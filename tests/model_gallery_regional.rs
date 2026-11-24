//! Regional and variant surface of the model gallery.
//!
//! These drive the real binary against the bundled registry, so they exercise
//! clap wiring, the registry contents and the output shape together. None of
//! them download anything: every assertion is about listing, or about rejecting
//! bad input before a single byte moves.
// Integration test crate. `unwrap`, `expect` and `panic` are how a test reports
// failure, not unhandled error paths, so rewriting them into propagated errors
// would only hide which assertion fired. Every exact float assertion in these
// tests is on a passed-through value (a literal parsed from a file, a
// coordinate round-tripped through JSON, a clip boundary clamped to a whole
// number) rather than a computed one, so exact equality is the assertion the
// test wants. The crate-level deny still governs everything birda ships.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use std::sync::LazyLock;

use assert_cmd::Command;
use birda::constants::CONFIG_DIR_ENV;
use predicates::prelude::*;
use tempfile::TempDir;

/// A throwaway config/data home shared by every test in this file.
///
/// Even the listing and reject-early cases reach `load_registry`, which
/// bootstraps `registry.json` into the config directory on a cache miss and
/// would otherwise read and write the developer's real profile (issue #328).
/// The override isolates on every platform, unlike HOME/XDG which `directories`
/// ignores on Windows.
///
/// The initializer primes the cache once, single-threaded, so the parallel
/// tests below all find `registry.json` already present and none of them races
/// to bootstrap it into the shared directory. The `TempDir` lives in the
/// `LazyLock` so it outlives every command; that intentionally leaks one
/// directory per test-binary run, which the OS temp reaper reclaims.
///
/// Every test here is read-only or rejects before writing, so one shared home
/// is safe. A test that mutates state (a real install, `config set`) must use
/// its own `TempDir` instead, or it would leak that state into its siblings.
static ISOLATED_HOME: LazyLock<TempDir> = LazyLock::new(|| {
    let home = TempDir::new().expect("create isolated home");
    let primed = Command::cargo_bin("birda")
        .expect("binary builds")
        .env(CONFIG_DIR_ENV, home.path())
        .env_remove("BIRDA_OUTPUT_MODE")
        .args(["models", "list-available"])
        .output()
        .expect("birda should run");
    assert!(
        primed.status.success(),
        "priming the registry cache failed: {}",
        String::from_utf8_lossy(&primed.stderr)
    );
    // Priming exists for its side effect: registry.json must now be present
    // under the override root (config_dir == home), or the parallel tests would
    // each bootstrap it and race. Assert it landed, so a future change that made
    // `list-available` stop caching fails loudly here instead of as an
    // occasional torn-read flake.
    assert!(
        home.path().join("registry.json").exists(),
        "priming did not create registry.json under the isolated home"
    );
    home
});

/// A birda invocation pinned to [`ISOLATED_HOME`].
///
/// `BIRDA_OUTPUT_MODE` is stripped because a developer with it exported would
/// otherwise flip every human-output assertion in this file to JSON, the same
/// scrub the sibling suites do.
fn birda() -> Command {
    let mut cmd = Command::cargo_bin("birda").expect("binary builds");
    cmd.env(CONFIG_DIR_ENV, ISOLATED_HOME.path())
        .env_remove("BIRDA_OUTPUT_MODE");
    cmd
}

/// Stdout of a successful run, split into lines.
fn stdout_lines(args: &[&str]) -> Vec<String> {
    let output = birda().args(args).assert().success().get_output().clone();
    String::from_utf8(output.stdout)
        .expect("stdout is UTF-8")
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn test_models_regions_lists_tiles_grouped_by_continent() {
    // The headers, in the order groups.json gives, each exactly once. A
    // `contains` check passes on a scrambled order or a header printed twice.
    let headers: Vec<String> = stdout_lines(&["models", "regions", "birdnet-plus-v30"])
        .into_iter()
        .filter(|line| line.ends_with(':') && !line.starts_with(' '))
        .collect();
    assert_eq!(
        headers,
        [
            "Regional variants of BirdNET+ V3.0:",
            "Europe:",
            "Asia:",
            "North America:",
            "South America:",
            "Africa:",
            "Oceania:",
        ]
    );
}

#[test]
fn test_models_regions_reports_species_counts_and_sizes() {
    // A user picking a region is choosing between memory footprints, so the
    // listing has to carry both numbers.
    birda()
        .args(["models", "regions", "birdnet-plus-v30"])
        .assert()
        .success()
        .stdout(predicate::str::contains("species"))
        .stdout(predicate::str::contains("MB"));
}

#[test]
fn test_models_regions_rejects_a_model_without_regions() {
    birda()
        .args(["models", "regions", "birdnet-v24"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("no regional variants"));
}

#[test]
fn test_models_regions_rejects_an_unknown_model() {
    birda()
        .args(["models", "regions", "not-a-model"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not-a-model"));
}

#[test]
fn test_models_install_rejects_an_unknown_region_before_downloading() {
    birda()
        .args([
            "models",
            "install",
            "birdnet-plus-v30",
            "--region",
            "atlantis",
            "--yes",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("atlantis"))
        .stderr(predicate::str::contains("nordic"));
}

#[test]
fn test_models_install_rejects_an_unknown_variant_before_downloading() {
    birda()
        .args([
            "models",
            "install",
            "birdnet-plus-v30",
            "--variant",
            "int4",
            "--yes",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("int4"))
        .stderr(predicate::str::contains("fp32"));
}

#[test]
fn test_models_install_rejects_a_region_on_a_model_that_has_none() {
    // Silently ignoring it would install the global model while the user
    // believed they had asked for a regional one.
    birda()
        .args([
            "models",
            "install",
            "birdnet-v24",
            "--region",
            "nordic",
            "--yes",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("no regional variants"));
}

#[test]
fn test_models_info_shows_the_exact_upstream_version_and_build() {
    // "3.0" would be a lie: the published weights are 3.0.1, and a later point
    // release is different weights under a version string that must not collide.
    let lines = stdout_lines(&["models", "info", "birdnet-plus-v30"]);
    assert!(
        lines.contains(&"Version: 3.0.1 (build 1)".to_string()),
        "no exact version line in {lines:?}"
    );
}

#[test]
fn test_models_info_reports_the_variants_and_the_regional_count() {
    let lines = stdout_lines(&["models", "info", "birdnet-plus-v30"]);
    for expected in [
        "Variants: fp32",
        "  Regional models: 44 (birda models regions birdnet-plus-v30)",
    ] {
        assert!(
            lines.contains(&expected.to_string()),
            "no line {expected:?} in {lines:?}"
        );
    }
}

#[test]
fn test_models_info_still_works_for_a_legacy_entry() {
    birda()
        .args(["models", "info", "birdnet-v24"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Files:"))
        .stdout(predicate::str::contains("languages available"));
}

#[test]
fn test_models_languages_explains_itself_on_a_variant_entry() {
    // Variant families ship one English labels file per region, so there is no
    // language list. Saying so beats printing an empty one.
    birda()
        .args(["models", "info", "birdnet-plus-v30", "--languages"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("language variants"));
}

#[test]
fn test_models_list_available_shows_every_family() {
    birda()
        .args(["models", "list-available"])
        .assert()
        .success()
        .stdout(predicate::str::contains("birdnet-v24"))
        .stdout(predicate::str::contains("birdnet-plus-v30"))
        .stdout(predicate::str::contains("perch-v2"));
}

#[test]
fn test_perch_publishes_regions_too() {
    birda()
        .args(["models", "regions", "perch-v2"])
        .assert()
        .success()
        .stdout(predicate::str::contains("nordic"));
}

#[test]
fn test_a_replaced_model_id_names_its_replacement() {
    let message = "error: model 'birdnet-v30' was replaced by 'birdnet-plus-v30'; \
                   run 'birda models install birdnet-plus-v30'\n";
    // One row per site that reports an unknown model id: install, `models
    // info` (its configuration fallback), regions and manifest.
    for args in [
        ["models", "install", "birdnet-v30"],
        ["models", "info", "birdnet-v30"],
        ["models", "regions", "birdnet-v30"],
        ["models", "manifest", "birdnet-v30"],
    ] {
        let output = birda().args(args).assert().failure().get_output().clone();
        assert_eq!(output.status.code(), Some(1), "{args:?}");
        assert_eq!(String::from_utf8_lossy(&output.stderr), message, "{args:?}");
    }
}

#[test]
fn test_a_configured_preview_model_still_shows_in_models_info() {
    // A config that names the retired id keeps working: the fallback maps only
    // its failure, never its success.
    let home = TempDir::new().unwrap();
    std::fs::write(
        home.path().join("config.toml"),
        "[models.birdnet-v30]\npath = \"/models/preview.onnx\"\nlabels = \"/models/preview.txt\"\n\
         type = \"birdnet-v30\"\n",
    )
    .unwrap();
    let output = Command::cargo_bin("birda")
        .unwrap()
        .env(CONFIG_DIR_ENV, home.path())
        .env_remove("BIRDA_OUTPUT_MODE")
        .args(["models", "info", "birdnet-v30"])
        .assert()
        .success()
        .get_output()
        .clone();
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Model: birdnet-v30\n  Type: birdnet-v30\n  Path: /models/preview.onnx\n  \
         Labels: /models/preview.txt\n"
    );
}
