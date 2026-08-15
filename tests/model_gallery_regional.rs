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

#[test]
fn test_models_regions_lists_tiles_grouped_by_continent() {
    birda()
        .args(["models", "regions", "birdnet-v30"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Europe:"))
        .stdout(predicate::str::contains("nordic"))
        .stdout(predicate::str::contains("Asia:"));
}

#[test]
fn test_models_regions_reports_species_counts_and_sizes() {
    // A user picking a region is choosing between memory footprints, so the
    // listing has to carry both numbers.
    birda()
        .args(["models", "regions", "birdnet-v30"])
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
            "birdnet-v30",
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
            "birdnet-v30",
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
    // "3.0" would be a lie: the published weights are preview3.1, and GA will
    // be different weights under a version string that must not collide.
    birda()
        .args(["models", "info", "birdnet-v30"])
        .assert()
        .success()
        .stdout(predicate::str::contains("3.0-preview3.1"))
        .stdout(predicate::str::contains("build 1"));
}

#[test]
fn test_models_info_reports_the_variants_and_the_regional_count() {
    birda()
        .args(["models", "info", "birdnet-v30"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Variants:"))
        .stdout(predicate::str::contains("Regional models: 39"));
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
        .args(["models", "info", "birdnet-v30", "--languages"])
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
        .stdout(predicate::str::contains("birdnet-v30"))
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
