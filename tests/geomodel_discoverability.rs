//! Integration tests for geomodel discoverability (#287).
//!
//! These drive the real binary rather than calling into `registry` directly,
//! and that is the point. `models info <id>` dispatches on
//! `registry::find_model`, which only searches `registry.models`; the geomodel
//! lives in `registry.range_filter`. A unit test calling
//! `show_range_filter_info(asset)` would pass while `birda models info
//! geomodel` still failed, because that function takes the asset directly and
//! the dispatch never reaches it for this id. Only driving the binary exercises
//! the dispatch, which is where the defect lived.
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

use std::time::Duration;

use assert_cmd::cargo::cargo_bin_cmd;
use birda::constants::CONFIG_DIR_ENV;
use serde_json::Value;

/// Registry commands are local, but `load_registry` can rewrite the cached
/// registry, so keep them bounded.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// Run birda against an isolated HOME.
///
/// `load_registry` writes an updated registry cache into the user's data
/// directory. Without this the suite would mutate the developer's real config.
fn run(args: &[&str]) -> std::process::Output {
    let home = tempfile::tempdir().expect("create temp home");
    run_in(home.path(), args)
}

/// Run birda against a caller-supplied HOME, so several invocations can share
/// one config directory. Needed to seed a registry file and then observe how a
/// later command reacts to it.
fn run_in(home: &std::path::Path, args: &[&str]) -> std::process::Output {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_DATA_HOME", home.join("data"))
        // The override is what actually isolates on Windows, where `directories`
        // ignores HOME/XDG_*; it points both the config dir (config.toml and the
        // cached registry.json) and the models dir at `home` (issue #328).
        .env(CONFIG_DIR_ENV, home)
        // `--output-mode` reads BIRDA_OUTPUT_MODE. A developer with that
        // exported turns every human-output assertion below into a failure, so
        // the isolation has to cover the environment, not just the filesystem.
        .env_remove("BIRDA_OUTPUT_MODE")
        .timeout(COMMAND_TIMEOUT);
    for arg in args {
        cmd.arg(arg);
    }

    cmd.output().expect("birda should run")
}

fn stdout_of(args: &[&str]) -> String {
    let output = run(args);
    assert!(
        output.status.success(),
        "`birda {}` failed with {:?}\nstderr: {}",
        args.join(" "),
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn test_models_info_geomodel_succeeds() {
    // Before the fix this exited non-zero with "model 'geomodel' not found",
    // despite every error message and doc page telling users to install it.
    let stdout = stdout_of(&["models", "info", "geomodel"]);

    assert!(
        stdout.contains("BirdNET Geomodel"),
        "should name the asset, got: {stdout}"
    );
}

#[test]
fn test_models_info_geomodel_shows_the_licence_terms() {
    // This is the "what am I about to download?" step, and the only place the
    // CC BY-SA terms surface before the user commits to the download. The
    // geomodel's share-alike obligation differs from the classifiers'
    // CC BY-NC-SA, so it has to be visible here specifically.
    let stdout = stdout_of(&["models", "info", "geomodel"]);

    assert!(
        stdout.contains("CC-BY-SA-4.0"),
        "should show the licence id, got: {stdout}"
    );
    assert!(
        stdout.contains("Share-alike required: Yes"),
        "should spell out the share-alike obligation, got: {stdout}"
    );
}

#[test]
fn test_models_info_geomodel_shows_coverage_and_size() {
    let stdout = stdout_of(&["models", "info", "geomodel"]);

    assert!(
        stdout.contains("12012"),
        "should report the species count, got: {stdout}"
    );
    assert!(
        stdout.contains("Download size"),
        "should report the download size, got: {stdout}"
    );
    // The label alone proves nothing: `total_download_size` returns None when
    // either file lacks a declared size, and `human_size(None)` renders
    // "unknown size" under that same label. Mutation testing confirmed a
    // always-None mutant survived the label-only assertion.
    assert!(
        !stdout.contains("unknown size"),
        "the size must actually resolve, not fall back to 'unknown size': {stdout}"
    );
}

#[test]
fn test_models_info_geomodel_languages_flag_is_handled() {
    // `--languages` routes to `show_languages`, which is a second consumer
    // behind the same dispatch. The geomodel has no language variants, so this
    // must explain that rather than fail or print an empty list.
    let stdout = stdout_of(&["models", "info", "geomodel", "--languages"]);

    assert!(
        stdout.contains("no label language variants"),
        "should explain the absence of language variants, got: {stdout}"
    );
}

/// Assert a command was rejected BY THE APPLICATION, naming the offending id.
///
/// `!status.success()` alone is not a verdict: it is equally true of a clap
/// parse error (exit 2), a panic (exit 101) and a timeout kill (no code at
/// all). Mutation testing confirmed it -- making the fallback panic left both
/// negative tests below green. Pinning exit code 1 plus the id in stderr is
/// what makes them observe the rejection they claim to.
fn assert_rejected_naming(args: &[&str], id: &str) {
    let output = run(args);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert_eq!(
        output.status.code(),
        Some(1),
        "expected a clean application error (exit 1), not a parse error, panic \
         or kill. stderr: {stderr}"
    );
    assert!(
        stderr.contains(id),
        "the error must name the rejected id '{id}', got: {stderr}"
    );
    // Against STDOUT, not stderr. `show_range_filter_info` prints with
    // `println!` while errors go to stderr via `eprintln!`, so asserting this
    // on stderr checked a stream the string can never reach: it passed for an
    // unrelated reason and would have kept passing if the range filter really
    // had been rendered for a rejected id.
    assert!(
        !stdout.contains("Range filter:"),
        "must not have rendered the range filter for a rejected id, got: {stdout}"
    );
}

#[test]
fn test_models_info_still_rejects_an_unknown_id() {
    // The geomodel branch must not swallow genuinely unknown ids.
    assert_rejected_naming(
        &["models", "info", "definitely-not-a-model"],
        "definitely-not-a-model",
    );
}

#[test]
fn test_models_info_uses_one_canonical_geomodel_handle() {
    // `models install` accepts only the install handle. If `models info` also
    // took the registry's internal asset id, `birda models info
    // birdnet-geomodel-v3` would succeed and then tell the user to run an
    // install command that rejects that id. One handle, or the two commands
    // disagree.
    assert_rejected_naming(
        &["models", "info", "birdnet-geomodel-v3"],
        "birdnet-geomodel-v3",
    );
}

#[test]
fn test_list_available_mentions_the_range_filter() {
    let stdout = stdout_of(&["models", "list-available"]);

    assert!(
        stdout.contains("Range filter"),
        "should carry a range filter section, got: {stdout}"
    );
    assert!(
        stdout.contains("geomodel"),
        "should name the install id, got: {stdout}"
    );
}

#[test]
fn test_list_available_json_exposes_range_filter_as_a_sibling_field() {
    let stdout = stdout_of(&["--output-mode", "json", "models", "list-available"]);
    let value: Value = serde_json::from_str(&stdout).expect("valid JSON envelope");
    let payload = &value["payload"];

    // Named `available_range_filter`, not `range_filter`: `DetectionsPayload`
    // already uses `range_filter` for a per-run coverage shape, and distinct
    // names keep the two readable side by side.
    let range_filter = &payload["available_range_filter"];
    assert!(
        !range_filter.is_null(),
        "available_range_filter must be present, got: {payload}"
    );

    // The install handle, not the registry asset id: this is what a user types.
    assert_eq!(range_filter["id"], "geomodel");
    assert_eq!(range_filter["share_alike"], true);
    assert_eq!(range_filter["species_count"], 12012);
    assert!(
        range_filter["size_bytes"].as_u64().is_some_and(|n| n > 0),
        "size_bytes must resolve to a real total, got: {range_filter}"
    );
}

#[test]
fn test_list_available_json_keeps_models_classifier_only() {
    // The additive claim. The geomodel is not selectable with `-m`, so a
    // consumer building a model picker from `models` must not see it; if it
    // ever leaks in, that picker offers an entry that fails on use.
    let stdout = stdout_of(&["--output-mode", "json", "models", "list-available"]);
    let value: Value = serde_json::from_str(&stdout).expect("valid JSON envelope");

    let models = value["payload"]["models"]
        .as_array()
        .expect("models is an array");

    // Prove the precondition before asserting an absence. Mutation testing
    // confirmed the `any()` check below passes vacuously on an empty array.
    assert!(
        !models.is_empty(),
        "the classifier list must be populated for this absence check to mean \
         anything, got: {models:?}"
    );

    assert!(
        !models
            .iter()
            .any(|m| m["id"] == "geomodel" || m["id"] == "birdnet-geomodel-v3"),
        "the geomodel must not appear in `models`, got: {models:?}"
    );
}

#[test]
fn test_models_info_geomodel_on_a_registry_without_the_asset() {
    // The case fix 4 exists for, and the only one that can distinguish it.
    //
    // The dispatch used to read `if let Some(asset) = registry.range_filter
    // && id == GEOMODEL_INSTALL_ID`, testing the asset BEFORE the id. With no
    // asset it fell through to `find_model` (never matches) and then
    // `config::get_model`, producing "model 'geomodel' not found in
    // configuration" -- the exact message the branch was written to prevent.
    // `models install geomodel` already ordered it the other way and reported
    // the accurate `RangeFilterAssetMissing`, so the two commands disagreed on
    // one registry state.
    //
    // Every other test here passes with the fix reverted, because for any id
    // that is not the install handle both orderings take the same path. This
    // one needs a registry that has no `range_filter` at all.
    let home = tempfile::tempdir().expect("create temp home");

    // Learn where this platform puts the config, rather than assuming a layout.
    let config_path = String::from_utf8_lossy(&run_in(home.path(), &["config", "path"]).stdout)
        .trim()
        .to_string();
    let config_dir = std::path::Path::new(&config_path)
        .parent()
        .expect("config path has a parent")
        .to_path_buf();

    // `load_registry` keeps the cached copy whenever its `registry_version` is
    // at least the bundled one, so a high version pins this stripped registry.
    std::fs::create_dir_all(&config_dir).expect("create config dir");
    std::fs::write(
        config_dir.join("registry.json"),
        r#"{"schema_version":"1.0","registry_version":9999,"models":[]}"#,
    )
    .expect("seed a registry with no range_filter");

    let output = run_in(home.path(), &["models", "info", "geomodel"]);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success(),
        "a registry with no range filter cannot describe the geomodel"
    );
    assert!(
        !stderr.contains("not found in configuration"),
        "must not claim the id is an unknown configured model; that is the \
         misleading fallback this branch exists to avoid. stderr: {stderr}"
    );
    assert!(
        stderr.contains("range filter"),
        "must report the missing range filter asset, matching what \
         `models install geomodel` reports for the same state. stderr: {stderr}"
    );
}
