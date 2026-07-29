//! Integration tests for geomodel discoverability (#287).
//!
//! These drive the real binary rather than calling `registry::show_info`
//! directly, and that is the point. `models info <id>` dispatches on
//! `registry::find_model`, which only searches `registry.models`; the geomodel
//! lives in `registry.range_filter`. A unit test calling `show_info` with the
//! geomodel id would therefore pass while `birda models info geomodel` still
//! failed, because the dispatch never reaches `show_info` for that id.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use assert_cmd::cargo::cargo_bin_cmd;
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

    let mut cmd = cargo_bin_cmd!("birda");
    cmd.env("HOME", home.path())
        .env("XDG_CONFIG_HOME", home.path().join("config"))
        .env("XDG_DATA_HOME", home.path().join("data"))
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

#[test]
fn test_models_info_still_rejects_an_unknown_id() {
    // The geomodel branch must not swallow genuinely unknown ids.
    let output = run(&["models", "info", "definitely-not-a-model"]);

    assert!(
        !output.status.success(),
        "an unknown id must still be an error"
    );
}

#[test]
fn test_models_info_uses_one_canonical_geomodel_handle() {
    // `models install` accepts only the install handle. If `models info` also
    // took the registry's internal asset id, `birda models info
    // birdnet-geomodel-v3` would succeed and then tell the user to run an
    // install command that rejects that id. One handle, or the two commands
    // disagree.
    let output = run(&["models", "info", "birdnet-geomodel-v3"]);

    assert!(
        !output.status.success(),
        "the registry asset id must not be a second accepted handle"
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

    let range_filter = &payload["range_filter"];
    assert!(
        !range_filter.is_null(),
        "range_filter must be present, got: {payload}"
    );

    // The install handle, not the registry asset id: this is what a user types.
    assert_eq!(range_filter["id"], "geomodel");
    assert_eq!(range_filter["share_alike"], true);
    assert_eq!(range_filter["species_count"], 12012);
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

    assert!(
        !models
            .iter()
            .any(|m| m["id"] == "geomodel" || m["id"] == "birdnet-geomodel-v3"),
        "the geomodel must not appear in `models`, got: {models:?}"
    );
}
