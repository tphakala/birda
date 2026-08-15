//! `models manifest <id>` projection surface.
//!
//! These drive the real binary against the bundled registry, so they exercise
//! clap wiring, the projection, and the JSON envelope together. Nothing is
//! downloaded: every assertion is about the emitted manifest.
// Integration test crate. `unwrap`, `expect` and `panic` are how a test reports
// failure, not unhandled error paths, so rewriting them into propagated errors
// would only hide which assertion fired. The crate-level deny still governs
// everything birda ships.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// Isolation rides on the BIRDA_CONFIG_DIR override (see `run` below), which
// points the config and data directories at a temp dir on every platform. It
// replaced a `#![cfg(unix)]` gate this suite carried because HOME / XDG_*
// redirect `directories` on Unix only: on Windows it resolves the config and
// data directories through SHGetKnownFolderPath, which reads no environment
// variable, so the command under test would have read and rewritten the
// developer's real registry cache (issue #328).

use std::time::Duration;

use assert_cmd::cargo::cargo_bin_cmd;
use birda::constants::CONFIG_DIR_ENV;
use serde_json::Value;

/// `load_registry` can rewrite the cached registry, so keep commands bounded.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// Run birda against an isolated HOME.
///
/// `load_registry` writes an updated registry cache into the user's data
/// directory. Without this isolation the suite would read, and on some
/// platforms rewrite, the developer's real config.
fn run(extra_env: &[(&str, &str)], args: &[&str]) -> std::process::Output {
    let home = tempfile::tempdir().expect("create temp home");
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.env("HOME", home.path())
        .env("XDG_CONFIG_HOME", home.path().join("config"))
        .env("XDG_DATA_HOME", home.path().join("data"))
        // The override is what actually isolates on Windows, where `directories`
        // ignores HOME/XDG_*; it points the config dir (config.toml and the
        // cached registry.json) and the models dir at `home` (issue #328).
        .env(CONFIG_DIR_ENV, home.path())
        // `--output-mode` also reads BIRDA_OUTPUT_MODE; a developer with it
        // exported would flip the human-output assertions below.
        .env_remove("BIRDA_OUTPUT_MODE")
        .env_remove("HF_ENDPOINT")
        .timeout(COMMAND_TIMEOUT);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    for arg in args {
        cmd.arg(arg);
    }
    cmd.output().expect("birda should run")
}

/// Run in JSON mode and return the parsed result envelope's payload.
fn manifest_payload(id: &str) -> Value {
    let output = run(&[], &["--output-mode", "json", "models", "manifest", id]);
    assert!(
        output.status.success(),
        "`models manifest {id}` failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf-8");
    let envelope: Value = serde_json::from_str(&stdout).expect("stdout should be valid JSON");
    assert_eq!(envelope["event"], "result");
    assert_eq!(envelope["payload"]["result_type"], "model_manifest");
    envelope["payload"].clone()
}

#[test]
fn test_manifest_projects_every_region_and_variant() {
    let payload = manifest_payload("birdnet-v30");
    let manifest = &payload["manifest"];
    assert_eq!(manifest["id"], "birdnet-v30");
    // The full catalogue: 2 global + 39 regions x 2 variants = 80.
    let variants = manifest["variants"].as_array().expect("variants array");
    assert_eq!(variants.len(), 80, "manifest keeps every combination");
    // The licence record travels whole, so a consumer can show the share-alike
    // obligation (this was the gap #300 closed for `list-available`).
    assert!(manifest["license"]["share_alike"].is_boolean());
    assert!(manifest["default_variant"].is_string());
    assert!(manifest["selection"].is_object());
}

#[test]
fn test_manifest_carries_country_coverage_for_a_region() {
    let payload = manifest_payload("birdnet-v30");
    let variants = payload["manifest"]["variants"].as_array().unwrap();
    let amazonia = variants
        .iter()
        .find(|v| v["region"] == "amazonia")
        .expect("amazonia is a published region");
    let core = amazonia["countries"]["core"]
        .as_array()
        .expect("core country list");
    assert!(
        core.iter().any(|c| c == "Brazil"),
        "amazonia core coverage should list Brazil, got: {core:?}"
    );
    assert!(amazonia["region_name"].is_string());
}

#[test]
fn test_manifest_emits_resolved_download_urls() {
    let payload = manifest_payload("birdnet-v30");
    let variants = payload["manifest"]["variants"].as_array().unwrap();
    let v = &variants[0];
    assert!(
        v["model_url"].as_str().unwrap().starts_with("https://"),
        "model_url must be a resolved URL: {v}"
    );
    assert!(
        v["labels_url"].as_str().unwrap().starts_with("https://"),
        "labels_url must be a resolved URL: {v}"
    );
}

#[test]
fn test_manifest_applies_hf_endpoint_once_in_birda() {
    // The GUI derives the coverage-map URL from the model URL and fetches
    // through the same mirror, so the rewrite has to happen here, not there.
    let output = run(
        &[("HF_ENDPOINT", "https://hf-mirror.com")],
        &["--output-mode", "json", "models", "manifest", "birdnet-v30"],
    );
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let envelope: Value = serde_json::from_str(&stdout).unwrap();
    let v = &envelope["payload"]["manifest"]["variants"][0];
    assert!(
        v["model_url"]
            .as_str()
            .unwrap()
            .starts_with("https://hf-mirror.com/"),
        "model_url should be rewritten to the mirror: {v}"
    );
    assert!(
        v["labels_url"]
            .as_str()
            .unwrap()
            .starts_with("https://hf-mirror.com/"),
        "labels_url should be rewritten to the mirror: {v}"
    );
}

#[test]
fn test_manifest_projects_a_legacy_model_as_one_global_variant() {
    // birdnet-v24 has `files`, not `variants`. It must still project to one
    // uniform variant so a consumer never branches on an empty list.
    let payload = manifest_payload("birdnet-v24");
    let variants = payload["manifest"]["variants"].as_array().unwrap();
    assert_eq!(
        variants.len(),
        1,
        "one synthetic variant for a legacy model"
    );
    let only = &variants[0];
    assert_eq!(only["id"], "global");
    assert!(only["region"].is_null(), "a legacy model has no region");
    assert!(
        only["countries"].is_null(),
        "no region means no country coverage"
    );
    assert!(only["model_url"].as_str().unwrap().starts_with("https://"));
}

#[test]
fn test_manifest_rejects_an_unknown_model() {
    let output = run(&[], &["models", "manifest", "not-a-model"]);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("not-a-model"),
        "the error should name the id the user typed"
    );
}

#[test]
fn test_manifest_human_form_is_a_short_summary() {
    let output = run(&[], &["models", "manifest", "birdnet-v30"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Global variants:"), "got: {stdout}");
    assert!(stdout.contains("Regions:"), "got: {stdout}");
    assert!(
        stdout.contains("--output-mode json"),
        "should point at the machine-readable form: {stdout}"
    );
}
