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
fn test_manifest_projects_every_region_variant_and_document() {
    let payload = manifest_payload("birdnet-plus-v30");
    let manifest = &payload["manifest"];
    assert_eq!(manifest["id"], "birdnet-plus-v30");
    // The full catalogue: 1 global + 44 regions, one fp32 file each. A trimmed
    // or wrong vendored file, or the retired preview's, changes this.
    let variants = manifest["variants"].as_array().expect("variants array");
    assert_eq!(variants.len(), 45, "manifest keeps every combination");
    // fp32 is the only precision BirdNET+ publishes, so no variant id and no
    // hardware key can resolve to anything else.
    assert!(variants.iter().all(|v| v["id"] == "fp32"));
    let selection = manifest["selection"].as_object().expect("selection map");
    assert!(selection.values().all(|v| v == "fp32"), "{selection:?}");
    assert_eq!(variants[0]["classes"], 11725);
    // The licence record travels whole, so a consumer can show the obligations
    // (this was the gap #300 closed for `list-available`).
    assert_eq!(manifest["license"]["type"], "Apache-2.0");
    assert_eq!(manifest["license"]["share_alike"], false);
    assert_eq!(manifest["default_variant"], "fp32");
    // Both documents, pinned to the revision they were published at. The
    // acceptable use policy is the one `models install` asks agreement to.
    let base = "https://huggingface.co/tphakala/BirdNET-Plus-V3.0/resolve/\
                f90c3ba818b179adf5f4c9eabf2e6f8b801e4c06";
    let revision = "f90c3ba818b179adf5f4c9eabf2e6f8b801e4c06";
    assert_eq!(
        manifest["documents"],
        serde_json::json!([
            {
                "kind": "license",
                "requires_agreement": false,
                "url": format!("{base}/LICENSE-v3.0.1.md"),
                "revision": revision,
                "sha256": "ec0c349bd674520989b4df63afd50601695bd99a01768b8e46a04694573afd47",
            },
            {
                "kind": "acceptable_use",
                "requires_agreement": true,
                "url": format!("{base}/ACCEPTABLE_USE-v3.0.1.md"),
                "revision": revision,
                "sha256": "4495ebf9a088a83f902bbbef95ec17f635b51c94ade5d95b5d248646ea70a04e",
            },
        ])
    );
}

#[test]
fn test_manifest_carries_country_coverage_for_a_region() {
    let payload = manifest_payload("birdnet-plus-v30");
    let variants = payload["manifest"]["variants"].as_array().unwrap();
    let nordic = variants
        .iter()
        .find(|v| v["region"] == "nordic")
        .expect("nordic is a published region");
    assert_eq!(
        nordic["countries"],
        serde_json::json!({"core": ["Åland Islands", "Denmark", "Finland", "Norway", "Sweden"]})
    );
    assert_eq!(
        nordic["country_codes"],
        serde_json::json!({"core": ["AX", "DK", "FI", "NO", "SE"]})
    );
    assert!(nordic["region_name"].is_string());
}

#[test]
fn test_manifest_emits_resolved_download_urls() {
    let payload = manifest_payload("birdnet-plus-v30");
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
        &[
            "--output-mode",
            "json",
            "models",
            "manifest",
            "birdnet-plus-v30",
        ],
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
    let output = run(&[], &["models", "manifest", "birdnet-plus-v30"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Global variants:"), "got: {stdout}");
    assert!(stdout.contains("Regions:"), "got: {stdout}");
    assert!(
        stdout.contains("--output-mode json"),
        "should point at the machine-readable form: {stdout}"
    );
}
