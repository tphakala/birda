//! Integration tests for bat model install and `--bat` resolution (#397).
//!
//! Two kinds of test here. The first parses the bundled `registry.json`
//! directly and asserts the shipped bat catalog matches `BatRegion`'s
//! filename derivation for every region: an install writes the registry
//! filenames, and `BatConfig::resolve` later reads `BatRegion::model_filename`,
//! so if the two ever diverge an install would look successful yet never be
//! found. The rest drive the real binary, because the install-id dispatch and
//! the `--bat` auto-resolution live in the CLI layer where a unit test would
//! not reach them.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use assert_cmd::cargo::cargo_bin_cmd;
use birda::constants::CONFIG_DIR_ENV;
use serde_json::Value;

/// The exact registry the binary ships with, parsed without touching any
/// user/config directory so the test is hermetic.
const REGISTRY_JSON: &str = include_str!("../registry.json");

const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

#[test]
fn bundled_registry_bat_catalog_matches_batregion_naming() {
    let registry: birda::registry::Registry =
        serde_json::from_str(REGISTRY_JSON).expect("bundled registry.json parses");
    let catalog = registry.bat.expect("bundled registry has a bat catalog");

    assert_eq!(
        catalog.backbone.model.filename, "birdnet-v24-embeddings.onnx",
        "backbone local filename is the stable name --bat resolves"
    );
    assert!(catalog.backbone.model.sha256.is_some());
    assert!(catalog.backbone.labels.sha256.is_some());
    assert_eq!(
        catalog.regions.len(),
        11,
        "all 11 BattyBirdNET regions ship"
    );

    for entry in &catalog.regions {
        let region = birda::registry::parse_bat_install_id(&format!("bat-{}", entry.region))
            .unwrap_or_else(|| panic!("region slug '{}' must parse to a BatRegion", entry.region));

        // The install↔resolve invariant: filenames the installer writes must be
        // exactly the ones BatConfig::resolve reads.
        assert_eq!(
            entry.model.filename,
            region.model_filename(),
            "model filename mismatch for region '{}'",
            entry.region
        );
        assert_eq!(
            entry.labels.filename,
            region.labels_filename(),
            "labels filename mismatch for region '{}'",
            entry.region
        );

        assert!(
            entry.model.sha256.is_some() && entry.labels.sha256.is_some(),
            "region '{}' must carry checksums",
            entry.region
        );
        assert!(
            entry.model.size_bytes.is_some() && entry.labels.size_bytes.is_some(),
            "region '{}' must carry sizes",
            entry.region
        );
        assert!(
            entry.species_count > 0,
            "region '{}' has species",
            entry.region
        );
    }

    assert!(
        !catalog.license.commercial_use,
        "bat models are CC-BY-NC-SA (non-commercial)"
    );
}

/// Run birda against an isolated HOME so the suite never touches the
/// developer's real config or models directory.
fn run(args: &[&str]) -> std::process::Output {
    let home = tempfile::tempdir().expect("create temp home");
    run_in(home.path(), args)
}

/// Run birda against a caller-supplied HOME, so a test can seed files under the
/// models directory and then observe how a command reacts.
fn run_in(home: &std::path::Path, args: &[&str]) -> std::process::Output {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_DATA_HOME", home.join("data"))
        .env(CONFIG_DIR_ENV, home)
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
        "`birda {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn list_available_lists_bat_regions() {
    let out = stdout_of(&["models", "list-available"]);
    assert!(out.contains("Bat classifiers"), "has a bat section:\n{out}");
    assert!(out.contains("bat-eu"), "lists the EU install id:\n{out}");
    assert!(
        out.contains("bat-usa-east-high"),
        "lists multi-hyphen ids:\n{out}"
    );
}

#[test]
fn list_available_json_exposes_bat_catalog() {
    let out = stdout_of(&["models", "list-available", "--output-mode", "json"]);
    let value: Value = serde_json::from_str(&out).expect("valid JSON");
    // Structured results are wrapped in a `{event, payload, ...}` envelope.
    let bat = &value["payload"]["available_bat"];
    assert!(bat.is_object(), "available_bat present: {value}");
    let regions = bat["regions"].as_array().expect("regions array");
    assert_eq!(regions.len(), 11);
    assert!(
        bat["backbone_size_bytes"].as_u64().is_some(),
        "backbone size reported once"
    );
    // Non-commercial license flows through, so a consumer can warn on it.
    assert_eq!(bat["commercial_use"], serde_json::json!(false));

    // Pin one region's projected fields against independently-known values, so a
    // field-mapping bug (name/region swapped, species from the wrong source)
    // cannot pass. EU: 30 species, id bat-eu.
    let eu = regions
        .iter()
        .find(|r| r["id"] == "bat-eu")
        .expect("bat-eu present");
    assert_eq!(eu["region"], "eu");
    assert_eq!(eu["name"], "BattyBirdNET EU");
    assert_eq!(eu["species_count"], serde_json::json!(30));
    assert!(eu["size_bytes"].as_u64().is_some_and(|n| n > 0));
}

#[test]
fn info_bat_region_json_reports_bat_model_type() {
    let out = stdout_of(&["models", "info", "bat-eu", "--output-mode", "json"]);
    let value: Value = serde_json::from_str(&out).expect("valid JSON");
    let model = &value["payload"]["model"];
    assert_eq!(model["id"], "bat-eu");
    assert_eq!(model["model_type"], "bat");
}

#[test]
fn bat_analyze_custom_model_path_requires_labels_path() {
    // A custom backbone via --model-path must bring its own labels; pairing it
    // with the registry v2.4 labels would risk a label-count mismatch at load.
    let output = run(&["--bat", "eu", "--model-path", "/tmp/custom.onnx", "rec.wav"]);
    assert!(!output.status.success(), "must reject a lone --model-path");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--labels-path is required with --model-path"),
        "actionable labels-required error:\n{stderr}"
    );
}

#[test]
fn models_check_lists_an_installed_bat_region() {
    let home = tempfile::tempdir().expect("temp home");
    // Seed the four files a bat region needs, with the exact registry filenames,
    // so installed_bat_region_ids sees bat-eu as installed. Content is irrelevant
    // (models check reports presence, not checksums).
    let bat_dir = home.path().join("models").join("bat");
    std::fs::create_dir_all(&bat_dir).expect("create bat dir");
    for name in [
        "birdnet-v24-embeddings.onnx",
        "birdnet-v24-embeddings-labels.txt",
        "BattyBirdNET-EU-256kHz_fp32.onnx",
        "BattyBirdNET-EU-256kHz_Labels.txt",
    ] {
        std::fs::write(bat_dir.join(name), b"x").expect("seed file");
    }

    let output = run_in(home.path(), &["models", "check"]);
    assert!(
        output.status.success(),
        "models check failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Bat classifiers: bat-eu installed"),
        "check must list the installed region:\n{stdout}"
    );
}

#[test]
fn models_remove_deletes_an_installed_bat_region() {
    let home = tempfile::tempdir().expect("temp home");
    let bat_dir = home.path().join("models").join("bat");
    std::fs::create_dir_all(&bat_dir).expect("create bat dir");
    let files = [
        "birdnet-v24-embeddings.onnx",
        "birdnet-v24-embeddings-labels.txt",
        "BattyBirdNET-EU-256kHz_fp32.onnx",
        "BattyBirdNET-EU-256kHz_Labels.txt",
    ];
    for name in files {
        std::fs::write(bat_dir.join(name), b"x").expect("seed file");
    }

    let output = run_in(home.path(), &["models", "remove", "bat-eu", "--yes"]);
    assert!(
        output.status.success(),
        "remove failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // EU was the only region, so its head and the now-unneeded shared backbone
    // are both gone.
    for name in files {
        assert!(
            !bat_dir.join(name).exists(),
            "{name} should have been removed"
        );
    }
}

#[test]
fn info_reports_a_bat_region() {
    let out = stdout_of(&["models", "info", "bat-eu"]);
    assert!(out.contains("BattyBirdNET EU"), "names the region:\n{out}");
    assert!(out.contains("--bat eu"), "shows how to run it:\n{out}");
}

#[test]
fn install_rejects_an_unknown_bat_region_with_the_valid_list() {
    let output = run(&["models", "install", "bat-atlantis", "--yes"]);
    assert!(!output.status.success(), "unknown region must fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("bat region 'atlantis'") && stderr.contains("bavaria"),
        "names the bad region and lists valid ones:\n{stderr}"
    );
}

#[test]
fn bat_analyze_without_backbone_fails_fast_with_install_hint() {
    // No install has run, so the backbone is absent. Resolution must fail before
    // any file work with an actionable message, not the cryptic per-segment
    // error from deep in the pipeline.
    let output = run(&["--bat", "eu", "/nonexistent/recording.wav"]);
    assert!(!output.status.success(), "must fail without a backbone");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("embeddings backbone") && stderr.contains("models install bat-"),
        "actionable backbone-not-installed error:\n{stderr}"
    );
}
