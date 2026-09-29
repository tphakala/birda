//! Integration tests for the geomodel command surface (#296).
//!
//! The defects here were dispatch defects: `models remove` searched
//! `config.models` while the geomodel lives in `defaults.geomodel`, and
//! `models check` keyed on the registry path while `analyze` prefers the
//! configured one. These tests drive the binary so the dispatch is exercised.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use assert_cmd::cargo::cargo_bin_cmd;
use birda::constants::CONFIG_DIR_ENV;
use serde_json::Value;

const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// Run birda against an isolated config and data directory.
fn run_in(home: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_DATA_HOME", home.join("data"))
        .env(CONFIG_DIR_ENV, home)
        .env_remove("BIRDA_OUTPUT_MODE")
        .env_remove("BIRDA_GEOMODEL_PATH")
        .env_remove("BIRDA_GEOMODEL_LABELS_PATH")
        .timeout(COMMAND_TIMEOUT);
    for arg in args {
        cmd.arg(arg);
    }
    cmd.output().expect("birda should run")
}

fn ok_in(home: &Path, args: &[&str]) -> std::process::Output {
    let output = run_in(home, args);
    assert!(
        output.status.success(),
        "`birda {}` failed with {:?}\nstderr: {}",
        args.join(" "),
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn payload_of(output: &std::process::Output) -> Value {
    let value: Value = serde_json::from_slice(&output.stdout).expect("valid JSON envelope");
    value["payload"].clone()
}

fn path_str(path: &Path) -> &str {
    path.to_str().expect("utf-8 path")
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Record a geomodel pair in the isolated config, as `models install geomodel` does.
fn configure_geomodel(home: &Path, model: &Path, labels: &Path) {
    ok_in(
        home,
        &["config", "set", "defaults.geomodel", path_str(model)],
    );
    ok_in(
        home,
        &[
            "config",
            "set",
            "defaults.geomodel_labels",
            path_str(labels),
        ],
    );
}

fn config_text(home: &Path) -> String {
    let path = home.join("config.toml");
    std::fs::read_to_string(&path).unwrap_or_default()
}

#[test]
fn test_remove_geomodel_clears_both_config_keys() {
    let home = tempfile::tempdir().unwrap();
    configure_geomodel(
        home.path(),
        &fixture("fixture-geomodel.onnx"),
        &fixture("fixture-geomodel-labels.txt"),
    );

    let output = ok_in(
        home.path(),
        &["--output-mode", "json", "models", "remove", "geomodel"],
    );
    let payload = payload_of(&output);
    assert_eq!(payload["result_type"], "model_removed");
    assert_eq!(payload["id"], "geomodel");
    assert_eq!(payload["purge_requested"], false);

    let config = config_text(home.path());
    assert!(
        !config.contains("geomodel"),
        "both keys must be gone from the saved config, got:\n{config}"
    );
    // Without --purge the files are left where they are.
    assert!(fixture("fixture-geomodel.onnx").is_file());
}

#[test]
fn test_remove_geomodel_when_not_configured_is_not_found() {
    let home = tempfile::tempdir().unwrap();
    let output = run_in(home.path(), &["models", "remove", "geomodel"]);
    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).trim(),
        "error: model 'geomodel' not found in configuration"
    );
}

#[test]
fn test_remove_geomodel_purge_deletes_only_files_in_the_models_dir() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    std::fs::create_dir_all(&models).unwrap();
    let managed_model = models.join("geomodel.onnx");
    std::fs::write(&managed_model, b"model").unwrap();

    // The labels file sits outside the models directory: a path the user
    // pointed at by hand, which --purge must not touch.
    let outside = tempfile::tempdir().unwrap();
    let hand_labels = outside.path().join("labels.txt");
    std::fs::write(&hand_labels, b"labels").unwrap();

    configure_geomodel(home.path(), &managed_model, &hand_labels);

    ok_in(
        home.path(),
        &[
            "--output-mode",
            "json",
            "models",
            "remove",
            "geomodel",
            "--purge",
        ],
    );

    assert!(!managed_model.exists(), "the managed copy must be deleted");
    assert!(hand_labels.is_file(), "a hand-placed file must survive");
}

#[test]
fn test_check_reports_the_configured_geomodel_not_the_registry_path() {
    let home = tempfile::tempdir().unwrap();
    let model = fixture("fixture-geomodel.onnx");
    let labels = fixture("fixture-geomodel-labels.txt");
    configure_geomodel(home.path(), &model, &labels);

    let output = ok_in(home.path(), &["--output-mode", "json", "models", "check"]);
    let geomodel = payload_of(&output)["geomodel"].clone();

    assert_eq!(geomodel["installed"], true, "got: {geomodel}");
    assert_eq!(geomodel["model_path"], path_str(&model));
    assert_eq!(geomodel["labels_path"], path_str(&labels));
}

#[test]
fn test_check_honours_the_geomodel_path_flags() {
    let home = tempfile::tempdir().unwrap();
    let model = fixture("fixture-geomodel.onnx");
    let labels = fixture("fixture-geomodel-labels.txt");

    let output = ok_in(
        home.path(),
        &[
            "--output-mode",
            "json",
            "--geomodel-path",
            path_str(&model),
            "--geomodel-labels-path",
            path_str(&labels),
            "models",
            "check",
        ],
    );
    let geomodel = payload_of(&output)["geomodel"].clone();

    assert_eq!(geomodel["installed"], true, "got: {geomodel}");
    assert_eq!(geomodel["model_path"], path_str(&model));
}

#[test]
fn test_check_without_any_geomodel_reports_not_installed() {
    let home = tempfile::tempdir().unwrap();
    let output = ok_in(home.path(), &["--output-mode", "json", "models", "check"]);
    let geomodel = payload_of(&output)["geomodel"].clone();

    assert_eq!(geomodel["installed"], false, "got: {geomodel}");
    assert!(geomodel.get("model_path").is_none());
}

#[test]
fn test_config_set_warns_about_a_half_configured_geomodel() {
    let home = tempfile::tempdir().unwrap();
    let model = fixture("fixture-geomodel.onnx");

    let first = ok_in(
        home.path(),
        &["config", "set", "defaults.geomodel", path_str(&model)],
    );
    let warning = String::from_utf8_lossy(&first.stderr).into_owned();
    assert!(
        warning
            .starts_with("warning: 'defaults.geomodel' is set without 'defaults.geomodel_labels'"),
        "the warning must name the key that was set and the one missing, got: {warning}"
    );

    // Completing the pair silences it.
    let second = ok_in(
        home.path(),
        &[
            "config",
            "set",
            "defaults.geomodel_labels",
            path_str(&fixture("fixture-geomodel-labels.txt")),
        ],
    );
    assert_eq!(String::from_utf8_lossy(&second.stderr), "");
}
