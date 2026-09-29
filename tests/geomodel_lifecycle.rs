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
    // Commands such as `species` write a default output file into the working
    // directory, which must not be the repository.
    cmd.current_dir(home)
        .env("HOME", home)
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

/// Seed the isolated home with a registry whose geomodel is already on disk.
///
/// The bundled registry with the geomodel's two filenames pointed at the test
/// fixtures and their checksums dropped, which makes `install_range_filter`
/// accept the files as they are instead of downloading 14 MB. A registry
/// version above the bundled one keeps the cached copy. Returns the model and
/// labels paths at the registry's own install location.
fn seed_installed_registry_geomodel(home: &Path) -> (PathBuf, PathBuf) {
    let bundled = Path::new(env!("CARGO_MANIFEST_DIR")).join("registry.json");
    let mut registry: Value =
        serde_json::from_slice(&std::fs::read(bundled).unwrap()).expect("bundled registry parses");
    registry["registry_version"] = Value::from(9999);
    let asset = &mut registry["range_filter"];
    for (key, filename) in [
        ("model", "seeded-geomodel.onnx"),
        ("labels", "seeded-geomodel-labels.txt"),
    ] {
        asset[key]["filename"] = Value::from(filename);
        asset[key].as_object_mut().unwrap().remove("sha256");
    }
    std::fs::write(home.join("registry.json"), registry.to_string()).unwrap();

    let models = home.join("models");
    std::fs::create_dir_all(&models).unwrap();
    let model = models.join("seeded-geomodel.onnx");
    let labels = models.join("seeded-geomodel-labels.txt");
    std::fs::copy(fixture("fixture-geomodel.onnx"), &model).unwrap();
    std::fs::copy(fixture("fixture-geomodel-labels.txt"), &labels).unwrap();
    (model, labels)
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

#[test]
fn test_check_reports_a_half_configured_geomodel_as_not_installed() {
    let home = tempfile::tempdir().unwrap();
    // A copy exists at the registry's own location, so a check that fell back to
    // that path would report "installed" for a config `analyze` cannot use.
    seed_installed_registry_geomodel(home.path());
    ok_in(
        home.path(),
        &[
            "config",
            "set",
            "defaults.geomodel",
            "/elsewhere/model.onnx",
        ],
    );

    let output = ok_in(home.path(), &["--output-mode", "json", "models", "check"]);
    let geomodel = payload_of(&output)["geomodel"].clone();

    assert_eq!(geomodel["installed"], false, "got: {geomodel}");
    assert!(geomodel.get("model_path").is_none(), "got: {geomodel}");
    assert!(geomodel.get("labels_path").is_none(), "got: {geomodel}");
}

#[test]
fn test_check_reports_the_registry_copy_when_nothing_is_configured() {
    // The counterpart of the test above: it is only meaningful if the seeded copy
    // is otherwise reported installed.
    let home = tempfile::tempdir().unwrap();
    let (model, labels) = seed_installed_registry_geomodel(home.path());

    let output = ok_in(home.path(), &["--output-mode", "json", "models", "check"]);
    let geomodel = payload_of(&output)["geomodel"].clone();

    assert_eq!(geomodel["installed"], true, "got: {geomodel}");
    assert_eq!(geomodel["model_path"], path_str(&model));
    assert_eq!(geomodel["labels_path"], path_str(&labels));
}

#[test]
fn test_remove_geomodel_purge_skips_a_path_that_climbs_out_of_the_models_dir() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    std::fs::create_dir_all(&models).unwrap();

    // Lexically under the models directory, but `..` puts it in the directory
    // above: a file the user never handed to birda.
    let outside = home.path().join("outside.txt");
    std::fs::write(&outside, b"not birda's").unwrap();
    let climbing = models.join("..").join("outside.txt");

    let managed_labels = models.join("labels.txt");
    std::fs::write(&managed_labels, b"labels").unwrap();

    configure_geomodel(home.path(), &climbing, &managed_labels);
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

    assert!(
        outside.is_file(),
        "a file outside the models dir must survive"
    );
    assert!(
        !managed_labels.exists(),
        "the managed file is still deleted"
    );
}

#[cfg(unix)]
#[test]
fn test_remove_geomodel_purge_removes_a_symlink_but_not_its_target() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    std::fs::create_dir_all(&models).unwrap();

    // A model kept on another volume and linked into the models directory: the
    // link is birda's to remove, the file it points at is not.
    let elsewhere = tempfile::tempdir().unwrap();
    let target = elsewhere.path().join("model.onnx");
    std::fs::write(&target, b"model").unwrap();
    let link = models.join("model.onnx");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let labels = models.join("labels.txt");
    std::fs::write(&labels, b"labels").unwrap();

    configure_geomodel(home.path(), &link, &labels);
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

    assert!(link.symlink_metadata().is_err(), "the link must be removed");
    assert!(target.is_file(), "the link's target must survive");
}

/// A home with one classifier configured (its labels file holds `classifier_labels`)
/// and the geomodel pair recorded in the config.
fn home_with_classifier_and_geomodel(
    classifier_labels: &str,
    geomodel_labels: &Path,
) -> tempfile::TempDir {
    let home = tempfile::tempdir().unwrap();
    let model = home.path().join("classifier.onnx");
    let labels = home.path().join("classifier-labels.txt");
    std::fs::write(&model, b"not run: species stops at the labels check").unwrap();
    std::fs::write(&labels, classifier_labels).unwrap();
    ok_in(
        home.path(),
        &[
            "models",
            "add",
            "test-classifier",
            "--path",
            path_str(&model),
            "--labels",
            path_str(&labels),
            "--type",
            "birdnet-v24",
            "--default",
        ],
    );
    configure_geomodel(
        home.path(),
        &fixture("fixture-geomodel.onnx"),
        geomodel_labels,
    );
    home
}

/// Whether an ONNX Runtime can be loaded here.
///
/// `birda species` initialises the runtime before it reads any labels, so the
/// three `species` tests below fail on a machine without one (a CI runner)
/// before reaching the check they exist to pin.
fn onnx_runtime_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| birda::inference::ensure_runtime_available().is_ok())
}

macro_rules! require_onnx_runtime {
    () => {
        if !onnx_runtime_available() {
            eprintln!("skipping: ONNX Runtime not available in this environment");
            return;
        }
    };
}

const SPECIES_ARGS: [&str; 6] = ["species", "--lat", "60.17", "--lon", "24.94", "--week=20"];

#[test]
fn test_species_rejects_a_geomodel_labels_file_of_the_wrong_size() {
    require_onnx_runtime!();
    // The fixture's labels file has 5 lines, not the geomodel's 12,012. `species`
    // used to skip this check and let birdnet-onnx report a bare count mismatch.
    let home = home_with_classifier_and_geomodel(
        "Parus major_Great Tit\n",
        &fixture("fixture-geomodel-labels.txt"),
    );

    let output = run_in(home.path(), &SPECIES_ARGS);

    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).trim(),
        "error: BirdNET Geomodel v3.0.2 labels file has 5 labels, expected 12012; \
         reinstall with 'birda models install geomodel'"
    );
}

#[test]
fn test_species_rejects_an_empty_geomodel_labels_file() {
    require_onnx_runtime!();
    let home = tempfile::tempdir().unwrap();
    let empty = home.path().join("empty-labels.txt");
    std::fs::write(&empty, b"").unwrap();
    let home = home_with_classifier_and_geomodel("Parus major_Great Tit\n", &empty);

    let output = run_in(home.path(), &SPECIES_ARGS);

    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).trim(),
        format!(
            "error: failed to load labels from {}: file contains no labels",
            empty.display()
        )
    );
}

#[test]
fn test_species_rejects_an_empty_classifier_labels_file() {
    require_onnx_runtime!();
    // `read_labels_file` returned Ok(vec![]) here, so `species` carried on with no
    // labels and reported zero coverage instead of naming the empty file.
    let home = home_with_classifier_and_geomodel("\n  \n", &fixture("fixture-geomodel-labels.txt"));

    let output = run_in(home.path(), &SPECIES_ARGS);

    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).trim(),
        format!(
            "error: failed to load labels from {}: file contains no labels",
            home.path().join("classifier-labels.txt").display()
        )
    );
}

#[test]
fn test_install_geomodel_emits_its_result_and_reports_ignored_path_flags() {
    let home = tempfile::tempdir().unwrap();
    let (model, labels) = seed_installed_registry_geomodel(home.path());

    let output = ok_in(
        home.path(),
        &[
            "--output-mode",
            "json",
            "--geomodel-path",
            "/ignored/model.onnx",
            "--geomodel-labels-path",
            "/ignored/labels.txt",
            "models",
            "install",
            "geomodel",
        ],
    );

    // stdout is what birda-gui parses: it used to be empty.
    let payload = payload_of(&output);
    assert_eq!(payload["result_type"], "model_installed");
    assert_eq!(payload["id"], "geomodel");
    assert_eq!(payload["set_as_default"], false);
    assert_eq!(payload["model_path"], path_str(&model));
    assert_eq!(payload["labels_path"], path_str(&labels));

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(
            "--geomodel-path and --geomodel-labels-path are ignored by 'models install geomodel'"
        ),
        "the ignored flags must be reported, got: {stderr}"
    );

    // The install records the models-directory paths, not the flags.
    let config = config_text(home.path());
    assert!(config.contains(path_str(&model)), "got:\n{config}");
    assert!(!config.contains("/ignored/"), "got:\n{config}");
}

#[test]
fn test_config_set_warns_for_labels_without_the_model_too() {
    let home = tempfile::tempdir().unwrap();

    let output = ok_in(
        home.path(),
        &[
            "config",
            "set",
            "defaults.geomodel_labels",
            path_str(&fixture("fixture-geomodel-labels.txt")),
        ],
    );

    assert!(
        String::from_utf8_lossy(&output.stderr)
            .starts_with("warning: 'defaults.geomodel_labels' is set without 'defaults.geomodel'"),
        "got: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

const NOT_RECORDED_ERROR: &str = "error: the geomodel is not recorded in the configuration, but a copy is installed in the models directory and birda finds it there; delete it with 'birda models remove geomodel --purge'";

#[test]
fn test_purge_deletes_an_unrecorded_registry_copy() {
    // A classifier install and the implicit download on `analyze` fetch the
    // geomodel without recording it in the config, which is the common case.
    let home = tempfile::tempdir().unwrap();
    let (model, labels) = seed_installed_registry_geomodel(home.path());

    let output = ok_in(
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

    let payload = payload_of(&output);
    assert_eq!(payload["result_type"], "model_removed");
    assert_eq!(payload["id"], "geomodel");
    assert_eq!(payload["purge_requested"], true);
    assert!(!model.exists(), "the model file must be deleted");
    assert!(!labels.exists(), "the labels file must be deleted");
    assert_eq!(
        config_text(home.path()),
        "",
        "an unrecorded copy has nothing to change in the config"
    );
}

#[test]
fn test_plain_remove_of_an_unrecorded_registry_copy_says_to_purge() {
    let home = tempfile::tempdir().unwrap();
    let (model, labels) = seed_installed_registry_geomodel(home.path());

    let output = run_in(home.path(), &["models", "remove", "geomodel"]);

    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).trim(),
        NOT_RECORDED_ERROR
    );
    assert!(
        model.is_file() && labels.is_file(),
        "nothing may be deleted"
    );
}

#[test]
fn test_plain_remove_says_the_files_are_still_in_use() {
    // `models install geomodel` records the registry paths, so clearing the keys
    // leaves the files where `analyze` finds them again.
    let home = tempfile::tempdir().unwrap();
    let (model, labels) = seed_installed_registry_geomodel(home.path());
    configure_geomodel(home.path(), &model, &labels);

    let output = ok_in(home.path(), &["models", "remove", "geomodel"]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.lines().any(|line| line
            == "The geomodel files are still in the models directory and birda still uses them; add --purge to delete them."),
        "got:\n{stdout}"
    );
    assert!(model.is_file() && labels.is_file());
}

#[test]
fn test_purge_reports_a_recorded_path_that_no_longer_exists() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    std::fs::create_dir_all(&models).unwrap();
    // The parent directory is gone as well, so resolving it fails with NotFound.
    let gone = models.join("removed-subdir").join("model.onnx");
    let labels = models.join("labels.txt");
    std::fs::write(&labels, b"labels").unwrap();
    configure_geomodel(home.path(), &gone, &labels);

    let output = ok_in(
        home.path(),
        &["models", "remove", "geomodel", "--purge", "--yes"],
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout
            .lines()
            .any(|line| line == format!("  Skipped (not found): {}", gone.display())),
        "got:\n{stdout}"
    );
    assert!(!labels.exists(), "the existing file is still deleted");
}

#[test]
fn test_check_says_why_a_half_configured_geomodel_is_not_installed() {
    let home = tempfile::tempdir().unwrap();
    ok_in(
        home.path(),
        &[
            "config",
            "set",
            "defaults.geomodel",
            "/elsewhere/model.onnx",
        ],
    );

    let output = ok_in(home.path(), &["--output-mode", "json", "models", "check"]);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(
            "geomodel path and geomodel labels path must be given together (received only defaults.geomodel)"
        ),
        "the missing key must be named, got: {stderr}"
    );
}

const STILL_IN_USE_NOTE: &str = "The geomodel files are still in the models directory and birda still uses them; add --purge to delete them.";

fn lines_of(output: &std::process::Output) -> Vec<String> {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn test_purge_of_a_recorded_registry_copy_deletes_each_file_once() {
    // `models install geomodel` records the registry paths, so the recorded and
    // the registry copy are the same two files and must not be handled twice.
    let home = tempfile::tempdir().unwrap();
    let (model, labels) = seed_installed_registry_geomodel(home.path());
    configure_geomodel(home.path(), &model, &labels);

    let output = ok_in(
        home.path(),
        &["models", "remove", "geomodel", "--purge", "--yes"],
    );

    let lines = lines_of(&output);
    assert_eq!(
        lines
            .iter()
            .filter(|l| l.starts_with("  Deleted: "))
            .collect::<Vec<_>>(),
        [
            &format!("  Deleted: {}", model.display()),
            &format!("  Deleted: {}", labels.display())
        ]
    );
    assert!(
        !lines.iter().any(|l| l.contains("Skipped")),
        "got: {lines:?}"
    );
    assert!(!lines.iter().any(|l| l == STILL_IN_USE_NOTE));
}

#[test]
fn test_plain_remove_of_a_hand_placed_geomodel_does_not_claim_it_is_still_used() {
    // No copy at the registry location, so nothing is left for birda to use.
    let home = tempfile::tempdir().unwrap();
    configure_geomodel(
        home.path(),
        &fixture("fixture-geomodel.onnx"),
        &fixture("fixture-geomodel-labels.txt"),
    );

    let output = ok_in(home.path(), &["models", "remove", "geomodel"]);

    let lines = lines_of(&output);
    assert_eq!(lines[0], "Geomodel removed from configuration.");
    assert!(
        lines[1].starts_with("Configuration saved to: "),
        "got: {lines:?}"
    );
    assert_eq!(lines.len(), 2, "got: {lines:?}");
}

#[test]
fn test_purge_with_nothing_recorded_or_installed_is_not_found() {
    let home = tempfile::tempdir().unwrap();

    let output = run_in(
        home.path(),
        &["models", "remove", "geomodel", "--purge", "--yes"],
    );

    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).trim(),
        "error: model 'geomodel' not found in configuration"
    );
}

#[test]
fn test_a_lone_registry_file_is_purged_but_not_reported_as_installed() {
    // Resolution needs both files, so one leftover is not "installed": a plain
    // remove has nothing to say about it, and --purge still cleans it up.
    let home = tempfile::tempdir().unwrap();
    let (model, labels) = seed_installed_registry_geomodel(home.path());
    std::fs::remove_file(&labels).unwrap();

    let plain = run_in(home.path(), &["models", "remove", "geomodel"]);
    assert!(!plain.status.success());
    assert_eq!(
        String::from_utf8_lossy(&plain.stderr).trim(),
        "error: model 'geomodel' not found in configuration"
    );
    assert!(model.is_file());

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
    assert!(!model.exists());
}

#[cfg(unix)]
#[test]
fn test_a_failed_purge_with_no_config_change_reports_no_removal() {
    use std::os::unix::fs::PermissionsExt;

    let home = tempfile::tempdir().unwrap();
    let (model, _labels) = seed_installed_registry_geomodel(home.path());
    let models = home.path().join("models");
    std::fs::set_permissions(&models, std::fs::Permissions::from_mode(0o555)).unwrap();

    // Root ignores directory permissions, so the deletion would succeed.
    let denied = std::fs::write(models.join("probe"), b"").is_err();
    let output = run_in(
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
    std::fs::set_permissions(&models, std::fs::Permissions::from_mode(0o755)).unwrap();
    if !denied {
        return;
    }

    assert!(!output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "",
        "no config changed and no file was deleted, so no removal may be reported"
    );
    assert!(model.is_file());
}

#[test]
fn test_structured_plain_remove_warns_on_stderr_that_the_files_are_still_used() {
    let home = tempfile::tempdir().unwrap();
    let (model, labels) = seed_installed_registry_geomodel(home.path());
    configure_geomodel(home.path(), &model, &labels);

    let output = ok_in(
        home.path(),
        &["--output-mode", "json", "models", "remove", "geomodel"],
    );

    assert_eq!(payload_of(&output)["result_type"], "model_removed");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(STILL_IN_USE_NOTE),
        "got: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_purge_that_finds_nothing_inside_the_models_dir_says_so() {
    // Both recorded paths are hand-placed elsewhere, so nothing is birda's to delete.
    let home = tempfile::tempdir().unwrap();
    configure_geomodel(
        home.path(),
        &fixture("fixture-geomodel.onnx"),
        &fixture("fixture-geomodel-labels.txt"),
    );

    let output = ok_in(
        home.path(),
        &["models", "remove", "geomodel", "--purge", "--yes"],
    );

    let lines = lines_of(&output);
    assert!(
        lines.iter().any(|l| l == "No geomodel files were deleted."),
        "got: {lines:?}"
    );
    // The models directory does not exist here, so the files are outside it and
    // present: not "not found".
    assert!(
        lines
            .iter()
            .any(|l| l.starts_with("  Skipped (outside the models directory): ")),
        "got: {lines:?}"
    );
    assert!(
        !lines.iter().any(|l| l.contains("not found")),
        "got: {lines:?}"
    );
    assert!(!lines.iter().any(|l| l == "Geomodel files deleted."));
    assert!(fixture("fixture-geomodel.onnx").is_file());
}

#[test]
fn test_a_registry_filename_that_leaves_the_models_dir_is_rejected() {
    // registry.json is user-writable. A filename with `..` must not point purge, or
    // the install, at a file outside the models directory.
    let home = tempfile::tempdir().unwrap();
    seed_installed_registry_geomodel(home.path());
    let path = home.path().join("registry.json");
    let mut registry: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    registry["range_filter"]["model"]["filename"] = Value::from("../escaped.onnx");
    std::fs::write(&path, registry.to_string()).unwrap();
    let outside = home.path().join("escaped.onnx");
    std::fs::write(&outside, b"not birda's").unwrap();

    let output = run_in(
        home.path(),
        &["models", "remove", "geomodel", "--purge", "--yes"],
    );

    assert!(!output.status.success());
    assert!(
        outside.is_file(),
        "a file outside the models dir must survive"
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).trim(),
        "error: configuration validation failed: invalid model filename in registry: \"../escaped.onnx\""
    );
}
