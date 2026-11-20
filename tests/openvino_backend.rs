//! Integration tests for the arguments and configuration of the native
//! `OpenVINO` backend.
//!
//! None of these needs an `OpenVINO` runtime or ONNX Runtime: every case is
//! decided before either is loaded, which is what the backend promises (an
//! explicit request fails up front, with no run on another engine). Anything
//! that does need a runtime is covered by the hardware recipe, not here.
//!
//! The runtime location is always given explicitly (a path that does not
//! exist), never left to the search, so a machine that happens to have
//! `OpenVINO` installed behaves the same as one that does not.
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

use std::path::Path;
use std::process::Output;
use std::time::Duration;

use assert_cmd::cargo::cargo_bin_cmd;
use birda::constants::CONFIG_DIR_ENV;

/// Bound each run so a hang fails the test instead of stalling the suite.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// Exit code `main` uses for every application error.
const APPLICATION_ERROR: i32 = 1;

/// Exit code clap uses for an argument it rejects.
const USAGE_ERROR: i32 = 2;

/// Run birda with its config directory redirected into `home`.
///
/// Every `BIRDA_*` variable is removed from the child's environment, because
/// each one overrides an argument and would change what is being tested.
fn run_in(home: &Path, args: &[&str]) -> Output {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_DATA_HOME", home.join("data"))
        .env(CONFIG_DIR_ENV, home)
        .timeout(COMMAND_TIMEOUT);
    for (key, _) in std::env::vars() {
        if key.starts_with("BIRDA_") && key != CONFIG_DIR_ENV {
            cmd.env_remove(key);
        }
    }
    cmd.args(args);
    cmd.output().expect("birda should run")
}

/// Write `config.toml` where the binary will read it.
///
/// The directory is the one [`CONFIG_DIR_ENV`] names, so the file lands inside
/// `home` and the developer's real config is never touched.
fn seed_config(home: &Path, contents: &str) {
    std::fs::write(home.join("config.toml"), contents).unwrap();
}

fn stderr_lines(output: &Output) -> Vec<String> {
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .map(str::to_string)
        .collect()
}

/// The first stderr line, which is where `main` prints the error.
fn first_error_line(output: &Output) -> String {
    stderr_lines(output)
        .into_iter()
        .next()
        .expect("birda printed no error")
}

fn missing_runtime_config(home: &Path) -> String {
    let missing = home.join("no-such-openvino");
    seed_config(
        home,
        &format!(
            "[inference]\nopenvino_path = {}\n",
            toml_string(&missing.display().to_string())
        ),
    );
    format!(
        "error: inference.openvino_path is set to '{}', but it does not exist",
        missing.display()
    )
}

/// A TOML basic string, with the escapes Windows paths need.
fn toml_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

#[test]
fn devices_with_a_runtime_path_that_does_not_exist_names_the_config_key() {
    let home = tempfile::tempdir().unwrap();
    let expected = missing_runtime_config(home.path());

    let output = run_in(home.path(), &["openvino", "devices"]);

    assert_eq!(output.status.code(), Some(APPLICATION_ERROR));
    assert_eq!(first_error_line(&output), expected);
}

#[test]
fn an_openvino_run_with_a_missing_runtime_path_fails_before_any_output() {
    let home = tempfile::tempdir().unwrap();
    let expected = missing_runtime_config(home.path());
    let input = home.path().join("input.wav");
    std::fs::write(&input, b"not really audio").unwrap();
    let out_dir = home.path().join("results");

    let output = run_in(
        home.path(),
        &[
            "--openvino",
            "--output-dir",
            out_dir.to_str().unwrap(),
            input.to_str().unwrap(),
        ],
    );

    assert_eq!(output.status.code(), Some(APPLICATION_ERROR));
    // The message is the OpenVINO one, not an ONNX Runtime or model error: the
    // request was refused without a fallback to another engine.
    assert_eq!(first_error_line(&output), expected);
    assert!(!out_dir.exists(), "no result may be written");
}

#[test]
fn the_config_file_can_select_openvino_and_gets_the_same_refusal() {
    let home = tempfile::tempdir().unwrap();
    let missing = home.path().join("no-such-openvino");
    seed_config(
        home.path(),
        &format!(
            "[inference]\ndevice = \"openvino\"\nopenvino_path = {}\n",
            toml_string(&missing.display().to_string())
        ),
    );
    let input = home.path().join("input.wav");
    std::fs::write(&input, b"not really audio").unwrap();

    let output = run_in(home.path(), &[input.to_str().unwrap()]);

    assert_eq!(output.status.code(), Some(APPLICATION_ERROR));
    assert_eq!(
        first_error_line(&output),
        format!(
            "error: inference.openvino_path is set to '{}', but it does not exist",
            missing.display()
        )
    );
}

#[test]
fn the_runtime_path_key_is_ignored_when_openvino_is_not_requested() {
    let home = tempfile::tempdir().unwrap();
    let expected = missing_runtime_config(home.path());
    let input = home.path().join("input.wav");
    std::fs::write(&input, b"not really audio").unwrap();

    let output = run_in(home.path(), &["--cpu", input.to_str().unwrap()]);

    assert!(
        !stderr_lines(&output).contains(&expected),
        "a CPU run must not read the OpenVINO path: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn the_device_flag_without_openvino_is_refused_naming_the_resolved_device() {
    let home = tempfile::tempdir().unwrap();
    let input = home.path().join("input.wav");
    std::fs::write(&input, b"not really audio").unwrap();

    let output = run_in(
        home.path(),
        &[
            "--cuda",
            "--openvino-device",
            "gpu",
            input.to_str().unwrap(),
        ],
    );

    assert_eq!(output.status.code(), Some(APPLICATION_ERROR));
    assert_eq!(
        first_error_line(&output),
        "error: configuration validation failed: --openvino-device needs --openvino \
         (or inference.device = \"openvino\"), but the inference device for this run is cuda"
    );
}

#[test]
fn a_device_spec_that_does_not_parse_is_rejected_by_the_argument_parser() {
    let home = tempfile::tempdir().unwrap();

    let output = run_in(
        home.path(),
        &["--openvino", "--openvino-device", "cpu", "input.wav"],
    );

    assert_eq!(output.status.code(), Some(USAGE_ERROR));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("'cpu' is not an OpenVINO device"),
        "got: {stderr}"
    );
}

#[test]
fn a_config_file_with_a_device_spec_that_does_not_parse_fails_to_load() {
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), "[inference]\nopenvino_device = \"bogus\"\n");

    let output = run_in(home.path(), &["config", "show"]);

    assert_eq!(output.status.code(), Some(APPLICATION_ERROR));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("'bogus' is not an OpenVINO device"),
        "got: {stderr}"
    );
}

#[test]
fn the_devices_help_describes_the_listing() {
    let home = tempfile::tempdir().unwrap();

    let output = run_in(home.path(), &["openvino", "devices", "--help"]);

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("List the Intel GPUs `OpenVINO` can use"),
        "got: {stdout}"
    );
}
