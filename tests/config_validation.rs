//! Integration tests for configuration validation on load (#295).
//!
//! These drive the real binary rather than calling `validate_config` directly,
//! and that is the point. Every rule in `src/config/validate.rs` already
//! existed and every unit test over it already passed; the defect was that
//! nothing on the load path ever called them. `validate_config` had exactly one
//! non-test caller, `handle_config_set`, so a value typed through the CLI (where
//! clap has usually already checked it) was validated and a value hand-edited
//! into `config.toml` (where it is most likely to be wrong) was not.
//!
//! A unit test cannot tell those two states apart, because both call the same
//! passing function. Only spawning the binary against a seeded `config.toml`
//! exercises the wiring, which is where the defect lived.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use assert_cmd::cargo::cargo_bin_cmd;

/// These commands are local (no network, no ONNX runtime), so they should be
/// quick. Bound them anyway so a hang fails rather than stalls the suite.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// A config that fails validation for a reason unrelated to the range filter,
/// so the tests do not all depend on one rule.
const OUT_OF_RANGE_LATITUDE: &str = "[defaults]\nlatitude = 200.0\n";

/// The reported case: NaN dropped every mapped species, because `score >= NaN`
/// is false for every score, and the only symptom was an info log reading
/// "0 species in range".
const NAN_RANGE_THRESHOLD: &str = "[defaults]\nrange_threshold = nan\n";

/// Run birda against a caller-supplied HOME.
///
/// `directories` resolves the config directory from `HOME` on macOS and from
/// `XDG_CONFIG_HOME` (falling back to `HOME`) on Linux, so both are set. Without
/// this the suite would read and rewrite the developer's real config.toml.
fn run_in(home: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_DATA_HOME", home.join("data"))
        // `--output-mode` reads BIRDA_OUTPUT_MODE. A developer with that
        // exported would turn the human-output assertions into failures, so the
        // isolation has to cover the environment, not just the filesystem.
        .env_remove("BIRDA_OUTPUT_MODE")
        .timeout(COMMAND_TIMEOUT);
    for arg in args {
        cmd.arg(arg);
    }

    cmd.output().expect("birda should run")
}

/// Ask the binary where its config lives rather than recomputing the platform
/// rules here. A test that hardcoded `~/.config/birda/config.toml` would seed a
/// file the binary never reads on macOS and then pass vacuously.
fn config_path_in(home: &Path) -> PathBuf {
    let output = run_in(home, &["config", "path"]);
    assert!(
        output.status.success(),
        "`config path` failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    PathBuf::from(String::from_utf8_lossy(&output.stdout).trim())
}

/// Write `contents` to the config file the binary will actually load.
fn seed_config(home: &Path, contents: &str) -> PathBuf {
    let path = config_path_in(home);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, contents).unwrap();
    path
}

fn stderr_of(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn test_out_of_range_latitude_is_rejected_on_load() {
    // Accepted on load before the fix. `validate_range_filter` has checked
    // latitude since long before this change; nothing called it.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), OUT_OF_RANGE_LATITUDE);

    let output = run_in(home.path(), &["models", "list"]);

    assert!(
        !output.status.success(),
        "an out-of-range latitude must not load clean"
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains("invalid latitude"),
        "the error should name the offending value, got: {stderr}"
    );
}

#[test]
fn test_nan_range_threshold_is_rejected_on_load() {
    // Reproduced in #295: `birda config show` printed `range_threshold: NaN`
    // and exited 0, and `models check` reported OK, while an analyze run
    // returned only the unmatched non-species labels.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), NAN_RANGE_THRESHOLD);

    let output = run_in(home.path(), &["models", "list"]);

    assert!(
        !output.status.success(),
        "a NaN range threshold must not load clean"
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains("invalid range threshold"),
        "the error should name the offending value, got: {stderr}"
    );
}

#[test]
fn test_config_show_still_works_with_an_invalid_config() {
    // The repair surface. Validating here would mean the only route out of a
    // broken config is the hand-editing that broke it, so `config` is exempt.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), OUT_OF_RANGE_LATITUDE);

    let output = run_in(home.path(), &["config", "show"]);

    assert!(
        output.status.success(),
        "`config show` must survive an invalid config: {}",
        stderr_of(&output)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("200.0"),
        "`config show` should display the offending value so it can be found, got: {stdout}"
    );
}

#[test]
fn test_config_set_repairs_an_invalid_config() {
    // The escape hatch end to end: the exemption is only worth having if the
    // exempt commands can actually get the user back to a loadable config.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), OUT_OF_RANGE_LATITUDE);

    let repair = run_in(
        home.path(),
        &["config", "set", "defaults.latitude", "60.17"],
    );
    assert!(
        repair.status.success(),
        "`config set` must be able to fix the value: {}",
        stderr_of(&repair)
    );

    let after = run_in(home.path(), &["models", "list"]);
    assert!(
        after.status.success(),
        "the repaired config should load: {}",
        stderr_of(&after)
    );
}

#[test]
fn test_config_set_refuses_to_persist_an_invalid_value() {
    // The save-side half of the change. `config set` validates through
    // `save_config`, so a bad value is rejected before the file is rewritten
    // and the previous contents survive.
    let home = tempfile::tempdir().unwrap();
    let path = seed_config(home.path(), "[defaults]\nlatitude = 60.17\n");
    let before = std::fs::read_to_string(&path).unwrap();

    let output = run_in(home.path(), &["config", "set", "defaults.latitude", "200"]);

    assert!(
        !output.status.success(),
        "`config set` must reject an out-of-range latitude"
    );
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        before,
        "a rejected `config set` must leave the file untouched"
    );
}

#[test]
fn test_bare_invocation_prints_help_with_an_invalid_config() {
    // Refusing to print help because the config is wrong is the least useful
    // moment to refuse, so the no-command no-inputs path is exempt too.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), OUT_OF_RANGE_LATITUDE);

    let output = run_in(home.path(), &[]);

    assert!(
        output.status.success(),
        "the bare invocation should still print help: {}",
        stderr_of(&output)
    );
}

#[test]
fn test_a_valid_config_still_loads() {
    // The control. Without it every assertion above would pass just as well if
    // the new check rejected every config it was handed.
    let home = tempfile::tempdir().unwrap();
    seed_config(
        home.path(),
        "[defaults]\nlatitude = 60.17\nlongitude = 24.94\nrange_threshold = 0.03\n",
    );

    let output = run_in(home.path(), &["models", "list"]);

    assert!(
        output.status.success(),
        "a valid config must still load: {}",
        stderr_of(&output)
    );
}
