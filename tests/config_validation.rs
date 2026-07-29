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
/// `seed_config` proves the redirection actually took effect before writing
/// anything; see the comment there.
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
///
/// The assertion is load-bearing, not a sanity check. `HOME` and `XDG_*`
/// redirect `directories` on macOS and Linux only: on Windows it resolves
/// through `SHGetKnownFolderPath`, which consults no environment variable at
/// all, so the path handed back would be the developer's real
/// `%APPDATA%\birda\config.toml` and this function would overwrite it with a
/// two-line stub, destroying every model they had configured. Refusing to write
/// outside the temp HOME turns that from silent destruction into a failure that
/// names its own cause, on any platform where the redirection stops working.
fn seed_config(home: &Path, contents: &str) -> PathBuf {
    let path = config_path_in(home);
    assert!(
        path.starts_with(home),
        "config path {} escaped the test HOME {}; refusing to write. \
         The environment no longer redirects this platform's config directory, \
         so writing here would clobber the real user config.",
        path.display(),
        home.display()
    );

    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, contents).unwrap();
    path
}

/// An input path that makes the run take the analyze branch.
///
/// Analysis is the only gated command, so every "must be rejected" assertion
/// has to go through it. The file only has to exist: validation happens in
/// `run()` before the runtime is initialised and long before anything tries to
/// decode audio, so its contents are never read.
fn dummy_input(home: &Path) -> PathBuf {
    let path = home.join("input.wav");
    std::fs::write(&path, b"not really audio").unwrap();
    path
}

fn stderr_of(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Assert the analyze path refused to start, naming the offending value.
///
/// The exit code is pinned rather than tested with `!success`, because a panic
/// (101), a clap failure (2) or the timeout kill would all satisfy the looser
/// form while proving nothing about validation. This follows the convention
/// `tests/geomodel_discoverability.rs` already established for the same reason.
fn assert_analysis_rejected(home: &Path, expected: &str) {
    let input = dummy_input(home);
    let output = run_in(home, &[input.to_str().unwrap()]);

    assert_eq!(
        output.status.code(),
        Some(APPLICATION_ERROR),
        "analysis must be rejected as an application error, got: {}",
        stderr_of(&output)
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains(expected),
        "the error should name the offending value ({expected}), got: {stderr}"
    );
}

/// Assert an analyze run cleared config validation and failed further in.
fn assert_analysis_passes_validation(home: &Path) {
    let input = dummy_input(home);
    let stderr = stderr_of(&run_in(home, &[input.to_str().unwrap()]));

    assert!(
        stderr.contains(PAST_VALIDATION_ERROR),
        "analysis should have reached model resolution, meaning it cleared \
         validation; expected {PAST_VALIDATION_ERROR:?}, got: {stderr}"
    );
}

/// The error an analyze run reaches once it is past config validation.
///
/// These tests install no model, so this is where a run that cleared the gate
/// lands. Asserting it positively is what makes the control meaningful: an
/// assertion that merely failed to find a validation error would also pass if
/// the binary died for any other reason, or if the error text drifted.
const PAST_VALIDATION_ERROR: &str = "no model specified";

/// The exit code birda uses for an application error, as opposed to a clap
/// parse failure (2) or a panic (101). Pinned so a rejection test cannot be
/// satisfied by the wrong kind of failure.
const APPLICATION_ERROR: i32 = 1;

/// Assert a command ran to completion, so it was not gated on the config.
///
/// A plain success assertion rather than the absence of a validation message.
/// Absence fails open twice over: it passes when the command breaks for an
/// unrelated reason, and it silently stops testing anything if the message it
/// looks for is ever reworded.
fn assert_command_succeeds(home: &Path, args: &[&str]) {
    let output = run_in(home, args);
    assert!(
        output.status.success(),
        "`birda {}` must run despite the invalid config, got {:?}: {}",
        args.join(" "),
        output.status.code(),
        stderr_of(&output)
    );
}

#[test]
fn test_out_of_range_latitude_is_rejected_on_load() {
    // Accepted on load before the fix. `validate_range_filter` has checked
    // latitude since long before this change; nothing called it.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), OUT_OF_RANGE_LATITUDE);

    assert_analysis_rejected(home.path(), "invalid latitude");
}

#[test]
fn test_nan_range_threshold_is_rejected_on_load() {
    // Reproduced in #295: `birda config show` printed `range_threshold: NaN`
    // and exited 0, while an analyze run returned only the unmatched
    // non-species labels.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), NAN_RANGE_THRESHOLD);

    assert_analysis_rejected(home.path(), "invalid range threshold");
}

#[test]
fn test_config_show_still_works_with_an_invalid_config() {
    // The repair surface. Validating here would mean the only route out of a
    // broken config is the hand-editing that broke it, so `config` is exempt.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), OUT_OF_RANGE_LATITUDE);

    // Asserted through the JSON envelope rather than the human output. The
    // structured payload is what birda-gui reads, and it is the only route
    // that shows a GUI user which value is wrong; the human form would pin the
    // assertion to `{config:#?}` Debug formatting, so a cosmetic change to how
    // floats print would break the test while the behaviour was intact.
    let output = run_in(home.path(), &["config", "show", "--output-mode", "json"]);

    assert!(
        output.status.success(),
        "`config show` must survive an invalid config: {}",
        stderr_of(&output)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let envelope: serde_json::Value =
        serde_json::from_str(&stdout).expect("`config show` must emit a parseable envelope");

    assert_eq!(
        envelope["payload"]["config"]["defaults"]["latitude"], 200.0,
        "`config show` should report the offending value so it can be found, got: {stdout}"
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

    // Pin that the requested value actually landed, not merely that the command
    // exited 0. A `config set` that persisted defaults instead would clear the
    // fault and satisfy every other assertion here.
    let shown = run_in(home.path(), &["config", "show", "--output-mode", "json"]);
    let envelope: serde_json::Value = serde_json::from_str(&String::from_utf8_lossy(&shown.stdout))
        .expect("`config show` must emit a parseable envelope");
    assert_eq!(
        envelope["payload"]["config"]["defaults"]["latitude"], 60.17,
        "the repaired value should be the one that was set"
    );

    assert_analysis_passes_validation(home.path());
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

    assert_eq!(
        output.status.code(),
        Some(APPLICATION_ERROR),
        "`config set` must reject an out-of-range latitude as an application \
         error; a panic or a parse failure would satisfy a bare !success check \
         while proving nothing, got: {}",
        stderr_of(&output)
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

    // Exiting 0 is not the claim; printing the help is. Without this the test
    // passes just as well against a `print_smart_help` that returns silently.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("birda models list-available"),
        "help should still guide a user with no models configured, got: {stdout}"
    );
}

#[test]
fn test_repair_and_diagnostic_commands_are_not_gated() {
    // The blast-radius guard. An earlier revision of this change gated every
    // subcommand, which blocked `models install` (the repair for a
    // `defaults.model` that names a missing model), and blocked `update`, the
    // route to a build that would fix the config handling, over a config none
    // of them reads.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), OUT_OF_RANGE_LATITUDE);

    // Only hermetic, local commands are driven here. Every exempt command
    // shares the single gate call site in `run()`, so proving the wiring with
    // one of them proves it for all; which commands are exempt is a table fact,
    // covered by the unit tests over the predicate. `update --check` and `clip`
    // were both in this loop briefly and neither belonged: `update --check`
    // performs a real release lookup and exits 1 with no network, so it would
    // fail for an offline developer and under GitHub API rate limiting on
    // shared CI runners, and `clip` currently exits 0 for a file that does not
    // exist, so asserting its success would pin a production oddity that ought
    // to be fixed.
    for args in [
        vec!["models", "list"],
        vec!["models", "check"],
        vec!["config", "path"],
        vec!["config", "show"],
    ] {
        assert_command_succeeds(home.path(), &args);
    }
}

#[test]
fn test_an_overlap_rule_violation_is_rejected_on_load() {
    // `validate_config` runs two halves and only `validate_range_filter` was
    // reached end to end: swapping the gate call to `validate_range_filter`
    // alone left the whole suite green. This drives a `validate_defaults` rule
    // through the binary, so the gate is pinned to the full check rather than
    // half of it.
    //
    // Overlap is the rule to use, because it is the one this branch tightened:
    // `overlap < 0.0` accepted NaN, which the saturating cast to `usize` then
    // turned into a silently ignored setting.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), "[defaults]\noverlap = nan\n");

    assert_analysis_rejected(home.path(), "overlap must be a finite non-negative number");
}

#[test]
fn test_a_dangling_default_model_does_not_block_the_models_commands() {
    // The one fault `config set` cannot fix by correcting the offending key:
    // `defaults.model` names a model that is not in the file, and installing
    // that model is the natural repair. Gating `models` made the repair
    // unreachable, so this pins it open.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), "[defaults]\nmodel = \"gone-model\"\n");

    assert_command_succeeds(home.path(), &["models", "list"]);
    assert_command_succeeds(home.path(), &["models", "check"]);
}

#[test]
fn test_a_valid_config_still_loads() {
    // The control. Without it every assertion above would pass just as well if
    // the new check rejected every config it was handed. Analysis may still
    // fail further on (no model is configured here), so this asserts only that
    // it got past validation.
    let home = tempfile::tempdir().unwrap();
    seed_config(
        home.path(),
        "[defaults]\nlatitude = 60.17\nlongitude = 24.94\nrange_threshold = 0.03\n",
    );

    assert_analysis_passes_validation(home.path());
}
