//! Integration tests for configuration validation on load (#295), for the
//! overlap rule reaching every input channel rather than only the file (#306),
//! and for the same parity on `batch_size` and `day_of_year` (#312).
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

/// An empty output-format list (#339).
///
/// The reason this rule needs a binary-level test more than its siblings do:
/// the symptom it fixes was an exit code, not a message. Before the rule, this
/// config made a run log "Skipping (output exists)" for every input file and
/// exit 0 having written nothing, so a unit test on `validate_config` can show
/// the rule fires but not that the run now stops.
const EMPTY_FORMATS: &str = "[defaults]\nformats = []\n";

/// A CSV column name no writer has an arm for (#339).
const UNKNOWN_CSV_COLUMN: &str = "[defaults.csv_columns]\ninclude = [\"lattitude\"]\n";

/// Every `BIRDA_*` variable clap binds to an argument.
///
/// All of them are cleared per run. These tests assert on what the config file
/// produces, and each of these overrides a config value, so any one of them
/// exported in a developer's shell would silently change what is being tested
/// rather than failing outright. Enumerated rather than cleared wholesale
/// because the child still needs `HOME`, `PATH` and the rest of its
/// environment.
///
/// Kept in sync with `grep -rhoE 'env = "BIRDA_[A-Z_]+"' src/`.
const BIRDA_ENV_VARS: [&str; 19] = [
    "BIRDA_BATCH_SIZE",
    "BIRDA_DAY_OF_YEAR",
    "BIRDA_FORMAT",
    "BIRDA_GEOMODEL_LABELS_PATH",
    "BIRDA_GEOMODEL_PATH",
    "BIRDA_LABELS_PATH",
    "BIRDA_LATITUDE",
    "BIRDA_LONGITUDE",
    "BIRDA_META_MODEL_PATH",
    "BIRDA_MIN_CONFIDENCE",
    "BIRDA_MODEL",
    "BIRDA_MODEL_PATH",
    "BIRDA_MODEL_TYPE",
    "BIRDA_OUTPUT_DIR",
    "BIRDA_OUTPUT_MODE",
    "BIRDA_OVERLAP",
    "BIRDA_RANGE_THRESHOLD",
    "BIRDA_RANGE_UNMATCHED",
    "BIRDA_SPECIES_LIST",
];

/// Run birda against a caller-supplied HOME.
///
/// `directories` resolves the config directory from `HOME` on macOS and from
/// `XDG_CONFIG_HOME` (falling back to `HOME`) on Linux, so both are set. Without
/// this the suite would read and rewrite the developer's real config.toml.
/// `seed_config` proves the redirection actually took effect before writing
/// anything; see the comment there.
///
/// The isolation has to cover the environment as well as the filesystem, hence
/// [`BIRDA_ENV_VARS`].
fn run_in(home: &Path, args: &[&str]) -> std::process::Output {
    run_in_with_env(home, &[], args)
}

/// [`run_in`], with `env` applied after the wholesale clearing.
///
/// The order matters: [`BIRDA_ENV_VARS`] is removed first so a variable
/// exported in the developer's shell cannot leak in, and the caller's overrides
/// go on afterwards. A test that set `BIRDA_OVERLAP` before the loop would have
/// it removed again and then pass vacuously against the config-file default.
fn run_in_with_env(home: &Path, env: &[(&str, &str)], args: &[&str]) -> std::process::Output {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_DATA_HOME", home.join("data"))
        .timeout(COMMAND_TIMEOUT);
    for var in BIRDA_ENV_VARS {
        cmd.env_remove(var);
    }
    for (key, value) in env {
        cmd.env(key, value);
    }
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

/// Read one `[defaults]` key back through the structured envelope.
///
/// Asserted through JSON rather than by parsing config.toml, for the reason
/// `test_config_show_still_works_with_an_invalid_config` gives: this payload is
/// what birda-gui consumes, so it is the surface worth pinning.
///
/// Returns `Value::Null` for a cleared key, whichever way it is cleared:
/// `day_of_year` carries `skip_serializing_if` so it disappears from the
/// payload, `batch_size` carries no serde attribute so it appears as an
/// explicit null, and indexing yields `Value::Null` either way.
fn config_value(home: &Path, key: &str) -> serde_json::Value {
    let shown = run_in(home, &["config", "show", "--output-mode", "json"]);
    assert!(
        shown.status.success(),
        "`config show` failed: {}",
        stderr_of(&shown)
    );
    let envelope: serde_json::Value = serde_json::from_str(&String::from_utf8_lossy(&shown.stdout))
        .expect("`config show` must emit a parseable envelope");
    envelope["payload"]["config"]["defaults"][key].clone()
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

/// Assert config validation did not stop an analyze run.
///
/// Stated as the absence of every rule message rather than as a positive
/// assertion on whatever comes next, because what comes next is not invariant.
/// `run()` initialises the ONNX runtime after validation but before it resolves
/// a model, so on a machine with the shared library present the run reaches "no
/// model specified", and on one without it stops at the runtime instead. CI is
/// the second kind, so pinning the later stage made this environment-dependent.
///
/// Absence is only safe here because each message is a shared constant that a
/// rejection test asserts positively. A reword therefore fails loudly there
/// instead of silently switching this check off, which is the trap an earlier
/// revision fell into with a list nothing positively pinned.
fn assert_validation_did_not_reject(home: &Path) {
    let input = dummy_input(home);
    let stderr = stderr_of(&run_in(home, &[input.to_str().unwrap()]));

    for message in [
        LATITUDE_VIOLATION,
        THRESHOLD_VIOLATION,
        OVERLAP_VIOLATION,
        BATCH_SIZE_VIOLATION,
        DAY_OF_YEAR_VIOLATION,
        EMPTY_FORMATS_VIOLATION,
        UNKNOWN_CSV_COLUMN_VIOLATION,
    ] {
        assert!(
            !stderr.contains(message),
            "a valid config must not be rejected ({message}), got: {stderr}"
        );
    }
}

/// A config that passes validation, for tests whose subject is not the file.
///
/// Named rather than repeated inline because the call to `seed_config` is
/// load-bearing even when the contents are irrelevant: it carries the assertion
/// that the temp HOME actually redirected the config directory. Without it a
/// test reaches the developer's real config.toml on any platform where the
/// redirection does not work. `test_a_valid_overlap_flag_is_accepted` explains
/// this at length; the constant keeps the next reader from having to find it.
const VALID_SEED: &str = "[defaults]\nbatch_size = 8\n";

/// The rule messages this suite drives, exactly as the user sees them.
///
/// Each is used twice: asserted positively by the rejection test for its rule,
/// and asserted absent by the control. That pairing is what keeps the control
/// honest, since a reworded message breaks its rejection test loudly rather
/// than quietly ceasing to match.
const LATITUDE_VIOLATION: &str = "invalid latitude";
const THRESHOLD_VIOLATION: &str = "invalid range threshold";
const OVERLAP_VIOLATION: &str = "overlap must be a finite non-negative number";
const BATCH_SIZE_VIOLATION: &str = "batch_size must be between";
const DAY_OF_YEAR_VIOLATION: &str = "day_of_year must be between";
const EMPTY_FORMATS_VIOLATION: &str = "defaults.formats must list at least one output format";
const UNKNOWN_CSV_COLUMN_VIOLATION: &str = "has an unknown column 'lattitude'";

/// The tail `cli::validators::parse_batch_size` adds and the config-file rule
/// does not.
///
/// Asserted on so the `config set` test pins its own layer. Both layers reject
/// an oversized batch size and both exit 1, so a test that only checked for
/// [`BATCH_SIZE_VIOLATION`] would pass just as well when the arm falls through
/// to the whole-config validation inside `save_config`, which is the state
/// before #312.
const BATCH_SIZE_CLI_ADVICE: &str = "GPU memory exhaustion";

/// The prefix `handle_config_set` puts on a value it refused itself.
///
/// Same purpose as [`BATCH_SIZE_CLI_ADVICE`]: it names the key the user typed,
/// which neither `validate_defaults` nor clap can do.
const CONFIG_SET_REJECTION: &str = "invalid value for";

/// The exit code birda uses for an application error, as opposed to a clap
/// parse failure (2) or a panic (101). Pinned so a rejection test cannot be
/// satisfied by the wrong kind of failure.
const APPLICATION_ERROR: i32 = 1;

/// The exit code clap uses when an argument fails its `value_parser`.
///
/// The overlap tests below pin this rather than [`APPLICATION_ERROR`], because
/// the whole point of #306 is that the flag and the environment variable are
/// rejected by the same mechanism as every sibling option, which is clap. A
/// rejection that arrived later, as an application error, would mean the value
/// parser is still not wired.
const CLAP_USAGE_ERROR: i32 = 2;

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

    assert_analysis_rejected(home.path(), LATITUDE_VIOLATION);
}

#[test]
fn test_an_empty_format_list_is_rejected_on_load() {
    // #339. The failure this replaces was silent: every input file was reported
    // as "Skipping (output exists)", the run summary said "0 processed", and
    // the exit code was 0. `assert_analysis_rejected` pins the exit code as
    // well as the message, which is the half a unit test cannot reach and the
    // half that was actually wrong.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), EMPTY_FORMATS);

    assert_analysis_rejected(home.path(), EMPTY_FORMATS_VIOLATION);
}

#[test]
fn test_an_unknown_csv_column_is_rejected_on_load() {
    // #339, the same no-CLI-route shape one key over. 'lattitude' reached the
    // CSV header verbatim and produced a column empty in every row, while the
    // Parquet writer dropped it, so the same config gave two different answers.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), UNKNOWN_CSV_COLUMN);

    assert_analysis_rejected(home.path(), UNKNOWN_CSV_COLUMN_VIOLATION);
}

#[test]
fn test_neither_new_rule_has_a_config_set_arm_to_repair_it() {
    // Pins the reason both messages carry recovery advice that the sibling
    // rules do not. `config set` cannot reach either key, and because
    // `save_config` validates the whole file, the fault also blocks `config
    // set` on every OTHER key. Hand-editing config.toml is the only way out,
    // and the README says so; this is what stops that becoming untrue quietly.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), EMPTY_FORMATS);

    // Both keys, because the name says both and the advice covers both. Written
    // with only `defaults.formats` first, this stayed green if someone added a
    // `defaults.csv_columns` arm, which would make half of `EDIT_THE_FILE`
    // false while nothing noticed.
    for key in ["defaults.formats", "defaults.csv_columns"] {
        let direct = run_in(home.path(), &["config", "set", key, "csv"]);
        assert_eq!(direct.status.code(), Some(APPLICATION_ERROR));
        assert!(
            stderr_of(&direct).contains("unknown configuration key"),
            "there is no arm for {key}, got: {}",
            stderr_of(&direct)
        );
    }

    let other = run_in(
        home.path(),
        &["config", "set", "defaults.min_confidence", "0.2"],
    );
    assert_eq!(other.status.code(), Some(APPLICATION_ERROR));
    assert!(
        stderr_of(&other).contains(EMPTY_FORMATS_VIOLATION),
        "an unrelated key must also be refused while the file is invalid, got: {}",
        stderr_of(&other)
    );

    // The recovery advice itself is user-facing output with nothing else
    // asserting a word of it.
    assert!(
        stderr_of(&other).contains("birda config path"),
        "the message must point at the file, got: {}",
        stderr_of(&other)
    );
}

#[test]
fn test_nan_range_threshold_is_rejected_on_load() {
    // Reproduced in #295: `birda config show` printed `range_threshold: NaN`
    // and exited 0, while an analyze run returned only the unmatched
    // non-species labels.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), NAN_RANGE_THRESHOLD);

    assert_analysis_rejected(home.path(), THRESHOLD_VIOLATION);
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

    assert_validation_did_not_reject(home.path());
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

    assert_analysis_rejected(home.path(), OVERLAP_VIOLATION);
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

/// Assert clap refused the run, naming the offending value.
///
/// Takes the whole invocation rather than just the value, because #306 and #312
/// are both about input channels agreeing, and the flag and the environment
/// variable reach clap by different routes.
///
/// `expected` is the rule message. For `OVERLAP_VIOLATION` and
/// `DAY_OF_YEAR_VIOLATION` it is the same string the config-file rule emits,
/// which is the claim being made: one rule, whichever route the value arrives
/// by.
///
/// `BATCH_SIZE_VIOLATION` matches on the upper bound only. At the lower bound
/// the flag says "`batch_size` must be at least 1" where the file says "must be
/// between 1 and 512", so passing it here with `--batch-size 0` would fail.
/// Same verdict, different sentence; `src/config/validate.rs` records why.
fn assert_flag_rejected(home: &Path, env: &[(&str, &str)], args: &[&str], expected: &str) {
    let input = dummy_input(home);
    let mut full: Vec<&str> = args.to_vec();
    let input = input.to_str().unwrap();
    full.push(input);

    let output = run_in_with_env(home, env, &full);

    assert_eq!(
        output.status.code(),
        Some(CLAP_USAGE_ERROR),
        "an invalid value must be refused by the value parser, got: {}",
        stderr_of(&output)
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains(expected),
        "the flag and the config file should give the same diagnostic \
         ({expected}), got: {stderr}"
    );
}

#[test]
fn test_a_nan_overlap_flag_is_rejected() {
    // #306. `--overlap` was the one numeric option on `AnalyzeArgs` with no
    // value parser, so it took whatever `f32::from_str` accepted. NaN reached
    // `overlap * sample_rate`, Rust saturates that cast rather than trapping,
    // and the setting became a silent zero: the run completed, produced
    // non-overlapping segments and said nothing.
    let home = tempfile::tempdir().unwrap();

    assert_flag_rejected(home.path(), &[], &["--overlap", "nan"], OVERLAP_VIOLATION);
}

#[test]
fn test_an_infinite_overlap_flag_is_rejected() {
    // The worst of the three, because it changed the result rather than
    // ignoring the setting. The same saturating cast sends infinity to
    // `usize::MAX`, `chunk_samples.saturating_sub(overlap_samples)` gives a
    // step of 0, and the run yields no segments at all: exit 0, empty output,
    // no diagnostic.
    let home = tempfile::tempdir().unwrap();

    assert_flag_rejected(home.path(), &[], &["--overlap", "inf"], OVERLAP_VIOLATION);
}

#[test]
fn test_a_negative_overlap_flag_is_rejected() {
    // Written as a separate argument rather than `--overlap=-1` to pin
    // `allow_hyphen_values`. Without it clap consumes the leading `-` as the
    // start of another flag and fails with "a value is required", which is a
    // usage error about the wrong thing: the user did supply a value, it was
    // just not a legal one. `--lat` and `--lon` carry the same attribute for
    // the same reason.
    let home = tempfile::tempdir().unwrap();

    assert_flag_rejected(home.path(), &[], &["--overlap", "-1"], OVERLAP_VIOLATION);
}

#[test]
fn test_the_overlap_env_var_is_validated() {
    // The channel the config-file fix did not reach. `BIRDA_OVERLAP` and
    // `--overlap` both win over the config value, so a rule enforced only on
    // the file left the two disagreeing: `overlap = -1` in config.toml was a
    // hard error while `BIRDA_OVERLAP=-1` was accepted and silently became zero
    // overlap. Driven through the environment because clap applies the value
    // parser to both sources and only an end-to-end run proves it.
    let home = tempfile::tempdir().unwrap();

    assert_flag_rejected(
        home.path(),
        &[("BIRDA_OVERLAP", "-1")],
        &[],
        OVERLAP_VIOLATION,
    );
}

#[test]
fn test_a_valid_overlap_flag_is_accepted() {
    // The control for the four above, which would all pass just as well if the
    // value parser rejected everything. 1.5 is finite and non-negative, so it
    // must reach the run; that it exceeds nothing is the point, since no upper
    // bound is imposed here (an oversized overlap is caught against the segment
    // length in `AudioDecoder::next_segment`, which is the only place that
    // knows it).
    //
    // `seed_config` is called even though the config content is irrelevant,
    // because it carries the assertion that the temp HOME actually redirected
    // the config directory. Without it this test reaches config load against
    // the developer's real config.toml on any platform where the redirection
    // does not work, and a real `overlap = nan` there would fail it spuriously.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), "[defaults]\noverlap = 0.0\n");
    let input = dummy_input(home.path());

    let output = run_in(home.path(), &["--overlap", "1.5", input.to_str().unwrap()]);

    // Asserted as "not a usage error", not merely as the absence of the
    // rule message. Absence alone passes when the flag does not exist: renaming
    // `--overlap` to `--overlapx` kills the four rejection tests above and this
    // one still passed, so it was a control against the parser being too strict
    // and not against the flag being unwired at all. Exit code 2 is what clap
    // returns for both "invalid value" and "unexpected argument", so pinning it
    // closes both.
    assert_ne!(
        output.status.code(),
        Some(CLAP_USAGE_ERROR),
        "a finite non-negative overlap must be accepted by the value parser, got: {}",
        stderr_of(&output)
    );
    assert!(
        !stderr_of(&output).contains(OVERLAP_VIOLATION),
        "a finite non-negative overlap must not be rejected, got: {}",
        stderr_of(&output)
    );
}

#[test]
fn test_an_oversized_batch_size_is_rejected_on_load() {
    // #312. `parse_batch_size` has capped the flag at 512 since #135 added the
    // cap, and config.toml was checked for `Some(0)` and nothing else, so
    // `--batch-size 100000` was refused while this exact value, hand-edited,
    // reached the inference path. The cap is there to prevent GPU memory
    // exhaustion, so the unchecked route was the one that could hang the
    // machine.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), "[defaults]\nbatch_size = 100000\n");

    assert_analysis_rejected(home.path(), BATCH_SIZE_VIOLATION);
}

#[test]
fn test_an_out_of_range_day_of_year_is_rejected_on_load() {
    // The other #312 field, and the wider hole of the two: `validate_defaults`
    // did not look at it at all and `config set` had no arm for it, so
    // config.toml was the only route to `defaults.day_of_year` and the only one
    // with no check. (`--day-of-year` and `BIRDA_DAY_OF_YEAR` set it for a
    // single run and were both bounded.) 999 reached the BSG SDM path and was
    // rejected there by birdnet-onnx, a long way from the key that carried it.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), "[defaults]\nday_of_year = 999\n");

    assert_analysis_rejected(home.path(), DAY_OF_YEAR_VIOLATION);
}

#[test]
fn test_the_day_of_year_flag_is_rejected_by_the_value_parser() {
    // The flag was already bounded, by an inline `range(1..=366)`. This pins
    // that swapping it for `parse_day_of_year` did not loosen it, which is what
    // makes the shared parser safe to depend on: clap's range and the config
    // rule were two literals, and only one of them existed.
    let home = tempfile::tempdir().unwrap();

    assert_flag_rejected(
        home.path(),
        &[],
        &["--day-of-year", "999"],
        DAY_OF_YEAR_VIOLATION,
    );
}

#[test]
fn test_the_day_of_year_env_var_is_validated() {
    // `BIRDA_DAY_OF_YEAR` wins over the config value the same way
    // `BIRDA_OVERLAP` does, so it gets the same treatment. clap applies the
    // value parser to both sources and only an end-to-end run proves it.
    let home = tempfile::tempdir().unwrap();

    assert_flag_rejected(
        home.path(),
        &[("BIRDA_DAY_OF_YEAR", "0")],
        &[],
        DAY_OF_YEAR_VIOLATION,
    );
}

#[test]
fn test_a_valid_day_of_year_flag_is_accepted() {
    // The control for the two above, which would both pass against a parser
    // that rejected everything. 366 is deliberate: the last day of a leap year
    // is a real date, and a bound written as 365 would refuse it.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), VALID_SEED);
    let input = dummy_input(home.path());

    let output = run_in(
        home.path(),
        &["--day-of-year", "366", input.to_str().unwrap()],
    );

    // "Not a usage error" rather than the absence of the message, for the
    // reason `test_a_valid_overlap_flag_is_accepted` documents: absence alone
    // also passes when the flag does not exist at all, and clap returns 2 for
    // both "invalid value" and "unexpected argument".
    assert_ne!(
        output.status.code(),
        Some(CLAP_USAGE_ERROR),
        "a day of year inside the range must be accepted, got: {}",
        stderr_of(&output)
    );
    assert!(
        !stderr_of(&output).contains(DAY_OF_YEAR_VIOLATION),
        "a day of year inside the range must not be rejected, got: {}",
        stderr_of(&output)
    );
}

#[test]
fn test_config_set_rejects_an_oversized_batch_size() {
    // The third route to the same setting, and the one #312 names explicitly:
    // the arm parsed a bare `usize` and left the bound to the whole-config
    // validation inside `save_config`, which did not carry it.
    //
    // Asserted on the parser's own wording rather than on "rejected", because
    // the save-side rule now refuses this value too and exits the same way.
    // Without pinning CONFIG_SET_REJECTION and BATCH_SIZE_CLI_ADVICE, both of
    // which only the arm can produce, this test passes with the arm reverted to
    // `value.parse::<usize>()` and proves nothing about the layer it names.
    let home = tempfile::tempdir().unwrap();
    let path = seed_config(home.path(), VALID_SEED);
    let before = std::fs::read_to_string(&path).unwrap();

    let output = run_in(
        home.path(),
        &["config", "set", "defaults.batch_size", "100000"],
    );

    assert_eq!(
        output.status.code(),
        Some(APPLICATION_ERROR),
        "`config set` must reject an oversized batch size as an application \
         error, got: {}",
        stderr_of(&output)
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains(CONFIG_SET_REJECTION) && stderr.contains(BATCH_SIZE_CLI_ADVICE),
        "the rejection should come from the shared value parser, naming the key \
         and carrying its advice, got: {stderr}"
    );
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        before,
        "a rejected `config set` must leave the file untouched"
    );
}

#[test]
fn test_config_set_writes_a_valid_day_of_year() {
    // A new route rather than a repaired one: the key had no arm, so this
    // failed with "invalid configuration key" and hand-editing config.toml was
    // the only way in.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), VALID_SEED);

    let output = run_in(
        home.path(),
        &["config", "set", "defaults.day_of_year", "200"],
    );
    assert!(
        output.status.success(),
        "`config set defaults.day_of_year` must be accepted: {}",
        stderr_of(&output)
    );

    // Pin the value that landed, not just the exit code. An arm that parsed the
    // value and then stored a default would satisfy a success assertion.
    assert_eq!(
        config_value(home.path(), "day_of_year"),
        200,
        "the stored day of year should be the one that was set"
    );
}

#[test]
fn test_config_set_rejects_an_out_of_range_day_of_year() {
    // The rejecting half of the arm above. Pinned on CONFIG_SET_REJECTION
    // because `validate_defaults` now emits the same rule message with the same
    // exit code, so the key prefix is the only thing that tells the two apart.
    let home = tempfile::tempdir().unwrap();
    let path = seed_config(home.path(), VALID_SEED);
    let before = std::fs::read_to_string(&path).unwrap();

    let output = run_in(
        home.path(),
        &["config", "set", "defaults.day_of_year", "999"],
    );

    assert_eq!(
        output.status.code(),
        Some(APPLICATION_ERROR),
        "`config set` must reject an out-of-range day of year as an \
         application error, got: {}",
        stderr_of(&output)
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains(CONFIG_SET_REJECTION) && stderr.contains(DAY_OF_YEAR_VIOLATION),
        "the rejection should name the key and the rule, got: {stderr}"
    );
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        before,
        "a rejected `config set` must leave the file untouched"
    );
}

#[test]
fn test_the_batch_size_flag_is_rejected_by_the_value_parser() {
    // The route README's "agree across their routes" claim rested on with no
    // test behind it: `batch_size` had end-to-end coverage for the file and for
    // `config set`, and none for the flag or the environment variable.
    //
    // Driven at the UPPER bound deliberately. That is where the two routes
    // share wording, so `BATCH_SIZE_VIOLATION` matches both; at the lower bound
    // they diverge, which `assert_flag_rejected`'s doc records.
    let home = tempfile::tempdir().unwrap();

    assert_flag_rejected(
        home.path(),
        &[],
        &["--batch-size", "100000"],
        BATCH_SIZE_VIOLATION,
    );
}

#[test]
fn test_the_batch_size_env_var_is_validated() {
    // `BIRDA_BATCH_SIZE` wins over the config value, so it gets the same
    // treatment as `BIRDA_OVERLAP` and `BIRDA_DAY_OF_YEAR`. clap applies the
    // value parser to both sources and only an end-to-end run proves it.
    let home = tempfile::tempdir().unwrap();

    assert_flag_rejected(
        home.path(),
        &[("BIRDA_BATCH_SIZE", "100000")],
        &[],
        BATCH_SIZE_VIOLATION,
    );
}

#[test]
fn test_config_set_writes_a_valid_batch_size() {
    // The accept side its day_of_year twin already had. Without it the arm is
    // only ever driven with a value it must refuse, so an arm that rejected
    // everything would pass the suite.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), VALID_SEED);

    let output = run_in(home.path(), &["config", "set", "defaults.batch_size", "32"]);
    assert!(
        output.status.success(),
        "`config set defaults.batch_size 32` must be accepted: {}",
        stderr_of(&output)
    );

    assert_eq!(
        config_value(home.path(), "batch_size"),
        32,
        "the stored batch size should be the one that was set"
    );
}

#[test]
fn test_config_set_clears_the_day_of_year() {
    // The empty-value branch of the new arm, which is the half that decides
    // between `None` and a parsed value. Nothing else drives it, and it is the
    // only way to get back to "auto-detect from the file timestamp" without
    // hand-editing.
    let home = tempfile::tempdir().unwrap();
    seed_config(home.path(), "[defaults]\nday_of_year = 200\n");

    let output = run_in(home.path(), &["config", "set", "defaults.day_of_year", ""]);
    assert!(
        output.status.success(),
        "clearing the key must be accepted: {}",
        stderr_of(&output)
    );

    assert!(
        config_value(home.path(), "day_of_year").is_null(),
        "an empty value must clear the key, not store a zero or a default"
    );
}

#[test]
fn test_config_set_rejects_an_out_of_range_min_confidence() {
    // A rewired arm that is neither of the two #312 fields. Five of the seven
    // numeric arms moved from a bare `value.parse()` to the shared validators
    // in this change, and without this only `batch_size`, `day_of_year` and
    // `latitude` were driven at all.
    //
    // Pinned on CONFIG_SET_REJECTION because `save_config`'s whole-config
    // validation rejects 1.5 as well, with `min_confidence must be between`;
    // only the arm names the key the user typed. So this fails if the arm is
    // reverted to a bare parse, which is the layer it is here to cover.
    let home = tempfile::tempdir().unwrap();
    let path = seed_config(home.path(), VALID_SEED);
    let before = std::fs::read_to_string(&path).unwrap();

    let output = run_in(
        home.path(),
        &["config", "set", "defaults.min_confidence", "1.5"],
    );

    assert_eq!(
        output.status.code(),
        Some(APPLICATION_ERROR),
        "`config set` must reject an out-of-range confidence as an application \
         error, got: {}",
        stderr_of(&output)
    );
    assert!(
        stderr_of(&output).contains(CONFIG_SET_REJECTION),
        "the arm should refuse it and name the key, got: {}",
        stderr_of(&output)
    );
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        before,
        "a rejected `config set` must leave the file untouched"
    );
}

#[test]
fn test_a_valid_config_still_loads() {
    // The control. Without it every assertion above would pass just as well if
    // the new check rejected every config it was handed. Analysis may still
    // fail further on (no model is configured here), so this asserts only that
    // it got past validation.
    // `batch_size` and `day_of_year` are set at their inclusive maxima rather
    // than left out. Omitted, `validate_defaults` short-circuits on `None` and
    // the two rule messages this control asserts absent could never have
    // appeared, so those assertions held by construction. At 512 and 366 they
    // are real, and the upper bounds get end-to-end coverage they otherwise
    // have only at unit level.
    let home = tempfile::tempdir().unwrap();
    seed_config(
        home.path(),
        "[defaults]\nlatitude = 60.17\nlongitude = 24.94\nrange_threshold = 0.03\n\
         batch_size = 512\nday_of_year = 366\n",
    );

    assert_validation_did_not_reject(home.path());
}
