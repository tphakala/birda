//! Integration tests for --stdout mode.
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

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

#[test]
fn test_stdout_requires_single_file() {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.arg("--stdout").arg("file1.wav").arg("file2.wav");

    cmd.assert().failure().stderr(predicate::str::contains(
        "--stdout requires exactly one input file",
    ));
}

#[test]
fn test_stdout_conflicts_with_output_dir() {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.arg("--stdout")
        .arg("--output-dir")
        .arg("/tmp")
        .arg("test.wav");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--stdout"))
        .stderr(predicate::str::contains("--output-dir"))
        .stderr(predicate::str::contains("cannot be used with"));
}

#[test]
fn test_stdout_conflicts_with_combine() {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.arg("--stdout").arg("--combine").arg("test.wav");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--stdout"))
        .stderr(predicate::str::contains("--combine"))
        .stderr(predicate::str::contains("cannot be used with"));
}

#[test]
fn test_stdout_conflicts_with_format() {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.arg("--stdout")
        .arg("--format")
        .arg("csv")
        .arg("test.wav");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--stdout"))
        .stderr(predicate::str::contains("--format"))
        .stderr(predicate::str::contains("cannot be used with"));
}
