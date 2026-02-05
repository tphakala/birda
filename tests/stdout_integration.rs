//! Integration tests for --stdout mode.

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

    cmd.assert().failure().stderr(predicate::str::contains(
        "--stdout cannot be used with --output-dir",
    ));
}

#[test]
fn test_stdout_conflicts_with_combine() {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.arg("--stdout").arg("--combine").arg("test.wav");

    cmd.assert().failure().stderr(predicate::str::contains(
        "--stdout cannot be used with --combine",
    ));
}

#[test]
fn test_stdout_conflicts_with_format() {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.arg("--stdout")
        .arg("--format")
        .arg("csv")
        .arg("test.wav");

    cmd.assert().failure().stderr(predicate::str::contains(
        "--stdout cannot be used with --format",
    ));
}
