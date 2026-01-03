//! Integration tests for CLI output enhancements.

use assert_cmd::cargo::cargo_bin;
use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

#[test]
fn test_timing_metrics_in_output() {
    let mut cmd = Command::new(cargo_bin("birda"));

    // This test requires a small test audio file
    // We'll skip it if the file doesn't exist
    if !std::path::Path::new("tests/fixtures/test.wav").exists() {
        eprintln!("Skipping test: test audio file not found");
        return;
    }

    cmd.arg("-v")
        .arg("--no-progress")
        .arg("tests/fixtures/test.wav");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("segments/sec"))
        .stdout(predicate::str::contains(" in ").or(predicate::str::contains("duration")));
}

#[test]
fn test_no_progress_flag() {
    let mut cmd = Command::new(cargo_bin("birda"));

    if !std::path::Path::new("tests/fixtures/test.wav").exists() {
        eprintln!("Skipping test: test audio file not found");
        return;
    }

    cmd.arg("-v")
        .arg("--no-progress")
        .arg("tests/fixtures/test.wav");

    // With --no-progress, stderr should not contain progress bar escape codes
    // Progress bars write to stderr by default and use ANSI escape sequences
    // Processing messages go to stdout via tracing
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Processing:").or(predicate::str::contains("Complete:")))
        .stderr(predicate::str::contains("\x1b[").not()); // Verify no ANSI escape sequences in stderr
}

#[test]
fn test_device_selection_logging_cpu() {
    let mut cmd = Command::new(cargo_bin("birda"));

    if !std::path::Path::new("tests/fixtures/test.wav").exists() {
        eprintln!("Skipping test: test audio file not found");
        return;
    }

    cmd.arg("-v")
        .arg("--cpu")
        .arg("--no-progress")
        .arg("tests/fixtures/test.wav");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("CPU"));
}

#[test]
fn test_device_selection_logging_auto() {
    let mut cmd = Command::new(cargo_bin("birda"));

    if !std::path::Path::new("tests/fixtures/test.wav").exists() {
        eprintln!("Skipping test: test audio file not found");
        return;
    }

    cmd.arg("-v")
        .arg("--no-progress")
        .arg("tests/fixtures/test.wav");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Auto").or(predicate::str::contains("device")));
}
