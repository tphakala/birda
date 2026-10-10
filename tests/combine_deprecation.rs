//! `--combine` is accepted, hidden, and warned about.
// Integration test crate. `unwrap`, `expect` and `panic` are how a test reports
// failure, not unhandled error paths, so rewriting them into propagated errors
// would only hide which assertion fired. The crate-level deny still governs
// everything birda ships.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use assert_cmd::cargo::cargo_bin_cmd;
use birda::constants::{COMBINE_DEPRECATED_WARNING, CONFIG_DIR_ENV};

#[test]
fn test_combine_warns_that_it_does_nothing() {
    let home = tempfile::tempdir().unwrap();
    let missing = home.path().join("missing.wav");

    let output = cargo_bin_cmd!("birda")
        .env(CONFIG_DIR_ENV, home.path())
        .arg("--combine")
        .arg(&missing)
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    let warning_lines: Vec<&str> = stderr
        .lines()
        .filter(|line| line.contains("--combine"))
        .collect();
    assert_eq!(
        warning_lines.len(),
        1,
        "expected one --combine line on stderr, got: {stderr}"
    );
    assert!(
        warning_lines[0].ends_with(COMBINE_DEPRECATED_WARNING),
        "got: {}",
        warning_lines[0]
    );
}

#[test]
fn test_no_warning_without_the_flag() {
    let home = tempfile::tempdir().unwrap();
    let missing = home.path().join("missing.wav");

    let output = cargo_bin_cmd!("birda")
        .env(CONFIG_DIR_ENV, home.path())
        .arg(&missing)
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains(COMBINE_DEPRECATED_WARNING),
        "got: {stderr}"
    );
}
