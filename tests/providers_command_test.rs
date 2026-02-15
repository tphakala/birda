//! Integration tests for providers command.

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

#[test]
fn test_providers_command_human_readable() {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.arg("providers");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Available execution providers:"))
        .stdout(predicate::str::contains("CPU"));
}
