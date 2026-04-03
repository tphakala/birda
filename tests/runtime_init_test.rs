//! Regression tests for ONNX Runtime startup failures.

use std::time::Duration;

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

/// Timeout for startup failure checks.
const STARTUP_FAILURE_TIMEOUT: Duration = Duration::from_secs(5);

/// Missing ONNX Runtime path used to force fast startup failure.
const MISSING_ORT_DYLIB_PATH: &str = "/definitely/missing/onnxruntime";

#[test]
fn test_invalid_ort_dylib_path_exits_with_error() {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.arg("providers")
        .env("ORT_DYLIB_PATH", MISSING_ORT_DYLIB_PATH)
        .timeout(STARTUP_FAILURE_TIMEOUT);

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains(
            "failed to initialize ONNX runtime",
        ))
        .stderr(predicate::str::contains("ORT_DYLIB_PATH"))
        .stderr(predicate::str::contains("does not exist"));
}
