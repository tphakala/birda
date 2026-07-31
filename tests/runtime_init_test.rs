//! Regression tests for ONNX Runtime startup failures.
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
