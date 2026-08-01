//! Integration tests for `TensorRT` library detection.
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

use birda::inference::{get_tensorrt_library_name, is_tensorrt_available};

#[test]
fn test_tensorrt_detection_doesnt_panic() {
    // Should not panic regardless of whether TensorRT is installed
    let _available = is_tensorrt_available();
    // If this completes, the detection logic is safe
}

#[test]
fn test_tensorrt_library_name_platform_specific() {
    let lib_name = get_tensorrt_library_name();

    #[cfg(target_os = "windows")]
    {
        assert_eq!(lib_name, "nvinfer_10.dll");
        assert!(lib_name.ends_with(".dll"));
    }

    #[cfg(target_os = "linux")]
    {
        assert_eq!(lib_name, "libnvinfer.so.10");
        assert!(lib_name.starts_with("lib"));
        assert!(lib_name.contains(".so"));
    }

    #[cfg(target_os = "macos")]
    {
        assert_eq!(lib_name, "libnvinfer.10.dylib");
        assert!(lib_name.starts_with("lib"));
        // `contains` rather than `ends_with`, matching the Linux arm above and
        // `tests/cuda_detection_test.rs`. See the comment there: Linux CI never
        // compiled this arm, so the lint survived the sweep in #338.
        assert!(lib_name.contains(".dylib"));
    }
}
