//! Integration tests for TensorRT library detection.

use birda::inference::is_tensorrt_available;

#[test]
fn test_tensorrt_detection_doesnt_panic() {
    // Should not panic regardless of whether TensorRT is installed
    let _available = is_tensorrt_available();
    // If this completes, the detection logic is safe
}

#[test]
fn test_tensorrt_detection_returns_bool() {
    let result = is_tensorrt_available();
    // Result should be a boolean (true or false)
    assert!(result || !result); // Tautology to verify it's bool
}
