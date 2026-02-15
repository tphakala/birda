//! Integration tests for TensorRT library detection.

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
        assert!(lib_name.ends_with(".dylib"));
    }
}
