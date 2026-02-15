//! `TensorRT` library detection for graceful fallback.

/// Get the expected `TensorRT` library filename for current platform.
#[allow(dead_code)]
fn get_tensorrt_library_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "nvinfer_10.dll"
    }
    #[cfg(target_os = "linux")]
    {
        "libnvinfer.so.10"
    }
    #[cfg(target_os = "macos")]
    {
        "libnvinfer.10.dylib"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tensorrt_library_name_windows() {
        #[cfg(target_os = "windows")]
        assert_eq!(get_tensorrt_library_name(), "nvinfer_10.dll");
    }

    #[test]
    fn test_get_tensorrt_library_name_linux() {
        #[cfg(target_os = "linux")]
        assert_eq!(get_tensorrt_library_name(), "libnvinfer.so.10");
    }

    #[test]
    fn test_get_tensorrt_library_name_macos() {
        #[cfg(target_os = "macos")]
        assert_eq!(get_tensorrt_library_name(), "libnvinfer.10.dylib");
    }
}
