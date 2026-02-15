//! `TensorRT` library detection for graceful fallback.
//!
//! This module provides runtime detection of NVIDIA `TensorRT` 10.x libraries
//! to enable graceful fallback when `TensorRT` is unavailable or version-incompatible.
//!
//! # Version Requirements
//!
//! Only `TensorRT` 10.x is supported due to ABI compatibility requirements with
//! the `birdnet-onnx` crate bindings. Earlier versions will not be detected.
//!
//! # Platform-Specific Search Paths
//!
//! - **Windows**: Searches `PATH` environment variable
//! - **Linux**: Searches `LD_LIBRARY_PATH` + standard paths (`/usr/lib`, `/usr/local/lib`, `/usr/lib/x86_64-linux-gnu`, `/usr/lib64`)
//! - **macOS**: Searches `DYLD_LIBRARY_PATH` + standard paths (`/usr/lib`, `/usr/local/lib`)

use super::library_detection::{check_library_exists, get_library_search_paths};
use tracing::debug;

/// Get the expected `TensorRT` library filename for current platform.
///
/// # `TensorRT` Version Requirement
///
/// This function explicitly checks for `TensorRT` 10.x to match the `birdnet-onnx`
/// crate's compiled bindings. Earlier `TensorRT` versions (8.x, 9.x) will NOT be
/// detected even if installed, as they are ABI-incompatible with the bindings.
///
/// If you have an older `TensorRT` version installed, the detection will return
/// `false` and execution will fall back to CPU or other available providers.
pub fn get_tensorrt_library_name() -> &'static str {
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

/// Check if `TensorRT` libraries are available on the system.
///
/// Searches for `TensorRT` runtime libraries in platform-specific locations:
/// - Windows: `nvinfer_10.dll` in PATH
/// - Linux: `libnvinfer.so.10` in `LD_LIBRARY_PATH` and standard directories
/// - macOS: `libnvinfer.10.dylib` in `DYLD_LIBRARY_PATH` and standard directories
///
/// Returns `true` if the library is found, `false` otherwise.
pub fn is_tensorrt_available() -> bool {
    let lib_name = get_tensorrt_library_name();
    let search_paths = get_library_search_paths();

    debug!(
        "Checking for TensorRT library '{}' in {} paths",
        lib_name,
        search_paths.len()
    );

    for path in &search_paths {
        debug!("Checking path: {}", path.display());
    }

    let found = check_library_exists(&search_paths, lib_name);

    if found {
        debug!("TensorRT libraries found");
    } else {
        debug!("TensorRT libraries not found");
    }

    found
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use serial_test::serial;

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

    #[test]
    #[serial]
    fn test_is_tensorrt_available_when_not_found() {
        // Save original environment
        let original_path = std::env::var("PATH").ok();
        let original_ld = std::env::var("LD_LIBRARY_PATH").ok();
        let original_dyld = std::env::var("DYLD_LIBRARY_PATH").ok();

        // Clear environment to ensure no paths
        unsafe {
            std::env::remove_var("PATH");
            std::env::remove_var("LD_LIBRARY_PATH");
            std::env::remove_var("DYLD_LIBRARY_PATH");
        }

        // Should return false when library not in standard paths
        let result = is_tensorrt_available();

        // Restore original environment
        unsafe {
            if let Some(orig) = original_path {
                std::env::set_var("PATH", orig);
            }
            if let Some(orig) = original_ld {
                std::env::set_var("LD_LIBRARY_PATH", orig);
            }
            if let Some(orig) = original_dyld {
                std::env::set_var("DYLD_LIBRARY_PATH", orig);
            }
        }

        let _ = result;
    }
}
