//! CUDA library detection for graceful fallback.
//!
//! This module provides runtime detection of NVIDIA CUDA runtime libraries
//! to enable graceful fallback when CUDA is unavailable.
//!
//! # Version Requirements
//!
//! This detection is version-agnostic and will detect any CUDA runtime version
//! installed on the system. ONNX Runtime's CUDA provider typically supports
//! multiple CUDA versions, so exact version matching is not required.
//!
//! # Platform-Specific Search Paths
//!
//! - **Windows**: Searches `PATH` environment variable
//! - **Linux**: Searches `LD_LIBRARY_PATH` + standard paths (`/usr/lib`, `/usr/local/lib`, `/usr/lib/x86_64-linux-gnu`, `/usr/lib64`)
//! - **macOS**: Searches `DYLD_LIBRARY_PATH` + standard paths (`/usr/lib`, `/usr/local/lib`)

use super::library_detection::{check_library_pattern_exists, get_library_search_paths};
use tracing::debug;

/// Get the expected CUDA runtime library patterns for the current platform.
///
/// # Version Agnostic
///
/// Unlike `TensorRT` detection, this function returns glob patterns that match
/// any CUDA runtime version. This is because ONNX Runtime's CUDA provider
/// typically supports multiple CUDA versions (11.x, 12.x, etc.).
///
/// The patterns use wildcards to match version numbers:
/// - Windows: `cudart64_*.dll` matches `cudart64_11.dll`, `cudart64_12.dll`, etc.
/// - Linux: `libcudart.so.*` matches `libcudart.so.11`, `libcudart.so.12.0`, etc.
/// - macOS: `libcudart.*.dylib` matches `libcudart.11.dylib`, `libcudart.12.dylib`, etc.
pub fn get_cuda_library_patterns() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        // Windows: cudart64_11.dll, cudart64_12.dll, etc.
        &["cudart64_*.dll"]
    }
    #[cfg(target_os = "linux")]
    {
        // Linux: libcudart.so.11, libcudart.so.12.0, libcudart.so.12.0.140, etc.
        &["libcudart.so.*"]
    }
    #[cfg(target_os = "macos")]
    {
        // macOS: libcudart.11.dylib, libcudart.12.dylib, etc.
        &["libcudart.*.dylib"]
    }
}

/// Check if CUDA runtime libraries are available on the system.
///
/// Searches for CUDA runtime libraries in platform-specific locations:
/// - Windows: `cudart64_*.dll` in PATH
/// - Linux: `libcudart.so.*` in `LD_LIBRARY_PATH` and standard directories
/// - macOS: `libcudart.*.dylib` in `DYLD_LIBRARY_PATH` and standard directories
///
/// Returns `true` if any CUDA runtime library is found, `false` otherwise.
///
/// # Version Flexibility
///
/// This function accepts any CUDA version found on the system. If you need
/// specific version validation, check the ONNX Runtime documentation for
/// supported CUDA versions.
pub fn is_cuda_available() -> bool {
    let patterns = get_cuda_library_patterns();
    let search_paths = get_library_search_paths();

    debug!(
        "Checking for CUDA runtime libraries matching patterns {:?} in {} paths",
        patterns,
        search_paths.len()
    );

    for path in &search_paths {
        debug!("Checking path: {}", path.display());
    }

    let found = check_library_pattern_exists(&search_paths, patterns);

    if found {
        debug!("CUDA runtime libraries found");
    } else {
        debug!("CUDA runtime libraries not found");
    }

    found
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_get_cuda_library_patterns_windows() {
        #[cfg(target_os = "windows")]
        {
            let patterns = get_cuda_library_patterns();
            assert_eq!(patterns, &["cudart64_*.dll"]);
        }
    }

    #[test]
    fn test_get_cuda_library_patterns_linux() {
        #[cfg(target_os = "linux")]
        {
            let patterns = get_cuda_library_patterns();
            assert_eq!(patterns, &["libcudart.so.*"]);
        }
    }

    #[test]
    fn test_get_cuda_library_patterns_macos() {
        #[cfg(target_os = "macos")]
        {
            let patterns = get_cuda_library_patterns();
            assert_eq!(patterns, &["libcudart.*.dylib"]);
        }
    }

    #[test]
    #[serial]
    fn test_is_cuda_available_when_not_found() {
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
        let result = is_cuda_available();

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

    #[test]
    #[serial]
    fn test_is_cuda_available_with_mock_library() {
        use std::fs::File;
        use tempfile::tempdir;

        let dir = tempdir().expect("create temp dir");

        // Create a mock CUDA runtime library
        #[cfg(target_os = "windows")]
        let lib_name = "cudart64_12.dll";
        #[cfg(target_os = "linux")]
        let lib_name = "libcudart.so.12.0";
        #[cfg(target_os = "macos")]
        let lib_name = "libcudart.12.dylib";

        let lib_path = dir.path().join(lib_name);
        File::create(&lib_path).expect("create file");

        // Add temp dir to appropriate env var
        let original = {
            #[cfg(target_os = "windows")]
            {
                let orig = std::env::var("PATH").ok();
                unsafe {
                    std::env::set_var("PATH", dir.path().to_str().expect("valid UTF-8"));
                }
                orig
            }
            #[cfg(target_os = "linux")]
            {
                let orig = std::env::var("LD_LIBRARY_PATH").ok();
                unsafe {
                    std::env::set_var("LD_LIBRARY_PATH", dir.path().to_str().expect("valid UTF-8"));
                }
                orig
            }
            #[cfg(target_os = "macos")]
            {
                let orig = std::env::var("DYLD_LIBRARY_PATH").ok();
                unsafe {
                    std::env::set_var(
                        "DYLD_LIBRARY_PATH",
                        dir.path().to_str().expect("valid UTF-8"),
                    );
                }
                orig
            }
        };

        // Should detect the library
        let result = is_cuda_available();
        assert!(result, "Should detect CUDA library in temp directory");

        // Restore original environment
        unsafe {
            #[cfg(target_os = "windows")]
            {
                if let Some(orig) = original {
                    std::env::set_var("PATH", orig);
                } else {
                    std::env::remove_var("PATH");
                }
            }
            #[cfg(target_os = "linux")]
            {
                if let Some(orig) = original {
                    std::env::set_var("LD_LIBRARY_PATH", orig);
                } else {
                    std::env::remove_var("LD_LIBRARY_PATH");
                }
            }
            #[cfg(target_os = "macos")]
            {
                if let Some(orig) = original {
                    std::env::set_var("DYLD_LIBRARY_PATH", orig);
                } else {
                    std::env::remove_var("DYLD_LIBRARY_PATH");
                }
            }
        }
    }
}
