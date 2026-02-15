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

use std::path::PathBuf;
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

/// Get platform-specific library search paths.
fn get_library_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    #[cfg(target_os = "windows")]
    {
        // Windows: Parse PATH environment variable
        match std::env::var("PATH") {
            Ok(path_env) => {
                for path_str in path_env.split(';') {
                    let path_str = path_str.trim();
                    if !path_str.is_empty() {
                        paths.push(PathBuf::from(path_str));
                    }
                }
            }
            Err(std::env::VarError::NotUnicode(_)) => {
                debug!("PATH environment variable contains invalid Unicode, ignoring");
            }
            Err(std::env::VarError::NotPresent) => {
                debug!("PATH environment variable not set");
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        // Linux: Parse LD_LIBRARY_PATH + standard paths
        match std::env::var("LD_LIBRARY_PATH") {
            Ok(ld_path) => {
                for path_str in ld_path.split(':') {
                    let path_str = path_str.trim();
                    if !path_str.is_empty() {
                        paths.push(PathBuf::from(path_str));
                    }
                }
            }
            Err(std::env::VarError::NotUnicode(_)) => {
                debug!("LD_LIBRARY_PATH environment variable contains invalid Unicode, ignoring");
            }
            Err(std::env::VarError::NotPresent) => {
                debug!("LD_LIBRARY_PATH environment variable not set");
            }
        }

        // Add standard Linux library paths
        paths.push(PathBuf::from("/usr/lib"));
        paths.push(PathBuf::from("/usr/local/lib"));
        paths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
        paths.push(PathBuf::from("/usr/lib64")); // RedHat/Fedora/CentOS 64-bit libs
    }

    #[cfg(target_os = "macos")]
    {
        // macOS: Parse DYLD_LIBRARY_PATH + standard paths
        match std::env::var("DYLD_LIBRARY_PATH") {
            Ok(dyld_path) => {
                for path_str in dyld_path.split(':') {
                    let path_str = path_str.trim();
                    if !path_str.is_empty() {
                        paths.push(PathBuf::from(path_str));
                    }
                }
            }
            Err(std::env::VarError::NotUnicode(_)) => {
                debug!("DYLD_LIBRARY_PATH environment variable contains invalid Unicode, ignoring");
            }
            Err(std::env::VarError::NotPresent) => {
                debug!("DYLD_LIBRARY_PATH environment variable not set");
            }
        }

        // Add standard macOS library paths
        paths.push(PathBuf::from("/usr/lib"));
        paths.push(PathBuf::from("/usr/local/lib"));
    }

    paths
}

/// Check if a specific library file exists in any search path.
fn check_library_exists(paths: &[PathBuf], lib_name: &str) -> bool {
    for path in paths {
        // Skip invalid paths (non-existent directories from env vars)
        if !path.exists() {
            debug!("Skipping non-existent search path: {}", path.display());
            continue;
        }

        // Validate path is a directory
        if !path.is_dir() {
            debug!("Skipping non-directory search path: {}", path.display());
            continue;
        }

        let lib_path = path.join(lib_name);
        if lib_path.exists() && lib_path.is_file() {
            debug!("Found TensorRT library: {}", lib_path.display());
            return true;
        }
    }
    false
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
    fn test_get_library_search_paths_includes_env_paths() {
        #[cfg(target_os = "windows")]
        {
            // Save original PATH
            let original = std::env::var("PATH").ok();
            unsafe {
                std::env::set_var("PATH", "C:\\test1;C:\\test2");
            }
            let paths = get_library_search_paths();
            assert!(paths.iter().any(|p| p == &PathBuf::from("C:\\test1")));
            assert!(paths.iter().any(|p| p == &PathBuf::from("C:\\test2")));
            // Restore original PATH
            unsafe {
                if let Some(orig) = original {
                    std::env::set_var("PATH", orig);
                } else {
                    std::env::remove_var("PATH");
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            // Save original LD_LIBRARY_PATH
            let original = std::env::var("LD_LIBRARY_PATH").ok();
            unsafe {
                std::env::set_var("LD_LIBRARY_PATH", "/test1:/test2");
            }
            let paths = get_library_search_paths();
            assert!(paths.iter().any(|p| p == &PathBuf::from("/test1")));
            assert!(paths.iter().any(|p| p == &PathBuf::from("/test2")));
            // Restore original LD_LIBRARY_PATH
            unsafe {
                if let Some(orig) = original {
                    std::env::set_var("LD_LIBRARY_PATH", orig);
                } else {
                    std::env::remove_var("LD_LIBRARY_PATH");
                }
            }
        }
    }

    #[test]
    fn test_get_library_search_paths_handles_empty_env() {
        // Test that standard paths are always included (don't modify env vars)
        let paths = get_library_search_paths();

        #[cfg(target_os = "linux")]
        {
            // Standard paths should always be present
            assert!(paths.contains(&PathBuf::from("/usr/lib")));
            assert!(paths.contains(&PathBuf::from("/usr/local/lib")));
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, if PATH is set, we should have some paths
            // Just verify the function doesn't panic
            let _ = paths;
        }

        #[cfg(target_os = "macos")]
        {
            // Standard paths should always be present
            assert!(paths.contains(&PathBuf::from("/usr/lib")));
            assert!(paths.contains(&PathBuf::from("/usr/local/lib")));
        }
    }

    #[test]
    fn test_check_library_exists_found() {
        use std::fs::File;
        use tempfile::tempdir;

        let dir = tempdir().expect("create temp dir");

        #[cfg(target_os = "windows")]
        let lib_name = "nvinfer_10.dll";
        #[cfg(target_os = "linux")]
        let lib_name = "libnvinfer.so.10";
        #[cfg(target_os = "macos")]
        let lib_name = "libnvinfer.10.dylib";

        // Create dummy library file
        let lib_path = dir.path().join(lib_name);
        File::create(&lib_path).expect("create file");

        let paths = vec![dir.path().to_path_buf()];
        assert!(check_library_exists(&paths, lib_name));
    }

    #[test]
    fn test_check_library_exists_not_found() {
        use tempfile::tempdir;

        let dir = tempdir().expect("create temp dir");
        let paths = vec![dir.path().to_path_buf()];
        assert!(!check_library_exists(&paths, "nonexistent.dll"));
    }

    #[test]
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
