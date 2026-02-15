//! `TensorRT` library detection for graceful fallback.

use std::path::PathBuf;

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

/// Get platform-specific library search paths.
#[allow(dead_code)]
fn get_library_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    #[cfg(target_os = "windows")]
    {
        // Windows: Parse PATH environment variable
        if let Ok(path_env) = std::env::var("PATH") {
            for path_str in path_env.split(';') {
                let path_str = path_str.trim();
                if !path_str.is_empty() {
                    paths.push(PathBuf::from(path_str));
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        // Linux: Parse LD_LIBRARY_PATH + standard paths
        if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
            for path_str in ld_path.split(':') {
                let path_str = path_str.trim();
                if !path_str.is_empty() {
                    paths.push(PathBuf::from(path_str));
                }
            }
        }

        // Add standard Linux library paths
        paths.push(PathBuf::from("/usr/lib"));
        paths.push(PathBuf::from("/usr/local/lib"));
        paths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
    }

    #[cfg(target_os = "macos")]
    {
        // macOS: Parse DYLD_LIBRARY_PATH + standard paths
        if let Ok(dyld_path) = std::env::var("DYLD_LIBRARY_PATH") {
            for path_str in dyld_path.split(':') {
                let path_str = path_str.trim();
                if !path_str.is_empty() {
                    paths.push(PathBuf::from(path_str));
                }
            }
        }

        // Add standard macOS library paths
        paths.push(PathBuf::from("/usr/lib"));
        paths.push(PathBuf::from("/usr/local/lib"));
    }

    paths
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
}
