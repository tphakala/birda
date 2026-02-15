//! Shared library detection utilities for GPU acceleration libraries.
//!
//! This module provides common functionality for detecting NVIDIA libraries
//! (CUDA, `TensorRT`, etc.) at runtime to enable graceful fallback when libraries
//! are unavailable or version-incompatible.
//!
//! # Platform-Specific Search Paths
//!
//! - **Windows**: Searches `PATH` environment variable
//! - **Linux**: Searches `LD_LIBRARY_PATH` + standard paths (`/usr/lib`, `/usr/local/lib`, `/usr/lib/x86_64-linux-gnu`, `/usr/lib64`)
//! - **macOS**: Searches `DYLD_LIBRARY_PATH` + standard paths (`/usr/lib`, `/usr/local/lib`)

use std::path::PathBuf;
use tracing::debug;

/// Get platform-specific library search paths.
///
/// Returns a vector of directories to search for shared libraries,
/// including both environment-based paths and standard system directories.
pub fn get_library_search_paths() -> Vec<PathBuf> {
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

/// Check if a specific library file exists in any search path (exact match).
///
/// This function performs exact filename matching, useful for version-specific
/// libraries like `TensorRT`.
///
/// # Arguments
///
/// * `paths` - Directories to search
/// * `lib_name` - Exact filename to match (e.g., `"libnvinfer.so.10"`)
///
/// # Returns
///
/// `true` if the library file is found, `false` otherwise.
pub fn check_library_exists(paths: &[PathBuf], lib_name: &str) -> bool {
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
            debug!("Found library: {}", lib_path.display());
            return true;
        }
    }
    false
}

/// Check if any library matching the given patterns exists in search paths.
///
/// This function performs pattern matching, useful for version-agnostic
/// detection like CUDA runtime libraries.
///
/// # Arguments
///
/// * `paths` - Directories to search
/// * `patterns` - Glob patterns to match (e.g., `["cudart64_*.dll", "cudart64_??.dll"]`)
///
/// # Returns
///
/// `true` if any matching library file is found, `false` otherwise.
pub fn check_library_pattern_exists(paths: &[PathBuf], patterns: &[&str]) -> bool {
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

        // Try to read directory entries
        let entries = match std::fs::read_dir(path) {
            Ok(entries) => entries,
            Err(e) => {
                debug!("Cannot read directory {}: {}", path.display(), e);
                continue;
            }
        };

        // Check each entry against patterns
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    debug!("Skipping entry in {}: {}", path.display(), e);
                    continue;
                }
            };
            let file_path = entry.path();

            // Only consider files (not directories or symlinks)
            if !file_path.is_file() {
                continue;
            }

            let Some(file_name) = file_path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };

            // Check if filename matches any pattern
            for pattern in patterns {
                if matches_pattern(file_name, pattern) {
                    debug!(
                        "Found library matching pattern '{}': {}",
                        pattern,
                        file_path.display()
                    );
                    return true;
                }
            }
        }
    }
    false
}

/// Simple glob pattern matcher for library filenames.
///
/// Supports `*` (matches any characters) and `?` (matches single character).
/// Optimized to work with byte slices to avoid heap allocations.
///
/// # Examples
///
/// ```ignore
/// assert!(matches_pattern("cudart64_12.dll", "cudart64_*.dll"));
/// assert!(matches_pattern("libcudart.so.11.8", "libcudart.so.*"));
/// assert!(!matches_pattern("cudnn64_8.dll", "cudart64_*.dll"));
/// ```
fn matches_pattern(filename: &str, pattern: &str) -> bool {
    matches_pattern_impl(filename.as_bytes(), pattern.as_bytes())
}

/// Internal pattern matching implementation working on byte slices.
///
/// This avoids heap allocations by working directly with slices rather than
/// creating new String instances during recursion.
fn matches_pattern_impl(filename: &[u8], pattern: &[u8]) -> bool {
    let mut f_idx = 0;
    let mut p_idx = 0;

    while p_idx < pattern.len() {
        match pattern[p_idx] {
            b'*' => {
                p_idx += 1;
                // '*' at end matches everything
                if p_idx >= pattern.len() {
                    return true;
                }

                // Try matching from current position onwards
                for i in f_idx..=filename.len() {
                    if matches_pattern_impl(&filename[i..], &pattern[p_idx..]) {
                        return true;
                    }
                }
                return false;
            }
            b'?' => {
                if f_idx >= filename.len() {
                    return false; // Pattern expects char but filename exhausted
                }
                f_idx += 1;
                p_idx += 1;
            }
            c => {
                if f_idx >= filename.len() || filename[f_idx] != c {
                    return false; // Mismatch
                }
                f_idx += 1;
                p_idx += 1;
            }
        }
    }

    // Pattern exhausted - filename must also be exhausted for match
    f_idx >= filename.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_pattern_exact() {
        assert!(matches_pattern("libcudart.so.12", "libcudart.so.12"));
        assert!(!matches_pattern("libcudart.so.11", "libcudart.so.12"));
    }

    #[test]
    fn test_matches_pattern_wildcard_star() {
        assert!(matches_pattern("cudart64_12.dll", "cudart64_*.dll"));
        assert!(matches_pattern("cudart64_11.dll", "cudart64_*.dll"));
        assert!(matches_pattern("libcudart.so.11.8", "libcudart.so.*"));
        assert!(matches_pattern("libcudart.so.12.0.140", "libcudart.so.*"));
        assert!(!matches_pattern("cudnn64_8.dll", "cudart64_*.dll"));
    }

    #[test]
    fn test_matches_pattern_wildcard_question() {
        assert!(matches_pattern("cudart64_11.dll", "cudart64_??.dll"));
        assert!(matches_pattern("cudart64_12.dll", "cudart64_??.dll"));
        assert!(!matches_pattern("cudart64_8.dll", "cudart64_??.dll"));
        assert!(!matches_pattern("cudart64_123.dll", "cudart64_??.dll"));
    }

    #[test]
    fn test_matches_pattern_no_wildcards() {
        assert!(matches_pattern("exact.so", "exact.so"));
        assert!(!matches_pattern("different.so", "exact.so"));
    }

    #[test]
    fn test_check_library_exists_with_tempdir() {
        use std::fs::File;
        use tempfile::tempdir;

        let dir = tempdir().expect("create temp dir");
        let lib_path = dir.path().join("test.so");
        File::create(&lib_path).expect("create file");

        let paths = vec![dir.path().to_path_buf()];
        assert!(check_library_exists(&paths, "test.so"));
        assert!(!check_library_exists(&paths, "missing.so"));
    }

    #[test]
    fn test_check_library_pattern_exists_with_tempdir() {
        use std::fs::File;
        use tempfile::tempdir;

        let dir = tempdir().expect("create temp dir");

        // Create test files
        File::create(dir.path().join("cudart64_12.dll")).expect("create file");
        File::create(dir.path().join("cudart64_11.dll")).expect("create file");

        let paths = vec![dir.path().to_path_buf()];
        assert!(check_library_pattern_exists(&paths, &["cudart64_*.dll"]));
        assert!(!check_library_pattern_exists(&paths, &["cudnn64_*.dll"]));
    }

    #[test]
    fn test_get_library_search_paths_not_empty() {
        let paths = get_library_search_paths();

        // Should have at least standard paths on Linux/macOS
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            assert!(!paths.is_empty());
            assert!(paths.contains(&PathBuf::from("/usr/lib")));
        }
    }
}
