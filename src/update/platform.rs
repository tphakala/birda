//! Platform and build variant detection for asset selection.
//!
//! Determines the correct manifest asset key based on the compile-time
//! target OS, architecture, and cargo features.

/// Returns the manifest asset key for the current platform and build variant.
///
/// Examples: `"linux-x64"`, `"linux-x64-cuda"`, `"windows-x64"`, `"macos-arm64"`.
pub fn asset_key() -> &'static str {
    let base = platform_base();

    if cfg!(feature = "cuda") {
        // CUDA builds have a separate key (only linux-x64 and windows-x64)
        match base {
            "linux-x64" => "linux-x64-cuda",
            "windows-x64" => "windows-x64-cuda",
            // macOS doesn't have CUDA builds; fall back to non-CUDA
            _ => base,
        }
    } else {
        base
    }
}

/// Returns the base platform identifier without variant suffix.
fn platform_base() -> &'static str {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "linux-x64"
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "windows-x64"
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "macos-arm64"
    }
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    )))]
    {
        compile_error!(
            "Unsupported platform for birda update. Supported: linux-x64, windows-x64, macos-arm64"
        );
    }
}

/// Returns the binary filename for the current platform.
pub fn binary_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "birda.exe"
    }
    #[cfg(not(target_os = "windows"))]
    {
        "birda"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_key_is_not_empty() {
        let key = asset_key();
        assert!(!key.is_empty());
    }

    #[test]
    fn test_platform_base_matches_expected() {
        let base = platform_base();
        let valid = ["linux-x64", "windows-x64", "macos-arm64"];
        assert!(valid.contains(&base), "unexpected platform: {base}");
    }

    #[test]
    fn test_binary_name_matches_platform() {
        let name = binary_name();
        if cfg!(target_os = "windows") {
            assert_eq!(name, "birda.exe");
        } else {
            assert_eq!(name, "birda");
        }
    }
}
