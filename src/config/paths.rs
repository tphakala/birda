//! Platform-specific configuration paths.

use crate::constants::{APP_NAME, tensorrt};
use crate::error::{Error, Result};
use directories::ProjectDirs;
use std::path::PathBuf;

/// Get the configuration directory for the current platform.
///
/// - Linux: `~/.config/birda/`
/// - macOS: `~/Library/Application Support/birda/`
/// - Windows: `%APPDATA%\birda\`
pub fn config_dir() -> Result<PathBuf> {
    ProjectDirs::from("", "", APP_NAME)
        .map(|dirs| dirs.config_dir().to_path_buf())
        .ok_or(Error::ConfigDirNotFound)
}

/// Get the full path to the config file.
pub fn config_file_path() -> Result<PathBuf> {
    Ok(config_dir()?.join("config.toml"))
}

/// Get the `TensorRT` cache directory for engine and timing caches.
///
/// - Linux: `~/.config/birda/tensorrt_cache/`
/// - macOS: `~/Library/Application Support/birda/tensorrt_cache/`
/// - Windows: `%APPDATA%\birda\tensorrt_cache\`
pub fn tensorrt_cache_dir() -> Result<PathBuf> {
    Ok(config_dir()?.join(tensorrt::CACHE_DIR))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_config_dir_returns_path() {
        let result = config_dir();
        assert!(result.is_ok());
        let path = result.ok();
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(path.to_string_lossy().contains("birda"));
    }

    #[test]
    fn test_config_file_path_ends_with_toml() {
        let result = config_file_path();
        assert!(result.is_ok());
        let path = result.ok().unwrap();
        assert!(path.to_string_lossy().ends_with("config.toml"));
    }

    #[test]
    fn test_tensorrt_cache_dir_returns_path() {
        let result = tensorrt_cache_dir();
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.to_string_lossy().contains("birda"));
        assert!(path.ends_with(tensorrt::CACHE_DIR));
    }
}
