//! Platform-specific configuration paths.

use crate::constants::{APP_NAME, CONFIG_DIR_ENV, tensorrt};
use crate::error::{Error, Result};
use directories::ProjectDirs;
use std::ffi::OsString;
use std::path::PathBuf;

/// Returns the [`CONFIG_DIR_ENV`] override root, if it is set to a non-empty
/// value.
///
/// Both [`config_dir`] and [`data_dir`] honour this, so the integration suite
/// can redirect every path birda reads or writes into a temporary directory on
/// any platform, Windows included. See [`CONFIG_DIR_ENV`].
fn config_dir_override() -> Option<PathBuf> {
    override_root(std::env::var_os(CONFIG_DIR_ENV))
}

/// Pure core of [`config_dir_override`], split out so the empty/unset handling
/// can be unit-tested without mutating the process environment.
///
/// An unset variable and one set to the empty string both mean "no override";
/// the value is otherwise used verbatim so a path containing spaces survives.
fn override_root(value: Option<OsString>) -> Option<PathBuf> {
    value.filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// Get the configuration directory for the current platform.
///
/// Honours the [`CONFIG_DIR_ENV`] override when it is set. Otherwise:
/// - Linux: `~/.config/birda/`
/// - macOS: `~/Library/Application Support/birda/`
/// - Windows: `%APPDATA%\birda\config\`
pub fn config_dir() -> Result<PathBuf> {
    if let Some(dir) = config_dir_override() {
        return Ok(dir);
    }
    ProjectDirs::from("", "", APP_NAME)
        .map(|dirs| dirs.config_dir().to_path_buf())
        .ok_or(Error::ConfigDirNotFound)
}

/// Get the data directory for the current platform.
///
/// Home of installed models. Honours the [`CONFIG_DIR_ENV`] override when set,
/// returning the same root as [`config_dir`]. Otherwise:
/// - Linux: `~/.local/share/birda/`
/// - macOS: `~/Library/Application Support/birda/`
/// - Windows: `%APPDATA%\birda\data\`
pub fn data_dir() -> Result<PathBuf> {
    if let Some(dir) = config_dir_override() {
        return Ok(dir);
    }
    ProjectDirs::from("", "", APP_NAME)
        .map(|dirs| dirs.data_dir().to_path_buf())
        .ok_or(Error::DataDirNotFound)
}

/// Get the cache directory for the current platform.
///
/// Deliberately NOT covered by the [`CONFIG_DIR_ENV`] override: it holds only
/// the regenerable `TensorRT` engine cache, which no test writes, so redirecting
/// it would widen the override's scope for no isolation benefit.
///
/// - Linux: `~/.cache/birda/`
/// - macOS: `~/Library/Caches/birda/`
/// - Windows: `%LOCALAPPDATA%\birda\cache\`
pub fn cache_dir() -> Result<PathBuf> {
    ProjectDirs::from("", "", APP_NAME)
        .map(|dirs| dirs.cache_dir().to_path_buf())
        .ok_or(Error::CacheDirNotFound)
}

/// Get the full path to the config file.
pub fn config_file_path() -> Result<PathBuf> {
    Ok(config_dir()?.join("config.toml"))
}

/// Get the `TensorRT` cache directory for engine and timing caches.
///
/// Uses the platform cache directory since `TensorRT` engines are:
/// - Large binary files (can be 100MB+)
/// - Machine/GPU-specific (not portable)
/// - Safely regenerable if deleted
///
/// - Linux: `~/.cache/birda/tensorrt_cache/`
/// - macOS: `~/Library/Caches/birda/tensorrt_cache/`
/// - Windows: `%LOCALAPPDATA%\birda\cache\tensorrt_cache\`
pub fn tensorrt_cache_dir() -> Result<PathBuf> {
    Ok(cache_dir()?.join(tensorrt::CACHE_DIR))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn override_root_unset_is_none() {
        assert_eq!(override_root(None), None);
    }

    #[test]
    fn override_root_empty_is_none() {
        // An exported-but-empty variable must not redirect anywhere.
        assert_eq!(override_root(Some(OsString::new())), None);
    }

    #[test]
    fn override_root_uses_value_verbatim() {
        // Paths with spaces are legitimate, so the value is not trimmed.
        let raw = "/tmp/birda test home";
        assert_eq!(
            override_root(Some(OsString::from(raw))),
            Some(PathBuf::from(raw))
        );
    }

    #[test]
    #[serial]
    fn override_redirects_config_and_data_dirs() {
        // Pins the override branch on every platform, including Linux CI where
        // the integration suites also set HOME/XDG at the same dir and would
        // therefore stay green even if this short-circuit were dropped (#328).
        // The override wins over the real platform dirs: both resolve to `root`,
        // not `~/.config/birda`, and config_dir and data_dir collapse together.
        let root = tempfile::TempDir::new().unwrap();
        temp_env::with_var(CONFIG_DIR_ENV, Some(root.path()), || {
            assert_eq!(config_dir().unwrap().as_path(), root.path());
            assert_eq!(data_dir().unwrap().as_path(), root.path());
        });
    }

    #[test]
    #[serial]
    fn test_config_dir_returns_path() {
        let result = config_dir();
        assert!(result.is_ok());
        let path = result.ok();
        assert!(path.is_some());
        let path = path.unwrap();
        // Under the override the path is the caller's root, which need not carry
        // the app name; only the default platform layout does.
        if config_dir_override().is_none() {
            assert!(path.to_string_lossy().contains("birda"));
        }
    }

    #[test]
    #[serial]
    fn test_data_dir_returns_path() {
        let result = data_dir();
        assert!(result.is_ok());
        let path = result.unwrap();
        if config_dir_override().is_none() {
            assert!(path.to_string_lossy().contains("birda"));
        }
    }

    #[test]
    fn test_cache_dir_returns_path() {
        let result = cache_dir();
        assert!(result.is_ok());
        let path = result.unwrap();
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

    #[test]
    fn test_tensorrt_cache_uses_cache_dir_not_config() {
        let cache = cache_dir().unwrap();
        let tensorrt = tensorrt_cache_dir().unwrap();
        // TensorRT cache should be under the cache directory, not config
        assert!(tensorrt.starts_with(&cache));
    }
}
