//! ONNX Runtime startup checks.

use crate::constants::onnx_runtime;
use crate::error::{Error, Result};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Ensure ONNX Runtime is discoverable before inference-related code touches `ort`.
///
/// This prevents the downstream `ort` loader from blocking indefinitely when the
/// dynamic library is missing at runtime.
pub fn ensure_runtime_available() -> Result<()> {
    let resolved_path = resolve_runtime_library_path()?;

    if resolved_path.is_some() {
        birdnet_onnx::init_runtime().map_err(|e| Error::RuntimeInitialization {
            reason: e.to_string(),
        })?;
        return Ok(());
    }

    if runtime_exists_in_default_search_paths() {
        return Ok(());
    }

    Err(missing_runtime_error())
}

fn resolve_runtime_library_path() -> Result<Option<PathBuf>> {
    if let Some(path) = env_override_path()? {
        return Ok(Some(path));
    }

    Ok(birdnet_onnx::find_ort_library())
}

fn env_override_path() -> Result<Option<PathBuf>> {
    std::env::var_os(onnx_runtime::DYLIB_PATH_ENV).map_or_else(
        || Ok(None),
        |value| {
            let path = PathBuf::from(&value);
            if path.is_file() {
                Ok(Some(path))
            } else {
                Err(invalid_env_override_error(&value))
            }
        },
    )
}

fn runtime_exists_in_default_search_paths() -> bool {
    search_path_directories()
        .into_iter()
        .chain(common_search_directories())
        .any(|dir| runtime_exists_in_directory(&dir))
}

fn search_path_directories() -> Vec<PathBuf> {
    std::env::var_os(onnx_runtime::SEARCH_PATH_ENV)
        .map(|value| std::env::split_paths(&value).collect())
        .unwrap_or_default()
}

fn common_search_directories() -> impl Iterator<Item = PathBuf> {
    onnx_runtime::COMMON_SEARCH_DIRS
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>()
        .into_iter()
}

fn runtime_exists_in_directory(directory: &Path) -> bool {
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir(directory) {
            return entries.flatten().any(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(onnx_runtime::LIBRARY_FILE_NAME)
            });
        }
        false
    }

    #[cfg(not(target_os = "linux"))]
    {
        directory.join(onnx_runtime::LIBRARY_FILE_NAME).is_file()
    }
}

fn invalid_env_override_error(value: &OsString) -> Error {
    let display = PathBuf::from(value).display().to_string();
    Error::RuntimeInitialization {
        reason: format!(
            "{} points to '{}', but that file does not exist. Set it to the full path of {}.",
            onnx_runtime::DYLIB_PATH_ENV,
            display,
            onnx_runtime::LIBRARY_FILE_NAME
        ),
    }
}

fn missing_runtime_error() -> Error {
    let searched_dirs = onnx_runtime::COMMON_SEARCH_DIRS.join(", ");
    Error::RuntimeInitialization {
        reason: format!(
            "could not locate {}. Install ONNX Runtime or set {} to the full path of the library. Also checked {} and common library directories ({})",
            onnx_runtime::LIBRARY_FILE_NAME,
            onnx_runtime::DYLIB_PATH_ENV,
            onnx_runtime::SEARCH_PATH_ENV,
            searched_dirs
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_exists_in_directory_matches_exact_file() {
        let temp_dir = tempfile::tempdir().expect("temp dir should be created");
        let library_path = temp_dir.path().join(onnx_runtime::LIBRARY_FILE_NAME);
        std::fs::write(&library_path, []).expect("library file should be created");

        assert!(runtime_exists_in_directory(temp_dir.path()));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_runtime_exists_in_directory_matches_versioned_linux_library() {
        let temp_dir = tempfile::tempdir().expect("temp dir should be created");
        let library_path = temp_dir
            .path()
            .join(format!("{}.1.22.0", onnx_runtime::LIBRARY_FILE_NAME));
        std::fs::write(&library_path, []).expect("library file should be created");

        assert!(runtime_exists_in_directory(temp_dir.path()));
    }
}
