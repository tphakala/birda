//! Binary self-replacement logic.
//!
//! On Unix: rename current binary to `.old`, move new binary into place.
//! On Windows: use `self_replace` crate to handle locked-binary replacement.

use crate::error::{Error, Result};
use std::path::Path;
use tracing::debug;

/// Check that the parent directory of the current binary is writable.
#[allow(unsafe_code)]
pub fn check_write_permission(exe_path: &Path) -> Result<()> {
    let parent = exe_path.parent().ok_or_else(|| Error::UpdateReplaceFailed {
        reason: format!(
            "cannot determine parent directory of '{}'",
            exe_path.display()
        ),
    })?;

    let metadata = std::fs::metadata(parent).map_err(|e| Error::UpdateReplaceFailed {
        reason: format!("cannot read metadata of '{}': {e}", parent.display()),
    })?;

    if metadata.permissions().readonly() {
        return Err(Error::UpdatePermissionDenied {
            path: parent.to_path_buf(),
        });
    }

    // On Unix, also check the actual write bits for the current user
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        use std::os::unix::fs::PermissionsExt;
        let mode = metadata.permissions().mode();
        // SAFETY: getuid() and getgid() are simple syscalls with no invariants to uphold.
        let uid = unsafe { libc::getuid() };
        let gid = unsafe { libc::getgid() };
        let file_uid = metadata.uid();
        let file_gid = metadata.gid();

        let writable = if uid == 0 {
            true // root can write anywhere
        } else if uid == file_uid {
            mode & 0o200 != 0 // owner write
        } else if gid == file_gid {
            mode & 0o020 != 0 // group write
        } else {
            mode & 0o002 != 0 // other write
        };

        if !writable {
            return Err(Error::UpdatePermissionDenied {
                path: parent.to_path_buf(),
            });
        }
    }

    Ok(())
}

/// Check if the binary appears to be a development build (inside a cargo target/ directory).
pub fn is_dev_build(exe_path: &Path) -> bool {
    let path_str = exe_path.to_string_lossy();
    path_str.contains("/target/") || path_str.contains("\\target\\")
}

/// Set executable permissions on a file (Unix only, no-op on Windows).
#[cfg(unix)]
pub fn set_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let perms = std::fs::Permissions::from_mode(0o755);
    std::fs::set_permissions(path, perms).map_err(|e| Error::UpdateReplaceFailed {
        reason: format!(
            "failed to set executable permissions on '{}': {e}",
            path.display()
        ),
    })
}

/// Set executable permissions on a file (Unix only, no-op on Windows).
#[cfg(not(unix))]
pub fn set_executable(_path: &Path) -> Result<()> {
    Ok(())
}

/// Replace the current binary with a new one.
///
/// On Unix: renames current to `.old`, moves new into place. If the second
/// rename fails, attempts to restore the original.
///
/// On Windows: uses `self_replace` crate.
///
/// Returns `true` if a `.old` backup was kept (Unix), `false` otherwise (Windows).
pub fn replace_binary(exe_path: &Path, new_binary_path: &Path) -> Result<bool> {
    #[cfg(unix)]
    {
        replace_unix(exe_path, new_binary_path)
    }
    #[cfg(windows)]
    {
        replace_windows(exe_path, new_binary_path)
    }
}

#[cfg(unix)]
fn replace_unix(exe_path: &Path, new_binary_path: &Path) -> Result<bool> {
    let backup_path = exe_path.with_extension("old");

    debug!(
        "renaming {} -> {}",
        exe_path.display(),
        backup_path.display()
    );
    std::fs::rename(exe_path, &backup_path).map_err(|e| Error::UpdateReplaceFailed {
        reason: format!(
            "failed to rename '{}' to '{}': {e}",
            exe_path.display(),
            backup_path.display()
        ),
    })?;

    debug!(
        "renaming {} -> {}",
        new_binary_path.display(),
        exe_path.display()
    );
    if let Err(e) = std::fs::rename(new_binary_path, exe_path) {
        // Rollback: restore the original
        debug!("second rename failed, rolling back");
        if let Err(rollback_err) = std::fs::rename(&backup_path, exe_path) {
            return Err(Error::UpdateReplaceFailed {
                reason: format!(
                    "failed to install new binary AND failed to restore backup: install error: {e}, rollback error: {rollback_err}"
                ),
            });
        }
        return Err(Error::UpdateReplaceFailed {
            reason: format!(
                "failed to rename '{}' to '{}': {e}",
                new_binary_path.display(),
                exe_path.display()
            ),
        });
    }

    Ok(true) // backup kept
}

#[cfg(windows)]
fn replace_windows(_exe_path: &Path, new_binary_path: &Path) -> Result<bool> {
    self_replace::self_replace(new_binary_path).map_err(|e| Error::UpdateReplaceFailed {
        reason: format!("self_replace failed: {e}"),
    })?;

    // Clean up the temp file
    let _ = std::fs::remove_file(new_binary_path);

    Ok(false) // no backup on Windows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_dev_build_detects_target_dir() {
        assert!(is_dev_build(Path::new(
            "/home/user/project/target/release/birda"
        )));
        assert!(is_dev_build(Path::new(
            "/home/user/project/target/debug/birda"
        )));
    }

    #[test]
    fn test_is_dev_build_false_for_install_paths() {
        assert!(!is_dev_build(Path::new("/usr/local/bin/birda")));
        assert!(!is_dev_build(Path::new("/home/user/.local/bin/birda")));
    }

    #[cfg(unix)]
    #[test]
    fn test_set_executable_on_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test-bin");
        std::fs::write(&path, b"fake binary").unwrap();
        set_executable(&path).unwrap();

        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o755);
    }
}
