//! File locking for distributed processing.

use crate::constants::LOCK_FILE_EXTENSION;
use crate::error::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Lock file content for debugging.
#[derive(Debug, Serialize, Deserialize)]
pub struct LockInfo {
    /// Process ID that holds the lock.
    pub pid: u32,
    /// Hostname of the machine.
    pub hostname: String,
    /// When the lock was acquired.
    pub started: DateTime<Utc>,
    /// Path to the input file being processed.
    pub input: PathBuf,
}

/// RAII guard for file locks.
pub struct FileLock {
    lock_path: PathBuf,
}

impl FileLock {
    /// Attempt to acquire a lock for processing a file.
    ///
    /// The lock file is created in the output directory.
    pub fn acquire(input_path: &Path, output_dir: &Path) -> Result<Self> {
        let lock_path = Self::lock_path_for(input_path, output_dir);

        // Try to create lock file exclusively
        let file = OpenOptions::new()
            .write(true)
            .create_new(true) // Fails if file exists
            .open(&lock_path);

        match file {
            Ok(mut f) => {
                // Write lock info
                let info = LockInfo {
                    pid: std::process::id(),
                    hostname: hostname::get().map_or_else(
                        |_| "unknown".to_string(),
                        |h| h.to_string_lossy().into_owned(),
                    ),
                    started: Utc::now(),
                    input: input_path.to_path_buf(),
                };

                let json = serde_json::to_string_pretty(&info).unwrap_or_else(|_| "{}".to_string());
                let _ = f.write_all(json.as_bytes());

                // Register for cleanup on signal
                register_lock(&lock_path);

                Ok(Self { lock_path })
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(Error::FileLocked { path: lock_path })
            }
            Err(e) => Err(Error::LockCreate {
                path: lock_path,
                source: e,
            }),
        }
    }

    /// Get the lock file path for an input file.
    pub fn lock_path_for(input_path: &Path, output_dir: &Path) -> PathBuf {
        let stem = input_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        output_dir.join(format!("{stem}{LOCK_FILE_EXTENSION}"))
    }

    /// Check if a lock file exists.
    pub fn is_locked(input_path: &Path, output_dir: &Path) -> bool {
        Self::lock_path_for(input_path, output_dir).exists()
    }

    /// Check if a lock is stale (older than `max_age`).
    pub fn is_stale(input_path: &Path, output_dir: &Path, max_age: Duration) -> bool {
        let lock_path = Self::lock_path_for(input_path, output_dir);

        if let Ok(metadata) = fs::metadata(&lock_path)
            && let Ok(modified) = metadata.modified()
        {
            return modified.elapsed().unwrap_or_default() > max_age;
        }
        false
    }

    /// Remove a stale lock.
    pub fn remove_stale(input_path: &Path, output_dir: &Path) -> Result<()> {
        let lock_path = Self::lock_path_for(input_path, output_dir);
        fs::remove_file(&lock_path).map_err(|e| Error::LockRemove {
            path: lock_path,
            source: e,
        })
    }

    /// Release the lock explicitly.
    pub fn release(self) -> Result<()> {
        // Drop will handle cleanup
        Ok(())
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock_path);
        unregister_lock(&self.lock_path);
    }
}

/// Global registry of active lock paths for cleanup on signal.
static ACTIVE_LOCKS: std::sync::LazyLock<std::sync::Mutex<Vec<PathBuf>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

/// Register a lock path for cleanup on signal.
pub fn register_lock(path: &Path) {
    if let Ok(mut locks) = ACTIVE_LOCKS.lock() {
        locks.push(path.to_path_buf());
    }
}

/// Unregister a lock path after normal cleanup.
pub fn unregister_lock(path: &Path) {
    if let Ok(mut locks) = ACTIVE_LOCKS.lock() {
        locks.retain(|p| p != path);
    }
}

/// Clean up all registered locks. Called on signal.
pub fn cleanup_all_locks() {
    if let Ok(locks) = ACTIVE_LOCKS.lock() {
        for lock_path in locks.iter() {
            let _ = fs::remove_file(lock_path);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::TempDir;

    #[test]
    fn test_acquire_and_release_lock() {
        let temp_dir = TempDir::new().unwrap();
        let input = temp_dir.path().join("test.wav");
        File::create(&input).unwrap();

        let lock = FileLock::acquire(&input, temp_dir.path());
        assert!(lock.is_ok());
        assert!(FileLock::is_locked(&input, temp_dir.path()));

        drop(lock);
        assert!(!FileLock::is_locked(&input, temp_dir.path()));
    }

    #[test]
    fn test_double_lock_fails() {
        let temp_dir = TempDir::new().unwrap();
        let input = temp_dir.path().join("test.wav");
        File::create(&input).unwrap();

        let lock1 = FileLock::acquire(&input, temp_dir.path());
        assert!(lock1.is_ok());

        let lock2 = FileLock::acquire(&input, temp_dir.path());
        assert!(lock2.is_err());
    }

    #[test]
    fn test_lock_path_format() {
        let path = FileLock::lock_path_for(Path::new("/data/audio.wav"), Path::new("/output"));
        assert_eq!(path.to_string_lossy(), "/output/audio.wav.birda.lock");
    }

    #[test]
    fn test_cleanup_all_locks_removes_registered_files() {
        let temp_dir = TempDir::new().unwrap();
        let lock_path = temp_dir.path().join("test.wav.birda.lock");

        // Create a lock file manually (simulating orphaned lock)
        File::create(&lock_path).unwrap();
        assert!(lock_path.exists());

        // Register and cleanup
        register_lock(&lock_path);
        cleanup_all_locks();

        // Lock file should be removed
        assert!(!lock_path.exists());
    }
}
