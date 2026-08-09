//! File locking for distributed processing.

use super::registry::LockRegistry;
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
#[derive(Debug)]
pub struct FileLock {
    lock_path: PathBuf,
}

impl FileLock {
    /// Attempt to acquire a lock for processing a file.
    ///
    /// The lock file is created in the output directory.
    pub fn acquire(input_path: &Path, output_dir: &Path) -> Result<Self> {
        let lock_path = Self::lock_path_for(input_path, output_dir);

        // Ensure output directory exists before creating lock file
        fs::create_dir_all(output_dir).map_err(|e| Error::OutputDirCreateFailed {
            path: output_dir.to_path_buf(),
            source: e,
        })?;

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

                // Registered only AFTER we own the file, so a path we do not own is
                // never in the registry and a Ctrl+C can never make
                // `cleanup_all_locks` delete a peer's lock. The cost is a Ctrl+C in
                // the create-then-register gap leaking our own lock, which is a
                // manual delete, not lost data. Registering before the create (the
                // old behaviour) meant an `AlreadyExists` create for a peer-owned
                // path left that path registered until the error arm unregistered
                // it, and a Ctrl+C in that window deleted the peer's live lock.
                ACTIVE_FILE_LOCKS.register(&lock_path);

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
        // Use to_string_lossy() to handle non-UTF-8 filenames gracefully
        let stem = input_path.file_name().map_or_else(
            || std::borrow::Cow::Borrowed("unknown"),
            |n| n.to_string_lossy(),
        );
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
        // release() unregisters before it unlinks; see LockRegistry::release for
        // why that ordering is what keeps a Ctrl+C from deleting a peer's lock.
        ACTIVE_FILE_LOCKS.release(&self.lock_path);
    }
}

/// Active per-input-file lock paths, removed if the process is interrupted. A
/// separate [`LockRegistry`] instance from the config lock's so neither type's
/// cleanup can remove the other's lock file.
static ACTIVE_FILE_LOCKS: LockRegistry = LockRegistry::new();

/// Clean up all registered file locks. Called from the Ctrl+C handler.
pub fn cleanup_all_locks() {
    ACTIVE_FILE_LOCKS.cleanup();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// Serialize locking tests to avoid race conditions with `cleanup_all_locks()`
    /// which drains the entire global registry.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_acquire_and_release_lock() {
        let _guard = TEST_LOCK.lock().unwrap();
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
        let _guard = TEST_LOCK.lock().unwrap();
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
        let _guard = TEST_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let lock_path = temp_dir.path().join("cleanup_test.wav.birda.lock");

        // Create a lock file manually (simulating orphaned lock)
        File::create(&lock_path).unwrap();
        assert!(lock_path.exists());

        // Register this path and call cleanup
        ACTIVE_FILE_LOCKS.register(&lock_path);
        cleanup_all_locks();

        // Our lock file should be removed
        assert!(
            !lock_path.exists(),
            "Lock file should be removed by cleanup_all_locks()"
        );
    }

    #[test]
    fn test_register_and_unregister_lock() {
        let _guard = TEST_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let lock_path = temp_dir.path().join("reg_test.wav.birda.lock");

        // Create a file
        File::create(&lock_path).unwrap();

        // Register and unregister - file should still exist
        ACTIVE_FILE_LOCKS.register(&lock_path);
        ACTIVE_FILE_LOCKS.unregister(&lock_path);

        assert!(lock_path.exists(), "unregister should not delete files");
    }

    #[test]
    fn test_acquire_registers_only_the_lock_it_owns() {
        // This guards the ownership INVARIANT: a held lock is registered, a lost
        // acquire registers nothing, and Drop unregisters. It does NOT prove the
        // register-after-create or unregister-before-remove ORDERING; those differ
        // from the old code only when a Ctrl+C lands inside a sub-instruction
        // window, which a synchronous test cannot reach. Of the three fixes, only
        // the poison recovery is deterministically red-green (see registry.rs).
        let _guard = TEST_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let input = temp_dir.path().join("test.wav");
        let lock_path = FileLock::lock_path_for(&input, temp_dir.path());

        // A peer's live lock: created out-of-band, so this process must never
        // register it, even though its own acquire loses the race to create it.
        File::create(&lock_path).unwrap();

        let contended = FileLock::acquire(&input, temp_dir.path());
        assert!(
            matches!(contended, Err(Error::FileLocked { .. })),
            "acquiring an already-locked input must fail: {contended:?}"
        );
        assert!(
            !ACTIVE_FILE_LOCKS.contains(&lock_path),
            "a failed acquire must not register a path this process does not own, \
             so a signal cleanup can never delete a peer's lock"
        );

        // With the peer's lock gone, winning the create registers the path we own.
        std::fs::remove_file(&lock_path).unwrap();
        let held = FileLock::acquire(&input, temp_dir.path()).unwrap();
        assert!(
            ACTIVE_FILE_LOCKS.contains(&lock_path),
            "a successful acquire must register the lock it owns"
        );

        drop(held);
        assert!(
            !ACTIVE_FILE_LOCKS.contains(&lock_path),
            "drop must unregister the lock it owned"
        );
    }

    #[test]
    fn test_is_stale_is_false_for_a_fresh_lock() {
        let _guard = TEST_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let input = temp_dir.path().join("test.wav");
        File::create(&input).unwrap();

        let _lock = FileLock::acquire(&input, temp_dir.path()).unwrap();

        assert!(
            !FileLock::is_stale(&input, temp_dir.path(), Duration::from_hours(1)),
            "a lock taken just now is not older than an hour"
        );
    }

    #[test]
    fn test_is_stale_is_false_when_no_lock_exists() {
        let temp_dir = TempDir::new().unwrap();
        let input = temp_dir.path().join("nope.wav");

        assert!(
            !FileLock::is_stale(&input, temp_dir.path(), Duration::from_secs(0)),
            "an absent lock is never stale"
        );
    }

    #[test]
    fn test_is_stale_detects_an_aged_lock_and_remove_stale_reclaims_it() {
        use std::time::SystemTime;

        let _guard = TEST_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let input = temp_dir.path().join("test.wav");

        // Age the lock deterministically rather than sleeping, so the timing is
        // not at the mercy of the test runner or the filesystem clock.
        let lock_path = FileLock::lock_path_for(&input, temp_dir.path());
        let lock_file = File::create(&lock_path).unwrap();
        lock_file
            .set_modified(SystemTime::now() - Duration::from_hours(1))
            .unwrap();

        assert!(
            FileLock::is_stale(&input, temp_dir.path(), Duration::from_mins(1)),
            "an hour-old lock is stale against a one-minute timeout"
        );
        assert!(
            !FileLock::is_stale(&input, temp_dir.path(), Duration::from_hours(2)),
            "the same lock is fresh against a two-hour timeout"
        );

        FileLock::remove_stale(&input, temp_dir.path()).unwrap();
        assert!(
            !FileLock::is_locked(&input, temp_dir.path()),
            "remove_stale must clear the lock so the file can be processed"
        );
    }

    #[test]
    fn test_remove_stale_errors_when_there_is_no_lock() {
        let temp_dir = TempDir::new().unwrap();
        let input = temp_dir.path().join("missing.wav");

        assert!(
            FileLock::remove_stale(&input, temp_dir.path()).is_err(),
            "removing a lock that is not there is an error, not a silent success"
        );
    }
}
