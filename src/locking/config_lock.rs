//! Serialising lock for the config file's read-modify-write (#313).
//!
//! #307 made each config write atomic, so a reader sees the whole old file or
//! the whole new one. It did not serialise a PAIR of writes: two `birda`
//! processes both load `{X}`, one saves `{X,Y}`, the other saves `{X,Z}`, and
//! `Y` is gone with no error at any point. This lock spans the whole
//! load-mutate-save, so the second writer waits for the first instead of loading
//! the same base and clobbering the edit.
//!
//! It is an `O_CREAT|O_EXCL` lock file beside the config (the issue's first
//! suggestion), mirroring [`super::FileLock`]'s create/registry/Drop shape. Two
//! deliberate choices:
//!
//! - It retries briefly instead of failing on the first miss, because config
//!   writes are sub-second and a fail-fast lock would turn ordinary back-to-back
//!   `config set`/`models` commands into spurious errors.
//! - It does NOT auto-break a lock left behind. Breaking an existence lock by age
//!   or liveness is a time-of-check/time-of-use race: two processes can both
//!   judge the same lock breakable, and the second's blind remove deletes the
//!   first's fresh lock, putting both into the critical section, which is the
//!   exact data loss this lock exists to prevent. Instead a lock is released on
//!   normal exit (Drop), on Ctrl+C ([`cleanup_all_config_locks`]), and on panic
//!   (unwind runs Drop); only a hard kill or power loss leaves one behind, and
//!   [`Error::ConfigLocked`] then tells the user to delete it. A future move to
//!   an advisory lock (`flock`/`LockFileEx`, released by the kernel on process
//!   death) would restore automatic recovery without the race.
//!
//! The registry is deliberately separate from `file_lock`'s: one shared `Vec`
//! would let either type's cleanup remove the other's lock file.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::constants::config_lock::{ACQUIRE_TIMEOUT, LOCK_SUFFIX, RETRY_INTERVAL};
use crate::error::{Error, Result};

/// Active config-lock paths, removed if the process is interrupted.
static ACTIVE_CONFIG_LOCKS: LazyLock<Mutex<Vec<PathBuf>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// Register a held lock path for signal cleanup.
fn register(path: &Path) {
    if let Ok(mut locks) = ACTIVE_CONFIG_LOCKS.lock() {
        locks.push(path.to_path_buf());
    }
}

/// Unregister a lock path once it is released.
fn unregister(path: &Path) {
    if let Ok(mut locks) = ACTIVE_CONFIG_LOCKS.lock() {
        locks.retain(|p| p != path);
    }
}

/// Remove every held config lock. Called from the Ctrl+C handler.
///
/// Recovers from a poisoned mutex so cleanup still runs after a panic, and
/// drains the registry so each path is removed once. Only paths this process has
/// successfully acquired are ever registered, so this never removes a peer's
/// lock.
pub fn cleanup_all_config_locks() {
    let paths = {
        let mut locks = ACTIVE_CONFIG_LOCKS
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        std::mem::take(&mut *locks)
    };
    for path in paths {
        let _ = std::fs::remove_file(&path);
    }
}

/// RAII guard serialising a config load-mutate-save against other processes.
#[derive(Debug)]
struct ConfigLock {
    lock_path: PathBuf,
}

impl ConfigLock {
    /// Acquire the exclusive config lock beside `config_path`.
    ///
    /// Retries on contention up to [`ACQUIRE_TIMEOUT`], then returns
    /// [`Error::ConfigLocked`]. Does not break a lock left by a crashed process;
    /// see the module docs for why, and for how a leftover lock is recovered.
    fn acquire(config_path: &Path) -> Result<Self> {
        Self::acquire_with(config_path, ACQUIRE_TIMEOUT)
    }

    /// [`Self::acquire`] with an explicit timeout, so tests need not wait seconds.
    fn acquire_with(config_path: &Path, timeout: Duration) -> Result<Self> {
        let lock_path = lock_path_for(config_path);

        // The lock create and the later save both need the directory to exist.
        if let Some(parent) = lock_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::LockCreate {
                path: lock_path.clone(),
                source: e,
            })?;
        }

        let deadline = Instant::now() + timeout;

        loop {
            match try_create(&lock_path) {
                Ok(()) => {
                    // Registered only AFTER we own the file, so a path we do not
                    // own is never in the registry and a Ctrl+C can never make
                    // `cleanup_all_config_locks` delete a peer's lock. The cost is
                    // a Ctrl+C in the create-then-register gap leaking our own
                    // lock, which is a manual delete, not lost data.
                    register(&lock_path);
                    return Ok(Self { lock_path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if Instant::now() >= deadline {
                        return Err(Error::ConfigLocked { path: lock_path });
                    }
                    std::thread::sleep(RETRY_INTERVAL);
                }
                Err(e) => {
                    return Err(Error::LockCreate {
                        path: lock_path,
                        source: e,
                    });
                }
            }
        }
    }
}

impl Drop for ConfigLock {
    fn drop(&mut self) {
        // Unregister BEFORE removing the file, not after. A Ctrl+C landing
        // between the two then leaks only our own lock file (a manual delete),
        // never a peer's: once the path is out of the registry our signal cleanup
        // ignores it, so a peer that re-creates a lock at this path right after
        // our remove cannot be caught by our cleanup. The other order leaves a
        // window where the path is still registered but the file at it is a
        // peer's fresh lock, which cleanup would then delete.
        unregister(&self.lock_path);
        // Safe to remove by path: nothing breaks a held lock, so while we hold it
        // the file at `lock_path` is always the one we created.
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

/// Run `f` while holding the config lock, releasing it when `f` returns.
///
/// The lock is dropped (and the lock file removed) whether `f` returns `Ok` or
/// `Err`. Wrap the whole load-mutate-save in `f`, not just the save, so a
/// concurrent writer cannot load the same base and clobber the edit.
///
/// Not re-entrant: `f` must not call `with_config_lock` or `update_config` again
/// for the same config, or it will block on its own lock until the timeout.
pub fn with_config_lock<T>(config_path: &Path, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let _lock = ConfigLock::acquire(config_path)?;
    f()
}

/// The lock file path beside `config_path`
/// (`config.toml` -> `config.toml.birda.lock`).
fn lock_path_for(config_path: &Path) -> PathBuf {
    let name = config_path.file_name().map_or_else(
        || std::borrow::Cow::Borrowed("config"),
        |n| n.to_string_lossy(),
    );
    config_path.with_file_name(format!("{name}{LOCK_SUFFIX}"))
}

/// Create the lock file exclusively, writing best-effort debug info.
fn try_create(lock_path: &Path) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(lock_path)?;
    // For a human debugging a stuck lock only; a write failure does not matter.
    let host = hostname::get().map_or_else(
        |_| "unknown".to_string(),
        |h| h.to_string_lossy().into_owned(),
    );
    let _ = writeln!(file, "pid={}\nhost={host}", std::process::id());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn cfg_path(dir: &TempDir) -> PathBuf {
        dir.path().join("config.toml")
    }

    #[test]
    fn test_acquire_creates_lock_and_drop_removes_it() {
        let dir = TempDir::new().unwrap();
        let path = cfg_path(&dir);
        let lock_path = lock_path_for(&path);
        {
            let _lock = ConfigLock::acquire(&path).unwrap();
            assert!(lock_path.exists(), "the lock file should exist while held");
        }
        assert!(
            !lock_path.exists(),
            "the lock file should be removed on drop"
        );
    }

    #[test]
    fn test_lock_path_is_a_sibling_of_the_config() {
        let path = Path::new("/etc/birda/config.toml");
        assert_eq!(
            lock_path_for(path),
            PathBuf::from("/etc/birda/config.toml.birda.lock")
        );
    }

    #[test]
    fn test_second_acquire_is_refused_while_held() {
        let dir = TempDir::new().unwrap();
        let path = cfg_path(&dir);
        let _held = ConfigLock::acquire(&path).unwrap();
        // Short timeout so the test does not wait the production five seconds.
        let contended = ConfigLock::acquire_with(&path, Duration::from_millis(80));
        assert!(
            matches!(contended, Err(Error::ConfigLocked { .. })),
            "a held lock must refuse a second acquirer: {contended:?}"
        );
    }

    #[test]
    fn test_acquire_succeeds_after_release() {
        let dir = TempDir::new().unwrap();
        let path = cfg_path(&dir);
        drop(ConfigLock::acquire(&path).unwrap());
        assert!(
            ConfigLock::acquire(&path).is_ok(),
            "a released lock must be re-acquirable"
        );
    }

    #[test]
    fn test_a_leftover_lock_blocks_until_removed() {
        // A lock left by a crashed process is NOT auto-broken (that would be a
        // TOCTOU race); it blocks until it is deleted.
        let dir = TempDir::new().unwrap();
        let path = cfg_path(&dir);
        let lock_path = lock_path_for(&path);
        std::fs::write(&lock_path, b"pid=999999\nhost=ghost").unwrap();

        let blocked = ConfigLock::acquire_with(&path, Duration::from_millis(80));
        assert!(
            matches!(blocked, Err(Error::ConfigLocked { .. })),
            "a leftover lock must block, not be stolen: {blocked:?}"
        );

        std::fs::remove_file(&lock_path).unwrap();
        assert!(
            ConfigLock::acquire(&path).is_ok(),
            "once the leftover lock is deleted, acquisition succeeds"
        );
    }

    #[test]
    fn test_with_config_lock_returns_value_and_releases() {
        let dir = TempDir::new().unwrap();
        let path = cfg_path(&dir);
        let lock_path = lock_path_for(&path);
        let out: i32 = with_config_lock(&path, || Ok(42)).unwrap();
        assert_eq!(out, 42);
        assert!(
            !lock_path.exists(),
            "with_config_lock must release the lock on return"
        );
    }

    #[test]
    fn test_with_config_lock_releases_on_error() {
        let dir = TempDir::new().unwrap();
        let path = cfg_path(&dir);
        let lock_path = lock_path_for(&path);
        let result: Result<()> = with_config_lock(&path, || {
            Err(Error::ConfigLocked {
                path: cfg_path(&dir),
            })
        });
        assert!(result.is_err());
        assert!(
            !lock_path.exists(),
            "the lock must be released even when the closure returns Err"
        );
        // Provably released: a fresh acquire wins immediately.
        assert!(ConfigLock::acquire(&path).is_ok());
    }
}
