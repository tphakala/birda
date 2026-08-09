//! A registry of held lock-file paths, drained if the process is interrupted.
//!
//! Both lock types ([`super::FileLock`] and the config lock) keep a registry of
//! the lock files they currently hold so the Ctrl+C handler can remove them on
//! the way out. The two registries are deliberately SEPARATE instances: one
//! shared `Vec` would let either type's cleanup remove the other type's lock
//! file. This primitive factors out the register/unregister/drain logic and the
//! three race and poison hardenings it carries (first applied to the config lock
//! in #313, extended to the file lock in #363) into one place that each lock type
//! instantiates as its own module-local `static`.
//!
//! The correctness rules the callers must follow are encoded in the method docs:
//! register only after winning the exclusive create, and unregister before
//! removing the file.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

/// The set of lock-file paths this process currently holds for one lock type.
///
/// Every method recovers from a poisoned mutex (`unwrap_or_else(into_inner)`)
/// rather than skipping the update. An asymmetry where `register`/`unregister`
/// silently no-op on poison while `cleanup` still drains could leave a stale
/// entry for a path this process no longer owns, which `cleanup` would then
/// remove, deleting a peer's fresh lock at that path.
pub(super) struct LockRegistry {
    active: Mutex<Vec<PathBuf>>,
}

impl LockRegistry {
    /// Create an empty registry. `const` so it can back a plain `static`.
    pub(super) const fn new() -> Self {
        Self {
            active: Mutex::new(Vec::new()),
        }
    }

    /// Record a held lock path so a signal can clean it up.
    ///
    /// Call only AFTER winning the exclusive create. A path this process does not
    /// own must never enter the registry, or a Ctrl+C in the window before the
    /// owner is established could make [`Self::cleanup`] delete a peer's lock. The
    /// cost of registering late is that a Ctrl+C in the create-then-register gap
    /// leaks this process's own lock file, which is a manual delete, not lost data.
    pub(super) fn register(&self, path: &Path) {
        let mut active = self.active.lock().unwrap_or_else(PoisonError::into_inner);
        active.push(path.to_path_buf());
    }

    /// Drop a lock path from the registry once it is released.
    ///
    /// Call BEFORE removing the file, not after. Once the path is out of the
    /// registry, a peer that re-creates a lock there is safe from this process's
    /// cleanup. The other order leaves a window where the path is still registered
    /// but the file at it is a peer's fresh lock, which cleanup would then delete.
    pub(super) fn unregister(&self, path: &Path) {
        let mut active = self.active.lock().unwrap_or_else(PoisonError::into_inner);
        active.retain(|p| p != path);
    }

    /// Remove every held lock file. Called from the Ctrl+C handler.
    ///
    /// Recovers from a poisoned mutex so cleanup still runs after a panic, and
    /// drains the registry so each path is removed once. Only paths this process
    /// has successfully acquired are ever registered, so this never removes a
    /// peer's lock.
    pub(super) fn cleanup(&self) {
        let paths = {
            let mut active = self.active.lock().unwrap_or_else(PoisonError::into_inner);
            std::mem::take(&mut *active)
        };
        for path in paths {
            let _ = std::fs::remove_file(&path);
        }
    }

    /// Release one held lock: drop it from the registry, then remove its file.
    ///
    /// Unregister BEFORE removing, so a Ctrl+C between the two leaks only this
    /// process's own lock file (a manual delete), never a peer's. Once the path is
    /// out of the registry, a peer re-creating a lock there is not caught by this
    /// process's cleanup; the reverse order leaves a window where the path is still
    /// registered but the file at it is a peer's fresh lock, which cleanup would
    /// delete. Removing by path is safe because nothing breaks a held lock, so
    /// while it is held the file at `path` is always the one we created.
    pub(super) fn release(&self, path: &Path) {
        self.unregister(path);
        let _ = std::fs::remove_file(path);
    }

    /// Whether `path` is currently registered. Test-only helper.
    #[cfg(test)]
    pub(super) fn contains(&self, path: &Path) -> bool {
        let active = self.active.lock().unwrap_or_else(PoisonError::into_inner);
        active.iter().any(|p| p == path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_register_then_unregister_leaves_no_entry() {
        let registry = LockRegistry::new();
        let path = Path::new("/tmp/does-not-matter.lock");

        registry.register(path);
        assert!(registry.contains(path), "register must record the path");

        registry.unregister(path);
        assert!(
            !registry.contains(path),
            "unregister must drop the path from the registry"
        );
    }

    #[test]
    fn test_unregister_does_not_remove_the_file() {
        let registry = LockRegistry::new();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("held.lock");
        std::fs::write(&path, b"held").unwrap();

        registry.register(&path);
        registry.unregister(&path);

        assert!(
            path.exists(),
            "unregister only touches the registry, never the filesystem"
        );
    }

    #[test]
    fn test_cleanup_removes_registered_files_and_drains() {
        let registry = LockRegistry::new();
        let dir = TempDir::new().unwrap();
        let one = dir.path().join("one.lock");
        let two = dir.path().join("two.lock");
        std::fs::write(&one, b"1").unwrap();
        std::fs::write(&two, b"2").unwrap();

        registry.register(&one);
        registry.register(&two);
        registry.cleanup();

        assert!(
            !one.exists(),
            "cleanup must remove the first registered lock"
        );
        assert!(
            !two.exists(),
            "cleanup must remove the second registered lock"
        );
        assert!(
            !registry.contains(&one) && !registry.contains(&two),
            "cleanup must drain the registry so a second call is a no-op"
        );
    }

    #[test]
    fn test_cleanup_ignores_an_unregistered_path() {
        let registry = LockRegistry::new();
        let dir = TempDir::new().unwrap();
        let unregistered = dir.path().join("peer.lock");
        std::fs::write(&unregistered, b"peer").unwrap();

        // Nothing registered: cleanup must not touch a file this process never
        // recorded, which is how a peer's lock stays safe.
        registry.cleanup();

        assert!(
            unregistered.exists(),
            "cleanup must leave an unregistered (e.g. peer-owned) file alone"
        );
    }

    #[test]
    fn test_registry_methods_recover_from_a_poisoned_mutex() {
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let registry = Arc::new(LockRegistry::new());

        // Registered BEFORE the poison, so unregister and cleanup have something to
        // act on: on the old `if let Ok(..)` behaviour they would silently no-op
        // through the poison and leave this entry (and its file) behind.
        let pre = dir.path().join("registered-before-poison.lock");
        std::fs::write(&pre, b"pre").unwrap();
        registry.register(&pre);

        // Poison the inner mutex by panicking while it is held.
        let poisoner = Arc::clone(&registry);
        let handle = std::thread::spawn(move || {
            let _guard = poisoner.active.lock().unwrap();
            panic!("poison the lock registry");
        });
        assert!(
            handle.join().is_err(),
            "the helper thread must have panicked"
        );

        // register must still record a path through the poison.
        let after = dir.path().join("registered-after-poison.lock");
        registry.register(&after);
        assert!(
            registry.contains(&after),
            "register must record a path even after the mutex was poisoned"
        );

        // unregister must still drop the entry through the poison.
        registry.unregister(&after);
        assert!(
            !registry.contains(&after),
            "unregister must still work after the mutex was poisoned"
        );

        // cleanup must still drain the registry AND remove the file through poison.
        registry.cleanup();
        assert!(
            !registry.contains(&pre),
            "cleanup must drain the registry even after the mutex was poisoned"
        );
        assert!(
            !pre.exists(),
            "cleanup must remove the registered file even after the mutex was poisoned"
        );
    }
}
