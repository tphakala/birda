//! Compiled-model cache housekeeping.
//!
//! `OpenVINO` writes the blobs itself under the directory birda hands it. birda
//! only chooses that directory and keeps its total size under a cap.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// One file in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    /// Path of the file.
    pub path: PathBuf,
    /// Size in bytes.
    pub size: u64,
    /// When the file was last used: the later of its access and modification
    /// times. Loading a cached blob reads it without writing it, so the
    /// modification time alone would rank a blob every run loads as old.
    pub last_used: SystemTime,
}

/// Name of the per-runtime subdirectory, derived from the build number.
///
/// Every character that is not alphanumeric, `.`, `-` or `_` becomes `_`. A
/// name made only of dots, which would be a relative path component, becomes
/// `unknown`, like an empty one.
#[must_use]
pub fn runtime_subdir(build_number: &str) -> String {
    let cleaned: String = build_number
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.chars().all(|c| c == '.') {
        "unknown".to_string()
    } else {
        cleaned
    }
}

/// Delete files, least recently used first, until the total size is at most
/// `cap`.
///
/// `remove` deletes one file and reports whether it is gone. A file it could
/// not delete still counts towards the total, so the next older one is tried.
/// Returns the files removed, in order.
fn prune_entries(
    entries: &[CacheEntry],
    cap: u64,
    remove: &mut dyn FnMut(&Path) -> bool,
) -> Vec<PathBuf> {
    let mut total: u64 = entries.iter().map(|entry| entry.size).sum();
    let mut by_age: Vec<&CacheEntry> = entries.iter().collect();
    by_age.sort_by(|a, b| {
        a.last_used
            .cmp(&b.last_used)
            .then_with(|| a.path.cmp(&b.path))
    });
    let mut removed = Vec::new();
    for entry in by_age {
        if total <= cap {
            break;
        }
        if remove(&entry.path) {
            total -= entry.size;
            removed.push(entry.path.clone());
        }
    }
    removed
}

/// Every regular file under `root`.
///
/// Symlinks are skipped, not followed: `DirEntry::metadata` describes the link
/// itself, which is neither a directory nor a regular file.
fn collect(root: &Path) -> Vec<CacheEntry> {
    let mut entries = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(read) = std::fs::read_dir(&dir) else {
            continue;
        };
        for item in read.flatten() {
            let Ok(metadata) = item.metadata() else {
                continue;
            };
            if metadata.is_dir() {
                pending.push(item.path());
            } else if metadata.is_file() {
                entries.push(CacheEntry {
                    path: item.path(),
                    size: metadata.len(),
                    last_used: metadata
                        .accessed()
                        .ok()
                        .max(metadata.modified().ok())
                        .unwrap_or(SystemTime::UNIX_EPOCH),
                });
            }
        }
    }
    entries
}

/// Delete the least recently used files under `root` until the cache is
/// within `cap`.
///
/// Returns how many files were removed. Failures to delete are skipped: the
/// cache is an optimisation and a file that stays only costs disk space.
pub fn prune(root: &Path, cap: u64) -> usize {
    prune_entries(&collect(root), cap, &mut |path| {
        std::fs::remove_file(path).is_ok()
    })
    .len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn entry(name: &str, size: u64, age_secs: u64) -> CacheEntry {
        CacheEntry {
            path: PathBuf::from(name),
            size,
            last_used: SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000 - age_secs),
        }
    }

    /// Prune with a remover that deletes every file.
    fn select_prune(entries: &[CacheEntry], cap: u64) -> Vec<PathBuf> {
        prune_entries(entries, cap, &mut |_| true)
    }

    #[test]
    fn nothing_is_deleted_within_the_cap() {
        let entries = [entry("a", 40, 30), entry("b", 60, 10)];
        assert_eq!(select_prune(&entries, 100), Vec::<PathBuf>::new());
        assert_eq!(select_prune(&[], 0), Vec::<PathBuf>::new());
    }

    #[test]
    fn a_file_that_cannot_be_deleted_does_not_stop_the_pruning() {
        // 30 + 30 + 60 = 120 against a cap of 60, and the oldest is stuck.
        let entries = [
            entry("stuck", 30, 30),
            entry("older", 30, 20),
            entry("newest", 60, 10),
        ];
        let removed = prune_entries(&entries, 60, &mut |path| path != Path::new("stuck"));
        // Removing older leaves 90, still over the cap, so newest goes too.
        assert_eq!(
            removed,
            vec![PathBuf::from("older"), PathBuf::from("newest")]
        );
    }

    #[test]
    fn the_oldest_files_go_first_until_the_total_fits() {
        // 30 + 40 + 50 + 60 = 180 against a cap of 100.
        let entries = [
            entry("newest", 60, 1),
            entry("oldest", 30, 40),
            entry("older", 40, 30),
            entry("newer", 50, 10),
        ];
        // Removing oldest (30) leaves 150, older (40) leaves 110, newer (50)
        // leaves 60: three files, and the newest survives.
        assert_eq!(
            select_prune(&entries, 100),
            vec![
                PathBuf::from("oldest"),
                PathBuf::from("older"),
                PathBuf::from("newer")
            ]
        );
    }

    #[test]
    fn equal_ages_are_broken_by_path() {
        let entries = [entry("b", 80, 5), entry("a", 80, 5)];
        assert_eq!(select_prune(&entries, 100), vec![PathBuf::from("a")]);
    }

    #[test]
    fn prune_removes_files_from_disk_and_keeps_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("2025.4.1");
        std::fs::create_dir_all(&nested).unwrap();
        let old = nested.join("old.blob");
        let new = dir.path().join("new.blob");
        std::fs::write(&old, vec![0_u8; 600]).unwrap();
        std::fs::write(&new, vec![0_u8; 600]).unwrap();
        let past = SystemTime::now() - Duration::from_secs(3600);
        set_times(&old, past, past);

        assert_eq!(prune(dir.path(), 1000), 1);
        assert!(!old.exists());
        assert!(new.exists());
        assert_eq!(prune(dir.path(), 1000), 0);
    }

    fn set_times(path: &Path, accessed: SystemTime, modified: SystemTime) {
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_times(
                std::fs::FileTimes::new()
                    .set_accessed(accessed)
                    .set_modified(modified),
            )
            .unwrap();
    }

    #[test]
    fn a_blob_read_recently_outlives_a_newer_one_left_unread() {
        let dir = tempfile::tempdir().unwrap();
        let read = dir.path().join("read.blob");
        let unread = dir.path().join("unread.blob");
        std::fs::write(&read, vec![0_u8; 600]).unwrap();
        std::fs::write(&unread, vec![0_u8; 600]).unwrap();
        let now = SystemTime::now();
        let hours_ago = |hours: u64| now - Duration::from_secs(hours * 3600);
        // Written three hours ago but read just now, against one written two
        // hours ago and not read since.
        set_times(&read, now, hours_ago(3));
        set_times(&unread, hours_ago(2), hours_ago(2));

        assert_eq!(prune(dir.path(), 1000), 1);
        assert!(read.exists());
        assert!(!unread.exists());
    }

    #[cfg(unix)]
    #[test]
    fn prune_never_follows_a_symlink_out_of_the_cache() {
        let cache = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let foreign = outside.path().join("foreign.bin");
        std::fs::write(&foreign, vec![0_u8; 600]).unwrap();
        std::os::unix::fs::symlink(outside.path(), cache.path().join("linked_dir")).unwrap();
        std::os::unix::fs::symlink(&foreign, cache.path().join("linked_file")).unwrap();
        let own = cache.path().join("own.blob");
        std::fs::write(&own, vec![0_u8; 600]).unwrap();

        // A cap of zero deletes every file the walk collects.
        assert_eq!(prune(cache.path(), 0), 1);
        assert!(!own.exists());
        assert!(foreign.exists());
    }

    #[test]
    fn the_runtime_subdirectory_is_a_safe_single_path_component() {
        assert_eq!(
            runtime_subdir("2025.4.1-19314-5a61fbc7b2d"),
            "2025.4.1-19314-5a61fbc7b2d"
        );
        assert_eq!(runtime_subdir("2026.2/../x y"), "2026.2_.._x_y");
        assert_eq!(runtime_subdir(""), "unknown");
        assert_eq!(runtime_subdir(".."), "unknown");
    }
}
