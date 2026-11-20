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
    /// Last modification time.
    pub modified: SystemTime,
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

/// Files to delete so the total size is at most `cap`, oldest first.
///
/// Deletes nothing when the total is already within the cap.
#[must_use]
pub fn select_prune(entries: &[CacheEntry], cap: u64) -> Vec<PathBuf> {
    let mut total: u64 = entries.iter().map(|entry| entry.size).sum();
    if total <= cap {
        return Vec::new();
    }
    let mut by_age: Vec<&CacheEntry> = entries.iter().collect();
    by_age.sort_by(|a, b| {
        a.modified
            .cmp(&b.modified)
            .then_with(|| a.path.cmp(&b.path))
    });
    let mut doomed = Vec::new();
    for entry in by_age {
        if total <= cap {
            break;
        }
        total -= entry.size;
        doomed.push(entry.path.clone());
    }
    doomed
}

/// Every regular file under `root`.
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
                    modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                });
            }
        }
    }
    entries
}

/// Delete the oldest files under `root` until the cache is within `cap`.
///
/// Returns how many files were removed. Failures to delete are skipped: the
/// cache is an optimisation and a file that stays only costs disk space.
pub fn prune(root: &Path, cap: u64) -> usize {
    select_prune(&collect(root), cap)
        .iter()
        .filter(|path| std::fs::remove_file(path).is_ok())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn entry(name: &str, size: u64, age_secs: u64) -> CacheEntry {
        CacheEntry {
            path: PathBuf::from(name),
            size,
            modified: SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000 - age_secs),
        }
    }

    #[test]
    fn nothing_is_deleted_within_the_cap() {
        let entries = [entry("a", 40, 30), entry("b", 60, 10)];
        assert_eq!(select_prune(&entries, 100), Vec::<PathBuf>::new());
        assert_eq!(select_prune(&[], 0), Vec::<PathBuf>::new());
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
        std::fs::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(past)
            .unwrap();

        assert_eq!(prune(dir.path(), 1000), 1);
        assert!(!old.exists());
        assert!(new.exists());
        assert_eq!(prune(dir.path(), 1000), 0);
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
