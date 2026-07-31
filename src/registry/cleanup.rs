//! Deleting model files an install no longer owns.
//!
//! Published model filenames never change, by the publishing policy that keeps
//! pinned checksums valid, so upgrading from one upstream version to the next
//! writes new files beside the old ones rather than over them. Left alone,
//! every upgrade leaks disk permanently: roughly 150 MB per regional slice and
//! 557 MB for a global `BirdNET` v3.0 fp32.
//!
//! Cleanup is precise rather than heuristic. `config.toml` records the exact
//! paths each entry owns, so the files to delete are the previous paths of the
//! entry being replaced, minus anything the new install reuses and anything
//! another entry still points at. Nothing is matched by filename pattern, and
//! nothing is deleted that a config entry still references.

use crate::config::{Config, ModelConfig};
use std::path::PathBuf;

/// Every file a config entry owns.
///
/// Not just the model and its labels: a BSG entry also owns its calibration,
/// migration and distribution-map files. Both halves of the cleanup decision
/// need the full set. Missing them on the left leaks those assets forever, and
/// missing them on the right lets cleanup delete a file another entry is still
/// using as one.
fn owned_paths(model: &ModelConfig) -> Vec<&PathBuf> {
    let mut paths = vec![&model.path, &model.labels];
    paths.extend(model.bsg_calibration.iter());
    paths.extend(model.bsg_migration.iter());
    paths.extend(model.bsg_distribution_maps.iter());
    paths
}

/// Files owned by `key` before this install that nothing references now.
///
/// Returns an empty list for a first install, which has no predecessor.
#[must_use]
pub fn orphaned_files(config: &Config, key: &str, keeping: &[PathBuf]) -> Vec<PathBuf> {
    let Some(previous) = config.models.get(key) else {
        return Vec::new();
    };

    let still_referenced: Vec<&PathBuf> = config
        .models
        .iter()
        .filter(|(other_key, _)| other_key.as_str() != key)
        .flat_map(|(_, model)| owned_paths(model))
        .collect();

    let mut orphans: Vec<PathBuf> = Vec::new();
    for path in owned_paths(previous) {
        if keeping.contains(path) {
            continue;
        }
        if still_referenced.contains(&path) {
            continue;
        }
        // A model whose path and labels are the same file would otherwise be
        // listed twice.
        if orphans.contains(path) {
            continue;
        }
        orphans.push(path.clone());
    }
    orphans
}

/// Delete orphaned files, returning the ones that could not be deleted.
///
/// A file that is already gone is in the desired state, so `NotFound` is not
/// reported. Failures are returned rather than propagated: the install itself
/// succeeded, and refusing to finish because a stale file could not be removed
/// would be a worse outcome than a warning.
#[must_use]
pub fn remove_orphans(paths: &[PathBuf]) -> Vec<(PathBuf, std::io::Error)> {
    paths
        .iter()
        .filter_map(|path| match std::fs::remove_file(path) {
            Ok(()) => None,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => Some((path.clone(), e)),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelType};

    fn model(path: &str, labels: &str) -> ModelConfig {
        ModelConfig {
            path: PathBuf::from(path),
            labels: PathBuf::from(labels),
            model_type: ModelType::BirdnetV30,
            meta_model: None,
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
            registry_id: Some("birdnet-v30".to_string()),
            installed_version: Some("3.0-preview3.1".to_string()),
            installed_build: Some(1),
            region: None,
            variant: Some("fp32".to_string()),
        }
    }

    #[test]
    fn test_orphaned_files_returns_the_previous_files_of_the_reinstalled_key() {
        let mut config = Config::default();
        config.models.insert(
            "birdnet-v30".to_string(),
            model("/m/old.onnx", "/m/old.txt"),
        );

        let orphans = orphaned_files(
            &config,
            "birdnet-v30",
            &[PathBuf::from("/m/new.onnx"), PathBuf::from("/m/new.txt")],
        );

        assert_eq!(
            orphans,
            vec![PathBuf::from("/m/old.onnx"), PathBuf::from("/m/old.txt")]
        );
    }

    #[test]
    fn test_orphaned_files_spares_a_file_another_entry_still_references() {
        // A global and a regional install can share a labels file. Deleting it
        // because the global was upgraded would break the regional install that
        // never changed.
        let mut config = Config::default();
        config.models.insert(
            "birdnet-v30".to_string(),
            model("/m/old.onnx", "/m/shared.txt"),
        );
        config.models.insert(
            "birdnet-v30-nordic".to_string(),
            model("/m/nordic.onnx", "/m/shared.txt"),
        );

        let orphans = orphaned_files(&config, "birdnet-v30", &[PathBuf::from("/m/new.onnx")]);

        assert_eq!(orphans, vec![PathBuf::from("/m/old.onnx")]);
    }

    #[test]
    fn test_orphaned_files_spares_files_the_new_install_reuses() {
        // Reinstalling the same variant, or switching variant while the labels
        // file stays the same, must not delete what was just downloaded.
        let mut config = Config::default();
        config.models.insert(
            "birdnet-v30".to_string(),
            model("/m/same.onnx", "/m/same.txt"),
        );

        let orphans = orphaned_files(
            &config,
            "birdnet-v30",
            &[PathBuf::from("/m/same.onnx"), PathBuf::from("/m/same.txt")],
        );

        assert!(orphans.is_empty());
    }

    #[test]
    fn test_orphaned_files_keeps_the_shared_labels_when_only_the_variant_changes() {
        // fp32 to fp16 for one region: the model file is superseded, the labels
        // file is byte-identical and reused.
        let mut config = Config::default();
        config.models.insert(
            "birdnet-v30-nordic".to_string(),
            model("/m/nordic-fp32.onnx", "/m/nordic-labels.txt"),
        );

        let orphans = orphaned_files(
            &config,
            "birdnet-v30-nordic",
            &[
                PathBuf::from("/m/nordic-fp16.onnx"),
                PathBuf::from("/m/nordic-labels.txt"),
            ],
        );

        assert_eq!(orphans, vec![PathBuf::from("/m/nordic-fp32.onnx")]);
    }

    #[test]
    fn test_orphaned_files_reclaims_the_bsg_assets_an_entry_owned() {
        // A BSG entry owns three files beyond the model and labels. Ignoring
        // them leaves them on disk forever after an upgrade.
        let mut config = Config::default();
        let mut bsg = model("/m/old.onnx", "/m/old.txt");
        bsg.bsg_calibration = Some(PathBuf::from("/m/old-cal.csv"));
        bsg.bsg_migration = Some(PathBuf::from("/m/old-mig.csv"));
        bsg.bsg_distribution_maps = Some(PathBuf::from("/m/old-maps.bin"));
        config.models.insert("bsg".to_string(), bsg);

        let orphans = orphaned_files(&config, "bsg", &[PathBuf::from("/m/new.onnx")]);

        assert!(orphans.contains(&PathBuf::from("/m/old-cal.csv")));
        assert!(orphans.contains(&PathBuf::from("/m/old-mig.csv")));
        assert!(orphans.contains(&PathBuf::from("/m/old-maps.bin")));
    }

    #[test]
    fn test_orphaned_files_spares_a_bsg_asset_another_entry_still_owns() {
        // The mirror of the case above: cleanup must not delete a file another
        // entry references, whichever slot each of them holds it in.
        let mut config = Config::default();
        let mut replaced = model("/m/old.onnx", "/m/old.txt");
        replaced.bsg_calibration = Some(PathBuf::from("/m/shared-cal.csv"));
        config.models.insert("bsg".to_string(), replaced);

        let mut other = model("/m/other.onnx", "/m/other.txt");
        other.bsg_calibration = Some(PathBuf::from("/m/shared-cal.csv"));
        config.models.insert("bsg-other".to_string(), other);

        let orphans = orphaned_files(&config, "bsg", &[]);

        assert!(!orphans.contains(&PathBuf::from("/m/shared-cal.csv")));
        assert!(orphans.contains(&PathBuf::from("/m/old.onnx")));
    }

    #[test]
    fn test_orphaned_files_is_empty_for_a_first_install() {
        let config = Config::default();
        let orphans = orphaned_files(&config, "birdnet-v30", &[PathBuf::from("/m/new.onnx")]);
        assert!(orphans.is_empty());
    }

    #[test]
    fn test_orphaned_files_does_not_list_one_path_twice() {
        let mut config = Config::default();
        config
            .models
            .insert("odd".to_string(), model("/m/one.bin", "/m/one.bin"));

        let orphans = orphaned_files(&config, "odd", &[]);

        assert_eq!(orphans, vec![PathBuf::from("/m/one.bin")]);
    }

    #[test]
    fn test_remove_orphans_deletes_the_files_it_is_given() {
        let dir = tempfile::tempdir().unwrap();
        let present = dir.path().join("gone.onnx");
        std::fs::write(&present, b"x").unwrap();

        let failures = remove_orphans(std::slice::from_ref(&present));

        assert!(!present.exists());
        assert!(failures.is_empty());
    }

    #[test]
    fn test_remove_orphans_treats_a_missing_file_as_already_done() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("never-existed.onnx");

        let failures = remove_orphans(&[missing]);

        assert!(failures.is_empty());
    }

    #[test]
    fn test_remove_orphans_reports_a_failure_without_skipping_the_rest() {
        // A directory cannot be removed with remove_file, which stands in for
        // any undeletable path. The file after it must still be removed.
        let dir = tempfile::tempdir().unwrap();
        let undeletable = dir.path().join("a-directory");
        std::fs::create_dir(&undeletable).unwrap();
        let deletable = dir.path().join("b.onnx");
        std::fs::write(&deletable, b"x").unwrap();

        let failures = remove_orphans(&[undeletable.clone(), deletable.clone()]);

        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, undeletable);
        assert!(!deletable.exists(), "the removable file must still go");
    }
}
