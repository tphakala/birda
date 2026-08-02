//! Registry loading and bootstrapping.

use super::types::{ModelEntry, Registry};
use crate::error::{Error, Result};
use std::path::PathBuf;

/// Load registry from user config or bundled default.
///
/// If a user registry exists but the bundled registry has a higher version,
/// the user registry is replaced with the bundled version.
pub fn load_registry() -> Result<Registry> {
    let registry_path = registry_file_path()?;
    let bundled_registry = load_bundled_registry()?;

    Ok(load_registry_from(&registry_path, bundled_registry))
}

/// Resolve the registry to use, given where the user's copy lives.
///
/// Split out from [`load_registry`] so it can be tested against a temporary
/// directory. [`load_registry`] resolves its path through
/// `crate::config::config_dir()`, so driving it directly in a test reads and
/// writes the developer's real config directory.
///
/// Infallible on purpose: every failure below is already a degradation to the
/// bundled registry, which is in hand before this is called, so there is nothing
/// left that could justify aborting the command that asked for it.
fn load_registry_from(registry_path: &std::path::Path, bundled_registry: Registry) -> Registry {
    if !registry_path.exists() {
        return bootstrap_registry(registry_path, &bundled_registry);
    }

    let user_registry = match load_from_file(registry_path) {
        Ok(registry) => registry,
        // Definitively corrupt: nothing can be recovered from the bytes on disk,
        // so replacing them with the bundled copy is the repair rather than a
        // loss. This is the behaviour that has always been here.
        Err(e @ Error::RegistryParse { .. }) => {
            tracing::warn!(
                "The model registry at {} could not be parsed: {}. Replacing it \
                 with the bundled registry.",
                registry_path.display(),
                e
            );
            persist_registry(registry_path, &bundled_registry);
            return bundled_registry;
        }
        // Could not be READ, which says nothing about the contents: EACCES, EIO,
        // EMFILE, a flaky network mount, fd exhaustion during a parallel batch.
        // The file is very possibly intact, so it is used from the bundled copy
        // in memory and left untouched on disk. Overwriting here was "could not
        // determine, so assume the destructive answer", and it silently discarded
        // hand-maintained registries that birda-gui also reads.
        //
        // The destructive branch is the one that has to name its variant, so an
        // error variant added to `load_from_file` later defaults to this arm.
        Err(e) => {
            tracing::warn!(
                "The model registry at {} could not be read: {}. Continuing with \
                 the bundled registry; the file on disk is unchanged.",
                registry_path.display(),
                e
            );
            return bundled_registry;
        }
    };

    // Compare versions - if bundled is newer, replace user's registry
    if bundled_registry.registry_version > user_registry.registry_version {
        tracing::info!(
            "Updating registry from version {} to {}",
            user_registry.registry_version,
            bundled_registry.registry_version
        );
        persist_registry(registry_path, &bundled_registry);
        bundled_registry
    } else {
        user_registry
    }
}

/// Write the registry to disk, warning rather than failing if it cannot be.
///
/// Persisting the registry is a cache side effect: `load_registry` already holds a
/// usable one in memory by the time this runs, so a failure to write it must not
/// abort the command that asked for it. Propagating it took down `species` and
/// every `models` subcommand, for a file the caller no longer needs. (`analyze`
/// already survived, because it matches on the error and continues without range
/// filtering, which is a different degradation and a worse one than not writing a
/// cache.)
///
/// Not a hypothetical. Replacing the file by rename needs write and execute on
/// the DIRECTORY, where the previous plain write needed only the write bit on the
/// file itself, so two ordinary layouts started failing here: a read-only config
/// directory holding a writable registry.json, and a registry.json bind-mounted
/// into a container as a FILE, where the rename fails with EBUSY. Both would
/// otherwise fail every invocation, and the version-bump branch above retries on
/// every single run because the file on disk never gets updated.
fn persist_registry(path: &std::path::Path, registry: &Registry) {
    if let Err(e) = write_registry_file(path, registry) {
        tracing::warn!(
            "Could not save the model registry to {}: {e}. Continuing with the bundled \
             registry; this will be retried on the next run.",
            path.display()
        );
    }
}

/// Get path to registry file in user config.
fn registry_file_path() -> Result<PathBuf> {
    Ok(crate::config::config_dir()?.join("registry.json"))
}

/// Load registry from existing file.
fn load_from_file(path: &std::path::Path) -> Result<Registry> {
    let content = std::fs::read_to_string(path).map_err(|e| Error::RegistryRead {
        path: path.to_path_buf(),
        source: e,
    })?;

    serde_json::from_str(&content).map_err(|e| Error::RegistryParse {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Load bundled registry from binary.
fn load_bundled_registry() -> Result<Registry> {
    const BUNDLED_REGISTRY: &str = include_str!("../../registry.json");

    serde_json::from_str(BUNDLED_REGISTRY).map_err(|e| Error::RegistryParse {
        path: PathBuf::from("bundled://registry.json"),
        source: e,
    })
}

/// Write registry to file, atomically.
///
/// Not a plain `fs::write`, which truncates the file and then writes it, leaving
/// registry.json empty in between. That is not a rare path: this function runs
/// whenever the file is absent, fails to parse, or is older than the bundled
/// version, and the last of those happens on the first launch after any upgrade
/// that bumps the bundled registry.
///
/// It partly self-heals, which caps the severity but does not remove it: a
/// zero-length file fails to parse on the next run and takes the re-bootstrap
/// branch, but any user-local edits are gone by then with no message, and the
/// recovery write could lose the same race again. It also has an external
/// consumer: birda-gui reads the file straight off disk and calls `JSON.parse`,
/// so a truncated registry surfaces there as a parse error rather than as
/// anything actionable, and model-language selection breaks.
///
/// Parent directories are created by the helper, so a missing config directory
/// now fails as a `RegistryWrite` naming the file rather than as a bare
/// `Error::Io`. Callers route through [`persist_registry`] rather than calling
/// this directly, because a failure to persist must not abort the command.
///
/// One behaviour change from the plain write, and it is not compensated for here:
/// a `registry.json` that is a **symlink whose target exists** is followed, but a
/// *dangling* one is replaced by a regular file rather than being written through.
/// Only `config.toml` resolves a dangling link, via `config::file::resolve_link`.
fn write_registry_file(path: &std::path::Path, registry: &Registry) -> Result<()> {
    let content = serde_json::to_string_pretty(registry)
        .map_err(|e| Error::RegistrySerialize { source: e })?;

    // `Umask` rather than `OwnerOnly`, which is what the `fs::write` this replaced
    // produced. registry.json holds the model catalogue, which ships bundled in
    // the binary and is not private, and it has an external reader: birda-gui
    // reads it straight off disk. Creating it 0600 would break a shared install
    // on first run and never recover, because nothing widens it afterwards.
    // config.toml is the file in this directory that is created private, and it
    // is private because it can name paths the user would rather not advertise.
    crate::utils::fs::write_atomic(
        path,
        content.as_bytes(),
        crate::utils::fs::NewFileMode::Umask,
    )
    .map_err(|e| Error::RegistryWrite {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Bootstrap registry from bundled default.
fn bootstrap_registry(dest: &std::path::Path, registry: &Registry) -> Registry {
    // Infallible, and that is the point: the bundled registry this persists is
    // already in hand, so a first run on a read-only config directory should work
    // rather than refuse to start. See [`persist_registry`].
    persist_registry(dest, registry);
    registry.clone()
}

/// Find model entry by ID.
pub fn find_model<'a>(registry: &'a Registry, id: &str) -> Option<&'a ModelEntry> {
    registry.models.iter().find(|m| m.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::types::{FileInfo, LabelsInfo, LanguageVariant, LicenseInfo, ModelFiles};

    /// A registry carrying nothing but its version, for the write-path tests.
    ///
    /// The version is the payload: it is what makes two writes distinguishable
    /// when the point of the test is which of them a given name resolves to.
    fn versioned_registry(version: u32) -> Registry {
        Registry {
            schema_version: "1.0".into(),
            registry_version: version,
            range_filter: None,
            models: vec![],
        }
    }

    /// The `registry_version` recorded in the file at `path`.
    fn version_at(path: &std::path::Path) -> u32 {
        load_from_file(path).unwrap().registry_version
    }

    #[test]
    fn test_write_registry_file_creates_the_config_directory() {
        // The bootstrap case: registry.json is written into a config directory
        // that does not exist yet on a fresh install. This function used to
        // create it itself and now leans on the write helper, so the coverage
        // has to move rather than disappear.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing").join("registry.json");

        write_registry_file(&path, &versioned_registry(7)).unwrap();

        assert_eq!(version_at(&path), 7);
    }

    #[test]
    #[cfg(unix)]
    fn test_write_registry_file_does_not_truncate_the_target_in_place() {
        // #311. `fs::write` truncates the file and then writes it, so
        // registry.json is empty in between, and this runs on the first launch
        // after any upgrade that bumps the bundled registry version.
        //
        // A hardlink is a second name for the same inode: truncate-and-rewrite
        // shows the new contents through the link, write-then-rename leaves the
        // old inode intact behind it. Reading the old version back through the
        // link is proof the file was replaced rather than rewritten.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");

        write_registry_file(&path, &versioned_registry(1)).unwrap();
        let link = dir.path().join("registry.json.link");
        std::fs::hard_link(&path, &link).unwrap();

        write_registry_file(&path, &versioned_registry(2)).unwrap();

        assert_eq!(
            version_at(&link),
            1,
            "the previous registry must survive behind its own name; seeing \
             version 2 here means the file was truncated in place"
        );
        assert_eq!(
            version_at(&path),
            2,
            "the registry path must carry the new contents"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_a_new_registry_is_not_narrowed_to_its_owner() {
        // The `fs::write` this replaced created registry.json at whatever the
        // umask allowed, usually 0644. Publishing by rename hands the path the
        // temporary's inode, and a temporary is owner-only, so without an explicit
        // policy a fresh install would get a 0600 registry and never recover:
        // nothing widens it afterwards, and birda-gui reads this file straight off
        // disk, so a shared install would break on first run.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");

        write_registry_file(&path, &versioned_registry(1)).unwrap();

        // Compared against a file `File::create` made under the same umask rather
        // than against a literal 0o644, which would fail for anyone whose umask is
        // 0o077 or 0o002.
        let reference = dir.path().join("reference");
        drop(std::fs::File::create(&reference).unwrap());

        let mode_of =
            |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode_of(&path),
            mode_of(&reference),
            "the registry must keep the mode File::create would have given it"
        );

        // What this cannot see, said out loud rather than left as a silent pass:
        // under a umask that masks the group and world bits away, `Umask` and
        // `OwnerOnly` both yield 0o600, so the assertion above holds whichever
        // policy production passes and the regression it guards could return
        // undetected.
        if mode_of(&reference) & 0o066 == 0 {
            eprintln!(
                "skipped the policy distinction: this umask masks both policies to \
                 {:o}, so a wrong one would not be detected here",
                mode_of(&reference)
            );
        }
    }

    #[test]
    fn test_write_registry_file_leaves_no_temporary_behind() {
        // The temporary has to be created beside the target, because rename is
        // only atomic within a filesystem and $TMPDIR is routinely a different
        // one. That puts it in the user's config directory, so a successful
        // write must clean up after itself.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");

        write_registry_file(&path, &versioned_registry(1)).unwrap();

        let strays: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name != "registry.json")
            .collect();
        assert!(
            strays.is_empty(),
            "a successful write must not litter the config directory, found: {strays:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_a_registry_that_could_not_be_read_is_left_alone() {
        // #323. Every failure to LOAD used to be treated as grounds to overwrite
        // the file with the bundled copy, and `load_from_file` returns two
        // semantically different errors. An unreadable file (EACCES, EIO, a flaky
        // network mount, fd exhaustion during a parallel batch) may be perfectly
        // intact, and replacing it destroys any user-local edits with no message.
        // birda-gui reads this file straight off disk, so a hand-maintained
        // registry disappearing is a user-visible outcome, not a cache miss.
        //
        // Mode 0000 blocks the read but not the rename: rename needs write and
        // execute on the DIRECTORY, not on the file. So the buggy version really
        // does replace this file, and this assertion really does distinguish them.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        let sentinel = serde_json::to_string_pretty(&versioned_registry(1)).unwrap();
        std::fs::write(&path, &sentinel).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();

        if std::fs::read_to_string(&path).is_ok() {
            eprintln!(
                "skipped: this process can read a mode-0000 file (running as root, \
                 or a filesystem that ignores modes), so the read-failure arm \
                 cannot be reached here"
            );
            return;
        }

        let loaded = load_registry_from(&path, versioned_registry(99));

        assert_eq!(
            loaded.registry_version, 99,
            "the caller must still get a usable registry, from the bundled copy"
        );

        // Widen it again so the assertion below can read it back.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            sentinel,
            "an intact but unreadable registry must survive byte for byte; \
             rewriting it destroys user-local edits for an error that says \
             nothing about the file's contents"
        );
    }

    #[test]
    fn test_a_corrupt_registry_is_still_replaced() {
        // The other half of #323, pinned so the fix cannot overshoot. A file that
        // does not parse holds nothing worth keeping, and leaving it would wedge
        // every run behind a permanent fallback that never repairs itself.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        std::fs::write(&path, b"{ this is not json").unwrap();

        let loaded = load_registry_from(&path, versioned_registry(42));

        assert_eq!(loaded.registry_version, 42);
        assert_eq!(
            version_at(&path),
            42,
            "a registry that cannot be parsed must be repaired on disk, not just \
             in memory"
        );
    }

    #[test]
    fn test_a_newer_bundled_registry_replaces_the_cached_one() {
        // The first launch after any upgrade that bumps the bundled registry
        // takes this branch, so it is the most-travelled write path in the file.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        write_registry_file(&path, &versioned_registry(1)).unwrap();

        let loaded = load_registry_from(&path, versioned_registry(2));

        assert_eq!(loaded.registry_version, 2);
        assert_eq!(version_at(&path), 2, "the cached copy must be updated too");
    }

    #[test]
    fn test_a_registry_newer_than_the_bundled_one_is_kept() {
        // The downgrade guard: a user whose registry.json is ahead of the binary
        // (a newer birda ran, or they edited it) must not have it rolled back.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        write_registry_file(&path, &versioned_registry(9)).unwrap();

        let loaded = load_registry_from(&path, versioned_registry(2));

        assert_eq!(loaded.registry_version, 9);
        assert_eq!(version_at(&path), 9, "the user's copy must survive");
    }

    #[test]
    fn test_find_model_by_id() {
        let registry = Registry {
            schema_version: "1.0".into(),
            registry_version: 0,
            range_filter: None,
            models: vec![
                ModelEntry {
                    id: "test-1".into(),
                    name: "Test Model 1".into(),
                    description: "First test model".into(),
                    vendor: "Test Vendor".into(),
                    version: "1.0".into(),
                    model_type: "birdnet-v24".into(),
                    license: LicenseInfo {
                        r#type: "MIT".into(),
                        url: "https://example.com".into(),
                        commercial_use: true,
                        attribution_required: false,
                        share_alike: false,
                    },
                    files: Some(ModelFiles {
                        model: FileInfo {
                            url: "https://example.com/model.onnx".into(),
                            filename: "model.onnx".into(),
                            sha256: None,
                            size_bytes: None,
                        },
                        labels: LabelsInfo {
                            default_language: "en".into(),
                            languages: vec![LanguageVariant {
                                code: "en".into(),
                                name: "English".into(),
                                url: "https://example.com/labels.txt".into(),
                                filename: "labels.txt".into(),
                            }],
                        },
                        bsg_calibration: None,
                        bsg_migration: None,
                        bsg_distribution_maps: None,
                    }),
                    build: None,
                    default_variant: None,
                    selection: std::collections::BTreeMap::new(),
                    variants: Vec::new(),
                    recommended: false,
                },
                ModelEntry {
                    id: "test-2".into(),
                    name: "Test Model 2".into(),
                    description: "Second test model".into(),
                    vendor: "Test Vendor".into(),
                    version: "2.0".into(),
                    model_type: "perch-v2".into(),
                    license: LicenseInfo {
                        r#type: "Apache-2.0".into(),
                        url: "https://example.com".into(),
                        commercial_use: true,
                        attribution_required: true,
                        share_alike: false,
                    },
                    files: Some(ModelFiles {
                        model: FileInfo {
                            url: "https://example.com/model2.onnx".into(),
                            filename: "model2.onnx".into(),
                            sha256: None,
                            size_bytes: None,
                        },
                        labels: LabelsInfo {
                            default_language: "en".into(),
                            languages: vec![LanguageVariant {
                                code: "en".into(),
                                name: "English".into(),
                                url: "https://example.com/labels2.txt".into(),
                                filename: "labels2.txt".into(),
                            }],
                        },
                        bsg_calibration: None,
                        bsg_migration: None,
                        bsg_distribution_maps: None,
                    }),
                    build: None,
                    default_variant: None,
                    selection: std::collections::BTreeMap::new(),
                    variants: Vec::new(),
                    recommended: true,
                },
            ],
        };

        // Test finding existing models
        assert!(find_model(&registry, "test-1").is_some());
        assert!(find_model(&registry, "test-2").is_some());

        let model1 = find_model(&registry, "test-1").unwrap();
        assert_eq!(model1.name, "Test Model 1");
        assert_eq!(model1.version, "1.0");

        let model2 = find_model(&registry, "test-2").unwrap();
        assert_eq!(model2.name, "Test Model 2");
        assert!(model2.recommended);

        // Test finding non-existent model
        assert!(find_model(&registry, "missing").is_none());
    }

    #[test]
    fn test_bundled_registry_parses() {
        // This test verifies that the bundled registry.json is valid
        const BUNDLED_REGISTRY: &str = include_str!("../../registry.json");

        let result = serde_json::from_str::<Registry>(BUNDLED_REGISTRY);
        assert!(result.is_ok(), "Bundled registry should parse successfully");

        let registry = result.unwrap();
        assert_eq!(registry.schema_version, "2.0");
        assert!(
            !registry.models.is_empty(),
            "Registry should contain models"
        );

        // Verify we have expected models
        assert!(find_model(&registry, "birdnet-v24").is_some());
        assert!(find_model(&registry, "birdnet-v30").is_some());
        assert!(find_model(&registry, "perch-v2").is_some());
        assert!(find_model(&registry, "bsg-fi-v44").is_some());
    }

    #[test]
    fn test_bundled_registry_defines_the_geomodel_range_filter() {
        const BUNDLED_REGISTRY: &str = include_str!("../../registry.json");

        let registry = serde_json::from_str::<Registry>(BUNDLED_REGISTRY).unwrap();
        let range_filter = registry
            .range_filter
            .expect("bundled registry must define range_filter");

        assert_eq!(range_filter.version, "3.0.2");
        assert_eq!(range_filter.species_count, 12012);
        assert!(
            range_filter.name.contains("BirdNET"),
            "geomodel display name must credit BirdNET"
        );
        assert!(
            range_filter.model.sha256.is_some(),
            "geomodel model file must be checksum verified"
        );
        assert!(
            range_filter.labels.sha256.is_some(),
            "geomodel labels file must be checksum verified"
        );
        assert!(
            registry.registry_version >= 4,
            "registry_version must be bumped so existing installs pick up the geomodel"
        );
    }
}
