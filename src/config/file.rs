//! Configuration file loading.

use crate::config::Config;
use crate::error::{Error, Result};
use std::path::Path;

/// Load configuration from a TOML file.
///
/// Returns default config if the file does not exist.
pub fn load_config_file(path: &Path) -> Result<Config> {
    if !path.exists() {
        return Ok(Config::default());
    }

    let contents = std::fs::read_to_string(path).map_err(|e| Error::ConfigRead {
        path: path.to_path_buf(),
        source: e,
    })?;

    let config: Config = toml::from_str(&contents).map_err(|e| Error::ConfigParse {
        path: path.to_path_buf(),
        source: e,
    })?;

    warn_deprecated_keys(&config);

    Ok(config)
}

/// Warn about configuration keys that are parsed only to report deprecation.
///
/// Serde ignores unknown keys, so a key that has been removed from the structs
/// vanishes without a word. Keeping the field and reporting it here is the only
/// way a user learns that their setting stopped taking effect.
fn warn_deprecated_keys(config: &Config) {
    if config.defaults.meta_model.is_some() {
        tracing::warn!(
            "config key 'defaults.meta_model' is deprecated and ignored; range filtering now \
             uses the BirdNET Geomodel v3.0.2. The key is dropped the next time the config is \
             saved. Run 'birda models install geomodel' if range filtering is not working."
        );
    }

    for (name, model) in &config.models {
        if model.meta_model.is_some() {
            tracing::warn!(
                "config key 'models.{name}.meta_model' is deprecated and ignored; range \
                 filtering now uses the BirdNET Geomodel v3.0.2."
            );
        }
    }
}

/// Load configuration from the default platform-specific path.
///
/// Returns default config if no config file exists.
pub fn load_default_config() -> Result<Config> {
    super::config_file_path().map_or_else(|_| Ok(Config::default()), |path| load_config_file(&path))
}

/// Wrap an I/O failure as a config write error naming the target path.
fn write_error(path: &Path, source: std::io::Error) -> Error {
    Error::ConfigWrite {
        path: path.to_path_buf(),
        source,
    }
}

/// Save configuration to a TOML file, atomically.
///
/// The configuration is validated first, so no writer can persist a value that
/// `config set` would have rejected. Several callers build a config by mutating
/// a loaded one (`models add`, `models install`, `models remove`, the geomodel
/// install) and reached this function without validating.
///
/// The write itself goes through [`crate::utils::fs::write_atomic`], which
/// writes to a temporary beside the target and renames it over. Both halves
/// matter and they cover different failure modes. Validating first stops an
/// invalid config destroying a good one on disk; replacing by rename stops an
/// *interrupted* write doing the same.
///
/// The second one is why this is not a plain `fs::write`. That call truncates
/// the file and then writes, so ENOSPC, SIGKILL or power loss in between leaves
/// a zero-length or partial config.toml. A truncated config is not a loud
/// failure: an empty file is valid TOML, deserialises to `Config::default()`,
/// and the user silently loses every `[models.*]` block, their default model
/// and their geomodel paths, with no error at any point. A rename is atomic
/// within a filesystem, so a reader sees either the whole old file or the whole
/// new one.
///
/// One consequence of replacing rather than rewriting in place is handled here
/// rather than in the helper: rename replaces the *name* it is given, where a
/// write would have followed a symlink at that name, so the link is resolved
/// first. See [`resolve_link`].
///
/// Two things this deliberately does not address. The whole file is
/// re-serialised from the struct, so comments and unrecognised keys are dropped
/// as they always were. And every caller is a lock-free load-mutate-save, so
/// two concurrent saves still lose one of the two edits; the rename makes each
/// write whole, not the pair of them serialised.
pub fn save_config(config: &Config, path: &Path) -> Result<()> {
    super::validate_config(config)?;

    // Rename replaces the name it is given, where a write follows it. A user
    // whose config.toml is a symlink into a dotfiles repository would have the
    // link replaced by a regular file and their real config left stale, so the
    // link is resolved first and the write lands on the file it points at.
    let target = resolve_link(path);
    let target = target.as_path();

    let contents =
        toml::to_string_pretty(config).map_err(|e| Error::ConfigSerialize { source: e })?;

    // `OwnerOnly`, so a config file that does not exist yet is created private
    // rather than at whatever the umask allows. A config directory is per-user
    // and the file can name paths the user would rather not advertise. An
    // existing file keeps the mode it already had, which the helper handles.
    crate::utils::fs::write_atomic(
        target,
        contents.as_bytes(),
        crate::utils::fs::NewFileMode::OwnerOnly,
    )
    .map_err(|e| write_error(path, e))
}

/// The real file `path` names, following it if it is a symlink.
///
/// `canonicalize` alone is not enough, and the gap is the case this function
/// exists for. It requires every component INCLUDING the last to exist, so a
/// symlink whose target has not been created yet fails with `NotFound`. Falling
/// back to the unresolved path there would hand `persist` the link itself and
/// the rename would replace it with a regular file, which is precisely the
/// regression following the link is meant to prevent. It is also the ordinary
/// bootstrap order: link config.toml into a dotfiles repository first, run
/// birda second. `fs::write` handled it correctly, because `O_CREAT` follows a
/// dangling link and creates the target.
///
/// So a failed `canonicalize` falls back to reading the link by hand, and the
/// link's target is used whether or not the directory holding it exists yet,
/// since `save_config` creates that directory anyway. The unresolved path is
/// returned when `read_link` fails, which is usually because `path` is not a
/// symlink at all (the ordinary case of saving a config that does not exist
/// yet) and otherwise because the link is unreadable. Both want the same
/// answer.
///
/// Deliberately no "does the target's parent exist?" guard. An earlier revision
/// had one, and it reintroduced the exact bug this function exists to prevent,
/// one directory deeper: linking config.toml into a dotfiles directory that had
/// not been created yet failed the guard, fell back to the link, and the rename
/// ate it, silently and with exit 0.
///
/// The cost of having no guard, stated because it goes past what `fs::write`
/// did: a save through a dangling link now CREATES the directories leading to
/// its target, where `fs::write` would have failed with ENOENT. For a link into
/// an unmounted path, that materialises a directory inside the mountpoint. The
/// link is always one the user placed at their own config path, so nothing
/// crosses a privilege boundary, and this is still the better failure mode than
/// eating the link.
///
/// One hop, not a loop. A chain of dangling links is not worth the cycle
/// detection it would need. For `a -> b -> missing` the write lands on `b`
/// rather than on `a` or on the end of the chain, so `a` survives as a link and
/// still reads through, which is the property that matters. Every cycle
/// terminates, but only a cycle of length two or more preserves that property:
/// `a -> a` resolves to `a` itself and is replaced by a regular file. That
/// costs nothing in practice, since a self-referential config link cannot be
/// read either.
fn resolve_link(path: &Path) -> std::path::PathBuf {
    if let Ok(resolved) = std::fs::canonicalize(path) {
        return resolved;
    }

    let Ok(link_target) = std::fs::read_link(path) else {
        return path.to_path_buf();
    };

    // A relative link resolves against the directory holding the link, not the
    // process's working directory.
    if link_target.is_absolute() {
        link_target
    } else {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join(link_target)
    }
}

/// Save configuration to the default platform-specific path.
///
/// Validates before writing; see [`save_config`].
pub fn save_default_config(config: &Config) -> Result<std::path::PathBuf> {
    let path = super::config_file_path()?;
    save_config(config, &path)?;
    Ok(path)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_nonexistent_file_returns_default() {
        let path = Path::new("/nonexistent/path/config.toml");
        let config = load_config_file(path);
        assert!(config.is_ok());
        let config = config.ok().unwrap();
        assert!(config.models.is_empty());
    }

    #[test]
    fn test_load_valid_config() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[models.test-model]
path = "/path/to/model.onnx"
labels = "/path/to/labels.txt"
type = "birdnet-v24"

[defaults]
min_confidence = 0.25
"#
        )
        .unwrap();

        let config = load_config_file(file.path());
        assert!(config.is_ok());
        let config = config.ok().unwrap();
        assert!(config.models.contains_key("test-model"));
        assert_eq!(config.defaults.min_confidence, 0.25);
    }

    #[test]
    fn test_load_invalid_toml_returns_error() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "this is not valid toml {{{{").unwrap();

        let config = load_config_file(file.path());
        assert!(config.is_err());
    }

    /// A config that `config set` would reject, for the save-path tests.
    ///
    /// `min_confidence` is used because `validate_defaults` checks it first, so
    /// these assert the whole chain runs and not just the range-filter half.
    fn invalid_config() -> Config {
        let mut config = Config::default();
        config.defaults.min_confidence = 1.5;
        config
    }

    #[test]
    fn test_save_config_rejects_an_invalid_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        let err = save_config(&invalid_config(), &path).unwrap_err();

        assert!(
            matches!(err, Error::ConfigValidation { .. }),
            "expected ConfigValidation, got {err:?}"
        );
    }

    #[test]
    fn test_save_config_writes_nothing_when_validation_fails() {
        // `models add`, `models install`, `models remove` and the geomodel
        // install all mutate a loaded config and save it without validating, so
        // rejecting has to happen before the file is touched, not after.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        assert!(save_config(&invalid_config(), &path).is_err());
        assert!(
            !path.exists(),
            "a rejected save must not leave a config file behind"
        );
    }

    #[test]
    fn test_save_config_does_not_truncate_an_existing_config() {
        // The regression guard that matters most. This function re-serialises the
        // whole file, and a half-written config parses as all-defaults, silently
        // losing every model the user had. Validating before the write is what
        // stops a bad config destroying a good one; the sibling test below covers
        // the other half, that an interrupted write cannot do it either.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        let mut good = Config::default();
        good.defaults.min_confidence = 0.25;
        save_config(&good, &path).unwrap();
        let before = std::fs::read_to_string(&path).unwrap();

        assert!(save_config(&invalid_config(), &path).is_err());

        let after = std::fs::read_to_string(&path).unwrap();
        assert_eq!(before, after, "the existing config must survive untouched");
        assert_eq!(
            load_config_file(&path).unwrap().defaults.min_confidence,
            0.25
        );
    }

    /// Every entry beside `path` that the caller did not expect to be there.
    ///
    /// A leaked temporary would sit next to the real config, which is both
    /// litter in the user's config directory and, on the failure paths,
    /// evidence that a partially written file survived. Stated as "anything
    /// unexpected" rather than "anything matching the temp-file prefix", so it
    /// still catches a leak if `tempfile` ever changes how it names them.
    fn strays_beside(path: &Path, expected: &[&str]) -> Vec<String> {
        std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| !expected.contains(&name.as_str()))
            .collect()
    }

    #[test]
    #[cfg(unix)]
    fn test_save_config_does_not_truncate_the_target_in_place() {
        // The #307 guard, and the only one that can fail deterministically
        // without simulating a crash.
        //
        // A hardlink is a second name for the same inode. `fs::write` truncates
        // and rewrites in place, so the link would show the new contents; a
        // write to a sibling temp file followed by `rename` gives the config
        // path a *different* inode, leaving the old one intact behind the link.
        // Reading the old contents back through the link is therefore proof the
        // target was never truncated, which is the specific regression #307 is
        // about.
        //
        // Named for what it proves rather than for atomicity, because it proves
        // less than that. It shows the config path acquired a new inode; it
        // cannot see whether a reader had an uninterrupted view throughout.
        // Unlinking the target immediately before the rename, which opens a
        // window where the config does not exist at all, leaves this green.
        // Closing that would need a concurrent reader or a crash harness.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        let mut before = Config::default();
        before.defaults.min_confidence = 0.25;
        save_config(&before, &path).unwrap();

        let link = dir.path().join("config.toml.link");
        std::fs::hard_link(&path, &link).unwrap();

        let mut after = Config::default();
        after.defaults.min_confidence = 0.75;
        save_config(&after, &path).unwrap();

        assert_eq!(
            load_config_file(&link).unwrap().defaults.min_confidence,
            0.25,
            "the previous file must survive untouched behind its own name; \
             seeing 0.75 here means the config path was truncated in place \
             rather than replaced by a rename"
        );
        assert_eq!(
            load_config_file(&path).unwrap().defaults.min_confidence,
            0.75,
            "the config path must carry the new contents"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_save_config_works_when_the_config_is_not_on_the_temp_filesystem() {
        // Pins the invariant every other test here is blind to: the temporary
        // must be created in the TARGET's directory, not in $TMPDIR. Rename is
        // only atomic within a filesystem and fails outright with EXDEV across
        // one, so a temporary in $TMPDIR breaks for the ordinary desktop layout
        // of /tmp on tmpfs and ~/.config on disk.
        //
        // Every other test saves into `tempfile::tempdir()`, which is itself
        // under $TMPDIR, so `persist` never crosses a device and swapping the
        // helper's `tempfile_in(dir)` for a plain `tempfile()` leaves them all
        // green. Saving into /dev/shm puts the target on a different filesystem
        // from /tmp, which is what makes this one able to fail.
        //
        // Two environments make this inert: /dev/shm missing, and $TMPDIR
        // pointed at /dev/shm so both ends share a device. It skips rather than
        // fails in both, because a red build over an environment quirk is worse
        // than a test that stops proving something, and `TMPDIR=/dev/shm` is a
        // real technique for speeding up tempfile-heavy suites.
        //
        // Being honest about the cost: libtest captures the output of passing
        // tests unless `--nocapture` is passed, and CI does not pass it, so on
        // such a machine this goes quiet with nothing to show for it. Preferred
        // anyway over the alternative of failing, and over a silent vacuous
        // pass, which is what this replaced.
        use std::os::unix::fs::MetadataExt;

        let shm = Path::new("/dev/shm");
        let Ok(dir) = tempfile::tempdir_in(shm) else {
            eprintln!("skipped: /dev/shm is unusable, cross-filesystem save not exercised");
            return;
        };

        // `.ok()` rather than `.unwrap()`: a broken $TMPDIR should not surface
        // here as a panic pointing at a device probe.
        let device_of = |p: &Path| std::fs::metadata(p).map(|m| m.dev()).ok();
        if device_of(dir.path()) == device_of(&std::env::temp_dir()) {
            eprintln!(
                "skipped: /dev/shm and $TMPDIR share a device, cross-filesystem save not exercised"
            );
            return;
        }

        let path = dir.path().join("config.toml");

        let mut config = Config::default();
        config.defaults.min_confidence = 0.37;
        save_config(&config, &path)
            .expect("a save must not depend on $TMPDIR sharing a filesystem");

        assert_eq!(
            load_config_file(&path).unwrap().defaults.min_confidence,
            0.37
        );
    }

    #[test]
    fn test_save_config_leaves_no_temporary_file_behind() {
        // The temporary has to be created beside the target, because `rename`
        // is only atomic within a filesystem and $TMPDIR is routinely a
        // different one. That puts it in the user's config directory, so a
        // successful save must clean up after itself.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        save_config(&Config::default(), &path).unwrap();

        // Bound once rather than called twice: two `read_dir` calls could in
        // principle disagree, so the message would explain a different state
        // from the one the condition rejected.
        let strays = strays_beside(&path, &["config.toml"]);
        assert!(
            strays.is_empty(),
            "a successful save must not litter the config directory, found: {strays:?}"
        );
    }

    #[test]
    fn test_a_rejected_save_leaves_no_temporary_file_behind() {
        // The failure path of the same concern. Validation runs before the
        // temporary is created, so there is nothing to clean up here; the test
        // exists so a later reordering that validates *after* writing cannot
        // pass silently.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        assert!(save_config(&invalid_config(), &path).is_err());

        let strays = strays_beside(&path, &[]);
        assert!(
            strays.is_empty(),
            "a rejected save must not leave a partial file behind, found: {strays:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_save_config_preserves_the_file_mode() {
        // Replacing by rename means the new file's permissions come from the
        // temporary, not from the file being replaced, so a rewrite would
        // silently reset whatever mode the user had chosen. A temporary is created
        // owner-only, so without the mode being carried across the first
        // `config set` after the upgrade would quietly narrow a shared 0644
        // config. The mechanism lives in `crate::utils::fs::write_atomic_with`;
        // this pins the contract at the config route.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        save_config(&Config::default(), &path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        let mut changed = Config::default();
        changed.defaults.min_confidence = 0.5;
        save_config(&changed, &path).unwrap();

        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o644,
            "a rewrite must keep the mode the user set"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_a_new_config_is_created_private() {
        // The other half of the rule above. There is no previous mode to carry
        // over for a file that does not exist yet, and a config directory is
        // per-user, so the file is created restrictively rather than at whatever
        // the umask allows.
        //
        // What this guards changed with the extraction: the privacy used to be
        // structural, because a temporary is created owner-only and there was no
        // way to ask for anything else. It is now the `NewFileMode::OwnerOnly`
        // argument in `save_config`, so this test is what catches that argument
        // being changed to `Umask`. It only catches it under a umask that admits
        // group or world bits, since a umask of 0o077 masks both policies to the
        // same 0o600; the helper's own mode test says so out loud when it happens.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        save_config(&Config::default(), &path).unwrap();

        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o077,
            0,
            "a newly created config must not be group or world accessible"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_save_config_writes_through_a_dangling_symlink() {
        // The twin of the test below, and the one that fails against a
        // `canonicalize`-only resolve. `canonicalize` needs the final component
        // to exist, so a link whose target has not been created yet does not
        // resolve, and falling back to the unresolved path makes the rename eat
        // the link. This is the ordinary bootstrap order for a dotfiles setup:
        // create the link, then run birda for the first time.
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("dotfiles.toml");
        let link_dir = dir.path().join("config");
        std::fs::create_dir(&link_dir).unwrap();
        let link = link_dir.join("config.toml");

        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(
            !real.exists(),
            "the target must be absent for this to test anything"
        );

        let mut config = Config::default();
        config.defaults.min_confidence = 0.33;
        save_config(&config, &link).unwrap();

        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "a dangling link must survive the save, not be replaced by a file"
        );
        assert_eq!(
            load_config_file(&real).unwrap().defaults.min_confidence,
            0.33,
            "the save must create the file the dangling link points at"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_save_config_writes_through_a_dangling_symlink_into_a_missing_directory() {
        // The same case one directory deeper, and the one a "does the target's
        // parent exist?" guard silently broke: the guard rejected, the code
        // fell back to the link, and the rename ate it with exit 0. Linking
        // config.toml into a dotfiles directory that does not exist yet is an
        // ordinary thing to do on a fresh machine.
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("dotfiles").join("birda.toml");
        let link = dir.path().join("config.toml");

        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(
            !real.parent().unwrap().exists(),
            "the target's directory must be absent for this to test anything"
        );

        let mut config = Config::default();
        config.defaults.min_confidence = 0.29;
        save_config(&config, &link).unwrap();

        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the link must survive even when its target's directory had to be created"
        );
        assert_eq!(
            load_config_file(&real).unwrap().defaults.min_confidence,
            0.29,
            "the save must create the directory and the file the link points at"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_save_config_writes_through_a_symlinked_config() {
        // The one behaviour a rename changes for the worse if left alone.
        // `fs::write` follows a symlink; `rename` replaces it. Keeping a
        // config.toml symlinked into a dotfiles repository is common enough
        // that silently swapping the link for a regular file, and leaving the
        // real file stale, would be a regression traded for the atomicity.
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("dotfiles.toml");
        let link_dir = dir.path().join("config");
        std::fs::create_dir(&link_dir).unwrap();
        let link = link_dir.join("config.toml");

        save_config(&Config::default(), &real).unwrap();
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let mut changed = Config::default();
        changed.defaults.min_confidence = 0.42;
        save_config(&changed, &link).unwrap();

        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the symlink must survive the save rather than be replaced by a file"
        );
        assert_eq!(
            load_config_file(&real).unwrap().defaults.min_confidence,
            0.42,
            "the write must land on the file the link points at"
        );
        let strays = strays_beside(&real, &["dotfiles.toml", "config"]);
        assert!(
            strays.is_empty(),
            "the temporary belongs beside the resolved file, and must be gone, found: {strays:?}"
        );
    }

    #[test]
    fn test_save_config_still_writes_a_valid_config() {
        // The happy path, so the guard above cannot pass by rejecting everything.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("config.toml");

        let mut config = Config::default();
        config.defaults.latitude = Some(60.17);
        config.defaults.longitude = Some(24.94);
        save_config(&config, &path).unwrap();

        let loaded = load_config_file(&path).unwrap();
        assert_eq!(loaded.defaults.latitude, Some(60.17));
        assert_eq!(loaded.defaults.longitude, Some(24.94));
    }
}
