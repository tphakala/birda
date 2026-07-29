//! Configuration file loading.

use crate::config::Config;
use crate::error::{Error, Result};
use std::io::Write;
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
/// The write itself goes to a temporary file beside the target, which is then
/// renamed over it. Both halves matter and they cover different failure modes.
/// Validating first stops an invalid config destroying a good one on disk;
/// replacing by rename stops an *interrupted* write doing the same.
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
/// Two consequences of replacing rather than rewriting in place. The temporary
/// must live in the target's directory, because rename across filesystems is
/// not atomic (and `$TMPDIR` is routinely a different one), so it is created
/// there and cleaned up on every path out. And the resulting file takes its
/// permissions from the temporary rather than from the file it replaced, so an
/// existing mode is copied across explicitly; see [`preserve_existing_mode`].
///
/// Still not addressed: the whole file is re-serialised from the struct, so
/// comments and unrecognised keys are dropped as they always were.
pub fn save_config(config: &Config, path: &Path) -> Result<()> {
    super::validate_config(config)?;

    // Rename replaces the name it is given, where a write follows it. A user
    // whose config.toml is a symlink into a dotfiles repository would have the
    // link replaced by a regular file and their real config left stale, so the
    // link is resolved first and the write lands on the file it points at.
    let target = resolve_link(path);
    let target = target.as_path();

    // `parent` is empty for a bare relative filename like `config.toml`, which
    // is a valid path to save to and not the same thing as having no parent.
    let dir = target
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir).map_err(|e| write_error(target, e))?;

    let contents =
        toml::to_string_pretty(config).map_err(|e| Error::ConfigSerialize { source: e })?;

    let mut temp = tempfile::NamedTempFile::new_in(dir).map_err(|e| write_error(target, e))?;
    temp.write_all(contents.as_bytes())
        .map_err(|e| write_error(target, e))?;

    // Flush to disk before the rename, not after. Without this the rename can
    // reach the disk while the data behind it has not, which on a crash leaves
    // the config path pointing at a file of zeros: exactly the outcome the
    // rename is here to prevent, just reached by a different route.
    temp.as_file()
        .sync_all()
        .map_err(|e| write_error(target, e))?;

    preserve_existing_mode(target, &temp)?;

    // Drops the temporary on failure, so a rejected save leaves nothing behind.
    temp.persist(target)
        .map_err(|e| write_error(target, e.error))?;

    sync_directory(dir);

    Ok(())
}

/// The real file `path` names, following it if it is a symlink.
///
/// Falls back to `path` unchanged whenever it cannot be resolved, which is the
/// ordinary case of saving a config that does not exist yet. `canonicalize`
/// also flattens `.`, `..` and any symlinked parent directory, which is
/// harmless here: every use of the result is relative to the file itself.
fn resolve_link(path: &Path) -> std::path::PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Give the temporary the mode of the file it is about to replace.
///
/// `NamedTempFile` creates at 0600, so without this the first save after this
/// change would silently narrow a config the user had deliberately made
/// readable to others. A file that does not exist yet has no mode to carry
/// over and keeps the restrictive default, which is the right direction for a
/// per-user config directory and is what a fresh install now gets.
///
/// A `metadata` failure other than "not found" is not fatal. It means the mode
/// cannot be read, not that the save cannot proceed, and refusing to write the
/// user's config over it would trade a cosmetic difference for a lost setting.
/// The fallback is the restrictive mode, so the failure direction is safe.
#[cfg(unix)]
fn preserve_existing_mode(target: &Path, temp: &tempfile::NamedTempFile) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let Ok(existing) = std::fs::metadata(target) else {
        return Ok(());
    };

    let mode = existing.permissions().mode() & 0o777;
    temp.as_file()
        .set_permissions(std::fs::Permissions::from_mode(mode))
        .map_err(|e| write_error(target, e))
}

/// No-op on platforms without Unix permission bits.
///
/// Windows carries an ACL rather than a mode, and `MoveFileEx` preserves the
/// ACL of the file being moved rather than of the one being replaced, so there
/// is nothing to copy across by hand.
#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
fn preserve_existing_mode(_target: &Path, _temp: &tempfile::NamedTempFile) -> Result<()> {
    Ok(())
}

/// Flush the directory entry the rename created.
///
/// Best effort, deliberately. The rename is already atomic as far as any reader
/// is concerned; this only decides whether the *new* name or the old one
/// survives a power loss, and both are whole, valid configs. Some filesystems
/// reject `fsync` on a directory outright, so failing a save that has already
/// completed would cost the user a setting to buy durability that was never
/// required for correctness.
#[cfg(unix)]
fn sync_directory(dir: &Path) {
    if let Ok(handle) = std::fs::File::open(dir) {
        drop(handle.sync_all());
    }
}

/// No-op on platforms where a directory is not an openable file.
#[cfg(not(unix))]
fn sync_directory(_dir: &Path) {}

/// Save configuration to the default platform-specific path.
///
/// Validates before writing; see [`save_config`].
pub fn save_default_config(config: &Config) -> Result<std::path::PathBuf> {
    let path = super::config_file_path()?;
    save_config(config, &path)?;
    Ok(path)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
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
        // The regression guard that matters most. This function truncates and
        // rewrites the whole file, and a half-written config parses as
        // all-defaults, silently losing every model the user had. Validating
        // before the write is what stops a bad config destroying a good one.
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
    fn test_save_config_replaces_the_file_by_rename() {
        // The atomicity guard (#307), and the only one that can fail
        // deterministically without simulating a crash.
        //
        // A hardlink is a second name for the same inode. `fs::write` truncates
        // and rewrites in place, so the link would show the new contents; a
        // write to a sibling temp file followed by `rename` gives the config
        // path a *different* inode, leaving the old one intact behind the link.
        // Reading the old contents back through the link is therefore proof
        // that the target was never truncated, which is what makes an
        // interrupted write survivable: the reader either sees the whole old
        // file or the whole new one, never a zero-length file that parses as
        // all-defaults and silently drops every configured model.
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
    fn test_save_config_leaves_no_temporary_file_behind() {
        // The temporary has to be created beside the target, because `rename`
        // is only atomic within a filesystem and $TMPDIR is routinely a
        // different one. That puts it in the user's config directory, so a
        // successful save must clean up after itself.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        save_config(&Config::default(), &path).unwrap();

        assert!(
            strays_beside(&path, &["config.toml"]).is_empty(),
            "a successful save must not litter the config directory, found: {:?}",
            strays_beside(&path, &["config.toml"])
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

        assert!(
            strays_beside(&path, &[]).is_empty(),
            "a rejected save must not leave a partial file behind, found: {:?}",
            strays_beside(&path, &[])
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_save_config_preserves_the_file_mode() {
        // Replacing by rename means the new file's permissions come from the
        // temporary, not from the file being replaced, so a rewrite would
        // silently reset whatever mode the user had chosen. `NamedTempFile`
        // creates at 0600, so without this the first `config set` after the
        // upgrade would quietly narrow a shared 0644 config.
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
        // per-user, so the restrictive default `NamedTempFile` gives is kept
        // rather than widened to match what `fs::write` used to produce.
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
        let expected = ["dotfiles.toml", "config"];
        assert!(
            strays_beside(&real, &expected).is_empty(),
            "the temporary belongs beside the resolved file, and must be gone, found: {:?}",
            strays_beside(&real, &expected)
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
