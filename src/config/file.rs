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

/// Save configuration to a TOML file.
///
/// The configuration is validated first, so no writer can persist a value that
/// `config set` would have rejected. Several callers build a config by mutating
/// a loaded one (`models add`, `models install`, `models remove`, the geomodel
/// install) and reached this function without validating.
///
/// Validating before the write also matters because this function truncates and
/// rewrites the whole file: refusing early means a bad config cannot destroy a
/// good one on disk.
pub fn save_config(config: &Config, path: &Path) -> Result<()> {
    super::validate_config(config)?;

    // Create parent directories if they don't exist
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::ConfigWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
    }

    let contents =
        toml::to_string_pretty(config).map_err(|e| Error::ConfigSerialize { source: e })?;

    std::fs::write(path, contents).map_err(|e| Error::ConfigWrite {
        path: path.to_path_buf(),
        source: e,
    })
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
