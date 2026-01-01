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

    toml::from_str(&contents).map_err(|e| Error::ConfigParse {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Load configuration from the default platform-specific path.
///
/// Returns default config if no config file exists.
pub fn load_default_config() -> Result<Config> {
    super::config_file_path().map_or_else(|_| Ok(Config::default()), |path| load_config_file(&path))
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
}
