//! Configuration validation.

use crate::config::{Config, ModelConfig};
use crate::constants::confidence;
use crate::error::{Error, Result};

/// Validate the entire configuration.
pub fn validate_config(config: &Config) -> Result<()> {
    validate_defaults(config)?;
    validate_range_filter(config)?;
    Ok(())
}

/// Validate default settings.
fn validate_defaults(config: &Config) -> Result<()> {
    let defaults = &config.defaults;

    // Validate min_confidence range
    if !(confidence::MIN..=confidence::MAX).contains(&defaults.min_confidence) {
        return Err(Error::ConfigValidation {
            message: format!(
                "min_confidence must be between {} and {}, got {}",
                confidence::MIN,
                confidence::MAX,
                defaults.min_confidence
            ),
        });
    }

    // Validate overlap is non-negative
    if defaults.overlap < 0.0 {
        return Err(Error::ConfigValidation {
            message: format!("overlap must be non-negative, got {}", defaults.overlap),
        });
    }

    // Validate batch_size is at least 1
    if defaults.batch_size == 0 {
        return Err(Error::ConfigValidation {
            message: "batch_size must be at least 1".to_string(),
        });
    }

    // Validate default model exists if specified
    if let Some(ref model_name) = defaults.model
        && !config.models.contains_key(model_name)
    {
        return Err(Error::ModelNotFound {
            name: model_name.clone(),
        });
    }

    Ok(())
}

/// Validate a model configuration and check files exist.
#[allow(clippy::needless_pass_by_value)]
pub fn validate_model_config(_name: &str, model: &ModelConfig) -> Result<()> {
    if !model.path.exists() {
        return Err(Error::ModelFileNotFound {
            path: model.path.clone(),
        });
    }

    if !model.labels.exists() {
        return Err(Error::LabelsFileNotFound {
            path: model.labels.clone(),
        });
    }

    // Model type validation is handled by the ModelType enum

    Ok(())
}

/// Get a model by name from the config.
pub fn get_model<'a>(config: &'a Config, name: &str) -> Result<&'a ModelConfig> {
    config.models.get(name).ok_or_else(|| Error::ModelNotFound {
        name: name.to_string(),
    })
}

/// Validate range filter configuration.
pub fn validate_range_filter(config: &Config) -> Result<()> {
    if let Some(lat) = config.defaults.latitude
        && !(-90.0..=90.0).contains(&lat)
    {
        return Err(Error::InvalidLatitude { value: lat });
    }

    if let Some(lon) = config.defaults.longitude
        && !(-180.0..=180.0).contains(&lon)
    {
        return Err(Error::InvalidLongitude { value: lon });
    }

    if let Some(meta_path) = &config.defaults.meta_model
        && !meta_path.exists()
    {
        return Err(Error::MetaModelNotFound {
            path: meta_path.clone(),
        });
    }

    // Validate per-model meta model paths
    for model_config in config.models.values() {
        if let Some(meta_path) = &model_config.meta_model
            && !meta_path.exists()
        {
            return Err(Error::MetaModelNotFound {
                path: meta_path.clone(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_valid_config() {
        let config = Config::default();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_invalid_confidence() {
        let mut config = Config::default();
        config.defaults.min_confidence = 1.5;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_negative_overlap() {
        let mut config = Config::default();
        config.defaults.overlap = -1.0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_zero_batch_size() {
        let mut config = Config::default();
        config.defaults.batch_size = 0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_missing_default_model() {
        let mut config = Config::default();
        config.defaults.model = Some("nonexistent".to_string());
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_range_filter_invalid_latitude() {
        let mut config = Config::default();
        config.defaults.latitude = Some(100.0);

        let result = validate_range_filter(&config);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidLatitude { .. }));
    }

    #[test]
    fn test_validate_range_filter_invalid_longitude() {
        let mut config = Config::default();
        config.defaults.longitude = Some(200.0);

        let result = validate_range_filter(&config);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidLongitude { .. }));
    }

    #[test]
    fn test_validate_range_filter_valid_coordinates() {
        let mut config = Config::default();
        config.defaults.latitude = Some(40.7128);
        config.defaults.longitude = Some(-74.0060);

        let result = validate_range_filter(&config);
        assert!(result.is_ok());
    }
}
