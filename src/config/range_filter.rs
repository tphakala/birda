//! Range filter configuration resolution.

use crate::cli::AnalyzeArgs;
use crate::config::types::{Config, ModelConfig};
use crate::error::{Error, Result};
use crate::inference::RangeFilterConfig;
use crate::utils::date::{date_to_week, day_of_year_to_date, week_to_start_day};

/// Build `RangeFilterConfig` from CLI args and config file.
///
/// Range filtering activates when:
/// - Coordinates are available (CLI or config)
/// - Time parameter is available (week OR month+day)
pub fn build_range_filter_config(
    args: &AnalyzeArgs,
    config: &Config,
    model_config: &ModelConfig,
    model_name: &str,
) -> Result<Option<RangeFilterConfig>> {
    // Get coordinates (CLI overrides config)
    let latitude = args.lat.or(config.defaults.latitude);
    let longitude = args.lon.or(config.defaults.longitude);

    // Range filtering requires both coordinates
    let (Some(latitude), Some(longitude)) = (latitude, longitude) else {
        return Ok(None); // No coordinates - range filtering disabled
    };

    // Get week number: either from CLI or convert from month/day
    let week = if let Some(week) = args.week {
        week
    } else if let (Some(month), Some(day)) = (args.month, args.day) {
        date_to_week(month, day)
    } else {
        // No time parameter - range filtering disabled
        return Ok(None);
    };

    // Convert week to month/day for RangeFilter::predict
    // Week 1 = Jan 1 (day 1), Week 48 = Dec 24 (day 358)
    let day_of_year = week_to_start_day(week);
    let (month, day) = day_of_year_to_date(day_of_year);

    // Get meta model path (per-model overrides global default)
    let meta_model_path = model_config
        .meta_model
        .as_ref()
        .or(config.defaults.meta_model.as_ref())
        .ok_or_else(|| Error::MetaModelMissing {
            model_name: model_name.to_string(),
        })?;

    // Get threshold (CLI overrides config)
    let threshold = args
        .range_threshold
        .unwrap_or(config.defaults.range_threshold);

    Ok(Some(RangeFilterConfig {
        meta_model_path: meta_model_path.clone(),
        threshold,
        latitude,
        longitude,
        month,
        day,
        rerank: args.rerank,
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_build_range_filter_with_week() {
        use crate::config::types::{Config, ModelConfig, ModelType};
        use std::path::PathBuf;

        let mut args = crate::cli::AnalyzeArgs::default();
        args.lat = Some(60.1699);
        args.lon = Some(24.9384);
        args.week = Some(24);

        let config = Config::default();

        let model_config = ModelConfig {
            path: PathBuf::from("test.onnx"),
            labels: PathBuf::from("test.txt"),
            model_type: ModelType::BirdnetV24,
            meta_model: Some(PathBuf::from("meta.onnx")),
        };

        let result = build_range_filter_config(&args, &config, &model_config, "test-model");

        assert!(result.is_ok());
        let rf_config = result.unwrap().unwrap();
        assert_eq!(rf_config.latitude, 60.1699);
        assert_eq!(rf_config.longitude, 24.9384);
        assert_eq!(rf_config.threshold, 0.01); // Default threshold
        // Week 24 = day 175 → June 24
        assert_eq!(rf_config.month, 6);
        assert_eq!(rf_config.day, 24);
    }

    #[test]
    fn test_build_range_filter_with_month_day() {
        use crate::config::types::{Config, ModelConfig, ModelType};
        use std::path::PathBuf;

        let mut args = crate::cli::AnalyzeArgs::default();
        args.lat = Some(60.1699);
        args.lon = Some(24.9384);
        args.month = Some(6);
        args.day = Some(15);

        let config = Config::default();

        let model_config = ModelConfig {
            path: PathBuf::from("test.onnx"),
            labels: PathBuf::from("test.txt"),
            model_type: ModelType::BirdnetV24,
            meta_model: Some(PathBuf::from("meta.onnx")),
        };

        let result = build_range_filter_config(&args, &config, &model_config, "test-model");

        assert!(result.is_ok());
        let rf_config = result.unwrap().unwrap();
        assert_eq!(rf_config.latitude, 60.1699);
        assert_eq!(rf_config.longitude, 24.9384);
        // June 15 → week 22 → day 160 → June 9 (precision loss in round-trip)
        assert_eq!(rf_config.month, 6);
        assert_eq!(rf_config.day, 9);
    }

    #[test]
    fn test_build_range_filter_with_config_defaults() {
        use crate::config::types::{Config, ModelConfig, ModelType};
        use std::path::PathBuf;

        let mut args = crate::cli::AnalyzeArgs::default();
        args.week = Some(24);

        let mut config = Config::default();
        config.defaults.latitude = Some(51.5074);
        config.defaults.longitude = Some(-0.1278);

        let model_config = ModelConfig {
            path: PathBuf::from("test.onnx"),
            labels: PathBuf::from("test.txt"),
            model_type: ModelType::BirdnetV24,
            meta_model: Some(PathBuf::from("meta.onnx")),
        };

        let result = build_range_filter_config(&args, &config, &model_config, "test-model");

        assert!(result.is_ok());
        let rf_config = result.unwrap().unwrap();
        // Should use config defaults
        assert_eq!(rf_config.latitude, 51.5074);
        assert_eq!(rf_config.longitude, -0.1278);
    }

    #[test]
    fn test_build_range_filter_disabled_without_coordinates() {
        use crate::config::types::{Config, ModelConfig, ModelType};
        use std::path::PathBuf;

        let mut args = crate::cli::AnalyzeArgs::default();
        args.week = Some(24);

        let config = Config::default();

        let model_config = ModelConfig {
            path: PathBuf::from("test.onnx"),
            labels: PathBuf::from("test.txt"),
            model_type: ModelType::BirdnetV24,
            meta_model: Some(PathBuf::from("meta.onnx")),
        };

        let result = build_range_filter_config(&args, &config, &model_config, "test-model");

        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // Should be disabled
    }

    #[test]
    fn test_build_range_filter_disabled_without_time() {
        use crate::config::types::{Config, ModelConfig, ModelType};
        use std::path::PathBuf;

        let mut args = crate::cli::AnalyzeArgs::default();
        args.lat = Some(60.1699);
        args.lon = Some(24.9384);

        let config = Config::default();

        let model_config = ModelConfig {
            path: PathBuf::from("test.onnx"),
            labels: PathBuf::from("test.txt"),
            model_type: ModelType::BirdnetV24,
            meta_model: Some(PathBuf::from("meta.onnx")),
        };

        let result = build_range_filter_config(&args, &config, &model_config, "test-model");

        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // Should be disabled
    }

    #[test]
    fn test_build_range_filter_meta_model_missing() {
        use crate::config::types::{Config, ModelConfig, ModelType};
        use std::path::PathBuf;

        let mut args = crate::cli::AnalyzeArgs::default();
        args.lat = Some(60.1699);
        args.lon = Some(24.9384);
        args.week = Some(24);

        let config = Config::default();

        let model_config = ModelConfig {
            path: PathBuf::from("test.onnx"),
            labels: PathBuf::from("test.txt"),
            model_type: ModelType::BirdnetV24,
            meta_model: None, // No meta model
        };

        let result = build_range_filter_config(&args, &config, &model_config, "test-model");

        assert!(result.is_err());
        assert!(matches!(result, Err(Error::MetaModelMissing { .. })));
    }
}
