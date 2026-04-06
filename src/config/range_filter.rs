//! Range filter configuration resolution.

use crate::cli::AnalyzeArgs;
use crate::config::types::{Config, ModelConfig, ModelType};
use crate::error::Result;
use crate::inference::RangeFilterConfig;
use crate::utils::date::{date_to_week, day_of_year_to_date, week_to_start_day};
use std::path::PathBuf;

/// Find a fallback meta model from other installed models.
///
/// Searches all configured models (excluding the current one and BSG models)
/// for one that has a `meta_model`. Prefers `BirdNET` models for largest species coverage.
/// Uses alphabetical model name ordering as a deterministic tiebreaker.
///
/// Returns `(model_name, model_config, meta_model_path)` so callers need no unwrapping.
fn find_fallback_meta_model<'a>(
    config: &'a Config,
    current_model_name: &str,
) -> Option<(&'a str, &'a ModelConfig, &'a PathBuf)> {
    let mut candidates: Vec<(&str, &ModelConfig, &PathBuf)> = config
        .models
        .iter()
        .filter_map(|(name, mc)| {
            if name.as_str() != current_model_name && mc.model_type != ModelType::BsgFinland {
                mc.meta_model.as_ref().map(|meta| (name.as_str(), mc, meta))
            } else {
                None
            }
        })
        .collect();

    // Sort: BirdNET models first, then alphabetical by name
    candidates.sort_by(|(name_a, mc_a, _), (name_b, mc_b, _)| {
        let a_is_birdnet = matches!(
            mc_a.model_type,
            ModelType::BirdnetV24 | ModelType::BirdnetV30
        );
        let b_is_birdnet = matches!(
            mc_b.model_type,
            ModelType::BirdnetV24 | ModelType::BirdnetV30
        );
        b_is_birdnet.cmp(&a_is_birdnet).then(name_a.cmp(name_b))
    });

    candidates.into_iter().next()
}

/// Find the model that owns a given meta model path.
///
/// When a meta model is configured directly (per-model or via defaults) for a
/// non-BirdNET model, we need to find the labels file of the model that the
/// meta model belongs to for cross-model label mapping.
fn find_meta_model_owner<'a>(
    config: &'a Config,
    meta_model_path: &PathBuf,
) -> Option<(&'a str, &'a ModelConfig)> {
    let mut candidates: Vec<(&str, &ModelConfig)> = config
        .models
        .iter()
        .filter_map(|(name, mc)| {
            mc.meta_model
                .as_ref()
                .filter(|p| *p == meta_model_path)
                .map(|_| (name.as_str(), mc))
        })
        .collect();

    // Sort: BirdNET models first for deterministic selection
    candidates.sort_by(|(name_a, mc_a), (name_b, mc_b)| {
        let a_is_birdnet = matches!(
            mc_a.model_type,
            ModelType::BirdnetV24 | ModelType::BirdnetV30
        );
        let b_is_birdnet = matches!(
            mc_b.model_type,
            ModelType::BirdnetV24 | ModelType::BirdnetV30
        );
        b_is_birdnet.cmp(&a_is_birdnet).then(name_a.cmp(name_b))
    });

    candidates.into_iter().next()
}

/// Check if a model type uses `BirdNET`-compatible labels for the range filter.
fn is_birdnet_model(model_type: ModelType) -> bool {
    matches!(model_type, ModelType::BirdnetV24 | ModelType::BirdnetV30)
}

/// Build `RangeFilterConfig` from CLI args and config file.
///
/// Range filtering activates when:
/// - Coordinates are available (CLI or config)
/// - Time parameter is available (week OR month+day)
/// - A meta model is configured (per-model, via defaults, or via cross-model fallback)
///
/// When the selected model has no meta model, other installed models are searched for
/// one that does. `BirdNET` models are preferred for largest species coverage.
///
/// Returns `Ok(None)` with a warning if any condition is unmet.
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

    // BSG models use their own species distribution mechanism, not meta-model range filtering
    if model_config.model_type == ModelType::BsgFinland {
        return Ok(None);
    }

    // Get meta model path (per-model overrides global default)
    let direct_meta_model = model_config
        .meta_model
        .as_ref()
        .or(config.defaults.meta_model.as_ref());

    // Resolve meta model path and determine if cross-model label mapping is needed.
    //
    // Cross-model mode activates when the current model (e.g., perch-v2) uses a
    // meta model from a different model family (e.g., BirdNET). In this case we
    // need the meta model owner's labels for correct output-size validation and
    // score remapping.
    let (meta_model_path, cross_model_labels, meta_model_source) =
        if let Some(path) = direct_meta_model {
            if is_birdnet_model(model_config.model_type) {
                // Same-model mode: BirdNET model using its own meta model
                (path.clone(), None, None)
            } else if let Some((source_name, source_config)) =
                find_meta_model_owner(config, path)
            {
                // Cross-model: non-BirdNET model with a directly configured meta model
                // that belongs to another installed model
                tracing::info!(
                    "Using range filter from model '{}' for model '{}' (cross-model mode)",
                    source_name,
                    model_name
                );
                (
                    path.clone(),
                    Some(source_config.labels.clone()),
                    Some(source_name.to_string()),
                )
            } else {
                // Meta model is configured but we can't find its owner's labels.
                // This happens when defaults.meta_model points to a file but the
                // owning model isn't installed. Try fallback discovery.
                tracing::warn!(
                    "Meta model configured for '{}' but no matching model found for labels, \
                     trying fallback discovery",
                    model_name
                );
                if let Some((source_name, source_config, meta_path)) =
                    find_fallback_meta_model(config, model_name)
                {
                    tracing::info!(
                        "Using range filter from model '{}' for model '{}' (cross-model mode)",
                        source_name,
                        model_name
                    );
                    (
                        meta_path.clone(),
                        Some(source_config.labels.clone()),
                        Some(source_name.to_string()),
                    )
                } else {
                    tracing::warn!(
                        "Range filtering disabled for model '{}': \
                         no labels found for meta model",
                        model_name
                    );
                    return Ok(None);
                }
            }
        } else if let Some((source_name, source_config, meta_path)) =
            find_fallback_meta_model(config, model_name)
        {
            // No direct meta model — cross-model fallback found
            tracing::info!(
                "Using range filter from model '{}' for model '{}' (cross-model mode)",
                source_name,
                model_name
            );
            (
                meta_path.clone(),
                Some(source_config.labels.clone()),
                Some(source_name.to_string()),
            )
        } else {
            tracing::warn!(
                "Range filtering disabled for model '{}': no meta model configured",
                model_name
            );
            return Ok(None);
        };

    // Get threshold (CLI overrides config)
    let threshold = args
        .range_threshold
        .unwrap_or(config.defaults.range_threshold);

    Ok(Some(RangeFilterConfig {
        meta_model_path,
        threshold,
        latitude,
        longitude,
        month,
        day,
        rerank: args.rerank,
        cross_model_labels,
        meta_model_source,
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
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
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
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
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
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
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
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
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
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
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
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        };

        let result = build_range_filter_config(&args, &config, &model_config, "test-model");

        // Should gracefully return None instead of erroring
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_build_range_filter_cross_model_fallback() {
        use crate::config::types::{Config, ModelConfig, ModelType};
        use std::path::PathBuf;

        let mut args = crate::cli::AnalyzeArgs::default();
        args.lat = Some(60.1699);
        args.lon = Some(24.9384);
        args.week = Some(24);

        // Config has another model with a meta model installed
        let mut config = Config::default();
        config.models.insert(
            "birdnet-v24".to_string(),
            ModelConfig {
                path: PathBuf::from("birdnet.onnx"),
                labels: PathBuf::from("birdnet_labels.txt"),
                model_type: ModelType::BirdnetV24,
                meta_model: Some(PathBuf::from("meta.onnx")),
                bsg_calibration: None,
                bsg_migration: None,
                bsg_distribution_maps: None,
            },
        );

        // The model being used has no meta model
        let model_config = ModelConfig {
            path: PathBuf::from("perch.onnx"),
            labels: PathBuf::from("perch_labels.csv"),
            model_type: ModelType::PerchV2,
            meta_model: None,
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        };

        let result = build_range_filter_config(&args, &config, &model_config, "perch-v2");

        assert!(result.is_ok());
        let rf_config = result.unwrap().unwrap();
        assert_eq!(rf_config.meta_model_path, PathBuf::from("meta.onnx"));
        assert_eq!(
            rf_config.cross_model_labels,
            Some(PathBuf::from("birdnet_labels.txt"))
        );
        assert_eq!(rf_config.meta_model_source, Some("birdnet-v24".to_string()));
    }

    #[test]
    fn test_build_range_filter_no_fallback_bsg_only() {
        use crate::config::types::{Config, ModelConfig, ModelType};
        use std::path::PathBuf;

        let mut args = crate::cli::AnalyzeArgs::default();
        args.lat = Some(60.1699);
        args.lon = Some(24.9384);
        args.week = Some(24);

        // Only BSG model installed (no meta model)
        let mut config = Config::default();
        config.models.insert(
            "bsg-fi-v44".to_string(),
            ModelConfig {
                path: PathBuf::from("bsg.onnx"),
                labels: PathBuf::from("bsg_labels.txt"),
                model_type: ModelType::BsgFinland,
                meta_model: None,
                bsg_calibration: Some(PathBuf::from("cal.csv")),
                bsg_migration: Some(PathBuf::from("mig.csv")),
                bsg_distribution_maps: Some(PathBuf::from("dist.bin")),
            },
        );

        let model_config = ModelConfig {
            path: PathBuf::from("perch.onnx"),
            labels: PathBuf::from("perch_labels.csv"),
            model_type: ModelType::PerchV2,
            meta_model: None,
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        };

        let result = build_range_filter_config(&args, &config, &model_config, "perch-v2");

        // No fallback found, should return None gracefully
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_perch_with_defaults_meta_model_uses_cross_model() {
        use crate::config::types::{Config, ModelConfig, ModelType};
        use std::path::PathBuf;

        let mut args = crate::cli::AnalyzeArgs::default();
        args.lat = Some(60.1699);
        args.lon = Some(24.9384);
        args.week = Some(24);

        // Config has birdnet installed (owns the meta model) and defaults.meta_model set
        let mut config = Config::default();
        config.defaults.meta_model = Some(PathBuf::from("meta.onnx"));
        config.models.insert(
            "birdnet-v24".to_string(),
            ModelConfig {
                path: PathBuf::from("birdnet.onnx"),
                labels: PathBuf::from("birdnet_labels.txt"),
                model_type: ModelType::BirdnetV24,
                meta_model: Some(PathBuf::from("meta.onnx")),
                bsg_calibration: None,
                bsg_migration: None,
                bsg_distribution_maps: None,
            },
        );

        // Perch model has no meta_model — relies on defaults.meta_model
        let model_config = ModelConfig {
            path: PathBuf::from("perch.onnx"),
            labels: PathBuf::from("perch_labels.csv"),
            model_type: ModelType::PerchV2,
            meta_model: None,
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        };

        let result = build_range_filter_config(&args, &config, &model_config, "perch-v2");

        assert!(result.is_ok());
        let rf_config = result.unwrap().unwrap();
        assert_eq!(rf_config.meta_model_path, PathBuf::from("meta.onnx"));
        // Must use cross-model mode with BirdNET labels, not perch labels
        assert_eq!(
            rf_config.cross_model_labels,
            Some(PathBuf::from("birdnet_labels.txt"))
        );
        assert_eq!(rf_config.meta_model_source, Some("birdnet-v24".to_string()));
    }

    #[test]
    fn test_perch_with_direct_meta_model_uses_cross_model() {
        use crate::config::types::{Config, ModelConfig, ModelType};
        use std::path::PathBuf;

        let mut args = crate::cli::AnalyzeArgs::default();
        args.lat = Some(60.1699);
        args.lon = Some(24.9384);
        args.week = Some(24);

        // Config has birdnet installed
        let mut config = Config::default();
        config.models.insert(
            "birdnet-v24".to_string(),
            ModelConfig {
                path: PathBuf::from("birdnet.onnx"),
                labels: PathBuf::from("birdnet_labels.txt"),
                model_type: ModelType::BirdnetV24,
                meta_model: Some(PathBuf::from("meta.onnx")),
                bsg_calibration: None,
                bsg_migration: None,
                bsg_distribution_maps: None,
            },
        );

        // Perch model has meta_model set directly (e.g., user configured it)
        let model_config = ModelConfig {
            path: PathBuf::from("perch.onnx"),
            labels: PathBuf::from("perch_labels.csv"),
            model_type: ModelType::PerchV2,
            meta_model: Some(PathBuf::from("meta.onnx")),
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        };

        let result = build_range_filter_config(&args, &config, &model_config, "perch-v2");

        assert!(result.is_ok());
        let rf_config = result.unwrap().unwrap();
        assert_eq!(rf_config.meta_model_path, PathBuf::from("meta.onnx"));
        // Must detect cross-model even when meta_model is set directly on perch
        assert_eq!(
            rf_config.cross_model_labels,
            Some(PathBuf::from("birdnet_labels.txt"))
        );
        assert_eq!(rf_config.meta_model_source, Some("birdnet-v24".to_string()));
    }

    #[test]
    fn test_birdnet_with_direct_meta_model_uses_same_model() {
        use crate::config::types::{Config, ModelConfig, ModelType};
        use std::path::PathBuf;

        let mut args = crate::cli::AnalyzeArgs::default();
        args.lat = Some(60.1699);
        args.lon = Some(24.9384);
        args.week = Some(24);

        let config = Config::default();

        // BirdNET model with its own meta model — should be same-model mode
        let model_config = ModelConfig {
            path: PathBuf::from("birdnet.onnx"),
            labels: PathBuf::from("birdnet_labels.txt"),
            model_type: ModelType::BirdnetV24,
            meta_model: Some(PathBuf::from("meta.onnx")),
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        };

        let result = build_range_filter_config(&args, &config, &model_config, "birdnet-v24");

        assert!(result.is_ok());
        let rf_config = result.unwrap().unwrap();
        assert_eq!(rf_config.meta_model_path, PathBuf::from("meta.onnx"));
        // BirdNET should use same-model mode (no cross_model_labels)
        assert!(rf_config.cross_model_labels.is_none());
        assert!(rf_config.meta_model_source.is_none());
    }
}
