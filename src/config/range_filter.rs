//! Range filter configuration resolution.
//!
//! Every classifier uses the same `BirdNET` Geomodel v3.0.2, so this is a
//! single uniform path. Earlier versions carried a cross-model fallback that
//! let Perch borrow `BirdNET`'s meta model, plus the meta-model owner lookup
//! and label plumbing that went with it; the shared geomodel makes all of that
//! redundant.

use crate::cli::AnalyzeArgs;
use crate::config::types::{Config, ModelConfig, ModelType};
use crate::constants::confidence;
use crate::error::{Error, Result};
use crate::inference::RangeFilterConfig;
use crate::registry::InstalledRangeFilter;
use crate::utils::date::{date_to_week, day_of_year_to_date, week_to_start_day};

/// Whether a model participates in geomodel range filtering.
///
/// BSG has its own species distribution mechanism. Bat detections cannot be
/// range filtered by this geomodel at all: v3.0.2 scores resident bats at
/// roughly 0.002 to 0.016 against 0.85 and above for birds, so every bat
/// detection would fall below any useful threshold.
///
/// This is the single authority on the question. Do not re-derive it at the
/// point of use: an earlier duplicate in the classifier omitted the bat case.
pub fn supports_range_filter(args: &AnalyzeArgs, model_type: ModelType) -> bool {
    if args.bat.is_some() {
        return false;
    }

    match model_type {
        ModelType::BirdnetV24 | ModelType::BirdnetV30 | ModelType::PerchV2 => true,
        ModelType::BsgFinland => false,
    }
}

/// Whether range filtering is wanted at all, before the geomodel is resolved.
///
/// Checked ahead of acquisition so birda never prompts for, downloads, or
/// records a 15 MB geomodel it is then going to discard. Coordinates alone are
/// not enough: a run also needs a time parameter, and BSG and bat mode never
/// range filter however complete the query is.
#[must_use]
pub fn wants_range_filter(args: &AnalyzeArgs, config: &Config, model_type: ModelType) -> bool {
    let has_coordinates = args.lat.or(config.defaults.latitude).is_some()
        && args.lon.or(config.defaults.longitude).is_some();
    let has_time = args.week.is_some() || (args.month.is_some() && args.day.is_some());

    has_coordinates && has_time && supports_range_filter(args, model_type)
}

/// Build `RangeFilterConfig` from CLI args, config file, and resolved geomodel.
///
/// Range filtering activates when:
/// - coordinates are available (CLI or config)
/// - a time parameter is available (week, or month plus day)
/// - the model supports it (not BSG, not bat mode)
///
/// Returns `Ok(None)` when any condition is unmet.
pub fn build_range_filter_config(
    args: &AnalyzeArgs,
    config: &Config,
    model_config: &ModelConfig,
    geomodel: &InstalledRangeFilter,
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

    if !supports_range_filter(args, model_config.model_type) {
        return Ok(None);
    }

    // Convert week to month/day for the geomodel query.
    // Week 1 = Jan 1 (day 1), Week 48 = Dec 24 (day 358)
    let day_of_year = week_to_start_day(week);
    let (month, day) = day_of_year_to_date(day_of_year);

    // Threshold and unmatched policy: CLI overrides config
    let threshold = args
        .range_threshold
        .unwrap_or(config.defaults.range_threshold);

    // Bounds-check here, at the point of use, and not only in
    // `validate_range_filter`. `validate_config` has exactly one caller,
    // `handle_config_set`, so nothing validates a config that was hand-edited
    // rather than written through the CLI. That hand-edited case is the one the
    // bug report describes: a negative threshold makes filtering a silent
    // no-op while the JSON envelope still reports it active, and NaN drops
    // every mapped species because `score >= NaN` is false for every score.
    // Validating only where configs are written would have left the path the
    // defect actually travels wide open.
    //
    // The CLI flag reaches here already bounded by `parse_confidence`, so this
    // fires only for a config-file value.
    if !(confidence::MIN..=confidence::MAX).contains(&threshold) {
        return Err(Error::InvalidRangeThreshold { value: threshold });
    }
    let unmatched = args
        .range_unmatched
        .unwrap_or(config.defaults.range_unmatched);

    Ok(Some(RangeFilterConfig {
        geomodel_path: geomodel.model.clone(),
        geomodel_labels_path: geomodel.labels.clone(),
        threshold,
        latitude,
        longitude,
        month,
        day,
        rerank: args.rerank,
        unmatched,
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Test setup code - panics are acceptable
mod tests {
    use super::*;
    use crate::config::types::UnmatchedPolicy;
    use std::path::PathBuf;

    fn geomodel() -> InstalledRangeFilter {
        InstalledRangeFilter {
            model: PathBuf::from("/models/birdnet-geomodel-v3.0.2.onnx"),
            labels: PathBuf::from("/models/birdnet-geomodel-v3.0.2-labels.txt"),
        }
    }

    fn model_of(model_type: ModelType) -> ModelConfig {
        ModelConfig {
            path: PathBuf::from("model.onnx"),
            labels: PathBuf::from("labels.txt"),
            model_type,
            meta_model: None,
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        }
    }

    fn args_at_helsinki_week_24() -> AnalyzeArgs {
        AnalyzeArgs {
            lat: Some(60.1699),
            lon: Some(24.9384),
            week: Some(24),
            ..AnalyzeArgs::default()
        }
    }

    /// A config as a hand-edit would leave it: coordinates set so range
    /// filtering is wanted, and a threshold nothing on this path validated.
    fn config_with_threshold(threshold: f32) -> Config {
        let mut config = Config::default();
        config.defaults.range_threshold = threshold;
        config
    }

    fn build_with(config: &Config) -> crate::error::Result<Option<RangeFilterConfig>> {
        build_range_filter_config(
            &args_at_helsinki_week_24(),
            config,
            &model_of(ModelType::BirdnetV24),
            &geomodel(),
        )
    }

    #[test]
    fn test_hand_edited_nan_threshold_is_rejected_on_the_analyze_path() {
        // The reported defect in full. `validate_config` has one caller,
        // `handle_config_set`, so a threshold typed into config.toml by hand
        // reached analysis unchecked: `score >= NaN` is false for every score,
        // so every mapped species was dropped and the only symptom was an info
        // log reading "0 species in range".
        //
        // Guarding only `validate_range_filter` would leave this green while
        // the bug stayed live, because nothing on this path calls it.
        let err = build_with(&config_with_threshold(f32::NAN)).unwrap_err();

        assert!(
            matches!(err, crate::error::Error::InvalidRangeThreshold { value } if value.is_nan()),
            "expected InvalidRangeThreshold(NaN), got {err:?}"
        );
    }

    #[test]
    fn test_hand_edited_negative_threshold_is_rejected_on_the_analyze_path() {
        // The sibling half: a negative threshold made filtering a silent no-op
        // while the JSON envelope still reported it active.
        let err = build_with(&config_with_threshold(-1.0)).unwrap_err();

        assert!(
            matches!(err, crate::error::Error::InvalidRangeThreshold { .. }),
            "expected InvalidRangeThreshold, got {err:?}"
        );
    }

    #[test]
    fn test_valid_thresholds_still_build_on_the_analyze_path() {
        // The normal case must be untouched, including both inclusive bounds.
        for threshold in [0.0, 0.01, 0.5, 1.0] {
            assert!(
                build_with(&config_with_threshold(threshold)).is_ok(),
                "threshold {threshold} must be accepted"
            );
        }
    }

    #[test]
    fn test_build_range_filter_with_week() {
        let rf_config = build_range_filter_config(
            &args_at_helsinki_week_24(),
            &Config::default(),
            &model_of(ModelType::BirdnetV24),
            &geomodel(),
        )
        .unwrap()
        .unwrap();

        assert!((rf_config.latitude - 60.1699).abs() < f64::EPSILON);
        assert!((rf_config.longitude - 24.9384).abs() < f64::EPSILON);
        assert!((rf_config.threshold - 0.01).abs() < f32::EPSILON);
        // Week 24 = day 175 -> June 24
        assert_eq!(rf_config.month, 6);
        assert_eq!(rf_config.day, 24);
        assert_eq!(rf_config.geomodel_path, geomodel().model);
        assert_eq!(rf_config.geomodel_labels_path, geomodel().labels);
    }

    #[test]
    fn test_build_range_filter_with_month_day() {
        let args = AnalyzeArgs {
            lat: Some(60.1699),
            lon: Some(24.9384),
            month: Some(6),
            day: Some(15),
            ..AnalyzeArgs::default()
        };

        let rf_config = build_range_filter_config(
            &args,
            &Config::default(),
            &model_of(ModelType::BirdnetV24),
            &geomodel(),
        )
        .unwrap()
        .unwrap();

        // June 15 -> week 22 -> day 160 -> June 9 (precision loss in round-trip)
        assert_eq!(rf_config.month, 6);
        assert_eq!(rf_config.day, 9);
    }

    #[test]
    fn test_build_range_filter_with_config_defaults() {
        let args = AnalyzeArgs {
            week: Some(24),
            ..AnalyzeArgs::default()
        };
        let mut config = Config::default();
        config.defaults.latitude = Some(51.5074);
        config.defaults.longitude = Some(-0.1278);

        let rf_config = build_range_filter_config(
            &args,
            &config,
            &model_of(ModelType::BirdnetV24),
            &geomodel(),
        )
        .unwrap()
        .unwrap();

        assert!((rf_config.latitude - 51.5074).abs() < f64::EPSILON);
        assert!((rf_config.longitude - (-0.1278)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_range_filter_disabled_without_coordinates() {
        let args = AnalyzeArgs {
            week: Some(24),
            ..AnalyzeArgs::default()
        };

        let result = build_range_filter_config(
            &args,
            &Config::default(),
            &model_of(ModelType::BirdnetV24),
            &geomodel(),
        );

        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_build_range_filter_disabled_without_time() {
        let args = AnalyzeArgs {
            lat: Some(60.1699),
            lon: Some(24.9384),
            ..AnalyzeArgs::default()
        };

        let result = build_range_filter_config(
            &args,
            &Config::default(),
            &model_of(ModelType::BirdnetV24),
            &geomodel(),
        );

        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_range_filter_applies_to_every_non_bsg_model_type() {
        // The point of the shared geomodel: no model needs to borrow another's
        // range filter any more.
        for model_type in [
            ModelType::BirdnetV24,
            ModelType::BirdnetV30,
            ModelType::PerchV2,
        ] {
            let result = build_range_filter_config(
                &args_at_helsinki_week_24(),
                &Config::default(),
                &model_of(model_type),
                &geomodel(),
            )
            .unwrap();

            assert!(
                result.is_some(),
                "{model_type} must support the geomodel range filter"
            );
        }
    }

    #[test]
    fn test_build_range_filter_disabled_for_bsg_model() {
        // BSG uses its own species distribution maps.
        let result = build_range_filter_config(
            &args_at_helsinki_week_24(),
            &Config::default(),
            &model_of(ModelType::BsgFinland),
            &geomodel(),
        );

        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_bat_mode_disables_range_filtering() {
        // Geomodel v3.0.2 scores resident bats at 0.002-0.016 against 0.85+ for
        // birds, so it cannot serve as a range filter for bat detections.
        let args = AnalyzeArgs {
            bat: Some(crate::config::BatRegion::Eu),
            ..args_at_helsinki_week_24()
        };

        let result = build_range_filter_config(
            &args,
            &Config::default(),
            &model_of(ModelType::BirdnetV24),
            &geomodel(),
        );

        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_unmatched_policy_defaults_to_keep() {
        let rf_config = build_range_filter_config(
            &args_at_helsinki_week_24(),
            &Config::default(),
            &model_of(ModelType::BirdnetV24),
            &geomodel(),
        )
        .unwrap()
        .unwrap();

        assert_eq!(rf_config.unmatched, UnmatchedPolicy::Keep);
    }

    #[test]
    fn test_unmatched_policy_cli_overrides_config() {
        let args = AnalyzeArgs {
            range_unmatched: Some(UnmatchedPolicy::Drop),
            ..args_at_helsinki_week_24()
        };
        let mut config = Config::default();
        config.defaults.range_unmatched = UnmatchedPolicy::Keep;

        let rf_config = build_range_filter_config(
            &args,
            &config,
            &model_of(ModelType::BirdnetV24),
            &geomodel(),
        )
        .unwrap()
        .unwrap();

        assert_eq!(rf_config.unmatched, UnmatchedPolicy::Drop);
    }

    #[test]
    fn test_unmatched_policy_falls_back_to_config() {
        let mut config = Config::default();
        config.defaults.range_unmatched = UnmatchedPolicy::Drop;

        let rf_config = build_range_filter_config(
            &args_at_helsinki_week_24(),
            &config,
            &model_of(ModelType::BirdnetV24),
            &geomodel(),
        )
        .unwrap()
        .unwrap();

        assert_eq!(rf_config.unmatched, UnmatchedPolicy::Drop);
    }

    #[test]
    fn test_threshold_cli_overrides_config() {
        let args = AnalyzeArgs {
            range_threshold: Some(0.25),
            ..args_at_helsinki_week_24()
        };

        let rf_config = build_range_filter_config(
            &args,
            &Config::default(),
            &model_of(ModelType::BirdnetV24),
            &geomodel(),
        )
        .unwrap()
        .unwrap();

        assert!((rf_config.threshold - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rerank_flag_is_carried_through() {
        let args = AnalyzeArgs {
            rerank: true,
            ..args_at_helsinki_week_24()
        };

        let rf_config = build_range_filter_config(
            &args,
            &Config::default(),
            &model_of(ModelType::BirdnetV24),
            &geomodel(),
        )
        .unwrap()
        .unwrap();

        assert!(rf_config.rerank);
    }
}
