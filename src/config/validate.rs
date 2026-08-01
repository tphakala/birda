//! Configuration validation.

use crate::config::{Config, ModelConfig};
use crate::constants::{MAX_BATCH_SIZE, MIN_BATCH_SIZE, confidence, day_of_year};
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

    // Validate overlap is a finite, non-negative number.
    //
    // The `is_finite` half is what catches NaN, and NaN is the case that
    // matters: a bare `overlap < 0.0` is the hand-rolled comparison
    // `validate_range_filter` warns against below, and `NaN < 0.0` is false, so
    // it was accepted. `overlap * sample_rate` is then cast to `usize`, and Rust
    // saturates that cast rather than trapping, so NaN became 0 and the setting
    // was silently ignored. That is the reported bug's signature exactly: a
    // configured value that changes nothing and says nothing.
    //
    // Infinity is also caught, but it was never silent, so the gain there is
    // only that it now fails as a config error up front instead of as an
    // `Error::Internal` from `AudioDecoder::next_segment`, which rejects an
    // overlap at or above the segment length. That reject still catches an
    // oversized *finite* overlap, which this rule deliberately does not: 5.0
    // and 1e15 are both finite and both fail there, the latter because the same
    // saturating cast sends it to `usize::MAX`.
    //
    // So no upper bound is imposed here. One belongs with the segment length
    // rather than in a rule that cannot see it, and adding it would be new
    // policy rather than a fix.
    //
    // `--overlap` and `BIRDA_OVERLAP` win over this value and were checked by
    // nothing at all, so the same rule now lives in `cli::validators::
    // parse_overlap` (#306). The two are kept identical, wording included, and
    // `test_parse_overlap_matches_the_config_file_rule` drives both and
    // compares their verdicts, so neither side can be changed alone. Adding an
    // upper bound here, for instance, fails that test rather than silently
    // making config.toml stricter than the flag.
    if !defaults.overlap.is_finite() || defaults.overlap < 0.0 {
        return Err(Error::ConfigValidation {
            message: format!(
                "overlap must be a finite non-negative number, got {}",
                defaults.overlap
            ),
        });
    }

    // The upper bound is the half that was missing (#312). `parse_batch_size`
    // has rejected anything above `MAX_BATCH_SIZE` since the flag existed and
    // this side checked only `Some(0)`, so `--batch-size 100000` was refused
    // while `batch_size = 100000` hand-edited into config.toml went straight to
    // the inference path. The limit is there to prevent GPU memory exhaustion,
    // so the route that skipped it was the one most likely to carry a value
    // nobody had checked.
    //
    // Both sides read `MIN_BATCH_SIZE`/`MAX_BATCH_SIZE`, and
    // `test_parse_batch_size_matches_the_config_file_rule` drives the two and
    // compares their verdicts, so neither can be tightened alone.
    //
    // The wording deliberately stops short of the "reduce it or use --cpu"
    // advice `parse_batch_size` adds. The two layers stay distinguishable in a
    // test that way, which is what lets the `config set` and load-path tests
    // each pin their own layer rather than being satisfied by the other's
    // error.
    if let Some(batch_size) = defaults.batch_size
        && !(MIN_BATCH_SIZE..=MAX_BATCH_SIZE).contains(&batch_size)
    {
        return Err(Error::ConfigValidation {
            message: format!(
                "batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}, got {batch_size}"
            ),
        });
    }

    // Same shape one field over. `--day-of-year` and `BIRDA_DAY_OF_YEAR` are
    // bounded by `parse_day_of_year`; the file was not, and `handle_config_set`
    // had no arm for the key at all, so hand-editing config.toml was both the
    // only way to set it and the one route with no check. `day_of_year = 999`
    // then reached the BSG SDM path in `run()`, which rejects it from
    // birdnet-onnx, far from the value that caused it.
    if let Some(day) = defaults.day_of_year
        && !(day_of_year::MIN..=day_of_year::MAX).contains(&day)
    {
        return Err(Error::ConfigValidation {
            message: format!(
                "day_of_year must be between {} and {}, got {day}",
                day_of_year::MIN,
                day_of_year::MAX
            ),
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
pub fn validate_model_config(name: &str, model: &ModelConfig) -> Result<()> {
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

    // BSG models require additional post-processing files
    if model.model_type == crate::config::types::ModelType::BsgFinland {
        if model.bsg_calibration.is_none() {
            return Err(Error::BsgConfig {
                message: format!(
                    "BSG model '{name}' requires calibration file. Run 'birda models install {name}' to download required files"
                ),
            });
        }
        if model.bsg_migration.is_none() {
            return Err(Error::BsgConfig {
                message: format!(
                    "BSG model '{name}' requires migration file. Run 'birda models install {name}' to download required files"
                ),
            });
        }
        if model.bsg_distribution_maps.is_none() {
            return Err(Error::BsgConfig {
                message: format!(
                    "BSG model '{name}' requires distribution maps file. Run 'birda models install {name}' to download required files"
                ),
            });
        }
    }

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

    // `--range-threshold` is bounds-checked by `parse_confidence`, but nothing
    // checked the config-file path, so a hand-edited value went straight
    // through. Two states were silently wrong rather than rejected: a negative
    // threshold made range filtering a no-op while the JSON envelope still
    // reported it active, and NaN dropped every mapped species because
    // `score >= NaN` is false, leaving only the unmatched non-species labels.
    //
    // `contains` handles NaN without a special case: every NaN comparison is
    // false, so the range does not contain it and the negation rejects it. A
    // hand-rolled `threshold < 0.0` test would let NaN straight through.
    let threshold = config.defaults.range_threshold;
    if !(confidence::MIN..=confidence::MAX).contains(&threshold) {
        return Err(Error::InvalidRangeThreshold { value: threshold });
    }

    // The geomodel is resolved and acquired at analysis time rather than
    // validated here: it is a shared asset that birda can offer to download,
    // so a missing file is not a configuration error.

    Ok(())
}

#[cfg(test)]
// Test setup code: unwrapping a known-Err result is how these assert, and the
// float comparisons check that a literal passed in came back unchanged, so an
// epsilon would weaken them. Matches the convention used by the other test
// modules in this crate.
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

    /// Build a config carrying nothing but the overlap under test.
    fn config_with_overlap(overlap: f32) -> Config {
        let mut config = Config::default();
        config.defaults.overlap = overlap;
        config
    }

    #[test]
    fn test_validate_rejects_nan_overlap() {
        // `NaN < 0.0` is false, so the previous bare comparison accepted this.
        // The cast to `usize` then saturated NaN to 0 and the setting was
        // silently ignored, which is the same shape as the range-threshold bug.
        assert!(validate_config(&config_with_overlap(f32::NAN)).is_err());
    }

    #[test]
    fn test_validate_rejects_infinite_overlap() {
        // Positive infinity is the case `is_finite` adds here. It was already
        // caught deeper in, by `AudioDecoder::next_segment` rejecting an overlap
        // at or above the segment length, so this only moves the failure to load
        // time where it reads as a config error rather than an internal one.
        assert!(validate_config(&config_with_overlap(f32::INFINITY)).is_err());

        // Negative infinity was already rejected by the old `overlap < 0.0`,
        // which is true for it. Asserted so the tightened rule is not assumed to
        // have changed this case.
        assert!(validate_config(&config_with_overlap(f32::NEG_INFINITY)).is_err());
    }

    #[test]
    fn test_validate_accepts_ordinary_overlap() {
        // The control: the tightened rule must not start rejecting the values
        // it is meant to allow, including the zero default.
        assert!(validate_config(&config_with_overlap(0.0)).is_ok());
        assert!(validate_config(&config_with_overlap(1.5)).is_ok());
    }

    #[test]
    fn test_validate_zero_batch_size() {
        let mut config = Config::default();
        config.defaults.batch_size = Some(0);
        assert!(validate_config(&config).is_err());
    }

    /// Build a config carrying nothing but the batch size under test.
    fn config_with_batch_size(batch_size: usize) -> Config {
        let mut config = Config::default();
        config.defaults.batch_size = Some(batch_size);
        config
    }

    #[test]
    fn test_validate_rejects_oversized_batch_size() {
        // #312. This is the case the old `== Some(0)` check let through, and
        // the reason it matters is in `parse_batch_size`: the cap exists to
        // prevent GPU memory exhaustion, and config.toml was the route that
        // skipped it.
        let err = validate_config(&config_with_batch_size(100_000)).unwrap_err();

        assert!(
            matches!(err, Error::ConfigValidation { .. }),
            "an oversized batch size is a config error, got {err:?}"
        );
        assert!(
            err.to_string().contains(&format!(
                "batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}"
            )),
            "the message should name the bound, got: {err}"
        );
    }

    #[test]
    fn test_validate_accepts_the_batch_size_boundaries() {
        // The control. Without it the rule above passes just as well when
        // written with an exclusive upper bound, which would refuse the largest
        // batch size the CLI accepts and make the two routes disagree in the
        // other direction.
        assert!(validate_config(&config_with_batch_size(MIN_BATCH_SIZE)).is_ok());
        assert!(validate_config(&config_with_batch_size(MAX_BATCH_SIZE)).is_ok());
    }

    #[test]
    fn test_validate_accepts_an_unset_batch_size() {
        // `None` means "pick a default from the model and provider", which
        // `determine_default_batch_size` does at run time. A range check
        // written over `unwrap_or_default()` would turn that into a rejected
        // zero, so the absent case is pinned separately.
        let mut config = Config::default();
        config.defaults.batch_size = None;
        assert!(validate_config(&config).is_ok());
    }

    /// Build a config carrying nothing but the day of year under test.
    fn config_with_day_of_year(day: u32) -> Config {
        let mut config = Config::default();
        config.defaults.day_of_year = Some(day);
        config
    }

    #[test]
    fn test_validate_rejects_out_of_range_day_of_year() {
        // The field `validate_defaults` did not look at at all (#312). 999 is
        // the value from the issue; it reached the BSG SDM path and was
        // rejected there by birdnet-onnx, far from the config key that set it.
        let err = validate_config(&config_with_day_of_year(999)).unwrap_err();

        assert!(
            matches!(err, Error::ConfigValidation { .. }),
            "an out-of-range day of year is a config error, got {err:?}"
        );
        assert!(
            err.to_string().contains(&format!(
                "day_of_year must be between {} and {}",
                day_of_year::MIN,
                day_of_year::MAX
            )),
            "the message should name the bound, got: {err}"
        );
    }

    #[test]
    fn test_validate_rejects_day_of_year_zero() {
        // The 1-based half. An off-by-one written as `0..=366` accepts this and
        // passes every other test here.
        assert!(validate_config(&config_with_day_of_year(0)).is_err());
    }

    #[test]
    fn test_validate_accepts_the_day_of_year_boundaries() {
        // 366 is deliberately included: the last day of a leap year is legal,
        // and a bound written as 365 would reject a real date.
        assert!(validate_config(&config_with_day_of_year(day_of_year::MIN)).is_ok());
        assert!(validate_config(&config_with_day_of_year(day_of_year::MAX)).is_ok());
        assert!(
            validate_config(&Config::default()).is_ok(),
            "unset is legal"
        );
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
        assert!(matches!(
            result.unwrap_err(),
            Error::InvalidLongitude { .. }
        ));
    }

    #[test]
    fn test_validate_range_filter_valid_coordinates() {
        let mut config = Config::default();
        config.defaults.latitude = Some(40.7128);
        config.defaults.longitude = Some(-74.0060);

        let result = validate_range_filter(&config);
        assert!(result.is_ok());
    }

    /// Build a config carrying nothing but the threshold under test, so a
    /// failure can only be attributed to the threshold.
    fn config_with_threshold(threshold: f32) -> Config {
        let mut config = Config::default();
        config.defaults.range_threshold = threshold;
        config
    }

    #[test]
    fn test_validate_range_filter_rejects_negative_threshold() {
        // Silently disabled range filtering while the JSON envelope still
        // reported it as active.
        let err = validate_range_filter(&config_with_threshold(-1.0)).unwrap_err();

        assert!(
            matches!(err, Error::InvalidRangeThreshold { value } if value == -1.0),
            "expected InvalidRangeThreshold(-1.0), got {err:?}"
        );
    }

    #[test]
    fn test_validate_range_filter_rejects_threshold_above_one() {
        let err = validate_range_filter(&config_with_threshold(1.5)).unwrap_err();

        assert!(
            matches!(err, Error::InvalidRangeThreshold { value } if value == 1.5),
            "expected InvalidRangeThreshold(1.5), got {err:?}"
        );
    }

    #[test]
    fn test_validate_range_filter_rejects_nan_threshold() {
        // The regression guard. NaN dropped every mapped species, because
        // `score >= NaN` is false for every score, and the only symptom was an
        // info log reading "0 species in range". It is caught here because
        // `(0.0..=1.0).contains(&NaN)` is false, not by an explicit is_nan test,
        // so this asserts the idiom keeps working.
        let err = validate_range_filter(&config_with_threshold(f32::NAN)).unwrap_err();

        assert!(
            matches!(err, Error::InvalidRangeThreshold { value } if value.is_nan()),
            "expected InvalidRangeThreshold(NaN), got {err:?}"
        );
    }

    #[test]
    fn test_validate_range_filter_accepts_threshold_boundaries() {
        // The range is inclusive at both ends; pin that so a later refactor to
        // an exclusive comparison is caught.
        assert!(validate_range_filter(&config_with_threshold(0.0)).is_ok());
        assert!(validate_range_filter(&config_with_threshold(1.0)).is_ok());
    }

    #[test]
    fn test_validate_config_rejects_bad_threshold() {
        // Every caller reaches this through `validate_config` rather than
        // calling `validate_range_filter` directly: the load gate in `run()`,
        // and `save_config` on behalf of every writer. So the chain has to hold
        // for `birda config set defaults.range_threshold -1` to be rejected.
        // Without this, the unit tests above could pass while the CLI path
        // stayed broken.
        let err = validate_config(&config_with_threshold(-1.0)).unwrap_err();

        assert!(
            matches!(err, Error::InvalidRangeThreshold { .. }),
            "validate_config must reach validate_range_filter, got {err:?}"
        );
    }
}
