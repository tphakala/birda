//! Configuration validation.

use crate::config::{Config, ModelConfig};
use crate::constants::{
    MAX_BATCH_SIZE, MIN_BATCH_SIZE, confidence, coordinates, csv_columns, day_of_year,
};
use crate::error::{Error, Result};

/// Validate the entire configuration.
///
/// Fail-fast: returns the first rule failure, exactly as the individual rules
/// below produce it, so every writer that persists a config through
/// [`crate::config::save_config`] still refuses to write an invalid one. For
/// every failure at once (and the one writer allowed to persist a config that
/// is invalid but no worse than it already was, #305) see [`collect_violations`].
pub fn validate_config(config: &Config) -> Result<()> {
    collect_violations(config)
        .into_iter()
        .next()
        .map_or(Ok(()), |violation| Err(violation.error))
}

/// A single configuration rule failure.
///
/// [`collect_violations`] returns every one at once rather than stopping at the
/// first. That is what lets `config set` tell "this edit left the existing
/// faults alone" apart from "this edit introduced a new one" (#305), and what
/// lets a diagnostic list them all instead of one per run.
pub struct Violation {
    /// Which configuration field the failure concerns.
    pub field: ViolationField,
    /// The exact error the fail-fast path raises for this rule, so a caller can
    /// surface the message the user has always seen.
    pub error: Error,
}

/// The configuration field a [`Violation`] concerns.
///
/// Identity is by field, not by value: two different out-of-range latitudes are
/// the same violation here. That is the granularity `config set` compares on
/// when it decides whether an edit added a fault to a field that was previously
/// clean. A value-level identity is neither needed (the `parse_*` validators
/// reject a malformed new value before the write) nor available (`Error` embeds
/// `std::io::Error`, which is not `PartialEq`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ViolationField {
    /// `defaults.min_confidence`.
    MinConfidence,
    /// `defaults.overlap`.
    Overlap,
    /// `defaults.batch_size`.
    BatchSize,
    /// `defaults.day_of_year`.
    DayOfYear,
    /// `defaults.formats`.
    Formats,
    /// `defaults.csv_columns.include`.
    CsvColumns,
    /// `defaults.model`.
    Model,
    /// `defaults.latitude`.
    Latitude,
    /// `defaults.longitude`.
    Longitude,
    /// `defaults.range_threshold`.
    RangeThreshold,
}

/// Collect every configuration rule failure rather than stopping at the first.
///
/// The rules are the ones [`validate_config`] enforces, in the same order, so
/// the first element (if any) is exactly what the fail-fast path returns.
pub fn collect_violations(config: &Config) -> Vec<Violation> {
    let mut violations = Vec::new();
    collect_defaults_violations(config, &mut violations);
    collect_range_filter_violations(config, &mut violations);
    violations
}

/// Recovery advice for the two keys `birda config set` cannot reach.
///
/// Every other rule in `collect_defaults_violations` guards a key with a `config set`
/// arm, so the tool can repair it and the message can stay short. `formats` and
/// `csv_columns` have no arm, and a bad value in either is refused by
/// `save_config` as well, so `config set` on any OTHER key is refused too until
/// the file is fixed by hand. That makes this text the only pointer the user
/// gets.
const EDIT_THE_FILE: &str = "This key has no 'birda config set' arm, so edit config.toml directly; \
     'birda config path' prints where it is.";

/// Collect `[defaults]` rule failures into `violations`. See [`collect_violations`].
fn collect_defaults_violations(config: &Config, violations: &mut Vec<Violation>) {
    let defaults = &config.defaults;

    // Validate min_confidence range
    if !(confidence::MIN..=confidence::MAX).contains(&defaults.min_confidence) {
        violations.push(Violation {
            field: ViolationField::MinConfidence,
            error: Error::ConfigValidation {
                message: format!(
                    "min_confidence must be between {} and {}, got {}",
                    confidence::MIN,
                    confidence::MAX,
                    defaults.min_confidence
                ),
            },
        });
    }

    // Validate overlap is a finite, non-negative number.
    //
    // The `is_finite` half is what catches NaN, and NaN is the case that
    // matters: a bare `overlap < 0.0` is the hand-rolled comparison
    // `collect_range_filter_violations` warns against below, and `NaN < 0.0` is false, so
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
        violations.push(Violation {
            field: ViolationField::Overlap,
            error: Error::ConfigValidation {
                message: format!(
                    "overlap must be a finite non-negative number, got {}",
                    defaults.overlap
                ),
            },
        });
    }

    // The upper bound is the half that was missing (#312). `parse_batch_size`
    // has rejected anything above `MAX_BATCH_SIZE` since #135 added the cap and
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
    // advice `parse_batch_size` adds. That is what keeps the two layers
    // distinguishable, so `test_config_set_rejects_an_oversized_batch_size` can
    // assert on the advice and fail if its arm falls through to this rule
    // instead. The load-path test is pinned differently, by passing no flag at
    // all, since this message is the only one reachable that way.
    //
    // The lower bound is the one place the two texts diverge without being
    // designed to: `parse_batch_size` says "must be at least 1" while this says
    // "must be between 1 and 512". Same verdict, different sentence, which is
    // why `BATCH_SIZE_VIOLATION` in the integration suite matches the flag only
    // at the upper bound.
    if let Some(batch_size) = defaults.batch_size
        && !(MIN_BATCH_SIZE..=MAX_BATCH_SIZE).contains(&batch_size)
    {
        violations.push(Violation {
            field: ViolationField::BatchSize,
            error: Error::ConfigValidation {
                message: format!(
                    "batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}, got {batch_size}"
                ),
            },
        });
    }

    // Same shape one field over. `--day-of-year` and `BIRDA_DAY_OF_YEAR` are
    // bounded by `parse_day_of_year`; the file was not, and `handle_config_set`
    // had no arm for the key at all, so hand-editing config.toml was the only
    // way to set `defaults.day_of_year` and the only route with no check. (The
    // flag and the environment variable set the value for one run, and both
    // were bounded.) `day_of_year = 999` then reached the BSG SDM path in
    // `run()`, which rejects it from birdnet-onnx, far from the value that
    // caused it.
    if let Some(day) = defaults.day_of_year
        && !(day_of_year::MIN..=day_of_year::MAX).contains(&day)
    {
        violations.push(Violation {
            field: ViolationField::DayOfYear,
            error: Error::ConfigValidation {
                message: format!(
                    "day_of_year must be between {} and {}, got {day}",
                    day_of_year::MIN,
                    day_of_year::MAX
                ),
            },
        });
    }

    // An empty `formats` made an analysis run skip every file it was given,
    // report success and write nothing (#339). `should_process` asks whether the
    // outputs already exist with `formats.iter().all(..)`, and `all` over an
    // empty slice is vacuously true, so "nothing to check" was answered as
    // "everything is already there": each file logged "Skipping (output
    // exists)", naming a reason that was not the real one, and the run exited 0.
    //
    // config.toml is the only route that can carry an EMPTY list, which is the
    // #312 and #306 shape once more. `--format` and `BIRDA_FORMAT` do reach
    // this setting and win over the file, but they are a `Vec<OutputFormat>`
    // of `ValueEnum`s, so clap refuses an empty or unknown value rather than
    // producing an empty list; `handle_config_set` has no `defaults.formats`
    // arm at all. Unlike those two predecessors the failure was silent rather
    // than loud, which is why it cost a whole run's output rather than an
    // error message.
    //
    // `DefaultsConfig::default()` sets `formats = ["csv"]`, so an empty list is
    // never what an untouched config carries and rejecting it cannot break one.
    // `should_process` no longer answers `SkipExists` for an empty slice either.
    // The two guards are independent on purpose: this one keeps the value out of
    // the config, and that one keeps the vacuous branch closed for a library
    // caller passing the slice directly.
    // The message describes the value, not the old symptom. "skips every input
    // file" was true before the coordinator guard went in alongside this rule,
    // and stating it here would have been this commit contradicting itself in
    // text the user reads.
    //
    // The rule is unconditional, like every sibling above, even though
    // `--stdout` never reads `formats` and an empty list is inert there. Making
    // it mode-aware would mean handing the args to a validator that is
    // deliberately blind to them, to preserve a setting whose only other mode
    // it silently breaks; and the repair costs a `--stdout` user nothing, since
    // naming a format writes no file in that mode either.
    // Both new messages name the full key and point at the file, which the
    // rules above deliberately do not. Those keys all have a `config set` arm,
    // so the tool can repair them and the message can stay short; these two
    // have none, so a hand edit is the only way out and the message is the only
    // place to say so.
    if defaults.formats.is_empty() {
        violations.push(Violation {
            field: ViolationField::Formats,
            error: Error::ConfigValidation {
                message: format!(
                    "defaults.formats must list at least one output format; with an empty list a \
                     run writes no output at all\n\n{EDIT_THE_FILE}"
                ),
            },
        });
    }

    // Same no-CLI-route, no-validation shape one field over, and the two writers
    // that read the list disagree about an unrecognised name: `CsvWriter` emits
    // it as a header over a column empty in every row, `parquet::build_schema`
    // drops it. `handle_config_set` has no `defaults.csv_columns` arm either, so
    // a hand-edited typo was the only way in and produced a phantom column in
    // one format and nothing at all in the other.
    //
    // This rejects a config that was previously accepted, which is the point:
    // the alternative to failing here is the phantom column, and a typo is the
    // only way to reach it.
    if let Some(unknown) = defaults
        .csv_columns
        .include
        .iter()
        .find(|column| !csv_columns::RECOGNISED.contains(&column.as_str()))
    {
        violations.push(Violation {
            field: ViolationField::CsvColumns,
            error: Error::ConfigValidation {
                message: format!(
                    "defaults.csv_columns.include has an unknown column '{unknown}'; the accepted \
                     columns are {}\n\n{EDIT_THE_FILE}",
                    csv_columns::RECOGNISED.join(", ")
                ),
            },
        });
    }

    // Validate default model exists if specified
    if let Some(ref model_name) = defaults.model
        && !config.models.contains_key(model_name)
    {
        violations.push(Violation {
            field: ViolationField::Model,
            error: Error::ModelNotFound {
                name: model_name.clone(),
            },
        });
    }
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

/// Fail-fast view of just the range-filter rules, for the unit tests that
/// exercise them in isolation. Not needed outside tests: production always goes
/// through [`validate_config`], which covers every rule.
#[cfg(test)]
fn validate_range_filter(config: &Config) -> Result<()> {
    let mut violations = Vec::new();
    collect_range_filter_violations(config, &mut violations);
    violations
        .into_iter()
        .next()
        .map_or(Ok(()), |violation| Err(violation.error))
}

/// Collect range filter rule failures into `violations`. See [`collect_violations`].
fn collect_range_filter_violations(config: &Config, violations: &mut Vec<Violation>) {
    // Both bounds read `constants::coordinates`, which `cli::validators` and the
    // `Error` message text also read (#340). Those three were the copies of the
    // numbers; the routes to the setting are a different set, being the flag
    // and its environment variable, `config set`, and a hand-edited file.
    // `config set` never held a copy, since it calls `parse_latitude`.
    if let Some(lat) = config.defaults.latitude
        && !(coordinates::LATITUDE_MIN..=coordinates::LATITUDE_MAX).contains(&lat)
    {
        violations.push(Violation {
            field: ViolationField::Latitude,
            error: Error::InvalidLatitude { value: lat },
        });
    }

    if let Some(lon) = config.defaults.longitude
        && !(coordinates::LONGITUDE_MIN..=coordinates::LONGITUDE_MAX).contains(&lon)
    {
        violations.push(Violation {
            field: ViolationField::Longitude,
            error: Error::InvalidLongitude { value: lon },
        });
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
        violations.push(Violation {
            field: ViolationField::RangeThreshold,
            error: Error::InvalidRangeThreshold { value: threshold },
        });
    }

    // The geomodel is resolved and acquired at analysis time rather than
    // validated here: it is a shared asset that birda can offer to download,
    // so a missing file is not a configuration error.
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
        // The field `collect_defaults_violations` did not look at at all (#312). 999 is
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

    /// Build a config carrying nothing but the format list under test.
    fn config_with_formats(formats: Vec<crate::config::OutputFormat>) -> Config {
        let mut config = Config::default();
        config.defaults.formats = formats;
        config
    }

    #[test]
    fn test_validate_rejects_an_empty_format_list() {
        // #339. Reachable only by hand-editing config.toml, and the reason it
        // cost a whole run rather than an error message: `should_process` asks
        // `formats.iter().all(..)`, which is vacuously true over an empty
        // slice, so every input file was reported as already having its output
        // and the run exited 0 having written nothing.
        let err = validate_config(&config_with_formats(vec![])).unwrap_err();

        assert!(
            matches!(&err, Error::ConfigValidation { message } if message.contains("at least one")),
            "expected the empty-formats rule, got {err:?}"
        );
    }

    #[test]
    fn test_validate_accepts_the_default_format_list() {
        // The control. `DefaultsConfig::default()` is `formats = ["csv"]`, so
        // the new rule must not reject a config nobody has touched.
        assert!(validate_config(&Config::default()).is_ok());
        assert!(
            validate_config(&config_with_formats(vec![
                crate::config::OutputFormat::Json
            ]))
            .is_ok()
        );
    }

    /// Build a config carrying nothing but the CSV column list under test.
    fn config_with_csv_columns(columns: &[&str]) -> Config {
        let mut config = Config::default();
        config.defaults.csv_columns.include = columns.iter().map(|c| (*c).to_string()).collect();
        config
    }

    #[test]
    fn test_validate_rejects_an_unknown_csv_column() {
        // Accepted before #339's fix, and then handled two different ways: the
        // CSV writer emitted 'lattitude' as a header over a column empty in
        // every row, the Parquet writer dropped it. Neither told the user.
        let err = validate_config(&config_with_csv_columns(&["lat", "lattitude"])).unwrap_err();

        assert!(
            matches!(&err, Error::ConfigValidation { message }
                if message.contains("lattitude") && message.contains("species_list")),
            "the message must name the bad column and list the valid ones, got {err:?}"
        );
    }

    #[test]
    fn test_validate_accepts_every_recognised_csv_column() {
        // Drives the whole of `csv_columns::RECOGNISED` through the rule that
        // reads it, so this rule cannot start refusing a name the writers
        // handle. The converse, a name accepted here that a writer does not
        // handle, is pinned separately and per writer, by
        // `test_every_recognised_column_is_written` and
        // `test_every_recognised_column_reaches_the_parquet_writer`.
        assert!(validate_config(&config_with_csv_columns(&csv_columns::RECOGNISED)).is_ok());
        assert!(validate_config(&config_with_csv_columns(&[])).is_ok());
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
