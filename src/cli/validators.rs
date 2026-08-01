//! CLI argument validators.
//!
//! Shared validation functions for CLI argument parsing.

use crate::constants::{MAX_BATCH_SIZE, MIN_BATCH_SIZE, confidence, coordinates, day_of_year};

/// Parse and validate confidence value (0.0-1.0).
pub fn parse_confidence(s: &str) -> Result<f32, String> {
    // Trimmed for the same reason `parse_batch_size` and `parse_overlap` trim:
    // every one of these is reachable through a `BIRDA_*` environment variable,
    // and a value that picked up a space in a shell profile or a Docker env
    // file should not be a hard error. Without this the crate had two spellings
    // of the same rule, so `BIRDA_OVERLAP=" 1.5 "` worked while
    // `BIRDA_MIN_CONFIDENCE=" 0.5 "` failed, from options a user sets side by
    // side. Trimming only widens what is accepted; it cannot change the value a
    // legal input parses to.
    let value: f32 = s
        .trim()
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    // The bounds come from the constants rather than literals so that
    // `clipper::command::validate_float_args`, which re-applies this rule at
    // the library boundary, reads the same two values instead of a second copy
    // of them.
    if !(confidence::MIN..=confidence::MAX).contains(&value) {
        return Err(format!(
            "confidence must be between {:.1} and {:.1}, got {value}",
            confidence::MIN,
            confidence::MAX
        ));
    }

    Ok(value)
}

/// Parse and validate a bounded float value.
///
/// # Arguments
///
/// * `s` - The string to parse
/// * `min` - Minimum allowed value (inclusive)
/// * `max` - Maximum allowed value (inclusive)
/// * `name` - Name of the parameter for error messages
pub fn parse_bounded_float(s: &str, min: f64, max: f64, name: &str) -> Result<f64, String> {
    // Trimmed; see `parse_confidence`. This one backs `--lat`/`BIRDA_LATITUDE`
    // and `--lon`/`BIRDA_LONGITUDE`.
    let value: f64 = s
        .trim()
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    if !(min..=max).contains(&value) {
        return Err(format!(
            "{name} must be between {min} and {max}, got {value}"
        ));
    }

    Ok(value)
}

/// Parse and validate a latitude in degrees.
///
/// Reads `coordinates::LATITUDE_MIN`/`MAX`, the same pair
/// `config::validate::validate_range_filter` and `Error::InvalidLatitude` read.
/// `test_parse_latitude_matches_the_config_file_rule` drives this and the file
/// rule together and compares verdicts (#340).
pub fn parse_latitude(s: &str) -> Result<f64, String> {
    parse_bounded_float(
        s,
        coordinates::LATITUDE_MIN,
        coordinates::LATITUDE_MAX,
        "latitude",
    )
}

/// Parse and validate a longitude in degrees.
///
/// See [`parse_latitude`] for the shared-constant arrangement.
pub fn parse_longitude(s: &str) -> Result<f64, String> {
    parse_bounded_float(
        s,
        coordinates::LONGITUDE_MIN,
        coordinates::LONGITUDE_MAX,
        "longitude",
    )
}

/// Parse and validate segment overlap in seconds (finite and non-negative).
///
/// The rule is deliberately identical to the config-file one in
/// `validate_defaults`, wording included, because the two are the same setting
/// reached by different routes: `--overlap` and `BIRDA_OVERLAP` both win over
/// `defaults.overlap`, so a rule enforced on the file alone left the channels
/// disagreeing. `overlap = -1` in config.toml was a hard error while
/// `BIRDA_OVERLAP=-1` was accepted.
///
/// What each rejected value used to do, since none of them failed loudly:
///
/// - NaN and a negative value were both silently ignored. `overlap *
///   sample_rate` is cast to `usize`, and Rust saturates that cast rather than
///   trapping, so both became 0 and the run produced non-overlapping segments
///   without a word.
/// - Infinity changed the result instead. The same cast sends it to
///   `usize::MAX`, `chunk_samples.saturating_sub(overlap_samples)` gives a step
///   of 0, and the run yields no segments at all: exit 0, empty output.
///
/// No upper bound is imposed, matching the config side. An oversized but finite
/// overlap is rejected by `AudioDecoder::next_segment`, which is the only place
/// that knows the segment length to compare against.
pub fn parse_overlap(s: &str) -> Result<f32, String> {
    let value: f32 = s
        .trim()
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    // `is_finite` is what catches NaN, and it has to be spelled this way round:
    // a bare `value < 0.0` test is false for NaN, which is exactly how NaN got
    // through before.
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "overlap must be a finite non-negative number, got {value}"
        ));
    }

    Ok(value)
}

/// Parse and validate batch size (must be between `MIN_BATCH_SIZE` and
/// `MAX_BATCH_SIZE`).
///
/// The same range is applied to `defaults.batch_size` by
/// `config::validate::validate_defaults`, and
/// `test_parse_batch_size_matches_the_config_file_rule` drives both and
/// compares their verdicts. Before #312 only this side carried the upper
/// bound, so `--batch-size 100000` was refused while the same value in
/// config.toml reached the inference path.
pub fn parse_batch_size(s: &str) -> Result<usize, String> {
    let value: usize = s
        .trim()
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    if value < MIN_BATCH_SIZE {
        return Err(format!(
            "batch_size must be at least {MIN_BATCH_SIZE}, got {value}"
        ));
    }

    if value > MAX_BATCH_SIZE {
        return Err(format!(
            "batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}, got {value}\n\n\
             This limit prevents GPU memory exhaustion.\n\
             If processing fails with batch_size={MAX_BATCH_SIZE}, try reducing it further or use --cpu."
        ));
    }

    Ok(value)
}

/// Parse and validate the day of year used for BSG SDM adjustment (1-366).
///
/// Shared with the config-file rule in `validate_defaults` for the reason
/// `parse_overlap` documents: `--day-of-year` and `BIRDA_DAY_OF_YEAR` both win
/// over `defaults.day_of_year`, so a bound enforced on one route left the
/// routes disagreeing about one setting.
///
/// This replaces an inline `clap::value_parser!(u32).range(1..=366)` on the
/// argument. That bounded the flag, but the bound was a literal only clap could
/// read, so neither `handle_config_set` nor `validate_defaults` could apply it.
pub fn parse_day_of_year(s: &str) -> Result<u32, String> {
    // Trimmed; see `parse_confidence`. This one backs `--day-of-year` and
    // `BIRDA_DAY_OF_YEAR`.
    //
    // Parsed as `i64` and then range-checked, rather than straight to `u32`,
    // because that is what the clap parser this replaced did: `value_parser!(u32)`
    // resolves to `RangedI64ValueParser<u32>`, which parses to `i64` first
    // (clap_builder 4.6.2, src/builder/value_parser.rs:2362 and :1418, read this
    // session). A negative therefore reached its bounds check and was reported as
    // out of range.
    //
    // Parsing straight to `u32` looks equivalent and is not: `-1` fails at the
    // parse step and gets reported as "not a valid number", which is false on its
    // face. That was measured against this branch before the `i64` step was put
    // back, and `--week` still shows the original behaviour for comparison.
    let value: i64 = s
        .trim()
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    let range = i64::from(day_of_year::MIN)..=i64::from(day_of_year::MAX);
    if !range.contains(&value) {
        return Err(format!(
            "day_of_year must be between {} and {}, got {value}",
            day_of_year::MIN,
            day_of_year::MAX
        ));
    }

    // Infallible: the range check above bounds `value` to 1..=366, which fits.
    u32::try_from(value).map_err(|_| format!("'{s}' is not a valid number"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_confidence_valid() {
        assert_eq!(parse_confidence("0.5").ok(), Some(0.5));
        assert_eq!(parse_confidence("0.0").ok(), Some(0.0));
        assert_eq!(parse_confidence("1.0").ok(), Some(1.0));
    }

    #[test]
    fn test_parse_confidence_invalid() {
        assert!(parse_confidence("1.1").is_err());
        assert!(parse_confidence("-0.1").is_err());
        assert!(parse_confidence("abc").is_err());
    }

    #[test]
    fn test_parse_bounded_float_valid() {
        assert_eq!(
            parse_bounded_float("50.0", -100.0, 100.0, "test").ok(),
            Some(50.0)
        );
        assert_eq!(
            parse_bounded_float("-100.0", -100.0, 100.0, "test").ok(),
            Some(-100.0)
        );
        assert_eq!(
            parse_bounded_float("100.0", -100.0, 100.0, "test").ok(),
            Some(100.0)
        );
    }

    #[test]
    fn test_parse_bounded_float_invalid_range() {
        let err = parse_bounded_float("101.0", -100.0, 100.0, "test");
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("test must be between"));
    }

    #[test]
    fn test_parse_bounded_float_invalid_number() {
        let err = parse_bounded_float("abc", -100.0, 100.0, "test");
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("not a valid number"));
    }

    #[test]
    fn test_parse_overlap_valid() {
        assert_eq!(parse_overlap("0").ok(), Some(0.0));
        assert_eq!(parse_overlap("1.5").ok(), Some(1.5));
        // No upper bound here on purpose; an oversized but finite overlap is
        // caught against the segment length in `AudioDecoder::next_segment`.
        assert_eq!(parse_overlap("1e15").ok(), Some(1e15));
    }

    #[test]
    fn test_parse_overlap_rejects_negative() {
        let err = parse_overlap("-1").unwrap_err();
        assert!(err.contains("finite non-negative"), "got: {err}");
        assert!(err.contains("-1"), "the error should name the value: {err}");
    }

    #[test]
    fn test_parse_overlap_rejects_nan() {
        // The case a hand-rolled `value < 0.0` test lets through, since every
        // NaN comparison is false. It was silently coerced to zero overlap.
        //
        // The message is asserted, not just `is_err()`. `f32::from_str` accepts
        // all of these, so a bare `is_err()` would also pass if they fell into
        // the "not a valid number" branch instead, and which branch catches
        // them is exactly the claim the shared wording rests on.
        for input in ["nan", "NaN", "-nan"] {
            let err = parse_overlap(input).unwrap_err();
            assert!(
                err.contains("finite non-negative"),
                "'{input}' must be rejected as non-finite, got: {err}"
            );
        }
    }

    #[test]
    fn test_parse_overlap_rejects_infinity() {
        // `f32::from_str` accepts all four spellings, and each one used to make
        // the run produce no segments at all. `1e40` is the interesting one: it
        // is written as a finite literal and overflows to infinity on the way
        // into an f32.
        for input in ["inf", "infinity", "-inf", "1e40"] {
            let err = parse_overlap(input).unwrap_err();
            assert!(
                err.contains("finite non-negative"),
                "'{input}' must be rejected as non-finite, got: {err}"
            );
        }
    }

    #[test]
    fn test_parse_overlap_rejects_non_numbers() {
        let err = parse_overlap("abc").unwrap_err();
        assert!(err.contains("not a valid number"), "got: {err}");
    }

    #[test]
    fn test_env_backed_validators_all_tolerate_whitespace() {
        // Environment variables pick up stray whitespace easily, and every one
        // of these is reachable through a `BIRDA_*` variable. Asserted over the
        // whole set rather than for overlap alone, because the failure this
        // pins is the crate holding two spellings of one rule: for a while
        // `BIRDA_OVERLAP=" 1.5 "` was accepted and `BIRDA_LATITUDE=" 60.1 "`
        // was not.
        //
        // Deliberately not stated as a count. This test said "all four" while
        // asserting five, and #312 then added a sixth (`parse_day_of_year`,
        // reached through `BIRDA_DAY_OF_YEAR`) without the number moving. Every
        // env-backed parser belongs in this list; the way to check is
        // `grep -rhoE 'env = "BIRDA_[A-Z_]+"' src/` against the ones that carry
        // a `value_parser`.
        assert_eq!(parse_overlap(" 1.5 ").ok(), Some(1.5));
        assert_eq!(parse_confidence(" 0.5 ").ok(), Some(0.5));
        assert_eq!(parse_latitude(" 60.17 ").ok(), Some(60.17));
        assert_eq!(parse_longitude(" 24.94 ").ok(), Some(24.94));
        assert_eq!(parse_batch_size(" 32 ").ok(), Some(32));
        assert_eq!(parse_day_of_year(" 200 ").ok(), Some(200));
    }

    #[test]
    fn test_parse_overlap_matches_the_config_file_rule() {
        // The whole point of #306: the flag, the environment variable and
        // config.toml must agree.
        //
        // This drives BOTH rules and compares their verdicts, rather than
        // asserting `parse_overlap` against a list of inputs a comment claims
        // `validate_defaults` agrees on. The list version was the shape this
        // test had first, and it did not work: giving `validate_defaults` an
        // upper bound the CLI does not have left every config test green, so
        // the guard the doc comments on both sides advertise was not a guard at
        // all. Comparing verdicts means the two cannot diverge on any input
        // here without this failing, whichever side moves.
        for input in [
            "0", "0.0", "1.5", "1e15", // accepted by both
            "-1", "nan", "inf", "-inf", "1e40", // rejected by both
        ] {
            let cli = parse_overlap(input);

            let mut config = crate::config::Config::default();
            config.defaults.overlap = input.trim().parse().expect("input must parse as f32");
            let file = crate::config::validate_config(&config);

            assert_eq!(
                cli.is_ok(),
                file.is_ok(),
                "'{input}': the flag says {}, config.toml says {}. The two rules must agree.",
                if cli.is_ok() { "ok" } else { "rejected" },
                if file.is_ok() { "ok" } else { "rejected" },
            );
        }
    }

    #[test]
    fn test_parse_batch_size_valid() {
        assert_eq!(parse_batch_size("1").ok(), Some(MIN_BATCH_SIZE));
        assert_eq!(parse_batch_size("8").ok(), Some(8));
        assert_eq!(parse_batch_size("128").ok(), Some(128));
    }

    #[test]
    fn test_parse_batch_size_invalid() {
        assert!(parse_batch_size("0").is_err());
        assert!(parse_batch_size("-1").is_err());
        assert!(parse_batch_size("abc").is_err());
    }

    #[test]
    fn test_parse_batch_size_at_maximum() {
        assert_eq!(parse_batch_size("512").ok(), Some(MAX_BATCH_SIZE));
    }

    #[test]
    fn test_parse_batch_size_above_maximum() {
        let result = parse_batch_size("513");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains(&format!(
            "batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}"
        )));
        assert!(err.contains("GPU memory exhaustion"));
    }

    #[test]
    fn test_parse_batch_size_way_above_maximum() {
        let result = parse_batch_size("2560");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains(&format!(
            "batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}"
        )));
        assert!(err.contains("GPU memory exhaustion"));
    }

    #[test]
    fn test_parse_batch_size_with_whitespace() {
        // Test leading/trailing whitespace (common in config files)
        assert_eq!(parse_batch_size(" 32").ok(), Some(32));
        assert_eq!(parse_batch_size("32 ").ok(), Some(32));
        assert_eq!(parse_batch_size(" 32 ").ok(), Some(32));
        assert_eq!(parse_batch_size("  64  ").ok(), Some(64));
    }

    #[test]
    fn test_parse_batch_size_matches_the_config_file_rule() {
        // #312, and deliberately the same shape as
        // `test_parse_overlap_matches_the_config_file_rule` above.
        //
        // Driving both rules and comparing verdicts is what makes this catch
        // the defect. A list-based test written from the CLI's point of view
        // ("512 ok, 513 rejected") passes against `parse_batch_size` alone and
        // says nothing about the config file, which is precisely where the
        // upper bound was missing: `--batch-size 100000` was refused while
        // `batch_size = 100000` in config.toml reached the inference path.
        // The boundaries are derived from the constants rather than spelled
        // out. Agreement would still be checked either way, since both rules
        // read the same constants, but with "512"/"513" written in, raising
        // `MAX_BATCH_SIZE` would leave the inputs no longer straddling the
        // bound, so the test would keep passing while covering less.
        let below = (MIN_BATCH_SIZE - 1).to_string();
        let at_max = MAX_BATCH_SIZE.to_string();
        let above = (MAX_BATCH_SIZE + 1).to_string();
        for input in [
            "1",
            "8",
            at_max.as_str(), // accepted by both
            below.as_str(),
            above.as_str(),
            "100000", // rejected by both
        ] {
            let cli = parse_batch_size(input);

            let mut config = crate::config::Config::default();
            config.defaults.batch_size =
                Some(input.trim().parse().expect("input must parse as usize"));
            let file = crate::config::validate_config(&config);

            assert_eq!(
                cli.is_ok(),
                file.is_ok(),
                "'{input}': the flag says {}, config.toml says {}. The two rules must agree.",
                if cli.is_ok() { "ok" } else { "rejected" },
                if file.is_ok() { "ok" } else { "rejected" },
            );
        }
    }

    #[test]
    fn test_parse_latitude_matches_the_config_file_rule() {
        // #340, the same shape as `test_parse_batch_size_matches_the_config_file_rule`
        // above. Three routes reach `defaults.latitude`: the flag, `config set`
        // (which routes through this parser), and a hand-edited config.toml.
        // Before this the bound was written out in `cli::validators`, in
        // `config::validate::validate_range_filter` and a third time in the
        // `Error::InvalidLatitude` message, with nothing keeping the three
        // equal. They agreed, so there was no live defect; this is what keeps
        // that true.
        //
        // The inputs are derived from the constants rather than spelled out, so
        // that moving a bound leaves them still straddling it.
        let below = (coordinates::LATITUDE_MIN - 1.0).to_string();
        let above = (coordinates::LATITUDE_MAX + 1.0).to_string();
        for input in [
            "0",
            "60.1699",
            &coordinates::LATITUDE_MIN.to_string(),
            &coordinates::LATITUDE_MAX.to_string(),
            &below,
            &above,
            "1000",
        ] {
            let cli = parse_latitude(input);

            let mut config = crate::config::Config::default();
            config.defaults.latitude = Some(input.trim().parse().expect("input must parse as f64"));
            let file = crate::config::validate_config(&config);

            assert_eq!(
                cli.is_ok(),
                file.is_ok(),
                "'{input}': the flag says {}, config.toml says {}. The two rules must agree.",
                if cli.is_ok() { "ok" } else { "rejected" },
                if file.is_ok() { "ok" } else { "rejected" },
            );
        }
    }

    #[test]
    fn test_parse_longitude_matches_the_config_file_rule() {
        // See `test_parse_latitude_matches_the_config_file_rule`. Kept separate
        // rather than folded into a loop over both, so a failure names which of
        // the two bounds drifted.
        let below = (coordinates::LONGITUDE_MIN - 1.0).to_string();
        let above = (coordinates::LONGITUDE_MAX + 1.0).to_string();
        for input in [
            "0",
            "24.9384",
            &coordinates::LONGITUDE_MIN.to_string(),
            &coordinates::LONGITUDE_MAX.to_string(),
            &below,
            &above,
            "1000",
        ] {
            let cli = parse_longitude(input);

            let mut config = crate::config::Config::default();
            config.defaults.longitude =
                Some(input.trim().parse().expect("input must parse as f64"));
            let file = crate::config::validate_config(&config);

            assert_eq!(
                cli.is_ok(),
                file.is_ok(),
                "'{input}': the flag says {}, config.toml says {}. The two rules must agree.",
                if cli.is_ok() { "ok" } else { "rejected" },
                if file.is_ok() { "ok" } else { "rejected" },
            );
        }
    }

    #[test]
    fn test_coordinate_error_messages_carry_the_enforced_bounds() {
        // The third copy of the numbers, and the one no parity test can reach:
        // `Error::InvalidLatitude` renders the bounds into text a user reads
        // when their value is refused. It interpolates the constants, so this
        // fails if the message is ever written back out as a literal that
        // disagrees with the rule that produced it.
        let latitude = crate::error::Error::InvalidLatitude { value: 91.0 }.to_string();
        assert!(
            latitude.contains(&format!("{:.1}", coordinates::LATITUDE_MIN))
                && latitude.contains(&format!("{:.1}", coordinates::LATITUDE_MAX)),
            "latitude message must state the enforced bounds, got '{latitude}'"
        );

        let longitude = crate::error::Error::InvalidLongitude { value: 181.0 }.to_string();
        assert!(
            longitude.contains(&format!("{:.1}", coordinates::LONGITUDE_MIN))
                && longitude.contains(&format!("{:.1}", coordinates::LONGITUDE_MAX)),
            "longitude message must state the enforced bounds, got '{longitude}'"
        );
    }

    #[test]
    fn test_parse_day_of_year_valid() {
        assert_eq!(parse_day_of_year("1").ok(), Some(day_of_year::MIN));
        assert_eq!(parse_day_of_year("200").ok(), Some(200));
        assert_eq!(parse_day_of_year("366").ok(), Some(day_of_year::MAX));
        // Trimmed like every sibling, because `BIRDA_DAY_OF_YEAR` can pick up a
        // space from a shell profile or a Docker env file.
        assert_eq!(parse_day_of_year(" 200 ").ok(), Some(200));
    }

    #[test]
    fn test_parse_day_of_year_invalid() {
        assert!(parse_day_of_year("0").is_err(), "the range is 1-based");
        assert!(parse_day_of_year("367").is_err());
        assert!(parse_day_of_year("999").is_err(), "the value from #312");
        assert!(parse_day_of_year("abc").is_err());

        // A negative is reported as OUT OF RANGE, not as a malformed number,
        // because `-1` plainly is a number. Asserted rather than left to
        // `is_err()`: parsing straight to `u32` also rejects it, but with
        // "'-1' is not a valid number", and that is what this branch shipped
        // until the gate caught it.
        let negative = parse_day_of_year("-1").unwrap_err();
        assert!(
            negative.contains("must be between") && negative.contains("got -1"),
            "a negative day should be reported as out of range, got: {negative}"
        );

        let err = parse_day_of_year("367").unwrap_err();
        assert!(
            err.contains(&format!(
                "day_of_year must be between {} and {}",
                day_of_year::MIN,
                day_of_year::MAX
            )),
            "the message should name the bound it enforces, got: {err}"
        );
    }

    #[test]
    fn test_parse_day_of_year_matches_the_config_file_rule() {
        // The other half of #312. `--day-of-year` carried an inline
        // `range(1..=366)`, `validate_defaults` did not look at the field at
        // all, and `config set` had no arm for the key, so config.toml was the
        // only route that could set `defaults.day_of_year` and the only one
        // with no check. The flag and `BIRDA_DAY_OF_YEAR` could set the value
        // for a single run, and both were bounded.
        //
        // Inputs are restricted to ones that parse as `u32`, since the config
        // side is already typed and cannot be handed "abc" or "-1"; those are
        // covered against the parser in `test_parse_day_of_year_invalid`.
        let at_max = day_of_year::MAX.to_string();
        let above = (day_of_year::MAX + 1).to_string();
        for input in ["1", "200", at_max.as_str(), "0", above.as_str(), "100000"] {
            let cli = parse_day_of_year(input);

            let mut config = crate::config::Config::default();
            config.defaults.day_of_year =
                Some(input.trim().parse().expect("input must parse as u32"));
            let file = crate::config::validate_config(&config);

            assert_eq!(
                cli.is_ok(),
                file.is_ok(),
                "'{input}': the flag says {}, config.toml says {}. The two rules must agree.",
                if cli.is_ok() { "ok" } else { "rejected" },
                if file.is_ok() { "ok" } else { "rejected" },
            );
        }
    }
}
