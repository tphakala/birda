//! CLI argument validators.
//!
//! Shared validation functions for CLI argument parsing.

use crate::constants::MAX_BATCH_SIZE;

/// Parse and validate confidence value (0.0-1.0).
pub fn parse_confidence(s: &str) -> Result<f32, String> {
    let value: f32 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    if !(0.0..=1.0).contains(&value) {
        return Err(format!(
            "confidence must be between 0.0 and 1.0, got {value}"
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
    let value: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    if !(min..=max).contains(&value) {
        return Err(format!(
            "{name} must be between {min} and {max}, got {value}"
        ));
    }

    Ok(value)
}

/// Parse and validate latitude value (-90.0 to 90.0).
pub fn parse_latitude(s: &str) -> Result<f64, String> {
    parse_bounded_float(s, -90.0, 90.0, "latitude")
}

/// Parse and validate longitude value (-180.0 to 180.0).
pub fn parse_longitude(s: &str) -> Result<f64, String> {
    parse_bounded_float(s, -180.0, 180.0, "longitude")
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

/// Parse and validate batch size (must be between 1 and `MAX_BATCH_SIZE`).
pub fn parse_batch_size(s: &str) -> Result<usize, String> {
    let value: usize = s
        .trim()
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    if value < 1 {
        return Err(format!("batch_size must be at least 1, got {value}"));
    }

    if value > MAX_BATCH_SIZE {
        return Err(format!(
            "batch_size must be between 1 and {MAX_BATCH_SIZE}, got {value}\n\n\
             This limit prevents GPU memory exhaustion.\n\
             If processing fails with batch_size={MAX_BATCH_SIZE}, try reducing it further or use --cpu."
        ));
    }

    Ok(value)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
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
        assert!(parse_overlap("nan").is_err());
        assert!(parse_overlap("NaN").is_err());
        assert!(parse_overlap("-nan").is_err());
    }

    #[test]
    fn test_parse_overlap_rejects_infinity() {
        // `f32::from_str` accepts all four spellings, and each one used to make
        // the run produce no segments at all.
        assert!(parse_overlap("inf").is_err());
        assert!(parse_overlap("infinity").is_err());
        assert!(parse_overlap("-inf").is_err());
        assert!(parse_overlap("1e40").is_err());
    }

    #[test]
    fn test_parse_overlap_rejects_non_numbers() {
        let err = parse_overlap("abc").unwrap_err();
        assert!(err.contains("not a valid number"), "got: {err}");
    }

    #[test]
    fn test_parse_overlap_with_whitespace() {
        // Environment variables pick up stray whitespace easily, and
        // `BIRDA_OVERLAP` is one of the two channels this parser exists for.
        assert_eq!(parse_overlap(" 1.5 ").ok(), Some(1.5));
    }

    #[test]
    fn test_parse_overlap_matches_the_config_file_rule() {
        // The whole point of #306: the flag, the environment variable and
        // config.toml must agree. This pins the CLI half against the same
        // inputs `validate_defaults` rules on, so the two cannot drift apart
        // without a test failing.
        for accepted in ["0", "0.0", "1.5"] {
            assert!(
                parse_overlap(accepted).is_ok(),
                "{accepted} is accepted by validate_defaults and must be accepted here"
            );
        }
        for rejected in ["-1", "nan", "inf"] {
            assert!(
                parse_overlap(rejected).is_err(),
                "{rejected} is rejected by validate_defaults and must be rejected here"
            );
        }
    }

    #[test]
    fn test_parse_batch_size_valid() {
        assert_eq!(parse_batch_size("1").ok(), Some(1));
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
            "batch_size must be between 1 and {MAX_BATCH_SIZE}"
        )));
        assert!(err.contains("GPU memory exhaustion"));
    }

    #[test]
    fn test_parse_batch_size_way_above_maximum() {
        let result = parse_batch_size("2560");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains(&format!(
            "batch_size must be between 1 and {MAX_BATCH_SIZE}"
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
}
