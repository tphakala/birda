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

/// Parse and validate batch size (must be between 1 and `MAX_BATCH_SIZE`).
pub fn parse_batch_size(s: &str) -> Result<usize, String> {
    let value: usize = s
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
        assert_eq!(parse_batch_size("512").ok(), Some(512));
    }

    #[test]
    fn test_parse_batch_size_above_maximum() {
        let result = parse_batch_size("513");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("batch_size must be between 1 and 512"));
        assert!(err.contains("GPU memory exhaustion"));
    }

    #[test]
    fn test_parse_batch_size_way_above_maximum() {
        let result = parse_batch_size("2560");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("batch_size must be between 1 and 512"));
        assert!(err.contains("GPU memory exhaustion"));
    }
}
