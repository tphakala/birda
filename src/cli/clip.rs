//! CLI for clip extraction subcommand.

use std::path::PathBuf;

use clap::Args;

use super::validators::parse_confidence;
use crate::constants::clipper::{
    DEFAULT_OUTPUT_DIR, DEFAULT_POST_PADDING, DEFAULT_PRE_PADDING, MAX_PADDING,
};

/// Arguments for the clip subcommand.
#[derive(Debug, Args)]
pub struct ClipArgs {
    /// Detection result files to process (CSV format).
    /// Mutually exclusive with --start/--end for direct extraction mode.
    #[arg(conflicts_with_all = ["start", "end"])]
    pub files: Vec<PathBuf>,

    /// Output directory for extracted clips.
    #[arg(short, long, default_value = DEFAULT_OUTPUT_DIR)]
    pub output: PathBuf,

    /// Minimum confidence threshold (0.0-1.0).
    #[arg(short, long, default_value = "0.0", value_parser = parse_confidence)]
    pub confidence: f32,

    /// Seconds of audio to include before each detection.
    #[arg(long, default_value_t = DEFAULT_PRE_PADDING, value_parser = parse_padding)]
    pub pre: f64,

    /// Seconds of audio to include after each detection.
    #[arg(long, default_value_t = DEFAULT_POST_PADDING, value_parser = parse_padding)]
    pub post: f64,

    /// Source audio file (auto-detected from detection file if omitted in CSV mode,
    /// required in direct extraction mode).
    #[arg(short, long)]
    pub audio: Option<PathBuf>,

    /// Base directory for resolving relative audio paths in detection files.
    /// If not specified, paths are resolved relative to the detection file location.
    #[arg(long)]
    pub base_dir: Option<PathBuf>,

    /// Start time in seconds for direct extraction mode.
    /// Requires --end and --audio.
    #[arg(long, requires = "end", requires = "audio", value_parser = parse_time)]
    pub start: Option<f64>,

    /// End time in seconds for direct extraction mode.
    /// Requires --start and --audio.
    #[arg(long, requires = "start", requires = "audio", value_parser = parse_time)]
    pub end: Option<f64>,
}

fn parse_padding(s: &str) -> Result<f64, String> {
    let value: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    // `is_finite` is what catches NaN, and it has to be spelled this way round:
    // a bare `value < 0.0` test is false for NaN, which is how NaN got through
    // before. The two paddings then failed differently, and neither failed
    // loudly:
    //
    // - `--pre nan` reached `(start - pre).max(0.0)`, and `f64::max` returns
    //   the other operand for a NaN receiver, so the start bound collapsed to
    //   0.0. Not "the padding was ignored": the clip began at the start of the
    //   file however late `--start` was.
    // - `--post nan` reached `end + post`, which has no `.max` to launder it,
    //   so the end bound stayed NaN all the way to the seconds-to-samples cast
    //   and came out as 0, underflowing the sample-count subtraction.
    //
    // Infinity used the same hole and pushed the bounds to a range no file
    // contains.
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "padding must be a finite non-negative number, got {value}"
        ));
    }

    if value > MAX_PADDING {
        return Err(format!(
            "padding cannot exceed {MAX_PADDING} seconds, got {value}"
        ));
    }

    Ok(value)
}

fn parse_time(s: &str) -> Result<f64, String> {
    let value: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    // See `parse_padding` for why the finiteness test comes first. Both values
    // this parser backs were reachable: `--end inf` saturated to `u64::MAX`
    // when the extractor converted seconds to samples and aborted the process
    // with a capacity overflow, while `--start nan` wrote a clip named
    // `detection_NaN-5` over a range nobody asked for, and exited 0.
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "time must be a finite non-negative number, got {value}"
        ));
    }

    Ok(value)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_time_accepts_finite_non_negative() {
        assert_eq!(parse_time("0").unwrap(), 0.0);
        assert_eq!(parse_time("12.5").unwrap(), 12.5);
    }

    #[test]
    fn test_parse_time_rejects_negative() {
        let err = parse_time("-1").unwrap_err();
        assert!(err.contains("finite non-negative"), "{err}");
    }

    #[test]
    fn test_parse_time_rejects_non_finite() {
        // The regression this file exists for: `--end inf` reached the
        // extractor as a saturated `u64::MAX` sample count and aborted the
        // process, and `--start nan` wrote a clip named after a NaN.
        for input in ["inf", "-inf", "infinity", "nan", "NaN", "-nan"] {
            let err = parse_time(input).expect_err("non-finite time must be rejected");
            assert!(err.contains("finite non-negative"), "{input}: {err}");
        }
    }

    #[test]
    fn test_parse_time_rejects_garbage() {
        let err = parse_time("abc").unwrap_err();
        assert!(err.contains("not a valid number"), "{err}");
    }

    #[test]
    fn test_parse_padding_accepts_finite_in_range() {
        assert_eq!(parse_padding("0").unwrap(), 0.0);
        assert_eq!(parse_padding("2.5").unwrap(), 2.5);
        assert_eq!(
            parse_padding(&MAX_PADDING.to_string()).unwrap(),
            MAX_PADDING
        );
    }

    #[test]
    fn test_parse_padding_rejects_non_finite() {
        // `--pre nan` used to be swallowed by `(start - pre).max(0.0)`, which
        // returns 0.0 for a NaN receiver, so the padding was dropped and the
        // run still exited 0.
        for input in ["inf", "-inf", "nan", "NaN"] {
            let err = parse_padding(input).expect_err("non-finite padding must be rejected");
            assert!(err.contains("finite non-negative"), "{input}: {err}");
        }
    }

    #[test]
    fn test_parse_padding_rejects_out_of_range() {
        assert!(
            parse_padding("-1")
                .unwrap_err()
                .contains("finite non-negative")
        );
        assert!(
            parse_padding(&(MAX_PADDING + 1.0).to_string())
                .unwrap_err()
                .contains("cannot exceed")
        );
    }
}
