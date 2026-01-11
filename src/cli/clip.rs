//! CLI for clip extraction subcommand.

use std::path::PathBuf;

use clap::Args;

/// Arguments for the clip subcommand.
#[derive(Debug, Args)]
pub struct ClipArgs {
    /// Detection result files to process (CSV format).
    #[arg(required = true)]
    pub files: Vec<PathBuf>,

    /// Output directory for extracted clips.
    #[arg(short, long, default_value = "clips")]
    pub output: PathBuf,

    /// Minimum confidence threshold (0.0-1.0).
    #[arg(short, long, default_value = "0.0", value_parser = parse_confidence)]
    pub confidence: f32,

    /// Seconds of audio to include before each detection.
    #[arg(long, default_value = "5.0", value_parser = parse_padding)]
    pub pre: f64,

    /// Seconds of audio to include after each detection.
    #[arg(long, default_value = "5.0", value_parser = parse_padding)]
    pub post: f64,

    /// Source audio file (auto-detected from detection file if omitted).
    #[arg(short, long)]
    pub audio: Option<PathBuf>,

    /// Base directory for resolving relative audio paths in detection files.
    /// If not specified, paths are resolved relative to the detection file location.
    #[arg(long)]
    pub base_dir: Option<PathBuf>,

    /// Process files even if clips already exist.
    #[arg(long)]
    pub force: bool,
}

fn parse_confidence(s: &str) -> Result<f32, String> {
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

fn parse_padding(s: &str) -> Result<f64, String> {
    let value: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    if value < 0.0 {
        return Err(format!("padding cannot be negative, got {value}"));
    }

    if value > 300.0 {
        return Err(format!("padding cannot exceed 300 seconds, got {value}"));
    }

    Ok(value)
}
