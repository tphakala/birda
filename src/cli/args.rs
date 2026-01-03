//! CLI argument definitions.

use crate::config::{ModelType, OutputFormat};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

/// Bird species detection using `BirdNET` and Perch models.
#[derive(Debug, Parser)]
#[command(name = "birda")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Subcommand to run.
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Input files or directories to analyze.
    pub inputs: Vec<PathBuf>,

    /// Common options for analysis.
    #[command(flatten)]
    pub analyze: AnalyzeArgs,
}

/// Available subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Manage configuration.
    Config {
        /// Configuration action to perform.
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Manage models.
    Models {
        /// Models action to perform.
        #[command(subcommand)]
        action: ModelsAction,
    },
}

/// Config subcommand actions.
#[derive(Debug, Clone, Copy, Subcommand)]
pub enum ConfigAction {
    /// Create default configuration file.
    Init,
    /// Display current configuration.
    Show,
    /// Print configuration file path.
    Path,
}

/// Models subcommand actions.
#[derive(Debug, Subcommand)]
pub enum ModelsAction {
    /// List configured models.
    List,
    /// List models available for download.
    ListAvailable,
    /// Add a new model to configuration.
    Add {
        /// Name for this model (e.g., "birdnet", "perch").
        name: String,
        /// Path to the ONNX model file.
        #[arg(long)]
        path: PathBuf,
        /// Path to the labels file.
        #[arg(long)]
        labels: PathBuf,
        /// Model type.
        #[arg(long, value_enum)]
        r#type: ModelType,
        /// Set as the default model.
        #[arg(long)]
        default: bool,
    },
    /// Verify model files exist and are valid.
    Check,
    /// Show details for a specific model.
    Info {
        /// Model ID from registry or name from configuration.
        id: String,
        /// Show available languages (for registry models).
        #[arg(long)]
        languages: bool,
    },
    /// Install a model from the registry.
    Install {
        /// Model ID to install.
        id: String,
        /// Language code for labels (default: en).
        #[arg(short, long)]
        language: Option<String>,
        /// Set as default model.
        #[arg(short, long)]
        default: bool,
    },
}

/// Arguments for the analyze command.
#[derive(Debug, Args)]
#[allow(clippy::struct_excessive_bools)]
pub struct AnalyzeArgs {
    /// Model name from configuration.
    #[arg(short, long, env = "BIRDA_MODEL")]
    pub model: Option<String>,

    /// Path to ONNX model file (overrides config).
    #[arg(long, env = "BIRDA_MODEL_PATH")]
    pub model_path: Option<PathBuf>,

    /// Path to labels file (overrides config).
    #[arg(long, env = "BIRDA_LABELS_PATH")]
    pub labels_path: Option<PathBuf>,

    /// Output formats (comma-separated: csv,raven,audacity,kaleidoscope).
    #[arg(short, long, value_delimiter = ',', env = "BIRDA_FORMAT")]
    pub format: Option<Vec<OutputFormat>>,

    /// Output directory (default: same as input).
    #[arg(short, long, env = "BIRDA_OUTPUT_DIR")]
    pub output_dir: Option<PathBuf>,

    /// Minimum confidence threshold (0.0-1.0).
    #[arg(short = 'c', long, value_parser = parse_confidence, env = "BIRDA_MIN_CONFIDENCE")]
    pub min_confidence: Option<f32>,

    /// Segment overlap in seconds.
    #[arg(long, env = "BIRDA_OVERLAP")]
    pub overlap: Option<f32>,

    /// Inference batch size.
    #[arg(short, long, env = "BIRDA_BATCH_SIZE")]
    pub batch_size: Option<usize>,

    /// Generate combined results file.
    #[arg(long)]
    pub combine: bool,

    /// Reprocess files even if output exists.
    #[arg(long)]
    pub force: bool,

    /// Stop on first error.
    #[arg(long)]
    pub fail_fast: bool,

    /// Suppress progress output.
    #[arg(short, long)]
    pub quiet: bool,

    /// Increase verbosity (-v: debug, -vv: trace+ORT info, -vvv: trace+ORT debug).
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Enable CUDA GPU acceleration.
    #[arg(long, conflicts_with = "cpu")]
    pub gpu: bool,

    /// Force CPU inference.
    #[arg(long, conflicts_with = "gpu")]
    pub cpu: bool,

    /// Latitude for range filtering (-90.0 to 90.0).
    #[arg(long, value_parser = parse_latitude, env = "BIRDA_LATITUDE")]
    pub lat: Option<f64>,

    /// Longitude for range filtering (-180.0 to 180.0).
    #[arg(long, value_parser = parse_longitude, env = "BIRDA_LONGITUDE")]
    pub lon: Option<f64>,

    /// Week number for range filtering (1-48).
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=48),
          conflicts_with_all = ["month", "day"])]
    pub week: Option<u32>,

    /// Month for range filtering (1-12).
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=12),
          requires = "day", conflicts_with = "week")]
    pub month: Option<u32>,

    /// Day of month for range filtering (1-31).
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=31),
          requires = "month", conflicts_with = "week")]
    pub day: Option<u32>,

    /// Range filter threshold (0.0-1.0).
    #[arg(long, value_parser = parse_confidence, env = "BIRDA_RANGE_THRESHOLD")]
    pub range_threshold: Option<f32>,

    /// Re-rank predictions by confidence × location score.
    #[arg(long)]
    pub rerank: bool,

    /// Path to species list file.
    /// File should contain one species per line in format: `"Genus species_Common Name"`.
    /// If lat/lon are provided, this will be ignored (dynamic filtering takes precedence).
    #[arg(long, env = "BIRDA_SPECIES_LIST")]
    pub slist: Option<PathBuf>,

    /// Remove locks older than this duration (e.g., 1h, 30m).
    #[arg(long)]
    pub stale_lock_timeout: Option<String>,
}

/// Parse and validate latitude value.
fn parse_latitude(s: &str) -> Result<f64, String> {
    let value: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    if !(-90.0..=90.0).contains(&value) {
        return Err(format!(
            "latitude must be between -90.0 and 90.0, got {value}"
        ));
    }

    Ok(value)
}

/// Parse and validate longitude value.
fn parse_longitude(s: &str) -> Result<f64, String> {
    let value: f64 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;

    if !(-180.0..=180.0).contains(&value) {
        return Err(format!(
            "longitude must be between -180.0 and 180.0, got {value}"
        ));
    }

    Ok(value)
}

/// Parse and validate confidence value.
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
        assert!(parse_confidence("1.5").is_err());
        assert!(parse_confidence("-0.1").is_err());
        assert!(parse_confidence("abc").is_err());
    }

    #[test]
    fn test_cli_parse_simple() {
        let cli = Cli::try_parse_from(["birda", "test.wav"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(cli.inputs.len(), 1);
    }

    #[test]
    fn test_cli_parse_with_options() {
        let cli =
            Cli::try_parse_from(["birda", "test.wav", "-m", "birdnet-v24", "-c", "0.25", "-q"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(cli.analyze.model, Some("birdnet-v24".to_string()));
        assert_eq!(cli.analyze.min_confidence, Some(0.25));
        assert!(cli.analyze.quiet);
    }

    #[test]
    fn test_cli_parse_config_subcommand() {
        let cli = Cli::try_parse_from(["birda", "config", "show"]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_parse_latitude_valid() {
        assert_eq!(parse_latitude("0.0").ok(), Some(0.0));
        assert_eq!(parse_latitude("90.0").ok(), Some(90.0));
        assert_eq!(parse_latitude("-90.0").ok(), Some(-90.0));
        assert_eq!(parse_latitude("40.7128").ok(), Some(40.7128));
    }

    #[test]
    fn test_parse_latitude_invalid() {
        assert!(parse_latitude("91.0").is_err());
        assert!(parse_latitude("-91.0").is_err());
        assert!(parse_latitude("abc").is_err());
    }

    #[test]
    fn test_parse_longitude_valid() {
        assert_eq!(parse_longitude("0.0").ok(), Some(0.0));
        assert_eq!(parse_longitude("180.0").ok(), Some(180.0));
        assert_eq!(parse_longitude("-180.0").ok(), Some(-180.0));
        assert_eq!(parse_longitude("-74.0060").ok(), Some(-74.0060));
    }

    #[test]
    fn test_parse_longitude_invalid() {
        assert!(parse_longitude("181.0").is_err());
        assert!(parse_longitude("-181.0").is_err());
        assert!(parse_longitude("abc").is_err());
    }

    #[test]
    fn test_cli_parse_range_filter_week() {
        let cli = Cli::try_parse_from([
            "birda",
            "test.wav",
            "--lat=40.7",
            "--lon=-74.0",
            "--week=24",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(cli.analyze.lat, Some(40.7));
        assert_eq!(cli.analyze.lon, Some(-74.0));
        assert_eq!(cli.analyze.week, Some(24));
    }

    #[test]
    fn test_cli_parse_range_filter_month_day() {
        let cli = Cli::try_parse_from([
            "birda",
            "test.wav",
            "--lat=40.7",
            "--lon=-74.0",
            "--month=6",
            "--day=15",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(cli.analyze.month, Some(6));
        assert_eq!(cli.analyze.day, Some(15));
    }

    #[test]
    fn test_cli_parse_range_filter_conflicts() {
        // week and month should conflict
        let cli = Cli::try_parse_from([
            "birda", "test.wav", "--week", "24", "--month", "6", "--day", "15",
        ]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_cli_parse_with_species_list() {
        let cli = Cli::try_parse_from(["birda", "test.wav", "--slist", "species_list.txt"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(cli.analyze.slist, Some(PathBuf::from("species_list.txt")));
    }
}
