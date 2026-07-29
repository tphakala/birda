//! CLI argument definitions.

use crate::config::{ModelType, OutputFormat, OutputMode};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use super::clip::ClipArgs;

/// Sort order for species list.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum SortOrder {
    /// Sort by occurrence probability (descending).
    Freq,
    /// Sort alphabetically.
    Alpha,
}

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

    /// CLI output format: human (default), json (buffered), or ndjson (streaming).
    /// Note: This controls CLI output format, not detection result file format.
    /// For file formats, use `-f/--format`.
    #[arg(long, value_enum, global = true, env = "BIRDA_OUTPUT_MODE")]
    pub output_mode: Option<OutputMode>,

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
    /// Show available execution providers (CPU, CUDA, etc.).
    Providers,
    /// Extract audio clips from detection results.
    Clip(ClipArgs),
    /// Check for and install updates from GitHub.
    Update {
        /// Only check for updates, don't install.
        #[arg(long)]
        check: bool,
    },
    /// Generate species list from range filter.
    #[command(group(
        clap::ArgGroup::new("time")
            .required(true)
            .args(["week", "month"]),
    ))]
    Species {
        /// Output file path (default: `species_list.txt` in current directory).
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Latitude for range filtering (-90.0 to 90.0).
        #[arg(long, allow_hyphen_values = true, value_parser = parse_latitude)]
        lat: f64,

        /// Longitude for range filtering (-180.0 to 180.0).
        #[arg(long, allow_hyphen_values = true, value_parser = parse_longitude)]
        lon: f64,

        /// Week number (1-48).
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..=48),
              conflicts_with_all = ["month", "day"])]
        week: Option<u32>,

        /// Month (1-12).
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..=12),
              requires = "day", conflicts_with = "week")]
        month: Option<u32>,

        /// Day of month (1-31).
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..=31),
              requires = "month", conflicts_with = "week")]
        day: Option<u32>,

        /// Range filter threshold (0.0-1.0).
        /// Note: Species list generation uses 0.03 default (vs 0.01 for live filtering)
        /// to reduce noise in generated lists.
        #[arg(long, value_parser = parse_confidence, default_value = "0.03")]
        threshold: f32,

        /// Sort order: freq (by occurrence probability) or alpha (alphabetically).
        #[arg(long, default_value = "freq")]
        sort: SortOrder,

        /// Model name whose label set the species list is written in.
        #[arg(short, long)]
        model: Option<String>,
    },
}

/// Config subcommand actions.
#[derive(Debug, Clone, Subcommand)]
pub enum ConfigAction {
    /// Create default configuration file.
    Init,
    /// Display current configuration.
    Show,
    /// Print configuration file path.
    Path,
    /// Set a configuration value.
    Set {
        /// Configuration key (dotted path, e.g., "defaults.model").
        key: String,
        // `allow_hyphen_values` because plenty of legitimate values start with a
        // minus. Without it clap reads `birda config set defaults.latitude
        // -33.9` as an unknown short flag and fails with "unexpected argument
        // '-3' found", which put every southern and western hemisphere
        // coordinate out of reach of this command. Kept as a `//` comment: clap
        // renders `///` into `--help`, and this is implementation rationale.
        /// Value to set. Values beginning with a hyphen are accepted, so
        /// coordinates such as -33.9 need no escaping.
        #[arg(allow_hyphen_values = true)]
        value: String,
    },
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
    /// Remove a model from configuration.
    Remove {
        /// Name of the model to remove (as shown in `models list`).
        name: String,
        /// Also delete model files from disk.
        #[arg(long)]
        purge: bool,
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
///
/// # Default Implementation
///
/// All fields default to `None`/`false`/`0`, representing "no user input".
/// This allows configuration file values to take precedence over defaults.
#[derive(Debug, Clone, Args, Default)]
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

    /// Model type for ad-hoc model (required with --model-path when no -m is provided).
    #[arg(long, value_enum, env = "BIRDA_MODEL_TYPE")]
    pub model_type: Option<ModelType>,

    /// Deprecated and ignored. Range filtering uses the `BirdNET` Geomodel
    /// v3.0.2; see `--geomodel-path`.
    #[arg(long, env = "BIRDA_META_MODEL_PATH", hide = true)]
    pub meta_model_path: Option<PathBuf>,

    // Global so `birda species` reaches the same field as `birda analyze`. See
    // the note on `yes` for why these are not duplicated per subcommand. Kept
    // as a `//` comment: clap renders `///` into `--help`.
    /// Path to the `BirdNET` Geomodel v3.0.2 ONNX file (overrides config).
    /// Must be given together with --geomodel-labels-path.
    #[arg(
        long,
        global = true,
        env = "BIRDA_GEOMODEL_PATH",
        requires = "geomodel_labels_path"
    )]
    pub geomodel_path: Option<PathBuf>,

    // Global for the same reason as the model path above.
    /// Path to the `BirdNET` Geomodel v3.0.2 labels file (overrides config).
    /// Must be given together with --geomodel-path.
    #[arg(
        long,
        global = true,
        env = "BIRDA_GEOMODEL_LABELS_PATH",
        requires = "geomodel_path"
    )]
    pub geomodel_labels_path: Option<PathBuf>,

    /// Enable bat detection with a regional classifier.
    /// Implies `BirdNET` v2.4 backbone with embedding extraction.
    #[arg(long, value_name = "REGION")]
    pub bat: Option<crate::config::BatRegion>,

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

    /// Inference batch size (must be at least 1).
    #[arg(short, long, value_parser = parse_batch_size, env = "BIRDA_BATCH_SIZE")]
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

    /// Disable progress bars (useful for scripting/logging).
    #[arg(long)]
    pub no_progress: bool,

    /// Disable UTF-8 BOM in CSV output (for compatibility with apps that don't handle BOM).
    #[arg(long)]
    pub no_csv_bom: bool,

    /// Auto-select best available GPU provider (priority: `TensorRT` → `CUDA` → `DirectML` → `CoreML` → `ROCm` → `OpenVINO`).
    /// Note: `CoreML` excluded from auto-selection on macOS (use `--coreml` to force). Run `birda providers` for platform-specific details.
    /// Warns and falls back to CPU if no GPU providers available.
    #[arg(long, group = "provider")]
    pub gpu: bool,

    /// Force CPU inference only.
    #[arg(long, group = "provider")]
    pub cpu: bool,

    /// Use `CUDA` provider explicitly (fail if unavailable).
    #[arg(long, group = "provider")]
    pub cuda: bool,

    /// Use `TensorRT` provider explicitly (fail if unavailable).
    #[arg(long, group = "provider")]
    pub tensorrt: bool,

    /// Use `DirectML` provider explicitly (fail if unavailable).
    #[arg(long, group = "provider")]
    pub directml: bool,

    /// Use `CoreML` provider explicitly (fail if unavailable).
    #[arg(long, group = "provider")]
    pub coreml: bool,

    /// Use `ROCm` provider explicitly (fail if unavailable).
    #[arg(long, group = "provider")]
    pub rocm: bool,

    /// Use `OpenVINO` provider explicitly (fail if unavailable).
    #[arg(long, group = "provider")]
    pub openvino: bool,

    /// Use `oneDNN` provider explicitly (fail if unavailable).
    #[arg(long, group = "provider")]
    pub onednn: bool,

    /// Use `QNN` provider explicitly (fail if unavailable).
    #[arg(long, group = "provider")]
    pub qnn: bool,

    /// Use `ACL` provider explicitly (fail if unavailable).
    #[arg(long, group = "provider")]
    pub acl: bool,

    /// Use `ArmNN` provider explicitly (fail if unavailable).
    #[arg(long, group = "provider")]
    pub armnn: bool,

    /// Use `XNNPACK` provider explicitly (optimized CPU for ARM/x86).
    #[arg(long, group = "provider")]
    pub xnnpack: bool,

    /// Latitude for range filtering (-90.0 to 90.0).
    #[arg(long, allow_hyphen_values = true, value_parser = parse_latitude, env = "BIRDA_LATITUDE")]
    pub lat: Option<f64>,

    /// Longitude for range filtering (-180.0 to 180.0).
    #[arg(long, allow_hyphen_values = true, value_parser = parse_longitude, env = "BIRDA_LONGITUDE")]
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

    /// Day of year for BSG SDM adjustment (1-366).
    /// If not provided and BSG model is used, auto-detected from file timestamp.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=366), env = "BIRDA_DAY_OF_YEAR")]
    pub day_of_year: Option<u32>,

    /// Range filter threshold (0.0-1.0).
    #[arg(long, value_parser = parse_confidence, env = "BIRDA_RANGE_THRESHOLD")]
    pub range_threshold: Option<f32>,

    /// Re-rank predictions by confidence × location score.
    ///
    /// Species with no geomodel entry are excluded when this is set, since
    /// there is no occurrence probability to weight them by.
    #[arg(long)]
    pub rerank: bool,

    /// What to do with species that have no `BirdNET` Geomodel entry.
    #[arg(long, value_enum, env = "BIRDA_RANGE_UNMATCHED")]
    pub range_unmatched: Option<crate::config::UnmatchedPolicy>,

    // Global so that every spelling reaches this one field. `AnalyzeArgs` is
    // flattened onto the root, so without `global` a subcommand could only see
    // this flag by declaring its own copy, and the two copies would populate
    // different fields: `birda --yes species` would set the root field while
    // `species` read its own, still-false one. That is the defect in #286, and
    // duplicating the flag would have moved it rather than fixed it.
    //
    // Kept as a `//` comment on purpose: clap renders `///` into `--help`, and
    // because this flag is global the text would appear under every subcommand.
    // Rationale belongs to the code, not to the user.
    //
    // Every prompt this flag can reach must read it. See `handle_models_install`
    // and `handle_models_remove`: a prompt that ignores `--yes`, or answers it
    // backwards, is the same defect as a flag that does not reach the command.
    /// Assume yes for prompts: the geomodel download offer, licence acceptance,
    /// setting an installed model as default, and `models remove --purge`.
    #[arg(short = 'y', long, global = true)]
    pub yes: bool,

    /// Path to species list file.
    /// File should contain one species per line in format: `"Genus species_Common Name"`.
    /// If lat/lon are provided, this will be ignored (dynamic filtering takes precedence).
    #[arg(long, env = "BIRDA_SPECIES_LIST")]
    pub slist: Option<PathBuf>,

    /// Remove locks older than this duration (e.g., 1h, 30m).
    #[arg(long)]
    pub stale_lock_timeout: Option<String>,

    /// Write results to stdout as NDJSON stream (single file only).
    #[arg(long, conflicts_with_all = ["output_dir", "combine", "format"])]
    pub stdout: bool,
}

impl AnalyzeArgs {
    /// The subset of these arguments that geomodel resolution actually reads.
    ///
    /// All three source fields are `global = true`, so this returns the same
    /// values whether the flags were given before the subcommand or after it.
    /// That is what lets `birda species` share one resolver with `analyze`
    /// instead of fabricating an `AnalyzeArgs` of its own, which is how #286
    /// arose.
    #[must_use]
    pub fn geomodel_request(&self) -> crate::config::GeomodelRequest<'_> {
        crate::config::GeomodelRequest {
            model_path: self.geomodel_path.as_deref(),
            labels_path: self.geomodel_labels_path.as_deref(),
            assume_yes: self.yes,
        }
    }
}

// Re-use shared validators
use super::validators::{parse_batch_size, parse_confidence, parse_latitude, parse_longitude};

#[cfg(test)]
// Test setup code: panicking on an unexpected parse shape is how these assert.
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use super::*;
    use std::path::Path;

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
    fn test_cli_parse_negative_coordinates() {
        // Verify negative lat/lon without = syntax works (southern/western hemisphere)
        let cli = Cli::try_parse_from([
            "birda",
            "test.wav",
            "--lat",
            "-33.86",
            "--lon",
            "-74.0",
            "--week=24",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(cli.analyze.lat, Some(-33.86));
        assert_eq!(cli.analyze.lon, Some(-74.0));
    }

    #[test]
    fn test_cli_parse_species_negative_coordinates() {
        let cli = Cli::try_parse_from([
            "birda",
            "species",
            "--lat",
            "-33.86",
            "--lon",
            "-74.0",
            "--week=24",
        ]);
        assert!(cli.is_ok());
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

    #[test]
    fn test_cli_parse_species_command_with_week() {
        let cli = Cli::try_parse_from([
            "birda",
            "species",
            "--lat=60.1699",
            "--lon=24.9384",
            "--week=24",
            "--output=my_species.txt",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_species_command_with_month_day() {
        let cli = Cli::try_parse_from([
            "birda",
            "species",
            "--lat=60.1699",
            "--lon=24.9384",
            "--month=6",
            "--day=15",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_species_command_requires_coordinates() {
        let cli = Cli::try_parse_from(["birda", "species", "--week=24"]);
        assert!(cli.is_err()); // Should fail without lat/lon
    }

    #[test]
    fn test_species_command_week_month_conflict() {
        let cli = Cli::try_parse_from([
            "birda",
            "species",
            "--lat=60.1699",
            "--lon=24.9384",
            "--week=24",
            "--month=6",
            "--day=15",
        ]);
        assert!(cli.is_err()); // week and month should conflict
    }

    #[test]
    fn test_cli_parse_no_csv_bom() {
        let cli = Cli::try_parse_from(["birda", "test.wav", "--no-csv-bom"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert!(cli.analyze.no_csv_bom);
    }

    #[test]
    fn test_cli_parse_default_csv_bom() {
        let cli = Cli::try_parse_from(["birda", "test.wav"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert!(!cli.analyze.no_csv_bom); // BOM enabled by default
    }

    #[test]
    fn test_cli_parse_model_type() {
        let cli = Cli::try_parse_from([
            "birda",
            "test.wav",
            "--model-type",
            "birdnet-v24",
            "--model-path",
            "/path/to/model.onnx",
            "--labels-path",
            "/path/to/labels.txt",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(cli.analyze.model_type, Some(ModelType::BirdnetV24));
        assert_eq!(
            cli.analyze.model_path,
            Some(PathBuf::from("/path/to/model.onnx"))
        );
        assert_eq!(
            cli.analyze.labels_path,
            Some(PathBuf::from("/path/to/labels.txt"))
        );
    }

    #[test]
    fn test_cli_parse_model_type_perch() {
        let cli = Cli::try_parse_from([
            "birda",
            "test.wav",
            "--model-type",
            "perch-v2",
            "--model-path",
            "/path/to/model.onnx",
            "--labels-path",
            "/path/to/labels.txt",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(cli.analyze.model_type, Some(ModelType::PerchV2));
    }

    #[test]
    fn test_cli_parse_model_type_birdnet_v30() {
        let cli = Cli::try_parse_from([
            "birda",
            "test.wav",
            "--model-type",
            "birdnet-v30",
            "--model-path",
            "/path/to/model.onnx",
            "--labels-path",
            "/path/to/labels.txt",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(cli.analyze.model_type, Some(ModelType::BirdnetV30));
    }

    #[test]
    fn test_cli_parse_invalid_model_type() {
        let cli = Cli::try_parse_from([
            "birda",
            "test.wav",
            "--model-type",
            "invalid-type",
            "--model-path",
            "/path/to/model.onnx",
        ]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_cli_parse_meta_model_path() {
        let cli = Cli::try_parse_from([
            "birda",
            "test.wav",
            "--model-type",
            "birdnet-v24",
            "--model-path",
            "/path/to/model.onnx",
            "--labels-path",
            "/path/to/labels.txt",
            "--meta-model-path",
            "/path/to/meta.onnx",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(
            cli.analyze.meta_model_path,
            Some(PathBuf::from("/path/to/meta.onnx"))
        );
    }

    #[test]
    fn test_cli_adhoc_model_complete() {
        // Full ad-hoc model specification with range filtering
        let cli = Cli::try_parse_from([
            "birda",
            "test.wav",
            "--model-type",
            "birdnet-v24",
            "--model-path",
            "/path/to/model.onnx",
            "--labels-path",
            "/path/to/labels.txt",
            "--meta-model-path",
            "/path/to/meta.onnx",
            "--lat",
            "60.17",
            "--lon",
            "24.94",
            "--week",
            "24",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(cli.analyze.model_type, Some(ModelType::BirdnetV24));
        assert!(cli.analyze.meta_model_path.is_some());
        assert_eq!(cli.analyze.lat, Some(60.17));
        assert_eq!(cli.analyze.lon, Some(24.94));
        assert_eq!(cli.analyze.week, Some(24));
    }

    #[test]
    fn test_cli_parse_with_stdout() {
        let cli = Cli::try_parse_from(["birda", "--stdout", "test.wav"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert!(cli.analyze.stdout);
    }

    #[test]
    fn test_cli_stdout_flag_exists() {
        let cli = Cli::try_parse_from(["birda", "--stdout", "test.wav"]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_stdout_conflicts_with_output_dir() {
        let cli = Cli::try_parse_from(["birda", "--stdout", "--output-dir", "/tmp", "test.wav"]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_cli_stdout_conflicts_with_combine() {
        let cli = Cli::try_parse_from(["birda", "--stdout", "--combine", "test.wav"]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_cli_stdout_conflicts_with_format() {
        let cli = Cli::try_parse_from(["birda", "--stdout", "--format", "csv", "test.wav"]);
        assert!(cli.is_err());
    }

    // ---- geomodel flag reach (#286) ----
    //
    // The defect was not that `--yes` was missing, but that the value never
    // reached the resolver. These assert on the resolved
    // `GeomodelRequest`, which is what the resolver actually consumes, rather
    // than on a struct field that might again be read by nobody.

    /// `species` requires a time argument, so every case below carries one.
    /// Without it clap fails with `MissingRequiredArgument` for `--week`, which
    /// would make the negative test below pass for entirely the wrong reason.
    fn species_argv(extra: &[&str]) -> Vec<String> {
        let mut argv: Vec<String> = [
            "birda", "species", "--lat", "60.2", "--lon", "24.9", "--week", "24",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        argv.extend(extra.iter().map(|s| (*s).to_string()));
        argv
    }

    fn request_from(argv: &[String]) -> crate::config::GeomodelRequest<'static> {
        // Leaked so the returned request can borrow for 'static, keeping these
        // tests to a single expression each. Test-only, and bounded by the
        // number of test cases.
        let cli = Box::leak(Box::new(
            Cli::try_parse_from(argv).expect("argv should parse"),
        ));
        cli.analyze.geomodel_request()
    }

    #[test]
    fn test_species_accepts_yes_after_the_subcommand() {
        // Rejected by clap before the fix, because `--yes` lived only on the
        // root and `species` had no such argument.
        let request = request_from(&species_argv(&["--yes"]));

        assert!(request.assume_yes);
    }

    #[test]
    fn test_species_accepts_short_yes_after_the_subcommand() {
        let request = request_from(&species_argv(&["-y"]));

        assert!(request.assume_yes);
    }

    #[test]
    fn test_species_sees_yes_given_before_the_subcommand() {
        // The regression guard for the shape of the fix. This spelling always
        // parsed, and always discarded the flag. An implementation that gave
        // `Species` its own `--yes` would leave this case broken, because the
        // two flags would populate different fields; `global = true` is what
        // makes both spellings land on the one field the resolver reads.
        let argv = [
            "birda", "--yes", "species", "--lat", "60.2", "--lon", "24.9", "--week", "24",
        ]
        .map(String::from);
        let request = request_from(&argv);

        assert!(
            request.assume_yes,
            "root --yes must reach the resolver for `birda --yes species`"
        );
    }

    #[test]
    fn test_species_accepts_the_geomodel_path_pair() {
        let request = request_from(&species_argv(&[
            "--geomodel-path",
            "/opt/g.onnx",
            "--geomodel-labels-path",
            "/opt/g.txt",
        ]));

        assert_eq!(request.model_path, Some(Path::new("/opt/g.onnx")));
        assert_eq!(request.labels_path, Some(Path::new("/opt/g.txt")));
    }

    #[test]
    fn test_species_rejects_a_half_specified_geomodel_pair() {
        // Making the flags global must not drop the `requires` pairing:
        // a new model with the previous version's labels fails the label count
        // check with a confusing message, so it is rejected at parse time.
        let err = Cli::try_parse_from(species_argv(&["--geomodel-path", "/opt/g.onnx"]))
            .expect_err("a lone --geomodel-path must be rejected");

        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
        // Assert it failed for the RIGHT missing argument. Without this the
        // test would also pass if `species` rejected the argv for an unrelated
        // reason, such as the missing time argument this helper supplies.
        assert!(
            err.to_string().contains("--geomodel-labels-path"),
            "must fail on the missing labels path, got: {err}"
        );
    }

    #[test]
    fn test_config_set_accepts_negative_values() {
        // `birda config set defaults.latitude -33.9` failed with "unexpected
        // argument '-3' found", because clap read the value as a short flag
        // group. That put every southern hemisphere latitude and western
        // hemisphere longitude out of reach of this command.
        let cli = Cli::try_parse_from(["birda", "config", "set", "defaults.latitude", "-33.9"])
            .expect("a negative value must be accepted");

        let Some(Command::Config {
            action: ConfigAction::Set { key, value },
        }) = cli.command
        else {
            panic!("expected a config set command");
        };

        assert_eq!(key, "defaults.latitude");
        assert_eq!(value, "-33.9");
    }

    #[test]
    fn test_config_set_accepts_negative_longitude() {
        let cli = Cli::try_parse_from(["birda", "config", "set", "defaults.longitude", "-74.006"])
            .expect("a negative longitude must be accepted");

        let Some(Command::Config {
            action: ConfigAction::Set { value, .. },
        }) = cli.command
        else {
            panic!("expected a config set command");
        };

        assert_eq!(value, "-74.006");
    }

    #[test]
    fn test_models_install_accepts_yes() {
        // `--yes` became reachable here when it was made global. It must parse
        // AND be read; the reading is asserted by the resolver-side tests
        // above sharing the one field this populates.
        let cli = Cli::try_parse_from(["birda", "models", "install", "birdnet-v24", "--yes"])
            .expect("models install must accept --yes");

        assert!(
            cli.analyze.yes,
            "--yes must reach the one field that is read"
        );
    }

    #[test]
    fn test_analyze_request_defaults_to_no_flags() {
        let argv = ["birda", "test.wav"].map(String::from);
        let request = request_from(&argv);

        assert!(!request.assume_yes);
        assert_eq!(request.model_path, None);
        assert_eq!(request.labels_path, None);
    }
}
