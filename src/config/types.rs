//! Configuration type definitions.

use crate::constants::{DEFAULT_MIN_CONFIDENCE, DEFAULT_OVERLAP};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Complete application configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Configured models by name.
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,

    /// Default settings.
    #[serde(default)]
    pub defaults: DefaultsConfig,

    /// Inference settings.
    #[serde(default)]
    pub inference: InferenceConfig,

    /// Output settings.
    #[serde(default)]
    pub output: OutputConfig,
}

/// Configuration for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the ONNX model file.
    pub path: PathBuf,

    /// Path to the labels file.
    pub labels: PathBuf,

    /// Model type (birdnet-v24, birdnet-v30, perch-v2, bsg-finland).
    #[serde(rename = "type")]
    pub model_type: ModelType,

    /// Deprecated. The `BirdNET` v2.4 meta model was replaced by the shared
    /// `BirdNET` Geomodel v3.0.2, configured under `defaults.geomodel`.
    ///
    /// Parsed only so a stale key can be reported. Serde ignores unknown keys,
    /// so dropping the field outright would make a per-model `meta_model`
    /// disappear without a word. Never written back.
    #[serde(default, skip_serializing)]
    pub meta_model: Option<PathBuf>,

    /// BSG calibration CSV file (required for BSG models).
    #[serde(default)]
    pub bsg_calibration: Option<PathBuf>,

    /// BSG migration CSV file (required for BSG models).
    #[serde(default)]
    pub bsg_migration: Option<PathBuf>,

    /// BSG distribution maps binary file (required for BSG models).
    #[serde(default)]
    pub bsg_distribution_maps: Option<PathBuf>,
}

/// Default analysis settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DefaultsConfig {
    /// Default model name to use.
    pub model: Option<String>,

    /// Minimum confidence threshold.
    pub min_confidence: f32,

    /// Segment overlap in seconds.
    pub overlap: f32,

    /// Output formats.
    pub formats: Vec<OutputFormat>,

    /// Batch size for inference. If None, a smart default based on model type
    /// and execution provider will be used.
    pub batch_size: Option<usize>,

    /// Latitude for range filtering.
    pub latitude: Option<f64>,

    /// Longitude for range filtering.
    pub longitude: Option<f64>,

    /// Range filter threshold.
    #[serde(default = "default_range_threshold")]
    pub range_threshold: f32,

    /// Path to the `BirdNET` Geomodel v3.0.2 ONNX file used for range filtering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geomodel: Option<PathBuf>,

    /// Path to the `BirdNET` Geomodel v3.0.2 labels file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geomodel_labels: Option<PathBuf>,

    /// How to treat classifier species that have no geomodel entry.
    #[serde(default)]
    pub range_unmatched: UnmatchedPolicy,

    /// Deprecated. Replaced by [`DefaultsConfig::geomodel`].
    ///
    /// Parsed only so a stale key can be reported; never written back.
    #[serde(default, skip_serializing)]
    pub meta_model: Option<PathBuf>,

    /// Optional species list file for filtering results.
    /// Format: one species per line as `"Genus species_Common Name"` (e.g., `"Parus major_Great Tit"`).
    /// Ignored if latitude/longitude are provided (dynamic filtering takes precedence).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub species_list_file: Option<PathBuf>,

    /// Day of year for BSG SDM adjustment (1-366).
    /// If not set, auto-detected from file timestamp when BSG model is used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub day_of_year: Option<u32>,

    /// CSV column configuration.
    #[serde(default)]
    pub csv_columns: CsvColumnsConfig,
}

/// What to do with classifier species that have no `BirdNET` Geomodel entry.
///
/// No classifier's label set is a subset of the geomodel's 12,012 species.
/// `BirdNET` v2.4 has 305 labels with no geomodel entry, mostly eBird taxonomic
/// revisions plus non-species labels such as Dog and Siren; Perch v2 has 3,650,
/// mostly sound classes, insects and amphibians.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum UnmatchedPolicy {
    /// Pass unmatched species through unfiltered, so no detection is ever lost
    /// to missing range data. This is the default.
    #[default]
    #[value(name = "keep")]
    Keep,
    /// Filter out unmatched species, treating absence from the geomodel as
    /// absence from the location.
    #[value(name = "drop")]
    Drop,
}

impl std::fmt::Display for UnmatchedPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Keep => write!(f, "keep"),
            Self::Drop => write!(f, "drop"),
        }
    }
}

/// Default range filter threshold.
fn default_range_threshold() -> f32 {
    crate::constants::range_filter::DEFAULT_THRESHOLD
}

impl Default for DefaultsConfig {
    fn default() -> Self {
        Self {
            model: None,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            overlap: DEFAULT_OVERLAP,
            formats: vec![OutputFormat::Csv],
            batch_size: None, // Use smart defaults based on model/EP
            latitude: None,
            longitude: None,
            range_threshold: default_range_threshold(),
            geomodel: None,
            geomodel_labels: None,
            range_unmatched: UnmatchedPolicy::default(),
            meta_model: None,
            species_list_file: None,
            day_of_year: None,
            csv_columns: CsvColumnsConfig::default(),
        }
    }
}

/// CSV additional columns configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CsvColumnsConfig {
    /// Additional columns to include.
    pub include: Vec<String>,
}

/// Inference device configuration.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum InferenceDevice {
    /// Automatically select (GPU if available, silent CPU fallback).
    #[default]
    Auto,
    /// Force CPU inference.
    Cpu,
    /// Auto-select best available GPU provider (warn on CPU fallback).
    Gpu,
    /// Explicit `CUDA` provider (fail if unavailable).
    Cuda,
    /// Explicit `TensorRT` provider (fail if unavailable).
    #[serde(rename = "tensorrt")]
    TensorRt,
    /// Explicit `DirectML` provider (fail if unavailable).
    #[serde(rename = "directml")]
    DirectMl,
    /// Explicit `CoreML` provider (fail if unavailable).
    #[serde(rename = "coreml")]
    CoreMl,
    /// Explicit `ROCm` provider (fail if unavailable).
    #[serde(rename = "rocm")]
    Rocm,
    /// Explicit `OpenVINO` provider (fail if unavailable).
    #[serde(rename = "openvino")]
    OpenVino,
    /// Explicit `oneDNN` provider (fail if unavailable).
    #[serde(rename = "onednn")]
    OneDnn,
    /// Explicit `QNN` provider (fail if unavailable).
    #[serde(rename = "qnn")]
    Qnn,
    /// Explicit `ACL` provider (fail if unavailable).
    #[serde(rename = "acl")]
    Acl,
    /// Explicit `ArmNN` provider (fail if unavailable).
    #[serde(rename = "armnn")]
    ArmNn,
    /// Explicit `XNNPACK` provider (fail if unavailable).
    /// Optimized CPU inference for ARM/x86 platforms.
    #[serde(rename = "xnnpack")]
    Xnnpack,
}

/// Inference settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    /// Device to use for inference.
    pub device: InferenceDevice,
}

/// CLI output mode for structured output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum OutputMode {
    /// Human-readable output with progress bars and colors.
    #[default]
    Human,
    /// Buffered JSON array of envelopes at completion.
    Json,
    /// Newline-delimited JSON, one event per line (streaming).
    Ndjson,
}

impl OutputMode {
    /// Check if output mode is structured (JSON or NDJSON).
    #[must_use]
    pub fn is_structured(self) -> bool {
        matches!(self, Self::Json | Self::Ndjson)
    }
}

impl std::fmt::Display for OutputMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Human => write!(f, "human"),
            Self::Json => write!(f, "json"),
            Self::Ndjson => write!(f, "ndjson"),
        }
    }
}

/// Output settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Prefix for combined output files.
    pub combined_prefix: String,

    /// Default CLI output format.
    pub default_format: OutputMode,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            combined_prefix: "BirdNET".to_string(),
            default_format: OutputMode::Human,
        }
    }
}

/// Supported output formats for detection results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// Generic CSV format.
    Csv,
    /// Raven selection table.
    Raven,
    /// Audacity labels.
    Audacity,
    /// Kaleidoscope CSV.
    Kaleidoscope,
    /// JSON format with metadata and summary.
    Json,
    /// Apache Parquet columnar format.
    Parquet,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Csv => write!(f, "csv"),
            Self::Raven => write!(f, "raven"),
            Self::Audacity => write!(f, "audacity"),
            Self::Kaleidoscope => write!(f, "kaleidoscope"),
            Self::Json => write!(f, "json"),
            Self::Parquet => write!(f, "parquet"),
        }
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = crate::error::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "csv" => Ok(Self::Csv),
            "raven" | "table" => Ok(Self::Raven),
            "audacity" => Ok(Self::Audacity),
            "kaleidoscope" => Ok(Self::Kaleidoscope),
            "json" => Ok(Self::Json),
            "parquet" => Ok(Self::Parquet),
            other => Err(crate::error::Error::InvalidOutputFormat {
                value: other.to_string(),
            }),
        }
    }
}

/// Supported model types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum ModelType {
    /// `BirdNET` v2.4 model.
    #[value(name = "birdnet-v24")]
    BirdnetV24,
    /// `BirdNET` v3.0 model.
    #[value(name = "birdnet-v30")]
    BirdnetV30,
    /// Google Perch v2 model.
    #[value(name = "perch-v2")]
    PerchV2,
    /// BSG Finland model (`BirdNET` v2.4 backbone + Finnish classification head).
    #[value(name = "bsg-finland")]
    BsgFinland,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BirdnetV24 => write!(f, "birdnet-v24"),
            Self::BirdnetV30 => write!(f, "birdnet-v30"),
            Self::PerchV2 => write!(f, "perch-v2"),
            Self::BsgFinland => write!(f, "bsg-finland"),
        }
    }
}

impl std::str::FromStr for ModelType {
    type Err = crate::error::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "birdnet-v24" => Ok(Self::BirdnetV24),
            "birdnet-v30" => Ok(Self::BirdnetV30),
            "perch-v2" => Ok(Self::PerchV2),
            "bsg-finland" => Ok(Self::BsgFinland),
            other => Err(crate::error::Error::InvalidModelType {
                value: other.to_string(),
            }),
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
#[allow(clippy::unwrap_used)] // Test setup code - panics are acceptable
mod tests {
    use super::*;

    #[test]
    fn test_unmatched_policy_defaults_to_keep() {
        assert_eq!(UnmatchedPolicy::default(), UnmatchedPolicy::Keep);
        assert_eq!(
            DefaultsConfig::default().range_unmatched,
            UnmatchedPolicy::Keep,
            "keeping unmatched species must be the default, so upgrading never \
             silently drops detections"
        );
    }

    #[test]
    fn test_unmatched_policy_parses_kebab_case() {
        let defaults: DefaultsConfig = toml::from_str("range_unmatched = \"drop\"\n").unwrap();
        assert_eq!(defaults.range_unmatched, UnmatchedPolicy::Drop);
    }

    #[test]
    fn test_unmatched_policy_displays_as_its_cli_value() {
        assert_eq!(UnmatchedPolicy::Keep.to_string(), "keep");
        assert_eq!(UnmatchedPolicy::Drop.to_string(), "drop");
    }

    #[test]
    fn test_defaults_config_reads_geomodel_paths() {
        let toml_src = r#"
geomodel = "/models/birdnet-geomodel-v3.0.2.onnx"
geomodel_labels = "/models/birdnet-geomodel-v3.0.2-labels.txt"
"#;
        let defaults: DefaultsConfig = toml::from_str(toml_src).unwrap();

        assert_eq!(
            defaults.geomodel.unwrap(),
            PathBuf::from("/models/birdnet-geomodel-v3.0.2.onnx")
        );
        assert_eq!(
            defaults.geomodel_labels.unwrap(),
            PathBuf::from("/models/birdnet-geomodel-v3.0.2-labels.txt")
        );
    }

    #[test]
    fn test_deprecated_meta_model_still_parses_in_defaults() {
        let defaults: DefaultsConfig =
            toml::from_str("meta_model = \"/models/birdnet-v24-meta.onnx\"\n").unwrap();

        assert!(
            defaults.meta_model.is_some(),
            "the key must still parse so its presence can be reported"
        );
    }

    #[test]
    fn test_deprecated_meta_model_still_parses_per_model() {
        // Keeping the field only on DefaultsConfig would make a per-model
        // meta_model vanish silently, because serde drops unknown keys.
        let toml_src = r#"
path = "/m.onnx"
labels = "/l.txt"
type = "birdnet-v24"
meta_model = "/models/birdnet-v24-meta.onnx"
"#;
        let model: ModelConfig = toml::from_str(toml_src).unwrap();

        assert!(model.meta_model.is_some());
    }

    #[test]
    fn test_deprecated_meta_model_is_not_written_back() {
        let defaults = DefaultsConfig {
            meta_model: Some(PathBuf::from("/models/birdnet-v24-meta.onnx")),
            ..Default::default()
        };

        let written = toml::to_string(&defaults).unwrap();

        assert!(
            !written.contains("meta_model"),
            "the deprecated key must be dropped on the next config write"
        );
    }

    #[test]
    fn test_geomodel_paths_are_written_back() {
        let defaults = DefaultsConfig {
            geomodel: Some(PathBuf::from("/models/birdnet-geomodel-v3.0.2.onnx")),
            geomodel_labels: Some(PathBuf::from("/models/birdnet-geomodel-v3.0.2-labels.txt")),
            ..Default::default()
        };

        let written = toml::to_string(&defaults).unwrap();

        assert!(written.contains("geomodel"));
        assert!(written.contains("geomodel_labels"));
    }

    #[test]
    fn test_output_format_from_str() {
        assert_eq!("csv".parse::<OutputFormat>().ok(), Some(OutputFormat::Csv));
        assert_eq!(
            "raven".parse::<OutputFormat>().ok(),
            Some(OutputFormat::Raven)
        );
        assert_eq!(
            "table".parse::<OutputFormat>().ok(),
            Some(OutputFormat::Raven)
        );
        assert_eq!(
            "audacity".parse::<OutputFormat>().ok(),
            Some(OutputFormat::Audacity)
        );
        assert_eq!(
            "kaleidoscope".parse::<OutputFormat>().ok(),
            Some(OutputFormat::Kaleidoscope)
        );
        assert_eq!(
            "json".parse::<OutputFormat>().ok(),
            Some(OutputFormat::Json)
        );
        assert!("unknown".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Csv.to_string(), "csv");
        assert_eq!(OutputFormat::Raven.to_string(), "raven");
        assert_eq!(OutputFormat::Json.to_string(), "json");
    }

    #[test]
    fn test_output_mode_display() {
        assert_eq!(OutputMode::Human.to_string(), "human");
        assert_eq!(OutputMode::Json.to_string(), "json");
        assert_eq!(OutputMode::Ndjson.to_string(), "ndjson");
    }

    #[test]
    fn test_output_config_default() {
        let config = OutputConfig::default();
        assert_eq!(config.combined_prefix, "BirdNET");
        assert_eq!(config.default_format, OutputMode::Human);
    }

    #[test]
    fn test_defaults_config_default_values() {
        let defaults = DefaultsConfig::default();
        assert_eq!(defaults.min_confidence, 0.1);
        assert_eq!(defaults.overlap, 0.0);
        assert_eq!(defaults.batch_size, None);
    }

    #[test]
    fn test_defaults_with_species_list_file() {
        let defaults = DefaultsConfig {
            species_list_file: Some(PathBuf::from("/path/to/species_list.txt")),
            ..Default::default()
        };
        assert!(defaults.species_list_file.is_some());
    }
}
