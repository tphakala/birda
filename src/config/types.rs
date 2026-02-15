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

    /// Optional meta model for range filtering.
    #[serde(default)]
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

    /// Global default meta model path.
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
mod tests {
    use super::*;

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
