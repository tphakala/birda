//! Configuration type definitions.

use crate::constants::{DEFAULT_BATCH_SIZE, DEFAULT_MIN_CONFIDENCE, DEFAULT_OVERLAP};
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

    /// Model type (birdnet-v24, birdnet-v30, perch-v2).
    #[serde(rename = "type")]
    pub model_type: ModelType,
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

    /// Batch size for inference.
    pub batch_size: usize,

    /// CSV column configuration.
    #[serde(default)]
    pub csv_columns: CsvColumnsConfig,
}

impl Default for DefaultsConfig {
    fn default() -> Self {
        Self {
            model: None,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            overlap: DEFAULT_OVERLAP,
            formats: vec![OutputFormat::Csv],
            batch_size: DEFAULT_BATCH_SIZE,
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
    /// Automatically select (GPU if available, else CPU).
    #[default]
    Auto,
    /// Force GPU (CUDA), fail if unavailable.
    Gpu,
    /// Force CPU inference.
    Cpu,
}

/// Inference settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    /// Device to use for inference.
    pub device: InferenceDevice,
}

/// Output settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Prefix for combined output files.
    pub combined_prefix: String,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            combined_prefix: "BirdNET".to_string(),
        }
    }
}

/// Supported output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Csv => write!(f, "csv"),
            Self::Raven => write!(f, "raven"),
            Self::Audacity => write!(f, "audacity"),
            Self::Kaleidoscope => write!(f, "kaleidoscope"),
        }
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "csv" => Ok(Self::Csv),
            "raven" | "table" => Ok(Self::Raven),
            "audacity" => Ok(Self::Audacity),
            "kaleidoscope" => Ok(Self::Kaleidoscope),
            other => Err(format!("unknown output format: {other}")),
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
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BirdnetV24 => write!(f, "birdnet-v24"),
            Self::BirdnetV30 => write!(f, "birdnet-v30"),
            Self::PerchV2 => write!(f, "perch-v2"),
        }
    }
}

impl std::str::FromStr for ModelType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "birdnet-v24" => Ok(Self::BirdnetV24),
            "birdnet-v30" => Ok(Self::BirdnetV30),
            "perch-v2" => Ok(Self::PerchV2),
            other => Err(format!("unknown model type: {other}")),
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
        assert!("unknown".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Csv.to_string(), "csv");
        assert_eq!(OutputFormat::Raven.to_string(), "raven");
    }

    #[test]
    fn test_defaults_config_default_values() {
        let defaults = DefaultsConfig::default();
        assert_eq!(defaults.min_confidence, 0.1);
        assert_eq!(defaults.overlap, 0.0);
        assert_eq!(defaults.batch_size, 1);
    }
}
