//! Bat detection configuration.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Regional bat classifier variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum BatRegion {
    /// Bavaria (Germany).
    Bavaria,
    /// Bavaria high-confidence variant.
    #[value(name = "bavaria-high")]
    BavariaHigh,
    /// European Union (broad coverage).
    Eu,
    /// Scotland.
    Scotland,
    /// South Wales.
    #[value(name = "south-wales")]
    SouthWales,
    /// Sweden.
    Sweden,
    /// United Kingdom.
    Uk,
    /// United States (full).
    Usa,
    /// United States East Coast.
    #[value(name = "usa-east")]
    UsaEast,
    /// United States East Coast high-confidence variant.
    #[value(name = "usa-east-high")]
    UsaEastHigh,
    /// United States West Coast.
    #[value(name = "usa-west")]
    UsaWest,
}

impl BatRegion {
    /// Model filename stem for this region.
    #[must_use]
    pub fn model_stem(&self) -> &'static str {
        match self {
            Self::Bavaria => "BattyBirdNET-Bavaria-256kHz",
            Self::BavariaHigh => "BattyBirdNET-Bavaria-256kHz-high",
            Self::Eu => "BattyBirdNET-EU-256kHz",
            Self::Scotland => "BattyBirdNET-Scotland-256kHz",
            Self::SouthWales => "BattyBirdNET-SouthWales-256kHz",
            Self::Sweden => "BattyBirdNET-Sweden-256kHz",
            Self::Uk => "BattyBirdNET-UK-256kHz",
            Self::Usa => "BattyBirdNET-USA-256kHz",
            Self::UsaEast => "BattyBirdNET-USA-EAST-256kHz",
            Self::UsaEastHigh => "BattyBirdNET-USA-EAST-256kHz-high",
            Self::UsaWest => "BattyBirdNET-USA-WEST-256kHz",
        }
    }

    /// ONNX model filename for this region.
    #[must_use]
    pub fn model_filename(&self) -> String {
        format!("{}_fp32.onnx", self.model_stem())
    }

    /// Labels filename for this region.
    #[must_use]
    pub fn labels_filename(&self) -> String {
        format!("{}_Labels.txt", self.model_stem())
    }
}

impl std::fmt::Display for BatRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model_stem())
    }
}

/// Resolved bat detection configuration.
#[derive(Debug, Clone)]
pub struct BatConfig {
    /// Selected bat region.
    pub region: BatRegion,
    /// Path to the bat classifier ONNX model.
    pub classifier_path: PathBuf,
    /// Path to the bat classifier labels file.
    pub labels_path: PathBuf,
}

impl BatConfig {
    /// Resolve bat config from a region and models directory.
    ///
    /// # Errors
    ///
    /// Returns `Error::ModelFileNotFound` if the classifier ONNX file is missing.
    /// Returns `Error::LabelsFileNotFound` if the labels file is missing.
    pub fn resolve(region: BatRegion, bat_models_dir: &std::path::Path) -> Result<Self> {
        let classifier_path = bat_models_dir.join(region.model_filename());
        let labels_path = bat_models_dir.join(region.labels_filename());

        if !classifier_path.exists() {
            return Err(Error::ModelFileNotFound {
                path: classifier_path,
            });
        }
        if !labels_path.exists() {
            return Err(Error::LabelsFileNotFound { path: labels_path });
        }

        Ok(Self {
            region,
            classifier_path,
            labels_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bat_region_model_filename() {
        assert_eq!(
            BatRegion::Uk.model_filename(),
            "BattyBirdNET-UK-256kHz_fp32.onnx"
        );
        assert_eq!(
            BatRegion::UsaEastHigh.model_filename(),
            "BattyBirdNET-USA-EAST-256kHz-high_fp32.onnx"
        );
    }

    #[test]
    fn test_bat_region_labels_filename() {
        assert_eq!(
            BatRegion::Eu.labels_filename(),
            "BattyBirdNET-EU-256kHz_Labels.txt"
        );
    }

    #[test]
    fn test_bat_region_all_variants_have_filenames() {
        let regions = [
            BatRegion::Bavaria,
            BatRegion::BavariaHigh,
            BatRegion::Eu,
            BatRegion::Scotland,
            BatRegion::SouthWales,
            BatRegion::Sweden,
            BatRegion::Uk,
            BatRegion::Usa,
            BatRegion::UsaEast,
            BatRegion::UsaEastHigh,
            BatRegion::UsaWest,
        ];
        for region in regions {
            assert!(!region.model_filename().is_empty());
            assert!(!region.labels_filename().is_empty());
            assert!(region.model_filename().ends_with("_fp32.onnx"));
            assert!(region.labels_filename().ends_with("_Labels.txt"));
        }
    }

    #[test]
    fn test_bat_config_resolve_missing_model() {
        let result = BatConfig::resolve(BatRegion::Uk, std::path::Path::new("/nonexistent"));
        assert!(result.is_err());
    }
}
