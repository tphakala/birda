//! Output type definitions.

use serde::Serialize;
use std::path::PathBuf;

/// A single bird detection.
#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    /// Path to the source audio file.
    pub file_path: PathBuf,
    /// Detection start time in seconds.
    pub start_time: f32,
    /// Detection end time in seconds.
    pub end_time: f32,
    /// Scientific name of the species.
    pub scientific_name: String,
    /// Common name of the species.
    pub common_name: String,
    /// Detection confidence (0.0 - 1.0).
    pub confidence: f32,
    /// Additional metadata.
    pub metadata: DetectionMetadata,
}

/// Optional metadata for detections.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DetectionMetadata {
    /// Recording latitude.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lat: Option<f64>,
    /// Recording longitude.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lon: Option<f64>,
    /// Week of year (1-48).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub week: Option<u8>,
    /// Model name used for detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Overlap setting used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlap: Option<f32>,
    /// Sensitivity setting used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sensitivity: Option<f32>,
    /// Minimum confidence threshold used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_conf: Option<f32>,
    /// Species list file path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub species_list: Option<String>,
}

impl Detection {
    /// Parse species label in `BirdNET` format.
    ///
    /// `BirdNET` labels are formatted as `ScientificName_CommonName`.
    pub fn from_label(
        label: &str,
        confidence: f32,
        start_time: f32,
        end_time: f32,
        file_path: PathBuf,
    ) -> Self {
        let (scientific_name, common_name) = label.find('_').map_or_else(
            || (label.to_string(), label.to_string()),
            |idx| (label[..idx].to_string(), label[idx + 1..].to_string()),
        );

        Self {
            file_path,
            start_time,
            end_time,
            scientific_name,
            common_name,
            confidence,
            metadata: DetectionMetadata::default(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_from_label() {
        let detection = Detection::from_label(
            "Passer domesticus_House Sparrow",
            0.95,
            0.0,
            3.0,
            PathBuf::from("test.wav"),
        );
        assert_eq!(detection.scientific_name, "Passer domesticus");
        assert_eq!(detection.common_name, "House Sparrow");
        assert_eq!(detection.confidence, 0.95);
    }

    #[test]
    fn test_detection_from_label_no_underscore() {
        let detection =
            Detection::from_label("Unknown Species", 0.5, 0.0, 3.0, PathBuf::from("test.wav"));
        assert_eq!(detection.scientific_name, "Unknown Species");
        assert_eq!(detection.common_name, "Unknown Species");
    }
}
