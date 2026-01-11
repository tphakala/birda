//! JSON output format writer.

use crate::error::Result;
use crate::output::{Detection, OutputWriter};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// JSON result file structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonResultFile {
    /// Source audio file name.
    pub source_file: String,
    /// Analysis timestamp.
    pub analysis_date: DateTime<Utc>,
    /// Model used for analysis.
    pub model: String,
    /// Analysis settings.
    pub settings: JsonSettings,
    /// Detection results.
    pub detections: Vec<JsonDetection>,
    /// Summary statistics.
    pub summary: JsonSummary,
}

/// Analysis settings for JSON output.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonSettings {
    /// Minimum confidence threshold.
    pub min_confidence: f32,
    /// Segment overlap.
    pub overlap: f32,
    /// Latitude (if range filtering).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lat: Option<f64>,
    /// Longitude (if range filtering).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lon: Option<f64>,
    /// Week number (if range filtering).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub week: Option<u8>,
}

/// Single detection in JSON format.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonDetection {
    /// Start time in seconds.
    pub start_time: f32,
    /// End time in seconds.
    pub end_time: f32,
    /// Scientific name.
    pub scientific_name: String,
    /// Common name.
    pub common_name: String,
    /// Confidence score.
    pub confidence: f32,
}

/// Summary statistics.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonSummary {
    /// Total number of detections.
    pub total_detections: usize,
    /// Number of unique species.
    pub unique_species: usize,
    /// Audio duration in seconds.
    pub audio_duration_seconds: f32,
}

/// Writer for JSON detection output files.
pub struct JsonResultWriter {
    /// Collected detections.
    detections: Vec<Detection>,
    /// Output file path.
    output_path: PathBuf,
    /// Source file name.
    source_file: String,
    /// Model name.
    model: String,
    /// Analysis settings.
    min_confidence: f32,
    /// Overlap setting.
    overlap: f32,
    /// Latitude.
    lat: Option<f64>,
    /// Longitude.
    lon: Option<f64>,
    /// Week.
    week: Option<u8>,
    /// Audio file duration in seconds (actual, not derived from detections).
    audio_duration: f32,
}

impl JsonResultWriter {
    /// Create a new JSON result writer.
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path to write the JSON file
    /// * `source_file` - Name of the source audio file
    /// * `audio_duration` - Actual duration of the audio file in seconds
    /// * `model` - Model name used for analysis
    /// * `min_confidence` - Minimum confidence threshold used
    /// * `overlap` - Segment overlap used
    /// * `lat` - Latitude for range filtering (if used)
    /// * `lon` - Longitude for range filtering (if used)
    /// * `week` - Week number for range filtering (if used)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        output_path: &Path,
        source_file: &str,
        audio_duration: f32,
        model: &str,
        min_confidence: f32,
        overlap: f32,
        lat: Option<f64>,
        lon: Option<f64>,
        week: Option<u8>,
    ) -> Result<Self> {
        Ok(Self {
            detections: Vec::new(),
            output_path: output_path.to_path_buf(),
            source_file: source_file.to_string(),
            model: model.to_string(),
            min_confidence,
            overlap,
            lat,
            lon,
            week,
            audio_duration,
        })
    }

    /// Compute summary from detections.
    fn compute_summary(&self) -> JsonSummary {
        let unique_species: HashSet<&str> = self
            .detections
            .iter()
            .map(|d| d.scientific_name.as_str())
            .collect();

        JsonSummary {
            total_detections: self.detections.len(),
            unique_species: unique_species.len(),
            audio_duration_seconds: self.audio_duration,
        }
    }
}

impl OutputWriter for JsonResultWriter {
    fn write_header(&mut self) -> Result<()> {
        // No header for JSON - written at finalize
        Ok(())
    }

    fn write_detection(&mut self, detection: &Detection) -> Result<()> {
        self.detections.push(detection.clone());
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        let json_detections: Vec<JsonDetection> = self
            .detections
            .iter()
            .map(|d| JsonDetection {
                start_time: d.start_time,
                end_time: d.end_time,
                scientific_name: d.scientific_name.clone(),
                common_name: d.common_name.clone(),
                confidence: d.confidence,
            })
            .collect();

        let result = JsonResultFile {
            source_file: self.source_file.clone(),
            analysis_date: Utc::now(),
            model: self.model.clone(),
            settings: JsonSettings {
                min_confidence: self.min_confidence,
                overlap: self.overlap,
                lat: self.lat,
                lon: self.lon,
                week: self.week,
            },
            detections: json_detections,
            summary: self.compute_summary(),
        };

        let file = File::create(&self.output_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &result).map_err(|e| {
            crate::error::Error::JsonWrite {
                path: self.output_path.clone(),
                source: e,
            }
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_json_writer_basic() {
        let dir = tempdir().expect("create temp dir");
        let output_path = dir.path().join("test.BirdNET.json");

        let mut writer = JsonResultWriter::new(
            &output_path,
            "test.wav",
            60.0, // audio_duration
            "birdnet-v24",
            0.1,
            0.0,
            None,
            None,
            None,
        )
        .expect("create writer");

        writer.write_header().expect("write header");

        let detection = Detection::from_label(
            "Passer domesticus_House Sparrow",
            0.95,
            0.0,
            3.0,
            PathBuf::from("test.wav"),
        );
        writer.write_detection(&detection).expect("write detection");

        writer.finalize().expect("finalize");

        // Verify file was created and is valid JSON
        let content = std::fs::read_to_string(&output_path).expect("read file");
        let result: JsonResultFile = serde_json::from_str(&content).expect("parse JSON");

        assert_eq!(result.source_file, "test.wav");
        assert_eq!(result.model, "birdnet-v24");
        assert_eq!(result.detections.len(), 1);
        assert_eq!(result.detections[0].scientific_name, "Passer domesticus");
        assert_eq!(result.summary.total_detections, 1);
        assert_eq!(result.summary.unique_species, 1);
        assert!((result.summary.audio_duration_seconds - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_json_summary_unique_species() {
        let dir = tempdir().expect("create temp dir");
        let output_path = dir.path().join("test.BirdNET.json");

        let mut writer = JsonResultWriter::new(
            &output_path,
            "test.wav",
            60.0, // audio_duration
            "birdnet-v24",
            0.1,
            0.0,
            Some(45.0),
            Some(-73.0),
            Some(24),
        )
        .expect("create writer");

        writer.write_header().expect("write header");

        // Add multiple detections, some same species
        let d1 = Detection::from_label(
            "Passer domesticus_House Sparrow",
            0.95,
            0.0,
            3.0,
            PathBuf::from("test.wav"),
        );
        let d2 = Detection::from_label(
            "Turdus migratorius_American Robin",
            0.87,
            15.0,
            18.0,
            PathBuf::from("test.wav"),
        );
        let d3 = Detection::from_label(
            "Passer domesticus_House Sparrow",
            0.92,
            30.0,
            33.0,
            PathBuf::from("test.wav"),
        );

        writer.write_detection(&d1).expect("write d1");
        writer.write_detection(&d2).expect("write d2");
        writer.write_detection(&d3).expect("write d3");

        writer.finalize().expect("finalize");

        let content = std::fs::read_to_string(&output_path).expect("read file");
        let result: JsonResultFile = serde_json::from_str(&content).expect("parse JSON");

        assert_eq!(result.summary.total_detections, 3);
        assert_eq!(result.summary.unique_species, 2);
        assert!(result.settings.lat.is_some());
        assert_eq!(result.settings.lat, Some(45.0));
    }
}
