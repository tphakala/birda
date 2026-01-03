//! Wrapper around birdnet-onnx RangeFilter.

use birdnet_onnx::{LocationScore, Prediction, RangeFilter as BirdnetRangeFilter};
use crate::error::{Error, Result};
use std::path::Path;

/// Wrapper around birdnet-onnx RangeFilter.
pub struct RangeFilter {
    inner: BirdnetRangeFilter,
}

impl RangeFilter {
    /// Build a range filter from configuration using classifier labels.
    pub fn from_config(
        meta_model_path: &Path,
        classifier_labels: &[String],
        threshold: f32,
    ) -> Result<Self> {
        let inner = BirdnetRangeFilter::builder()
            .model_path(meta_model_path.to_string_lossy().to_string())
            .from_classifier_labels(classifier_labels)
            .threshold(threshold)
            .build()
            .map_err(|e| Error::RangeFilterBuild {
                reason: e.to_string(),
            })?;

        Ok(Self { inner })
    }

    /// Get location scores for species at given coordinates and date.
    pub fn predict(
        &self,
        latitude: f64,
        longitude: f64,
        month: u32,
        day: u32,
    ) -> Result<Vec<LocationScore>> {
        #[allow(clippy::cast_possible_truncation)]
        self.inner
            .predict(latitude as f32, longitude as f32, month, day)
            .map_err(|e| Error::RangeFilterPredict {
                reason: e.to_string(),
            })
    }

    /// Filter predictions using location scores.
    /// Uses library's built-in filtering logic.
    pub fn filter_predictions(
        &self,
        predictions: &[Prediction],
        location_scores: &[LocationScore],
        rerank: bool,
    ) -> Vec<Prediction> {
        self.inner
            .filter_predictions(predictions, location_scores, rerank)
    }

    /// Filter multiple prediction sets efficiently.
    /// Useful for batch processing.
    pub fn filter_batch_predictions(
        &self,
        predictions: Vec<Vec<Prediction>>,
        location_scores: &[LocationScore],
        rerank: bool,
    ) -> Vec<Vec<Prediction>> {
        self.inner
            .filter_batch_predictions(predictions, location_scores, rerank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require actual model files
    // These are placeholder unit tests for structure

    #[test]
    fn test_range_filter_struct_exists() {
        // Just verify the struct compiles
        // Real tests will be integration tests with actual models
    }
}
