//! Wrapper around birdnet-onnx `RangeFilter`.

use crate::error::{Error, Result};
use birdnet_onnx::{LocationScore, RangeFilter as BirdnetRangeFilter};
use std::path::Path;

/// Wrapper around birdnet-onnx `RangeFilter`.
pub struct RangeFilter {
    inner: BirdnetRangeFilter,
}

impl RangeFilter {
    /// Build a range filter from the geomodel and ITS OWN labels.
    ///
    /// `geomodel_labels` must be the geomodel's label set, never a
    /// classifier's: birdnet-onnx validates that the label count equals the
    /// model's output size, and no classifier has the geomodel's 12,012
    /// classes. Scores are projected into a classifier's label space
    /// afterwards, by `crate::inference::geomodel`.
    pub fn from_config(
        geomodel_path: &Path,
        geomodel_labels: &[String],
        threshold: f32,
    ) -> Result<Self> {
        let inner = BirdnetRangeFilter::builder()
            .model_path(geomodel_path.to_string_lossy().to_string())
            .from_classifier_labels(geomodel_labels)
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
}
