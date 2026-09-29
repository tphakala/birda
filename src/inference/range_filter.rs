//! Wrapper around birdnet-onnx `RangeFilter`.

use crate::constants::range_filter::GEOMODEL_SPECIES_COUNT;
use crate::error::{Error, Result};
use birdnet_onnx::{LocationScore, RangeFilter as BirdnetRangeFilter};
use std::path::Path;

/// Check that a geomodel labels file matches the model's output size.
///
/// A mismatch means the labels and the ONNX file came from different versions,
/// which `birdnet_onnx` would otherwise report as a bare label-count error.
fn validate_geomodel_labels(labels: &[String], expected: usize) -> Result<()> {
    if labels.len() == expected {
        return Ok(());
    }

    Err(Error::GeomodelLabelCount {
        expected,
        actual: labels.len(),
    })
}

/// Read the geomodel's labels file and check it holds the geomodel's species.
///
/// Every route that builds a [`RangeFilter`] reads the labels through here, so
/// the empty-file guard and the label-count check apply to `analyze` and
/// `species` alike, and both report the purpose-built error instead of a raw
/// `birdnet_onnx` one.
///
/// # Errors
/// The errors of [`crate::utils::labels::read_label_lines`], and
/// [`Error::GeomodelLabelCount`] when the file does not list
/// [`GEOMODEL_SPECIES_COUNT`] species.
pub fn read_geomodel_labels(path: &Path) -> Result<Vec<String>> {
    let labels = crate::utils::labels::read_label_lines(path)?;
    validate_geomodel_labels(&labels, GEOMODEL_SPECIES_COUNT)?;
    Ok(labels)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(count: usize) -> Vec<String> {
        (0..count)
            .map(|i| format!("Genus species{i}_Name"))
            .collect()
    }

    #[test]
    fn test_validate_accepts_exactly_the_expected_count() {
        assert!(validate_geomodel_labels(&labels(5), 5).is_ok());
    }

    #[test]
    fn test_validate_reports_both_counts_on_mismatch() {
        // A version-mismatched labels file: the message must carry the count the
        // registry declares and the count found, so the user can see which way
        // the mismatch goes.
        let err = validate_geomodel_labels(&labels(6_522), GEOMODEL_SPECIES_COUNT).unwrap_err();

        assert!(
            matches!(
                err,
                Error::GeomodelLabelCount {
                    expected: 12_012,
                    actual: 6_522
                }
            ),
            "got: {err:?}"
        );
    }

    #[test]
    fn test_read_geomodel_labels_rejects_a_classifier_sized_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.txt");
        std::fs::write(&path, labels(3).join("\n")).unwrap();

        let err = read_geomodel_labels(&path).unwrap_err();

        assert!(
            matches!(
                err,
                Error::GeomodelLabelCount {
                    expected: 12_012,
                    actual: 3
                }
            ),
            "got: {err:?}"
        );
    }

    #[test]
    fn test_read_geomodel_labels_accepts_a_full_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.txt");
        std::fs::write(&path, labels(GEOMODEL_SPECIES_COUNT).join("\n")).unwrap();

        assert_eq!(
            read_geomodel_labels(&path).unwrap().len(),
            GEOMODEL_SPECIES_COUNT
        );
    }

    #[test]
    fn test_read_geomodel_labels_reports_an_empty_file_as_such() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.txt");
        std::fs::write(&path, "").unwrap();

        let err = read_geomodel_labels(&path).unwrap_err();

        assert!(
            matches!(&err, Error::LabelLoad { reason, .. } if reason == "file contains no labels"),
            "got: {err:?}"
        );
    }
}
