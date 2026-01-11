//! Detection grouping and merging.

use super::ParsedDetection;

/// A group of detections for the same species that overlap temporally.
#[derive(Debug, Clone)]
pub struct DetectionGroup {
    /// The species scientific name.
    pub scientific_name: String,
    /// The species common name.
    pub common_name: String,
    /// Start time of the merged clip (including padding).
    pub start: f64,
    /// End time of the merged clip (including padding).
    pub end: f64,
    /// Maximum confidence among grouped detections.
    pub max_confidence: f32,
    /// Number of detections in this group.
    pub detection_count: usize,
}

/// Group detections by species and merge overlapping time ranges.
#[allow(clippy::needless_pass_by_value, clippy::todo)]
pub fn group_detections(
    _detections: Vec<ParsedDetection>,
    _pre_padding: f64,
    _post_padding: f64,
) -> Vec<DetectionGroup> {
    todo!()
}
