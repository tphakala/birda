//! Detection grouping and merging.
//!
//! Groups detections by species and merges overlapping time ranges
//! to produce consolidated clip regions.

use std::collections::HashMap;

use super::ParsedDetection;

/// A group of detections for the same species that overlap temporally.
#[derive(Debug, Clone)]
pub struct DetectionGroup {
    /// The species scientific name.
    pub scientific_name: String,
    /// The species common name.
    pub common_name: String,
    /// Start time of the merged clip (including padding, clamped to 0).
    pub start: f64,
    /// End time of the merged clip (including padding).
    pub end: f64,
    /// Maximum confidence among grouped detections.
    pub max_confidence: f32,
    /// Number of detections in this group.
    pub detection_count: usize,
}

/// A time range that can be merged with overlapping ranges.
#[derive(Debug, Clone)]
struct TimeRange {
    start: f64,
    end: f64,
    max_confidence: f32,
    detection_count: usize,
}

impl TimeRange {
    fn new(start: f64, end: f64, confidence: f32) -> Self {
        Self {
            start,
            end,
            max_confidence: confidence,
            detection_count: 1,
        }
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    fn merge(&mut self, other: &Self) {
        self.start = self.start.min(other.start);
        self.end = self.end.max(other.end);
        self.max_confidence = self.max_confidence.max(other.max_confidence);
        self.detection_count += other.detection_count;
    }
}

/// Group detections by species and merge overlapping time ranges.
///
/// Detections for the same species that overlap (considering padding) are
/// merged into a single clip region. The resulting groups are sorted by
/// start time.
///
/// # Arguments
///
/// * `detections` - List of parsed detections
/// * `pre_padding` - Seconds to add before each detection
/// * `post_padding` - Seconds to add after each detection
///
/// # Returns
///
/// Sorted list of detection groups with merged time ranges.
#[must_use]
pub fn group_detections(
    detections: Vec<ParsedDetection>,
    pre_padding: f64,
    post_padding: f64,
) -> Vec<DetectionGroup> {
    // Group by species first
    let mut species_detections: HashMap<String, Vec<ParsedDetection>> = HashMap::new();

    for detection in detections {
        species_detections
            .entry(detection.scientific_name.clone())
            .or_default()
            .push(detection);
    }

    let mut groups = Vec::new();

    for (scientific_name, mut detections) in species_detections {
        // Sort by start time
        detections.sort_unstable_by(|a, b| {
            a.start
                .partial_cmp(&b.start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Get common name from first detection
        let common_name = detections
            .first()
            .map(|d| d.common_name.clone())
            .unwrap_or_default();

        // Create time ranges with padding
        let ranges: Vec<TimeRange> = detections
            .iter()
            .map(|d| {
                let start = (d.start - pre_padding).max(0.0);
                let end = d.end + post_padding;
                TimeRange::new(start, end, d.confidence)
            })
            .collect();

        // Merge overlapping ranges (already sorted by start time above)
        let merged = merge_overlapping_ranges(&ranges);

        // Convert to groups
        for range in merged {
            groups.push(DetectionGroup {
                scientific_name: scientific_name.clone(),
                common_name: common_name.clone(),
                start: range.start,
                end: range.end,
                max_confidence: range.max_confidence,
                detection_count: range.detection_count,
            });
        }
    }

    // Sort all groups by start time
    groups.sort_unstable_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    groups
}

/// Merge overlapping time ranges into consolidated ranges.
///
/// Assumes ranges are already sorted by start time.
fn merge_overlapping_ranges(ranges: &[TimeRange]) -> Vec<TimeRange> {
    if ranges.is_empty() {
        return Vec::new();
    }

    let mut merged = Vec::new();
    let mut current = ranges[0].clone();

    for range in ranges.iter().skip(1) {
        if current.overlaps(range) {
            current.merge(range);
        } else {
            merged.push(current);
            current = range.clone();
        }
    }

    merged.push(current);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_range_overlaps() {
        let r1 = TimeRange::new(0.0, 5.0, 0.8);
        let r2 = TimeRange::new(4.0, 8.0, 0.9);
        let r3 = TimeRange::new(10.0, 15.0, 0.7);

        assert!(r1.overlaps(&r2));
        assert!(r2.overlaps(&r1));
        assert!(!r1.overlaps(&r3));
    }

    #[test]
    fn test_time_range_merge() {
        let mut r1 = TimeRange::new(0.0, 5.0, 0.8);
        let r2 = TimeRange::new(4.0, 8.0, 0.9);

        r1.merge(&r2);

        assert_eq!(r1.start, 0.0);
        assert_eq!(r1.end, 8.0);
        assert!((r1.max_confidence - 0.9).abs() < 0.001);
        assert_eq!(r1.detection_count, 2);
    }
}
