//! Tests for detection grouping.

use birda::clipper::{ParsedDetection, group_detections};

fn make_detection(start: f64, end: f64, species: &str, confidence: f32) -> ParsedDetection {
    ParsedDetection {
        start,
        end,
        scientific_name: species.to_string(),
        common_name: format!("{species} Common"),
        confidence,
    }
}

#[test]
fn test_group_single_detection() {
    let detections = vec![make_detection(0.0, 3.0, "Parus major", 0.85)];

    let groups = group_detections(detections, 0.0, 0.0);

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].scientific_name, "Parus major");
    assert_eq!(groups[0].start, 0.0);
    assert_eq!(groups[0].end, 3.0);
    assert_eq!(groups[0].detection_count, 1);
}

#[test]
fn test_group_overlapping_same_species() {
    let detections = vec![
        make_detection(0.0, 3.0, "Parus major", 0.85),
        make_detection(2.0, 5.0, "Parus major", 0.90),
        make_detection(4.0, 7.0, "Parus major", 0.75),
    ];

    let groups = group_detections(detections, 0.0, 0.0);

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].start, 0.0);
    assert_eq!(groups[0].end, 7.0);
    assert!((groups[0].max_confidence - 0.90).abs() < 0.001);
    assert_eq!(groups[0].detection_count, 3);
}

#[test]
fn test_group_with_padding_causes_merge() {
    let detections = vec![
        make_detection(0.0, 3.0, "Parus major", 0.85),
        make_detection(5.0, 8.0, "Parus major", 0.90), // Gap of 2s, but 3s padding merges them
    ];

    let groups = group_detections(detections, 3.0, 3.0);

    assert_eq!(groups.len(), 1);
    // With 3s padding: first clip 0-3 becomes -3 to 6, second 5-8 becomes 2-11
    // They overlap, so merged to -3 to 11, clamped to 0.0
    assert_eq!(groups[0].start, 0.0); // Clamped from -3.0
    assert_eq!(groups[0].end, 11.0);
}

#[test]
fn test_group_different_species_separate() {
    let detections = vec![
        make_detection(0.0, 3.0, "Parus major", 0.85),
        make_detection(1.0, 4.0, "Cyanistes caeruleus", 0.90),
    ];

    let groups = group_detections(detections, 0.0, 0.0);

    assert_eq!(groups.len(), 2);
}

#[test]
fn test_group_non_overlapping_same_species() {
    let detections = vec![
        make_detection(0.0, 3.0, "Parus major", 0.85),
        make_detection(10.0, 13.0, "Parus major", 0.90), // Not overlapping
    ];

    let groups = group_detections(detections, 0.0, 0.0);

    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].start, 0.0);
    assert_eq!(groups[1].start, 10.0);
}

#[test]
fn test_groups_sorted_by_start_time() {
    let detections = vec![
        make_detection(10.0, 13.0, "Parus major", 0.85),
        make_detection(0.0, 3.0, "Cyanistes caeruleus", 0.90),
        make_detection(5.0, 8.0, "Parus major", 0.75),
    ];

    let groups = group_detections(detections, 0.0, 0.0);

    assert!(groups[0].start <= groups[1].start);
    if groups.len() > 2 {
        assert!(groups[1].start <= groups[2].start);
    }
}
