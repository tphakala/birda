//! End-to-end tests for the BirdNET Geomodel range filter path.
//!
//! These run against a tiny fixture ONNX with the same input and output
//! contract as the real geomodel (3 float32 inputs, N sigmoid outputs), so the
//! whole chain (build filter, predict, map, project, filter) is exercised in CI
//! without downloading 14.7 MB. See `tests/fixtures/make_fixture_geomodel.py`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use birda::config::UnmatchedPolicy;
use birda::inference::range_filter::RangeFilter;
use birda::inference::{FilterSettings, GeomodelScores, SpeciesMapping, filter_predictions};
use birdnet_onnx::Prediction;
use std::path::PathBuf;

/// Helsinki, used as a stable query point.
const TEST_LAT: f64 = 60.1699;
const TEST_LON: f64 = 24.9384;
const TEST_MONTH: u32 = 6;
const TEST_DAY: u32 = 15;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn fixture_labels() -> Vec<String> {
    std::fs::read_to_string(fixture_path("fixture-geomodel-labels.txt"))
        .expect("fixture labels must exist")
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect()
}

fn fixture_filter(labels: &[String]) -> RangeFilter {
    RangeFilter::from_config(&fixture_path("fixture-geomodel.onnx"), labels, 0.0)
        .expect("fixture geomodel must load")
}

fn prediction(species: &str, confidence: f32) -> Prediction {
    Prediction {
        species: species.to_string(),
        confidence,
        index: 0,
    }
}

#[test]
fn test_zero_threshold_returns_every_class() {
    // birda queries the geomodel at threshold 0.0 and thresholds afterwards, so
    // that every mapped species has a score and "no range data" stays
    // distinguishable from "out of range".
    let labels = fixture_labels();
    let filter = fixture_filter(&labels);

    let scores = filter
        .predict(TEST_LAT, TEST_LON, TEST_MONTH, TEST_DAY)
        .unwrap();

    assert_eq!(scores.len(), labels.len());
}

#[test]
fn test_classifier_labels_are_rejected_as_geomodel_labels() {
    // Regression guard for the failure this whole design exists to avoid:
    // building the filter from the classifier's labels fails birdnet-onnx's
    // label-count validation, because no classifier has the geomodel's classes.
    let classifier_labels = vec!["Parus major_Great Tit".to_string()];

    let result = RangeFilter::from_config(
        &fixture_path("fixture-geomodel.onnx"),
        &classifier_labels,
        0.0,
    );

    assert!(
        result.is_err(),
        "a 1-label set must not build against a 5-output model"
    );
}

#[test]
fn test_scores_project_onto_localized_classifier_labels() {
    // The geomodel ships English labels only, so a localized classifier label
    // must still receive its score.
    let geomodel_labels = fixture_labels();
    let filter = fixture_filter(&geomodel_labels);
    let scores = filter
        .predict(TEST_LAT, TEST_LON, TEST_MONTH, TEST_DAY)
        .unwrap();

    let classifier_labels = vec![
        "Parus major_Talitiainen".to_string(),
        "Cyanistes caeruleus_Sinitiainen".to_string(),
        "Dog_Dog".to_string(),
    ];
    let mapping = SpeciesMapping::build(&geomodel_labels, &classifier_labels);

    assert_eq!(mapping.mapped_count(), 2);
    assert_eq!(mapping.unmatched_count(), 1);

    let projected = GeomodelScores::project(&scores, &mapping);

    assert!(projected.score_of("Parus major_Talitiainen").is_some());
    assert!(
        projected
            .score_of("Cyanistes caeruleus_Sinitiainen")
            .is_some()
    );
    assert!(
        projected.score_of("Dog_Dog").is_none(),
        "a species with no geomodel entry must have no score"
    );
}

#[test]
fn test_keep_policy_passes_unmatched_species_through_end_to_end() {
    let geomodel_labels = fixture_labels();
    let filter = fixture_filter(&geomodel_labels);
    let scores = filter
        .predict(TEST_LAT, TEST_LON, TEST_MONTH, TEST_DAY)
        .unwrap();

    let classifier_labels = vec!["Parus major_Great Tit".to_string(), "Dog_Dog".to_string()];
    let mapping = SpeciesMapping::build(&geomodel_labels, &classifier_labels);
    let projected = GeomodelScores::project(&scores, &mapping);

    let filtered = filter_predictions(
        &[prediction("Dog_Dog", 0.7)],
        &projected,
        FilterSettings {
            threshold: 0.01,
            unmatched: UnmatchedPolicy::Keep,
            rerank: false,
        },
    );

    assert_eq!(
        filtered.len(),
        1,
        "non-species labels must survive the default policy"
    );
}

#[test]
fn test_drop_policy_removes_unmatched_species_end_to_end() {
    let geomodel_labels = fixture_labels();
    let filter = fixture_filter(&geomodel_labels);
    let scores = filter
        .predict(TEST_LAT, TEST_LON, TEST_MONTH, TEST_DAY)
        .unwrap();

    let classifier_labels = vec!["Parus major_Great Tit".to_string(), "Dog_Dog".to_string()];
    let mapping = SpeciesMapping::build(&geomodel_labels, &classifier_labels);
    let projected = GeomodelScores::project(&scores, &mapping);

    let filtered = filter_predictions(
        &[prediction("Dog_Dog", 0.7)],
        &projected,
        FilterSettings {
            threshold: 0.01,
            unmatched: UnmatchedPolicy::Drop,
            rerank: false,
        },
    );

    assert!(filtered.is_empty());
}

#[test]
fn test_out_of_range_species_is_filtered_end_to_end() {
    // The fixture gives species index 3 a large negative bias, so its sigmoid
    // score sits far below any sensible threshold at this query point.
    let geomodel_labels = fixture_labels();
    let filter = fixture_filter(&geomodel_labels);
    let scores = filter
        .predict(TEST_LAT, TEST_LON, TEST_MONTH, TEST_DAY)
        .unwrap();

    let mapping = SpeciesMapping::build(&geomodel_labels, &geomodel_labels);
    let projected = GeomodelScores::project(&scores, &mapping);

    let unlikely = "Turdus merula_Common Blackbird";
    let score = projected
        .score_of(unlikely)
        .expect("mapped species must have a score");
    assert!(
        score < 0.01,
        "fixture must keep this species below threshold, got {score}"
    );

    let filtered = filter_predictions(
        &[prediction(unlikely, 0.95)],
        &projected,
        FilterSettings {
            threshold: 0.01,
            unmatched: UnmatchedPolicy::Keep,
            rerank: false,
        },
    );

    assert!(
        filtered.is_empty(),
        "a confident detection of an out-of-range species must be filtered"
    );
}

#[test]
fn test_rerank_scales_confidence_by_the_predicted_score() {
    let geomodel_labels = fixture_labels();
    let filter = fixture_filter(&geomodel_labels);
    let scores = filter
        .predict(TEST_LAT, TEST_LON, TEST_MONTH, TEST_DAY)
        .unwrap();

    let mapping = SpeciesMapping::build(&geomodel_labels, &geomodel_labels);
    let projected = GeomodelScores::project(&scores, &mapping);

    let species = "Parus major_Great Tit";
    let score = projected.score_of(species).unwrap();

    let filtered = filter_predictions(
        &[prediction(species, 0.8)],
        &projected,
        FilterSettings {
            threshold: 0.01,
            unmatched: UnmatchedPolicy::Keep,
            rerank: true,
        },
    );

    assert_eq!(filtered.len(), 1);
    assert!((filtered[0].confidence - 0.8 * score).abs() < 1e-6);
}

#[test]
fn test_a_different_location_produces_different_scores() {
    // Guards against the query parameters being ignored, which would make the
    // filter a constant and silently useless.
    let labels = fixture_labels();
    let filter = fixture_filter(&labels);

    let helsinki = filter
        .predict(TEST_LAT, TEST_LON, TEST_MONTH, TEST_DAY)
        .unwrap();
    let patagonia = filter.predict(-51.6, -69.2, TEST_MONTH, TEST_DAY).unwrap();

    let differs = helsinki
        .iter()
        .zip(patagonia.iter())
        .any(|(a, b)| (a.score - b.score).abs() > 1e-6);

    assert!(differs, "coordinates must affect the predicted scores");
}
