//! Applying `BirdNET` Geomodel occurrence scores to predictions.
//!
//! Filtering lives here rather than delegating to
//! `birdnet_onnx::RangeFilter::filter_predictions`, whose implementation drops
//! every species absent from the score map. That makes the `keep` policy, where
//! a species with no geomodel entry survives, impossible to express upstream.

use crate::config::UnmatchedPolicy;
use crate::inference::geomodel::GeomodelScores;
use birdnet_onnx::Prediction;

/// How a set of predictions should be filtered against geomodel scores.
#[derive(Debug, Clone, Copy)]
pub struct FilterSettings {
    /// Minimum occurrence score for a species to be considered present.
    pub threshold: f32,
    /// What to do with species that have no geomodel entry.
    pub unmatched: UnmatchedPolicy,
    /// Multiply confidence by occurrence score and re-sort.
    pub rerank: bool,
}

impl FilterSettings {
    /// Whether species with no geomodel entry survive filtering.
    ///
    /// Reranking always drops them, whatever the policy says. Reranking
    /// computes `confidence * P(species present)`, and a species with no
    /// geomodel entry has no such term. Substituting 1.0 would hand it the
    /// maximum possible prior, so the species we know least about would
    /// systematically outrank well-supported in-range ones; any other constant
    /// would be invented. See `keeps_unmatched` callers for the warning path.
    const fn keeps_unmatched(self) -> bool {
        matches!(self.unmatched, UnmatchedPolicy::Keep) && !self.rerank
    }
}

/// Filter predictions against geomodel occurrence scores.
///
/// | | score >= threshold | score < threshold | no geomodel entry |
/// |---|---|---|---|
/// | rerank off, keep | keep | drop | keep, confidence untouched |
/// | rerank off, drop | keep | drop | drop |
/// | rerank on | keep, scaled | drop | drop |
#[must_use]
pub fn filter_predictions(
    predictions: &[Prediction],
    scores: &GeomodelScores,
    settings: FilterSettings,
) -> Vec<Prediction> {
    let keeps_unmatched = settings.keeps_unmatched();

    let mut filtered: Vec<Prediction> = predictions
        .iter()
        .filter_map(|prediction| match scores.score_of(&prediction.species) {
            Some(score) if score >= settings.threshold => {
                let confidence = if settings.rerank {
                    prediction.confidence * score
                } else {
                    prediction.confidence
                };
                Some(Prediction {
                    species: prediction.species.clone(),
                    confidence,
                    index: prediction.index,
                })
            }
            // In range data, but not expected here at this time of year.
            Some(_) => None,
            // No range data for this species at all.
            None => keeps_unmatched.then(|| prediction.clone()),
        })
        .collect();

    if settings.rerank {
        filtered.sort_unstable_by(|a, b| b.confidence.total_cmp(&a.confidence));
    }

    filtered
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
#[allow(clippy::unwrap_used)] // Test setup code - panics are acceptable
mod tests {
    use super::*;
    use crate::inference::geomodel::SpeciesMapping;
    use birdnet_onnx::LocationScore;

    fn prediction(species: &str, confidence: f32) -> Prediction {
        Prediction {
            species: species.to_string(),
            confidence,
            index: 0,
        }
    }

    /// Build scores where every listed species is both mapped and scored.
    fn scores_of(entries: &[(&str, f32)]) -> GeomodelScores {
        let labels: Vec<String> = entries.iter().map(|(s, _)| (*s).to_string()).collect();
        let mapping = SpeciesMapping::build(&labels, &labels);
        let location_scores: Vec<LocationScore> = entries
            .iter()
            .enumerate()
            .map(|(index, (species, score))| LocationScore {
                species: (*species).to_string(),
                score: *score,
                index,
            })
            .collect();
        GeomodelScores::project(&location_scores, &mapping)
    }

    fn settings(unmatched: UnmatchedPolicy, rerank: bool) -> FilterSettings {
        FilterSettings {
            threshold: 0.01,
            unmatched,
            rerank,
        }
    }

    #[test]
    fn test_keeps_mapped_species_above_threshold() {
        let scores = scores_of(&[("Parus major_x", 0.5)]);

        let out = filter_predictions(
            &[prediction("Parus major_x", 0.8)],
            &scores,
            settings(UnmatchedPolicy::Keep, false),
        );

        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].confidence, 0.8,
            "confidence must be untouched without rerank"
        );
    }

    #[test]
    fn test_drops_mapped_species_below_threshold() {
        let scores = scores_of(&[("Parus major_x", 0.005)]);

        let out = filter_predictions(
            &[prediction("Parus major_x", 0.9)],
            &scores,
            settings(UnmatchedPolicy::Keep, false),
        );

        assert!(out.is_empty());
    }

    #[test]
    fn test_keeps_species_exactly_at_threshold() {
        let scores = scores_of(&[("Parus major_x", 0.01)]);

        let out = filter_predictions(
            &[prediction("Parus major_x", 0.9)],
            &scores,
            settings(UnmatchedPolicy::Keep, false),
        );

        assert_eq!(out.len(), 1, "the threshold is inclusive");
    }

    #[test]
    fn test_keep_policy_passes_unmatched_species_through() {
        let scores = scores_of(&[("Parus major_x", 0.5)]);

        let out = filter_predictions(
            &[prediction("Dog_Dog", 0.7)],
            &scores,
            settings(UnmatchedPolicy::Keep, false),
        );

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].confidence, 0.7);
    }

    #[test]
    fn test_drop_policy_removes_unmatched_species() {
        let scores = scores_of(&[("Parus major_x", 0.5)]);

        let out = filter_predictions(
            &[prediction("Dog_Dog", 0.7)],
            &scores,
            settings(UnmatchedPolicy::Drop, false),
        );

        assert!(out.is_empty());
    }

    #[test]
    fn test_rerank_scales_confidence_by_score() {
        let scores = scores_of(&[("Parus major_x", 0.5)]);

        let out = filter_predictions(
            &[prediction("Parus major_x", 0.8)],
            &scores,
            settings(UnmatchedPolicy::Keep, true),
        );

        assert!((out[0].confidence - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_rerank_drops_unmatched_even_under_keep_policy() {
        // Reranking has no occurrence probability to weight an unmatched
        // species by, and 1.0 would make it outrank everything in range.
        let scores = scores_of(&[("Parus major_x", 0.5)]);

        let out = filter_predictions(
            &[prediction("Parus major_x", 0.8), prediction("Dog_Dog", 0.9)],
            &scores,
            settings(UnmatchedPolicy::Keep, true),
        );

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].species, "Parus major_x");
    }

    #[test]
    fn test_rerank_orders_plausible_species_above_implausible_one() {
        // The whole point of rerank: a slightly less confident but far more
        // likely species must outrank a confident but out-of-place one.
        let scores = scores_of(&[("Parus major_x", 0.9), ("Rara avis_y", 0.02)]);

        let out = filter_predictions(
            &[
                prediction("Rara avis_y", 0.80),
                prediction("Parus major_x", 0.70),
            ],
            &scores,
            settings(UnmatchedPolicy::Keep, true),
        );

        assert_eq!(
            out[0].species, "Parus major_x",
            "0.70 * 0.9 must outrank 0.80 * 0.02"
        );
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_no_rerank_preserves_input_order() {
        let scores = scores_of(&[("Aaa aaa_x", 0.2), ("Bbb bbb_y", 0.9)]);

        let out = filter_predictions(
            &[prediction("Aaa aaa_x", 0.5), prediction("Bbb bbb_y", 0.4)],
            &scores,
            settings(UnmatchedPolicy::Keep, false),
        );

        assert_eq!(out[0].species, "Aaa aaa_x");
        assert_eq!(out[1].species, "Bbb bbb_y");
    }

    #[test]
    fn test_prediction_index_survives_filtering() {
        let scores = scores_of(&[("Parus major_x", 0.5)]);
        let prediction = Prediction {
            species: "Parus major_x".to_string(),
            confidence: 0.8,
            index: 42,
        };

        let out = filter_predictions(
            &[prediction],
            &scores,
            settings(UnmatchedPolicy::Keep, true),
        );

        assert_eq!(out[0].index, 42);
    }

    #[test]
    fn test_empty_predictions_yield_empty_output() {
        let scores = scores_of(&[("Aaa aaa_x", 0.9)]);

        let out = filter_predictions(&[], &scores, settings(UnmatchedPolicy::Keep, true));

        assert!(out.is_empty());
    }

    #[test]
    fn test_every_species_unmatched_under_keep_is_a_passthrough() {
        // Guards the case where the geomodel and classifier share nothing:
        // filtering must not silently delete every detection.
        let scores = scores_of(&[]);

        let out = filter_predictions(
            &[prediction("Dog_Dog", 0.7), prediction("Siren_Siren", 0.6)],
            &scores,
            settings(UnmatchedPolicy::Keep, false),
        );

        assert_eq!(out.len(), 2);
    }
}
