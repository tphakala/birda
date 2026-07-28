//! Mapping between `BirdNET` Geomodel species and classifier labels.
//!
//! The geomodel scores 12,012 species, but every classifier has its own label
//! set: `BirdNET` v2.4 has 6,522, Perch v2 has 14,795, and neither is a subset
//! of the geomodel's. Worse, classifier labels are localized (birda ships 37
//! label languages for `BirdNET` v2.4) while the geomodel's are English only.
//!
//! Both sides are therefore keyed on the scientific name, lowercased, so
//! `Parus major_Great Tit` and `Parus major_Talitiainen` resolve to the same
//! species. This module is pure: it performs no I/O and touches no ONNX.

use birdnet_onnx::LocationScore;
use std::collections::HashMap;
use tracing::warn;

/// Extract the scientific name from a species label.
///
/// Labels come in two shapes: `Scientific name_Common name`, used by `BirdNET`
/// and by the geomodel, and a bare scientific name, used by Perch.
///
/// The part before the first underscore is treated as a scientific name only
/// when it contains a space. Every geomodel key is a binomial and so always
/// does, while Perch carries FSD50K sound classes such as
/// `Accelerating_and_revving_and_vroom`, which splitting would truncate to
/// `Accelerating`. Those labels have no geomodel entry either way, but leaving
/// them intact keeps them distinguishable from each other.
#[must_use]
pub fn scientific_name(label: &str) -> &str {
    match label.split_once('_') {
        Some((prefix, _)) if prefix.contains(' ') => prefix,
        _ => label,
    }
}

/// Lookup key for a species: its scientific name, case-folded.
fn species_key(label: &str) -> String {
    scientific_name(label).to_lowercase()
}

/// Mapping from a classifier's labels to geomodel species.
///
/// Built once at startup and then consulted per prediction batch.
#[derive(Debug, Clone)]
pub struct SpeciesMapping {
    /// Geomodel species key to the classifier label that species maps onto.
    by_species_key: HashMap<String, String>,
    /// Total labels in the classifier's label set.
    total_classifier_species: usize,
}

impl SpeciesMapping {
    /// Build the mapping between a geomodel label set and a classifier's.
    ///
    /// Two classifier labels can resolve to the same scientific name. The first
    /// wins and the collision is logged, since silently preferring one over the
    /// other would make range filtering depend on label file ordering.
    #[must_use]
    pub fn build(geomodel_labels: &[String], classifier_labels: &[String]) -> Self {
        let mut classifier_by_key: HashMap<String, &String> =
            HashMap::with_capacity(classifier_labels.len());

        for label in classifier_labels {
            let key = species_key(label);
            if let Some(existing) = classifier_by_key.get(&key) {
                warn!(
                    "Classifier labels '{}' and '{}' share the scientific name '{}'; \
                     range filtering will use the first",
                    existing, label, key
                );
            } else {
                classifier_by_key.insert(key, label);
            }
        }

        let mut by_species_key = HashMap::new();
        for geomodel_label in geomodel_labels {
            let key = species_key(geomodel_label);
            if let Some(classifier_label) = classifier_by_key.get(&key) {
                by_species_key.insert(key, (*classifier_label).clone());
            }
        }

        Self {
            by_species_key,
            total_classifier_species: classifier_labels.len(),
        }
    }

    /// Classifier label a geomodel label maps onto, if any.
    #[must_use]
    pub fn classifier_label_for(&self, geomodel_label: &str) -> Option<&str> {
        self.by_species_key
            .get(&species_key(geomodel_label))
            .map(String::as_str)
    }

    /// Number of classifier species that have a geomodel entry.
    #[must_use]
    pub fn mapped_count(&self) -> usize {
        self.by_species_key.len()
    }

    /// Number of classifier species with no geomodel entry.
    #[must_use]
    pub fn unmatched_count(&self) -> usize {
        self.total_classifier_species
            .saturating_sub(self.mapped_count())
    }

    /// Total labels in the classifier's label set.
    #[must_use]
    pub const fn total_classifier_species(&self) -> usize {
        self.total_classifier_species
    }

    /// Every classifier label that has a geomodel entry.
    fn mapped_classifier_labels(&self) -> impl Iterator<Item = &String> {
        self.by_species_key.values()
    }
}

/// Geomodel occurrence scores, projected into a classifier's label space.
///
/// A label is absent from this table exactly when the classifier species has no
/// geomodel entry. That distinction matters: "no range data" and "out of range"
/// are handled differently by the filter, so a mapped species scoring zero must
/// still have an entry.
#[derive(Debug, Clone, Default)]
pub struct GeomodelScores {
    by_classifier_label: HashMap<String, f32>,
}

impl GeomodelScores {
    /// Project raw geomodel scores into the classifier's label space.
    ///
    /// Every mapped classifier label receives an entry, defaulting to zero for
    /// species the geomodel did not report. Scores for geomodel species with no
    /// classifier counterpart are dropped: they cannot be predicted anyway.
    #[must_use]
    pub fn project(scores: &[LocationScore], mapping: &SpeciesMapping) -> Self {
        // Seed every mapped species at zero so a species the geomodel omitted
        // reads as "out of range" rather than "no range data".
        let mut by_classifier_label: HashMap<String, f32> = mapping
            .mapped_classifier_labels()
            .map(|label| (label.clone(), 0.0))
            .collect();

        for score in scores {
            if let Some(classifier_label) = mapping.classifier_label_for(&score.species) {
                by_classifier_label.insert(classifier_label.to_string(), score.score);
            }
        }

        Self {
            by_classifier_label,
        }
    }

    /// Occurrence score for a classifier label, or `None` when the species has
    /// no geomodel entry at all.
    #[must_use]
    pub fn score_of(&self, classifier_label: &str) -> Option<f32> {
        self.by_classifier_label.get(classifier_label).copied()
    }

    /// Number of mapped species scoring at or above `threshold`.
    #[must_use]
    pub fn in_range_count(&self, threshold: f32) -> usize {
        self.by_classifier_label
            .values()
            .filter(|score| **score >= threshold)
            .count()
    }

    /// Whether any species has range data.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_classifier_label.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
#[allow(clippy::unwrap_used)] // Test setup code - panics are acceptable
mod tests {
    use super::*;

    fn labels(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_string()).collect()
    }

    fn score(species: &str, score: f32, index: usize) -> LocationScore {
        LocationScore {
            species: species.to_string(),
            score,
            index,
        }
    }

    #[test]
    fn test_scientific_name_extracts_binomial_from_birdnet_label() {
        assert_eq!(scientific_name("Parus major_Great Tit"), "Parus major");
    }

    #[test]
    fn test_scientific_name_extracts_from_localized_label() {
        assert_eq!(scientific_name("Parus major_Talitiainen"), "Parus major");
    }

    #[test]
    fn test_scientific_name_passes_through_bare_binomial() {
        assert_eq!(scientific_name("Parus major"), "Parus major");
    }

    #[test]
    fn test_scientific_name_keeps_fsd50k_label_intact() {
        // Splitting on '_' unconditionally would truncate this to "Accelerating".
        let label = "Accelerating_and_revving_and_vroom";
        assert_eq!(scientific_name(label), label);
    }

    #[test]
    fn test_scientific_name_keeps_single_word_label_intact() {
        assert_eq!(scientific_name("Accordion"), "Accordion");
        assert_eq!(scientific_name("Dog_Dog"), "Dog_Dog");
    }

    #[test]
    fn test_scientific_name_splits_on_the_first_underscore_only() {
        assert_eq!(scientific_name("Parus major_Great_Tit"), "Parus major");
    }

    #[test]
    fn test_scientific_name_handles_an_empty_label() {
        assert_eq!(scientific_name(""), "");
    }

    #[test]
    fn test_mapping_matches_localized_classifier_labels() {
        // The geomodel ships English labels only. A Finnish BirdNET label must
        // still map, because matching is on the binomial, not the common name.
        let mapping = SpeciesMapping::build(
            &labels(&["Parus major_Great Tit"]),
            &labels(&["Parus major_Talitiainen"]),
        );

        assert_eq!(mapping.mapped_count(), 1);
        assert_eq!(mapping.unmatched_count(), 0);
        assert_eq!(
            mapping.classifier_label_for("Parus major_Great Tit"),
            Some("Parus major_Talitiainen")
        );
    }

    #[test]
    fn test_mapping_matches_bare_binomial_perch_labels() {
        let mapping = SpeciesMapping::build(
            &labels(&["Parus major_Great Tit"]),
            &labels(&["Parus major"]),
        );

        assert_eq!(mapping.mapped_count(), 1);
    }

    #[test]
    fn test_mapping_is_case_insensitive() {
        let mapping = SpeciesMapping::build(
            &labels(&["parus major_Great Tit"]),
            &labels(&["Parus Major_Talitiainen"]),
        );

        assert_eq!(mapping.mapped_count(), 1);
    }

    #[test]
    fn test_mapping_counts_unmatched_classifier_species() {
        let mapping = SpeciesMapping::build(
            &labels(&["Parus major_Great Tit"]),
            &labels(&[
                "Parus major_Great Tit",
                // eBird revision: BirdNET v2.4 still says Accipiter.
                "Accipiter gentilis_Northern Goshawk",
                // Non-species label, absent from any geomodel.
                "Dog_Dog",
            ]),
        );

        assert_eq!(mapping.mapped_count(), 1);
        assert_eq!(mapping.unmatched_count(), 2);
        assert_eq!(mapping.total_classifier_species(), 3);
    }

    #[test]
    fn test_mapping_ignores_geomodel_species_absent_from_the_classifier() {
        // The geomodel covers mammals, insects and amphibians that no bird
        // classifier predicts. Those must not inflate the mapped count.
        let mapping = SpeciesMapping::build(
            &labels(&[
                "Parus major_Great Tit",
                "Petaurista albiventer_White-bellied Giant Flying Squirrel",
            ]),
            &labels(&["Parus major_Great Tit"]),
        );

        assert_eq!(mapping.mapped_count(), 1);
        assert_eq!(mapping.unmatched_count(), 0);
    }

    #[test]
    fn test_mapping_keeps_the_first_of_two_colliding_classifier_labels() {
        let mapping = SpeciesMapping::build(
            &labels(&["Parus major_Great Tit"]),
            &labels(&["Parus major_First", "Parus major_Second"]),
        );

        assert_eq!(mapping.mapped_count(), 1);
        assert_eq!(
            mapping.classifier_label_for("Parus major_Great Tit"),
            Some("Parus major_First")
        );
    }

    #[test]
    fn test_mapping_of_empty_label_sets_is_empty() {
        let mapping = SpeciesMapping::build(&[], &[]);

        assert_eq!(mapping.mapped_count(), 0);
        assert_eq!(mapping.unmatched_count(), 0);
        assert_eq!(mapping.total_classifier_species(), 0);
    }

    #[test]
    fn test_projection_keys_by_classifier_label() {
        let mapping = SpeciesMapping::build(
            &labels(&["Parus major_Great Tit"]),
            &labels(&["Parus major_Talitiainen"]),
        );

        let projected =
            GeomodelScores::project(&[score("Parus major_Great Tit", 0.8, 0)], &mapping);

        assert_eq!(projected.score_of("Parus major_Talitiainen"), Some(0.8));
        assert_eq!(
            projected.score_of("Parus major_Great Tit"),
            None,
            "scores must be keyed by the classifier's label, not the geomodel's"
        );
    }

    #[test]
    fn test_projection_includes_mapped_species_the_geomodel_omitted() {
        // A mapped species must have an entry even with no reported score, so
        // the filter can tell "no range data" apart from "out of range".
        let mapping = SpeciesMapping::build(
            &labels(&["Parus major_Great Tit"]),
            &labels(&["Parus major_Great Tit"]),
        );

        let projected = GeomodelScores::project(&[], &mapping);

        assert_eq!(projected.score_of("Parus major_Great Tit"), Some(0.0));
    }

    #[test]
    fn test_projection_omits_unmatched_species() {
        let mapping =
            SpeciesMapping::build(&labels(&["Parus major_Great Tit"]), &labels(&["Dog_Dog"]));

        let projected = GeomodelScores::project(&[], &mapping);

        assert_eq!(projected.score_of("Dog_Dog"), None);
        assert!(projected.is_empty());
    }

    #[test]
    fn test_projection_drops_geomodel_species_with_no_classifier_match() {
        let mapping = SpeciesMapping::build(
            &labels(&["Parus major_Great Tit", "Vulpes vulpes_Red Fox"]),
            &labels(&["Parus major_Great Tit"]),
        );

        let projected = GeomodelScores::project(
            &[
                score("Parus major_Great Tit", 0.8, 0),
                score("Vulpes vulpes_Red Fox", 0.9, 1),
            ],
            &mapping,
        );

        assert_eq!(projected.score_of("Parus major_Great Tit"), Some(0.8));
        assert_eq!(projected.score_of("Vulpes vulpes_Red Fox"), None);
    }

    #[test]
    fn test_in_range_count_applies_the_threshold() {
        let geomodel = labels(&["Aaa aaa_X", "Bbb bbb_Y", "Ccc ccc_Z"]);
        let mapping = SpeciesMapping::build(&geomodel, &geomodel);

        let projected = GeomodelScores::project(
            &[
                score("Aaa aaa_X", 0.9, 0),
                score("Bbb bbb_Y", 0.005, 1),
                score("Ccc ccc_Z", 0.02, 2),
            ],
            &mapping,
        );

        assert_eq!(projected.in_range_count(0.01), 2);
        assert_eq!(projected.in_range_count(0.5), 1);
        assert_eq!(projected.in_range_count(0.99), 0);
    }
}
