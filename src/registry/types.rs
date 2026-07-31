//! Data structures for model registry.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Registry schema version and model entries.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Registry {
    /// Schema version string (e.g., "1.0").
    pub schema_version: String,
    /// Registry content version (increments when models are added/updated).
    #[serde(default)]
    pub registry_version: u32,
    /// List of available models.
    pub models: Vec<ModelEntry>,
    /// Shared range filter asset (`BirdNET` Geomodel), used by all classifiers.
    ///
    /// Optional so a registry written by an older birda still deserializes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range_filter: Option<RangeFilterAsset>,
}

/// Shared range filter asset available to every classifier.
///
/// Unlike [`ModelEntry`] this has no language variants: the geomodel ships a
/// single English labels file, and the species names shown to the user come
/// from the active classifier's own labels.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RangeFilterAsset {
    /// Unique identifier, e.g. "birdnet-geomodel-v3".
    pub id: String,
    /// Display name. Always includes "`BirdNET`" for attribution.
    pub name: String,
    /// Upstream model version, e.g. "3.0.2". Authoritative over the filename.
    pub version: String,
    /// Organization/author.
    pub vendor: String,
    /// License information.
    pub license: LicenseInfo,
    /// Number of species the model scores.
    pub species_count: usize,
    /// ONNX model file.
    pub model: FileInfo,
    /// Labels file, one `Scientific name_Common name` per line.
    pub labels: FileInfo,
}

/// Single model entry in registry.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ModelEntry {
    /// Unique identifier (kebab-case, matches `model_type`).
    pub id: String,
    /// Display name.
    pub name: String,
    /// Short description (1-2 sentences).
    pub description: String,
    /// Organization/author.
    pub vendor: String,
    /// Model version string.
    pub version: String,
    /// Must match `crate::config::ModelType` enum values.
    pub model_type: String,
    /// License information.
    pub license: LicenseInfo,
    /// Model and labels file information, legacy single-file entries only.
    ///
    /// `None` on variant-based entries, and deliberately so: see
    /// [`ModelEntry::is_variant_based`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files: Option<ModelFiles>,
    /// Our conversion revision of the upstream weights.
    ///
    /// [`ModelEntry::version`] says whose weights these are; this says which of
    /// our exports of them. Both are needed: upstream alone cannot express "we
    /// improved the conversion", and build alone cannot express "the weights
    /// changed underneath".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<u32>,
    /// Variant id to install when no signal identifies a better one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_variant: Option<String>,
    /// Hardware key to variant id, copied from the publisher's manifest.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub selection: BTreeMap<String, String>,
    /// Every downloadable region and variant combination.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<ModelVariant>,
    /// Show as recommended to users.
    #[serde(default)]
    pub recommended: bool,
}

/// One downloadable combination of region and hardware variant.
///
/// A model family publishes the same weights as several files: a global slice
/// and 39 regional ones, each in one or more hardware variants. The variant id
/// is the publisher's own string (`fp32`, `fp16`, `int8-arm`, `no-dft-fp32`),
/// never reinterpreted here, because Perch's variants are not precisions and no
/// precision vocabulary can name `no-dft-fp32`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ModelVariant {
    /// Publisher's variant identifier.
    pub id: String,
    /// Region slug, or `None` for the global model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Human-readable region name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_name: Option<String>,
    /// Continental group slug used to organise the region listing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    /// Human-readable continental group name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_name: Option<String>,
    /// Display order of the continental group.
    #[serde(default)]
    pub group_order: u32,
    /// Number of classes this variant scores, when the publisher states it.
    ///
    /// `Option` rather than a plain count because not every manifest carries
    /// it: Perch declares class counts only in its per-region metadata, and not
    /// at all for its global model. Defaulting a missing count to zero would
    /// print "0 species" next to a perfectly good model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classes: Option<usize>,
    /// ONNX model file.
    pub model: FileInfo,
    /// Labels file matching this variant's class list.
    pub labels: FileInfo,
}

impl ModelEntry {
    /// Whether this entry publishes regional and hardware variants.
    ///
    /// Variant-based entries omit `files` on purpose. `registry.json` is cached
    /// in the user config directory and kept across a birda downgrade, so an
    /// older binary can read an entry naming a `model_type` its `ModelType`
    /// enum lacks. Omitting `files` makes that older binary fail to parse the
    /// registry, which its loader already handles by falling back to its own
    /// bundled copy, rather than letting it install a model it cannot run and
    /// then choke on its own `config.toml` on every later invocation.
    #[must_use]
    pub const fn is_variant_based(&self) -> bool {
        !self.variants.is_empty()
    }

    /// Find the variant for a region and variant id.
    #[must_use]
    pub fn find_variant(&self, region: Option<&str>, id: &str) -> Option<&ModelVariant> {
        self.variants
            .iter()
            .find(|v| v.id == id && v.region.as_deref() == region)
    }

    /// Variant ids available for a region.
    #[must_use]
    pub fn variant_ids_for(&self, region: Option<&str>) -> Vec<&str> {
        self.variants
            .iter()
            .filter(|v| v.region.as_deref() == region)
            .map(|v| v.id.as_str())
            .collect()
    }

    /// One variant per region, for listing the available tiles.
    ///
    /// Regions are what a user chooses; variant ids are chosen for them, so
    /// listing every combination would show each tile once per hardware
    /// variant. The global model is not a region and is excluded.
    ///
    /// The representative is the default variant wherever the region publishes
    /// one, not simply the first in manifest order. Those differ: Perch lists
    /// `int8-arm` before `no-dft-fp32` for every region while the default is
    /// `no-dft-fp32`, so taking the first would advertise a 41 MB download for
    /// a tile whose default install actually fetches 57 MB.
    #[must_use]
    pub fn regions(&self) -> Vec<&ModelVariant> {
        let default = self.default_variant.as_deref();
        let mut by_region: BTreeMap<&str, &ModelVariant> = BTreeMap::new();
        for variant in &self.variants {
            let Some(region) = variant.region.as_deref() else {
                continue;
            };
            let slot = by_region.entry(region).or_insert(variant);
            if default == Some(variant.id.as_str()) {
                *slot = variant;
            }
        }

        let mut out: Vec<&ModelVariant> = by_region.into_values().collect();
        out.sort_by(|a, b| {
            a.group_order
                .cmp(&b.group_order)
                .then_with(|| a.region_name.cmp(&b.region_name))
        });
        out
    }
}

/// License information for a model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct LicenseInfo {
    /// SPDX license identifier.
    #[serde(rename = "type")]
    pub r#type: String,
    /// URL to full license text.
    pub url: String,
    /// Whether commercial use is allowed.
    pub commercial_use: bool,
    /// Whether attribution is required.
    pub attribution_required: bool,
    /// Whether share-alike is required.
    pub share_alike: bool,
}

/// Model and labels file information.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ModelFiles {
    /// Model file information.
    pub model: FileInfo,
    /// Labels file information with language variants.
    pub labels: LabelsInfo,
    /// BSG calibration CSV file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bsg_calibration: Option<FileInfo>,
    /// BSG migration CSV file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bsg_migration: Option<FileInfo>,
    /// BSG distribution maps binary file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bsg_distribution_maps: Option<FileInfo>,
}

/// Single file download information.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct FileInfo {
    /// Direct download URL.
    pub url: String,
    /// Local filename after download.
    pub filename: String,
    /// Optional SHA256 checksum for verification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Optional file size in bytes, used to tell the user what a download costs
    /// before they agree to it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

/// Labels with language variants.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct LabelsInfo {
    /// ISO 639-1 code for default language.
    pub default_language: String,
    /// Available language variants.
    pub languages: Vec<LanguageVariant>,
}

/// Single language variant for labels.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct LanguageVariant {
    /// ISO 639-1 language code.
    pub code: String,
    /// Human-readable language name.
    pub name: String,
    /// URL to labels file for this language.
    pub url: String,
    /// Local filename after download.
    pub filename: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variant(id: &str, region: Option<&str>, classes: usize) -> ModelVariant {
        ModelVariant {
            id: id.to_string(),
            region: region.map(str::to_string),
            region_name: region.map(str::to_uppercase),
            group: region.map(|_| "europe".to_string()),
            group_name: region.map(|_| "Europe".to_string()),
            group_order: 0,
            classes: Some(classes),
            model: FileInfo {
                url: format!("https://huggingface.co/x/m-{id}.onnx"),
                filename: format!("m-{id}.onnx"),
                sha256: Some("abc".to_string()),
                size_bytes: Some(10),
            },
            labels: FileInfo {
                url: "https://huggingface.co/x/l.txt".to_string(),
                filename: "l.txt".to_string(),
                sha256: Some("def".to_string()),
                size_bytes: Some(1),
            },
        }
    }

    fn variant_entry() -> ModelEntry {
        ModelEntry {
            id: "birdnet-v30".to_string(),
            name: "BirdNET v3.0".to_string(),
            description: "test".to_string(),
            vendor: "test".to_string(),
            version: "3.0-preview3.1".to_string(),
            model_type: "birdnet-v30".to_string(),
            license: LicenseInfo {
                r#type: "CC-BY-NC-SA-4.0".to_string(),
                url: "https://example.com".to_string(),
                commercial_use: false,
                attribution_required: true,
                share_alike: true,
            },
            files: None,
            build: Some(1),
            default_variant: Some("fp32".to_string()),
            selection: std::iter::once(("cuda".to_string(), "fp16".to_string())).collect(),
            variants: vec![
                variant("fp32", None, 11560),
                variant("fp16", None, 11560),
                variant("fp32", Some("nordic"), 422),
                variant("fp16", Some("nordic"), 422),
            ],
            recommended: true,
        }
    }

    #[test]
    fn test_variant_entry_is_variant_based() {
        assert!(variant_entry().is_variant_based());
    }

    #[test]
    fn test_legacy_entry_is_not_variant_based() {
        // The three shipped entries carry `files` and no `variants`. They must
        // keep taking the legacy install path.
        let json = r#"{
            "id": "birdnet-v24", "name": "n", "description": "d", "vendor": "v",
            "version": "2.4", "model_type": "birdnet-v24",
            "license": {"type":"CC","url":"https://x","commercial_use":false,
                        "attribution_required":true,"share_alike":true},
            "files": {"model":{"url":"https://x/m.onnx","filename":"m.onnx"},
                      "labels":{"default_language":"en","languages":[
                        {"code":"en","name":"English","url":"https://x/l.txt","filename":"l.txt"}]}}
        }"#;
        let entry: ModelEntry = serde_json::from_str(json).unwrap();
        assert!(!entry.is_variant_based());
        assert!(entry.files.is_some());
    }

    #[test]
    fn test_find_variant_matches_region_and_id() {
        let entry = variant_entry();
        let found = entry.find_variant(Some("nordic"), "fp32").unwrap();
        assert_eq!(found.classes, Some(422));
        let global = entry.find_variant(None, "fp32").unwrap();
        assert_eq!(global.classes, Some(11560));
    }

    #[test]
    fn test_find_variant_rejects_unknown_region() {
        assert!(
            variant_entry()
                .find_variant(Some("atlantis"), "fp32")
                .is_none()
        );
    }

    #[test]
    fn test_regions_deduplicates_across_variant_ids() {
        // nordic appears twice, once per variant id. `regions` lists tiles, not
        // downloads, so it must report nordic exactly once and skip the global.
        let entry = variant_entry();
        let regions = entry.regions();
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].region.as_deref(), Some("nordic"));
    }

    #[test]
    fn test_regions_sorts_by_group_order_then_name() {
        let mut entry = variant_entry();
        let mut asia = variant("fp32", Some("japan"), 500);
        asia.group = Some("asia".to_string());
        asia.group_name = Some("Asia".to_string());
        asia.group_order = 1;
        let mut iberia = variant("fp32", Some("iberia"), 600);
        iberia.group_order = 0;
        entry.variants.push(asia);
        entry.variants.push(iberia);

        let slugs: Vec<&str> = entry
            .regions()
            .iter()
            .filter_map(|v| v.region.as_deref())
            .collect();

        // Europe (group_order 0) before Asia (1), and inside Europe the display
        // names IBERIA and NORDIC sort alphabetically.
        assert_eq!(slugs, vec!["iberia", "nordic", "japan"]);
    }

    #[test]
    fn test_regions_represents_each_tile_with_the_variant_an_install_would_pick() {
        // Perch lists int8-arm before no-dft-fp32 for every region while the
        // default is no-dft-fp32, so taking the first in manifest order made
        // `models regions` advertise a 41 MB download for a tile whose default
        // install actually fetches 57 MB.
        let mut entry = variant_entry();
        entry.default_variant = Some("fp16".to_string());

        let regions = entry.regions();

        assert_eq!(regions.len(), 1);
        assert_eq!(
            regions[0].id, "fp16",
            "the listing must quote the variant an install without --variant gets"
        );
    }

    #[test]
    fn test_regions_falls_back_when_a_tile_lacks_the_default_variant() {
        let mut entry = variant_entry();
        entry.default_variant = Some("fp16".to_string());
        entry.variants.push(variant("fp32", Some("iberia"), 600));

        let iberia = entry
            .regions()
            .into_iter()
            .find(|v| v.region.as_deref() == Some("iberia"))
            .expect("iberia is listed");

        assert_eq!(iberia.id, "fp32");
    }

    #[test]
    fn test_variant_ids_for_region() {
        let entry = variant_entry();
        let mut ids = entry.variant_ids_for(Some("nordic"));
        ids.sort_unstable();
        assert_eq!(ids, vec!["fp16", "fp32"]);
    }

    #[test]
    fn test_variant_ids_for_an_unknown_region_is_empty() {
        assert!(variant_entry().variant_ids_for(Some("atlantis")).is_empty());
    }

    #[test]
    fn test_a_variant_entry_is_rejected_by_the_legacy_schema() {
        // The downgrade guard. An older birda declares `files` as required, so a
        // variant-only entry must fail to parse there, which sends its loader
        // down the "fall back to the bundled registry" path instead of letting
        // it install a model type its ModelType enum does not have.
        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct LegacyModelEntry {
            id: String,
            name: String,
            description: String,
            vendor: String,
            version: String,
            model_type: String,
            license: LicenseInfo,
            files: ModelFiles,
        }

        let json = serde_json::to_string(&variant_entry()).unwrap();
        assert!(serde_json::from_str::<LegacyModelEntry>(&json).is_err());
    }

    #[test]
    fn test_a_variant_entry_round_trips() {
        let entry = variant_entry();
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: ModelEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, entry);
    }

    #[test]
    fn test_deserialize_registry_with_range_filter() {
        let json = r#"{"schema_version":"1.1","registry_version":4,"models":[],
            "range_filter":{"id":"birdnet-geomodel-v3","name":"BirdNET Geomodel v3.0.2",
            "version":"3.0.2","vendor":"Cornell Lab","species_count":12012,
            "license":{"type":"CC-BY-SA-4.0","url":"https://x","commercial_use":true,
                       "attribution_required":true,"share_alike":true},
            "model":{"url":"https://x/m.onnx","filename":"m.onnx","sha256":"abc"},
            "labels":{"url":"https://x/l.txt","filename":"l.txt","sha256":"def"}}}"#;
        let registry: Registry = serde_json::from_str(json).unwrap();
        let rf = registry.range_filter.unwrap();
        assert_eq!(rf.version, "3.0.2");
        assert_eq!(rf.species_count, 12012);
        assert_eq!(rf.model.sha256.as_deref(), Some("abc"));
        assert!(
            rf.name.contains("BirdNET"),
            "geomodel display name must credit BirdNET"
        );
    }

    #[test]
    fn test_deserialize_registry_without_range_filter() {
        let json = r#"{"schema_version":"1.0","registry_version":3,"models":[]}"#;
        let registry: Registry = serde_json::from_str(json).unwrap();
        assert!(registry.range_filter.is_none());
    }

    #[test]
    fn test_deserialize_legacy_model_files_with_meta_model() {
        // A registry.json cached from schema 1.0 still carries files.meta_model.
        // It must deserialize (serde ignores unknown fields) so the loader can
        // replace it with the newer bundled registry.
        let json = r#"{
            "model": {"url":"https://x/m.onnx","filename":"m.onnx"},
            "labels": {"default_language":"en","languages":[
                {"code":"en","name":"English","url":"https://x/l.txt","filename":"l.txt"}]},
            "meta_model": {"url":"https://x/meta.onnx","filename":"meta.onnx"}
        }"#;
        let files: ModelFiles = serde_json::from_str(json).unwrap();
        assert_eq!(files.model.filename, "m.onnx");
    }

    #[test]
    fn test_deserialize_empty_registry() {
        let json = r#"{"schema_version":"1.0","registry_version":0,"models":[]}"#;
        let registry: Registry = serde_json::from_str(json).unwrap();
        assert_eq!(registry.schema_version, "1.0");
        assert_eq!(registry.registry_version, 0);
        assert!(registry.models.is_empty());
    }

    #[test]
    fn test_deserialize_model_entry() {
        let json = r#"{
            "id": "test",
            "name": "Test Model",
            "description": "A test model",
            "vendor": "Test Vendor",
            "version": "1.0",
            "model_type": "birdnet-v24",
            "license": {
                "type": "MIT",
                "url": "https://example.com",
                "commercial_use": true,
                "attribution_required": false,
                "share_alike": false
            },
            "files": {
                "model": {
                    "url": "https://example.com/model.onnx",
                    "filename": "model.onnx",
                    "sha256": null
                },
                "labels": {
                    "default_language": "en",
                    "languages": [
                        {
                            "code": "en",
                            "name": "English",
                            "url": "https://example.com/labels.txt",
                            "filename": "labels.txt"
                        }
                    ]
                }
            },
            "recommended": true
        }"#;

        let entry: ModelEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.id, "test");
        assert_eq!(entry.name, "Test Model");
        assert_eq!(entry.license.r#type, "MIT");
        assert!(entry.recommended);
    }

    #[test]
    fn test_deserialize_registry_without_version() {
        // Test backward compatibility - old registries without registry_version
        let json = r#"{"schema_version":"1.0","models":[]}"#;
        let registry: Registry = serde_json::from_str(json).unwrap();
        assert_eq!(registry.schema_version, "1.0");
        assert_eq!(registry.registry_version, 0); // Should default to 0
        assert!(registry.models.is_empty());
    }

    #[test]
    fn test_model_entry_default_recommended() {
        let json = r#"{
            "id": "test",
            "name": "Test Model",
            "description": "A test model",
            "vendor": "Test Vendor",
            "version": "1.0",
            "model_type": "birdnet-v24",
            "license": {
                "type": "MIT",
                "url": "https://example.com",
                "commercial_use": true,
                "attribution_required": false,
                "share_alike": false
            },
            "files": {
                "model": {
                    "url": "https://example.com/model.onnx",
                    "filename": "model.onnx",
                    "sha256": null
                },
                "labels": {
                    "default_language": "en",
                    "languages": [
                        {
                            "code": "en",
                            "name": "English",
                            "url": "https://example.com/labels.txt",
                            "filename": "labels.txt"
                        }
                    ]
                }
            }
        }"#;

        let entry: ModelEntry = serde_json::from_str(json).unwrap();
        assert!(!entry.recommended); // Default is false
    }
}
