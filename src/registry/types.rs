//! Data structures for model registry.

use serde::{Deserialize, Serialize};

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
    /// Model and labels file information.
    pub files: ModelFiles,
    /// Show as recommended to users.
    #[serde(default)]
    pub recommended: bool,
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
