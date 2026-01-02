//! Data structures for model registry.

use serde::{Deserialize, Serialize};

/// Registry schema version and model entries.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Registry {
    /// Schema version string (e.g., "1.0").
    pub schema_version: String,
    /// List of available models.
    pub models: Vec<ModelEntry>,
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
    fn test_deserialize_empty_registry() {
        let json = r#"{"schema_version":"1.0","models":[]}"#;
        let registry: Registry = serde_json::from_str(json).unwrap();
        assert_eq!(registry.schema_version, "1.0");
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
