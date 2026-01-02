//! Registry loading and bootstrapping.

use super::types::{ModelEntry, Registry};
use crate::error::{Error, Result};
use std::path::PathBuf;

/// Load registry from user config or bundled default.
pub fn load_registry() -> Result<Registry> {
    let registry_path = registry_file_path()?;

    if registry_path.exists() {
        load_from_file(&registry_path)
    } else {
        bootstrap_registry(&registry_path)
    }
}

/// Get path to registry file in user config.
fn registry_file_path() -> Result<PathBuf> {
    let config_dir = directories::ProjectDirs::from("", "", "birda")
        .ok_or(Error::ConfigDirNotFound)?
        .config_dir()
        .to_path_buf();

    Ok(config_dir.join("registry.json"))
}

/// Load registry from existing file.
fn load_from_file(path: &std::path::Path) -> Result<Registry> {
    let content = std::fs::read_to_string(path).map_err(|e| Error::RegistryRead {
        path: path.to_path_buf(),
        source: e,
    })?;

    serde_json::from_str(&content).map_err(|e| Error::RegistryParse {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Bootstrap registry from bundled default.
fn bootstrap_registry(dest: &std::path::Path) -> Result<Registry> {
    const BUNDLED_REGISTRY: &str = include_str!("../../registry.json");

    // Parse bundled registry
    let registry: Registry =
        serde_json::from_str(BUNDLED_REGISTRY).map_err(|e| Error::RegistryParse {
            path: PathBuf::from("bundled://registry.json"),
            source: e,
        })?;

    // Ensure config directory exists
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }

    // Write to user config
    let content = serde_json::to_string_pretty(&registry)
        .map_err(|e| Error::RegistrySerialize { source: e })?;

    std::fs::write(dest, content).map_err(|e| Error::RegistryWrite {
        path: dest.to_path_buf(),
        source: e,
    })?;

    Ok(registry)
}

/// Find model entry by ID.
pub fn find_model<'a>(registry: &'a Registry, id: &str) -> Option<&'a ModelEntry> {
    registry.models.iter().find(|m| m.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::types::{FileInfo, LabelsInfo, LanguageVariant, LicenseInfo, ModelFiles};

    #[test]
    fn test_find_model_by_id() {
        let registry = Registry {
            schema_version: "1.0".into(),
            models: vec![
                ModelEntry {
                    id: "test-1".into(),
                    name: "Test Model 1".into(),
                    description: "First test model".into(),
                    vendor: "Test Vendor".into(),
                    version: "1.0".into(),
                    model_type: "birdnet-v24".into(),
                    license: LicenseInfo {
                        r#type: "MIT".into(),
                        url: "https://example.com".into(),
                        commercial_use: true,
                        attribution_required: false,
                        share_alike: false,
                    },
                    files: ModelFiles {
                        model: FileInfo {
                            url: "https://example.com/model.onnx".into(),
                            filename: "model.onnx".into(),
                            sha256: None,
                        },
                        labels: LabelsInfo {
                            default_language: "en".into(),
                            languages: vec![LanguageVariant {
                                code: "en".into(),
                                name: "English".into(),
                                url: "https://example.com/labels.txt".into(),
                                filename: "labels.txt".into(),
                            }],
                        },
                    },
                    recommended: false,
                },
                ModelEntry {
                    id: "test-2".into(),
                    name: "Test Model 2".into(),
                    description: "Second test model".into(),
                    vendor: "Test Vendor".into(),
                    version: "2.0".into(),
                    model_type: "perch-v2".into(),
                    license: LicenseInfo {
                        r#type: "Apache-2.0".into(),
                        url: "https://example.com".into(),
                        commercial_use: true,
                        attribution_required: true,
                        share_alike: false,
                    },
                    files: ModelFiles {
                        model: FileInfo {
                            url: "https://example.com/model2.onnx".into(),
                            filename: "model2.onnx".into(),
                            sha256: None,
                        },
                        labels: LabelsInfo {
                            default_language: "en".into(),
                            languages: vec![LanguageVariant {
                                code: "en".into(),
                                name: "English".into(),
                                url: "https://example.com/labels2.txt".into(),
                                filename: "labels2.txt".into(),
                            }],
                        },
                    },
                    recommended: true,
                },
            ],
        };

        // Test finding existing models
        assert!(find_model(&registry, "test-1").is_some());
        assert!(find_model(&registry, "test-2").is_some());

        let model1 = find_model(&registry, "test-1").unwrap();
        assert_eq!(model1.name, "Test Model 1");
        assert_eq!(model1.version, "1.0");

        let model2 = find_model(&registry, "test-2").unwrap();
        assert_eq!(model2.name, "Test Model 2");
        assert!(model2.recommended);

        // Test finding non-existent model
        assert!(find_model(&registry, "missing").is_none());
    }

    #[test]
    fn test_bundled_registry_parses() {
        // This test verifies that the bundled registry.json is valid
        const BUNDLED_REGISTRY: &str = include_str!("../../registry.json");

        let result = serde_json::from_str::<Registry>(BUNDLED_REGISTRY);
        assert!(result.is_ok(), "Bundled registry should parse successfully");

        let registry = result.unwrap();
        assert_eq!(registry.schema_version, "1.0");
        assert!(
            !registry.models.is_empty(),
            "Registry should contain models"
        );

        // Verify we have expected models
        assert!(find_model(&registry, "birdnet-v24").is_some());
        assert!(find_model(&registry, "perch-v2").is_some());
    }
}
