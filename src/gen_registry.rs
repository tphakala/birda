//! Generating `registry.json` from published `models.json` manifests.
//!
//! Feature-gated behind `gen-registry` because it is a maintenance tool, not
//! part of the shipped CLI. Written in Rust rather than as a script so it
//! serializes the very types the CLI deserializes: a registry this produces
//! cannot fail to load.
//!
//! Inputs are vendored under `manifests/`, so generation and the test that
//! guards against drift are hermetic and need no network.

use crate::error::{Error, Result};
use crate::registry::types::{
    Countries, FileInfo, LicenseInfo, ModelEntry, ModelVariant, Registry,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// Hugging Face origin used to build download URLs.
const HF_ORIGIN: &str = "https://huggingface.co";
/// Branch the published files are resolved from.
const HF_REVISION: &str = "main";
/// Directory holding the vendored manifests, relative to the repository root.
const MANIFEST_DIR: &str = "manifests";
/// The generated file, relative to the repository root.
const REGISTRY_FILE: &str = "registry.json";
/// The curation file, relative to the repository root.
const SOURCES_FILE: &str = "registry-sources.toml";

/// One published file, the subset of a manifest entry the gallery needs.
#[derive(Debug, Deserialize)]
struct ManifestModel {
    path: String,
    variant: String,
    #[serde(default)]
    region: Option<String>,
    #[serde(default)]
    classes: Option<usize>,
    #[serde(default)]
    labels: Option<String>,
    upstream_version: String,
    build: u32,
    sha256: String,
    size_bytes: u64,
    #[serde(default)]
    superseded_by: Option<String>,
}

/// A repository's `models.json`.
#[derive(Debug, Deserialize)]
struct Manifest {
    repo: String,
    #[serde(default)]
    selection: BTreeMap<String, String>,
    models: Vec<ManifestModel>,
}

/// Display metadata for one region, from its `metadata.json`.
#[derive(Debug, Deserialize)]
struct RegionMetadata {
    name: String,
    #[serde(default)]
    group: Option<String>,
    #[serde(default)]
    group_display: Option<String>,
    #[serde(default)]
    group_order: u32,
    /// Class count, where the manifest entry itself does not carry one.
    #[serde(default)]
    classes: Option<usize>,
    /// Countries this region covers, carried through verbatim to the variant so
    /// a consumer can offer a country-name search over the region list.
    #[serde(default)]
    countries: Option<Countries>,
}

/// Curation entry from `registry-sources.toml`.
#[derive(Debug, Deserialize)]
struct SourceModel {
    id: String,
    manifest: String,
    regions: String,
    name: String,
    description: String,
    vendor: String,
    model_type: String,
    #[serde(default)]
    recommended: bool,
    default_variant: String,
    license: LicenseInfo,
}

/// `registry-sources.toml`.
#[derive(Debug, Deserialize)]
struct Sources {
    schema_version: String,
    #[serde(rename = "model")]
    models: Vec<SourceModel>,
}

/// The `registry_version` for freshly generated content.
///
/// The loader replaces a user's cached registry only when the bundled version
/// is strictly greater, so any content change must bump the version or it never
/// reaches a cached user (the corrected Perch class counts that shipped without
/// a bump and left every region showing "0 species"). This bumps by one when
/// `generated` differs from `existing` in anything but the version field, and
/// keeps the version when they match, so a no-op regeneration stays a no-op and
/// a real change forces exactly one increment.
///
/// The comparison is against the on-disk registry, from which the frozen legacy
/// entries and the range filter are also carried through verbatim, so it tracks
/// changes to the generator-owned (manifest-derived) models. A manual edit to a
/// carried-through entry appears on both sides and so does not bump; those
/// entries are frozen, which is why that is acceptable.
fn next_registry_version(generated: &Registry, existing: &Registry) -> u32 {
    let mut normalized = generated.clone();
    normalized.registry_version = existing.registry_version;
    if normalized == *existing {
        existing.registry_version
    } else {
        existing.registry_version.saturating_add(1)
    }
}

/// Build the registry JSON text from the vendored manifests under `root`.
///
/// Returns the exact bytes that belong in `registry.json`, so the caller can
/// either write them or compare them with what is checked in.
pub fn generate_from_repo_root(root: &str) -> Result<String> {
    let root = Path::new(root);
    let sources: Sources =
        toml::from_str(&read_to_string(&root.join(SOURCES_FILE))?).map_err(|e| {
            Error::Internal {
                message: format!("{SOURCES_FILE} is not valid TOML: {e}"),
            }
        })?;

    // The entries this generator does not own are preserved byte for byte from
    // the registry being replaced. That is what keeps the frozen legacy entries
    // and the range filter asset alive across a regeneration.
    let existing: Registry = serde_json::from_str(&read_to_string(&root.join(REGISTRY_FILE))?)
        .map_err(|e| Error::Internal {
            message: format!("{REGISTRY_FILE} is not valid JSON: {e}"),
        })?;

    let generated_ids: Vec<&str> = sources.models.iter().map(|m| m.id.as_str()).collect();
    let mut models: Vec<ModelEntry> = Vec::new();
    for source in &sources.models {
        models.push(build_entry(root, source)?);
    }
    models.extend(
        existing
            .models
            .iter()
            .filter(|m| !generated_ids.contains(&m.id.as_str()))
            .cloned(),
    );

    // Stable order regardless of how the sources file is arranged, so a
    // regeneration produces a reviewable diff rather than a reshuffle.
    models.sort_by(|a, b| a.id.cmp(&b.id));

    let mut registry = Registry {
        schema_version: sources.schema_version.clone(),
        // Seeded from the on-disk registry; bumped just below only when the
        // content actually changed, so the version is generator-managed rather
        // than a manual step in registry-sources.toml that is easy to forget.
        registry_version: existing.registry_version,
        models,
        range_filter: existing.range_filter.clone(),
    };
    registry.registry_version = next_registry_version(&registry, &existing);

    let mut json = serde_json::to_string_pretty(&registry).map_err(|e| Error::Internal {
        message: format!("could not serialize the registry: {e}"),
    })?;
    json.push('\n');
    Ok(json)
}

/// Read a file, naming it in the error rather than reporting a bare IO failure.
fn read_to_string(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| Error::Internal {
        message: format!("could not read {}: {e}", path.display()),
    })
}

/// Turn one manifest plus its curation into a registry entry.
fn build_entry(root: &Path, source: &SourceModel) -> Result<ModelEntry> {
    let manifest_dir = root.join(MANIFEST_DIR);
    let manifest: Manifest = serde_json::from_str(&read_to_string(
        &manifest_dir.join(&source.manifest),
    )?)
    .map_err(|e| Error::Internal {
        message: format!("{} is not a valid manifest: {e}", source.manifest),
    })?;
    let regions: BTreeMap<String, RegionMetadata> = serde_json::from_str(&read_to_string(
        &manifest_dir.join(&source.regions),
    )?)
    .map_err(|e| Error::Internal {
        message: format!("{} is not valid region metadata: {e}", source.regions),
    })?;

    // A superseded file stays published so older clients keep working, but it
    // must never be offered to a new install.
    let publishable: Vec<&ManifestModel> = manifest
        .models
        .iter()
        .filter(|m| m.superseded_by.is_none())
        .collect();

    let first = publishable.first().ok_or_else(|| Error::Internal {
        message: format!("{} lists no installable files", source.manifest),
    })?;
    // Deliberately not filtered by the `legacy` flag: that flag records whether
    // a filename predates the current naming scheme, which says nothing about
    // whether the version string is usable. Perch marks every file legacy, so
    // filtering on it would leave the entry with no version at all.
    let version = first.upstream_version.clone();
    let build = first.build;

    let mut variants: Vec<ModelVariant> = Vec::new();
    for model in &publishable {
        let labels = model.labels.as_ref().ok_or_else(|| Error::Internal {
            message: format!("{} has no labels file and cannot be installed", model.path),
        })?;
        let region_meta = model
            .region
            .as_ref()
            .and_then(|region| regions.get(region.as_str()));

        variants.push(ModelVariant {
            id: model.variant.clone(),
            region: model.region.clone(),
            region_name: region_meta.map(|m| m.name.clone()),
            group: region_meta.and_then(|m| m.group.clone()),
            group_name: region_meta.and_then(|m| m.group_display.clone()),
            group_order: region_meta.map_or(0, |m| m.group_order),
            // The manifest entry first, then the region's own metadata. Perch
            // states class counts only in the latter, and not at all for its
            // global model, which is why this can legitimately end up None.
            classes: model
                .classes
                .or_else(|| region_meta.and_then(|m| m.classes)),
            model: FileInfo {
                url: file_url(&manifest.repo, &model.path),
                filename: basename(&model.path)?.to_string(),
                sha256: Some(model.sha256.clone()),
                size_bytes: Some(model.size_bytes),
            },
            labels: FileInfo {
                url: file_url(&manifest.repo, labels),
                filename: basename(labels)?.to_string(),
                sha256: None,
                size_bytes: None,
            },
            countries: region_meta.and_then(|m| m.countries.clone()),
        });
    }

    Ok(ModelEntry {
        id: source.id.clone(),
        name: source.name.clone(),
        description: source.description.clone(),
        vendor: source.vendor.clone(),
        version,
        model_type: source.model_type.clone(),
        license: source.license.clone(),
        // Omitted on purpose. See ModelEntry::is_variant_based.
        files: None,
        build: Some(build),
        default_variant: Some(source.default_variant.clone()),
        selection: translate_selection(&manifest, &publishable)?,
        variants,
        recommended: source.recommended,
    })
}

/// Marker for a manifest selection value that is a template, not a file.
///
/// Perch's `low-ram` key resolves to `regional/<region>/perch_v2_<region>_...`,
/// which names a shape rather than a download.
const SELECTION_TEMPLATE_MARKER: char = '<';

/// Turn the manifest's `hardware key -> path` map into `hardware key -> variant id`.
///
/// Template values are skipped deliberately. `low-ram` says "use a regional
/// model", which is a choice along the region axis, not the variant axis, and
/// birda must not pick a region for the user: a region is geographic, and
/// guessing one from available memory would hand someone the wrong continent's
/// species list. The regional models it points at are reachable through
/// `--region`, which is where that choice belongs.
///
/// A non-template value that matches no installable file is an error rather
/// than a silent drop. That means the manifest and the file list disagree, and
/// carrying on would leave a hardware key quietly missing from the gallery.
fn translate_selection(
    manifest: &Manifest,
    publishable: &[&ManifestModel],
) -> Result<BTreeMap<String, String>> {
    let mut selection = BTreeMap::new();
    for (key, path) in &manifest.selection {
        if path.contains(SELECTION_TEMPLATE_MARKER) {
            continue;
        }
        let model = publishable
            .iter()
            .find(|m| &m.path == path)
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "{} maps {key} to {path}, which is not an installable file in that manifest",
                    manifest.repo
                ),
            })?;
        selection.insert(key.clone(), model.variant.clone());
    }
    Ok(selection)
}

/// Download URL for a path inside a Hugging Face repository.
fn file_url(repo: &str, path: &str) -> String {
    format!("{HF_ORIGIN}/{repo}/resolve/{HF_REVISION}/{path}")
}

/// Final path component, which is what the file is called once downloaded.
///
/// Published basenames already encode family, upstream version, region, variant
/// and build, and are immutable by the publishing policy, so they cannot
/// collide across versions and need no flattening.
fn basename(path: &str) -> Result<&str> {
    // `rsplit` always yields at least one item, so the interesting failure is
    // an empty final component: a manifest path ending in `/` would otherwise
    // produce an empty filename and be downloaded to the models directory
    // itself.
    path.rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| Error::Internal {
            message: format!("manifest path has no file name: {path}"),
        })
}

/// Write the generated registry to `registry.json` under `root`.
pub fn write_registry(root: &str) -> Result<()> {
    let json = generate_from_repo_root(root)?;
    std::fs::write(Path::new(root).join(REGISTRY_FILE), json).map_err(Error::Io)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry(schema: &str, version: u32) -> Registry {
        Registry {
            schema_version: schema.to_string(),
            registry_version: version,
            models: Vec::new(),
            range_filter: None,
        }
    }

    #[test]
    fn test_next_registry_version_keeps_version_when_content_is_unchanged() {
        let existing = registry("2.0", 6);
        // Same content, even though the generated struct carries a different
        // version: the version is held equal for the comparison, so a no-op
        // regeneration does not bump.
        let generated = registry("2.0", 999);
        assert_eq!(next_registry_version(&generated, &existing), 6);
    }

    #[test]
    fn test_next_registry_version_bumps_once_when_content_changes() {
        let existing = registry("2.0", 6);
        // Any field but the version differing is a content change, and the
        // loader only replaces a cached registry when the bundled version is
        // strictly greater, so it must bump.
        let generated = registry("2.1", 6);
        assert_eq!(next_registry_version(&generated, &existing), 7);
    }

    #[test]
    fn test_next_registry_version_saturates_at_u32_max() {
        // A content change at the ceiling stays at the ceiling rather than
        // wrapping to 0, which every cache would read as a downgrade.
        let existing = registry("2.0", u32::MAX);
        let generated = registry("2.1", u32::MAX);
        assert_eq!(next_registry_version(&generated, &existing), u32::MAX);
    }

    #[test]
    fn test_next_registry_version_bumps_on_a_model_level_change() {
        // The #329/#332 failure was a model-level change (a corrected class
        // count), not a schema_version change, so pin that a difference inside
        // `models` alone, with schema_version held constant, still bumps.
        let root = env!("CARGO_MANIFEST_DIR");
        let existing: Registry = serde_json::from_str(
            &std::fs::read_to_string(Path::new(root).join(REGISTRY_FILE)).unwrap(),
        )
        .unwrap();
        let mut generated = existing.clone();
        let model = generated
            .models
            .first_mut()
            .expect("the registry has at least one model");
        model.version = format!("{}-changed", model.version);

        assert_eq!(
            next_registry_version(&generated, &existing),
            existing.registry_version.saturating_add(1)
        );
    }

    #[test]
    fn test_generate_bumps_the_version_when_the_content_changes() {
        // End-to-end through the real generator: a hermetic copy of its inputs
        // plus a stale on-disk registry proves a content change forces a bump.
        // This reproduces the failure the versioned-model work (#329) exposed: a
        // corrected class count shipped without a bump and reached no cached user.
        let real_root = env!("CARGO_MANIFEST_DIR");
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();

        std::fs::copy(
            Path::new(real_root).join(SOURCES_FILE),
            root.join(SOURCES_FILE),
        )
        .unwrap();
        let manifests = root.join(MANIFEST_DIR);
        std::fs::create_dir(&manifests).unwrap();
        for entry in std::fs::read_dir(Path::new(real_root).join(MANIFEST_DIR)).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), manifests.join(entry.file_name())).unwrap();
        }

        // Stand in for a cached registry at version 6 whose content is stale.
        let mut stale: Registry =
            serde_json::from_str(&generate_from_repo_root(real_root).unwrap()).unwrap();
        stale.registry_version = 6;
        stale.schema_version = "0.0-stale".to_string();
        std::fs::write(
            root.join(REGISTRY_FILE),
            serde_json::to_string_pretty(&stale).unwrap(),
        )
        .unwrap();

        let regenerated: Registry =
            serde_json::from_str(&generate_from_repo_root(root.to_str().unwrap()).unwrap())
                .unwrap();

        // The generated content differs from the stale on-disk copy, so the
        // version bumps exactly once.
        assert_eq!(regenerated.registry_version, 7);
    }
}
