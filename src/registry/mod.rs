//! Model registry system for discovering and installing models.

#![allow(clippy::print_stdout)]

pub mod cleanup;
pub mod installer;
pub mod license;
pub mod loader;
pub mod selection;
pub mod types;

// Re-export commonly used types and functions
pub use cleanup::{orphaned_files, remove_orphans};
pub use installer::{
    GEOMODEL_INSTALL_ID, InstallProvenance, InstalledRangeFilter, download_file,
    find_obsolete_files, find_stale_part_files, geomodel_paths, install_model,
    install_range_filter, install_variant, models_dir, resolve_url,
};
pub use license::{LicensedAsset, prompt_license_acceptance};
pub use loader::{find_model, load_registry};
// Only what callers outside this module actually name. `HardwareProbe`,
// `VariantChoice` and `SelectionReason` stay reachable as
// `registry::selection::*` rather than crowding the root: they are the shape of
// how a variant is chosen, not part of the gallery's surface.
pub use selection::{SystemProbe, select_variant};
pub use types::{
    Countries, FileInfo, LabelsInfo, LanguageVariant, LicenseInfo, ModelEntry, ModelFiles,
    ModelVariant, RangeFilterAsset, Registry,
};

use crate::error::{Error, Result};

/// List all available models from the registry.
pub fn list_available(registry: &Registry, output_mode: crate::config::OutputMode) {
    use crate::output::{
        AvailableModelEntry, AvailableModelsPayload, ResultType, emit_json_result,
    };

    // JSON/NDJSON output
    if output_mode.is_structured() {
        let models: Vec<AvailableModelEntry> = registry
            .models
            .iter()
            .map(|m| AvailableModelEntry {
                id: m.id.clone(),
                name: m.name.clone(),
                description: m.description.clone(),
                vendor: m.vendor.clone(),
                version: m.version.clone(),
                model_type: m.model_type.clone(),
                recommended: m.recommended,
                license: m.license.r#type.clone(),
                commercial_use: m.license.commercial_use,
            })
            .collect();
        let payload = AvailableModelsPayload {
            result_type: ResultType::AvailableModels,
            models,
            available_range_filter: registry.range_filter.as_ref().map(available_range_filter),
        };
        emit_json_result(&payload);
        return;
    }

    // Human-readable output
    println!("Available models:");
    println!();

    for model in &registry.models {
        let recommended = if model.recommended {
            " (recommended)"
        } else {
            ""
        };
        println!("  {}{}", model.id, recommended);
        println!("    {} - {}", model.name, model.description);
        println!("    Vendor: {}", model.vendor);

        println!("    License: {}", license_line(&model.license));
        println!();
    }

    // The geomodel lives in `registry.range_filter`, not `registry.models`, so
    // every loop over `models` skips it. Listing it here is what makes the
    // asset every error message tells users to install actually discoverable.
    if let Some(asset) = registry.range_filter.as_ref() {
        println!("Range filter (shared by all classifiers):");
        println!();
        println!("  {GEOMODEL_INSTALL_ID}");
        // RangeFilterAsset has no `description` field, unlike ModelEntry, so
        // this shows the name alone rather than the "name - description" pair
        // used above.
        println!("    {}", asset.name);
        println!("    Vendor: {}", asset.vendor);
        println!("    License: {}", license_line(&asset.license));
        println!("    Covers {} species", asset.species_count);
        println!();
    }

    println!("Run 'birda models info <id>' for details.");
}

/// Render a variant's class count, or say the publisher did not state one.
///
/// Perch declares class counts only in its per-region metadata and not at all
/// for its global model, so the count is genuinely unknown for some entries.
/// Printing "0 species" there would be a lie about the model rather than an
/// admission about the manifest.
#[must_use]
pub fn species_count_label(classes: Option<usize>) -> String {
    classes.map_or_else(
        || "species count not published".to_string(),
        |count| format!("{count} species"),
    )
}

/// Render a licence identifier with the restrictions that apply to it.
///
/// One renderer for classifiers and the range filter alike. Listing them
/// separately taught a falsehood: the classifier loop showed only
/// `(non-commercial)` and the range filter showed only `(share-alike)`, so
/// `birdnet-v24` and `bsg-fi-v44` listed without a share-alike note even though
/// both carry that obligation. Whichever restrictions apply are now named on
/// every entry.
fn license_line(license: &LicenseInfo) -> String {
    let mut notes = Vec::new();
    if !license.commercial_use {
        notes.push("non-commercial");
    }
    if license.share_alike {
        notes.push("share-alike");
    }

    if notes.is_empty() {
        license.r#type.clone()
    } else {
        format!("{} ({})", license.r#type, notes.join(", "))
    }
}

/// Project the shared range filter asset into its structured-output shape.
fn available_range_filter(asset: &RangeFilterAsset) -> crate::output::AvailableRangeFilterEntry {
    crate::output::AvailableRangeFilterEntry {
        // The install handle, not `asset.id`: this is the string a user types.
        id: GEOMODEL_INSTALL_ID.to_string(),
        name: asset.name.clone(),
        version: asset.version.clone(),
        vendor: asset.vendor.clone(),
        license: asset.license.r#type.clone(),
        commercial_use: asset.license.commercial_use,
        share_alike: asset.license.share_alike,
        species_count: asset.species_count,
        size_bytes: total_download_size(asset),
    }
}

/// Combined download size of the geomodel's two files.
///
/// Both files are required, so the number a user weighing the download cares
/// about is the total. Returns `None` unless both sizes are declared, rather
/// than reporting a half-total that reads as the whole.
fn total_download_size(asset: &RangeFilterAsset) -> Option<u64> {
    let model = asset.model.size_bytes?;
    let labels = asset.labels.size_bytes?;
    model.checked_add(labels)
}

/// Render the shared range filter asset for `birda models info geomodel`.
///
/// Separate from [`show_info`] because the geomodel is not a [`ModelEntry`]:
/// it has no per-language labels, no model type and no description, and it
/// carries a species count and a download size that no classifier entry does.
///
/// Note that share-alike is NOT what distinguishes it: `birdnet-v24`
/// (CC BY-NC-SA) and `bsg-fi-v44` (BSG-NC) are share-alike too. What differs is
/// commercial use, which the geomodel permits and those two do not.
pub fn show_range_filter_info(asset: &RangeFilterAsset) {
    println!("Range filter: {}", asset.name);
    println!("ID: {GEOMODEL_INSTALL_ID}");
    println!("Version: {}", asset.version);
    println!("Vendor: {}", asset.vendor);
    println!();

    println!("Description:");
    println!(
        "  Scores {} species by location and time of year. Shared by every",
        asset.species_count
    );
    println!("  classifier; it is not selectable with -m.");
    println!();

    println!("License:");
    println!("  Type: {}", asset.license.r#type);
    println!("  URL: {}", asset.license.url);
    println!(
        "  Commercial use: {}",
        if asset.license.commercial_use {
            "Yes"
        } else {
            "No"
        }
    );
    println!(
        "  Attribution required: {}",
        if asset.license.attribution_required {
            "Yes"
        } else {
            "No"
        }
    );
    println!(
        "  Share-alike required: {}",
        if asset.license.share_alike {
            "Yes"
        } else {
            "No"
        }
    );
    println!();

    println!("Files:");
    println!("  Model: {}", asset.model.url);
    println!("  Labels: {}", asset.labels.url);
    println!(
        "  Download size: {}",
        crate::config::geomodel::human_size(total_download_size(asset))
    );
    println!();

    println!("To install: birda models install {GEOMODEL_INSTALL_ID}");
}

/// Show detailed information about a specific model.
pub fn show_info(registry: &Registry, id: &str) -> Result<()> {
    let model = find_model(registry, id)
        .ok_or_else(|| Error::ModelNotFoundInRegistry { id: id.to_string() })?;

    println!("Model: {}", model.name);
    println!("ID: {}", model.id);
    // The version is the exact upstream identity, preview status included, and
    // the build is our conversion revision of those same weights. Showing only
    // the first would let two different files answer to one version string.
    if let Some(build) = model.build {
        println!("Version: {} (build {build})", model.version);
    } else {
        println!("Version: {}", model.version);
    }
    println!("Vendor: {}", model.vendor);
    println!();

    println!("Description:");
    println!("  {}", model.description);
    println!();

    println!("License:");
    println!("  Type: {}", model.license.r#type);
    println!("  URL: {}", model.license.url);
    println!(
        "  Commercial use: {}",
        if model.license.commercial_use {
            "Yes"
        } else {
            "No"
        }
    );
    println!(
        "  Attribution required: {}",
        if model.license.attribution_required {
            "Yes"
        } else {
            "No"
        }
    );
    println!(
        "  Share-alike required: {}",
        if model.license.share_alike {
            "Yes"
        } else {
            "No"
        }
    );
    println!();

    if let Some(files) = model.files.as_ref() {
        println!("Files:");
        println!("  Model: {}", files.model.url);

        let lang_count = files.labels.languages.len();
        let default_lang = files
            .labels
            .languages
            .iter()
            .find(|l| l.code == files.labels.default_language)
            .map_or("Unknown", |l| l.name.as_str());

        if lang_count == 1 {
            println!("  Labels: {default_lang} only");
        } else {
            println!("  Labels: {lang_count} languages available (default: {default_lang})");
        }
        println!();
    }

    if model.is_variant_based() {
        let variant_ids = model.variant_ids_for(None).join(", ");
        let global = model
            .default_variant
            .as_deref()
            .and_then(|id| model.find_variant(None, id));

        println!("Variants: {variant_ids}");
        if let Some(global) = global {
            println!(
                "  Global model: {}, {}",
                species_count_label(global.classes),
                crate::config::geomodel::human_size(global.model.size_bytes)
            );
        }
        println!(
            "  Regional models: {} (birda models regions {})",
            model.regions().len(),
            model.id
        );
        println!();
    }

    println!("To install: birda models install {}", model.id);

    Ok(())
}

/// Variant id used for the single synthetic variant of a legacy single-file
/// model, which has no publisher variant id of its own.
const LEGACY_VARIANT_ID: &str = "global";

/// Emit the documented manifest projection for a model (`models manifest <id>`).
///
/// This is the machine-readable catalogue a consumer builds a region-aware
/// model gallery on: every region and variant, each with its class count,
/// download size, resolved model and labels URLs, and the countries the region
/// covers. It is a deliberate projection, never a dump of `registry.json`,
/// which uses field omission as a downgrade guard (see
/// [`ModelEntry::is_variant_based`]) and must not become a public contract.
///
/// The human rendering is a short summary; the value here is the JSON.
pub fn show_manifest(
    registry: &Registry,
    id: &str,
    output_mode: crate::config::OutputMode,
) -> Result<()> {
    let model = find_model(registry, id)
        .ok_or_else(|| Error::ModelNotFoundInRegistry { id: id.to_string() })?;

    let manifest = project_manifest(model);

    if output_mode.is_structured() {
        use crate::output::{ModelManifestPayload, ResultType, emit_json_result};
        let payload = ModelManifestPayload {
            result_type: ResultType::ModelManifest,
            manifest,
        };
        emit_json_result(&payload);
        return Ok(());
    }

    print_manifest_human(&manifest);
    Ok(())
}

/// Build the projection for one registry entry.
///
/// A variant-based entry projects every region and variant combination (unlike
/// [`ModelEntry::regions`], which deduplicates to one tile per region). A legacy
/// single-file entry has no variants, so it is projected as one synthetic
/// `global` variant from its `files`, giving a consumer one uniform shape across
/// old and new models.
fn project_manifest(model: &ModelEntry) -> crate::output::ModelManifest {
    let variants = if model.is_variant_based() {
        model.variants.iter().map(project_variant).collect()
    } else {
        legacy_variant(model).into_iter().collect()
    };

    crate::output::ModelManifest {
        id: model.id.clone(),
        name: model.name.clone(),
        version: model.version.clone(),
        build: model.build,
        model_type: model.model_type.clone(),
        license: model.license.clone(),
        default_variant: model.default_variant.clone(),
        selection: model.selection.clone(),
        variants,
    }
}

/// Project one variant, resolving both URLs through `HF_ENDPOINT` once here so a
/// consumer never reimplements mirror rewriting.
///
/// `ModelVariant` is destructured exhaustively on purpose: a field added to the
/// registry type then fails to compile here until someone decides whether the
/// projection should carry it, rather than silently dropping it from the public
/// contract.
fn project_variant(variant: &ModelVariant) -> crate::output::ManifestVariant {
    let ModelVariant {
        id,
        region,
        region_name,
        group,
        group_name,
        group_order,
        classes,
        model,
        labels,
        countries,
    } = variant;

    crate::output::ManifestVariant {
        id: id.clone(),
        region: region.clone(),
        region_name: region_name.clone(),
        group: group.clone(),
        group_name: group_name.clone(),
        group_order: *group_order,
        classes: *classes,
        size_bytes: model.size_bytes,
        model_url: resolve_url(&model.url),
        labels_url: resolve_url(&labels.url),
        countries: countries.clone(),
    }
}

/// Synthesize a single `global` variant for a legacy single-file model.
///
/// A legacy entry carries `files` (one model file plus a per-language labels
/// set), not `variants`. It projects to one variant so a consumer sees the same
/// shape as a variant-based model; the labels URL is the default-language file,
/// and there is no region, so no countries and no class count.
fn legacy_variant(model: &ModelEntry) -> Option<crate::output::ManifestVariant> {
    let files = model.files.as_ref()?;
    let labels_url = files
        .labels
        .languages
        .iter()
        .find(|l| l.code == files.labels.default_language)
        .or_else(|| files.labels.languages.first())
        .map_or_else(String::new, |l| resolve_url(&l.url));

    Some(crate::output::ManifestVariant {
        id: LEGACY_VARIANT_ID.to_string(),
        region: None,
        region_name: None,
        group: None,
        group_name: None,
        group_order: 0,
        classes: None,
        size_bytes: files.model.size_bytes,
        model_url: resolve_url(&files.model.url),
        labels_url,
        countries: None,
    })
}

/// Short human summary of a projected manifest, for parity with [`show_info`].
///
/// The full per-region detail is intentionally left to the JSON form; dumping
/// 80 variants as text would bury the summary a person actually wants.
fn print_manifest_human(manifest: &crate::output::ModelManifest) {
    println!("Model: {}", manifest.name);
    println!("ID: {}", manifest.id);
    if let Some(build) = manifest.build {
        println!("Version: {} (build {build})", manifest.version);
    } else {
        println!("Version: {}", manifest.version);
    }
    println!("Type: {}", manifest.model_type);
    println!("License: {}", license_line(&manifest.license));
    if let Some(default) = manifest.default_variant.as_deref() {
        println!("Default variant: {default}");
    }
    println!();

    let global: Vec<&str> = manifest
        .variants
        .iter()
        .filter(|v| v.region.is_none())
        .map(|v| v.id.as_str())
        .collect();
    if !global.is_empty() {
        println!("Global variants: {}", global.join(", "));
    }

    let regions: std::collections::BTreeSet<&str> = manifest
        .variants
        .iter()
        .filter_map(|v| v.region.as_deref())
        .collect();
    if !regions.is_empty() {
        println!("Regions: {}", regions.len());
    }

    println!();
    println!("Run with --output-mode json for the full machine-readable manifest,");
    println!("including per-region country coverage and resolved download URLs.");
}

/// List the regional tiles a model publishes, grouped by continent.
///
/// Regions are what a user picks; the variant is picked for them, so this lists
/// each tile once rather than once per hardware variant.
pub fn show_regions(registry: &Registry, id: &str) -> Result<()> {
    let model = find_model(registry, id)
        .ok_or_else(|| Error::ModelNotFoundInRegistry { id: id.to_string() })?;

    let regions = model.regions();
    if regions.is_empty() {
        return Err(Error::RegionsNotSupported {
            model_id: id.to_string(),
        });
    }

    println!("Regional variants of {}:", model.name);
    println!();

    let mut current_group: Option<&str> = None;
    for variant in regions {
        let group = variant.group_name.as_deref().unwrap_or("Other");
        if current_group != Some(group) {
            if current_group.is_some() {
                println!();
            }
            println!("{group}:");
            current_group = Some(group);
        }
        println!(
            "  {:<24} {:>28}   {}",
            variant.region.as_deref().unwrap_or("global"),
            species_count_label(variant.classes),
            crate::config::geomodel::human_size(variant.model.size_bytes),
        );
    }

    println!();
    println!("A regional model scores only the species of that region, which cuts");
    println!("memory use and latency. It is otherwise the same model.");
    println!();
    println!("To install: birda models install {id} --region <slug>");

    Ok(())
}

/// Show available languages for a model.
pub fn show_languages(registry: &Registry, id: &str) -> Result<()> {
    let model = find_model(registry, id)
        .ok_or_else(|| Error::ModelNotFoundInRegistry { id: id.to_string() })?;

    // Variant-based families publish a labels file per region, all English, so
    // there are no translations to list. Saying that is more use than printing
    // an empty list.
    let files = model
        .files
        .as_ref()
        .ok_or_else(|| Error::ModelHasNoLanguages {
            model_id: id.to_string(),
        })?;

    println!("Model: {}", model.name);
    println!();
    println!("Available label languages:");
    println!();

    for lang in &files.labels.languages {
        let default_marker = if lang.code == files.labels.default_language {
            " (default)"
        } else {
            ""
        };
        println!("  {} - {}{}", lang.code, lang.name, default_marker);
    }

    println!();
    println!("To install with specific language:");
    println!("  birda models install {} --language <code>", model.id);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn license(commercial_use: bool, share_alike: bool) -> LicenseInfo {
        LicenseInfo {
            r#type: "TEST-1.0".into(),
            url: "https://example.com/licence".into(),
            commercial_use,
            attribution_required: true,
            share_alike,
        }
    }

    #[test]
    fn test_license_line_names_every_restriction_that_applies() {
        // The defect this replaced: the classifier loop showed only
        // "(non-commercial)" and the range filter only "(share-alike)", so
        // birdnet-v24 and bsg-fi-v44 listed with no share-alike note despite
        // carrying that obligation. Both restrictions must show together.
        let line = license_line(&license(false, true));

        assert!(line.contains("non-commercial"), "got: {line}");
        assert!(line.contains("share-alike"), "got: {line}");
    }

    #[test]
    fn test_license_line_names_share_alike_on_a_commercial_licence() {
        // The geomodel's shape: CC BY-SA permits commercial use but still binds
        // share-alike, so the note must not be suppressed by commercial_use.
        let line = license_line(&license(true, true));

        assert!(!line.contains("non-commercial"), "got: {line}");
        assert!(line.contains("share-alike"), "got: {line}");
    }

    #[test]
    fn test_license_line_adds_nothing_for_an_unrestricted_licence() {
        assert_eq!(license_line(&license(true, false)), "TEST-1.0");
    }

    fn projection_variant(id: &str, region: Option<&str>) -> ModelVariant {
        ModelVariant {
            id: id.to_string(),
            region: region.map(str::to_string),
            region_name: region.map(str::to_uppercase),
            group: None,
            group_name: None,
            group_order: 0,
            classes: Some(100),
            model: FileInfo {
                url: format!("https://huggingface.co/o/r/resolve/main/{id}.onnx"),
                filename: format!("{id}.onnx"),
                sha256: None,
                size_bytes: Some(123),
            },
            labels: FileInfo {
                url: "https://huggingface.co/o/r/resolve/main/labels.txt".to_string(),
                filename: "labels.txt".to_string(),
                sha256: None,
                size_bytes: None,
            },
            countries: region.map(|_| Countries {
                core: vec!["Brazil".to_string()],
                partial: Vec::new(),
            }),
        }
    }

    fn projection_entry() -> ModelEntry {
        ModelEntry {
            id: "birdnet-v30".to_string(),
            name: "BirdNET v3.0".to_string(),
            description: "d".to_string(),
            vendor: "v".to_string(),
            version: "3.0".to_string(),
            model_type: "birdnet-v30".to_string(),
            license: license(false, true),
            files: None,
            build: Some(1),
            default_variant: Some("fp32".to_string()),
            selection: std::iter::once(("cuda".to_string(), "fp16".to_string())).collect(),
            variants: vec![
                projection_variant("fp32", None),
                projection_variant("fp16", None),
                projection_variant("fp32", Some("nordic")),
                projection_variant("fp16", Some("nordic")),
            ],
            recommended: true,
        }
    }

    fn legacy_projection_entry() -> ModelEntry {
        ModelEntry {
            id: "birdnet-v24".to_string(),
            name: "BirdNET v2.4".to_string(),
            description: "d".to_string(),
            vendor: "v".to_string(),
            version: "2.4".to_string(),
            model_type: "birdnet-v24".to_string(),
            license: license(false, true),
            files: Some(ModelFiles {
                model: FileInfo {
                    url: "https://huggingface.co/o/r/resolve/main/birdnet.onnx".to_string(),
                    filename: "birdnet.onnx".to_string(),
                    sha256: None,
                    size_bytes: Some(50),
                },
                labels: LabelsInfo {
                    default_language: "en".to_string(),
                    languages: vec![
                        LanguageVariant {
                            code: "fr".to_string(),
                            name: "French".to_string(),
                            url: "https://example.com/labels-fr.txt".to_string(),
                            filename: "labels-fr.txt".to_string(),
                        },
                        LanguageVariant {
                            code: "en".to_string(),
                            name: "English".to_string(),
                            url: "https://example.com/labels-en.txt".to_string(),
                            filename: "labels-en.txt".to_string(),
                        },
                    ],
                },
                bsg_calibration: None,
                bsg_migration: None,
                bsg_distribution_maps: None,
            }),
            build: None,
            default_variant: None,
            selection: std::collections::BTreeMap::new(),
            variants: Vec::new(),
            recommended: false,
        }
    }

    #[test]
    fn test_project_manifest_keeps_every_region_variant_combination() {
        // Unlike regions(), which lists one tile per region, the manifest keeps
        // every combination so a consumer can enumerate every download.
        let manifest = project_manifest(&projection_entry());
        assert_eq!(manifest.variants.len(), 4);
        assert_eq!(manifest.default_variant.as_deref(), Some("fp32"));
        assert_eq!(
            manifest.selection.get("cuda").map(String::as_str),
            Some("fp16")
        );
        // A resolved URL is present; it is identity without a mirror, and the
        // filename stem survives the rewrite either way.
        assert!(manifest.variants[0].model_url.contains("fp32"));
        assert!(manifest.variants[0].labels_url.contains("labels"));
    }

    #[test]
    fn test_project_manifest_carries_countries_only_on_regional_variants() {
        let manifest = project_manifest(&projection_entry());
        let nordic = manifest
            .variants
            .iter()
            .find(|v| v.region.as_deref() == Some("nordic"))
            .unwrap();
        assert_eq!(
            nordic.countries.as_ref().unwrap().core,
            vec!["Brazil".to_string()]
        );
        let global = manifest
            .variants
            .iter()
            .find(|v| v.region.is_none())
            .unwrap();
        assert!(global.countries.is_none());
    }

    #[test]
    fn test_project_manifest_synthesizes_one_global_variant_for_a_legacy_model() {
        // birdnet-v24 has `files`, not `variants`; it must still project to one
        // uniform variant so a consumer never branches on an empty list.
        let manifest = project_manifest(&legacy_projection_entry());
        assert_eq!(manifest.variants.len(), 1);
        let only = &manifest.variants[0];
        assert_eq!(only.id, LEGACY_VARIANT_ID);
        assert!(only.region.is_none());
        assert!(only.countries.is_none());
        assert!(only.classes.is_none());
        // The default-language labels file, not simply the first listed.
        assert!(
            only.labels_url.contains("labels-en"),
            "should pick the default language, got: {}",
            only.labels_url
        );
    }
}
