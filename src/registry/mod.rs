//! Model registry system for discovering and installing models.

#![allow(clippy::print_stdout)]

pub mod installer;
pub mod license;
pub mod loader;
pub mod selection;
pub mod types;

// Re-export commonly used types and functions
pub use installer::{
    GEOMODEL_INSTALL_ID, InstalledRangeFilter, download_file, find_obsolete_files, geomodel_paths,
    install_model, install_range_filter, models_dir,
};
pub use license::{LicensedAsset, prompt_license_acceptance};
pub use loader::{find_model, load_registry};
pub use selection::{HardwareProbe, SelectionReason, SystemProbe, VariantChoice, select_variant};
pub use types::{
    FileInfo, LabelsInfo, LanguageVariant, LicenseInfo, ModelEntry, ModelFiles, ModelVariant,
    RangeFilterAsset, Registry,
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
                "  Global model: {} species, {}",
                global.classes,
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

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Test setup code - panics are acceptable
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
}

/// Show available languages for a model.
pub fn show_languages(registry: &Registry, id: &str) -> Result<()> {
    let model = find_model(registry, id)
        .ok_or_else(|| Error::ModelNotFoundInRegistry { id: id.to_string() })?;

    // Variant-based families publish one labels file per region, in English
    // only, so there are no language variants to list. Saying that is more use
    // than printing an empty list.
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
