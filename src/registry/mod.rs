//! Model registry system for discovering and installing models.

#![allow(clippy::print_stdout)]

pub mod installer;
pub mod license;
pub mod loader;
pub mod types;

// Re-export commonly used types and functions
pub use installer::{download_file, install_model, models_dir};
pub use license::prompt_license_acceptance;
pub use loader::{find_model, load_registry};
pub use types::{
    FileInfo, LabelsInfo, LanguageVariant, LicenseInfo, ModelEntry, ModelFiles, Registry,
};

use crate::error::{Error, Result};

/// List all available models from the registry.
pub fn list_available(registry: &Registry) {
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

        let license_note = if model.license.commercial_use {
            &model.license.r#type
        } else {
            &format!("{} (non-commercial)", model.license.r#type)
        };
        println!("    License: {license_note}");
        println!();
    }

    println!("Run 'birda models info <id>' for details.");
}

/// Show detailed information about a specific model.
pub fn show_info(registry: &Registry, id: &str) -> Result<()> {
    let model = find_model(registry, id)
        .ok_or_else(|| Error::ModelNotFoundInRegistry { id: id.to_string() })?;

    println!("Model: {}", model.name);
    println!("ID: {}", model.id);
    println!("Version: {}", model.version);
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

    println!("Files:");
    println!("  Model: {}", model.files.model.url);

    let lang_count = model.files.labels.languages.len();
    let default_lang = model
        .files
        .labels
        .languages
        .iter()
        .find(|l| l.code == model.files.labels.default_language)
        .map_or("Unknown", |l| l.name.as_str());

    if lang_count == 1 {
        println!("  Labels: {default_lang} only");
    } else {
        println!("  Labels: {lang_count} languages available (default: {default_lang})");
    }
    println!();

    println!("To install: birda models install {}", model.id);

    Ok(())
}

/// Show available languages for a model.
pub fn show_languages(registry: &Registry, id: &str) -> Result<()> {
    let model = find_model(registry, id)
        .ok_or_else(|| Error::ModelNotFoundInRegistry { id: id.to_string() })?;

    println!("Model: {}", model.name);
    println!();
    println!("Available label languages:");
    println!();

    for lang in &model.files.labels.languages {
        let default_marker = if lang.code == model.files.labels.default_language {
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
