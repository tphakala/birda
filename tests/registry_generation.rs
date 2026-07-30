//! The checked-in `registry.json` must equal what the generator produces.
//!
//! Without this, a manifest can gain a region or a corrected checksum and the
//! gallery silently keeps shipping the old one. Run under the maintenance
//! feature, which is where the generator lives:
//!
//! ```text
//! cargo test --features gen-registry --test registry_generation
//! ```

#![cfg(feature = "gen-registry")]

use birda::registry::Registry;

fn generated() -> String {
    birda::gen_registry::generate_from_repo_root(env!("CARGO_MANIFEST_DIR"))
        .expect("generation succeeds")
}

fn parsed() -> Registry {
    serde_json::from_str(&generated()).expect("the generated registry parses")
}

#[test]
fn test_checked_in_registry_matches_the_generated_one() {
    let checked_in = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("registry.json"),
    )
    .expect("registry.json is readable");

    assert_eq!(
        generated().trim_end(),
        checked_in.trim_end(),
        "registry.json is stale. Regenerate with: \
         cargo run --features gen-registry --bin gen-registry"
    );
}

#[test]
fn test_every_generated_variant_has_a_checksum_and_a_size() {
    // A variant without these cannot be verified after download, and cannot
    // tell the user what the download costs before they agree to it.
    for model in &parsed().models {
        for variant in &model.variants {
            assert!(
                variant.model.sha256.is_some(),
                "{} variant {} has no checksum",
                model.id,
                variant.id
            );
            assert!(
                variant.model.size_bytes.is_some_and(|s| s > 0),
                "{} variant {} has no size",
                model.id,
                variant.id
            );
        }
    }
}

#[test]
fn test_no_variant_claims_zero_classes() {
    // A missing class count must stay missing rather than defaulting to zero.
    // Perch declares counts only in its per-region metadata, so an unwrap_or_default
    // in the generator printed "0 species" next to a working 62 MB model.
    for model in parsed().models.iter().filter(|m| m.is_variant_based()) {
        for variant in &model.variants {
            assert_ne!(
                variant.classes,
                Some(0),
                "{} variant {} claims zero classes",
                model.id,
                variant.id
            );
        }
    }
}

#[test]
fn test_every_regional_variant_states_its_class_count() {
    // The count is the whole point of choosing a region, so a regional tile
    // without one means the metadata vendoring missed that region.
    for model in parsed().models.iter().filter(|m| m.is_variant_based()) {
        for variant in model.variants.iter().filter(|v| v.region.is_some()) {
            assert!(
                variant.classes.is_some(),
                "{} region {:?} has no class count",
                model.id,
                variant.region
            );
        }
    }
}

#[test]
fn test_every_regional_variant_has_display_metadata() {
    // Without a group the region falls into an "Other" bucket in the listing,
    // which is what a missing metadata.json looks like from the outside.
    for model in parsed().models.iter().filter(|m| m.is_variant_based()) {
        for variant in model.variants.iter().filter(|v| v.region.is_some()) {
            assert!(
                variant.region_name.is_some() && variant.group_name.is_some(),
                "{} region {:?} has no display metadata",
                model.id,
                variant.region
            );
        }
    }
}

#[test]
fn test_every_variant_entry_declares_a_default_variant_that_exists() {
    // Selection falls back to this, so an entry naming a variant it does not
    // publish would fail at install time on exactly the hosts with no better
    // signal, which is most of them.
    for model in parsed().models.iter().filter(|m| m.is_variant_based()) {
        let default = model
            .default_variant
            .as_deref()
            .unwrap_or_else(|| panic!("{} publishes variants but names no default", model.id));
        assert!(
            model.find_variant(None, default).is_some(),
            "{} defaults to variant {default}, which it does not publish globally",
            model.id
        );
    }
}

#[test]
fn test_every_region_publishes_the_default_variant() {
    // Auto-selection degrades to the default when a region lacks the variant a
    // hardware key names. If a region lacks the default too, that last rung
    // fails and the region is uninstallable without an explicit --variant.
    for model in parsed().models.iter().filter(|m| m.is_variant_based()) {
        let Some(default) = model.default_variant.as_deref() else {
            continue;
        };
        for region in model.regions() {
            let slug = region.region.as_deref().unwrap_or_default();
            assert!(
                model.find_variant(Some(slug), default).is_some(),
                "{} region {slug} does not publish the default variant {default}",
                model.id
            );
        }
    }
}

#[test]
fn test_variant_entries_omit_the_legacy_file_set() {
    // The downgrade guard: an older birda declares `files` as required, so its
    // parse of this registry must fail and send it back to its own bundled
    // copy rather than let it install a model type it does not have.
    for model in parsed().models.iter().filter(|m| m.is_variant_based()) {
        assert!(
            model.files.is_none(),
            "{} carries both variants and a legacy file set",
            model.id
        );
    }
}

#[test]
fn test_the_frozen_legacy_entries_survive_regeneration() {
    // birdnet-v24 and bsg-fi-v44 are not in registry-sources.toml. They must be
    // carried through untouched rather than dropped.
    let registry = parsed();
    for id in ["birdnet-v24", "bsg-fi-v44"] {
        let entry = registry
            .models
            .iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("{id} was dropped by regeneration"));
        assert!(entry.files.is_some(), "{id} lost its file set");
        assert!(!entry.is_variant_based(), "{id} was converted unasked");
    }

    assert!(
        registry.range_filter.is_some(),
        "the shared range filter asset was dropped by regeneration"
    );
}

#[test]
fn test_every_url_points_at_the_repository_the_manifest_names() {
    for model in parsed().models.iter().filter(|m| m.is_variant_based()) {
        for variant in &model.variants {
            for url in [&variant.model.url, &variant.labels.url] {
                assert!(
                    url.starts_with("https://huggingface.co/tphakala/"),
                    "{} variant {} points outside the published repositories: {url}",
                    model.id,
                    variant.id
                );
            }
        }
    }
}

#[test]
fn test_no_two_variants_share_a_filename() {
    // Files land in one flat models directory, so a collision would have two
    // different models overwrite each other. The published naming scheme
    // prevents this; this asserts the generator did not flatten it away.
    for model in parsed().models.iter().filter(|m| m.is_variant_based()) {
        let mut seen = std::collections::HashSet::new();
        for variant in &model.variants {
            assert!(
                seen.insert(variant.model.filename.clone()),
                "{} has two variants named {}",
                model.id,
                variant.model.filename
            );
        }
    }
}

#[test]
fn test_the_selection_map_names_only_variants_that_exist() {
    // The manifest maps hardware keys to file paths; the generator translates
    // them to variant ids. A key surviving that translation with an id nothing
    // publishes would silently resolve to nothing at install time.
    for model in parsed().models.iter().filter(|m| m.is_variant_based()) {
        for (key, id) in &model.selection {
            assert!(
                model.find_variant(None, id).is_some(),
                "{} maps {key} to variant {id}, which it does not publish globally",
                model.id
            );
        }
    }
}
