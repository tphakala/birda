//! The checked-in `registry.json` must equal what the generator produces.
//!
//! Without this, a manifest can gain a region or a corrected checksum and the
//! gallery silently keeps shipping the old one. Run under the maintenance
//! feature, which is where the generator lives:
//!
//! ```text
//! cargo test --features gen-registry --test registry_generation
//! ```
// Integration test crate. `unwrap`, `expect` and `panic` are how a test reports
// failure, not unhandled error paths, so rewriting them into propagated errors
// would only hide which assertion fired. Every exact float assertion in these
// tests is on a passed-through value (a literal parsed from a file, a
// coordinate round-tripped through JSON, a clip boundary clamped to a whole
// number) rather than a computed one, so exact equality is the assertion the
// test wants. The crate-level deny still governs everything birda ships.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
#![cfg(feature = "gen-registry")]

use birda::registry::{BoundingBox, GeoPoint, Registry};

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
fn test_every_regional_variant_carries_country_coverage() {
    // The vendored region metadata declares core/partial countries, and the
    // manifest projection exposes them as the search index for the region
    // picker. If the generator drops them, that search has nothing to match.
    for model in parsed().models.iter().filter(|m| m.is_variant_based()) {
        for variant in model.variants.iter().filter(|v| v.region.is_some()) {
            let countries = variant.countries.as_ref().unwrap_or_else(|| {
                panic!(
                    "{} region {:?} lost its countries",
                    model.id, variant.region
                )
            });
            assert!(
                !countries.core.is_empty(),
                "{} region {:?} has empty core country coverage",
                model.id,
                variant.region
            );
        }
    }
}

#[test]
fn test_global_variants_carry_no_country_coverage() {
    // The global model is not a region, so it must not gain a countries object;
    // that is what keeps the field a reliable regional/global discriminator.
    for model in parsed().models.iter().filter(|m| m.is_variant_based()) {
        for variant in model.variants.iter().filter(|v| v.region.is_none()) {
            assert!(
                variant.countries.is_none(),
                "{} global variant {} must not carry countries",
                model.id,
                variant.id
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
fn test_every_variant_of_an_entry_points_at_one_repository() {
    // Not a hardcoded owner: the assertion is that the generator kept every
    // file of an entry inside the single repository its manifest names, which
    // stays true if a manifest for a different owner is added later.
    for model in parsed().models.iter().filter(|m| m.is_variant_based()) {
        let first = &model.variants[0].model.url;
        let repo_prefix = first
            .split("/resolve/")
            .next()
            .expect("a resolve URL has a prefix");

        for variant in &model.variants {
            for url in [&variant.model.url, &variant.labels.url] {
                assert!(
                    url.starts_with(repo_prefix),
                    "{} variant {} points outside {repo_prefix}: {url}",
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

/// Every hardware key each source manifest declares, minus template values.
///
/// Read from the manifest rather than from the generated registry, so this is a
/// statement about what the publisher asked for rather than a restatement of
/// what the generator did.
fn manifest_selection_keys() -> Vec<(String, Vec<String>)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("manifests");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("manifests directory is readable") {
        let path = entry.expect("directory entry is readable").path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !name.ends_with(".models.json") {
            continue;
        }
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("manifest is readable"))
                .expect("manifest is valid JSON");
        let repo = manifest["repo"].as_str().unwrap_or_default().to_string();
        let keys = manifest["selection"]
            .as_object()
            .map(|map| {
                map.iter()
                    // A template value names a shape, not a file. Perch's
                    // low-ram key is `regional/<region>/...`, which says "use a
                    // regional model": a choice along the region axis, which the
                    // gallery exposes through --region rather than by guessing a
                    // continent for the user.
                    .filter(|(_, value)| !value.as_str().unwrap_or_default().contains('<'))
                    .map(|(key, _)| key.clone())
                    .collect()
            })
            .unwrap_or_default();
        out.push((repo, keys));
    }
    assert!(!out.is_empty(), "no manifests were found to check against");
    out
}

#[test]
fn test_no_hardware_key_from_a_manifest_is_silently_dropped() {
    // The earlier version of this test compared the generated selection against
    // the generated variants, which is the generator agreeing with itself. It
    // could not have noticed a hardware key vanishing in translation. This
    // compares against the source instead.
    let registry = parsed();

    for (repo, keys) in manifest_selection_keys() {
        let model = registry
            .models
            .iter()
            .find(|m| {
                m.variants
                    .first()
                    .is_some_and(|v| v.model.url.contains(&repo))
            })
            .unwrap_or_else(|| panic!("no generated entry sources its files from {repo}"));

        for key in keys {
            assert!(
                model.selection.contains_key(&key),
                "{} lost hardware key {key} from {repo}'s manifest",
                model.id
            );
        }
    }
}

/// Generate from a temp root holding the schema 2 fixture family (wrapped
/// regions, a groups file, documents, per-file builds, labels checksums) and
/// the real country table, with `registry.json` as given.
fn generate_schema_2_family(registry_json: &str) -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let temp = tempfile::tempdir().unwrap();
    let manifests = temp.path().join("manifests");
    std::fs::create_dir(&manifests).unwrap();

    let fixtures = root.join("tests/fixtures/registry-schema2");
    for name in [
        "BirdNET-Plus-V3.0.models.json",
        "BirdNET-Plus-V3.0.regions.json",
        "BirdNET-Plus-V3.0.groups.json",
    ] {
        std::fs::copy(fixtures.join(name), manifests.join(name)).unwrap();
    }
    std::fs::copy(
        root.join("manifests/iso3166-1-alpha2.json"),
        manifests.join("iso3166-1-alpha2.json"),
    )
    .unwrap();
    std::fs::write(
        temp.path().join("registry-sources.toml"),
        r#"schema_version = "2.0"

[[model]]
id = "birdnet-plus"
manifest = "BirdNET-Plus-V3.0.models.json"
regions = "BirdNET-Plus-V3.0.regions.json"
groups = "BirdNET-Plus-V3.0.groups.json"
name = "BirdNET+ V3.0"
description = "fixture"
vendor = "fixture"
model_type = "birdnet-v30"
default_variant = "fp32"

[model.license]
type = "Apache-2.0"
url = "https://www.apache.org/licenses/LICENSE-2.0"
commercial_use = true
attribution_required = true
share_alike = false
"#,
    )
    .unwrap();
    std::fs::write(temp.path().join("registry.json"), registry_json).unwrap();

    birda::gen_registry::generate_from_repo_root(temp.path().to_str().unwrap())
        .expect("the schema 2 fixture generates")
}

#[test]
fn test_a_schema_2_family_generates_and_regenerates_unchanged() {
    let first =
        generate_schema_2_family(r#"{"schema_version":"2.0","registry_version":1,"models":[]}"#);
    let registry: Registry = serde_json::from_str(&first).unwrap();
    // New content over a registry at version 1 bumps exactly once.
    assert_eq!(registry.registry_version, 2);
    let entry = &registry.models[0];
    let variant = |region: Option<&str>| {
        entry
            .variants
            .iter()
            .find(|v| v.region.as_deref() == region)
            .unwrap_or_else(|| panic!("no variant for {region:?}"))
    };

    // The superseded nordic b1 file is dropped, so the first publishable file
    // is the global b1 and the regions state a build only where it differs.
    assert_eq!(entry.build, Some(1));
    assert_eq!(variant(None).build, None);
    assert_eq!(variant(Some("nordic")).build, Some(2));
    assert_eq!(variant(Some("central-india")).build, Some(3));
    assert_eq!(entry.variants.len(), 4);

    let nordic = variant(Some("nordic"));
    assert_eq!(
        nordic.countries.as_ref().unwrap().core,
        vec!["Åland Islands", "Denmark", "Finland", "Norway", "Sweden"]
    );
    assert_eq!(
        nordic.country_codes.as_ref().unwrap().core,
        vec!["AX", "DK", "FI", "NO", "SE"]
    );
    assert_eq!(
        nordic.labels.sha256.as_deref(),
        Some("5f18020e5c3dc5af6f8961da6b37afd9e7ace033f764fc1a2c8c988088e8e71a")
    );
    assert_eq!(nordic.group.as_deref(), Some("europe"));
    assert_eq!(nordic.group_order, 0);
    assert_eq!(variant(Some("central-india")).group_order, 1);
    assert_eq!(nordic.tier, Some(50));
    assert_eq!(
        nordic.centroid,
        Some(GeoPoint {
            lat: 62.88,
            lon: 17.685,
        })
    );
    assert_eq!(
        nordic.bboxes,
        vec![BoundingBox {
            lat_min: 53.1,
            lat_max: 72.66,
            lon_min: 3.32,
            lon_max: 32.05,
        }]
    );
    let western_palearctic = variant(Some("western-palearctic"));
    assert_eq!(western_palearctic.tier, Some(10));
    assert_eq!(western_palearctic.bboxes.len(), 5);
    assert_eq!(
        western_palearctic.bboxes[1],
        BoundingBox {
            lat_min: 26.0,
            lat_max: 40.0,
            lon_min: -32.0,
            lon_max: -13.0,
        }
    );
    // The global model is not a region, so it has no geometry.
    assert_eq!(variant(None).tier, None);
    assert_eq!(variant(None).centroid, None);
    assert_eq!(variant(None).bboxes, Vec::<BoundingBox>::new());

    let documents: Vec<(&str, bool, &str)> = entry
        .documents
        .iter()
        .map(|d| (d.kind.as_str(), d.requires_agreement, d.url.as_str()))
        .collect();
    assert_eq!(
        documents,
        vec![
            (
                "license",
                false,
                "https://huggingface.co/tphakala/BirdNET-Plus-V3.0/resolve/\
                 f90c3ba818b179adf5f4c9eabf2e6f8b801e4c06/LICENSE-v3.0.1.md"
            ),
            (
                "acceptable_use",
                true,
                "https://huggingface.co/tphakala/BirdNET-Plus-V3.0/resolve/\
                 f90c3ba818b179adf5f4c9eabf2e6f8b801e4c06/ACCEPTABLE_USE-v3.0.1.md"
            ),
        ]
    );

    // Feed the output back in as the registry on disk. The fixture carries
    // floating-point geometry, which the vendored files do not, so this is the
    // run that proves the version stays put once the content stops changing
    // with geometry present (`test_manifest_floats_parse_to_the_nearest_double`
    // pins the float parsing itself).
    let second = generate_schema_2_family(&first);
    assert_eq!(second, first);
    assert_eq!(
        serde_json::from_str::<Registry>(&second)
            .unwrap()
            .registry_version,
        2
    );
}
