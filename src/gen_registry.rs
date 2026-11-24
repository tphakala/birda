//! Generating `registry.json` from published `models.json` manifests.
//!
//! Feature-gated behind `gen-registry` because it is a maintenance tool, not
//! part of the shipped CLI. Written in Rust rather than as a script so it
//! serializes the very types the CLI deserializes: a registry this produces
//! cannot fail to load.
//!
//! Inputs are vendored under `manifests/`, so generation and the test that
//! guards against drift are hermetic and need no network. They are the
//! publisher's `models.json` (schema 1 or 2), the matching `regions.json` (flat
//! with country names, or wrapped with ISO 3166-1 alpha-2 codes), the shared
//! country-name table, and, for a wrapped regions file, a `groups.json` that
//! orders the regions.

use crate::constants::coordinates::{LATITUDE_MAX, LATITUDE_MIN, LONGITUDE_MAX, LONGITUDE_MIN};
use crate::error::{Error, Result};
use crate::registry::types::{
    BoundingBox, Countries, FileInfo, GeoPoint, LicenseInfo, ModelDocument, ModelEntry,
    ModelVariant, Registry, ScoreCalibration,
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
/// ISO 3166-1 alpha-2 code to English country name, inside [`MANIFEST_DIR`].
const COUNTRY_NAMES_FILE: &str = "iso3166-1-alpha2.json";
/// The `models.json` schema versions this generator reads.
const SUPPORTED_MANIFEST_SCHEMAS: [u32; 2] = [1, 2];
/// The only wrapped `regions.json` schema version this generator reads.
const WRAPPED_REGIONS_SCHEMA: u32 = 1;
/// Top-level key whose presence marks a `regions.json` as the wrapped form.
const WRAPPED_REGIONS_KEY: &str = "schema";

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
    #[serde(default)]
    labels_sha256: Option<String>,
    #[serde(default)]
    labels_size_bytes: Option<u64>,
    upstream_version: String,
    build: u32,
    sha256: String,
    size_bytes: u64,
    #[serde(default)]
    superseded_by: Option<String>,
    /// `null` or absent means live; any JSON object means withdrawn, and its
    /// `date` and `reason` are not needed here. `false`, a string and an array
    /// fail the parse rather than being read one way or the other: a map type
    /// refuses a sequence, which an empty braced struct would accept as `[]`.
    #[serde(default)]
    withdrawn: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    score_calibration: Option<ScoreCalibration>,
}

/// A licence or usage document in `family_meta`. Published either as a `path`
/// inside the repository at a pinned `revision`, or as a plain `url`.
#[derive(Debug, Deserialize)]
struct ManifestDocument {
    kind: String,
    requires_agreement: bool,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    revision: Option<String>,
    #[serde(default)]
    sha256: Option<String>,
}

/// The part of `family_meta` the registry carries.
#[derive(Debug, Deserialize)]
struct FamilyMeta {
    #[serde(default)]
    documents: Vec<ManifestDocument>,
}

/// Just the `schema` number of a `models.json`, read before the rest so an
/// unsupported schema is reported as such rather than as whichever field
/// changed.
#[derive(Debug, Deserialize)]
struct SchemaProbe {
    schema: u32,
}

/// A repository's `models.json`.
#[derive(Debug, Deserialize)]
struct Manifest {
    repo: String,
    #[serde(default)]
    family_meta: Option<FamilyMeta>,
    #[serde(default)]
    selection: BTreeMap<String, String>,
    models: Vec<ManifestModel>,
}

/// Display metadata for one region in the flat `regions.json` form, which
/// carries country names and its own group order.
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
    /// Country names this region covers, used as written so a consumer can
    /// offer a country-name search over the region list.
    #[serde(default)]
    countries: Option<Countries>,
}

/// The wrapped `regions.json` form: ISO country codes and geometry, with the
/// group and its order taken from the groups file instead.
#[derive(Debug, Deserialize)]
struct WrappedRegions {
    regions: BTreeMap<String, WrappedRegion>,
}

/// One region of the wrapped form. Geometry is required: every published region
/// carries it, and a region that cannot be placed on a map is not worth listing.
#[derive(Debug, Deserialize)]
struct WrappedRegion {
    name: String,
    tier: u32,
    /// Each box is `[lat_min, lat_max, lon_min, lon_max]`.
    bboxes: Vec<[f64; 4]>,
    /// `[lat, lon]`.
    centroid: [f64; 2],
    /// ISO 3166-1 alpha-2 codes.
    countries: Countries,
    #[serde(default)]
    classes: Option<usize>,
}

/// A family's `groups.json`: which tiles sit in which continental group, and
/// the order the groups are shown in.
#[derive(Debug, Deserialize)]
struct GroupsFile {
    groups: Vec<GroupEntry>,
}

/// One continental group of a `groups.json`.
#[derive(Debug, Deserialize)]
struct GroupEntry {
    group: String,
    display: String,
    order: u32,
    tiles: Vec<String>,
}

/// Where a tile sits: its group's slug and display name, and the group's order.
#[derive(Debug)]
struct Placement {
    group: String,
    display: String,
    order: u32,
}

/// ISO 3166-1 alpha-2 code to English country name.
#[derive(Debug, Deserialize)]
struct CountryNames {
    names: BTreeMap<String, String>,
}

impl CountryNames {
    fn parse(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|e| Error::Internal {
            message: format!("{COUNTRY_NAMES_FILE} is not a valid country table: {e}"),
        })
    }

    /// Replace each code in `codes` by its name, keeping the publisher's order.
    fn translate(&self, regions_file: &str, region: &str, codes: &Countries) -> Result<Countries> {
        let name = |code: &String| -> Result<String> {
            self.names
                .get(code)
                .cloned()
                .ok_or_else(|| Error::Internal {
                    message: format!(
                        "{regions_file}: region {region} lists unknown country code {code}"
                    ),
                })
        };
        Ok(Countries {
            core: codes.core.iter().map(name).collect::<Result<_>>()?,
            partial: codes.partial.iter().map(name).collect::<Result<_>>()?,
        })
    }
}

/// A region after the flat and wrapped forms are reduced to one shape.
#[derive(Debug)]
struct Region {
    name: String,
    group: Option<String>,
    group_name: Option<String>,
    group_order: u32,
    classes: Option<usize>,
    countries: Option<Countries>,
    country_codes: Option<Countries>,
    tier: Option<u32>,
    bboxes: Vec<BoundingBox>,
    centroid: Option<GeoPoint>,
}

/// Curation entry from `registry-sources.toml`.
#[derive(Debug, Deserialize)]
struct SourceModel {
    id: String,
    manifest: String,
    regions: String,
    /// The family's `groups.json`: required exactly when `regions` is wrapped.
    #[serde(default)]
    groups: Option<String>,
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

    let manifest_dir = root.join(MANIFEST_DIR);
    let country_names =
        CountryNames::parse(&read_to_string(&manifest_dir.join(COUNTRY_NAMES_FILE))?)?;

    let generated_ids: Vec<&str> = sources.models.iter().map(|m| m.id.as_str()).collect();
    let mut models: Vec<ModelEntry> = Vec::new();
    for source in &sources.models {
        models.push(load_entry(&manifest_dir, source, &country_names)?);
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
        // Hand-maintained like `range_filter`: carried through verbatim so a
        // `gen-registry` run never drops the bat catalog. Edits to it, including
        // the `registry_version` bump, are made directly in `registry.json`.
        bat: existing.bat.clone(),
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

/// Read one source's manifest, regions and groups files and build its entry.
fn load_entry(
    manifest_dir: &Path,
    source: &SourceModel,
    country_names: &CountryNames,
) -> Result<ModelEntry> {
    let manifest = parse_manifest(
        &source.manifest,
        &read_to_string(&manifest_dir.join(&source.manifest))?,
    )?;
    let groups = match &source.groups {
        Some(file) => Some((file.as_str(), read_to_string(&manifest_dir.join(file))?)),
        None => None,
    };
    let regions = parse_regions(
        &source.regions,
        &read_to_string(&manifest_dir.join(&source.regions))?,
        groups.as_ref().map(|(file, text)| (*file, text.as_str())),
        country_names,
    )?;
    build_entry(source, &manifest, &regions)
}

/// Parse a `models.json`, refusing a schema this generator does not know.
///
/// The schema is read first so a future schema that renames a field is reported
/// as an unsupported schema, not as a missing field, or worse, read into a
/// registry with silently empty checksums.
fn parse_manifest(file_name: &str, text: &str) -> Result<Manifest> {
    let invalid = |e: serde_json::Error| Error::Internal {
        message: format!("{file_name} is not a valid manifest: {e}"),
    };
    let SchemaProbe { schema } = serde_json::from_str(text).map_err(invalid)?;
    if !SUPPORTED_MANIFEST_SCHEMAS.contains(&schema) {
        return Err(Error::Internal {
            message: format!(
                "{file_name} has manifest schema {schema}, which this generator does not read \
                 (it reads {SUPPORTED_MANIFEST_SCHEMAS:?})"
            ),
        });
    }
    serde_json::from_str(text).map_err(invalid)
}

/// Parse a `regions.json` in either form into one shape.
///
/// The wrapped form is recognised by its top-level `schema` key. `groups` is
/// the family's `groups.json` as `(file name, text)`: required for the wrapped
/// form, which carries no group order of its own, and refused for the flat
/// form, which does, because two sources of group order would disagree.
fn parse_regions(
    file_name: &str,
    text: &str,
    groups: Option<(&str, &str)>,
    country_names: &CountryNames,
) -> Result<BTreeMap<String, Region>> {
    let invalid = |e: serde_json::Error| Error::Internal {
        message: format!("{file_name} is not valid region metadata: {e}"),
    };
    let top: serde_json::Value = serde_json::from_str(text).map_err(invalid)?;
    let Some(schema) = top.get(WRAPPED_REGIONS_KEY) else {
        return parse_flat_regions(file_name, text, groups);
    };

    if schema.as_u64() != Some(u64::from(WRAPPED_REGIONS_SCHEMA)) {
        return Err(Error::Internal {
            message: format!(
                "{file_name} has regions schema {schema}, which this generator does not read \
                 (it reads {WRAPPED_REGIONS_SCHEMA})"
            ),
        });
    }
    let wrapped: WrappedRegions = serde_json::from_str(text).map_err(invalid)?;
    let (groups_file, groups_text) = groups.ok_or_else(|| Error::Internal {
        message: format!(
            "{file_name} lists countries by ISO code and has no group order of its own: \
             name a groups file for it with `groups` in {SOURCES_FILE}"
        ),
    })?;
    let tile_groups = parse_groups(groups_file, groups_text)?;

    let mut regions = BTreeMap::new();
    for (slug, region) in wrapped.regions {
        let group = tile_groups.get(&slug).ok_or_else(|| Error::Internal {
            message: format!("{file_name}: region {slug} is in no group of {groups_file}"),
        })?;
        let bboxes = region
            .bboxes
            .iter()
            .map(|b| validate_box(file_name, &slug, *b))
            .collect::<Result<Vec<_>>>()?;
        if bboxes.is_empty() {
            return Err(Error::Internal {
                message: format!("{file_name}: region {slug} has no bounding box"),
            });
        }
        let centroid = validate_point(file_name, &slug, region.centroid)?;
        let countries = country_names.translate(file_name, &slug, &region.countries)?;

        regions.insert(
            slug,
            Region {
                name: region.name,
                group: Some(group.group.clone()),
                group_name: Some(group.display.clone()),
                group_order: group.order,
                classes: region.classes,
                countries: Some(countries),
                country_codes: Some(region.countries),
                tier: Some(region.tier),
                bboxes,
                centroid: Some(centroid),
            },
        );
    }
    Ok(regions)
}

/// The flat `regions.json` form: names as written, group order of its own.
fn parse_flat_regions(
    file_name: &str,
    text: &str,
    groups: Option<(&str, &str)>,
) -> Result<BTreeMap<String, Region>> {
    if let Some((groups_file, _)) = groups {
        return Err(Error::Internal {
            message: format!(
                "{file_name} carries its own group order, so {groups_file} would be a second \
                 source of it: remove `groups` from its entry in {SOURCES_FILE}"
            ),
        });
    }
    let flat: BTreeMap<String, RegionMetadata> =
        serde_json::from_str(text).map_err(|e| Error::Internal {
            message: format!("{file_name} is not valid region metadata: {e}"),
        })?;
    Ok(flat
        .into_iter()
        .map(|(slug, m)| {
            let region = Region {
                name: m.name,
                group: m.group,
                group_name: m.group_display,
                group_order: m.group_order,
                classes: m.classes,
                countries: m.countries,
                country_codes: None,
                tier: None,
                bboxes: Vec::new(),
                centroid: None,
            };
            (slug, region)
        })
        .collect())
}

/// Map each tile to its group, refusing data that would make a tile's group or
/// a group's position ambiguous.
///
/// A tile listed here but absent from the regions file is fine: one groups file
/// serves every family, and a family publishes only some of its tiles.
fn parse_groups(file_name: &str, text: &str) -> Result<BTreeMap<String, Placement>> {
    let file: GroupsFile = serde_json::from_str(text).map_err(|e| Error::Internal {
        message: format!("{file_name} is not a valid groups file: {e}"),
    })?;

    let mut order_owner: BTreeMap<u32, &str> = BTreeMap::new();
    let mut tile_groups: BTreeMap<String, Placement> = BTreeMap::new();
    for group in &file.groups {
        if let Some(other) = order_owner.insert(group.order, &group.group) {
            return Err(Error::Internal {
                message: format!(
                    "{file_name}: groups {other} and {} share order {}",
                    group.group, group.order
                ),
            });
        }
        for tile in &group.tiles {
            let placement = Placement {
                group: group.group.clone(),
                display: group.display.clone(),
                order: group.order,
            };
            if let Some(other) = tile_groups.insert(tile.clone(), placement) {
                return Err(Error::Internal {
                    message: format!(
                        "{file_name}: tile {tile} is in both group {} and group {}",
                        other.group, group.group
                    ),
                });
            }
        }
    }
    Ok(tile_groups)
}

/// Whether every value is a latitude, bounds included.
fn valid_latitudes(values: &[f64]) -> bool {
    values
        .iter()
        .all(|v| (LATITUDE_MIN..=LATITUDE_MAX).contains(v))
}

/// Whether every value is a longitude, bounds included.
fn valid_longitudes(values: &[f64]) -> bool {
    values
        .iter()
        .all(|v| (LONGITUDE_MIN..=LONGITUDE_MAX).contains(v))
}

/// Check a `[lat_min, lat_max, lon_min, lon_max]` box and name its fields.
///
/// A box with `lon_min > lon_max` is refused rather than read as crossing the
/// antimeridian: the registry's [`BoundingBox`] never crosses it, so a region
/// that does must be published as two boxes.
fn validate_box(file_name: &str, region: &str, raw: [f64; 4]) -> Result<BoundingBox> {
    let [lat_min, lat_max, lon_min, lon_max] = raw;
    let fail = |what: String| Error::Internal {
        message: format!("{file_name}: region {region} has a bounding box {what}"),
    };
    if !valid_latitudes(&[lat_min, lat_max]) {
        return Err(fail(format!(
            "with a latitude outside {LATITUDE_MIN} to {LATITUDE_MAX} ({lat_min}, {lat_max})"
        )));
    }
    if !valid_longitudes(&[lon_min, lon_max]) {
        return Err(fail(format!(
            "with a longitude outside {LONGITUDE_MIN} to {LONGITUDE_MAX} ({lon_min}, {lon_max})"
        )));
    }
    if lat_min > lat_max {
        return Err(fail(format!(
            "whose southern edge {lat_min} is north of its northern edge {lat_max}"
        )));
    }
    if lon_min > lon_max {
        return Err(fail(format!(
            "that crosses the antimeridian (west edge {lon_min} is east of east edge {lon_max}); \
             publish it as two boxes"
        )));
    }
    Ok(BoundingBox {
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    })
}

/// Check a `[lat, lon]` centroid and name its fields.
fn validate_point(file_name: &str, region: &str, raw: [f64; 2]) -> Result<GeoPoint> {
    let [lat, lon] = raw;
    if !valid_latitudes(&[lat]) || !valid_longitudes(&[lon]) {
        return Err(Error::Internal {
            message: format!(
                "{file_name}: region {region} has a centroid outside the valid range ({lat}, {lon})"
            ),
        });
    }
    Ok(GeoPoint { lat, lon })
}

/// Turn the manifest's `family_meta` documents into registry documents.
///
/// A `path` document resolves to the repository file at the revision the
/// publisher states; a `url` document is passed through as the publisher wrote
/// it, keeping a `revision` it states.
fn resolve_documents(
    manifest_file: &str,
    repo: &str,
    documents: &[ManifestDocument],
) -> Result<Vec<ModelDocument>> {
    documents
        .iter()
        .map(|doc| {
            let fail = |why: &str| Error::Internal {
                message: format!("{manifest_file}: document {} {why}", doc.kind),
            };
            let (url, revision) = match (&doc.path, &doc.url) {
                (Some(path), None) => {
                    let revision = doc
                        .revision
                        .as_ref()
                        .ok_or_else(|| fail("has a path but no revision"))?;
                    (url_at(repo, revision, path), Some(revision.clone()))
                }
                (None, Some(url)) => (url.clone(), doc.revision.clone()),
                (Some(_), Some(_)) => return Err(fail("has both a path and a url")),
                (None, None) => return Err(fail("has neither a path nor a url")),
            };
            Ok(ModelDocument {
                kind: doc.kind.clone(),
                requires_agreement: doc.requires_agreement,
                url,
                revision,
                sha256: doc.sha256.clone(),
            })
        })
        .collect()
}

/// Turn one manifest plus its curation into a registry entry.
fn build_entry(
    source: &SourceModel,
    manifest: &Manifest,
    regions: &BTreeMap<String, Region>,
) -> Result<ModelEntry> {
    // A superseded file stays published so older clients keep working, but it
    // must never be offered to a new install. A withdrawn file is pulled
    // outright, with or without a replacement named.
    let publishable: Vec<&ManifestModel> = manifest
        .models
        .iter()
        .filter(|m| m.superseded_by.is_none() && m.withdrawn.is_none())
        .collect();

    let first = publishable.first().ok_or_else(|| Error::Internal {
        message: format!("{} lists no installable files", source.manifest),
    })?;
    // Deliberately not filtered by the `legacy` flag: that flag records whether
    // a filename predates the current naming scheme, which says nothing about
    // whether the version string is usable. Perch marks every file legacy, so
    // filtering on it would leave the entry with no version at all.
    let version = first.upstream_version.clone();
    // The entry keeps the first file's build; a file at another build states its
    // own on the variant, so the registry stays unchanged while every file
    // shares one build.
    let build = first.build;
    // One entry has one `version`, so files at different upstream versions would
    // mislabel all but one of them. A release replaces the old files with
    // `superseded_by` or `withdrawn`, which the filter above has already dropped.
    if let Some(odd) = publishable.iter().find(|m| m.upstream_version != version) {
        return Err(Error::Internal {
            message: format!(
                "{}: {} has upstream_version {} but {} has {version}; one entry holds one version",
                source.manifest, odd.path, odd.upstream_version, first.path
            ),
        });
    }

    let mut variants: Vec<ModelVariant> = Vec::new();
    let mut seen: BTreeMap<(Option<&str>, &str), &str> = BTreeMap::new();
    for model in &publishable {
        let labels = model.labels.as_ref().ok_or_else(|| Error::Internal {
            message: format!("{} has no labels file and cannot be installed", model.path),
        })?;
        // A region the regions file does not describe would be listed with no
        // name, group or countries, so refuse it rather than publish it bare.
        let region_meta = match &model.region {
            Some(region) => Some(
                regions
                    .get(region.as_str())
                    .ok_or_else(|| Error::Internal {
                        message: format!(
                            "{}: {} is in region {region}, which {} does not describe",
                            source.manifest, model.path, source.regions
                        ),
                    })?,
            ),
            None => None,
        };
        // `find_variant` takes the first file of a (region, variant) while
        // `ModelEntry::regions` takes the last file of the default variant, so a
        // duplicate would make install and the region listing disagree about
        // which file a tile is.
        if let Some(earlier) = seen.insert((model.region.as_deref(), &model.variant), &model.path) {
            return Err(Error::Internal {
                message: format!(
                    "{}: {earlier} and {} are both variant {} of {}",
                    source.manifest,
                    model.path,
                    model.variant,
                    model
                        .region
                        .as_ref()
                        .map_or_else(|| "the global model".to_string(), |r| format!("region {r}"))
                ),
            });
        }

        variants.push(ModelVariant {
            id: model.variant.clone(),
            region: model.region.clone(),
            region_name: region_meta.map(|m| m.name.clone()),
            group: region_meta.and_then(|m| m.group.clone()),
            group_name: region_meta.and_then(|m| m.group_name.clone()),
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
                sha256: model.labels_sha256.clone(),
                size_bytes: model.labels_size_bytes,
            },
            countries: region_meta.and_then(|m| m.countries.clone()),
            build: (model.build != build).then_some(model.build),
            country_codes: region_meta.and_then(|m| m.country_codes.clone()),
            tier: region_meta.and_then(|m| m.tier),
            bboxes: region_meta.map_or_else(Vec::new, |m| m.bboxes.clone()),
            centroid: region_meta.and_then(|m| m.centroid),
            score_calibration: model.score_calibration.clone(),
        });
    }

    let documents = match &manifest.family_meta {
        Some(meta) => resolve_documents(&source.manifest, &manifest.repo, &meta.documents)?,
        None => Vec::new(),
    };

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
        selection: translate_selection(manifest, &publishable)?,
        variants,
        documents,
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

/// Download URL for a path inside a Hugging Face repository at `revision`.
fn url_at(repo: &str, revision: &str, path: &str) -> String {
    format!("{HF_ORIGIN}/{repo}/resolve/{revision}/{path}")
}

/// Download URL for a published file, resolved from the branch.
fn file_url(repo: &str, path: &str) -> String {
    url_at(repo, HF_REVISION, path)
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
            bat: None,
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

    // ---- schema 2 inputs --------------------------------------------------

    const REGIONS_FILE: &str = "r.json";
    const GROUPS_FILE: &str = "g.json";
    const MANIFEST_FILE: &str = "m.json";

    /// A two-group `groups.json`: `nordic` and `iberia` in Europe, `japan` in Asia.
    const GROUPS: &str = r#"{"groups":[
        {"group":"europe","display":"Europe","order":0,"tiles":["nordic","iberia"]},
        {"group":"asia","display":"Asia","order":1,"tiles":["japan"]}]}"#;

    fn names() -> CountryNames {
        CountryNames::parse(r#"{"names":{"AX":"Åland Islands","FI":"Finland","NO":"Norway"}}"#)
            .unwrap()
    }

    /// A wrapped regions file holding `regions`, a comma-separated list of
    /// `"slug": {...}` members.
    fn wrapped(regions: &str) -> String {
        format!(r#"{{"schema":1,"repo":"o/r","updated":"x","regions":{{{regions}}}}}"#)
    }

    /// One wrapped region with the given `bboxes` and `centroid` JSON.
    fn region_json(slug: &str, bboxes: &str, centroid: &str) -> String {
        format!(
            r#""{slug}":{{"name":"Nordic","tier":50,"bboxes":{bboxes},"centroid":{centroid},
            "countries":{{"core":["AX","FI"],"partial":["NO"]}},"classes":464}}"#
        )
    }

    fn nordic() -> String {
        region_json("nordic", "[[53.1,72.66,3.32,32.05]]", "[62.88,17.685]")
    }

    fn message<T: std::fmt::Debug>(result: Result<T>) -> String {
        result.unwrap_err().to_string()
    }

    fn parse_wrapped(text: &str) -> Result<BTreeMap<String, Region>> {
        parse_regions(REGIONS_FILE, text, Some((GROUPS_FILE, GROUPS)), &names())
    }

    fn source() -> SourceModel {
        SourceModel {
            id: "fam".to_string(),
            manifest: MANIFEST_FILE.to_string(),
            regions: REGIONS_FILE.to_string(),
            groups: None,
            name: "Fam".to_string(),
            description: "d".to_string(),
            vendor: "v".to_string(),
            model_type: "birdnet-v30".to_string(),
            recommended: false,
            default_variant: "fp32".to_string(),
            license: LicenseInfo {
                r#type: "MIT".to_string(),
                url: "https://example.com".to_string(),
                commercial_use: true,
                attribution_required: false,
                share_alike: false,
            },
        }
    }

    /// One manifest file entry. `extra` is appended inside the object and must
    /// start with a comma when it is not empty.
    fn file_json(
        path: &str,
        variant: &str,
        region: Option<&str>,
        build: u32,
        extra: &str,
    ) -> String {
        let region = region.map_or_else(|| "null".to_string(), |r| format!("\"{r}\""));
        format!(
            r#"{{"path":"{path}","variant":"{variant}","region":{region},"upstream_version":"3.0.1",
            "build":{build},"labels":"labels/{variant}.txt","sha256":"aa","size_bytes":1{extra}}}"#
        )
    }

    fn manifest_json(schema: u32, family_meta: &str, files: &[String]) -> String {
        format!(
            r#"{{"schema":{schema},"repo":"o/r"{family_meta},"models":[{}]}}"#,
            files.join(",")
        )
    }

    fn entry_from(text: &str, regions: &BTreeMap<String, Region>) -> Result<ModelEntry> {
        build_entry(&source(), &parse_manifest(MANIFEST_FILE, text)?, regions)
    }

    #[test]
    fn test_wrapped_regions_translate_codes_and_carry_geometry() {
        let regions = parse_wrapped(&wrapped(&nordic())).unwrap();
        let nordic = &regions["nordic"];

        assert_eq!(
            nordic.countries,
            Some(Countries {
                core: vec!["Åland Islands".to_string(), "Finland".to_string()],
                partial: vec!["Norway".to_string()],
            })
        );
        assert_eq!(
            nordic.country_codes,
            Some(Countries {
                core: vec!["AX".to_string(), "FI".to_string()],
                partial: vec!["NO".to_string()],
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
        assert_eq!(
            nordic.centroid,
            Some(GeoPoint {
                lat: 62.88,
                lon: 17.685,
            })
        );
        assert_eq!(nordic.tier, Some(50));
    }

    #[test]
    fn test_an_unknown_country_code_fails_generation() {
        // One row per list, so neither can skip the check on its own.
        let in_core = nordic().replace("\"AX\"", "\"QQ\"");
        let in_partial = nordic().replace("\"NO\"", "\"QQ\"");
        for text in [in_core, in_partial] {
            assert_eq!(
                message(parse_wrapped(&wrapped(&text))),
                "internal error: r.json: region nordic lists unknown country code QQ"
            );
        }
    }

    #[test]
    fn test_group_placement_comes_from_the_groups_file() {
        // Each region names the wrong group on purpose: the groups file is the
        // only source, so the region's own group and display must be ignored.
        let nordic = nordic().replace(
            "\"name\":\"Nordic\"",
            "\"name\":\"Nordic\",\"group\":\"asia\",\"group_display\":\"Wrong\"",
        );
        let japan = region_json("japan", "[[24,46,122,146]]", "[35,138]").replace(
            "\"name\":\"Nordic\"",
            "\"name\":\"Japan\",\"group\":\"europe\",\"group_display\":\"Wrong\"",
        );
        let regions = parse_wrapped(&wrapped(&format!("{nordic},{japan}"))).unwrap();

        let placement = |slug: &str| {
            let r = &regions[slug];
            (r.group.clone(), r.group_name.clone(), r.group_order)
        };
        assert_eq!(
            placement("nordic"),
            (Some("europe".to_string()), Some("Europe".to_string()), 0)
        );
        assert_eq!(
            placement("japan"),
            (Some("asia".to_string()), Some("Asia".to_string()), 1)
        );
    }

    #[test]
    fn test_inconsistent_group_data_fails_generation() {
        let flat = r#"{"nordic":{"name":"Nordic","group_order":0}}"#;
        let faroe = region_json("faroe", "[[61,63,-8,-6]]", "[62,-7]");
        let two_homes = r#"{"groups":[
            {"group":"europe","display":"Europe","order":0,"tiles":["nordic"]},
            {"group":"asia","display":"Asia","order":1,"tiles":["nordic"]}]}"#;
        let one_order = r#"{"groups":[
            {"group":"europe","display":"Europe","order":0,"tiles":["nordic"]},
            {"group":"asia","display":"Asia","order":0,"tiles":["japan"]}]}"#;
        let with_groups = |text: &str, groups: &str| {
            message(parse_regions(
                REGIONS_FILE,
                text,
                Some((GROUPS_FILE, groups)),
                &names(),
            ))
        };

        assert_eq!(
            message(parse_regions(
                REGIONS_FILE,
                &wrapped(&nordic()),
                None,
                &names()
            )),
            "internal error: r.json lists countries by ISO code and has no group order of its own: \
             name a groups file for it with `groups` in registry-sources.toml"
        );
        assert_eq!(
            with_groups(flat, GROUPS),
            "internal error: r.json carries its own group order, so g.json would be a second \
             source of it: remove `groups` from its entry in registry-sources.toml"
        );
        assert_eq!(
            with_groups(&wrapped(&faroe), GROUPS),
            "internal error: r.json: region faroe is in no group of g.json"
        );
        assert_eq!(
            with_groups(&wrapped(&nordic()), two_homes),
            "internal error: g.json: tile nordic is in both group europe and group asia"
        );
        assert_eq!(
            with_groups(&wrapped(&nordic()), one_order),
            "internal error: g.json: groups europe and asia share order 0"
        );
    }

    #[test]
    fn test_region_geometry_is_validated() {
        let prefix = "internal error: r.json: region nordic has";
        let bad_box = |bboxes: &str| {
            message(parse_wrapped(&wrapped(&region_json(
                "nordic", bboxes, "[0,0]",
            ))))
        };
        let bad_centroid = |centroid: &str| {
            message(parse_wrapped(&wrapped(&region_json(
                "nordic",
                "[[0,10,0,10]]",
                centroid,
            ))))
        };
        let lat_outside = |values: &str| {
            format!("{prefix} a bounding box with a latitude outside -90 to 90 ({values})")
        };
        let lon_outside = |values: &str| {
            format!("{prefix} a bounding box with a longitude outside -180 to 180 ({values})")
        };
        let antimeridian = format!(
            "{prefix} a bounding box that crosses the antimeridian (west edge 170 is east of \
             east edge -170); publish it as two boxes"
        );

        // One row per bound, so no edge can go unchecked.
        assert_eq!(bad_box("[[-91,10,0,10]]"), lat_outside("-91, 10"));
        assert_eq!(bad_box("[[0,91,0,10]]"), lat_outside("0, 91"));
        assert_eq!(bad_box("[[0,10,-181,10]]"), lon_outside("-181, 10"));
        assert_eq!(bad_box("[[0,10,0,181]]"), lon_outside("0, 181"));
        assert_eq!(
            bad_box("[[20,10,0,10]]"),
            format!(
                "{prefix} a bounding box whose southern edge 20 is north of its northern edge 10"
            )
        );
        assert_eq!(bad_box("[[0,10,170,-170]]"), antimeridian);
        // Every box is checked, not only the first.
        assert_eq!(bad_box("[[0,10,0,10],[0,10,170,-170]]"), antimeridian);
        assert_eq!(bad_box("[]"), format!("{prefix} no bounding box"));

        let outside =
            |values: &str| format!("{prefix} a centroid outside the valid range ({values})");
        assert_eq!(bad_centroid("[-91,0]"), outside("-91, 0"));
        assert_eq!(bad_centroid("[91,0]"), outside("91, 0"));
        assert_eq!(bad_centroid("[0,-181]"), outside("0, -181"));
        assert_eq!(bad_centroid("[0,181]"), outside("0, 181"));

        // The bounds are inclusive: the whole globe is a valid box, and the
        // corners are valid centroids.
        let regions = parse_wrapped(&wrapped(&region_json(
            "nordic",
            "[[-90,90,-180,180]]",
            "[90,180]",
        )))
        .unwrap();
        assert_eq!(
            regions["nordic"].bboxes,
            vec![BoundingBox {
                lat_min: -90.0,
                lat_max: 90.0,
                lon_min: -180.0,
                lon_max: 180.0,
            }]
        );
        for (centroid, lat, lon) in [
            ("[90,180]", 90.0, 180.0),
            ("[-90,-180]", -90.0, -180.0),
            ("[-90,180]", -90.0, 180.0),
            ("[90,-180]", 90.0, -180.0),
        ] {
            let regions =
                parse_wrapped(&wrapped(&region_json("nordic", "[[0,10,0,10]]", centroid))).unwrap();
            assert_eq!(regions["nordic"].centroid, Some(GeoPoint { lat, lon }));
        }
    }

    #[test]
    fn test_a_variant_states_its_build_only_when_it_differs_from_the_entry() {
        let text = manifest_json(
            2,
            "",
            &[
                file_json("full/a-b1.onnx", "fp32", None, 1, ""),
                file_json("regional/nordic/a-b2.onnx", "fp32", Some("nordic"), 2, ""),
            ],
        );
        let regions = parse_wrapped(&wrapped(&nordic())).unwrap();
        let entry = entry_from(&text, &regions).unwrap();

        assert_eq!(entry.build, Some(1));
        let builds: Vec<Option<u32>> = entry.variants.iter().map(|v| v.build).collect();
        // The override is written only where the file differs from the entry, so
        // a family whose files share one build leaves the registry unchanged.
        assert_eq!(builds, vec![None, Some(2)]);
        let effective: Vec<Option<u32>> =
            entry.variants.iter().map(|v| entry.build_for(v)).collect();
        assert_eq!(effective, vec![Some(1), Some(2)]);
    }

    #[test]
    fn test_publishable_files_must_share_one_upstream_version() {
        let newer = |extra: &str| {
            file_json("full/b.onnx", "fp16", None, 1, extra).replace(
                "\"upstream_version\":\"3.0.1\"",
                "\"upstream_version\":\"3.0.2\"",
            )
        };
        let text = manifest_json(
            2,
            "",
            &[file_json("full/a.onnx", "fp32", None, 1, ""), newer("")],
        );
        assert_eq!(
            message(entry_from(&text, &BTreeMap::new())),
            "internal error: m.json: full/b.onnx has upstream_version 3.0.2 but full/a.onnx has 3.0.1; \
             one entry holds one version"
        );

        // A release replaces the old files with `superseded_by`, so a superseded
        // file at another version is expected and must not trip the check.
        let text = manifest_json(
            2,
            "",
            &[
                file_json("full/a.onnx", "fp32", None, 1, ""),
                newer(r#","superseded_by":"full/a.onnx""#),
            ],
        );
        assert!(entry_from(&text, &BTreeMap::new()).is_ok());

        // Withdrawn alone also takes a file out before the check, with no
        // replacement named.
        let text = manifest_json(
            2,
            "",
            &[
                file_json("full/a.onnx", "fp32", None, 1, ""),
                newer(r#","withdrawn":{"date":"2026-11-22","reason":"bad export"}"#),
            ],
        );
        assert!(entry_from(&text, &BTreeMap::new()).is_ok());
    }

    #[test]
    fn test_labels_checksum_and_size_come_from_the_manifest() {
        let text = manifest_json(
            2,
            "",
            &[
                file_json(
                    "full/a.onnx",
                    "fp32",
                    None,
                    1,
                    r#","labels_sha256":"cc","labels_size_bytes":9"#,
                ),
                file_json("full/b.onnx", "fp16", None, 1, ""),
            ],
        );
        let entry = entry_from(&text, &BTreeMap::new()).unwrap();

        let labels = &entry.variants[0].labels;
        assert_eq!(labels.sha256.as_deref(), Some("cc"));
        assert_eq!(labels.size_bytes, Some(9));
        // A manifest without them still generates, as the vendored schema 1 ones do.
        assert_eq!(entry.variants[1].labels.sha256, None);
        assert_eq!(entry.variants[1].labels.size_bytes, None);
    }

    #[test]
    fn test_a_withdrawn_file_is_excluded_without_superseded_by() {
        let text = manifest_json(
            2,
            "",
            &[
                file_json("full/a.onnx", "fp32", None, 1, ""),
                file_json(
                    "full/b.onnx",
                    "fp16",
                    None,
                    1,
                    r#","withdrawn":{"date":"2026-11-22","reason":"bad export"}"#,
                ),
                file_json("full/c.onnx", "int8", None, 1, r#","withdrawn":null"#),
            ],
        );
        let entry = entry_from(&text, &BTreeMap::new()).unwrap();

        let ids: Vec<&str> = entry.variants.iter().map(|v| v.id.as_str()).collect();
        assert_eq!(ids, vec!["fp32", "int8"]);
    }

    #[test]
    fn test_score_calibration_is_typed() {
        let calibrated = |method_and_fields: &str| {
            manifest_json(
                2,
                "",
                &[file_json(
                    "full/a.onnx",
                    "fp32",
                    None,
                    1,
                    &format!(r#","score_calibration":{{{method_and_fields}}}"#),
                )],
            )
        };

        let entry = entry_from(
            &calibrated(r#""method":"softmax-rest-logit","rest_logit":-1.25"#),
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(
            entry.variants[0].score_calibration,
            Some(ScoreCalibration::SoftmaxRestLogit { rest_logit: -1.25 })
        );

        // An unknown method must stop generation, because `ScoreCalibration`
        // is closed and the registry could not carry it.
        let unknown = message(parse_manifest(
            MANIFEST_FILE,
            &calibrated(r#""method":"platt","a":1"#),
        ));
        assert!(
            unknown.starts_with(
                "internal error: m.json is not a valid manifest: unknown variant `platt`, \
                 expected `softmax-rest-logit` at line "
            ),
            "{unknown}"
        );
    }

    #[test]
    fn test_documents_resolve_against_their_revision() {
        let meta = r#","family_meta":{"documents":[
            {"kind":"license","requires_agreement":false,"path":"LICENSE.md","revision":"abc123","sha256":"dd"},
            {"kind":"terms","requires_agreement":true,"url":"https://example.com/terms","revision":"r2"}]}"#;
        let text = manifest_json(2, meta, &[file_json("full/a.onnx", "fp32", None, 1, "")]);
        let entry = entry_from(&text, &BTreeMap::new()).unwrap();

        assert_eq!(
            entry.documents,
            vec![
                ModelDocument {
                    kind: "license".to_string(),
                    requires_agreement: false,
                    url: "https://huggingface.co/o/r/resolve/abc123/LICENSE.md".to_string(),
                    revision: Some("abc123".to_string()),
                    sha256: Some("dd".to_string()),
                },
                ModelDocument {
                    kind: "terms".to_string(),
                    requires_agreement: true,
                    url: "https://example.com/terms".to_string(),
                    revision: Some("r2".to_string()),
                    sha256: None,
                },
            ]
        );
    }

    #[test]
    fn test_malformed_documents_fail_generation() {
        let with_document = |document: &str| {
            let meta = format!(r#","family_meta":{{"documents":[{document}]}}"#);
            let text = manifest_json(2, &meta, &[file_json("full/a.onnx", "fp32", None, 1, "")]);
            message(entry_from(&text, &BTreeMap::new()))
        };

        assert_eq!(
            with_document(
                r#"{"kind":"license","requires_agreement":false,"path":"L.md","url":"https://x/l","revision":"r"}"#
            ),
            "internal error: m.json: document license has both a path and a url"
        );
        assert_eq!(
            with_document(r#"{"kind":"license","requires_agreement":false}"#),
            "internal error: m.json: document license has neither a path nor a url"
        );
        assert_eq!(
            with_document(r#"{"kind":"license","requires_agreement":false,"path":"L.md"}"#),
            "internal error: m.json: document license has a path but no revision"
        );
    }

    #[test]
    fn test_unsupported_schemas_fail_generation() {
        let manifest = manifest_json(3, "", &[file_json("full/a.onnx", "fp32", None, 1, "")]);
        assert_eq!(
            message(parse_manifest(MANIFEST_FILE, &manifest)),
            "internal error: m.json has manifest schema 3, which this generator does not read (it reads [1, 2])"
        );

        assert_eq!(
            message(parse_wrapped(r#"{"schema":2,"regions":{}}"#)),
            "internal error: r.json has regions schema 2, which this generator does not read (it reads 1)"
        );
    }

    #[test]
    fn test_withdrawn_accepts_only_null_or_an_object() {
        let with = |withdrawn: &str| {
            let file = file_json(
                "full/a.onnx",
                "fp32",
                None,
                1,
                &format!(r#","withdrawn":{withdrawn}"#),
            );
            parse_manifest(MANIFEST_FILE, &manifest_json(2, "", &[file]))
        };

        assert!(with("null").is_ok());
        assert!(with("{}").is_ok());
        assert!(with(r#"{"date":"2026-11-22"}"#).is_ok());
        // A sequence, a boolean and a string are not "withdrawn" in either
        // direction; they are a manifest this generator does not understand.
        for (value, found) in [
            ("[]", "sequence"),
            ("false", "boolean `false`"),
            ("\"gone\"", "string \"gone\""),
        ] {
            let message = message(with(value));
            assert!(
                message.starts_with(&format!(
                    "internal error: m.json is not a valid manifest: invalid type: {found}, \
                     expected a map at line "
                )),
                "{message}"
            );
        }
    }

    #[test]
    fn test_a_regional_file_without_a_region_entry_fails_generation() {
        let files = [file_json(
            "regional/faroe/a.onnx",
            "fp32",
            Some("faroe"),
            1,
            "",
        )];
        let text = manifest_json(2, "", &files);

        assert_eq!(
            message(entry_from(
                &text,
                &parse_wrapped(&wrapped(&nordic())).unwrap()
            )),
            "internal error: m.json: regional/faroe/a.onnx is in region faroe, which r.json does not describe"
        );
        // The flat form is held to the same rule.
        let flat = parse_regions(
            REGIONS_FILE,
            r#"{"nordic":{"name":"Nordic","group_order":0}}"#,
            None,
            &names(),
        )
        .unwrap();
        assert_eq!(
            message(entry_from(&text, &flat)),
            "internal error: m.json: regional/faroe/a.onnx is in region faroe, which r.json does not describe"
        );
    }

    #[test]
    fn test_two_files_for_one_region_and_variant_fail_generation() {
        let regions = parse_wrapped(&wrapped(&nordic())).unwrap();
        let regional = manifest_json(
            2,
            "",
            &[
                file_json("regional/nordic/a-b1.onnx", "fp32", Some("nordic"), 1, ""),
                file_json("regional/nordic/a-b2.onnx", "fp32", Some("nordic"), 2, ""),
            ],
        );
        assert_eq!(
            message(entry_from(&regional, &regions)),
            "internal error: m.json: regional/nordic/a-b1.onnx and regional/nordic/a-b2.onnx \
             are both variant fp32 of region nordic"
        );

        let global = manifest_json(
            2,
            "",
            &[
                file_json("full/a-b1.onnx", "fp32", None, 1, ""),
                file_json("full/a-b2.onnx", "fp32", None, 2, ""),
            ],
        );
        assert_eq!(
            message(entry_from(&global, &regions)),
            "internal error: m.json: full/a-b1.onnx and full/a-b2.onnx are both variant fp32 \
             of the global model"
        );

        // The same variant id in two regions, or in a region and the global
        // model, is how every family is laid out.
        let distinct = manifest_json(
            2,
            "",
            &[
                file_json("full/a.onnx", "fp32", None, 1, ""),
                file_json("regional/nordic/a.onnx", "fp32", Some("nordic"), 1, ""),
            ],
        );
        assert!(entry_from(&distinct, &regions).is_ok());
    }

    #[test]
    fn test_manifest_floats_parse_to_the_nearest_double() {
        // Geometry floats are parsed from the manifest and again from the
        // registry on disk, and `next_registry_version` compares the two. Without
        // serde_json's `float_roundtrip` feature this 17-digit literal parsed one
        // ULP away from `str::parse` (0.8255026276548036 against
        // 0.8255026276548038, measured with serde_json 1.0.154), so such a
        // coordinate could bump the version on every regeneration.
        let text = "0.82550262765480373";
        let parsed: f64 = serde_json::from_str(text).unwrap();
        assert_eq!(parsed.to_bits(), text.parse::<f64>().unwrap().to_bits());
    }
}
