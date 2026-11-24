//! Picking a regional model from a location.
//!
//! One rule serves `birda models install --region auto` and
//! `birda models regions <id> --lat --lon`, whose JSON result lets a client such
//! as birda-gui show the same pick: the tiles whose boxes contain the point
//! (edges included), then the highest `tier`, then the nearest `centroid` by great-circle distance rounded to whole
//! kilometres, then the slug. A point no tile contains is answered by the global
//! model. Nothing here reads the network or the host: the pick is a function of
//! the point and the entry's region data.

use super::types::{BoundingBox, GeoPoint, ModelEntry, ModelVariant};
use crate::constants::coordinates::{LATITUDE_MAX, LATITUDE_MIN, LONGITUDE_MAX, LONGITUDE_MIN};
use crate::constants::regions as consts;
use crate::error::{Error, Result};

impl BoundingBox {
    /// Whether `point` lies inside the box, edges included.
    ///
    /// Longitudes -180 and 180 are the same meridian, so a point on one also
    /// matches a box that reaches the other. A box never crosses the
    /// antimeridian (`types.rs`, the `BoundingBox` doc), which keeps this a
    /// one-twin check.
    #[must_use]
    pub fn contains(&self, point: GeoPoint) -> bool {
        if !(self.lat_min <= point.lat && point.lat <= self.lat_max) {
            return false;
        }
        let lon_inside = |lon: f64| self.lon_min <= lon && lon <= self.lon_max;
        lon_inside(point.lon)
            || (point.lon <= LONGITUDE_MIN && lon_inside(LONGITUDE_MAX))
            || (point.lon >= LONGITUDE_MAX && lon_inside(LONGITUDE_MIN))
    }
}

/// Great-circle distance between two points, in kilometres (haversine, mean
/// Earth radius).
///
/// Planar degree distance is wrong at high latitude and across the
/// antimeridian, which is where two tiles' centres can be nearest to a point.
#[must_use]
pub fn great_circle_km(a: GeoPoint, b: GeoPoint) -> f64 {
    let half_dlat = (b.lat - a.lat).to_radians() / 2.0;
    let half_dlon = (b.lon - a.lon).to_radians() / 2.0;
    let h = half_dlon.sin().powi(2).mul_add(
        a.lat.to_radians().cos() * b.lat.to_radians().cos(),
        half_dlat.sin().powi(2),
    );
    2.0 * consts::EARTH_MEAN_RADIUS_KM * h.clamp(0.0, 1.0).sqrt().asin()
}

/// Whether any tile of `entry` publishes boundaries, so a location can pick one.
#[must_use]
pub fn has_geometry(entry: &ModelEntry) -> bool {
    entry.regions().iter().any(|v| !v.bboxes.is_empty())
}

/// The region slug of a tile, empty for the global model.
fn slug_of(variant: &ModelVariant) -> &str {
    variant.region.as_deref().unwrap_or_default()
}

/// Distance from `point` to the tile's centroid in whole kilometres, as a float
/// so it orders with [`f64::total_cmp`]. A missing or non-finite centroid is
/// infinitely far, which sorts after every tile that has one.
fn rounded_distance_km(variant: &ModelVariant, point: GeoPoint) -> f64 {
    variant
        .centroid
        .filter(|c| c.lat.is_finite() && c.lon.is_finite())
        .map_or(f64::INFINITY, |c| great_circle_km(point, c).round())
}

/// The tiles of `entry` whose boxes contain `point`, best first.
///
/// Order: highest `tier` (a missing tier is lowest), then the nearest centroid
/// in whole kilometres, then the slug. Rounding makes near-equal pairs fall to
/// the slug rather than to the last bit of a floating-point result, so the pick
/// is reproducible. The global model is never listed; an empty result means it
/// is the answer.
#[must_use]
pub fn regions_at(entry: &ModelEntry, point: GeoPoint) -> Vec<&ModelVariant> {
    let mut hits: Vec<(&ModelVariant, f64)> = entry
        .regions()
        .into_iter()
        .filter(|v| v.bboxes.iter().any(|b| b.contains(point)))
        .map(|v| (v, rounded_distance_km(v, point)))
        .collect();

    hits.sort_by(|(a, dist_a), (b, dist_b)| {
        b.tier
            .cmp(&a.tier)
            .then_with(|| dist_a.total_cmp(dist_b))
            .then_with(|| slug_of(a).cmp(slug_of(b)))
    });
    hits.into_iter().map(|(v, _)| v).collect()
}

/// The matches after the first that share its tier.
///
/// `matches` comes from [`regions_at`], where tier descends, so these are the
/// tiles the pick beat only on distance or slug.
#[must_use]
pub fn top_tier_ties<'a, 'b>(matches: &'b [&'a ModelVariant]) -> &'b [&'a ModelVariant] {
    let Some((first, rest)) = matches.split_first() else {
        return &[];
    };
    let ties = rest.iter().take_while(|v| v.tier == first.tier).count();
    &rest[..ties]
}

/// The global model's representative variant: the default variant when the
/// family publishes one without a region, else the first without a region.
#[must_use]
pub fn global_variant(entry: &ModelEntry) -> Option<&ModelVariant> {
    entry
        .default_variant
        .as_deref()
        .and_then(|id| entry.find_variant(None, id))
        .or_else(|| entry.variants.iter().find(|v| v.region.is_none()))
}

/// Lowercase `text` and map the accented characters of ISO country names to
/// ASCII, so "Cote d'Ivoire" finds "Côte d’Ivoire".
///
/// Not Unicode normalisation: combining marks ([`consts::COMBINING_MARKS`]) are
/// dropped, so a decomposed spelling folds like the precomposed one, and the
/// table in [`consts::FOLDS`] covers the names the ISO list contains; a test
/// fails when an update adds a character.
#[must_use]
pub fn fold(text: &str) -> String {
    text.chars()
        .filter(|c| !consts::COMBINING_MARKS.contains(c))
        .flat_map(char::to_lowercase)
        .map(|c| {
            consts::FOLDS
                .iter()
                .find(|(from, _)| *from == c)
                .map_or(c, |(_, to)| *to)
        })
        .collect()
}

/// Whether any entry of `list` satisfies `matches`, as a rank: 0 for a core
/// country, 1 for a partial one.
fn country_rank(
    core: Option<&[String]>,
    partial: Option<&[String]>,
    matches: &dyn Fn(&str) -> bool,
) -> Option<u8> {
    if core.is_some_and(|names| names.iter().any(|n| matches(n))) {
        return Some(0);
    }
    if partial.is_some_and(|names| names.iter().any(|n| matches(n))) {
        return Some(1);
    }
    None
}

/// Tiles with a rank, best rank first and `regions()` order within a rank.
fn ranked<'a>(
    tiles: &[&'a ModelVariant],
    rank: impl Fn(&ModelVariant) -> Option<u8>,
) -> Vec<&'a ModelVariant> {
    let mut hits: Vec<(u8, &ModelVariant)> = tiles
        .iter()
        .filter_map(|v| rank(v).map(|r| (r, *v)))
        .collect();
    // Stable, so equal ranks keep the order `regions()` gave.
    hits.sort_by_key(|(r, _)| *r);
    hits.into_iter().map(|(_, v)| v).collect()
}

/// The tiles that serve a country the user typed, best first.
///
/// In order, the first rule that finds anything wins:
///
/// 1. A two-letter query that some tile lists as an ISO code (case ignored).
/// 2. A folded query equal to a folded country name.
/// 3. A folded query found at the start of a word of a region name (rank 0), a
///    core country (1), a partial country (2) or a group name (3), the rank
///    order birda-gui's search uses. A word starts the text or follows a
///    character that is not alphanumeric, so "aland" finds "Åland Islands"
///    and not "New Zealand".
///
/// Rules 1 and 2 put tiles that cover the country wholly before partial ones.
/// Within a rank the order is that of [`ModelEntry::regions`]. A two-letter
/// query no tile lists as a code falls through, so a registry published without
/// codes still finds "fi" by name.
#[must_use]
pub fn regions_for_country<'a>(entry: &'a ModelEntry, query: &str) -> Vec<&'a ModelVariant> {
    let query = query.trim();
    if query.is_empty() {
        return Vec::new();
    }
    let tiles = entry.regions();

    if query.chars().count() == consts::ISO_CODE_LEN
        && query.chars().all(|c| c.is_ascii_alphabetic())
    {
        let by_code = ranked(&tiles, |v| {
            let codes = v.country_codes.as_ref();
            country_rank(
                codes.map(|c| c.core.as_slice()),
                codes.map(|c| c.partial.as_slice()),
                &|code| code.eq_ignore_ascii_case(query),
            )
        });
        if !by_code.is_empty() {
            return by_code;
        }
    }

    let folded = fold(query);
    // A query of only combining marks folds to nothing, and an empty needle
    // would start a word in every name.
    if folded.is_empty() {
        return Vec::new();
    }
    let by_name = ranked(&tiles, |v| {
        let names = v.countries.as_ref();
        country_rank(
            names.map(|c| c.core.as_slice()),
            names.map(|c| c.partial.as_slice()),
            &|name| fold(name) == folded,
        )
    });
    if !by_name.is_empty() {
        return by_name;
    }

    ranked(&tiles, |v| search_rank(v, &folded))
}

/// Whether `needle` occurs in `haystack` starting at a word: at the start of the
/// text, or right after a character that is not alphanumeric.
///
/// Every character position is tried, not only the non-overlapping matches
/// `str::match_indices` would return, so an occurrence that overlaps an
/// earlier mid-word one is still found.
fn starts_a_word(haystack: &str, needle: &str) -> bool {
    let mut before: Option<char> = None;
    for (at, current) in haystack.char_indices() {
        if before.is_none_or(|c| !c.is_alphanumeric()) && haystack[at..].starts_with(needle) {
            return true;
        }
        before = Some(current);
    }
    false
}

/// The rank of a folded query among a tile's texts, with birda-gui's order:
/// region name 0, core country 1, partial country 2, group name 3. A text
/// counts only where the query starts a word of it.
fn search_rank(variant: &ModelVariant, folded: &str) -> Option<u8> {
    let has = |text: &str| starts_a_word(&fold(text), folded);
    if variant.region_name.as_deref().is_some_and(has) {
        return Some(0);
    }
    if let Some(names) = &variant.countries {
        if names.core.iter().any(|n| has(n)) {
            return Some(1);
        }
        if names.partial.iter().any(|n| has(n)) {
            return Some(2);
        }
    }
    variant.group_name.as_deref().is_some_and(has).then_some(3)
}

/// What the user asked for with `--region`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionRequest<'a> {
    /// No `--region`.
    Unspecified,
    /// `--region auto`: the region that covers a location.
    Auto,
    /// `--region global`: the global model.
    Global,
    /// A region slug, validated later by variant selection.
    Named(&'a str),
}

impl<'a> RegionRequest<'a> {
    /// Read the `--region` argument. Exact lowercase, as the flag is documented.
    #[must_use]
    pub fn from_arg(arg: Option<&'a str>) -> Self {
        match arg {
            None => Self::Unspecified,
            Some(consts::AUTO) => Self::Auto,
            Some(consts::GLOBAL) => Self::Global,
            Some(slug) => Self::Named(slug),
        }
    }
}

/// The region options `models install` hands its handler.
#[derive(Debug, Clone, Copy)]
pub struct InstallRegion<'a> {
    /// The `--region` argument.
    pub region: Option<&'a str>,
    /// `--lat`/`--lon` given after `models install`.
    pub point: Option<GeoPoint>,
    /// Whether `--lat` or `--lon` (or their environment variables) were given
    /// before `models`, where they belong to the analyze command.
    pub analyze_location_set: bool,
}

/// Everything [`resolve_region`] decides from.
#[derive(Debug, Clone, Copy)]
pub struct RegionInputs<'a> {
    /// The `--region` argument.
    pub request: RegionRequest<'a>,
    /// `--lat`/`--lon` given after `models install`.
    pub point: Option<GeoPoint>,
    /// `defaults.latitude` from the config file.
    pub configured_lat: Option<f64>,
    /// `defaults.longitude` from the config file.
    pub configured_lon: Option<f64>,
    /// Whether `--lat` or `--lon` were given before `models`; see
    /// [`InstallRegion::analyze_location_set`].
    pub analyze_location_set: bool,
    /// A person is present and can be asked.
    pub interactive: bool,
    /// `--yes`: do not ask.
    pub assume_yes: bool,
    /// The `--variant` argument, checked early so a typo fails before a question.
    pub variant: Option<&'a str>,
}

/// What an install does about the region.
#[derive(Debug, Clone, PartialEq)]
pub enum RegionResolution<'a> {
    /// No `--region` and nothing to ask: the global model, as before regional
    /// installs existed.
    DefaultGlobal,
    /// `--region global`.
    Global,
    /// `--region <slug>`; variant selection validates the slug.
    Named(String),
    /// `--region auto`: the tiles that cover the point, best first. Empty means
    /// the global model.
    Located {
        /// The location used.
        point: GeoPoint,
        /// Containing tiles in [`regions_at`] order.
        matches: Vec<&'a ModelVariant>,
    },
    /// Ask the user, who may have a configured location to propose.
    Ask {
        /// The configured location, when it is set and the entry can use it.
        configured: Option<GeoPoint>,
    },
}

/// Range-check the configured coordinate stored under `key`, reporting a value
/// outside `min..=max` as [`Error::ConfiguredCoordinateOutOfRange`].
fn checked(value: Option<f64>, min: f64, max: f64, key: &'static str) -> Result<Option<f64>> {
    match value {
        Some(v) if !(min..=max).contains(&v) => {
            Err(Error::ConfiguredCoordinateOutOfRange { key, min, max })
        }
        other => Ok(other),
    }
}

/// The configured location, when both keys are set and in range.
///
/// `models` commands skip config validation, so a hand-edited out-of-range value
/// reaches this point and would otherwise fail only when the install is saved.
fn configured_point(lat: Option<f64>, lon: Option<f64>) -> Result<Option<GeoPoint>> {
    let lat = checked(lat, LATITUDE_MIN, LATITUDE_MAX, consts::CONFIG_LATITUDE_KEY)?;
    let lon = checked(
        lon,
        LONGITUDE_MIN,
        LONGITUDE_MAX,
        consts::CONFIG_LONGITUDE_KEY,
    )?;
    Ok(lat.zip(lon).map(|(lat, lon)| GeoPoint { lat, lon }))
}

/// Decide what an install does about the region.
///
/// Checks run in this order: (1) a point with any request but `Auto` is
/// refused; (2) `Global` is `Global` on any entry; (3) an entry without regions
/// refuses `Named` and `Auto` and installs its only model for `Unspecified`;
/// (4) `Named` passes through; (5) `Auto` needs boundaries, then a location;
/// (6) `Unspecified` asks only when `interactive` and not `assume_yes`.
///
/// # Errors
///
/// [`Error::LocationNeedsRegionAuto`], [`Error::RegionsNotSupported`],
/// [`Error::RegionLocationMissing`], [`Error::RegionLocationBeforeModels`],
/// [`Error::RegionGeometryMissing`], [`Error::VariantNotFound`] for a `variant`
/// no variant of the entry has, and [`Error::ConfiguredCoordinateOutOfRange`]
/// for an out-of-range configured location (read
/// only for `Auto` and for a question, and only for an entry with boundaries,
/// so a broken config never changes a plain install or a model that cannot use
/// a location).
pub fn resolve_region<'a>(
    entry: &'a ModelEntry,
    inputs: RegionInputs<'_>,
) -> Result<RegionResolution<'a>> {
    let not_supported = || Error::RegionsNotSupported {
        model_id: entry.id.clone(),
    };

    if inputs.point.is_some() && inputs.request != RegionRequest::Auto {
        return Err(Error::LocationNeedsRegionAuto);
    }
    if inputs.request == RegionRequest::Global {
        return Ok(RegionResolution::Global);
    }
    if entry.regions().is_empty() {
        return match inputs.request {
            RegionRequest::Named(_) | RegionRequest::Auto => Err(not_supported()),
            _ => Ok(RegionResolution::DefaultGlobal),
        };
    }

    match inputs.request {
        RegionRequest::Named(slug) => Ok(RegionResolution::Named(slug.to_string())),
        RegionRequest::Auto => {
            if !has_geometry(entry) {
                return Err(Error::RegionGeometryMissing {
                    model_id: entry.id.clone(),
                });
            }
            let point = match inputs.point {
                Some(point) => point,
                None if inputs.analyze_location_set => {
                    return Err(Error::RegionLocationBeforeModels);
                }
                None => configured_point(inputs.configured_lat, inputs.configured_lon)?
                    .ok_or_else(|| Error::RegionLocationMissing {
                        model_id: entry.id.clone(),
                    })?,
            };

            Ok(RegionResolution::Located {
                point,
                matches: regions_at(entry, point),
            })
        }
        RegionRequest::Unspecified | RegionRequest::Global => {
            if !inputs.interactive || inputs.assume_yes {
                return Ok(RegionResolution::DefaultGlobal);
            }
            if let Some(requested) = inputs.variant
                && entry.variants.iter().all(|v| v.id != requested)
            {
                let mut ids: Vec<&str> = Vec::new();
                for v in &entry.variants {
                    if !ids.contains(&v.id.as_str()) {
                        ids.push(&v.id);
                    }
                }
                return Err(Error::VariantNotFound {
                    model_id: entry.id.clone(),
                    variant: requested.to_string(),
                    available: ids.join(", "),
                });
            }
            let configured = if has_geometry(entry) {
                configured_point(inputs.configured_lat, inputs.configured_lon)?
            } else {
                None
            };
            Ok(RegionResolution::Ask { configured })
        }
    }
}

/// Coverage facts for a point, one or two lines: the pick and any tile equally
/// specific, or that nothing covers it. The caller adds its own action sentence.
///
/// Names the point, so it is for `birda models regions <id> --lat --lon`,
/// where the coordinates are the user's own query. An install uses
/// [`install_location_summary`] instead.
#[must_use]
pub fn located_summary(point: GeoPoint, matches: &[&ModelVariant]) -> String {
    let GeoPoint { lat, lon } = point;
    coverage_summary(&format!("{lat}, {lon}"), matches)
}

/// The same coverage facts as [`located_summary`], without the coordinates.
///
/// Install output often ends up in provisioning logs, and where a recorder
/// sits can be sensitive, so an install names the region that covers the
/// location, if one does, but never the point.
#[must_use]
pub fn install_location_summary(matches: &[&ModelVariant]) -> String {
    coverage_summary(consts::INSTALL_LOCATION_SUBJECT, matches)
}

/// The coverage lines for `subject`, the point as the caller describes it.
fn coverage_summary(subject: &str, matches: &[&ModelVariant]) -> String {
    let Some(best) = matches.first() else {
        return format!("No regional model covers {subject}.");
    };
    let mut text = format!(
        "Region for {subject}: {} ({}).",
        slug_of(best),
        best.region_name.as_deref().unwrap_or_default()
    );
    let ties = top_tier_ties(matches);
    if !ties.is_empty() {
        let slugs: Vec<&str> = ties.iter().map(|v| slug_of(v)).collect();
        text = format!("{text}\nAlso equally specific here: {}.", slugs.join(", "));
    }
    text
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    //! Entries for the lookup, resolver and prompt tests.
    //!
    //! `real_tiles` copies the geometry, tier and centroid of twelve `BirdNET+`
    //! V3.0.1 regions from the bundled `registry.json` (registry version 10, the
    //! `birdnet-plus-v30` entry). The country lists are trimmed to what the tests
    //! use, names and codes kept in step.

    use super::{GeoPoint, ModelEntry, ModelVariant};
    use crate::registry::types::{BoundingBox, Countries, FileInfo, LicenseInfo};

    pub fn file(name: &str, size_bytes: u64) -> FileInfo {
        FileInfo {
            url: format!("https://huggingface.co/x/{name}"),
            filename: name.to_string(),
            sha256: Some("abc".to_string()),
            size_bytes: Some(size_bytes),
        }
    }

    pub fn countries(core: &[&str], partial: &[&str]) -> Countries {
        let own = |names: &[&str]| names.iter().map(|n| (*n).to_string()).collect();
        Countries {
            core: own(core),
            partial: own(partial),
        }
    }

    /// A tile with no geometry; callers set the rest through the helpers below.
    pub fn tile(slug: &str, name: &str, group_order: u32) -> ModelVariant {
        ModelVariant {
            id: "fp32".to_string(),
            region: Some(slug.to_string()),
            region_name: Some(name.to_string()),
            group: Some("g".to_string()),
            group_name: Some(format!("Group {group_order}")),
            group_order,
            classes: Some(100),
            model: file(&format!("{slug}.onnx"), 100_000_000),
            labels: file(&format!("{slug}.txt"), 1_000),
            countries: None,
            build: None,
            country_codes: None,
            tier: None,
            bboxes: Vec::new(),
            centroid: None,
            score_calibration: None,
        }
    }

    pub fn bbox(lat_min: f64, lat_max: f64, lon_min: f64, lon_max: f64) -> BoundingBox {
        BoundingBox {
            lat_min,
            lat_max,
            lon_min,
            lon_max,
        }
    }

    /// A tile covering `boxes` with the given tier and centroid.
    pub fn placed(
        slug: &str,
        tier: Option<u32>,
        boxes: &[BoundingBox],
        centroid: Option<GeoPoint>,
    ) -> ModelVariant {
        let mut v = tile(slug, slug, 0);
        v.tier = tier;
        v.bboxes = boxes.to_vec();
        v.centroid = centroid;
        v
    }

    pub fn point(lat: f64, lon: f64) -> GeoPoint {
        GeoPoint { lat, lon }
    }

    /// The global model plus `tiles`, default variant `fp32`.
    pub fn entry_with(tiles: Vec<ModelVariant>) -> ModelEntry {
        let mut global = tile("global", "Global", 0);
        global.region = None;
        global.region_name = None;
        global.group = None;
        global.group_name = None;
        global.classes = Some(11_000);
        global.model = file("global.onnx", 200_000_000);
        let mut variants = vec![global];
        variants.extend(tiles);
        ModelEntry {
            id: "geo-test".to_string(),
            name: "Geo Test".to_string(),
            description: "d".to_string(),
            vendor: "v".to_string(),
            version: "1.0".to_string(),
            model_type: "birdnet-v30".to_string(),
            license: LicenseInfo {
                r#type: "CC".to_string(),
                url: "https://x".to_string(),
                commercial_use: true,
                attribution_required: false,
                share_alike: false,
            },
            files: None,
            build: Some(1),
            default_variant: Some("fp32".to_string()),
            selection: std::collections::BTreeMap::new(),
            variants,
            documents: Vec::new(),
            recommended: false,
        }
    }

    /// One published region: slug, name, group order, group name, species,
    /// size, tier, centroid, boxes, core and partial country names and codes.
    struct Real {
        slug: &'static str,
        name: &'static str,
        group_order: u32,
        group_name: &'static str,
        classes: usize,
        size_bytes: u64,
        tier: u32,
        centroid: (f64, f64),
        boxes: &'static [(f64, f64, f64, f64)],
        core: &'static [&'static str],
        partial: &'static [&'static str],
        core_codes: &'static [&'static str],
        partial_codes: &'static [&'static str],
    }

    const REAL: &[Real] = &[
        Real {
            slug: "nordic",
            name: "Nordic",
            group_order: 0,
            group_name: "Europe",
            classes: 464,
            size_bytes: 102_224_864,
            tier: 50,
            centroid: (62.88, 17.685),
            boxes: &[(53.1, 72.66, 3.32, 32.05)],
            core: &["Åland Islands", "Denmark", "Finland", "Norway", "Sweden"],
            partial: &[],
            core_codes: &["AX", "DK", "FI", "NO", "SE"],
            partial_codes: &[],
        },
        Real {
            slug: "baltics",
            name: "Baltics",
            group_order: 0,
            group_name: "Europe",
            classes: 479,
            size_bytes: 102_547_724,
            tier: 50,
            centroid: (56.65, 25.0),
            boxes: &[(53.5, 59.8, 20.0, 30.0)],
            core: &["Belarus", "Estonia", "Lithuania", "Latvia", "Russia"],
            partial: &[],
            core_codes: &["BY", "EE", "LT", "LV", "RU"],
            partial_codes: &[],
        },
        Real {
            slug: "western-palearctic",
            name: "Western Palearctic",
            group_order: 0,
            group_name: "Europe",
            classes: 1446,
            size_bytes: 123_361_440,
            tier: 10,
            centroid: (41.5, 14.0),
            boxes: &[
                (35.0, 71.0, -11.0, 45.0),
                (26.0, 40.0, -32.0, -13.0),
                (20.0, 37.0, -13.0, 37.0),
                (30.0, 42.0, 34.0, 50.0),
                (12.0, 32.0, 34.0, 60.0),
            ],
            core: &[
                "Åland Islands",
                "Denmark",
                "Finland",
                "Niger",
                "Norway",
                "Sweden",
            ],
            partial: &["Russia"],
            core_codes: &["AX", "DK", "FI", "NE", "NO", "SE"],
            partial_codes: &["RU"],
        },
        Real {
            slug: "central-europe",
            name: "Central Europe",
            group_order: 0,
            group_name: "Europe",
            classes: 673,
            size_bytes: 106_723_388,
            tier: 50,
            centroid: (50.25, 13.0),
            boxes: &[(45.0, 55.5, 2.0, 24.0)],
            core: &["Germany", "Poland"],
            partial: &["Denmark", "Lithuania"],
            core_codes: &["DE", "PL"],
            partial_codes: &["DK", "LT"],
        },
        Real {
            slug: "southern-europe",
            name: "Southern Europe",
            group_order: 0,
            group_name: "Europe",
            classes: 653,
            size_bytes: 106_292_908,
            tier: 50,
            centroid: (41.5, 17.5),
            boxes: &[(36.0, 47.0, 6.0, 29.0)],
            core: &["Italy", "Greece"],
            partial: &["Türkiye"],
            core_codes: &["IT", "GR"],
            partial_codes: &["TR"],
        },
        Real {
            slug: "central-india",
            name: "Central India",
            group_order: 1,
            group_name: "Asia",
            classes: 708,
            size_bytes: 107_476_728,
            tier: 90,
            centroid: (21.0, 79.75),
            boxes: &[(16.0, 26.0, 73.0, 86.5)],
            core: &["India"],
            partial: &[],
            core_codes: &["IN"],
            partial_codes: &[],
        },
        Real {
            slug: "indo-gangetic",
            name: "Indo-Gangetic Plain",
            group_order: 1,
            group_name: "Asia",
            classes: 1212,
            size_bytes: 118_324_824,
            tier: 50,
            centroid: (25.5, 79.0),
            boxes: &[(21.0, 30.0, 68.0, 90.0)],
            core: &["Bangladesh", "India", "Nepal"],
            partial: &["China"],
            core_codes: &["BD", "IN", "NP"],
            partial_codes: &["CN"],
        },
        Real {
            slug: "east-africa",
            name: "East Africa",
            group_order: 4,
            group_name: "Africa",
            classes: 1840,
            size_bytes: 131_841_896,
            tier: 50,
            centroid: (0.5, 40.0),
            boxes: &[(-17.0, 18.0, 28.0, 52.0)],
            core: &["Kenya", "Tanzania"],
            partial: &["Sudan"],
            core_codes: &["KE", "TZ"],
            partial_codes: &["SD"],
        },
        Real {
            slug: "central-africa",
            name: "Central Africa",
            group_order: 4,
            group_name: "Africa",
            classes: 1556,
            size_bytes: 125_729_080,
            tier: 50,
            centroid: (-3.0, 19.5),
            boxes: &[(-17.0, 11.0, 8.0, 31.0)],
            core: &["Angola", "Gabon"],
            partial: &["Nigeria"],
            core_codes: &["AO", "GA"],
            partial_codes: &["NG"],
        },
        Real {
            slug: "north-america-east",
            name: "North America (East)",
            group_order: 2,
            group_name: "North America",
            classes: 846,
            size_bytes: 110_447_040,
            tier: 50,
            centroid: (37.5, -77.5),
            boxes: &[(25.0, 50.0, -95.0, -60.0)],
            core: &["Canada", "United States"],
            partial: &[],
            core_codes: &["CA", "US"],
            partial_codes: &[],
        },
        Real {
            slug: "amazonia",
            name: "Amazonia",
            group_order: 3,
            group_name: "South America",
            classes: 3196,
            size_bytes: 161_028_440,
            tier: 50,
            centroid: (-1.5, -61.5),
            boxes: &[(-13.0, 10.0, -79.0, -44.0)],
            core: &["Brazil", "Ecuador"],
            partial: &["Bolivia"],
            core_codes: &["BR", "EC"],
            partial_codes: &["BO"],
        },
        Real {
            slug: "andes",
            name: "Andes",
            group_order: 3,
            group_name: "South America",
            classes: 3325,
            size_bytes: 163_805_036,
            tier: 50,
            centroid: (-14.0, -71.5),
            boxes: &[(-40.0, 12.0, -81.0, -62.0)],
            core: &["Chile", "Ecuador", "Peru"],
            partial: &[],
            core_codes: &["CL", "EC", "PE"],
            partial_codes: &[],
        },
    ];

    /// The twelve published tiles, in the order of the table above.
    pub fn real_tiles() -> Vec<ModelVariant> {
        REAL.iter()
            .map(|r| {
                let mut v = tile(r.slug, r.name, r.group_order);
                v.group_name = Some(r.group_name.to_string());
                v.classes = Some(r.classes);
                v.model = file(&format!("{}.onnx", r.slug), r.size_bytes);
                v.tier = Some(r.tier);
                v.centroid = Some(point(r.centroid.0, r.centroid.1));
                v.bboxes = r.boxes.iter().map(|b| bbox(b.0, b.1, b.2, b.3)).collect();
                v.countries = Some(countries(r.core, r.partial));
                v.country_codes = Some(countries(r.core_codes, r.partial_codes));
                v
            })
            .collect()
    }

    /// `geo-test` with the published tiles.
    pub fn real_entry() -> ModelEntry {
        entry_with(real_tiles())
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::{
        bbox, countries, entry_with, placed, point, real_entry, real_tiles, tile,
    };
    use super::*;

    fn slugs(variants: &[&ModelVariant]) -> Vec<String> {
        variants.iter().map(|v| slug_of(v).to_string()).collect()
    }

    /// Slugs of `regions_at` for the tiles, then for the same tiles reversed:
    /// the pick must not depend on variant order.
    fn lookup(tiles: &[ModelVariant], at: GeoPoint) -> Vec<String> {
        let forward = slugs(&regions_at(&entry_with(tiles.to_vec()), at));
        let mut reversed = tiles.to_vec();
        reversed.reverse();
        let backward = slugs(&regions_at(&entry_with(reversed), at));
        assert_eq!(forward, backward, "order of variants changed the pick");
        forward
    }

    #[test]
    fn test_regions_at_contains_and_orders_by_tier_centre_and_slug() {
        let real = real_tiles();
        let rows: &[(&str, GeoPoint, &[&str])] = &[
            (
                "Helsinki",
                point(60.17, 24.94),
                &["nordic", "western-palearctic"],
            ),
            (
                "Berlin",
                point(52.52, 13.40),
                &["central-europe", "western-palearctic"],
            ),
            (
                "Nagpur",
                point(21.15, 79.09),
                &["central-india", "indo-gangetic"],
            ),
            ("Nairobi", point(-1.29, 36.82), &["east-africa"]),
            ("Ithaca", point(42.44, -76.50), &["north-america-east"]),
            ("mid Atlantic", point(30.0, -40.0), &[]),
            // Tie at tier 50: baltics' centre is 311 km away, nordic's 538 km.
            (
                "Tallinn",
                point(59.44, 24.75),
                &["baltics", "nordic", "western-palearctic"],
            ),
            // 1718 km to the Andes centre, 1892 km to Amazonia's.
            ("Quito", point(-0.18, -78.47), &["andes", "amazonia"]),
        ];
        for (name, at, expected) in rows {
            assert_eq!(&lookup(&real, *at), expected, "{name}");
        }

        // Edges are inside.
        let synthetic = [placed(
            "box",
            Some(1),
            &[bbox(10.0, 20.0, 30.0, 40.0)],
            None,
        )];
        for (lat, lon) in [(10.0, 35.0), (20.0, 35.0), (15.0, 30.0), (15.0, 40.0)] {
            assert_eq!(
                lookup(&synthetic, point(lat, lon)),
                ["box"],
                "edge {lat}, {lon}"
            );
        }
        assert_eq!(lookup(&synthetic, point(9.99, 35.0)), Vec::<String>::new());

        // Same everything: the slug decides, whatever the insertion order.
        let around = [bbox(-10.0, 10.0, -10.0, 10.0)];
        // Display names in the opposite order to the slugs, so `regions()` order
        // alone would put `b-tile` first.
        let mut twins = [
            placed("b-tile", Some(5), &around, Some(point(1.0, 1.0))),
            placed("a-tile", Some(5), &around, Some(point(1.0, 1.0))),
        ];
        twins[0].region_name = Some("Alpha".to_string());
        twins[1].region_name = Some("Zulu".to_string());
        assert_eq!(lookup(&twins, point(0.0, 0.0)), ["a-tile", "b-tile"]);

        // Rounding: on the equator one degree of longitude is 111.19 km, so
        // 0.9020 degrees is 100.30 km and 0.9002 degrees is 100.10 km. Both are
        // 100 km after rounding, so the slug wins although `b-tile` is nearer.
        let near_ties = [
            placed("a-tile", Some(5), &around, Some(point(0.0, 0.9020))),
            placed("b-tile", Some(5), &around, Some(point(0.0, 0.9002))),
        ];
        assert_eq!(lookup(&near_ties, point(0.0, 0.0)), ["a-tile", "b-tile"]);
        // A kilometre apart is a real difference.
        let apart = [
            placed("a-tile", Some(5), &around, Some(point(0.0, 0.9020))),
            placed("b-tile", Some(5), &around, Some(point(0.0, 0.8900))),
        ];
        assert_eq!(lookup(&apart, point(0.0, 0.0)), ["b-tile", "a-tile"]);

        // Tier beats distance.
        let tiers = [
            placed("near", Some(50), &around, Some(point(0.0, 0.1))),
            placed("far", Some(90), &around, Some(point(8.0, 8.0))),
        ];
        assert_eq!(lookup(&tiers, point(0.0, 0.0)), ["far", "near"]);

        // A missing tier is the lowest.
        let untiered = [
            placed("none", None, &around, Some(point(0.0, 0.0))),
            placed("some", Some(10), &around, Some(point(8.0, 8.0))),
        ];
        assert_eq!(lookup(&untiered, point(0.0, 0.0)), ["some", "none"]);

        // A missing or NaN centroid sorts after a tile that has one.
        let centres = [
            placed("a-missing", Some(5), &around, None),
            placed("b-nan", Some(5), &around, Some(point(f64::NAN, 0.0))),
            placed("c-real", Some(5), &around, Some(point(9.0, 9.0))),
        ];
        assert_eq!(
            lookup(&centres, point(0.0, 0.0)),
            ["c-real", "a-missing", "b-nan"]
        );

        // A tile without boxes is never listed.
        let boxless = [placed("boxless", Some(99), &[], Some(point(0.0, 0.0)))];
        assert_eq!(lookup(&boxless, point(0.0, 0.0)), Vec::<String>::new());
    }

    #[test]
    fn test_regions_at_measures_distance_across_the_antimeridian() {
        // Point (0, 179). `west` is centred 8 degrees away, `east` 3 degrees
        // away across the antimeridian, so a planar metric ranks them the other
        // way (8 against 357).
        let tiles = [
            placed(
                "west",
                Some(5),
                &[bbox(-10.0, 10.0, 170.0, 180.0)],
                Some(point(0.0, 171.0)),
            ),
            placed(
                "east",
                Some(5),
                &[
                    bbox(-10.0, 10.0, 175.0, 180.0),
                    bbox(-10.0, 10.0, -180.0, -170.0),
                ],
                Some(point(0.0, -178.0)),
            ),
        ];
        assert_eq!(lookup(&tiles, point(0.0, 179.0)), ["east", "west"]);

        // -180 and 180 are one meridian, whichever side the box is written on.
        let west_only = [tiles[0].clone()];
        assert_eq!(lookup(&west_only, point(0.0, -180.0)), ["west"]);
        let east_edge = [placed(
            "edge",
            Some(5),
            &[bbox(-10.0, 10.0, -180.0, -170.0)],
            None,
        )];
        assert_eq!(lookup(&east_edge, point(0.0, 180.0)), ["edge"]);
        // And a point well inside neither is not pulled in by the twin rule.
        assert_eq!(lookup(&west_only, point(0.0, -179.0)), Vec::<String>::new());
    }

    #[test]
    fn test_top_tier_ties_names_what_the_pick_beat_on_distance_only() {
        let entry = real_entry();
        let tallinn = regions_at(&entry, point(59.44, 24.75));
        assert_eq!(slugs(top_tier_ties(&tallinn)), ["nordic"]);
        let helsinki = regions_at(&entry, point(60.17, 24.94));
        assert_eq!(slugs(top_tier_ties(&helsinki)), Vec::<String>::new());
        assert_eq!(slugs(top_tier_ties(&[])), Vec::<String>::new());
    }

    #[test]
    fn test_regions_for_country_matches_codes_then_names_then_search() {
        let entry = real_entry();
        let by = |query: &str| slugs(&regions_for_country(&entry, query));

        assert_eq!(by("FI"), ["nordic", "western-palearctic"]);
        assert_eq!(by("fi"), ["nordic", "western-palearctic"]);
        // central-europe lists Denmark as partial and sorts first in `regions()`
        // order, so it is last only because core matches come first.
        let denmark = ["nordic", "western-palearctic", "central-europe"];
        assert_eq!(by("DK"), denmark);
        assert_eq!(by("Denmark"), denmark);
        assert_eq!(by("FINLAND"), ["nordic", "western-palearctic"]);
        assert_eq!(by("aland islands"), ["nordic", "western-palearctic"]);
        // The whole name, not a prefix: Nigeria is not Niger.
        assert_eq!(by("Niger"), ["western-palearctic"]);
        assert_eq!(by("nordic"), ["nordic"]);
        // Region names first, then the group name.
        assert_eq!(
            by("Europe"),
            [
                "central-europe",
                "southern-europe",
                "baltics",
                "nordic",
                "western-palearctic"
            ]
        );
        assert_eq!(by("xyz"), Vec::<String>::new());
        assert_eq!(by("  "), Vec::<String>::new());

        // One tile lists the code: only that tile, not every name containing "in".
        let mut india = tile("india", "India", 0);
        india.country_codes = Some(countries(&["IN"], &[]));
        india.countries = Some(countries(&["India"], &[]));
        let mut finland = tile("finland", "Finland", 0);
        finland.countries = Some(countries(&["Finland"], &[]));
        let single = entry_with(vec![india, finland]);
        assert_eq!(slugs(&regions_for_country(&single, "in")), ["india"]);
        // No tile carries codes, so "fi" falls through to a name search.
        assert_eq!(slugs(&regions_for_country(&single, "fi")), ["finland"]);

        // A tile that lists the accented name and one that lists the typographic
        // apostrophe.
        let mut ivory = tile("west-africa", "West Africa", 0);
        ivory.countries = Some(countries(&["Côte d\u{2019}Ivoire"], &[]));
        let ivory = entry_with(vec![ivory]);
        assert_eq!(
            slugs(&regions_for_country(&ivory, "cote d'ivoire")),
            ["west-africa"]
        );
    }

    #[test]
    fn test_regions_for_country_ranks_search_hits_name_core_partial_group() {
        // Alphabetical order of names is the reverse of the rank order, so only
        // ranking puts these right. Each hit starts a word, as the search needs.
        let mut alpha = tile("alpha", "Alpha", 0);
        alpha.group_name = Some("New World".to_string());
        let mut beta = tile("beta", "Beta", 0);
        beta.countries = Some(countries(&[], &["Papua New Guinea"]));
        let mut gamma = tile("gamma", "Gamma", 0);
        gamma.countries = Some(countries(&["New Zealand"], &[]));
        let delta = tile("delta", "New Highlands", 0);
        let entry = entry_with(vec![alpha, beta, gamma, delta]);
        assert_eq!(
            slugs(&regions_for_country(&entry, "new")),
            ["delta", "gamma", "beta", "alpha"]
        );
    }

    #[test]
    fn test_regions_for_country_matches_only_at_word_starts() {
        let mut new_zealand = tile("new-zealand", "New Zealand", 0);
        new_zealand.countries = Some(countries(&["New Zealand"], &[]));
        let mut nordic = tile("nordic", "Nordic", 0);
        nordic.countries = Some(countries(&["Åland Islands", "Finland"], &[]));
        let mut palearctic = tile("western-palearctic", "Western Palearctic", 1);
        palearctic.countries = Some(countries(&["Åland Islands"], &[]));
        // Synthetic names: "man" is mid-word in "Romania" before it starts
        // "Mansion", and the second "a a" in "Ba A A" overlaps the first.
        let mansion = tile("mansion", "Romania Mansion", 2);
        let overlap = tile("overlap", "Ba A A", 2);
        let entry = entry_with(vec![new_zealand, nordic, palearctic, mansion, overlap]);
        let by = |query: &str| slugs(&regions_for_country(&entry, query));

        // "aland" sits inside "zealand" and starts "Åland Islands": only the
        // tiles where it starts a word.
        assert_eq!(by("Aland"), ["nordic", "western-palearctic"]);
        // A fragment that starts a word still matches; one that does not, never.
        assert_eq!(by("zealand"), ["new-zealand"]);
        assert_eq!(by("ealand"), Vec::<String>::new());
        assert_eq!(by("land"), Vec::<String>::new());
        // A later word of a name starts a word too.
        assert_eq!(by("islands"), ["nordic", "western-palearctic"]);
        // The first occurrence is mid-word and a later one starts a word:
        // every occurrence is checked, not only the first.
        assert_eq!(by("man"), ["mansion"]);
        // A word-start occurrence that overlaps an earlier mid-word one.
        assert_eq!(by("a a"), ["overlap"]);
        // Only combining marks: folds to nothing, so it matches nothing.
        assert_eq!(by("\u{0301}"), Vec::<String>::new());
    }

    #[test]
    fn test_fold_drops_combining_marks_so_decomposed_spellings_match() {
        // Decomposed (u + U+0308), precomposed (U+00FC) and ASCII spellings of
        // one name all fold to the same text.
        assert_eq!(fold("Tu\u{0308}rkiye"), "turkiye");
        assert_eq!(fold("T\u{fc}rkiye"), "turkiye");
        assert_eq!(fold("Turkiye"), "turkiye");
    }

    #[test]
    fn test_fold_maps_iso_country_names_to_ascii() {
        let table: serde_json::Value =
            serde_json::from_str(include_str!("../../manifests/iso3166-1-alpha2.json")).unwrap();
        let names = table["names"].as_object().expect("names object");
        let not_ascii: Vec<String> = names
            .values()
            .filter_map(|v| v.as_str())
            .filter(|n| !fold(n).is_ascii())
            .map(|n| format!("{n} -> {}", fold(n)))
            .collect();
        assert!(not_ascii.is_empty(), "fold leaves non-ASCII: {not_ascii:?}");

        for (code, folded) in [
            ("AX", "åland islands"),
            ("BL", "st. barthelemy"),
            ("CI", "cote d'ivoire"),
            ("CW", "curacao"),
            ("RE", "reunion"),
            ("ST", "sao tome & principe"),
            ("TR", "turkiye"),
        ] {
            let name = names[code].as_str().unwrap();
            let expected = folded.replace('å', "a");
            assert_eq!(fold(name), expected, "{code}");
        }
    }

    fn inputs(request: RegionRequest<'_>) -> RegionInputs<'_> {
        RegionInputs {
            request,
            point: None,
            configured_lat: None,
            configured_lon: None,
            analyze_location_set: false,
            interactive: false,
            assume_yes: false,
            variant: None,
        }
    }

    /// A comparable rendering of a resolution.
    fn shown(resolved: Result<RegionResolution<'_>>) -> String {
        match resolved {
            Ok(RegionResolution::DefaultGlobal) => "DefaultGlobal".to_string(),
            Ok(RegionResolution::Global) => "Global".to_string(),
            Ok(RegionResolution::Named(slug)) => format!("Named({slug})"),
            Ok(RegionResolution::Located { matches, .. }) => {
                format!("Located[{}]", slugs(&matches).join(","))
            }
            Ok(RegionResolution::Ask { configured }) => configured.map_or_else(
                || "Ask(none)".to_string(),
                |p| format!("Ask({}, {})", p.lat, p.lon),
            ),
            Err(e) => format!("error: {e}"),
        }
    }

    #[test]
    fn test_resolve_region_follows_the_request_and_the_terminal() {
        let entry = real_entry();
        let mut flat_tiles = vec![tile("one", "One", 0), tile("two", "Two", 1)];
        flat_tiles.reverse();
        let flat = entry_with(flat_tiles);
        let mut no_regions = entry_with(Vec::new());
        no_regions.id = "global-only".to_string();
        let mut single_file = no_regions.clone();
        single_file.variants.clear();

        let helsinki = Some(point(60.17, 24.94));
        let configured = |lat, lon| RegionInputs {
            configured_lat: lat,
            configured_lon: lon,
            ..inputs(RegionRequest::Unspecified)
        };
        let ask = |base: RegionInputs<'static>| RegionInputs {
            interactive: true,
            ..base
        };
        let auto = |base: RegionInputs<'static>| RegionInputs {
            request: RegionRequest::Auto,
            ..base
        };

        // I1: not interactive means today's global install, configured or not.
        let at_home = configured(Some(60.17), Some(24.94));
        assert_eq!(shown(resolve_region(&entry, at_home)), "DefaultGlobal");
        let yes = RegionInputs {
            assume_yes: true,
            ..ask(at_home)
        };
        assert_eq!(shown(resolve_region(&entry, yes)), "DefaultGlobal");
        // Not even a broken configured location matters when nothing asks.
        let broken = configured(Some(200.0), None);
        assert_eq!(shown(resolve_region(&entry, broken)), "DefaultGlobal");

        // Interactive: ask, proposing the configured location when usable.
        assert_eq!(
            shown(resolve_region(&entry, ask(at_home))),
            "Ask(60.17, 24.94)"
        );
        assert_eq!(
            shown(resolve_region(&entry, ask(configured(None, None)))),
            "Ask(none)"
        );
        assert_eq!(shown(resolve_region(&flat, ask(at_home))), "Ask(none)");
        // A model that cannot use a location never rejects one.
        assert_eq!(
            shown(resolve_region(
                &flat,
                ask(configured(Some(200.0), Some(10.0)))
            )),
            "Ask(none)"
        );
        assert_eq!(
            shown(resolve_region(
                &entry,
                ask(configured(Some(200.0), Some(10.0)))
            )),
            "error: defaults.latitude in config.toml is out of range (must be -90.0 to 90.0); fix it with birda config set defaults.latitude <value>"
        );
        assert_eq!(
            shown(resolve_region(&entry, ask(configured(None, Some(200.0))))),
            "error: defaults.longitude in config.toml is out of range (must be -180.0 to 180.0); fix it with birda config set defaults.longitude <value>"
        );
        let typo = RegionInputs {
            variant: Some("int4"),
            ..ask(configured(None, None))
        };
        assert_eq!(
            shown(resolve_region(&entry, typo)),
            "error: model 'geo-test' has no variant 'int4'. Available: fp32"
        );
        assert_eq!(
            shown(resolve_region(&no_regions, ask(at_home))),
            "DefaultGlobal"
        );

        // Global and Named.
        let global = inputs(RegionRequest::Global);
        assert_eq!(shown(resolve_region(&single_file, global)), "Global");
        assert_eq!(shown(resolve_region(&entry, global)), "Global");
        let named_input = inputs(RegionRequest::Named("nordic"));
        assert_eq!(shown(resolve_region(&entry, named_input)), "Named(nordic)");
        assert_eq!(
            shown(resolve_region(&no_regions, named_input)),
            "error: model 'global-only' has no regional variants"
        );

        // Auto: the flag point beats the configured one.
        let flag_and_config = RegionInputs {
            point: helsinki,
            ..auto(configured(Some(42.44), Some(-76.5)))
        };
        assert_eq!(
            shown(resolve_region(&entry, flag_and_config)),
            "Located[nordic,western-palearctic]"
        );
        assert_eq!(
            shown(resolve_region(
                &entry,
                auto(configured(Some(60.17), Some(24.94)))
            )),
            "Located[nordic,western-palearctic]"
        );
        let mid_atlantic = RegionInputs {
            point: Some(point(30.0, -40.0)),
            ..auto(configured(None, None))
        };
        assert_eq!(shown(resolve_region(&entry, mid_atlantic)), "Located[]");

        let missing = "error: --region auto needs a location for 'geo-test': pass --lat and \
                       --lon after 'models install', or set defaults.latitude and \
                       defaults.longitude with 'birda config set'";
        for partial in [
            configured(Some(60.0), None),
            configured(None, Some(24.0)),
            configured(None, None),
        ] {
            assert_eq!(shown(resolve_region(&entry, auto(partial))), missing);
        }
        assert_eq!(
            shown(resolve_region(
                &entry,
                auto(configured(Some(60.0), Some(200.0)))
            )),
            "error: defaults.longitude in config.toml is out of range (must be -180.0 to 180.0); fix it with birda config set defaults.longitude <value>"
        );
        assert_eq!(
            shown(resolve_region(
                &no_regions,
                auto(configured(Some(60.0), Some(24.0)))
            )),
            "error: model 'global-only' has no regional variants"
        );
        assert_eq!(
            shown(resolve_region(
                &flat,
                auto(configured(Some(60.0), Some(24.0)))
            )),
            "error: model 'geo-test' publishes no region boundaries, so a region cannot be \
             picked from a location; choose one with --region <slug> (see 'birda models \
             regions geo-test')"
        );
        // The boundaries come first: no location, or a broken one, cannot
        // matter to a model that cannot use it.
        for bare in [configured(None, None), configured(Some(200.0), Some(24.0))] {
            let refused = shown(resolve_region(&flat, auto(bare)));
            assert!(
                refused.starts_with("error: model 'geo-test' publishes no region boundaries"),
                "{refused}"
            );
        }

        // Analyze options before `models` never pick the tile, but a point given
        // after `models install` does, whatever the analyze options say.
        let before_models = RegionInputs {
            analyze_location_set: true,
            ..auto(configured(Some(52.52), Some(13.4)))
        };
        assert_eq!(
            shown(resolve_region(&entry, before_models)),
            "error: --lat and --lon before 'models' (or BIRDA_LATITUDE/BIRDA_LONGITUDE) are \
             analyze options; pass --lat and --lon after 'models install' to pick a region"
        );
        let both = RegionInputs {
            point: helsinki,
            ..before_models
        };
        assert_eq!(
            shown(resolve_region(&entry, both)),
            "Located[nordic,western-palearctic]"
        );

        // A point is only for `auto`.
        for request in [RegionRequest::Global, RegionRequest::Named("nordic")] {
            let with_point = RegionInputs {
                point: helsinki,
                ..inputs(request)
            };
            assert_eq!(
                shown(resolve_region(&entry, with_point)),
                "error: --lat and --lon pick a region only with --region auto"
            );
        }
    }

    #[test]
    fn test_region_request_reads_the_flag_exactly() {
        for (arg, expected) in [
            (None, RegionRequest::Unspecified),
            (Some("auto"), RegionRequest::Auto),
            (Some("global"), RegionRequest::Global),
            (Some("Auto"), RegionRequest::Named("Auto")),
            (Some("nordic"), RegionRequest::Named("nordic")),
        ] {
            assert_eq!(RegionRequest::from_arg(arg), expected, "{arg:?}");
        }
    }

    #[test]
    fn test_located_summary_names_the_pick_the_ties_and_the_gap() {
        let entry = real_entry();
        let tallinn = point(59.44, 24.75);
        assert_eq!(
            located_summary(tallinn, &regions_at(&entry, tallinn)),
            "Region for 59.44, 24.75: baltics (Baltics).\nAlso equally specific here: nordic."
        );
        let helsinki = point(60.17, 24.94);
        assert_eq!(
            located_summary(helsinki, &regions_at(&entry, helsinki)),
            "Region for 60.17, 24.94: nordic (Nordic)."
        );
        let sea = point(30.0, -40.0);
        assert_eq!(
            located_summary(sea, &regions_at(&entry, sea)),
            "No regional model covers 30, -40."
        );
    }

    #[test]
    fn test_install_location_summary_names_the_region_but_never_the_point() {
        let entry = real_entry();
        let tallinn = point(59.44, 24.75);
        assert_eq!(
            install_location_summary(&regions_at(&entry, tallinn)),
            "Region for this location: baltics (Baltics).\nAlso equally specific here: nordic."
        );
        let sea = point(30.0, -40.0);
        assert_eq!(
            install_location_summary(&regions_at(&entry, sea)),
            "No regional model covers this location."
        );
    }

    #[test]
    fn test_global_variant_prefers_the_default_among_unregioned_variants() {
        let mut entry = real_entry();
        assert_eq!(global_variant(&entry).map(|v| v.id.as_str()), Some("fp32"));
        let mut fp16 = entry.variants[0].clone();
        fp16.id = "fp16".to_string();
        entry.variants.insert(0, fp16);
        assert_eq!(global_variant(&entry).map(|v| v.id.as_str()), Some("fp32"));
        entry.default_variant = None;
        assert_eq!(global_variant(&entry).map(|v| v.id.as_str()), Some("fp16"));
    }
}
