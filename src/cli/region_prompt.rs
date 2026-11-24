//! The question `birda models install` asks when it can: where the user records.
//!
//! The dialogue runs over an injected reader and writer so a test can drive it
//! without a terminal. It only picks a slug; what the slug installs is decided
//! by variant selection, and which slug a location maps to by
//! [`crate::registry::region`].

use std::io::{BufRead, Write};

use super::validators::{parse_latitude, parse_longitude};
use crate::config::geomodel::human_size;
use crate::constants::regions::{COORDINATE_SEPARATOR, PROMPT_ATTEMPTS};
use crate::error::Result;
use crate::registry::region::{
    global_variant, has_geometry, located_summary, regions_at, regions_for_country,
};
use crate::registry::{GeoPoint, ModelEntry, ModelVariant, species_count_label};

/// What the user typed at "Where do you record?".
#[derive(Debug, Clone, PartialEq)]
pub enum LocationInput {
    /// Nothing, which asks for the global model.
    Empty,
    /// A valid `lat,lon`.
    Point(GeoPoint),
    /// Anything else, taken as a country name or ISO code.
    Country(String),
    /// `lat,lon` with an unusable number; the message says which.
    Invalid(String),
}

/// How the question ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptedRegion {
    /// Install this region.
    Region(String),
    /// Install the global model.
    Global,
    /// The user gave up: end of input, or too many unusable answers.
    Cancelled,
}

/// Read what the user typed as a location.
///
/// Exactly two comma-separated numbers are a point, checked against the same
/// bounds as `--lat` and `--lon`. Anything else is a country, so a name that
/// contains commas ("St. Helena, Ascension & Tristan da Cunha") stays one.
#[must_use]
pub fn parse_location_input(text: &str) -> LocationInput {
    let text = text.trim();
    if text.is_empty() {
        return LocationInput::Empty;
    }
    let parts: Vec<&str> = text.split(COORDINATE_SEPARATOR).collect();
    if let [lat, lon] = parts.as_slice()
        && lat.trim().parse::<f64>().is_ok()
        && lon.trim().parse::<f64>().is_ok()
    {
        return match (parse_latitude(lat), parse_longitude(lon)) {
            (Ok(lat), Ok(lon)) => LocationInput::Point(GeoPoint { lat, lon }),
            (Err(message), _) | (_, Err(message)) => LocationInput::Invalid(message),
        };
    }
    LocationInput::Country(text.to_string())
}

/// Write `text`, flush, and read one trimmed line; `None` at end of input.
fn read_answer(input: &mut dyn BufRead, out: &mut dyn Write, text: &str) -> Result<Option<String>> {
    write!(out, "{text}")?;
    out.flush()?;
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        // The prompt line was never ended by an answer.
        writeln!(out)?;
        return Ok(None);
    }
    Ok(Some(line.trim().to_string()))
}

/// "479 species, 97.8 MB", the two numbers a user picks between.
fn size_line(variant: &ModelVariant) -> String {
    format!(
        "{}, {}",
        species_count_label(variant.classes),
        human_size(variant.model.size_bytes)
    )
}

/// Ask where the user records and which model that means.
///
/// With a `configured` location the first question proposes the region it falls
/// in; answering no, or having no location, leads to the second, which takes a
/// country or coordinates and lists the matching regions beside the global
/// model. Every question gives up after [`PROMPT_ATTEMPTS`] unusable answers.
///
/// # Errors
///
/// I/O errors from the reader or writer.
pub fn ask_region(
    entry: &ModelEntry,
    configured: Option<GeoPoint>,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
) -> Result<PromptedRegion> {
    if let Some(point) = configured
        && let Some(decided) = propose(entry, point, input, out)?
    {
        return Ok(decided);
    }
    ask_location(entry, input, out)
}

/// Step A: propose the region for the configured location. `None` when the user
/// declined and should be asked where they record instead.
fn propose(
    entry: &ModelEntry,
    point: GeoPoint,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
) -> Result<Option<PromptedRegion>> {
    let matches = regions_at(entry, point);
    writeln!(out, "{}", located_summary(point, &matches))?;
    let (question, accepted) = if let Some(best) = matches.first() {
        let slug = best.region.clone().unwrap_or_default();
        writeln!(out, "{slug}: {}", size_line(best))?;
        (
            format!("Install {slug}? [Y/n]: "),
            PromptedRegion::Region(slug),
        )
    } else {
        if let Some(global) = global_variant(entry) {
            writeln!(out, "Global model: {}", size_line(global))?;
        }
        (
            "Install the global model? [Y/n]: ".to_string(),
            PromptedRegion::Global,
        )
    };

    for _ in 0..PROMPT_ATTEMPTS {
        let Some(answer) = read_answer(input, out, &question)? else {
            return Ok(Some(PromptedRegion::Cancelled));
        };
        match answer.to_ascii_lowercase().as_str() {
            "" | "y" => return Ok(Some(accepted)),
            "n" => return Ok(None),
            _ => writeln!(out, "Please answer y or n.")?,
        }
    }
    Ok(Some(PromptedRegion::Cancelled))
}

/// Step B: a country or coordinates, then a numbered choice.
fn ask_location(
    entry: &ModelEntry,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
) -> Result<PromptedRegion> {
    let geometry = has_geometry(entry);
    if geometry {
        writeln!(
            out,
            "Where do you record? Enter a country (name or ISO code) or coordinates as lat,lon. \
             Press Enter for the global model."
        )?;
    } else {
        writeln!(
            out,
            "Where do you record? Enter a country (name or ISO code). \
             Press Enter for the global model."
        )?;
    }

    for _ in 0..PROMPT_ATTEMPTS {
        let Some(answer) = read_answer(input, out, "Location: ")? else {
            return Ok(PromptedRegion::Cancelled);
        };
        let tiles = match parse_location_input(&answer) {
            LocationInput::Empty => return Ok(PromptedRegion::Global),
            LocationInput::Invalid(message) => {
                writeln!(out, "{message}")?;
                continue;
            }
            LocationInput::Point(_) if !geometry => {
                writeln!(
                    out,
                    "This model publishes no region boundaries; enter a country."
                )?;
                continue;
            }
            LocationInput::Point(point) => {
                let tiles = regions_at(entry, point);
                writeln!(out, "{}", located_summary(point, &tiles))?;
                tiles
            }
            LocationInput::Country(name) => {
                let tiles = regions_for_country(entry, &name);
                if tiles.is_empty() {
                    writeln!(out, "No region lists '{name}'.")?;
                    continue;
                }
                tiles
            }
        };
        return choose(entry, &tiles, input, out);
    }
    Ok(PromptedRegion::Cancelled)
}

/// List `tiles` and the global model as numbered choices and read one.
fn choose(
    entry: &ModelEntry,
    tiles: &[&ModelVariant],
    input: &mut dyn BufRead,
    out: &mut dyn Write,
) -> Result<PromptedRegion> {
    for (index, tile) in tiles.iter().enumerate() {
        writeln!(
            out,
            "  {}) {} ({}): {}",
            index + 1,
            tile.region.as_deref().unwrap_or_default(),
            tile.region_name.as_deref().unwrap_or_default(),
            size_line(tile)
        )?;
    }
    let global_number = tiles.len() + 1;
    match global_variant(entry) {
        Some(global) => writeln!(
            out,
            "  {global_number}) Global model: {}",
            size_line(global)
        )?,
        None => writeln!(out, "  {global_number}) Global model")?,
    }

    let question = format!("Choose 1-{global_number}: ");
    for _ in 0..PROMPT_ATTEMPTS {
        let Some(answer) = read_answer(input, out, &question)? else {
            return Ok(PromptedRegion::Cancelled);
        };
        match answer.parse::<usize>() {
            Ok(n) if n == global_number => return Ok(PromptedRegion::Global),
            Ok(n) if (1..global_number).contains(&n) => {
                let slug = n
                    .checked_sub(1)
                    .and_then(|i| tiles.get(i))
                    .and_then(|tile| tile.region.clone());
                if let Some(slug) = slug {
                    return Ok(PromptedRegion::Region(slug));
                }
            }
            _ => {}
        }
        writeln!(out, "Enter a number from 1 to {global_number}.")?;
    }
    Ok(PromptedRegion::Cancelled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::region::test_fixtures::{countries, entry_with, point, real_entry, tile};

    #[test]
    fn test_parse_location_input() {
        for (text, expected) in [
            ("60.17,24.94", LocationInput::Point(point(60.17, 24.94))),
            (" -1.29 , 36.82 ", LocationInput::Point(point(-1.29, 36.82))),
            // Longitude past 90 is fine; latitude bounds must not be applied to it.
            ("35.68, 139.69", LocationInput::Point(point(35.68, 139.69))),
            (
                "95,10",
                LocationInput::Invalid(parse_latitude("95").unwrap_err()),
            ),
            (
                "10,195",
                LocationInput::Invalid(parse_longitude("195").unwrap_err()),
            ),
            (
                "nan,10",
                LocationInput::Invalid(parse_latitude("nan").unwrap_err()),
            ),
            (
                "St. Helena, Ascension & Tristan da Cunha",
                LocationInput::Country("St. Helena, Ascension & Tristan da Cunha".to_string()),
            ),
            ("Finland", LocationInput::Country("Finland".to_string())),
            ("", LocationInput::Empty),
            ("   ", LocationInput::Empty),
        ] {
            assert_eq!(parse_location_input(text), expected, "{text:?}");
        }
    }

    /// Run the dialogue on `answers`; returns the outcome and everything written.
    fn run(
        entry: &ModelEntry,
        configured: Option<GeoPoint>,
        answers: &str,
    ) -> (PromptedRegion, String) {
        let mut input = answers.as_bytes();
        let mut out = Vec::new();
        let outcome = ask_region(entry, configured, &mut input, &mut out).unwrap();
        (outcome, String::from_utf8(out).unwrap())
    }

    fn region(slug: &str) -> PromptedRegion {
        PromptedRegion::Region(slug.to_string())
    }

    #[test]
    fn test_ask_region_proposes_the_configured_location_and_shows_ties() {
        let entry = real_entry();

        let (outcome, shown) = run(&entry, Some(point(59.44, 24.75)), "\n");
        assert_eq!(outcome, region("baltics"));
        assert_eq!(
            shown,
            "Region for 59.44, 24.75: baltics (Baltics).\n\
             Also equally specific here: nordic.\n\
             baltics: 479 species, 97.8 MB\n\
             Install baltics? [Y/n]: "
        );

        for yes in ["y\n", "Y\n", "\n"] {
            assert_eq!(
                run(&entry, Some(point(60.17, 24.94)), yes).0,
                region("nordic"),
                "{yes:?}"
            );
        }

        let (outcome, shown) = run(&entry, Some(point(30.0, -40.0)), "\n");
        assert_eq!(outcome, PromptedRegion::Global);
        assert_eq!(
            shown,
            "No regional model covers 30, -40.\n\
             Global model: 11000 species, 190.7 MB\n\
             Install the global model? [Y/n]: "
        );
    }

    #[test]
    fn test_ask_region_falls_through_to_the_country_question_on_no() {
        let entry = real_entry();
        let (outcome, shown) = run(&entry, Some(point(60.17, 24.94)), "n\nFinland\n2\n");
        assert_eq!(outcome, region("western-palearctic"));
        assert_eq!(
            shown,
            "Region for 60.17, 24.94: nordic (Nordic).\n\
             nordic: 464 species, 97.5 MB\n\
             Install nordic? [Y/n]: \
             Where do you record? Enter a country (name or ISO code) or coordinates as lat,lon. \
             Press Enter for the global model.\n\
             Location: \
             \x20 1) nordic (Nordic): 464 species, 97.5 MB\n\
             \x20 2) western-palearctic (Western Palearctic): 1446 species, 117.6 MB\n\
             \x20 3) Global model: 11000 species, 190.7 MB\n\
             Choose 1-3: "
        );
    }

    #[test]
    fn test_ask_region_without_a_location_takes_a_country_or_a_point() {
        let entry = real_entry();
        let ask = |answers: &str| run(&entry, None, answers).0;

        // The last row is the global model, with tiles present.
        assert_eq!(ask("Finland\n3\n"), PromptedRegion::Global);
        assert_eq!(ask("60.17, 24.94\n1\n"), region("nordic"));
        // A point no tile covers lists only the global row.
        assert_eq!(ask("30,-40\n1\n"), PromptedRegion::Global);
        assert_eq!(ask("\n"), PromptedRegion::Global);
        assert_eq!(ask("xyz\nFI\n1\n"), region("nordic"));
        assert_eq!(ask("95,10\n60.17,24.94\n1\n"), region("nordic"));
        // Zero re-asks without underflowing the index.
        assert_eq!(ask("Finland\n0\n1\n"), region("nordic"));
    }

    #[test]
    fn test_ask_region_gives_up_at_end_of_input_and_after_three_bad_answers() {
        let entry = real_entry();
        let at_home = Some(point(60.17, 24.94));
        for (location, answers) in [
            (None, ""),
            (at_home, ""),
            (None, "Finland\n9\n9\n9\n"),
            (None, "x\nx\nx\n"),
            (at_home, "q\nq\nq\n"),
            (None, "Finland\n"),
        ] {
            assert_eq!(
                run(&entry, location, answers).0,
                PromptedRegion::Cancelled,
                "{answers:?}"
            );
        }
        // Two bad answers are survivable.
        assert_eq!(run(&entry, None, "x\nx\nFI\n1\n").0, region("nordic"));
    }

    #[test]
    fn test_ask_region_on_an_entry_without_boundaries_refuses_points() {
        let mut finland = tile("finland", "Finland", 0);
        finland.countries = Some(countries(&["Finland"], &[]));
        let flat = entry_with(vec![finland]);
        let (outcome, shown) = run(&flat, None, "60,24\nFinland\n1\n");
        assert_eq!(outcome, region("finland"));
        assert!(
            shown.starts_with(
                "Where do you record? Enter a country (name or ISO code). \
                 Press Enter for the global model.\n"
            ),
            "{shown}"
        );
        assert!(
            shown.contains("This model publishes no region boundaries; enter a country.\n"),
            "{shown}"
        );
    }
}
