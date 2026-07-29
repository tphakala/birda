//! Species list generation from range filter.

use crate::cli::SortOrder;
use crate::config::{OutputMode, load_default_config};
use crate::error::{Error, Result};
use crate::inference::SpeciesMapping;
use crate::inference::range_filter::RangeFilter;
use crate::output::{ResultType, SpeciesEntry, SpeciesListPayload, emit_json_result};
use crate::utils::date::{date_to_week, day_of_year_to_date, week_to_start_day};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

/// Default output file name.
const DEFAULT_OUTPUT_FILE: &str = "species_list.txt";

/// Generate species list from range filter predictions.
///
/// # Arguments
/// - `output`: Output file path (None = default to `species_list.txt`)
/// - `lat`: Latitude (-90.0 to 90.0)
/// - `lon`: Longitude (-180.0 to 180.0)
/// - `week`: Week number (1-48)
/// - `month`: Month (1-12)
/// - `day`: Day of month (1-31)
/// - `threshold`: Range filter threshold (0.0-1.0)
/// - `sort`: Sort order (Freq or Alpha)
/// - `model`: Model name to use
/// - `output_mode`: Output mode (Human, Json, Ndjson)
///
/// # Errors
/// Returns error if:
/// - Config cannot be loaded
/// - Model not found
/// - The `BirdNET` Geomodel is not installed and could not be acquired
/// - Range filter prediction fails
/// - Cannot write output file
#[allow(clippy::too_many_arguments)]
pub fn generate_species_list(
    output: Option<PathBuf>,
    lat: f64,
    lon: f64,
    week: Option<u32>,
    month: Option<u32>,
    day: Option<u32>,
    threshold: f32,
    sort: SortOrder,
    model: Option<String>,
    output_mode: OutputMode,
    geomodel_request: crate::config::GeomodelRequest<'_>,
) -> Result<()> {
    // Load configuration
    let config = load_default_config()?;

    // Determine model to use
    let model_name = model
        .or_else(|| config.defaults.model.clone())
        .ok_or_else(|| Error::ConfigValidation {
            message: "no model specified (use -m or set defaults.model in config)".to_string(),
        })?;

    let model_config = crate::config::get_model(&config, &model_name)?;

    // Resolve the shared geomodel, offering to download it when absent.
    //
    // Unlike the analyze path, an unavailable geomodel is fatal here: producing
    // a species list is the entire purpose of this command, so there is nothing
    // useful to fall back to.
    let registry = crate::registry::load_registry()?;
    let geomodel =
        match crate::config::resolve_geomodel(geomodel_request, &config, &registry, output_mode)? {
            crate::config::GeomodelResolution::Ready(paths) => paths,
            crate::config::GeomodelResolution::Unavailable(hint) => {
                return Err(Error::GeomodelNotInstalled { hint });
            }
        };

    let is_json = output_mode.is_structured();

    // Read classifier labels
    if !is_json {
        println!(
            "Loading model labels from: {}",
            model_config.labels.display()
        );
    }
    let labels = read_labels_file(&model_config.labels)?;
    if !is_json {
        println!("Loaded {} species labels", labels.len());
    }

    // Get month/day for range filter
    let (filter_month, filter_day) = if let Some(week_num) = week {
        // Convert week to approximate month/day
        week_to_date(week_num)
    } else if let (Some(m), Some(d)) = (month, day) {
        // Use exact date specified by user
        (m, d)
    } else {
        return Err(Error::ConfigValidation {
            message: "either --week or --month+--day must be specified".to_string(),
        });
    };

    // Calculate week for JSON output using canonical date_to_week function
    let week_num = week.unwrap_or_else(|| date_to_week(filter_month, filter_day));

    // Build the range filter from the GEOMODEL's labels, not the classifier's.
    // birdnet-onnx validates that the label count matches the model's output
    // size, and no classifier has the geomodel's 12,012 classes. Scores are
    // projected back into the classifier's label space afterwards, so the
    // output stays usable as a --species-list for that model.
    if !is_json {
        println!(
            "Loading BirdNET Geomodel v3.0.2: {}",
            geomodel.model.display()
        );
    }
    let geomodel_labels = read_labels_file(&geomodel.labels)?;
    let range_filter = RangeFilter::from_config(
        &geomodel.model,
        &geomodel_labels,
        crate::constants::range_filter::GEOMODEL_QUERY_THRESHOLD,
    )?;

    // Get location scores
    if !is_json {
        println!(
            "Predicting species for: lat={lat:.4}, lon={lon:.4}, month={filter_month}, day={filter_day}, threshold={threshold}"
        );
    }
    let location_scores = range_filter.predict(lat, lon, filter_month, filter_day)?;

    let mapping = SpeciesMapping::build(&geomodel_labels, &labels);
    let species_list = build_species_entries(&location_scores, &mapping, threshold, sort);

    if !is_json {
        println!(
            "{} of {} model species have BirdNET Geomodel v3.0.2 coverage",
            mapping.mapped_count(),
            mapping.total_classifier_species()
        );
        println!(
            "Found {} species above threshold {:.3}",
            species_list.len(),
            threshold
        );
    }

    // Determine output file path (only used for human mode)
    let output_path = output.unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_FILE));

    // Write species list to file (only in human mode)
    if !is_json {
        write_species_list(&output_path, &species_list)?;
    }

    // JSON/NDJSON output
    if is_json {
        // Parse species names into scientific/common name pairs
        let species_entries: Vec<SpeciesEntry> = species_list
            .iter()
            .map(|(label, score)| {
                // Label format is typically "Genus species_Common Name"
                let (scientific, common) = if let Some((s, c)) = label.split_once('_') {
                    (s.to_string(), c.to_string())
                } else {
                    (label.clone(), String::new())
                };
                SpeciesEntry {
                    scientific_name: scientific,
                    common_name: common,
                    frequency: *score,
                }
            })
            .collect();

        let payload = SpeciesListPayload {
            result_type: ResultType::SpeciesList,
            lat,
            lon,
            week: week_num,
            threshold,
            species_count: species_entries.len(),
            output_file: None, // No file written in JSON mode
            species: species_entries,
        };
        emit_json_result(&payload);
        return Ok(());
    }

    // Human-readable output
    println!("Species list written to: {}", output_path.display());
    println!(
        "Sort order: {}",
        match sort {
            SortOrder::Freq => "by occurrence probability",
            SortOrder::Alpha => "alphabetically",
        }
    );

    Ok(())
}

/// Project geomodel scores onto the classifier's labels and sort them.
///
/// Emits the classifier's own label for each species, so the resulting file can
/// be fed straight back in with `--slist` for that model. Species below
/// `threshold`, and geomodel species the classifier cannot predict, are omitted.
fn build_species_entries(
    location_scores: &[birdnet_onnx::LocationScore],
    mapping: &SpeciesMapping,
    threshold: f32,
    sort: SortOrder,
) -> Vec<(String, f32)> {
    let mut species_list: Vec<(String, f32)> = location_scores
        .iter()
        .filter(|score| score.score >= threshold)
        .filter_map(|score| {
            mapping
                .classifier_label_for(&score.species)
                .map(|label| (label.to_string(), score.score))
        })
        .collect();

    // sort_unstable_by is fine here: ties between equal scores or equal labels
    // are not meaningful.
    match sort {
        SortOrder::Freq => species_list.sort_unstable_by(|a, b| b.1.total_cmp(&a.1)),
        SortOrder::Alpha => species_list.sort_unstable_by(|a, b| a.0.cmp(&b.0)),
    }

    species_list
}

/// Convert week number to month/day.
///
/// Week 1 = Jan 1 (day 1), Week 48 = Dec 24 (day 358)
fn week_to_date(week: u32) -> (u32, u32) {
    day_of_year_to_date(week_to_start_day(week))
}

/// Read labels file.
fn read_labels_file(path: &std::path::Path) -> Result<Vec<String>> {
    use std::io::BufRead;

    let file = File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Error::LabelsFileNotFound {
                path: path.to_path_buf(),
            }
        } else {
            Error::Io(e)
        }
    })?;

    let reader = std::io::BufReader::new(file);
    let mut labels = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(Error::Io)?;
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            labels.push(trimmed.to_string());
        }
    }

    Ok(labels)
}

/// Write species list to file.
///
/// Format: `Genus species_Common Name` (one per line)
fn write_species_list(path: &std::path::Path, species: &[(String, f32)]) -> Result<()> {
    let mut file = File::create(path).map_err(Error::Io)?;

    for (label, _score) in species {
        writeln!(file, "{label}").map_err(Error::Io)?;
    }

    file.flush().map_err(Error::Io)?;

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    fn labels_of(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_string()).collect()
    }

    fn score(species: &str, score: f32, index: usize) -> birdnet_onnx::LocationScore {
        birdnet_onnx::LocationScore {
            species: species.to_string(),
            score,
            index,
        }
    }

    #[test]
    fn test_species_list_emits_classifier_labels_not_geomodel_labels() {
        // The output is fed back in as a --species-list for this model, so it
        // must carry the model's own labels, in the user's chosen language.
        let geomodel = labels_of(&["Parus major_Great Tit"]);
        let classifier = labels_of(&["Parus major_Talitiainen"]);
        let mapping = SpeciesMapping::build(&geomodel, &classifier);

        let entries = build_species_entries(
            &[score("Parus major_Great Tit", 0.9, 0)],
            &mapping,
            0.01,
            SortOrder::Freq,
        );

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "Parus major_Talitiainen");
    }

    #[test]
    fn test_species_list_excludes_scores_below_threshold() {
        let geomodel = labels_of(&["Aaa aaa_X", "Bbb bbb_Y"]);
        let mapping = SpeciesMapping::build(&geomodel, &geomodel);

        let entries = build_species_entries(
            &[score("Aaa aaa_X", 0.9, 0), score("Bbb bbb_Y", 0.001, 1)],
            &mapping,
            0.01,
            SortOrder::Freq,
        );

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "Aaa aaa_X");
    }

    #[test]
    fn test_species_list_omits_species_the_model_cannot_predict() {
        // The geomodel covers mammals and insects that no bird classifier
        // predicts; listing them would produce an unusable species list.
        let geomodel = labels_of(&["Parus major_Great Tit", "Vulpes vulpes_Red Fox"]);
        let classifier = labels_of(&["Parus major_Great Tit"]);
        let mapping = SpeciesMapping::build(&geomodel, &classifier);

        let entries = build_species_entries(
            &[
                score("Parus major_Great Tit", 0.9, 0),
                score("Vulpes vulpes_Red Fox", 0.95, 1),
            ],
            &mapping,
            0.01,
            SortOrder::Freq,
        );

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "Parus major_Great Tit");
    }

    #[test]
    fn test_species_list_freq_sort_is_most_likely_first() {
        let geomodel = labels_of(&["Zzz zzz_Zebra Finch", "Aaa aaa_Ant Bird"]);
        let mapping = SpeciesMapping::build(&geomodel, &geomodel);

        let entries = build_species_entries(
            &[
                score("Aaa aaa_Ant Bird", 0.5, 0),
                score("Zzz zzz_Zebra Finch", 0.9, 1),
            ],
            &mapping,
            0.01,
            SortOrder::Freq,
        );

        assert_eq!(entries[0].0, "Zzz zzz_Zebra Finch");
    }

    #[test]
    fn test_species_list_alpha_sort_orders_by_label() {
        let geomodel = labels_of(&["Zzz zzz_Zebra Finch", "Aaa aaa_Ant Bird"]);
        let mapping = SpeciesMapping::build(&geomodel, &geomodel);

        let entries = build_species_entries(
            &[
                score("Zzz zzz_Zebra Finch", 0.9, 0),
                score("Aaa aaa_Ant Bird", 0.5, 1),
            ],
            &mapping,
            0.01,
            SortOrder::Alpha,
        );

        assert_eq!(entries[0].0, "Aaa aaa_Ant Bird");
    }

    #[test]
    fn test_species_list_is_empty_when_nothing_maps() {
        let mapping = SpeciesMapping::build(
            &labels_of(&["Parus major_Great Tit"]),
            &labels_of(&["Dog_Dog"]),
        );

        let entries = build_species_entries(
            &[score("Parus major_Great Tit", 0.9, 0)],
            &mapping,
            0.01,
            SortOrder::Freq,
        );

        assert!(entries.is_empty());
    }

    #[test]
    fn test_week_to_date_week_1() {
        assert_eq!(week_to_date(1), (1, 1));
    }

    #[test]
    fn test_week_to_date_week_24() {
        // Week 24 = day 175 → June 24
        assert_eq!(week_to_date(24), (6, 24));
    }

    #[test]
    fn test_week_to_date_week_48() {
        // Week 48 = day 358 → Dec 24
        assert_eq!(week_to_date(48), (12, 24));
    }
}
