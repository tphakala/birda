//! Species list generation from range filter.

use crate::cli::SortOrder;
use crate::config::{OutputMode, load_default_config};
use crate::error::{Error, Result};
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
/// - Model not found or has no meta model
/// - Meta model file not found
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

    // Get meta model path
    let meta_model_path = model_config
        .meta_model
        .as_ref()
        .or(config.defaults.meta_model.as_ref())
        .ok_or_else(|| Error::MetaModelMissing {
            model_name: model_name.clone(),
        })?;

    // Verify meta model file exists
    if !meta_model_path.exists() {
        return Err(Error::MetaModelNotFound {
            path: meta_model_path.clone(),
        });
    }

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

    // Build range filter
    if !is_json {
        println!("Loading range filter model: {}", meta_model_path.display());
    }
    let range_filter = RangeFilter::from_config(meta_model_path, &labels, threshold)?;

    // Get location scores
    if !is_json {
        println!(
            "Predicting species for: lat={lat:.4}, lon={lon:.4}, month={filter_month}, day={filter_day}, threshold={threshold}"
        );
    }
    let location_scores = range_filter.predict(lat, lon, filter_month, filter_day)?;

    // Filter species based on threshold and create list
    let mut species_list: Vec<(String, f32)> = location_scores
        .iter()
        .filter(|score| score.score >= threshold)
        .map(|score| (score.species.clone(), score.score))
        .collect();

    if !is_json {
        println!(
            "Found {} species above threshold {:.3}",
            species_list.len(),
            threshold
        );
    }

    // Sort according to user preference
    // Use sort_unstable_by for performance - stability not needed here
    match sort {
        SortOrder::Freq => {
            // Sort by score descending (most likely first)
            species_list.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        }
        SortOrder::Alpha => {
            // Sort alphabetically
            species_list.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        }
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
