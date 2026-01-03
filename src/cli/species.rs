//! Species list generation from range filter.

use crate::cli::SortOrder;
use crate::config::load_default_config;
use crate::constants::range_filter::DAYS_PER_WEEK;
use crate::error::{Error, Result};
use crate::inference::range_filter::RangeFilter;
use crate::utils::date::day_of_year_to_date;
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

    // Read classifier labels
    println!(
        "Loading model labels from: {}",
        model_config.labels.display()
    );
    let labels = read_labels_file(&model_config.labels)?;
    println!("Loaded {} species labels", labels.len());

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

    // Build range filter
    println!("Loading range filter model: {}", meta_model_path.display());
    let range_filter = RangeFilter::from_config(meta_model_path, &labels, threshold)?;

    // Get location scores
    println!(
        "Predicting species for: lat={lat:.4}, lon={lon:.4}, month={filter_month}, day={filter_day}, threshold={threshold}"
    );
    let location_scores = range_filter.predict(lat, lon, filter_month, filter_day)?;

    // Filter species based on threshold and create list
    let mut species_list: Vec<(String, f32)> = location_scores
        .iter()
        .filter(|score| score.score >= threshold)
        .map(|score| (score.species.clone(), score.score))
        .collect();

    println!(
        "Found {} species above threshold {:.3}",
        species_list.len(),
        threshold
    );

    // Sort according to user preference
    match sort {
        SortOrder::Freq => {
            // Sort by score descending (most likely first)
            species_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        SortOrder::Alpha => {
            // Sort alphabetically
            species_list.sort_by(|a, b| a.0.cmp(&b.0));
        }
    }

    // Determine output file path
    let output_path = output.unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_FILE));

    // Write species list to file
    write_species_list(&output_path, &species_list)?;

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
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let day_of_year = ((week - 1) as f32).mul_add(DAYS_PER_WEEK, 1.0) as u32;
    day_of_year_to_date(day_of_year)
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

    #[test]
    fn test_day_of_year_to_date_jan_1() {
        assert_eq!(day_of_year_to_date(1), (1, 1));
    }

    #[test]
    fn test_day_of_year_to_date_dec_31() {
        assert_eq!(day_of_year_to_date(365), (12, 31));
    }

    #[test]
    fn test_day_of_year_to_date_jun_15() {
        assert_eq!(day_of_year_to_date(166), (6, 15));
    }

    #[test]
    fn test_day_of_year_to_date_overflow() {
        // Day 400 should return Dec 31 (overflow protection)
        assert_eq!(day_of_year_to_date(400), (12, 31));
    }
}
