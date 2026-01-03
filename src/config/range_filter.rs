//! Range filter configuration resolution.

use crate::cli::AnalyzeArgs;
use crate::config::types::{Config, ModelConfig};
use crate::constants::range_filter::DAYS_PER_WEEK;
use crate::error::{Error, Result};
use crate::inference::RangeFilterConfig;
use crate::utils::date::date_to_week;

/// Build `RangeFilterConfig` from CLI args and config file.
///
/// Range filtering activates when:
/// - Coordinates are available (CLI or config)
/// - Time parameter is available (week OR month+day)
pub fn build_range_filter_config(
    args: &AnalyzeArgs,
    config: &Config,
    model_config: &ModelConfig,
) -> Result<Option<RangeFilterConfig>> {
    // Get coordinates (CLI overrides config)
    let latitude = args.lat.or(config.defaults.latitude);
    let longitude = args.lon.or(config.defaults.longitude);

    // Range filtering requires both coordinates
    let (Some(latitude), Some(longitude)) = (latitude, longitude) else {
        return Ok(None); // No coordinates - range filtering disabled
    };

    // Determine time: week or month/day
    let (month, day) = if let Some(_week) = args.week {
        // Use week directly - convert to approximate month/day
        // Week 1 = Jan 1, Week 48 = Dec 31
        // This is approximate but week will be used for actual filtering
        (1, 1) // Placeholder, actual implementation uses week
    } else if let (Some(month), Some(day)) = (args.month, args.day) {
        (month, day)
    } else {
        // No time parameter - range filtering disabled
        return Ok(None);
    };

    // Get week number (convert from month/day if needed)
    let week = args.week.unwrap_or_else(|| date_to_week(month, day));

    // Convert week back to month/day for RangeFilter::predict
    // Week 1 = ~Jan 4 (day 4), Week 48 = ~Dec 29 (day 363)
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let day_of_year = ((week - 1) as f32).mul_add(DAYS_PER_WEEK, 1.0) as u32;
    let (actual_month, actual_day) = day_of_year_to_date(day_of_year);

    // Get meta model path (per-model overrides global default)
    let meta_model_path = model_config
        .meta_model
        .as_ref()
        .or(config.defaults.meta_model.as_ref())
        .ok_or_else(|| Error::MetaModelMissing {
            model_name: "unknown".to_string(), // TODO: pass model name
        })?;

    // Get threshold (CLI overrides config)
    let threshold = args
        .range_threshold
        .unwrap_or(config.defaults.range_threshold);

    Ok(Some(RangeFilterConfig {
        meta_model_path: meta_model_path.clone(),
        threshold,
        latitude,
        longitude,
        month: actual_month,
        day: actual_day,
        rerank: args.rerank,
    }))
}

/// Convert day of year (1-365) to (month, day).
fn day_of_year_to_date(day_of_year: u32) -> (u32, u32) {
    const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    let mut remaining = day_of_year;
    for (month_idx, &days_in_month) in DAYS_IN_MONTH.iter().enumerate() {
        if remaining <= days_in_month {
            #[allow(clippy::cast_possible_truncation)]
            return ((month_idx + 1) as u32, remaining);
        }
        remaining -= days_in_month;
    }

    // If we overflow, return Dec 31
    (12, 31)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Day 166
        assert_eq!(day_of_year_to_date(166), (6, 15));
    }
}
