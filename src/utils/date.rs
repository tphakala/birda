//! Date conversion utilities for range filtering.

use crate::constants::range_filter::{DAYS_PER_WEEK, WEEKS_PER_YEAR};

/// Convert month/day to week number (1-48).
///
/// `BirdNET` uses 48 weeks per year, approximately 7.6 days per week.
/// `Week = floor((day_of_year - 1) / 7.6) + 1`
pub fn date_to_week(month: u32, day: u32) -> u32 {
    const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    let day_of_year: u32 = DAYS_IN_MONTH.iter().take((month - 1) as usize).sum::<u32>() + day;

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let week = ((day_of_year - 1) as f32 / DAYS_PER_WEEK).floor() as u32 + 1;

    week.min(WEEKS_PER_YEAR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_to_week_jan_1() {
        assert_eq!(date_to_week(1, 1), 1);
    }

    #[test]
    fn test_date_to_week_dec_31() {
        assert_eq!(date_to_week(12, 31), 48);
    }

    #[test]
    fn test_date_to_week_jun_15() {
        // June 15 is day 166 of year
        // (166 - 1) / 7.6 = 21.71 -> floor = 21, + 1 = 22
        assert_eq!(date_to_week(6, 15), 22);
    }

    #[test]
    fn test_date_to_week_mid_year() {
        // July 1 is day 182
        // (182 - 1) / 7.6 = 23.81 -> floor = 23, + 1 = 24
        assert_eq!(date_to_week(7, 1), 24);
    }
}
