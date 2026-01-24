//! Date conversion utilities for range filtering.

use crate::constants::calendar::DAYS_IN_MONTH;
use crate::constants::range_filter::{DAYS_PER_WEEK, WEEKS_PER_YEAR, YEAR_START_DAY};

/// Convert month/day to week number (1-48).
///
/// `BirdNET` uses 48 weeks per year, approximately 7.6 days per year.
/// `Week = floor((day_of_year - 1) / 7.6) + 1`
///
/// # Limitations
///
/// - Assumes non-leap years (February = 28 days). For leap years, calculations
///   after February will be off by 1 day, resulting in ~0.13 week error.
///   This is acceptable given `BirdNET`'s approximate 48-week system.
/// - Does not validate month/day combinations (e.g., Feb 31 will produce
///   incorrect results).
pub fn date_to_week(month: u32, day: u32) -> u32 {
    let day_of_year: u32 = DAYS_IN_MONTH.iter().take((month - 1) as usize).sum::<u32>() + day;

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let week = ((day_of_year - 1) as f32 / DAYS_PER_WEEK).floor() as u32 + 1;

    week.min(WEEKS_PER_YEAR)
}

/// Convert day of year (1-365) to (month, day).
pub fn day_of_year_to_date(day_of_year: u32) -> (u32, u32) {
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

/// Convert a `BirdNET` week number (1-48) to the starting day of that week.
///
/// `BirdNET` uses 48 weeks of ~7.6 days each. Week 1 starts on day 1 (Jan 1).
///
/// # Formula
///
/// `day_of_year = (week - 1) * DAYS_PER_WEEK + YEAR_START_DAY`
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn week_to_start_day(week: u32) -> u32 {
    ((week - 1) as f32).mul_add(DAYS_PER_WEEK, YEAR_START_DAY) as u32
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

    #[test]
    fn test_week_to_start_day_week_1() {
        // Week 1 starts on day 1 (Jan 1)
        assert_eq!(week_to_start_day(1), 1);
    }

    #[test]
    fn test_week_to_start_day_week_24() {
        // Week 24: (24-1) * 7.6 + 1 = 175.8 -> 175
        assert_eq!(week_to_start_day(24), 175);
    }

    #[test]
    fn test_week_to_start_day_week_48() {
        // Week 48: (48-1) * 7.6 + 1 = 358.2 -> 358
        assert_eq!(week_to_start_day(48), 358);
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
        // Day 166
        assert_eq!(day_of_year_to_date(166), (6, 15));
    }

    #[test]
    fn test_day_of_year_to_date_overflow() {
        // Day 400 should return Dec 31 (overflow protection)
        assert_eq!(day_of_year_to_date(400), (12, 31));
    }
}
