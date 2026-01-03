//! Application-wide constants.
//!
//! All magic numbers and strings are defined here to ensure consistency
//! and make changes easy to track.

/// Application name used for config directories and user-facing messages.
pub const APP_NAME: &str = "birda";

/// Default minimum confidence threshold for detections.
pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.1;

/// Default segment overlap in seconds.
pub const DEFAULT_OVERLAP: f32 = 0.0;

/// Default batch size (1 = no batching).
pub const DEFAULT_BATCH_SIZE: usize = 1;

/// Default number of top predictions to return per segment.
pub const DEFAULT_TOP_K: usize = 5;

/// Lock file extension.
pub const LOCK_FILE_EXTENSION: &str = ".birda.lock";

/// Output file extensions by format.
pub mod output_extensions {
    /// CSV output extension.
    pub const CSV: &str = ".BirdNET.results.csv";
    /// Raven selection table extension.
    pub const RAVEN: &str = ".BirdNET.selection.table.txt";
    /// Audacity labels extension.
    pub const AUDACITY: &str = ".BirdNET.results.txt";
    /// Kaleidoscope CSV extension.
    pub const KALEIDOSCOPE: &str = ".BirdNET.results.kaleidoscope.csv";
}

/// Combined output file names.
pub mod combined_filenames {
    /// Combined CSV filename.
    pub const CSV: &str = "BirdNET_CombinedTable.csv";
    /// Combined Raven filename.
    pub const RAVEN: &str = "BirdNET_SelectionTable.txt";
    /// Combined Kaleidoscope filename.
    pub const KALEIDOSCOPE: &str = "BirdNET_Kaleidoscope.csv";
}

/// Confidence value bounds.
pub mod confidence {
    /// Minimum valid confidence value.
    pub const MIN: f32 = 0.0;
    /// Maximum valid confidence value.
    pub const MAX: f32 = 1.0;
    /// Decimal places for confidence formatting.
    pub const DECIMAL_PLACES: usize = 4;
}

/// Raven format constants.
pub mod raven {
    /// View column value.
    pub const VIEW: &str = "Spectrogram 1";
    /// Channel column value.
    pub const CHANNEL: u8 = 1;
    /// Default low frequency bound in Hz.
    pub const DEFAULT_LOW_FREQ: u32 = 150;
    /// Default high frequency bound in Hz.
    pub const DEFAULT_HIGH_FREQ: u32 = 15000;
}

/// Range filter constants.
pub mod range_filter {
    /// `BirdNET` uses 48 weeks per year.
    pub const WEEKS_PER_YEAR: u32 = 48;

    /// Days per `BirdNET` week (365.25 / 48).
    pub const DAYS_PER_WEEK: f32 = 7.6;

    /// Default range filter threshold.
    pub const DEFAULT_THRESHOLD: f32 = 0.01;
}

/// Calendar constants.
pub mod calendar {
    /// Days in each month (non-leap year).
    pub const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
}

/// UTF-8 Byte Order Mark for Excel compatibility in CSV files.
pub const UTF8_BOM: &[u8; 3] = b"\xEF\xBB\xBF";
