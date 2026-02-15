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

/// Default batch size for inference.
///
/// This is the baseline batch size used for CPU inference across all models.
/// GPU-specific defaults are defined in the `batch_size` module.
///
/// See `determine_default_batch_size()` in `lib.rs` for dynamic batch size selection.
pub const DEFAULT_BATCH_SIZE: usize = 8;

/// Maximum allowed batch size to prevent GPU memory exhaustion.
///
/// This hard limit prevents users from specifying absurdly large batch sizes
/// that would cause GPU memory exhaustion and system hangs. The limit is
/// conservative enough to work on most consumer GPUs while still allowing
/// efficient processing of large files.
///
/// Batch sizes larger than the number of segments in a file are automatically
/// adjusted down at runtime to avoid unnecessary memory allocation and padding.
pub const MAX_BATCH_SIZE: usize = 512;

/// Batch size defaults by execution provider and model type.
pub mod batch_size {
    /// CPU batch size for all models.
    pub const CPU: usize = super::DEFAULT_BATCH_SIZE;

    /// CUDA batch size for `BirdNET` v2.4 and BSG Finland models.
    pub const CUDA_BIRDNET_V24: usize = 64;

    /// CUDA batch size for `BirdNET` v3.0 and Perch v2 models.
    pub const CUDA_BIRDNET_V30: usize = 32;

    /// `TensorRT` batch size for all models.
    pub const TENSORRT: usize = 32;

    /// Conservative default for unknown/other GPU providers.
    pub const OTHER_GPU: usize = 16;
}

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
    /// JSON output extension.
    pub const JSON: &str = ".BirdNET.json";
    /// Parquet output extension.
    pub const PARQUET: &str = ".BirdNET.results.parquet";
}

/// Combined output file names.
pub mod combined_filenames {
    /// Combined CSV filename.
    pub const CSV: &str = "BirdNET_CombinedTable.csv";
    /// Combined Raven filename.
    pub const RAVEN: &str = "BirdNET_SelectionTable.txt";
    /// Combined Kaleidoscope filename.
    pub const KALEIDOSCOPE: &str = "BirdNET_Kaleidoscope.csv";
    /// Combined Parquet filename.
    pub const PARQUET: &str = "BirdNET_CombinedTable.parquet";
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

    /// First day of the year (January 1st) for week-to-day offset calculation.
    pub const YEAR_START_DAY: f32 = 1.0;

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

/// `TensorRT` execution provider constants.
pub mod tensorrt {
    /// Directory name for `TensorRT` engine and timing cache files.
    pub const CACHE_DIR: &str = "tensorrt_cache";
}

/// Clipper constants for clip extraction.
pub mod clipper {
    /// Default pre-padding for clip extraction in seconds.
    pub const DEFAULT_PRE_PADDING: f64 = 5.0;

    /// Default post-padding for clip extraction in seconds.
    pub const DEFAULT_POST_PADDING: f64 = 5.0;

    /// Maximum allowed padding in seconds.
    pub const MAX_PADDING: f64 = 300.0;

    /// Default output directory for clips.
    pub const DEFAULT_OUTPUT_DIR: &str = "clips";

    /// Minimum start time (in seconds) before seeking is attempted.
    /// For clips starting before this threshold, we decode from the beginning.
    pub const SEEK_THRESHOLD_SECS: f64 = 10.0;

    /// `BirdNET` results suffix in detection filenames.
    pub const BIRDNET_RESULTS_SUFFIX: &str = ".BirdNET.results";

    /// `BirdNET` suffix in detection filenames.
    pub const BIRDNET_SUFFIX: &str = ".BirdNET";

    /// Supported audio file extensions for source audio resolution.
    pub const AUDIO_EXTENSIONS: &[&str] = &["wav", "flac", "mp3", "m4a", "aac"];
}
