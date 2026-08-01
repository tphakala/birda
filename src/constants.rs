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

/// Minimum allowed batch size.
///
/// One segment per inference call. Zero means "run inference on nothing", and
/// it was the only batch size the config-file route rejected before #312: the
/// upper bound lived in `cli::validators::parse_batch_size` alone. Both routes
/// read this constant now, so neither can be tightened without the other.
pub const MIN_BATCH_SIZE: usize = 1;

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

/// Valid range for the day of year used by BSG SDM seasonal adjustment.
///
/// A calendar position rather than an offset, so it is 1-based, and the upper
/// bound is 366 so the last day of a leap year is reachable.
///
/// Read by `cli::validators::parse_day_of_year` and by `config::validate`, for
/// the reason above: the bound used to be an inline `range(1..=366)` on the
/// clap argument, which nothing else could see.
pub mod day_of_year {
    /// First day of the year.
    pub const MIN: u32 = 1;

    /// Last day of a leap year.
    pub const MAX: u32 = 366;
}

/// Geographic coordinate bounds, in degrees.
///
/// Three routes reach one setting: `--lat`/`--lon` and their `BIRDA_LATITUDE`/
/// `BIRDA_LONGITUDE` variables, `birda config set defaults.latitude`, and a
/// hand-edited config.toml. Before #340 each of `cli::validators`,
/// `config::validate::validate_range_filter` and the `Error::InvalidLatitude`
/// message text carried its own copy of these numbers, with nothing keeping
/// them equal. `test_parse_latitude_matches_the_config_file_rule` and its
/// longitude twin drive the flag and the file together and compare verdicts,
/// so neither rule can be changed alone.
pub mod coordinates {
    /// Southernmost latitude.
    pub const LATITUDE_MIN: f64 = -90.0;

    /// Northernmost latitude.
    pub const LATITUDE_MAX: f64 = 90.0;

    /// Westernmost longitude.
    pub const LONGITUDE_MIN: f64 = -180.0;

    /// Easternmost longitude.
    pub const LONGITUDE_MAX: f64 = 180.0;
}

/// Optional metadata columns for the CSV and Parquet writers.
pub mod csv_columns {
    /// Every name `defaults.csv_columns.include` accepts.
    ///
    /// Every writer that consumes `include` matches on these names, and a name
    /// added here needs an arm in each of them. Today that is three arms, and
    /// they disagree about what an unhandled name does: `CsvWriter::
    /// write_detection` leaves the cell empty, so the name becomes a header
    /// over a column empty in every row; `parquet::build_schema` drops it; and
    /// `parquet::build_metadata_column` returns `Error::InvalidColumnName`,
    /// which fails the write outright. The count is deliberately not the point
    /// of this sentence: the invariant survives a fourth writer, a count does
    /// not.
    ///
    /// `config::validate` rejects an unrecognised name, so none of those three
    /// behaviours is reachable from an analyze run. It is not a guarantee for a
    /// library caller: `CsvWriter::new` and `ParquetWriter::new` are public and
    /// take the list directly, which is the same hole `should_process` keeps
    /// its own guard for.
    ///
    /// `test_every_recognised_column_is_written` and
    /// `test_every_recognised_column_reaches_the_parquet_writer` drive both
    /// writers with every name here.
    pub const RECOGNISED: [&str; 8] = [
        "lat",
        "lon",
        "week",
        "model",
        "overlap",
        "sensitivity",
        "min_conf",
        "species_list",
    ];
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
    /// First week of the year, the lower bound for `--week`.
    pub const WEEK_MIN: u32 = 1;

    /// `BirdNET` uses 48 weeks per year.
    ///
    /// Doubles as the upper bound for `--week`, which is why both `--week`
    /// declarations in `cli::args` read this rather than restating 48 (#340).
    /// `utils::date::date_to_week` clamps its result to the same value, so a
    /// separately written CLI bound could have admitted a week that function
    /// never returns.
    pub const WEEKS_PER_YEAR: u32 = 48;

    /// Days per `BirdNET` week (365.25 / 48).
    pub const DAYS_PER_WEEK: f32 = 7.6;

    /// First day of the year (January 1st) for week-to-day offset calculation.
    pub const YEAR_START_DAY: f32 = 1.0;

    /// Default range filter threshold.
    pub const DEFAULT_THRESHOLD: f32 = 0.01;

    /// Version of the `BirdNET` Geomodel birda range filters with.
    ///
    /// Kept in step with the `range_filter.version` field in `registry.json`.
    pub const GEOMODEL_VERSION: &str = "3.0.2";

    /// Number of species the `BirdNET` Geomodel v3.0.2 scores.
    pub const GEOMODEL_SPECIES_COUNT: usize = 12_012;

    /// Threshold used when querying the geomodel itself.
    ///
    /// Zero so the model returns a score for every class. Thresholding and the
    /// unmatched-species policy are applied afterwards, in birda, against the
    /// classifier's own label space.
    pub const GEOMODEL_QUERY_THRESHOLD: f32 = 0.0;
}

/// Download constants.
pub mod download {
    /// Suffix for in-progress downloads.
    ///
    /// The file is renamed onto the real destination only once the transfer
    /// completes, so a concurrent or interrupted download can never leave a
    /// truncated destination file behind.
    pub const PARTIAL_SUFFIX: &str = "part";

    /// Connection timeout for registry downloads, in seconds.
    pub const CONNECT_TIMEOUT_SECS: u64 = 30;

    /// Total request timeout for registry downloads, in minutes.
    pub const REQUEST_TIMEOUT_MINS: u64 = 5;

    /// Timeout for the connectivity probe made before offering a download.
    ///
    /// Short, because its only job is to tell "offline" apart from "slow".
    pub const CONNECTIVITY_PROBE_TIMEOUT_SECS: u64 = 5;

    /// Canonical Hugging Face origin, the prefix a mirror endpoint replaces.
    pub const HUGGING_FACE_ENDPOINT: &str = "https://huggingface.co";

    /// Environment variable naming a Hugging Face mirror.
    ///
    /// Named `HF_ENDPOINT` because that is what the Hugging Face client
    /// libraries already read, so a user who has set it for `huggingface-cli`
    /// does not have to learn a birda-specific spelling.
    pub const HF_ENDPOINT_ENV: &str = "HF_ENDPOINT";
}

/// Files left behind by earlier birda versions that are no longer used.
pub mod obsolete_files {
    /// The `BirdNET` v2.4 meta model, replaced by the `BirdNET` Geomodel v3.0.2.
    pub const NAMES: &[&str] = &["birdnet-v24-meta.onnx"];
}

/// Calendar constants.
pub mod calendar {
    /// Days in each month (non-leap year).
    pub const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    /// First month of the year, the lower bound for `--month`.
    pub const MONTH_MIN: u32 = 1;

    /// Last month of the year, the upper bound for `--month`.
    pub const MONTH_MAX: u32 = 12;

    /// First day of any month, the lower bound for `--day`.
    pub const DAY_MIN: u32 = 1;

    /// Last day of the longest month, the upper bound for `--day`.
    ///
    /// `--day` is not checked against the month chosen alongside it, so the
    /// bound is the longest month rather than the selected one.
    pub const DAY_MAX: u32 = 31;

    // Pin both upper bounds to the table they describe. A month added to or
    // removed from `DAYS_IN_MONTH`, or a change to its longest entry, fails the
    // build here rather than leaving `--month` and `--day` bounded by numbers
    // the calendar they came from no longer supports.
    const _: () = {
        let mut months: u32 = 0;
        let mut longest: u32 = 0;
        let mut index = 0;
        while index < DAYS_IN_MONTH.len() {
            if DAYS_IN_MONTH[index] > longest {
                longest = DAYS_IN_MONTH[index];
            }
            months += 1;
            index += 1;
        }
        assert!(months == MONTH_MAX);
        assert!(longest == DAY_MAX);
    };
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

    /// Upper bound, in seconds of audio, on what an extracted clip reserves up
    /// front.
    ///
    /// Denominated in seconds rather than samples on purpose: the extractor
    /// multiplies by the file's own sample rate. A flat sample count would
    /// mean 60 seconds at 48 kHz but only 11.25 at `bat::SAMPLE_RATE`, which
    /// is short enough that the default bat clip (a 3-second detection plus
    /// the default padding) would exceed it before anything unusual happened.
    ///
    /// The reservation is only a sizing hint: the decode loop pushes the
    /// samples it actually finds and the buffer grows on demand, so a longer
    /// clip costs geometric reallocations, the last of which holds both
    /// buffers and so peaks around 1.5 times the final size. Below the cap
    /// that costs nothing at all, and above it a modest fraction of the
    /// extraction.
    ///
    /// The cap exists because the requested range is caller-supplied and its
    /// length is not otherwise bounded. `--end 1e12` is finite, non-negative
    /// and passes every range check, yet at 48 kHz it asks for 4.8e16 samples,
    /// and a reservation that large aborts the process rather than returning
    /// an error.
    ///
    /// Clips longer than this exist and are fine: `MAX_PADDING` alone allows a
    /// 603-second clip around a 3-second detection, and `group_detections`
    /// merges overlapping padded ranges without limit, so a species calling
    /// through a dawn recording routinely produces one much longer group.
    pub const MAX_CLIP_PREALLOC_SECS: usize = 60;

    /// Absolute ceiling on an extracted clip's reservation, whatever the file
    /// claims its sample rate to be.
    ///
    /// `MAX_CLIP_PREALLOC_SECS` is scaled by the rate read out of the
    /// container, and that rate is not validated anywhere: a hand-built WAV
    /// can declare `u32::MAX`, which scales the ceiling to 257 billion samples
    /// and asks for a terabyte. That is the reservation #310 was filed about,
    /// reintroduced through the mechanism added to prevent it, so the
    /// rate-scaled term needs a term beside it that no input can move.
    ///
    /// 60 seconds at the highest rate the tool actually handles, so it binds
    /// only when a file is lying.
    pub const MAX_CLIP_PREALLOC_SAMPLES: usize =
        MAX_CLIP_PREALLOC_SECS * super::bat::SAMPLE_RATE as usize;

    /// Malformed detection rows reported individually before the rest are
    /// summarised.
    ///
    /// The row count is caller-supplied, so a file of nothing but malformed
    /// rows would otherwise emit one formatted warning per row, which costs
    /// several times what parsing the rows costs.
    pub const MAX_SKIPPED_ROW_WARNINGS: usize = 10;
}

/// Bat detection constants.
pub mod bat {
    /// Audio sample rate for bat recordings (256 kHz).
    pub const SAMPLE_RATE: u32 = 256_000;

    /// Number of audio samples per segment.
    /// Equals `BirdNET` v2.4's 144,000 samples; this is the "slow-down trick".
    pub const CHUNK_SAMPLES: usize = 144_000;

    /// Segment duration in seconds, derived from chunk samples and sample rate.
    #[allow(clippy::cast_precision_loss)]
    pub const SEGMENT_DURATION: f32 = CHUNK_SAMPLES as f32 / SAMPLE_RATE as f32;

    /// Fraction of segment duration used as overlap between segments.
    pub const OVERLAP_FRACTION: f32 = 0.25;

    /// Overlap between segments in seconds.
    pub const OVERLAP: f32 = SEGMENT_DURATION * OVERLAP_FRACTION;
}

/// ONNX Runtime discovery constants.
pub mod onnx_runtime {
    /// Environment variable used to override the ONNX Runtime dynamic library path.
    pub const DYLIB_PATH_ENV: &str = "ORT_DYLIB_PATH";

    #[cfg(target_os = "linux")]
    /// Name of the ONNX Runtime shared library on Linux.
    pub const LIBRARY_FILE_NAME: &str = "libonnxruntime.so";

    #[cfg(target_os = "macos")]
    /// Name of the ONNX Runtime shared library on macOS.
    pub const LIBRARY_FILE_NAME: &str = "libonnxruntime.dylib";

    #[cfg(target_os = "windows")]
    /// Name of the ONNX Runtime shared library on Windows.
    pub const LIBRARY_FILE_NAME: &str = "onnxruntime.dll";

    #[cfg(target_os = "linux")]
    /// Search path environment variable used by the dynamic linker on Linux.
    pub const SEARCH_PATH_ENV: &str = "LD_LIBRARY_PATH";

    #[cfg(target_os = "macos")]
    /// Search path environment variable used by the dynamic linker on macOS.
    pub const SEARCH_PATH_ENV: &str = "DYLD_LIBRARY_PATH";

    #[cfg(target_os = "windows")]
    /// Search path environment variable used by the dynamic linker on Windows.
    pub const SEARCH_PATH_ENV: &str = "PATH";

    #[cfg(target_os = "linux")]
    /// Common Linux library directories to probe before letting ORT fall back to loader defaults.
    pub const COMMON_SEARCH_DIRS: &[&str] = &[
        "/usr/lib",
        "/usr/local/lib",
        "/lib",
        "/lib64",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ];

    #[cfg(target_os = "macos")]
    /// Common macOS library directories to probe before letting ORT fall back to loader defaults.
    pub const COMMON_SEARCH_DIRS: &[&str] = &["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib"];

    #[cfg(target_os = "windows")]
    /// Common Windows library directories to probe before letting ORT fall back to loader defaults.
    pub const COMMON_SEARCH_DIRS: &[&str] = &["C:\\Windows\\System32", "C:\\Windows", "."];
}
