//! Error types for birda.

/// Result type alias for birda operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error type for birda.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// I/O operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration directory could not be determined.
    #[error("could not determine configuration directory for this platform")]
    ConfigDirNotFound,

    /// Cache directory could not be determined.
    #[error("could not determine cache directory for this platform")]
    CacheDirNotFound,

    /// Failed to read configuration file.
    #[error("failed to read config file '{path}'")]
    ConfigRead {
        /// Path to the config file.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to parse configuration file.
    #[error("failed to parse config file '{path}'")]
    ConfigParse {
        /// Path to the config file.
        path: std::path::PathBuf,
        /// Underlying parse error.
        #[source]
        source: toml::de::Error,
    },

    /// Configuration validation failed.
    #[error("configuration validation failed: {message}")]
    ConfigValidation {
        /// Description of the validation failure.
        message: String,
    },

    /// Invalid configuration key.
    #[error("unknown configuration key: '{key}'")]
    InvalidConfigKey {
        /// The invalid key.
        key: String,
    },

    /// Model not found in configuration.
    #[error("model '{name}' not found in configuration")]
    ModelNotFound {
        /// Name of the missing model.
        name: String,
    },

    /// Model file does not exist.
    #[error("model file does not exist: {path}")]
    ModelFileNotFound {
        /// Path to the missing model file.
        path: std::path::PathBuf,
    },

    /// Labels file does not exist.
    #[error("labels file does not exist: {path}")]
    LabelsFileNotFound {
        /// Path to the missing labels file.
        path: std::path::PathBuf,
    },

    /// Model already exists in configuration.
    #[error("model '{name}' already exists in configuration")]
    ModelAlreadyExists {
        /// Name of the existing model.
        name: String,
    },

    /// Failed to write configuration file.
    #[error("failed to write config file '{path}'")]
    ConfigWrite {
        /// Path to the config file.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to serialize configuration.
    #[error("failed to serialize config")]
    ConfigSerialize {
        /// Underlying serialization error.
        #[source]
        source: toml::ser::Error,
    },

    /// No valid audio files found.
    #[error("no valid audio files found in the provided paths")]
    NoValidAudioFiles,

    /// Output path escapes the intended output directory (path traversal attempt).
    #[error("output path '{output_path}' escapes output directory '{output_dir}'")]
    PathTraversal {
        /// The generated output path.
        output_path: std::path::PathBuf,
        /// The intended output directory.
        output_dir: std::path::PathBuf,
    },

    /// Failed to open audio file.
    #[error("failed to open audio file '{path}'")]
    AudioOpen {
        /// Path to the audio file.
        path: std::path::PathBuf,
        /// Underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Unsupported audio format.
    #[error("unsupported audio format: {format}")]
    UnsupportedAudioFormat {
        /// The unsupported format.
        format: String,
    },

    /// Failed to decode audio.
    #[error("failed to decode audio from '{path}'")]
    AudioDecode {
        /// Path to the audio file.
        path: std::path::PathBuf,
        /// Underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// No audio tracks found.
    #[error("no audio tracks found in '{path}'")]
    NoAudioTracks {
        /// Path to the audio file.
        path: std::path::PathBuf,
    },

    /// Failed to resample audio.
    #[error("failed to resample audio: {reason}")]
    Resample {
        /// Description of the resampling failure.
        reason: String,
    },

    /// Failed to acquire lock.
    #[error("file is locked by another process: {path}")]
    FileLocked {
        /// Path to the lock file.
        path: std::path::PathBuf,
    },

    /// Failed to create lock file.
    #[error("failed to create lock file '{path}'")]
    LockCreate {
        /// Path to the lock file.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to remove lock file.
    #[error("failed to remove lock file '{path}'")]
    LockRemove {
        /// Path to the lock file.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to initialize ONNX runtime.
    #[error("failed to initialize ONNX runtime: {reason}")]
    RuntimeInitialization {
        /// Description of the initialization failure.
        reason: String,
    },

    /// Failed to build classifier.
    #[error("failed to build classifier: {reason}")]
    ClassifierBuild {
        /// Description of the build failure.
        reason: String,
    },

    /// Inference failed.
    #[error("inference failed: {reason}")]
    Inference {
        /// Description of the inference failure.
        reason: String,
    },

    /// Failed to read registry file.
    #[error("failed to read registry file '{path}'")]
    RegistryRead {
        /// Path to the registry file.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to parse registry file.
    #[error("failed to parse registry file '{path}'")]
    RegistryParse {
        /// Path to the registry file.
        path: std::path::PathBuf,
        /// Underlying parse error.
        #[source]
        source: serde_json::Error,
    },

    /// Failed to serialize registry.
    #[error("failed to serialize registry")]
    RegistrySerialize {
        /// Underlying serialization error.
        #[source]
        source: serde_json::Error,
    },

    /// Failed to move a completed download onto its destination.
    #[error("failed to install downloaded file '{dest}'")]
    DownloadInstallFailed {
        /// Path the download was being renamed onto.
        dest: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to write registry file.
    #[error("failed to write registry file '{path}'")]
    RegistryWrite {
        /// Path to the registry file.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Model not found in registry.
    #[error("model '{id}' not found in registry")]
    ModelNotFoundInRegistry {
        /// ID of the missing model.
        id: String,
    },

    /// Model publishes no translated label sets.
    #[error("model '{model_id}' has no label language variants")]
    ModelHasNoLanguages {
        /// Registry id of the model.
        model_id: String,
    },

    /// Region not published for this model.
    #[error("model '{model_id}' has no region '{region}'. Available: {available}")]
    RegionNotFound {
        /// Registry id of the model.
        model_id: String,
        /// Region slug the user asked for.
        region: String,
        /// Comma-separated list of valid region slugs.
        available: String,
    },

    /// Variant not published for this model and region.
    #[error("model '{model_id}' has no variant '{variant}'. Available: {available}")]
    VariantNotFound {
        /// Registry id of the model.
        model_id: String,
        /// Variant id the user asked for.
        variant: String,
        /// Comma-separated list of valid variant ids.
        available: String,
    },

    /// Model publishes no regional variants.
    #[error("model '{model_id}' has no regional variants")]
    RegionsNotSupported {
        /// Registry id of the model.
        model_id: String,
    },

    /// Language not available for model.
    #[error("language '{code}' not available for model '{model_id}'")]
    LanguageNotFound {
        /// Language code.
        code: String,
        /// Model ID.
        model_id: String,
    },

    /// Download failed.
    #[error("failed to download from '{url}'")]
    DownloadFailed {
        /// URL that failed.
        url: String,
        /// Underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Invalid model type string.
    #[error("invalid model type: {value}")]
    InvalidModelType {
        /// Invalid value.
        value: String,
    },

    /// Internal error (for unexpected failures).
    #[error("internal error: {message}")]
    Internal {
        /// Error message.
        message: String,
    },

    /// Decode thread channel was closed unexpectedly.
    #[error("decode channel closed unexpectedly")]
    DecodeChannelClosed,

    /// Failed to build range filter.
    #[error("failed to build range filter: {reason}")]
    RangeFilterBuild {
        /// Description of the build failure.
        reason: String,
    },

    /// Failed to predict location scores.
    #[error("failed to predict location scores: {reason}")]
    RangeFilterPredict {
        /// Description of the prediction failure.
        reason: String,
    },

    /// Registry does not describe a range filter asset.
    #[error("registry does not describe a range filter asset; update birda to a newer version")]
    RangeFilterAssetMissing,

    /// The geomodel is not installed and could not be acquired.
    #[error("BirdNET Geomodel v3.0.2 is not installed: {hint}")]
    GeomodelNotInstalled {
        /// What the user should do next.
        hint: String,
    },

    /// Only one of the two geomodel paths was supplied.
    #[error(
        "geomodel path and geomodel labels path must be given together (received only {given})"
    )]
    GeomodelPathsIncomplete {
        /// Which of the two paths was supplied.
        given: String,
    },

    /// The geomodel labels file does not match the model's output size.
    #[error(
        "BirdNET Geomodel v3.0.2 labels file has {actual} labels, expected {expected}; \
         reinstall with 'birda models install geomodel'"
    )]
    GeomodelLabelCount {
        /// Number of labels the registry declares.
        expected: usize,
        /// Number of labels actually read from the file.
        actual: usize,
    },

    /// No network connectivity when a download was requested.
    #[error("no network connectivity to {host}; run 'birda models install geomodel' when online")]
    NoNetworkConnectivity {
        /// Host that could not be reached.
        host: String,
    },

    /// Invalid latitude value.
    #[error("invalid latitude: {value} (must be -90.0 to 90.0)")]
    InvalidLatitude {
        /// Invalid latitude value.
        value: f64,
    },

    /// Invalid longitude value.
    #[error("invalid longitude: {value} (must be -180.0 to 180.0)")]
    InvalidLongitude {
        /// Invalid longitude value.
        value: f64,
    },

    /// Invalid range filter threshold.
    #[error("invalid range threshold: {value} (must be 0.0 to 1.0)")]
    InvalidRangeThreshold {
        /// Invalid threshold value.
        value: f32,
    },

    /// Failed to read species list file.
    #[error("failed to read species list file '{path}'")]
    SpeciesListRead {
        /// Path to the species list file.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to write a species list file.
    #[error("failed to write species list '{path}'")]
    SpeciesListWrite {
        /// Path to the species list file.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    // Clipper errors
    /// Failed to parse detection file.
    #[error("failed to parse detection file '{path}'")]
    DetectionParseFailed {
        /// Path to the detection file.
        path: std::path::PathBuf,
        /// Underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Invalid detection file format.
    #[error("invalid detection file format: {message}")]
    InvalidDetectionFormat {
        /// Description of the format error.
        message: String,
    },

    /// Failed to write WAV file.
    #[error("failed to write WAV file '{path}'")]
    WavWriteFailed {
        /// Path to the WAV file.
        path: std::path::PathBuf,
        /// Underlying error.
        #[source]
        source: hound::Error,
    },

    /// Failed to create output directory.
    #[error("failed to create output directory '{path}'")]
    OutputDirCreateFailed {
        /// Path to the output directory.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Source audio file not found for detection file.
    #[error(
        "source audio file not found for detection file '{detection_path}', expected '{audio_path}'"
    )]
    SourceAudioNotFound {
        /// Path to the detection file.
        detection_path: std::path::PathBuf,
        /// Expected path to the audio file.
        audio_path: std::path::PathBuf,
    },

    /// Failed to write JSON output file.
    #[error("failed to write JSON output file '{path}'")]
    JsonWrite {
        /// Path to the JSON file.
        path: std::path::PathBuf,
        /// Underlying serialization error.
        #[source]
        source: serde_json::Error,
    },

    /// Failed to flush a JSON output file to disk.
    ///
    /// Separate from [`Self::JsonWrite`] because that one carries a serialization
    /// error, and the final flush fails with an I/O error instead. It is the
    /// failure that used to be discarded entirely.
    #[error("failed to flush JSON output file '{path}'")]
    JsonFlush {
        /// Path to the JSON file.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Invalid output format string.
    #[error("invalid output format: {value}")]
    InvalidOutputFormat {
        /// The invalid value.
        value: String,
    },

    /// Invalid time range for clip extraction.
    ///
    /// Covers a non-finite bound as well as an out-of-order one, because the
    /// ordering test alone never rejects NaN.
    // Unit outside the parentheses, as in `InvalidPadding`: a NaN bound would
    // otherwise render as `NaNs`.
    #[error(
        "invalid time range: start {start}, end {end} (both must be finite non-negative seconds, with end greater than start)"
    )]
    InvalidTimeRange {
        /// Start time in seconds.
        start: f64,
        /// End time in seconds.
        end: f64,
    },

    /// Invalid clip padding value.
    // No unit suffix directly after `{value}`: this variant exists to report
    // non-finite values, and `NaNs` reads as a plural rather than a quantity.
    // The parenthesised-constraint shape matches `InvalidLatitude` and its
    // siblings.
    #[error(
        "invalid padding: {value} (must be a finite number of seconds from 0.0 to {:.1})",
        crate::constants::clipper::MAX_PADDING
    )]
    InvalidPadding {
        /// The rejected padding value in seconds.
        value: f64,
    },

    /// Invalid clip confidence threshold.
    #[error(
        "invalid confidence: {value} (must be a finite number from {:.1} to {:.1})",
        crate::constants::confidence::MIN,
        crate::constants::confidence::MAX
    )]
    InvalidConfidence {
        /// The rejected confidence value.
        value: f32,
    },

    /// BSG model configuration error.
    #[error("BSG configuration error: {message}")]
    BsgConfig {
        /// Error message.
        message: String,
    },

    /// BSG calibration file missing or invalid.
    #[error("BSG calibration file error: {0}")]
    BsgCalibration(String),

    /// BSG migration file missing or invalid.
    #[error("BSG migration file error: {0}")]
    BsgMigration(String),

    /// BSG distribution maps file missing or invalid.
    #[error("BSG distribution maps file error: {0}")]
    BsgDistributionMaps(String),

    /// Failed to delete a model file from disk.
    #[error("failed to delete file '{path}'")]
    FileDeletionFailed {
        /// Path to the file that couldn't be deleted.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Day of year auto-detection failed.
    #[error("could not auto-detect day of year from file {path}: {reason}")]
    DayOfYearAutoDetect {
        /// Path to the file.
        path: std::path::PathBuf,
        /// Reason for failure.
        reason: String,
    },

    /// Failed to create Parquet file.
    #[error("failed to create Parquet file '{path}'")]
    ParquetFileCreate {
        /// Path to the Parquet file.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to open Parquet file for reading.
    #[error("failed to open Parquet file '{path}'")]
    ParquetFileOpen {
        /// Path to the Parquet file.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to write Parquet data.
    #[error("Parquet write error: {context}")]
    ParquetWrite {
        /// Context of the write operation.
        context: String,
        /// Underlying Parquet error.
        #[source]
        source: parquet::errors::ParquetError,
    },

    /// Invalid column name in Parquet schema.
    #[error("invalid Parquet column name: {name}")]
    InvalidColumnName {
        /// The invalid column name.
        name: String,
    },

    /// No input files provided for combining operation.
    #[error("no input files were provided to combine")]
    NoInputFilesToCombine,

    /// Failed to load labels from file.
    #[error("failed to load labels from {path}: {reason}")]
    LabelLoad {
        /// Path to the labels file.
        path: String,
        /// Description of the failure.
        reason: String,
    },

    /// Failed to fetch update manifest from GitHub.
    #[error("failed to fetch update manifest: {reason}")]
    UpdateFetchFailed {
        /// Description of the failure.
        reason: String,
    },

    /// Update manifest JSON was malformed.
    #[error("failed to parse update manifest")]
    UpdateManifestParse {
        /// Underlying parse error.
        #[source]
        source: serde_json::Error,
    },

    /// SHA256 checksum mismatch after download.
    #[error("checksum mismatch for '{file}': expected {expected}, got {actual}")]
    UpdateChecksumMismatch {
        /// File name that failed verification.
        file: String,
        /// Expected SHA256 hash.
        expected: String,
        /// Actual SHA256 hash.
        actual: String,
    },

    /// Binary replacement failed.
    #[error("failed to replace binary: {reason}")]
    UpdateReplaceFailed {
        /// Description of the failure.
        reason: String,
    },

    /// Update blocked due to ONNX Runtime ABI incompatibility.
    #[error(
        "update blocked: ONNX Runtime version changed ({current} -> {required}), binary-only update would break birda\nPlease download the full package from: {release_url}"
    )]
    UpdateBlocked {
        /// Current ONNX Runtime version.
        current: String,
        /// Required ONNX Runtime version.
        required: String,
        /// URL to the full release for manual download.
        release_url: String,
    },

    /// No write permission to the binary's parent directory.
    #[error("no write permission to '{path}', try running with elevated privileges")]
    UpdatePermissionDenied {
        /// Path that lacks write permission.
        path: std::path::PathBuf,
    },

    /// No matching update asset for this platform/variant.
    #[error("no update available for platform '{platform}'")]
    UpdateUnsupportedPlatform {
        /// Platform key that wasn't found in manifest.
        platform: String,
    },

    /// Archive extraction failed.
    #[error("failed to extract update archive: {reason}")]
    UpdateExtractFailed {
        /// Description of the failure.
        reason: String,
    },

    /// Update refused because binary is a dev build (in target/ directory).
    #[error("refusing to update a development build (binary is in a cargo target/ directory)")]
    UpdateDevBuild,

    /// Failed to determine the current executable path.
    #[error("failed to determine current executable path")]
    UpdateExeNotFound {
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_output_format_error_display() {
        let err = Error::InvalidOutputFormat {
            value: "foobar".to_string(),
        };
        assert_eq!(err.to_string(), "invalid output format: foobar");
    }

    /// The three #310 messages, asserted on their rendered text rather than
    /// their variant. The rest of the suite matches on the variant, so the
    /// wording was free to drift, and the wording is the whole user-visible
    /// half of the fix. Each case is a non-finite value on purpose: that is
    /// the population these variants exist to report, and it is where a unit
    /// suffix placed straight after the value renders as `NaNs`.
    #[test]
    fn test_invalid_time_range_renders_both_bounds() {
        let err = Error::InvalidTimeRange {
            start: f64::NAN,
            end: 5.0,
        };
        assert_eq!(
            err.to_string(),
            "invalid time range: start NaN, end 5 \
             (both must be finite non-negative seconds, with end greater than start)"
        );
    }

    #[test]
    fn test_invalid_padding_renders_the_value_and_the_ceiling() {
        let err = Error::InvalidPadding {
            value: f64::INFINITY,
        };
        assert_eq!(
            err.to_string(),
            "invalid padding: inf (must be a finite number of seconds from 0.0 to 300.0)"
        );
    }

    #[test]
    fn test_invalid_confidence_renders_the_value_and_the_range() {
        let err = Error::InvalidConfidence { value: f32::NAN };
        assert_eq!(
            err.to_string(),
            "invalid confidence: NaN (must be a finite number from 0.0 to 1.0)"
        );
    }
}
