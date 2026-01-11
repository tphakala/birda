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

    /// Range filtering requires meta model.
    #[error(
        "range filtering requires meta model (model {model_name} has no meta model configured)"
    )]
    MetaModelMissing {
        /// Name of the model.
        model_name: String,
    },

    /// Meta model file not found.
    #[error("meta model file not found: {path}")]
    MetaModelNotFound {
        /// Path to the missing meta model file.
        path: std::path::PathBuf,
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

    /// Failed to read species list file.
    #[error("failed to read species list file '{path}'")]
    SpeciesListRead {
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
}
