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

    /// No input files provided.
    #[error("no input files provided")]
    NoInputFiles,

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
}
