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
}
