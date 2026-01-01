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
}
