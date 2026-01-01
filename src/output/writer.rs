//! Output writer trait definition.

use crate::error::Result;
use crate::output::Detection;

/// Trait for writing detection results.
pub trait OutputWriter {
    /// Write the file header (if applicable).
    fn write_header(&mut self) -> Result<()>;

    /// Write a single detection.
    fn write_detection(&mut self, detection: &Detection) -> Result<()>;

    /// Finalize the output (flush, close, etc.).
    fn finalize(&mut self) -> Result<()>;
}
