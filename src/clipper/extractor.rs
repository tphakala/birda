//! Audio clip extraction.

use std::path::Path;

use super::DetectionGroup;

/// Extracts audio clips from source files.
pub struct ClipExtractor {
    /// Pre-padding in seconds.
    pre_padding: f64,
    /// Post-padding in seconds.
    post_padding: f64,
}

impl ClipExtractor {
    /// Create a new clip extractor with the given padding settings.
    #[must_use]
    pub fn new(pre_padding: f64, post_padding: f64) -> Self {
        Self {
            pre_padding,
            post_padding,
        }
    }

    /// Extract a clip from the source audio file.
    ///
    /// # Errors
    ///
    /// Returns an error if the audio file cannot be read or the clip cannot be extracted.
    #[allow(clippy::todo)]
    pub fn extract_clip(
        &self,
        _source_path: &Path,
        _group: &DetectionGroup,
    ) -> Result<Vec<f32>, crate::Error> {
        let _ = (self.pre_padding, self.post_padding);
        todo!()
    }
}
