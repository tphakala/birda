//! Detection file parsing.

/// A detection parsed from a results file.
#[derive(Debug, Clone)]
pub struct ParsedDetection {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Scientific name of the species.
    pub scientific_name: String,
    /// Common name of the species.
    pub common_name: String,
    /// Detection confidence (0.0-1.0).
    pub confidence: f32,
}

/// Parse a detection file and return detections.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
#[allow(clippy::todo)]
pub fn parse_detection_file(_path: &std::path::Path) -> Result<Vec<ParsedDetection>, crate::Error> {
    todo!()
}
