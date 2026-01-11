//! Detection file parsing.
//!
//! Parses birda CSV detection files to extract detection information
//! for clip extraction. Uses the `csv` crate for robust parsing.

use std::path::Path;

use serde::Deserialize;

use crate::Error;

/// Internal record for CSV deserialization.
#[derive(Debug, Deserialize)]
struct DetectionRecord {
    #[serde(rename = "Start (s)")]
    start: f64,
    #[serde(rename = "End (s)")]
    end: f64,
    #[serde(rename = "Scientific name")]
    scientific_name: String,
    #[serde(rename = "Common name")]
    common_name: String,
    #[serde(rename = "Confidence")]
    confidence: f32,
}

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
/// Supports birda CSV format with columns:
/// - Start (s), End (s), Scientific name, Common name, Confidence
///
/// Handles UTF-8 BOM if present, quoted fields with embedded commas,
/// and escaped quotes within fields.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - Required columns are missing
/// - Values cannot be parsed
///
/// Returns `Ok(vec![])` if the file contains no detections (empty or header-only).
pub fn parse_detection_file(path: &Path) -> Result<Vec<ParsedDetection>, Error> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_path(path)
        .map_err(|e| Error::DetectionParseFailed {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?;

    let mut detections = Vec::new();

    for (line_num, result) in reader.deserialize::<DetectionRecord>().enumerate() {
        let record = result.map_err(|e| Error::InvalidDetectionFormat {
            message: format!("line {}: {e}", line_num + 2),
        })?;

        // Validate time range
        if record.end <= record.start {
            return Err(Error::InvalidDetectionFormat {
                message: format!(
                    "line {}: end time ({}) must be greater than start time ({})",
                    line_num + 2,
                    record.end,
                    record.start
                ),
            });
        }

        detections.push(ParsedDetection {
            start: record.start,
            end: record.end,
            scientific_name: record.scientific_name,
            common_name: record.common_name,
            confidence: record.confidence,
        });
    }

    Ok(detections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_simple_csv() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "Start (s),End (s),Scientific name,Common name,Confidence"
        )
        .unwrap();
        writeln!(file, "0.0,3.0,Turdus merula,Eurasian Blackbird,0.85").unwrap();
        writeln!(file, "5.0,8.0,Parus major,Great Tit,0.92").unwrap();
        file.flush().unwrap();

        let detections = parse_detection_file(file.path()).unwrap();
        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].scientific_name, "Turdus merula");
        assert!((detections[0].confidence - 0.85).abs() < 0.001);
        assert_eq!(detections[1].scientific_name, "Parus major");
    }

    #[test]
    fn test_parse_quoted_fields_with_commas() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "Start (s),End (s),Scientific name,Common name,Confidence"
        )
        .unwrap();
        writeln!(file, "1.0,4.0,Tyto alba,\"Owl, Barn\",0.78").unwrap();
        file.flush().unwrap();

        let detections = parse_detection_file(file.path()).unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].common_name, "Owl, Barn");
    }

    #[test]
    fn test_parse_escaped_quotes() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "Start (s),End (s),Scientific name,Common name,Confidence"
        )
        .unwrap();
        // CSV escaped quotes: "" becomes "
        writeln!(file, "2.0,5.0,Test species,\"The \"\"Big\"\" Bird\",0.65").unwrap();
        file.flush().unwrap();

        let detections = parse_detection_file(file.path()).unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].common_name, "The \"Big\" Bird");
    }

    #[test]
    fn test_parse_with_bom() {
        let mut file = NamedTempFile::new().unwrap();
        // Write UTF-8 BOM
        file.write_all(b"\xEF\xBB\xBF").unwrap();
        writeln!(
            file,
            "Start (s),End (s),Scientific name,Common name,Confidence"
        )
        .unwrap();
        writeln!(file, "0.0,3.0,Turdus merula,Eurasian Blackbird,0.85").unwrap();
        file.flush().unwrap();

        let detections = parse_detection_file(file.path()).unwrap();
        assert_eq!(detections.len(), 1);
    }

    #[test]
    fn test_empty_file_returns_empty_vec() {
        let file = NamedTempFile::new().unwrap();
        // Empty file returns empty vec (csv crate handles gracefully)
        let result = parse_detection_file(file.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_header_only_returns_empty_vec() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "Start (s),End (s),Scientific name,Common name,Confidence"
        )
        .unwrap();
        file.flush().unwrap();

        let result = parse_detection_file(file.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_invalid_time_range_error() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "Start (s),End (s),Scientific name,Common name,Confidence"
        )
        .unwrap();
        // End time before start time
        writeln!(file, "5.0,3.0,Turdus merula,Eurasian Blackbird,0.85").unwrap();
        file.flush().unwrap();

        let result = parse_detection_file(file.path());
        assert!(matches!(result, Err(Error::InvalidDetectionFormat { .. })));
    }
}
