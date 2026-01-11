//! Detection file parsing.
//!
//! Parses birda CSV detection files to extract detection information
//! for clip extraction.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::Error;

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

/// Column indices for CSV parsing.
struct ColumnIndices {
    start: usize,
    end: usize,
    scientific_name: usize,
    common_name: usize,
    confidence: usize,
}

impl ColumnIndices {
    fn from_header(header: &str) -> Result<Self, Error> {
        let columns = split_csv_line(header);

        let find_column = |name: &str| -> Result<usize, Error> {
            columns
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| Error::MissingDetectionColumn {
                    column: name.to_string(),
                })
        };

        Ok(Self {
            start: find_column("Start (s)")?,
            end: find_column("End (s)")?,
            scientific_name: find_column("Scientific name")?,
            common_name: find_column("Common name")?,
            confidence: find_column("Confidence")?,
        })
    }
}

/// Split a CSV line respecting quoted fields.
///
/// Handles commas within double-quoted fields correctly.
/// Strips quotes from quoted fields.
fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current_field = String::new();
    let mut in_quotes = false;

    for c in line.chars() {
        match c {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                fields.push(current_field.trim().to_string());
                current_field.clear();
            }
            _ => current_field.push(c),
        }
    }
    fields.push(current_field.trim().to_string());
    fields
}

/// Parse a detection file and return detections.
///
/// Supports birda CSV format with columns:
/// - Start (s), End (s), Scientific name, Common name, Confidence
///
/// Handles UTF-8 BOM if present and quoted fields with embedded commas.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - Required columns are missing
/// - Values cannot be parsed
/// - No detections are found
pub fn parse_detection_file(path: &Path) -> Result<Vec<ParsedDetection>, Error> {
    let file = File::open(path).map_err(|e| Error::DetectionParseFailed {
        path: path.to_path_buf(),
        source: Box::new(e),
    })?;

    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Read header line
    let header = lines
        .next()
        .ok_or_else(|| Error::InvalidDetectionFormat {
            message: "file is empty".to_string(),
        })?
        .map_err(|e| Error::DetectionParseFailed {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?;

    // Strip UTF-8 BOM if present
    let header = header.strip_prefix('\u{FEFF}').unwrap_or(&header);

    let indices = ColumnIndices::from_header(header)?;

    let mut detections = Vec::new();

    for (line_num, line_result) in lines.enumerate() {
        let line = line_result.map_err(|e| Error::DetectionParseFailed {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?;

        if line.trim().is_empty() {
            continue;
        }

        let fields = split_csv_line(&line);

        let parse_field = |idx: usize, name: &str| -> Result<&str, Error> {
            fields
                .get(idx)
                .map(String::as_str)
                .ok_or_else(|| Error::InvalidDetectionFormat {
                    message: format!("line {}: missing field '{name}'", line_num + 2),
                })
        };

        let start: f64 = parse_field(indices.start, "Start (s)")?
            .trim()
            .parse()
            .map_err(|_| Error::InvalidDetectionFormat {
                message: format!("line {}: invalid start time", line_num + 2),
            })?;

        let end: f64 = parse_field(indices.end, "End (s)")?
            .trim()
            .parse()
            .map_err(|_| Error::InvalidDetectionFormat {
                message: format!("line {}: invalid end time", line_num + 2),
            })?;

        let scientific_name = parse_field(indices.scientific_name, "Scientific name")?
            .trim()
            .to_string();

        let common_name = parse_field(indices.common_name, "Common name")?
            .trim()
            .to_string();

        let confidence: f32 = parse_field(indices.confidence, "Confidence")?
            .trim()
            .parse()
            .map_err(|_| Error::InvalidDetectionFormat {
                message: format!("line {}: invalid confidence", line_num + 2),
            })?;

        // Validate time range
        if end <= start {
            return Err(Error::InvalidDetectionFormat {
                message: format!(
                    "line {}: end time ({end}) must be greater than start time ({start})",
                    line_num + 2
                ),
            });
        }

        detections.push(ParsedDetection {
            start,
            end,
            scientific_name,
            common_name,
            confidence,
        });
    }

    if detections.is_empty() {
        return Err(Error::NoDetectionsFound {
            path: path.to_path_buf(),
        });
    }

    Ok(detections)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_csv_line_simple() {
        let fields = split_csv_line("a,b,c");
        assert_eq!(fields, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_csv_line_quoted() {
        let fields = split_csv_line("a,\"b,c\",d");
        assert_eq!(fields, vec!["a", "b,c", "d"]);
    }

    #[test]
    fn test_split_csv_line_quoted_with_spaces() {
        let fields = split_csv_line("1.0, \"Owl, Barn\", 0.85");
        assert_eq!(fields, vec!["1.0", "Owl, Barn", "0.85"]);
    }
}
