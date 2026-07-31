//! Detection file parsing.
//!
//! Parses birda CSV detection files to extract detection information
//! for clip extraction. Uses the `csv` crate for robust parsing.

use std::path::Path;

use serde::Deserialize;
use tracing::warn;

use crate::Error;
use crate::constants::clipper::MAX_SKIPPED_ROW_WARNINGS;

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
/// A row whose start, end or confidence is not finite is **dropped**, with a
/// warning naming its line, and the call still succeeds. Every returned
/// [`ParsedDetection`] therefore has finite fields and `end > start`, and the
/// returned length may be shorter than the file. The alternative, failing the
/// file, costs every good row beside the bad one, because a caller that gets
/// an error here has no way to recover the rest.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - Required columns are missing
/// - Values cannot be parsed
/// - A row's end time is not greater than its start time
///
/// Returns `Ok(vec![])` if the file contains no detections (empty, header-only,
/// or every row dropped).
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
    let mut skipped = 0usize;

    for (line_num, result) in reader.deserialize::<DetectionRecord>().enumerate() {
        let record = result.map_err(|e| Error::InvalidDetectionFormat {
            message: format!("line {}: {e}", line_num + 2),
        })?;

        // The ordering test below cannot reject a non-finite value:
        // `record.end <= record.start` is false whenever either side is NaN,
        // and false again for an infinite end, so both used to reach the
        // extractor. An infinite end aborted the process with a capacity
        // overflow, a NaN start was laundered into 0.0 by the grouper's
        // `.max(0.0)` and produced a clip over a range the file never claimed,
        // and a NaN confidence made `confidence >= threshold` false so the row
        // vanished with no diagnostic at all.
        //
        // Skipped rather than rejected, and the distinction is the whole point
        // of the check. `process_detection_file` discards an entire file when
        // this function returns an error, so a hard error here trades one
        // unusable row for every good row beside it: a 10,000-row file with a
        // single bad line yields nothing. The extractor's own guard already
        // contains a bad range safely one row at a time, so failing the file
        // buys no safety and costs the other 9,999 clips.
        //
        // The pre-existing `end <= start` rejection below keeps its hard-error
        // contract, which `test_invalid_time_range_error` here and
        // `test_parse_invalid_time_range_returns_error` in
        // `tests/clipper_parser_test.rs` both pin. The argument above applies
        // to it just as well, so the two rejections now differ in blast radius
        // for no reason a user could infer, but widening a twice-pinned
        // contract is a policy change rather than a fix for #310. Tracked in
        // #319 with the rest of the command's error contract.
        //
        // The per-row warnings are counted and capped, because the row count
        // is caller-supplied: a file of nothing but malformed rows would
        // otherwise spend several times the parsing cost on formatting
        // diagnostics nobody will read past the first few.
        if !record.start.is_finite() || !record.end.is_finite() {
            skipped += 1;
            if skipped <= MAX_SKIPPED_ROW_WARNINGS {
                warn!(
                    "line {}: skipping detection, start ({}) and end ({}) must both be finite",
                    line_num + 2,
                    record.start,
                    record.end
                );
            }
            continue;
        }

        if !record.confidence.is_finite() {
            skipped += 1;
            if skipped <= MAX_SKIPPED_ROW_WARNINGS {
                warn!(
                    "line {}: skipping detection, confidence ({}) must be finite",
                    line_num + 2,
                    record.confidence
                );
            }
            continue;
        }

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

    if skipped > MAX_SKIPPED_ROW_WARNINGS {
        warn!(
            "skipped {skipped} malformed detections in '{}'; {} further warnings suppressed",
            path.display(),
            skipped - MAX_SKIPPED_ROW_WARNINGS
        );
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

    /// Write a one-detection CSV and parse it, so the non-finite cases below
    /// differ only in the row under test.
    fn parse_single_row(row: &str) -> Result<Vec<ParsedDetection>, Error> {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "Start (s),End (s),Scientific name,Common name,Confidence"
        )
        .unwrap();
        writeln!(file, "{row}").unwrap();
        file.flush().unwrap();

        parse_detection_file(file.path())
    }

    #[test]
    fn test_parse_skips_rows_with_non_finite_times() {
        // `end <= start` is false whenever either side is NaN, and false again
        // for an infinite end, so both used to reach the extractor: infinity
        // aborted the process on a `Vec::with_capacity`, NaN wrote a clip over
        // a range the file never claimed.
        for row in [
            "0.0,inf,Parus major,Great Tit,0.85",
            "0.0,-inf,Parus major,Great Tit,0.85",
            "nan,3.0,Parus major,Great Tit,0.85",
            "0.0,nan,Parus major,Great Tit,0.85",
            "inf,inf,Parus major,Great Tit,0.85",
        ] {
            let detections = parse_single_row(row).unwrap();
            assert!(detections.is_empty(), "{row} produced {detections:?}");
        }
    }

    #[test]
    fn test_parse_keeps_the_rows_around_a_skipped_time_range() {
        // The measured cost of getting this wrong: a hard error here discards
        // the whole file, so a 10,000-row detection file with one bad line
        // yields nothing at all. The extractor's own guard already contains a
        // bad range one row at a time.
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "Start (s),End (s),Scientific name,Common name,Confidence"
        )
        .unwrap();
        writeln!(file, "0.0,3.0,Parus major,Great Tit,0.85").unwrap();
        writeln!(file, "5.0,inf,Parus major,Great Tit,0.85").unwrap();
        writeln!(file, "10.0,13.0,Turdus merula,Eurasian Blackbird,0.91").unwrap();
        file.flush().unwrap();

        let detections = parse_detection_file(file.path()).unwrap();
        assert_eq!(
            detections.len(),
            2,
            "only the infinite row should be dropped"
        );
        assert!(
            detections
                .iter()
                .all(|d| d.start.is_finite() && d.end.is_finite())
        );
    }

    #[test]
    fn test_parse_skips_every_bad_row_even_past_the_warning_cap() {
        // The warnings are capped, the skipping is not. A file with more
        // malformed rows than `MAX_SKIPPED_ROW_WARNINGS` must still drop all
        // of them and keep all the good ones.
        let bad_rows = MAX_SKIPPED_ROW_WARNINGS * 3;

        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "Start (s),End (s),Scientific name,Common name,Confidence"
        )
        .unwrap();
        for i in 0..bad_rows {
            writeln!(file, "{i}.0,inf,Parus major,Great Tit,0.85").unwrap();
        }
        writeln!(file, "0.0,3.0,Turdus merula,Eurasian Blackbird,0.85").unwrap();
        file.flush().unwrap();

        let detections = parse_detection_file(file.path()).unwrap();
        assert_eq!(detections.len(), 1, "{bad_rows} bad rows should all skip");
        assert_eq!(detections[0].scientific_name, "Turdus merula");
    }

    #[test]
    fn test_parse_skips_a_confidence_that_overflows_f32() {
        // `1e40` is an ordinary decimal a hand-edited or third-party CSV can
        // carry, and it overflows `f32` to infinity on the way in. Before it
        // was skipped here it survived as `inf`, comparing greater than every
        // threshold.
        let detections = parse_single_row("0.0,3.0,Parus major,Great Tit,1e40").unwrap();
        assert!(detections.is_empty(), "{detections:?}");
    }

    #[test]
    fn test_parse_skips_a_row_with_a_non_finite_confidence() {
        // A NaN confidence made `confidence >= threshold` false, dropping the
        // detection with no diagnostic and a zero exit code. It is now skipped
        // with a warning rather than failing the file: `process_detection_file`
        // discards a whole file when this function errors, so one odd
        // confidence must not cost the rows around it.
        let detections = parse_single_row("0.0,3.0,Parus major,Great Tit,nan").unwrap();
        assert!(detections.is_empty(), "{detections:?}");
    }

    #[test]
    fn test_parse_keeps_the_rows_around_a_skipped_confidence() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "Start (s),End (s),Scientific name,Common name,Confidence"
        )
        .unwrap();
        writeln!(file, "0.0,3.0,Parus major,Great Tit,0.85").unwrap();
        writeln!(file, "5.0,8.0,Parus major,Great Tit,nan").unwrap();
        writeln!(file, "10.0,13.0,Turdus merula,Eurasian Blackbird,0.91").unwrap();
        file.flush().unwrap();

        let detections = parse_detection_file(file.path()).unwrap();
        assert_eq!(detections.len(), 2, "only the NaN row should be dropped");
        assert!(detections.iter().all(|d| d.confidence.is_finite()));
    }

    #[test]
    fn test_parse_still_rejects_an_inverted_range_by_line() {
        // The pre-existing hard-error contract, unchanged by #310: a non-finite
        // value skips its row, an inverted range still fails the file.
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "Start (s),End (s),Scientific name,Common name,Confidence"
        )
        .unwrap();
        writeln!(file, "0.0,3.0,Parus major,Great Tit,0.85").unwrap();
        writeln!(file, "8.0,5.0,Parus major,Great Tit,0.85").unwrap();
        file.flush().unwrap();

        let err =
            parse_detection_file(file.path()).expect_err("an inverted range must be rejected");
        assert!(err.to_string().contains("line 3"), "{err}");
    }

    #[test]
    fn test_parse_accepts_finite_boundary_values() {
        let detections = parse_single_row("0.0,0.001,Parus major,Great Tit,0.0").unwrap();
        assert_eq!(detections.len(), 1);
        assert!(detections[0].confidence.abs() < f32::EPSILON);
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
