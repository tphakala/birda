//! CSV output format writer.

use crate::constants::UTF8_BOM;
use crate::constants::confidence::DECIMAL_PLACES;
use crate::error::Result;
use crate::output::{Detection, OutputWriter};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// CSV format output writer.
pub struct CsvWriter {
    writer: BufWriter<File>,
    include_columns: Vec<String>,
}

impl CsvWriter {
    /// Create a new CSV writer.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the output CSV file
    /// * `include_columns` - Additional columns to include in output
    /// * `include_bom` - Whether to write UTF-8 BOM for Excel compatibility
    pub fn new(path: &Path, include_columns: Vec<String>, include_bom: bool) -> Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write UTF-8 BOM for Excel compatibility (unless disabled)
        if include_bom {
            writer.write_all(UTF8_BOM)?;
        }

        Ok(Self {
            writer,
            include_columns,
        })
    }
}

impl OutputWriter for CsvWriter {
    fn write_header(&mut self) -> Result<()> {
        let mut header =
            "Start (s),End (s),Scientific name,Common name,Confidence,File".to_string();

        for col in &self.include_columns {
            header.push(',');
            header.push_str(col);
        }

        writeln!(self.writer, "{header}")?;
        Ok(())
    }

    fn write_detection(&mut self, detection: &Detection) -> Result<()> {
        write!(
            self.writer,
            "{:.1},{:.1},{},{},{:.decimal$},{}",
            detection.start_time,
            detection.end_time,
            escape_csv(&detection.scientific_name),
            escape_csv(&detection.common_name),
            detection.confidence,
            escape_csv(&detection.file_path.display().to_string()),
            decimal = DECIMAL_PLACES,
        )?;

        for col in &self.include_columns {
            write!(self.writer, ",")?;
            match col.as_str() {
                "lat" => {
                    if let Some(lat) = detection.metadata.lat {
                        write!(self.writer, "{lat}")?;
                    }
                }
                "lon" => {
                    if let Some(lon) = detection.metadata.lon {
                        write!(self.writer, "{lon}")?;
                    }
                }
                "week" => {
                    if let Some(week) = detection.metadata.week {
                        write!(self.writer, "{week}")?;
                    }
                }
                "model" => {
                    if let Some(ref model) = detection.metadata.model {
                        write!(self.writer, "{}", escape_csv(model))?;
                    }
                }
                "overlap" => {
                    if let Some(overlap) = detection.metadata.overlap {
                        write!(self.writer, "{overlap}")?;
                    }
                }
                "sensitivity" => {
                    if let Some(sens) = detection.metadata.sensitivity {
                        write!(self.writer, "{sens}")?;
                    }
                }
                "min_conf" => {
                    if let Some(min_conf) = detection.metadata.min_conf {
                        write!(self.writer, "{min_conf}")?;
                    }
                }
                "species_list" => {
                    if let Some(ref list) = detection.metadata.species_list {
                        write!(self.writer, "{}", escape_csv(list))?;
                    }
                }
                _ => {}
            }
        }

        writeln!(self.writer)?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Escape a value for CSV output.
fn escape_csv(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_writer_basic() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = CsvWriter::new(file.path(), vec![], true).unwrap();

        writer.write_header().unwrap();

        let detection = Detection::from_label(
            "Passer domesticus_House Sparrow",
            0.8542,
            0.0,
            3.0,
            PathBuf::from("/path/to/audio.wav"),
        );
        writer.write_detection(&detection).unwrap();
        writer.finalize().unwrap();

        let contents = std::fs::read_to_string(file.path()).unwrap();
        assert!(contents.contains("Start (s),End (s)"));
        assert!(contents.contains("House Sparrow"));
        assert!(contents.contains("0.8542"));
    }

    #[test]
    fn test_every_recognised_column_is_written() {
        // Pins `csv_columns::RECOGNISED`, which `config::validate` accepts, to
        // what this writer actually handles. The two are matched on strings in
        // arms the compiler cannot connect to the constant, and the fall-through
        // is silent: an unhandled name becomes a header over a column empty in
        // every row. Adding a name to the constant without an arm here, or
        // removing an arm, fails this test rather than shipping that column.
        let columns: Vec<String> = crate::constants::csv_columns::RECOGNISED
            .iter()
            .map(|c| (*c).to_string())
            .collect();

        let file = NamedTempFile::new().unwrap();
        let mut writer = CsvWriter::new(file.path(), columns.clone(), false).unwrap();

        let mut detection = Detection::from_label(
            "Passer domesticus_House Sparrow",
            0.85,
            0.0,
            3.0,
            PathBuf::from("/path/to/audio.wav"),
        );
        // Every optional field set, so an empty cell can only mean the writer
        // did not handle the column.
        detection.metadata = crate::output::DetectionMetadata {
            lat: Some(60.1699),
            lon: Some(24.9384),
            week: Some(24),
            model: Some("birdnet-v24".to_string()),
            overlap: Some(1.5),
            sensitivity: Some(1.25),
            min_conf: Some(0.1),
            species_list: Some("finland.txt".to_string()),
        };

        writer.write_header().unwrap();
        writer.write_detection(&detection).unwrap();
        writer.finalize().unwrap();

        let contents = std::fs::read_to_string(file.path()).unwrap();
        let mut lines = contents.lines();
        let header: Vec<&str> = lines.next().unwrap().split(',').collect();
        let row: Vec<&str> = lines.next().unwrap().split(',').collect();
        assert_eq!(header.len(), row.len(), "header and row must line up");

        // Each cell is asserted against the value that column is FOR, not
        // merely against being non-empty. Non-emptiness alone passes when an
        // arm emits the wrong field: rewriting the `lat` arm to write
        // `metadata.lon` leaves a populated `lat` column carrying a longitude,
        // and a presence check cannot see it. Every value here is comma-free,
        // which is what keeps the naive `split(',')` above honest; the
        // `header.len() == row.len()` assertion fails loudly rather than
        // silently misaligning if that ever stops being true.
        let expected = [
            ("lat", "60.1699"),
            ("lon", "24.9384"),
            ("week", "24"),
            ("model", "birdnet-v24"),
            ("overlap", "1.5"),
            ("sensitivity", "1.25"),
            ("min_conf", "0.1"),
            ("species_list", "finland.txt"),
        ];
        assert_eq!(
            expected.len(),
            columns.len(),
            "every name in RECOGNISED needs an expected value here"
        );

        for (column, want) in expected {
            let index = header
                .iter()
                .position(|h| *h == column)
                .unwrap_or_else(|| panic!("'{column}' is missing from the header"));
            assert_eq!(
                row[index], want,
                "'{column}' must carry its own metadata field; an empty cell means no arm in \
                 the match, a wrong value means the arm reads the wrong field"
            );
        }
    }

    #[test]
    fn test_escape_csv() {
        assert_eq!(escape_csv("simple"), "simple");
        assert_eq!(escape_csv("with,comma"), "\"with,comma\"");
        assert_eq!(escape_csv("with\"quote"), "\"with\"\"quote\"");
    }

    #[test]
    fn test_csv_writer_with_bom() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = CsvWriter::new(file.path(), vec![], true).unwrap();

        writer.write_header().unwrap();
        writer.finalize().unwrap();

        let bytes = std::fs::read(file.path()).unwrap();
        // Check UTF-8 BOM is present at start
        assert_eq!(&bytes[0..3], UTF8_BOM);
        // Check header follows BOM
        let content = String::from_utf8_lossy(&bytes[3..]);
        assert!(content.starts_with("Start (s),End (s)"));
    }

    #[test]
    fn test_csv_writer_without_bom() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = CsvWriter::new(file.path(), vec![], false).unwrap();

        writer.write_header().unwrap();
        writer.finalize().unwrap();

        let bytes = std::fs::read(file.path()).unwrap();
        // Check no BOM at start
        assert_ne!(&bytes[0..3], UTF8_BOM);
        // Check header starts immediately
        let content = String::from_utf8_lossy(&bytes);
        assert!(content.starts_with("Start (s),End (s)"));
    }
}
