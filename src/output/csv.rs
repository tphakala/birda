//! CSV output format writer.

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
    pub fn new(path: &Path, include_columns: Vec<String>) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_writer_basic() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = CsvWriter::new(file.path(), vec![]).unwrap();

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
    fn test_escape_csv() {
        assert_eq!(escape_csv("simple"), "simple");
        assert_eq!(escape_csv("with,comma"), "\"with,comma\"");
        assert_eq!(escape_csv("with\"quote"), "\"with\"\"quote\"");
    }
}
