//! Audacity labels output format writer.

use crate::constants::confidence::DECIMAL_PLACES;
use crate::error::Result;
use crate::output::{Detection, OutputWriter};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Audacity labels output writer.
pub struct AudacityWriter {
    writer: BufWriter<File>,
}

impl AudacityWriter {
    /// Create a new Audacity writer.
    pub fn new(path: &Path) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

impl OutputWriter for AudacityWriter {
    fn write_header(&mut self) -> Result<()> {
        // Audacity format has no header
        Ok(())
    }

    fn write_detection(&mut self, detection: &Detection) -> Result<()> {
        // Replace underscores with commas for Audacity format
        let species_name = detection.common_name.replace('_', ", ");

        writeln!(
            self.writer,
            "{:.1}\t{:.1}\t{}\t{:.decimal$}",
            detection.start_time,
            detection.end_time,
            species_name,
            detection.confidence,
            decimal = DECIMAL_PLACES,
        )?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    #[test]
    fn test_audacity_writer_basic() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = AudacityWriter::new(file.path()).unwrap();

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
        assert!(contents.contains("0.0\t3.0\tHouse Sparrow\t0.8542"));
    }

    #[test]
    fn test_audacity_no_header() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = AudacityWriter::new(file.path()).unwrap();
        writer.write_header().unwrap();
        writer.finalize().unwrap();

        let contents = std::fs::read_to_string(file.path()).unwrap();
        assert!(contents.is_empty());
    }
}
