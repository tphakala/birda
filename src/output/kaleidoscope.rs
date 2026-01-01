//! Kaleidoscope CSV output format writer.

use crate::constants::confidence::DECIMAL_PLACES;
use crate::error::Result;
use crate::output::{Detection, OutputWriter};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Kaleidoscope CSV output writer.
pub struct KaleidoscopeWriter {
    writer: BufWriter<File>,
}

impl KaleidoscopeWriter {
    /// Create a new Kaleidoscope writer.
    pub fn new(path: &Path) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

impl OutputWriter for KaleidoscopeWriter {
    fn write_header(&mut self) -> Result<()> {
        writeln!(self.writer, "INDIR,FOLDER,IN FILE,OFFSET,DURATION,TOP1MATCH,TOP1DIST")?;
        Ok(())
    }

    fn write_detection(&mut self, detection: &Detection) -> Result<()> {
        let path = &detection.file_path;

        // Get parent directory and grandparent
        let folder = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("");

        let indir = path
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.display().to_string())
            .unwrap_or_default();

        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        let duration = detection.end_time - detection.start_time;

        writeln!(
            self.writer,
            "{},{},{},{:.1},{:.1},{},{:.decimal$}",
            indir,
            folder,
            filename,
            detection.start_time,
            duration,
            detection.common_name.replace(' ', "_"),
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
    fn test_kaleidoscope_writer_basic() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = KaleidoscopeWriter::new(file.path()).unwrap();

        writer.write_header().unwrap();

        let detection = Detection::from_label(
            "Passer domesticus_House Sparrow",
            0.8542,
            0.0,
            3.0,
            PathBuf::from("/home/user/recordings/morning/audio.wav"),
        );
        writer.write_detection(&detection).unwrap();
        writer.finalize().unwrap();

        let contents = std::fs::read_to_string(file.path()).unwrap();
        assert!(contents.contains("INDIR,FOLDER,IN FILE"));
        assert!(contents.contains("morning"));
        assert!(contents.contains("audio.wav"));
        assert!(contents.contains("House_Sparrow"));
    }
}
