//! Raven selection table output format writer.

use crate::constants::{confidence::DECIMAL_PLACES, raven};
use crate::error::Result;
use crate::output::{Detection, OutputWriter};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Raven selection table output writer.
pub struct RavenWriter {
    writer: BufWriter<File>,
    selection_id: u32,
}

impl RavenWriter {
    /// Create a new Raven writer.
    pub fn new(path: &Path) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            selection_id: 0,
        })
    }
}

impl OutputWriter for RavenWriter {
    fn write_header(&mut self) -> Result<()> {
        writeln!(
            self.writer,
            "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tCommon Name\tSpecies Code\tConfidence\tBegin Path\tFile Offset (s)"
        )?;
        Ok(())
    }

    fn write_detection(&mut self, detection: &Detection) -> Result<()> {
        self.selection_id += 1;

        // Replace spaces with underscores in common name for Raven format
        let common_name = detection.common_name.replace(' ', "_");

        // Species code would normally come from eBird taxonomy
        // For now, use a placeholder based on common name
        let species_code = generate_species_code(&detection.common_name);

        writeln!(
            self.writer,
            "{}\t{}\t{}\t{:.1}\t{:.1}\t{}\t{}\t{}\t{}\t{:.decimal$}\t{}\t{:.1}",
            self.selection_id,
            raven::VIEW,
            raven::CHANNEL,
            detection.start_time,
            detection.end_time,
            raven::DEFAULT_LOW_FREQ,
            raven::DEFAULT_HIGH_FREQ,
            common_name,
            species_code,
            detection.confidence,
            detection.file_path.display(),
            detection.start_time,
            decimal = DECIMAL_PLACES,
        )?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Generate a simple species code from common name.
///
/// This is a placeholder - real implementation should use eBird taxonomy.
fn generate_species_code(common_name: &str) -> String {
    let words: Vec<&str> = common_name.split_whitespace().collect();
    match words.len() {
        0 => "unkn".to_string(),
        1 => words[0].chars().take(4).collect::<String>().to_lowercase(),
        _ => {
            let first: String = words[0].chars().take(3).collect();
            let last: String = words.last().unwrap_or(&"").chars().take(3).collect();
            format!("{}{}", first.to_lowercase(), last.to_lowercase())
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    #[test]
    fn test_raven_writer_basic() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = RavenWriter::new(file.path()).unwrap();

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
        assert!(contents.contains("Selection\tView"));
        assert!(contents.contains("House_Sparrow"));
        assert!(contents.contains("Spectrogram 1"));
    }

    #[test]
    fn test_generate_species_code() {
        assert_eq!(generate_species_code("House Sparrow"), "houspa");
        assert_eq!(generate_species_code("Robin"), "robi");
        assert_eq!(generate_species_code("European Robin"), "eurrob");
    }
}
