//! WAV file writing.
//!
//! Writes audio clips to WAV files organized by species.

use std::fs;
use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavSpec, WavWriter as HoundWriter};

use crate::Error;

/// Writes audio data to WAV files.
pub struct WavWriter {
    /// Output directory for clips.
    output_dir: PathBuf,
}

impl WavWriter {
    /// Create a new WAV writer with the given output directory.
    #[must_use]
    pub fn new(output_dir: PathBuf) -> Self {
        Self { output_dir }
    }

    /// Write audio samples to a WAV file.
    ///
    /// Creates a species subdirectory and writes the clip with a descriptive
    /// filename containing species, confidence, and time range.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples as f32 (-1.0 to 1.0)
    /// * `sample_rate` - Sample rate in Hz
    /// * `species` - Species name for directory and filename
    /// * `confidence` - Detection confidence (0.0-1.0)
    /// * `start_time` - Clip start time in seconds
    /// * `end_time` - Clip end time in seconds
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or the file
    /// cannot be written.
    pub fn write_clip(
        &self,
        samples: &[f32],
        sample_rate: u32,
        species: &str,
        confidence: f32,
        start_time: f64,
        end_time: f64,
    ) -> Result<PathBuf, Error> {
        // Sanitize species name for filesystem
        let safe_species = sanitize_filename(species);

        // Create species subdirectory
        let species_dir = self.output_dir.join(&safe_species);
        fs::create_dir_all(&species_dir).map_err(|e| Error::OutputDirCreateFailed {
            path: species_dir.clone(),
            source: e,
        })?;

        // Generate filename
        let filename = generate_filename(&safe_species, confidence, start_time, end_time);
        let output_path = species_dir.join(filename);

        // Write WAV file
        write_wav_file(&output_path, samples, sample_rate)?;

        Ok(output_path)
    }
}

/// Sanitize a string for use as a filename/directory name.
///
/// Replaces characters that are invalid in filenames across platforms
/// and prevents path traversal attacks.
fn sanitize_filename(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => c,
        })
        .collect();

    // Prevent path traversal: replace ".." with "__"
    sanitized.replace("..", "__")
}

/// Generate a filename for a clip.
///
/// Format: `species_confidence_start-end.wav`
/// Example: `Parus major_85p_10.5-13.5.wav`
fn generate_filename(species: &str, confidence: f32, start_time: f64, end_time: f64) -> String {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let confidence_pct = (confidence * 100.0).round() as u32;
    format!("{species}_{confidence_pct}p_{start_time:.1}-{end_time:.1}.wav")
}

/// Write samples to a WAV file.
fn write_wav_file(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), Error> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = HoundWriter::create(path, spec).map_err(|e| Error::WavWriteFailed {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Convert f32 samples to i16
    for &sample in samples {
        #[allow(clippy::cast_possible_truncation)]
        let sample_i16 = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
        writer
            .write_sample(sample_i16)
            .map_err(|e| Error::WavWriteFailed {
                path: path.to_path_buf(),
                source: e,
            })?;
    }

    writer.finalize().map_err(|e| Error::WavWriteFailed {
        path: path.to_path_buf(),
        source: e,
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("Parus major"), "Parus major");
        assert_eq!(sanitize_filename("a/b:c*d"), "a_b_c_d");
        assert_eq!(sanitize_filename("file?name"), "file_name");
    }

    #[test]
    fn test_sanitize_filename_prevents_path_traversal() {
        // Direct traversal attempts
        assert_eq!(sanitize_filename(".."), "__");
        assert_eq!(sanitize_filename("../etc"), "___etc"); // "/" -> "_", ".." -> "__"
        assert_eq!(sanitize_filename("foo/../bar"), "foo____bar"); // "/" -> "_", ".." -> "__"
        // Preserves single dots (e.g., species abbreviations)
        assert_eq!(sanitize_filename("P. major"), "P. major");
        assert_eq!(sanitize_filename("sp."), "sp.");
    }

    #[test]
    fn test_generate_filename() {
        let filename = generate_filename("Species", 0.8542, 10.5, 13.5);
        assert_eq!(filename, "Species_85p_10.5-13.5.wav");
    }
}
