//! WAV file writing.

/// Writes audio data to WAV files.
pub struct WavWriter {
    /// Output directory for clips.
    output_dir: std::path::PathBuf,
}

impl WavWriter {
    /// Create a new WAV writer with the given output directory.
    #[must_use]
    pub fn new(output_dir: std::path::PathBuf) -> Self {
        Self { output_dir }
    }

    /// Write audio samples to a WAV file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    #[allow(clippy::todo)]
    pub fn write_clip(
        &self,
        _samples: &[f32],
        _sample_rate: u32,
        _species: &str,
        _confidence: f32,
        _start_time: f64,
        _end_time: f64,
    ) -> Result<std::path::PathBuf, crate::Error> {
        let _ = &self.output_dir;
        todo!()
    }
}
