//! Pipeline coordination for file processing.

use crate::config::OutputFormat;
use crate::constants::output_extensions;
use crate::error::Result;
use crate::locking::FileLock;
use std::path::{Path, PathBuf};
use tracing::warn;

/// Options for processing a single file.
#[derive(Debug, Clone)]
pub struct ProcessOptions {
    /// Output directory (None = same as input).
    pub output_dir: Option<PathBuf>,
    /// Output formats to generate.
    pub formats: Vec<OutputFormat>,
    /// Force reprocessing even if output exists.
    pub force: bool,
    /// Minimum confidence threshold.
    pub min_confidence: f32,
    /// Segment overlap in seconds.
    pub overlap: f32,
    /// Batch size for inference.
    pub batch_size: usize,
    /// Model name.
    pub model_name: String,
}

/// Result of checking whether a file should be processed.
#[derive(Debug)]
pub enum ProcessCheck {
    /// File should be processed.
    Process,
    /// Skip - output already exists.
    SkipExists,
    /// Skip - file is locked by another process.
    SkipLocked,
}

/// Determine the output directory for a file.
pub fn output_dir_for(input: &Path, explicit_output_dir: Option<&Path>) -> PathBuf {
    explicit_output_dir.map_or_else(
        || {
            input
                .parent()
                .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
        },
        Path::to_path_buf,
    )
}

/// Get output file path for a given format.
pub fn output_path_for(input: &Path, output_dir: &Path, format: OutputFormat) -> PathBuf {
    // Use to_string_lossy() to handle non-UTF-8 filenames gracefully
    // Invalid UTF-8 sequences will be replaced with the Unicode replacement character
    let stem = input.file_stem().map_or_else(
        || std::borrow::Cow::Borrowed("output"),
        |s| s.to_string_lossy(),
    );

    let extension = match format {
        OutputFormat::Csv => output_extensions::CSV,
        OutputFormat::Raven => output_extensions::RAVEN,
        OutputFormat::Audacity => output_extensions::AUDACITY,
        OutputFormat::Kaleidoscope => output_extensions::KALEIDOSCOPE,
    };

    output_dir.join(format!("{stem}{extension}"))
}

/// Check if a file should be processed.
pub fn should_process(
    input: &Path,
    output_dir: &Path,
    formats: &[OutputFormat],
    force: bool,
) -> ProcessCheck {
    // Check if locked
    if FileLock::is_locked(input, output_dir) {
        return ProcessCheck::SkipLocked;
    }

    // Check if all outputs exist (unless force)
    if !force {
        let all_exist = formats
            .iter()
            .all(|fmt| output_path_for(input, output_dir, *fmt).exists());
        if all_exist {
            return ProcessCheck::SkipExists;
        }
    }

    ProcessCheck::Process
}

/// Collect input files from paths (files and directories).
pub fn collect_input_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for path in paths {
        if path.is_file() {
            if is_audio_file(path) {
                files.push(path.clone());
            }
        } else if path.is_dir() {
            collect_audio_files_recursive(path, &mut files)?;
        } else {
            warn!("Skipping non-existent path: {}", path.display());
        }
    }

    Ok(files)
}

/// Recursively collect audio files from a directory.
fn collect_audio_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            collect_audio_files_recursive(&path, files)?;
        } else if is_audio_file(&path) {
            files.push(path);
        }
    }

    Ok(())
}

/// Check if a file is a supported audio format.
fn is_audio_file(path: &Path) -> bool {
    use std::ffi::OsStr;

    path.extension().is_some_and(|ext| {
        // Compare extension directly as OsStr to handle non-UTF-8 filenames
        ext.eq_ignore_ascii_case(OsStr::new("wav"))
            || ext.eq_ignore_ascii_case(OsStr::new("flac"))
            || ext.eq_ignore_ascii_case(OsStr::new("mp3"))
            || ext.eq_ignore_ascii_case(OsStr::new("m4a"))
            || ext.eq_ignore_ascii_case(OsStr::new("aac"))
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_output_dir_for_with_explicit() {
        let input = Path::new("/data/audio.wav");
        let output = output_dir_for(input, Some(Path::new("/results")));
        assert_eq!(output, PathBuf::from("/results"));
    }

    #[test]
    fn test_output_dir_for_without_explicit() {
        let input = Path::new("/data/audio.wav");
        let output = output_dir_for(input, None);
        assert_eq!(output, PathBuf::from("/data"));
    }

    #[test]
    fn test_output_path_for_csv() {
        let path = output_path_for(
            Path::new("test.wav"),
            Path::new("/output"),
            OutputFormat::Csv,
        );
        assert!(path.to_string_lossy().ends_with(".BirdNET.results.csv"));
    }

    #[test]
    fn test_is_audio_file() {
        assert!(is_audio_file(Path::new("test.wav")));
        assert!(is_audio_file(Path::new("test.FLAC")));
        assert!(is_audio_file(Path::new("test.mp3")));
        assert!(!is_audio_file(Path::new("test.txt")));
    }

    #[test]
    fn test_is_audio_file_with_unicode() {
        // Test with Finnish/Swedish characters
        assert!(is_audio_file(Path::new("ääni_tiedostö.wav")));
        assert!(is_audio_file(Path::new("räkä.flac")));
        assert!(is_audio_file(Path::new("öljy.mp3")));
        assert!(is_audio_file(Path::new("テスト.wav"))); // Japanese
    }

    #[test]
    fn test_output_path_for_unicode() {
        // Test that Unicode filenames preserve their names in output
        let path = output_path_for(
            Path::new("ääni_tiedostö.wav"),
            Path::new("/output"),
            OutputFormat::Csv,
        );
        assert!(path.to_string_lossy().contains("ääni_tiedostö"));
    }
}
