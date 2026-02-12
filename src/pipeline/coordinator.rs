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

/// Sanitize a filename to prevent path traversal attacks.
///
/// Replaces path separators with underscores.
fn sanitize_filename(filename: &str) -> String {
    filename.replace(['/', '\\'], "_")
}

/// Get output file path for a given format.
///
/// The filename is sanitized to prevent path traversal attacks.
pub fn output_path_for(input: &Path, output_dir: &Path, format: OutputFormat) -> PathBuf {
    // Use to_string_lossy() to handle non-UTF-8 filenames gracefully
    // Invalid UTF-8 sequences will be replaced with the Unicode replacement character
    let stem = input.file_stem().map_or_else(
        || std::borrow::Cow::Borrowed("output"),
        |s| s.to_string_lossy(),
    );

    // Sanitize filename to prevent path traversal
    let safe_stem = sanitize_filename(&stem);

    let extension = match format {
        OutputFormat::Csv => output_extensions::CSV,
        OutputFormat::Raven => output_extensions::RAVEN,
        OutputFormat::Audacity => output_extensions::AUDACITY,
        OutputFormat::Kaleidoscope => output_extensions::KALEIDOSCOPE,
        OutputFormat::Json => output_extensions::JSON,
    };

    let output_path = output_dir.join(format!("{safe_stem}{extension}"));

    // Verify the output path is within the output directory (defense in depth)
    debug_assert!(
        output_path.starts_with(output_dir),
        "Output path escapes output directory: {}",
        output_path.display()
    );

    output_path
}

/// Check if a file should be processed.
pub fn should_process(
    input: &Path,
    output_dir: &Path,
    formats: &[OutputFormat],
    force: bool,
    stdout_mode: bool,
) -> ProcessCheck {
    // Check if locked
    if FileLock::is_locked(input, output_dir) {
        return ProcessCheck::SkipLocked;
    }

    // Skip file existence check in stdout mode (no files written)
    if stdout_mode {
        return ProcessCheck::Process;
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
    const AUDIO_EXTENSIONS: &[&str] = &["wav", "flac", "mp3", "m4a", "aac"];

    // Compare extension directly as OsStr to handle non-UTF-8 filenames
    path.extension().is_some_and(|ext| {
        AUDIO_EXTENSIONS
            .iter()
            .any(|&audio_ext| ext.eq_ignore_ascii_case(audio_ext))
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

    #[test]
    fn test_sanitize_filename_normal() {
        // Normal filenames should pass through unchanged
        assert_eq!(sanitize_filename("audio_file"), "audio_file");
        assert_eq!(sanitize_filename("recording-2024"), "recording-2024");
        assert_eq!(sanitize_filename("test123"), "test123");
    }

    #[test]
    fn test_sanitize_filename_path_separators() {
        // Path separators should be replaced with underscores
        assert_eq!(sanitize_filename("../etc/passwd"), ".._etc_passwd");
        assert_eq!(sanitize_filename("subdir/file"), "subdir_file");
        assert_eq!(
            sanitize_filename("..\\windows\\system32"),
            ".._windows_system32"
        );
    }

    #[test]
    fn test_sanitize_filename_parent_directory() {
        // Path separators in parent directory references are sanitized
        assert_eq!(sanitize_filename(".."), ".."); // No slashes to replace
        assert_eq!(sanitize_filename("../audio"), ".._audio");
        assert_eq!(sanitize_filename("../../file"), ".._.._file");
    }

    #[test]
    fn test_output_path_for_prevents_traversal() {
        // Even with malicious input, output should stay in output_dir
        let output_dir = Path::new("/safe/output");

        let path1 = output_path_for(
            Path::new("../etc/passwd.wav"),
            output_dir,
            OutputFormat::Json,
        );
        assert!(path1.starts_with(output_dir));

        let path2 = output_path_for(
            Path::new("../../root/.ssh/id_rsa.wav"),
            output_dir,
            OutputFormat::Csv,
        );
        assert!(path2.starts_with(output_dir));

        // Path should not contain actual path separators after sanitization
        let path_str = path1.to_string_lossy();
        assert!(!path_str.contains("../"));
    }
}
