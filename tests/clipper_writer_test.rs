//! Tests for WAV file writer.

use birda::clipper::WavWriter;
use tempfile::TempDir;

#[test]
fn test_write_clip_creates_species_directory() {
    let temp_dir = TempDir::new().unwrap();
    let writer = WavWriter::new(temp_dir.path().to_path_buf());

    // Simple sine wave samples
    let samples: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.01).sin()).collect();

    let path = writer
        .write_clip(&samples, 48000, "Parus major", 0.85, 10.5, 11.5)
        .unwrap();

    assert!(path.exists());
    assert!(path.parent().unwrap().ends_with("Parus major"));
}

#[test]
fn test_write_clip_filename_format() {
    let temp_dir = TempDir::new().unwrap();
    let writer = WavWriter::new(temp_dir.path().to_path_buf());

    let samples: Vec<f32> = vec![0.0; 48000];

    let path = writer
        .write_clip(&samples, 48000, "Cyanistes caeruleus", 0.9234, 5.0, 8.0)
        .unwrap();

    let filename = path.file_name().unwrap().to_str().unwrap();
    // Format: species_confidence_start-end.wav
    assert!(filename.starts_with("Cyanistes caeruleus_92p_"));
    assert!(filename.ends_with(".wav"));
}

#[test]
fn test_write_clip_sanitizes_species_name() {
    let temp_dir = TempDir::new().unwrap();
    let writer = WavWriter::new(temp_dir.path().to_path_buf());

    let samples: Vec<f32> = vec![0.0; 48000];

    // Species name with characters that need sanitization
    let path = writer
        .write_clip(
            &samples,
            48000,
            "Species/with:special*chars",
            0.80,
            0.0,
            1.0,
        )
        .unwrap();

    assert!(path.exists());
    // Directory name should be sanitized
    let dir_name = path
        .parent()
        .unwrap()
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    assert!(!dir_name.contains('/'));
    assert!(!dir_name.contains(':'));
    assert!(!dir_name.contains('*'));
}

#[test]
fn test_written_wav_is_valid() {
    let temp_dir = TempDir::new().unwrap();
    let writer = WavWriter::new(temp_dir.path().to_path_buf());

    let samples: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.01).sin()).collect();

    let path = writer
        .write_clip(&samples, 48000, "Test Species", 0.85, 0.0, 1.0)
        .unwrap();

    // Verify we can read the WAV back
    let reader = hound::WavReader::open(&path).unwrap();
    let spec = reader.spec();

    assert_eq!(spec.sample_rate, 48000);
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.bits_per_sample, 16);
}
