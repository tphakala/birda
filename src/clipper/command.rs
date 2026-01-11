//! Clip command execution.

use std::path::{Path, PathBuf};

use tracing::{info, warn};

use crate::Error;
use crate::cli::ClipArgs;

use super::{ClipExtractor, ParsedDetection, WavWriter, group_detections, parse_detection_file};

/// Execute the clip command.
///
/// # Errors
///
/// Returns an error if clip extraction fails.
pub fn execute(args: &ClipArgs) -> Result<(), Error> {
    let extractor = ClipExtractor::new(args.pre, args.post);
    let writer = WavWriter::new(args.output.clone());

    let mut total_clips = 0;
    let mut total_files = 0;

    for detection_file in &args.files {
        match process_detection_file(detection_file, args, &extractor, &writer) {
            Ok(clip_count) => {
                total_clips += clip_count;
                total_files += 1;
            }
            Err(e) => {
                warn!("Failed to process {}: {e}", detection_file.display());
            }
        }
    }

    info!(
        "Extracted {total_clips} clips from {total_files} detection files to {}",
        args.output.display()
    );

    Ok(())
}

fn process_detection_file(
    detection_file: &Path,
    args: &ClipArgs,
    extractor: &ClipExtractor,
    writer: &WavWriter,
) -> Result<usize, Error> {
    info!("Processing {}", detection_file.display());

    // Parse detections
    let detections = parse_detection_file(detection_file)?;

    // Filter by confidence
    let filtered: Vec<ParsedDetection> = detections
        .into_iter()
        .filter(|d| d.confidence >= args.confidence)
        .collect();

    if filtered.is_empty() {
        info!(
            "No detections above confidence threshold {} in {}",
            args.confidence,
            detection_file.display()
        );
        return Ok(0);
    }

    info!(
        "Found {} detections above threshold {}",
        filtered.len(),
        args.confidence
    );

    // Group detections
    let groups = group_detections(filtered, args.pre, args.post);

    info!("Grouped into {} clips", groups.len());

    // Find source audio file
    let audio_path =
        find_source_audio(detection_file, args.audio.as_ref(), args.base_dir.as_ref())?;

    info!("Using source audio: {}", audio_path.display());

    // Extract and write clips
    let mut clip_count = 0;

    for group in &groups {
        match extractor.extract_clip(&audio_path, group) {
            Ok(clip) => {
                match writer.write_clip(
                    &clip.samples,
                    clip.sample_rate,
                    &group.scientific_name,
                    group.max_confidence,
                    group.start,
                    group.end,
                ) {
                    Ok(path) => {
                        info!(
                            "  {} ({:.0}%): {:.1}s-{:.1}s -> {}",
                            group.scientific_name,
                            group.max_confidence * 100.0,
                            group.start,
                            group.end,
                            path.file_name().unwrap_or_default().to_string_lossy()
                        );
                        clip_count += 1;
                    }
                    Err(e) => {
                        warn!("Failed to write clip: {e}");
                    }
                }
            }
            Err(e) => {
                warn!(
                    "Failed to extract clip for {} at {:.1}s-{:.1}s: {e}",
                    group.scientific_name, group.start, group.end
                );
            }
        }
    }

    Ok(clip_count)
}

/// Find the source audio file for a detection file.
///
/// Resolution order:
/// 1. Explicit --audio path if provided
/// 2. Infer from detection filename in --base-dir (if provided)
/// 3. Infer from detection filename in detection file's directory
fn find_source_audio(
    detection_file: &Path,
    explicit_audio: Option<&PathBuf>,
    base_dir: Option<&PathBuf>,
) -> Result<PathBuf, Error> {
    if let Some(audio_path) = explicit_audio {
        if audio_path.exists() {
            return Ok(audio_path.clone());
        }
        return Err(Error::SourceAudioNotFound {
            detection_path: detection_file.to_path_buf(),
            audio_path: audio_path.clone(),
        });
    }

    // Try to infer audio file from detection filename
    // Detection files are named: original.wav.BirdNET.results.csv
    // or: original.flac.BirdNET.results.csv
    let file_name = detection_file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // Common suffixes to strip
    let suffixes = [
        ".BirdNET.results.csv",
        ".BirdNET.selection.table.txt",
        ".BirdNET.results.txt",
        ".BirdNET.results.kaleidoscope.csv",
    ];

    // Determine search directory: --base-dir if provided, otherwise detection file's parent
    let search_dir = base_dir.map_or_else(
        || detection_file.parent().unwrap_or_else(|| Path::new(".")),
        PathBuf::as_path,
    );

    for suffix in suffixes {
        if let Some(base) = file_name.strip_suffix(suffix) {
            let audio_path = search_dir.join(base);
            if audio_path.exists() {
                return Ok(audio_path);
            }
        }
    }

    // Try common audio extensions
    let stem = detection_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    // Remove any remaining ".BirdNET" or similar suffixes from stem
    let clean_stem = stem
        .strip_suffix(".BirdNET.results")
        .or_else(|| stem.strip_suffix(".BirdNET"))
        .unwrap_or(stem);

    for ext in ["wav", "flac", "mp3", "ogg", "m4a"] {
        let audio_path = search_dir.join(format!("{clean_stem}.{ext}"));
        if audio_path.exists() {
            return Ok(audio_path);
        }
    }

    Err(Error::SourceAudioNotFound {
        detection_path: detection_file.to_path_buf(),
        audio_path: search_dir.join(clean_stem),
    })
}
