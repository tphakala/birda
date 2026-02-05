//! Clip command execution.

use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use tracing::{info, warn};

use crate::Error;
use crate::cli::ClipArgs;
use crate::config::OutputMode;
use crate::constants::{clipper, output_extensions};
use crate::output::{ClipExtractionEntry, ClipExtractionPayload, ResultType, emit_json_result};

use super::{
    ClipExtractor, DetectionGroup, ParsedDetection, WavWriter, group_detections,
    parse_detection_file,
};

/// Execute the clip command.
///
/// # Errors
///
/// Returns an error if clip extraction fails.
pub fn execute(args: &ClipArgs, output_mode: OutputMode) -> Result<(), Error> {
    // Detect mode based on presence of --start/--end
    if let (Some(start), Some(end)) = (args.start, args.end) {
        execute_direct_extraction(args, start, end, output_mode)
    } else {
        execute_csv_mode(args, output_mode)
    }
}

/// Execute clip extraction from CSV detection files.
#[allow(clippy::unnecessary_wraps)]
fn execute_csv_mode(args: &ClipArgs, output_mode: OutputMode) -> Result<(), Error> {
    let extractor = ClipExtractor::new();
    let writer = WavWriter::new(args.output.clone());
    let is_json = output_mode.is_structured();

    let mut total_clips = 0;
    let mut total_files = 0;
    let mut all_clips: Vec<ClipExtractionEntry> = Vec::new();

    for detection_file in &args.files {
        match process_detection_file(detection_file, args, &extractor, &writer, is_json) {
            Ok((clip_count, clips)) => {
                total_clips += clip_count;
                total_files += 1;
                all_clips.extend(clips);
            }
            Err(e) => {
                warn!("Failed to process {}: {e}", detection_file.display());
            }
        }
    }

    // JSON/NDJSON output
    if is_json {
        let payload = ClipExtractionPayload {
            result_type: ResultType::ClipExtraction,
            output_dir: args.output.clone(),
            total_clips,
            total_files,
            clips: all_clips,
        };
        emit_json_result(&payload);
        return Ok(());
    }

    // Human-readable output
    info!(
        "Extracted {total_clips} clips from {total_files} detection files to {}",
        args.output.display()
    );

    Ok(())
}

/// Execute direct clip extraction from time range.
fn execute_direct_extraction(
    args: &ClipArgs,
    start: f64,
    end: f64,
    output_mode: OutputMode,
) -> Result<(), Error> {
    // Validation
    if end <= start {
        return Err(Error::InvalidTimeRange { start, end });
    }

    // audio is guaranteed by clap constraints
    let audio_path = args.audio.as_ref().ok_or_else(|| Error::Internal {
        message: "audio path required in direct extraction mode".to_string(),
    })?;

    if !audio_path.exists() {
        return Err(Error::SourceAudioNotFound {
            detection_path: PathBuf::new(),
            audio_path: audio_path.clone(),
        });
    }

    // Apply padding
    let padded_start = (start - args.pre).max(0.0);
    let padded_end = end + args.post;

    // Create synthetic DetectionGroup for extraction
    let group = DetectionGroup {
        scientific_name: format!("detection_{start:.0}-{end:.0}"),
        common_name: String::new(), // Empty for generic clips
        start: padded_start,
        end: padded_end,
        max_confidence: 1.0, // No confidence for direct extraction
        detection_count: 1,
    };

    // Extract and write clip
    let extractor = ClipExtractor::new();
    let writer = WavWriter::new(args.output.clone());

    let clip = extractor.extract_clip(audio_path, &group)?;
    let output_path = writer.write_clip(
        &clip.samples,
        clip.sample_rate,
        &group.scientific_name,
        group.max_confidence,
        padded_start,
        padded_end,
    )?;

    // Output handling
    if output_mode.is_structured() {
        // JSON/NDJSON output
        let payload = ClipExtractionPayload {
            result_type: ResultType::ClipExtraction,
            output_dir: args.output.clone(),
            total_clips: 1,
            total_files: 1,
            clips: vec![ClipExtractionEntry {
                source_audio: audio_path.clone(),
                scientific_name: group.scientific_name,
                confidence: group.max_confidence,
                start_time: padded_start,
                end_time: padded_end,
                output_file: output_path,
            }],
        };
        emit_json_result(&payload);
    } else {
        // Human-readable: print only the clip path to stdout
        println!("{}", output_path.display());
    }

    Ok(())
}

fn process_detection_file(
    detection_file: &Path,
    args: &ClipArgs,
    extractor: &ClipExtractor,
    writer: &WavWriter,
    is_json: bool,
) -> Result<(usize, Vec<ClipExtractionEntry>), Error> {
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
        return Ok((0, Vec::new()));
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

    // Create progress bar for clip extraction (only in human mode)
    #[allow(clippy::cast_possible_truncation)]
    let pb = if is_json {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(groups.len() as u64);
        // Template is hardcoded and known to be valid
        #[allow(clippy::expect_used)]
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} clips ({msg})")
                .expect("valid progress template")
                .progress_chars("#>-"),
        );
        pb
    };

    // Extract and write clips
    let mut clip_count = 0;
    let mut clip_entries: Vec<ClipExtractionEntry> = Vec::new();

    for group in &groups {
        pb.set_message(group.scientific_name.clone());

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
                        // Record clip entry for JSON output
                        clip_entries.push(ClipExtractionEntry {
                            source_audio: audio_path.clone(),
                            scientific_name: group.scientific_name.clone(),
                            confidence: group.max_confidence,
                            start_time: group.start,
                            end_time: group.end,
                            output_file: path.clone(),
                        });

                        if !is_json {
                            // Use pb.println to avoid progress bar stuttering
                            pb.println(format!(
                                "  {} ({:.0}%): {:.1}s-{:.1}s -> {}",
                                group.scientific_name,
                                group.max_confidence * 100.0,
                                group.start,
                                group.end,
                                path.file_name().unwrap_or_default().to_string_lossy()
                            ));
                        }
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

        pb.inc(1);
    }

    pb.finish_with_message("done");

    Ok((clip_count, clip_entries))
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

    // Common suffixes to strip (use constants from output_extensions)
    let suffixes = [
        output_extensions::CSV,
        output_extensions::RAVEN,
        output_extensions::AUDACITY,
        output_extensions::KALEIDOSCOPE,
        output_extensions::JSON,
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
        .strip_suffix(clipper::BIRDNET_RESULTS_SUFFIX)
        .or_else(|| stem.strip_suffix(clipper::BIRDNET_SUFFIX))
        .unwrap_or(stem);

    // If clean_stem has an audio extension (e.g., "recording.wav"), strip it
    // This handles edge cases like recording.wav.BirdNET.results.csv -> recording.flac
    let base_stem = clipper::AUDIO_EXTENSIONS
        .iter()
        .find_map(|ext| clean_stem.strip_suffix(&format!(".{ext}")))
        .unwrap_or(clean_stem);

    // Prevent path traversal: reject stems containing ".." or path separators
    if base_stem.contains("..") || base_stem.contains('/') || base_stem.contains('\\') {
        return Err(Error::SourceAudioNotFound {
            detection_path: detection_file.to_path_buf(),
            audio_path: search_dir.join(base_stem),
        });
    }

    for ext in clipper::AUDIO_EXTENSIONS {
        let audio_path = search_dir.join(format!("{base_stem}.{ext}"));
        if audio_path.exists() {
            return Ok(audio_path);
        }
    }

    Err(Error::SourceAudioNotFound {
        detection_path: detection_file.to_path_buf(),
        audio_path: search_dir.join(base_stem),
    })
}
