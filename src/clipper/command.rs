//! Clip command execution.

use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use tracing::{info, warn};

use crate::Error;
use crate::cli::ClipArgs;
use crate::config::OutputMode;
use crate::constants::{clipper, confidence, output_extensions};
use crate::output::{ClipExtractionEntry, ClipExtractionPayload, ResultType, emit_json_result};

use super::{
    ClipExtractor, DetectionGroup, ParsedDetection, WavWriter, group_detections,
    parse_detection_file, validate_time_range,
};

/// Execute the clip command.
///
/// # Errors
///
/// Returns [`Error::InvalidPadding`] or [`Error::InvalidConfidence`] if the
/// caller built `args` outside clap with a value the CLI would have refused,
/// [`Error::InvalidTimeRange`] for a bad direct-extraction range, and an error
/// if clip extraction fails.
pub fn execute(args: &ClipArgs, output_mode: OutputMode) -> Result<(), Error> {
    validate_float_args(args)?;

    // Detect mode based on presence of --start/--end
    if let (Some(start), Some(end)) = (args.start, args.end) {
        execute_direct_extraction(args, start, end, output_mode)
    } else {
        execute_csv_mode(args, output_mode)
    }
}

/// Re-check the float arguments at the library boundary.
///
/// `ClipArgs` is public and so is this module, so a caller can build one
/// without going through clap's `value_parser`s. Every value checked here
/// fails quietly rather than loudly when it is not finite, which is why the
/// check is worth repeating rather than delegated to the CLI:
///
/// - a NaN `pre` collapses the start bound to 0.0, because `f64::max` returns
///   the other operand for a NaN receiver, so `(start - pre).max(0.0)` yields
///   the beginning of the file however late the detection was;
/// - a NaN `post` leaves the end bound NaN, which the seconds-to-samples cast
///   turns into 0;
/// - every comparison against a NaN `confidence` is false, so the filter in
///   `process_detection_file` discards every detection and the run reports
///   success over an empty result.
///
/// The bounds are deliberately the same ones `cli::clip`'s parsers and
/// `cli::validators::parse_confidence` enforce. #306 was filed because a rule
/// held on one route and not on another, and this is the other route.
fn validate_float_args(args: &ClipArgs) -> Result<(), Error> {
    // The negated `contains` is what rejects NaN and infinity, and it has to
    // be spelled this way round: a bare `value < 0.0 || value > MAX` is false
    // for NaN on both halves. Same spelling as `cli::validators`.
    for value in [args.pre, args.post] {
        if !(0.0..=clipper::MAX_PADDING).contains(&value) {
            return Err(Error::InvalidPadding { value });
        }
    }

    if !(confidence::MIN..=confidence::MAX).contains(&args.confidence) {
        return Err(Error::InvalidConfidence {
            value: args.confidence,
        });
    }

    Ok(())
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
    validate_time_range(start, end)?;

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
        // Template is hardcoded and known to be valid, so the only way this
        // expect fires is a typo in the literal above, which the tests catch.
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
        output_extensions::PARQUET,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A `ClipArgs` with every value in range, for the library-boundary tests
    /// to perturb one field at a time. Built by hand rather than through clap
    /// on purpose: the check under test exists precisely for callers that skip
    /// clap.
    fn valid_args() -> ClipArgs {
        ClipArgs {
            files: Vec::new(),
            output: PathBuf::from("clips"),
            confidence: 0.0,
            pre: clipper::DEFAULT_PRE_PADDING,
            post: clipper::DEFAULT_POST_PADDING,
            audio: None,
            base_dir: None,
            start: None,
            end: None,
        }
    }

    #[test]
    fn test_validate_float_args_accepts_the_defaults() {
        assert!(validate_float_args(&valid_args()).is_ok());
    }

    #[test]
    fn test_validate_float_args_rejects_non_finite_padding() {
        // Not reachable through clap, which rejects these in its
        // `value_parser`. Reachable through the library, which is the point.
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            for mutate in [
                (|a: &mut ClipArgs, v: f64| a.pre = v) as fn(&mut ClipArgs, f64),
                |a: &mut ClipArgs, v: f64| a.post = v,
            ] {
                let mut args = valid_args();
                mutate(&mut args, value);
                assert!(
                    matches!(
                        validate_float_args(&args),
                        Err(Error::InvalidPadding { .. })
                    ),
                    "padding {value} was not rejected"
                );
            }
        }
    }

    #[test]
    fn test_validate_float_args_applies_the_same_padding_ceiling_as_the_cli() {
        // The CLI parser caps padding at `MAX_PADDING`. A library caller that
        // skipped clap used to get no ceiling at all, which is the #306 shape:
        // one rule, two routes, two answers.
        let mut args = valid_args();
        args.pre = clipper::MAX_PADDING;
        assert!(
            validate_float_args(&args).is_ok(),
            "the ceiling is inclusive"
        );

        args.pre = clipper::MAX_PADDING + 1.0;
        assert!(matches!(
            validate_float_args(&args),
            Err(Error::InvalidPadding { .. })
        ));
    }

    #[test]
    fn test_validate_float_args_rejects_an_out_of_range_confidence() {
        // A NaN confidence makes `d.confidence >= args.confidence` false for
        // every detection, so the run discards them all and still exits 0.
        for value in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            let mut args = valid_args();
            args.confidence = value;
            assert!(
                matches!(
                    validate_float_args(&args),
                    Err(Error::InvalidConfidence { .. })
                ),
                "confidence {value} was not rejected"
            );
        }
    }

    /// Both guards run before any I/O, so `execute` can be driven with no
    /// fixture at all. These exist because the five tests around them call
    /// `validate_float_args` directly, which covers the rule and says nothing
    /// about whether anything calls it: deleting either `?` in `execute` left
    /// the whole suite green.
    #[test]
    fn test_execute_applies_the_float_guard() {
        let mut args = valid_args();
        args.pre = f64::NAN;
        assert!(matches!(
            execute(&args, OutputMode::Human),
            Err(Error::InvalidPadding { .. })
        ));

        let mut args = valid_args();
        args.confidence = f32::NAN;
        assert!(matches!(
            execute(&args, OutputMode::Human),
            Err(Error::InvalidConfidence { .. })
        ));
    }

    #[test]
    fn test_execute_applies_the_range_guard() {
        // An inverted range, which is also the CLI-route case no test drove:
        // clap accepts each bound on its own and only the command sees the
        // pair.
        let mut args = valid_args();
        args.audio = Some(PathBuf::from("/nonexistent/birda-command-test.wav"));
        args.start = Some(5.0);
        args.end = Some(1.0);
        assert!(matches!(
            execute(&args, OutputMode::Human),
            Err(Error::InvalidTimeRange { .. })
        ));

        args.end = Some(f64::INFINITY);
        assert!(matches!(
            execute(&args, OutputMode::Human),
            Err(Error::InvalidTimeRange { .. })
        ));

        // Finite and increasing, and still wrong. `parse_time` refuses a
        // negative start, and until the same rule reached the shared helper a
        // library caller with this got seconds 0 to 4 back, under a name
        // claiming -100 to -1, because `(start - pre).max(0.0)` clamps.
        args.start = Some(-100.0);
        args.end = Some(-1.0);
        assert!(matches!(
            execute(&args, OutputMode::Human),
            Err(Error::InvalidTimeRange { .. })
        ));
    }

    /// The rule `validate_float_args` claims to share with the CLI, enforced
    /// rather than asserted in prose. The two spell it differently
    /// (`confidence::MIN..=MAX` here, the literals `0.0..=1.0` in
    /// `parse_confidence`), so only a differential test keeps them together.
    #[test]
    fn test_the_confidence_rule_matches_the_cli_parser() {
        for value in ["-0.1", "0.0", "0.5", "1.0", "1.1", "nan", "inf", "-inf"] {
            let cli_ok = crate::cli::validators::parse_confidence(value).is_ok();

            let mut args = valid_args();
            args.confidence = value.parse().unwrap_or(f32::NAN);
            let library_ok = validate_float_args(&args).is_ok();

            assert_eq!(cli_ok, library_ok, "the two routes disagree on {value}");
        }
    }

    #[test]
    fn test_validate_float_args_accepts_the_confidence_bounds() {
        for value in [confidence::MIN, confidence::MAX] {
            let mut args = valid_args();
            args.confidence = value;
            assert!(
                validate_float_args(&args).is_ok(),
                "{value} should be legal"
            );
        }
    }
}
