//! Single file processing pipeline.

use crate::audio::AudioChunk;
use crate::config::OutputFormat;
use crate::error::Result;
use crate::inference::{BatchInferenceContext, BirdClassifier, InferenceOptions};
use crate::locking::FileLock;
use crate::output::{
    AudacityWriter, CsvWriter, Detection, JsonResultWriter, KaleidoscopeWriter, OutputWriter,
    RavenWriter,
};
use crate::pipeline::output_path_for;
use std::path::Path;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};
use tracing::{debug, info};

/// Result type for chunks sent through the decode channel.
type ChunkResult = std::result::Result<AudioChunk, crate::error::Error>;

/// Spawn a thread that decodes audio and sends chunks through the channel.
fn spawn_decode_thread(
    path: std::path::PathBuf,
    source_rate: u32,
    target_rate: u32,
    segment_samples: usize,
    overlap_samples: usize,
    tx: SyncSender<ChunkResult>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let result = decode_and_stream(
            &path,
            source_rate,
            target_rate,
            segment_samples,
            overlap_samples,
            &tx,
        );
        if let Err(e) = result {
            // Send error through channel, ignore if receiver dropped
            let _ = tx.send(Err(e));
        }
        // tx drops here, closing channel
    })
}

/// Decode audio file and stream chunks through the channel.
fn decode_and_stream(
    path: &Path,
    source_rate: u32,
    target_rate: u32,
    segment_samples: usize,
    overlap_samples: usize,
    tx: &SyncSender<ChunkResult>,
) -> Result<()> {
    use crate::audio::{StreamingDecoder, resample_chunk};

    let mut decoder = StreamingDecoder::open(path)?;

    // Calculate source segment size based on rate ratio
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let source_segment_samples = if source_rate == target_rate {
        segment_samples
    } else {
        ((segment_samples as f64) * f64::from(source_rate) / f64::from(target_rate)).ceil() as usize
    };

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let source_overlap_samples = if source_rate == target_rate {
        overlap_samples
    } else {
        ((overlap_samples as f64) * f64::from(source_rate) / f64::from(target_rate)).ceil() as usize
    };

    while let Some(raw) = decoder.next_segment(source_segment_samples, source_overlap_samples)? {
        // Resample to target rate and ensure exact segment length
        let mut samples = resample_chunk(raw.samples, source_rate, target_rate)?;
        samples.resize(segment_samples, 0.0);

        // Calculate time offsets from decoder position (more accurate than index-based)
        #[allow(clippy::cast_precision_loss)]
        let start_time = raw.start_sample as f32 / source_rate as f32;
        #[allow(clippy::cast_precision_loss)]
        let segment_duration = segment_samples as f32 / target_rate as f32;
        let end_time = start_time + segment_duration;

        let chunk = AudioChunk {
            samples,
            start_time,
            end_time,
        };

        // Send chunk, blocks if channel full (backpressure)
        tx.send(Ok(chunk))
            .map_err(|_| crate::error::Error::DecodeChannelClosed)?;
    }

    Ok(())
}

/// Run inference on chunks received from the decode channel.
///
/// Returns detections and the total segment count processed.
#[allow(clippy::too_many_arguments)]
fn run_streaming_inference(
    rx: Receiver<ChunkResult>,
    classifier: &BirdClassifier,
    file_path: &Path,
    min_confidence: f32,
    batch_size: usize,
    progress: Option<&indicatif::ProgressBar>,
    batch_context: &mut Option<BatchInferenceContext>,
    reporter: Option<&dyn crate::output::ProgressReporter>,
    estimated_segments: usize,
    bsg_params: Option<(f64, f64, Option<u32>)>,
) -> Result<(Vec<Detection>, usize)> {
    let mut detections = Vec::new();
    let mut batch: Vec<AudioChunk> = Vec::with_capacity(batch_size);
    let mut segment_count = 0usize;
    let mut segments_done = 0usize;

    for item in rx {
        let chunk = item?; // Propagate decode errors
        batch.push(chunk);
        segment_count += 1;

        if batch.len() >= batch_size {
            process_batch(
                &batch,
                classifier,
                file_path,
                min_confidence,
                &mut detections,
                progress,
                batch_context,
                batch_size,
                reporter,
                &mut segments_done,
                estimated_segments,
                bsg_params,
            )?;
            batch.clear();
        }
    }

    // Process remaining partial batch (padding handled inside process_batch)
    if !batch.is_empty() {
        process_batch(
            &batch,
            classifier,
            file_path,
            min_confidence,
            &mut detections,
            progress,
            batch_context,
            batch_size, // Target size for TensorRT alignment
            reporter,
            &mut segments_done,
            estimated_segments,
            bsg_params,
        )?;
    }

    // Sort by start time, then by confidence (descending)
    // Using unstable sort for performance - stability doesn't matter for detections
    detections.sort_unstable_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    Ok((detections, segment_count))
}

/// Default watchdog timeout for inference operations (in seconds).
/// Can be overridden via `BIRDA_INFERENCE_TIMEOUT` environment variable.
const DEFAULT_INFERENCE_WATCHDOG_SECS: u64 = 10;

/// Minimum and maximum allowed watchdog timeout values.
const MIN_WATCHDOG_SECS: u64 = 1;
const MAX_WATCHDOG_SECS: u64 = 3600;

/// Get the inference watchdog timeout from environment or use default.
///
/// Override with `BIRDA_INFERENCE_TIMEOUT=<seconds>` for different hardware.
/// Normal inference is ~74ms per batch, so 10s default is generous while catching hangs.
/// Valid range: 1-3600 seconds. Invalid values use default.
fn inference_watchdog_timeout() -> u64 {
    std::env::var("BIRDA_INFERENCE_TIMEOUT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| (MIN_WATCHDOG_SECS..=MAX_WATCHDOG_SECS).contains(&v))
        .unwrap_or(DEFAULT_INFERENCE_WATCHDOG_SECS)
}

/// Process a batch of chunks through the classifier.
///
/// # Arguments
///
/// * `target_batch_size` - Target batch size for `TensorRT` alignment (pads with silence if needed)
/// * `bsg_params` - Optional (lat, lon, `day_of_year`) for BSG SDM, `day_of_year=None` for auto-detect
#[allow(clippy::too_many_arguments)]
fn process_batch(
    batch: &[AudioChunk],
    classifier: &BirdClassifier,
    file_path: &Path,
    min_confidence: f32,
    detections: &mut Vec<Detection>,
    progress: Option<&indicatif::ProgressBar>,
    batch_context: &mut Option<BatchInferenceContext>,
    target_batch_size: usize,
    reporter: Option<&dyn crate::output::ProgressReporter>,
    segments_done: &mut usize,
    estimated_segments: usize,
    bsg_params: Option<(f64, f64, Option<u32>)>,
) -> Result<()> {
    use crate::gpu::start_inference_watchdog;
    use crate::output::progress::inc_progress;
    use std::time::Duration;

    let valid_count = batch.len();
    let mut segments: Vec<&[f32]> = batch.iter().map(|c| c.samples.as_slice()).collect();

    // Pad segments with silence for TensorRT batch size alignment (single allocation, no cloning)
    let padding_buffer: Vec<f32>;
    if valid_count < target_batch_size {
        let sample_count = classifier.sample_count();
        padding_buffer = vec![0.0f32; sample_count];
        let padding_needed = target_batch_size - valid_count;
        tracing::debug!(
            "Padding partial batch: {} → {} segments ({} padding)",
            valid_count,
            target_batch_size,
            padding_needed
        );
        segments.extend(std::iter::repeat_n(
            padding_buffer.as_slice(),
            padding_needed,
        ));
    }

    let batch_size = segments.len();

    // Start watchdog timer - kills process if inference hangs
    let _watchdog = start_inference_watchdog(
        Duration::from_secs(inference_watchdog_timeout()),
        batch_size,
    );

    let options = InferenceOptions::default();
    let mut results = if batch_size == 1 {
        vec![classifier.predict(segments[0], &options)?]
    } else if let Some(ctx) = batch_context.as_mut() {
        // Use pre-allocated context for memory-efficient batch inference
        classifier.predict_batch_with_context(ctx, &segments, &options)?
    } else {
        // Fallback for PerchV2 or when context not available
        classifier.predict_batch(&segments, &options)?
    };

    // Watchdog is automatically cancelled when _watchdog drops here

    // Apply BSG post-processing (calibration always, SDM optional)
    // For BSG models, calibration is always applied even without location/date
    // Day-of-year auto-detection happens once per file in process_file()
    if classifier.has_bsg_processor() {
        if let Some((lat, lon, day_of_year)) = bsg_params {
            // Apply BSG post-processing (calibration + SDM if day available)
            #[allow(clippy::cast_possible_truncation)]
            {
                results = results
                    .into_iter()
                    .map(|r| {
                        if let Some(day) = day_of_year {
                            // Apply calibration + SDM
                            classifier.apply_bsg_postprocessing(
                                r,
                                Some(lat as f32),
                                Some(lon as f32),
                                Some(day),
                            )
                        } else {
                            // Apply calibration only (SDM disabled due to missing day-of-year)
                            classifier.apply_bsg_postprocessing(r, None, None, None)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
            }
        } else {
            // No SDM parameters - apply calibration only
            results = results
                .into_iter()
                .map(|r| classifier.apply_bsg_postprocessing(r, None, None, None))
                .collect::<Result<Vec<_>>>()?;
        }
    }

    // Apply range filtering if configured (skipped for BSG models)
    let results = classifier.apply_range_filter(results)?;

    // Only process results from valid segments (excludes padding)
    for (chunk, result) in batch.iter().zip(results.iter()).take(valid_count) {
        for pred in &result.predictions {
            if pred.confidence >= min_confidence {
                let detection = Detection::from_label(
                    &pred.species,
                    pred.confidence,
                    chunk.start_time,
                    chunk.end_time,
                    file_path.to_path_buf(),
                );
                detections.push(detection);
            }
        }
        inc_progress(progress);

        // Report progress via NDJSON reporter if available
        if let Some(reporter) = reporter {
            *segments_done += 1;
            #[allow(clippy::cast_precision_loss)]
            let percent = if estimated_segments > 0 {
                (*segments_done as f32 / estimated_segments as f32 * 100.0).min(100.0)
            } else {
                0.0
            };

            let file_progress = crate::output::json_envelope::FileProgress {
                path: file_path.to_path_buf(),
                segments_done: *segments_done,
                segments_total: estimated_segments,
                percent,
            };

            reporter.progress(None, Some(&file_progress));
        }
    }

    Ok(())
}

/// Process a single audio file and write detection results.
///
/// # Arguments
///
/// * `input_path` - Path to input audio file
/// * `output_dir` - Directory for output files
/// * `classifier` - `BirdNET` classifier for inference
/// * `formats` - Output formats to generate
/// * `min_confidence` - Minimum confidence threshold (0.0-1.0)
/// * `overlap` - Overlap between chunks in seconds
/// * `batch_size` - Number of chunks to process in parallel
/// * `csv_columns` - Additional columns to include in CSV output
/// * `progress_enabled` - Whether to show progress bars
/// * `csv_bom_enabled` - Whether to include UTF-8 BOM in CSV output for Excel compatibility
/// * `model_name` - Model name for JSON output metadata
/// * `range_filter_params` - Optional (lat, lon, week) for JSON output metadata
/// * `bsg_params` - Optional (lat, lon, `day_of_year`) for BSG SDM, `day_of_year=None` for auto-detect
/// * `reporter` - Optional reporter for stdout mode (emits detections instead of writing files)
#[allow(clippy::too_many_arguments)]
pub fn process_file(
    input_path: &Path,
    output_dir: &Path,
    classifier: &BirdClassifier,
    formats: &[OutputFormat],
    min_confidence: f32,
    overlap: f32,
    batch_size: usize,
    csv_columns: &[String],
    progress_enabled: bool,
    csv_bom_enabled: bool,
    model_name: &str,
    range_filter_params: Option<(f64, f64, u8)>,
    bsg_params: Option<(f64, f64, Option<u32>)>,
    reporter: Option<&dyn crate::output::ProgressReporter>,
    dual_output_mode: bool,
) -> Result<ProcessResult> {
    use crate::audio::StreamingDecoder;
    use crate::output::progress::{self, estimate_segment_count};
    use std::time::Instant;

    let start_time = Instant::now();

    info!("Processing: {}", input_path.display());

    // Acquire lock when writing files (file mode or dual output mode)
    let _lock = if reporter.is_none() || dual_output_mode {
        // File mode or dual output mode - need lock to prevent concurrent writes
        Some(FileLock::acquire(input_path, output_dir)?)
    } else {
        // Pure stdout mode - no files written, no lock needed
        None
    };

    // Open decoder to get metadata
    let decoder = StreamingDecoder::open(input_path)?;
    let source_rate = decoder.sample_rate();
    let duration_hint = decoder.duration_hint();
    let target_rate = classifier.sample_rate();
    let segment_duration = classifier.segment_duration();

    // Resolve BSG parameters with day-of-year auto-detection (once per file, not per batch)
    let resolved_bsg_params = if let Some((lat, lon, day_of_year)) = bsg_params {
        // Auto-detect day-of-year if not provided
        let resolved_day = day_of_year.map_or_else(
            || {
                use crate::utils::date::auto_detect_day_of_year;
                match auto_detect_day_of_year(input_path) {
                    Ok(d) => {
                        debug!("Auto-detected day-of-year: {}", d);
                        Some(d)
                    }
                    Err(e) => {
                        tracing::warn!("{}, SDM will not be applied", e);
                        None
                    }
                }
            },
            Some,
        );
        Some((lat, lon, resolved_day))
    } else {
        None
    };

    // Create batch context for GPU memory efficiency (if batch_size > 1)
    // Context is created once and reused for all batches in this file
    let mut batch_context = if batch_size > 1 {
        match classifier.create_batch_context(batch_size) {
            Ok(ctx) => {
                debug!(
                    "Created BatchInferenceContext for up to {} segments ({} bytes input buffer)",
                    batch_size,
                    ctx.input_buffer_bytes()
                );
                Some(ctx)
            }
            Err(e) => {
                // PerchV2 doesn't support BatchInferenceContext - fall back gracefully
                debug!(
                    "BatchInferenceContext not available: {}, using standard predict_batch",
                    e
                );
                None
            }
        }
    } else {
        None
    };

    // Log audio info
    if let Some(duration) = duration_hint {
        info!(
            "Processing ~{} of audio ({:.1}s)",
            progress::format_duration(duration),
            duration
        );
    } else {
        info!("Processing audio (duration unknown)");
    }

    // Calculate segment parameters
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let segment_samples = (segment_duration * target_rate as f32) as usize;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let overlap_samples = (overlap * target_rate as f32) as usize;

    // Estimate segment count for progress bar
    let estimated_segments = estimate_segment_count(duration_hint, segment_duration, overlap);

    // Create progress bar
    let file_name = input_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    // Sanitize filename to prevent template injection (curly braces are special in indicatif)
    let safe_name = file_name.replace(['{', '}'], "");

    #[allow(clippy::cast_possible_truncation)]
    let segment_progress = estimated_segments.map_or_else(
        || {
            // No duration hint - create spinner-style progress
            if progress_enabled {
                let pb = indicatif::ProgressBar::new_spinner();
                pb.set_style(
                    indicatif::ProgressStyle::default_spinner()
                        .template(&format!(
                            "{{spinner:.green}} [{{elapsed_precise}}] {{pos}} segments - {safe_name}"
                        ))
                        .unwrap_or_else(|_| indicatif::ProgressStyle::default_spinner()),
                );
                pb.enable_steady_tick(std::time::Duration::from_millis(100));
                Some(pb)
            } else {
                None
            }
        },
        |est| progress::create_segment_progress(est as usize, file_name, progress_enabled),
    );

    let progress_guard = progress::ProgressGuard::new(segment_progress, "Inference complete");

    // Create channel with capacity for 2 batches (backpressure)
    let channel_capacity = batch_size.saturating_mul(2).max(4);
    let (tx, rx) = sync_channel::<ChunkResult>(channel_capacity);

    // Spawn decode thread
    // Note: We re-open the decoder in the thread since StreamingDecoder
    // has consumed state. This is a minor overhead but keeps ownership clean.
    let path_for_thread = input_path.to_path_buf();
    let decode_handle = spawn_decode_thread(
        path_for_thread,
        source_rate,
        target_rate,
        segment_samples,
        overlap_samples,
        tx,
    );

    // Run inference on main thread
    #[allow(clippy::cast_possible_truncation)]
    let estimated_segments_usize = estimated_segments.unwrap_or(0) as usize;
    let (detections, actual_segments) = run_streaming_inference(
        rx,
        classifier,
        input_path,
        min_confidence,
        batch_size,
        progress_guard.get(),
        &mut batch_context,
        reporter,
        estimated_segments_usize,
        resolved_bsg_params,
    )?;

    // Wait for decode thread to finish
    // Errors are sent through the channel, so we just wait for cleanup
    // If the thread panicked, log a warning (panics indicate bugs, but shouldn't crash batch jobs)
    if let Err(panic_payload) = decode_handle.join() {
        tracing::warn!("Decode thread panicked: {:?}", panic_payload);
    }

    // Finish progress bar
    drop(progress_guard);

    info!(
        "Found {} detections above {:.1}% confidence",
        detections.len(),
        min_confidence * 100.0
    );

    // Prepare JSON config if JSON output is requested
    // Use decoder hint if available, otherwise estimate from processed segments
    let audio_duration_secs = duration_hint.unwrap_or_else(|| {
        // Estimate: segment_duration + (n-1) * (segment_duration - overlap)
        if actual_segments > 0 {
            let seg_dur = f64::from(segment_duration);
            let ovr = f64::from(overlap);
            let non_overlap = seg_dur - ovr;
            #[allow(clippy::cast_precision_loss)]
            let estimated = (actual_segments as f64 - 1.0).mul_add(non_overlap, seg_dur);
            estimated
        } else {
            0.0
        }
    });
    let json_config = if formats.contains(&OutputFormat::Json) {
        #[allow(clippy::cast_possible_truncation)]
        let audio_duration_f32 = audio_duration_secs as f32;
        Some(JsonOutputConfig {
            model: model_name.to_string(),
            min_confidence,
            overlap,
            audio_duration: audio_duration_f32,
            lat: range_filter_params.map(|(lat, _, _)| lat),
            lon: range_filter_params.map(|(_, lon, _)| lon),
            week: range_filter_params.map(|(_, _, week)| week),
        })
    } else {
        None
    };

    // Write output files or emit detections event
    if dual_output_mode {
        // Dual output mode: write files, don't emit detection events
        // (Progress events already sent via reporter)
        for format in formats {
            write_output(
                input_path,
                output_dir,
                *format,
                &detections,
                csv_columns,
                csv_bom_enabled,
                json_config.as_ref(),
            )?;
        }
    } else if let Some(reporter) = reporter {
        // Pure stdout mode - emit detections event instead of writing files

        // Construct BSG metadata if BSG model is used
        let bsg_metadata = if classifier.has_bsg_processor() {
            use crate::output::BsgMetadata;

            if let Some((lat, lon, day_of_year)) = resolved_bsg_params {
                // SDM parameters provided (lat/lon), day may be auto-detected or missing
                #[allow(clippy::cast_possible_truncation)]
                Some(BsgMetadata {
                    calibration_applied: true,
                    sdm_applied: day_of_year.is_some(), // SDM only applied if day available
                    latitude: Some(lat as f32),
                    longitude: Some(lon as f32),
                    day_of_year,
                })
            } else {
                // Calibration-only mode - no SDM parameters provided
                Some(BsgMetadata {
                    calibration_applied: true,
                    sdm_applied: false,
                    latitude: None,
                    longitude: None,
                    day_of_year: None,
                })
            }
        } else {
            None
        };

        reporter.detections(input_path, &detections, bsg_metadata.as_ref());
    } else {
        // Pure file mode - write output files
        for format in formats {
            write_output(
                input_path,
                output_dir,
                *format,
                &detections,
                csv_columns,
                csv_bom_enabled,
                json_config.as_ref(),
            )?;
        }
    }

    let duration_secs = start_time.elapsed().as_secs_f64();

    #[allow(clippy::cast_precision_loss)]
    let segments_per_sec = if duration_secs > 0.0 && actual_segments > 0 {
        actual_segments as f64 / duration_secs
    } else {
        0.0
    };
    let realtime_factor = if duration_secs > 0.0 && audio_duration_secs > 0.0 {
        audio_duration_secs / duration_secs
    } else {
        0.0
    };

    info!(
        "Processed {} segments in {:.2}s ({:.1} segments/sec, {:.1}x realtime)",
        actual_segments, duration_secs, segments_per_sec, realtime_factor
    );

    Ok(ProcessResult {
        detections: detections.len(),
        segments: actual_segments,
        duration_secs,
        audio_duration_secs,
    })
}

/// Configuration for JSON output writer.
#[derive(Debug, Clone)]
pub struct JsonOutputConfig {
    /// Model name used for analysis.
    pub model: String,
    /// Minimum confidence threshold.
    pub min_confidence: f32,
    /// Segment overlap.
    pub overlap: f32,
    /// Audio file duration in seconds.
    pub audio_duration: f32,
    /// Latitude for range filtering.
    pub lat: Option<f64>,
    /// Longitude for range filtering.
    pub lon: Option<f64>,
    /// Week for range filtering.
    pub week: Option<u8>,
}

/// Write detections to an output file.
#[allow(clippy::too_many_arguments)]
fn write_output(
    input_path: &Path,
    output_dir: &Path,
    format: OutputFormat,
    detections: &[Detection],
    csv_columns: &[String],
    csv_bom_enabled: bool,
    json_config: Option<&JsonOutputConfig>,
) -> Result<()> {
    let output_path = output_path_for(input_path, output_dir, format);
    debug!("Writing {} output: {}", format, output_path.display());

    let mut writer: Box<dyn OutputWriter> = match format {
        OutputFormat::Csv => Box::new(CsvWriter::new(
            &output_path,
            csv_columns.to_vec(),
            csv_bom_enabled,
        )?),
        OutputFormat::Raven => Box::new(RavenWriter::new(&output_path)?),
        OutputFormat::Audacity => Box::new(AudacityWriter::new(&output_path)?),
        OutputFormat::Kaleidoscope => Box::new(KaleidoscopeWriter::new(&output_path)?),
        OutputFormat::Json => {
            let source_file = input_path.file_name().map_or_else(
                || "unknown".to_string(),
                |n| n.to_string_lossy().to_string(),
            );

            // JsonOutputConfig must be provided when JSON format is requested
            let config = json_config.ok_or_else(|| crate::error::Error::Internal {
                message: "JsonOutputConfig required for JSON format".to_string(),
            })?;

            Box::new(JsonResultWriter::new(
                &output_path,
                &source_file,
                config.audio_duration,
                &config.model,
                config.min_confidence,
                config.overlap,
                config.lat,
                config.lon,
                config.week,
            )?)
        }
    };

    writer.write_header()?;
    for detection in detections {
        writer.write_detection(detection)?;
    }
    writer.finalize()?;

    Ok(())
}

/// Result of processing a single file.
#[derive(Debug)]
pub struct ProcessResult {
    /// Number of detections found.
    pub detections: usize,
    /// Number of segments processed.
    pub segments: usize,
    /// Processing duration in seconds.
    pub duration_secs: f64,
    /// Audio duration in seconds.
    pub audio_duration_secs: f64,
}
