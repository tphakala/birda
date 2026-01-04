//! Single file processing pipeline.

use crate::audio::{AudioChunk, chunk_audio, decode_audio_file, resample};
use crate::config::OutputFormat;
use crate::error::Result;
use crate::inference::BirdClassifier;
use crate::locking::FileLock;
use crate::output::{
    AudacityWriter, CsvWriter, Detection, KaleidoscopeWriter, OutputWriter, RavenWriter,
};
use crate::pipeline::output_path_for;
use indicatif::MultiProgress;
use std::path::Path;
use tracing::{debug, info};

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
/// * `multi_progress` - `MultiProgress` for managing progress bars
/// * `progress_enabled` - Whether to show progress bars
/// * `csv_bom_enabled` - Whether to include UTF-8 BOM in CSV output for Excel compatibility
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
    multi_progress: &MultiProgress,
    progress_enabled: bool,
    csv_bom_enabled: bool,
) -> Result<ProcessResult> {
    use crate::output::progress;
    use std::time::Instant;

    let start_time = Instant::now();

    info!("Processing: {}", input_path.display());

    // Acquire lock
    let _lock = FileLock::acquire(input_path, output_dir)?;

    // Decode audio
    info!("Decoding audio...");
    let decoded = decode_audio_file(input_path)?;
    let audio_duration_secs = decoded.duration_secs;
    info!(
        "Decoded {} of audio ({:.1}s)",
        progress::format_duration(audio_duration_secs),
        audio_duration_secs
    );

    // Resample to model's expected sample rate
    let target_rate = classifier.sample_rate();
    let samples = if decoded.sample_rate == target_rate {
        decoded.samples
    } else {
        debug!(
            "Resampling from {} Hz to {} Hz...",
            decoded.sample_rate, target_rate
        );
        resample(decoded.samples, decoded.sample_rate, target_rate)?
    };

    // Chunk audio into segments
    let segment_duration = classifier.segment_duration();
    debug!(
        "Chunking into {:.1}s segments with {:.1}s overlap...",
        segment_duration, overlap
    );
    let chunks = chunk_audio(&samples, target_rate, segment_duration, overlap);

    if chunks.is_empty() {
        info!("No segments to process (audio too short)");
        let duration_secs = start_time.elapsed().as_secs_f64();
        return Ok(ProcessResult {
            detections: 0,
            segments: 0,
            duration_secs,
            audio_duration_secs,
        });
    }

    // Create segment progress bar
    let file_name = input_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    let segment_progress = if progress_enabled {
        progress::create_segment_progress(chunks.len(), file_name, true)
            .map(|pb| multi_progress.add(pb))
    } else {
        None
    };

    // Wrap in guard to ensure cleanup on both success and error
    let multi_progress_opt = segment_progress.as_ref().map(|_| multi_progress.clone());
    let progress_guard =
        progress::ProgressGuard::new(segment_progress, multi_progress_opt, "Inference complete");

    // Run inference
    debug!("Running inference on {} segments...", chunks.len());
    let detections = run_inference(
        &chunks,
        classifier,
        input_path,
        min_confidence,
        batch_size,
        progress_guard.get(),
    )?;

    // Guard will automatically finish progress bar when dropped here

    info!(
        "Found {} detections above {:.1}% confidence",
        detections.len(),
        min_confidence * 100.0
    );

    // Write output files
    for format in formats {
        write_output(
            input_path,
            output_dir,
            *format,
            &detections,
            csv_columns,
            csv_bom_enabled,
        )?;
    }

    let duration_secs = start_time.elapsed().as_secs_f64();
    #[allow(clippy::cast_precision_loss)]
    let segments_per_sec = if duration_secs > 0.0 {
        chunks.len() as f64 / duration_secs
    } else {
        0.0
    };
    let realtime_factor = if duration_secs > 0.0 {
        f64::from(audio_duration_secs) / duration_secs
    } else {
        0.0
    };
    info!(
        "Processed {} segments in {:.2}s ({:.1} segments/sec, {:.1}x realtime)",
        chunks.len(),
        duration_secs,
        segments_per_sec,
        realtime_factor
    );

    Ok(ProcessResult {
        detections: detections.len(),
        segments: chunks.len(),
        duration_secs,
        audio_duration_secs,
    })
}

/// Run inference on audio chunks.
fn run_inference(
    chunks: &[AudioChunk],
    classifier: &BirdClassifier,
    file_path: &Path,
    min_confidence: f32,
    batch_size: usize,
    segment_progress: Option<&indicatif::ProgressBar>,
) -> Result<Vec<Detection>> {
    use crate::output::progress;
    let mut detections = Vec::new();

    // Process in batches
    for batch_chunks in chunks.chunks(batch_size) {
        let segments: Vec<&[f32]> = batch_chunks.iter().map(|c| c.samples.as_slice()).collect();

        let results = if segments.len() == 1 {
            vec![classifier.predict(segments[0])?]
        } else {
            classifier.predict_batch(&segments)?
        };

        // Apply range filtering if configured
        let results = classifier.apply_range_filter(results)?;

        for (chunk, result) in batch_chunks.iter().zip(results.iter()) {
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
            // Increment progress for each segment processed
            progress::inc_progress(segment_progress);
        }
    }

    // Sort by start time, then by confidence (descending)
    detections.sort_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    Ok(detections)
}

/// Write detections to an output file.
fn write_output(
    input_path: &Path,
    output_dir: &Path,
    format: OutputFormat,
    detections: &[Detection],
    csv_columns: &[String],
    csv_bom_enabled: bool,
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
    pub audio_duration_secs: f32,
}
