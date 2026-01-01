//! Single file processing pipeline.

use crate::audio::{chunk_audio, decode_audio_file, resample, AudioChunk};
use crate::config::OutputFormat;
use crate::error::Result;
use crate::inference::BirdClassifier;
use crate::locking::FileLock;
use crate::output::{
    AudacityWriter, CsvWriter, Detection, KaleidoscopeWriter, OutputWriter, RavenWriter,
};
use crate::pipeline::output_path_for;
use std::path::Path;
use tracing::{debug, info};

/// Process a single audio file and write detection results.
pub fn process_file(
    input_path: &Path,
    output_dir: &Path,
    classifier: &BirdClassifier,
    formats: &[OutputFormat],
    min_confidence: f32,
    overlap: f32,
    batch_size: usize,
    csv_columns: Vec<String>,
) -> Result<ProcessResult> {
    info!("Processing: {}", input_path.display());

    // Acquire lock
    let _lock = FileLock::acquire(input_path, output_dir)?;

    // Decode audio
    debug!("Decoding audio...");
    let decoded = decode_audio_file(input_path)?;

    // Resample to model's expected sample rate
    let target_rate = classifier.sample_rate();
    let samples = if decoded.sample_rate != target_rate {
        debug!(
            "Resampling from {} Hz to {} Hz...",
            decoded.sample_rate, target_rate
        );
        resample(decoded.samples, decoded.sample_rate, target_rate)?
    } else {
        decoded.samples
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
        return Ok(ProcessResult {
            detections: 0,
            segments: 0,
        });
    }

    // Run inference
    debug!("Running inference on {} segments...", chunks.len());
    let detections = run_inference(
        &chunks,
        classifier,
        input_path,
        min_confidence,
        batch_size,
    )?;

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
            &csv_columns,
        )?;
    }

    Ok(ProcessResult {
        detections: detections.len(),
        segments: chunks.len(),
    })
}

/// Run inference on audio chunks.
fn run_inference(
    chunks: &[AudioChunk],
    classifier: &BirdClassifier,
    file_path: &Path,
    min_confidence: f32,
    batch_size: usize,
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();

    // Process in batches
    for batch_chunks in chunks.chunks(batch_size) {
        let segments: Vec<&[f32]> = batch_chunks.iter().map(|c| c.samples.as_slice()).collect();

        let results = if segments.len() == 1 {
            vec![classifier.predict(segments[0])?]
        } else {
            classifier.predict_batch(&segments)?
        };

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
) -> Result<()> {
    let output_path = output_path_for(input_path, output_dir, format);
    debug!("Writing {} output: {}", format, output_path.display());

    let mut writer: Box<dyn OutputWriter> = match format {
        OutputFormat::Csv => Box::new(CsvWriter::new(&output_path, csv_columns.to_vec())?),
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
}
