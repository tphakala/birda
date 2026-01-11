//! Audio clip extraction with seeking support.
//!
//! Extracts audio segments from source files using Symphonia's seeking
//! for efficient random-access extraction of clips from large files.

use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::{FormatOptions, SeekMode, SeekTo};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::units::Time;

use crate::Error;
use crate::constants::clipper::SEEK_THRESHOLD_SECS;

use super::DetectionGroup;

/// Result of clip extraction.
pub struct ExtractedClip {
    /// Audio samples as f32 (-1.0 to 1.0).
    pub samples: Vec<f32>,
    /// Sample rate of the extracted audio.
    pub sample_rate: u32,
}

/// Extracts audio clips from source files.
pub struct ClipExtractor {
    /// Pre-padding in seconds (applied during grouping, not here).
    #[allow(dead_code)]
    pre_padding: f64,
    /// Post-padding in seconds (applied during grouping, not here).
    #[allow(dead_code)]
    post_padding: f64,
}

impl ClipExtractor {
    /// Create a new clip extractor with the given padding settings.
    ///
    /// Note: Padding is applied during detection grouping, not during extraction.
    #[must_use]
    pub fn new(pre_padding: f64, post_padding: f64) -> Self {
        Self {
            pre_padding,
            post_padding,
        }
    }

    /// Extract a clip from the source audio file.
    ///
    /// Uses seeking for efficient extraction when the clip starts beyond
    /// the seek threshold. Falls back to sequential decoding for early clips.
    ///
    /// # Arguments
    ///
    /// * `source_path` - Path to the source audio file
    /// * `group` - Detection group with start/end times (already includes padding)
    ///
    /// # Errors
    ///
    /// Returns an error if the audio file cannot be read or decoded.
    pub fn extract_clip(
        &self,
        source_path: &Path,
        group: &DetectionGroup,
    ) -> Result<ExtractedClip, Error> {
        // Open the audio file
        let file = std::fs::File::open(source_path).map_err(|e| Error::AudioOpen {
            path: source_path.to_path_buf(),
            source: Box::new(e),
        })?;

        let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

        // Probe the format
        let hint = Hint::new();
        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| Error::AudioDecode {
                path: source_path.to_path_buf(),
                source: Box::new(e),
            })?;

        let mut format = probed.format;

        // Get the audio track
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or_else(|| Error::NoAudioTracks {
                path: source_path.to_path_buf(),
            })?;

        let track_id = track.id;
        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| Error::AudioDecode {
                path: source_path.to_path_buf(),
                source: "missing sample rate".into(),
            })?;

        // Calculate sample positions
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let start_sample = (group.start * f64::from(sample_rate)) as u64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let end_sample = (group.end * f64::from(sample_rate)) as u64;
        #[allow(clippy::cast_possible_truncation)]
        let expected_samples = (end_sample - start_sample) as usize;

        // Create decoder
        let decoder_opts = DecoderOptions::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .map_err(|e| Error::AudioDecode {
                path: source_path.to_path_buf(),
                source: Box::new(e),
            })?;

        // Attempt to seek if start time is beyond threshold.
        // If seek fails, fall back to sequential decoding from start.
        let mut current_sample: u64 = 0;
        if group.start >= SEEK_THRESHOLD_SECS
            && let Ok(seeked_to) = format.seek(
                SeekMode::Coarse,
                SeekTo::Time {
                    time: Time::from(group.start),
                    track_id: Some(track_id),
                },
            )
        {
            // Update current position based on seek result
            current_sample = seeked_to.actual_ts;
            // Reset decoder state after seek
            decoder.reset();
        }

        let mut samples = Vec::with_capacity(expected_samples);

        // Decode and extract samples
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => {
                    return Err(Error::AudioDecode {
                        path: source_path.to_path_buf(),
                        source: Box::new(e),
                    });
                }
            };

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = match decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
                Err(e) => {
                    return Err(Error::AudioDecode {
                        path: source_path.to_path_buf(),
                        source: Box::new(e),
                    });
                }
            };

            let spec = *decoded.spec();
            #[allow(clippy::cast_possible_truncation)]
            let num_frames = decoded.frames() as u64;
            let packet_end_sample = current_sample + num_frames;

            // Check if this packet overlaps our target range
            if packet_end_sample > start_sample && current_sample < end_sample {
                let mut sample_buf: SampleBuffer<f32> = SampleBuffer::new(num_frames, spec);
                sample_buf.copy_interleaved_ref(decoded);

                let packet_samples = sample_buf.samples();
                let channels = spec.channels.count();

                for (frame_idx, frame_samples) in packet_samples.chunks(channels).enumerate() {
                    #[allow(clippy::cast_possible_truncation)]
                    let sample_pos = current_sample + frame_idx as u64;

                    if sample_pos >= start_sample && sample_pos < end_sample {
                        // Average channels to mono
                        #[allow(clippy::cast_precision_loss)]
                        let mono_sample: f32 = frame_samples.iter().sum::<f32>() / channels as f32;
                        samples.push(mono_sample);
                    }
                }
            }

            current_sample = packet_end_sample;

            // Early exit if we've passed the end
            if current_sample >= end_sample {
                break;
            }
        }

        Ok(ExtractedClip {
            samples,
            sample_rate,
        })
    }
}
