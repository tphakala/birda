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
use tracing::trace;

use crate::Error;
use crate::constants::clipper::{
    MAX_CLIP_PREALLOC_SAMPLES, MAX_CLIP_PREALLOC_SECS, SEEK_THRESHOLD_SECS,
};

use super::{DetectionGroup, validate_time_range};

/// How many samples an extracted clip may reserve up front, for a file at
/// `sample_rate`.
///
/// Two terms, because each covers the other's blind spot.
///
/// The rate-scaled one keeps the cap meaning the same duration whatever the
/// recording is: a flat sample count would be a full minute at 48 kHz but a
/// fifth of that on ultrasonic material, short enough that the default bat
/// clip would exceed it before anything unusual happened.
///
/// The absolute one is not belt and braces. `sample_rate` is read straight out
/// of the container and validated nowhere, so a hand-built WAV declaring
/// `u32::MAX` scales the first term to 257 billion samples and asks for a
/// terabyte, which is #310 again by another road. `unwrap_or(0)` fails towards
/// no reservation rather than an unbounded one, for the same reason.
fn prealloc_cap(sample_rate: u32) -> usize {
    MAX_CLIP_PREALLOC_SECS
        .saturating_mul(usize::try_from(sample_rate).unwrap_or(0))
        .min(MAX_CLIP_PREALLOC_SAMPLES)
}

/// Result of clip extraction.
pub struct ExtractedClip {
    /// Audio samples as f32 (-1.0 to 1.0).
    pub samples: Vec<f32>,
    /// Sample rate of the extracted audio.
    pub sample_rate: u32,
}

/// Extracts audio clips from source files.
///
/// Note: Padding is applied during detection grouping, not during extraction.
/// The extractor receives groups with already-padded time ranges.
pub struct ClipExtractor;

impl Default for ClipExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl ClipExtractor {
    /// Create a new clip extractor.
    #[must_use]
    pub fn new() -> Self {
        Self
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
    /// Returns [`Error::InvalidTimeRange`] if `group`'s bounds are not finite
    /// or do not increase, and an error if the audio file cannot be read or
    /// decoded.
    pub fn extract_clip(
        &self,
        source_path: &Path,
        group: &DetectionGroup,
    ) -> Result<ExtractedClip, Error> {
        // `ClipExtractor` is public and the range reaches an allocation below,
        // so validate here instead of trusting every path in.
        validate_time_range(group.start, group.end)?;

        // Open the audio file
        let file = std::fs::File::open(source_path).map_err(|e| Error::AudioOpen {
            path: source_path.to_path_buf(),
            source: Box::new(e),
        })?;

        let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

        // Probe the format with file extension hint
        let mut hint = Hint::new();
        if let Some(ext) = source_path.extension() {
            hint.with_extension(&ext.to_string_lossy());
        }
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

        // Capped, because a range that passes validation can still be
        // enormous: an end of 1e12 seconds is finite and non-negative, and at
        // 48 kHz it works out to 4.8e16 samples, which `Vec::with_capacity`
        // answers by aborting the process. Undersizing only costs
        // reallocations, since the buffer grows to whatever the decode loop
        // actually produces.
        //
        // `saturating_sub` cannot trigger given the check above (the cast is
        // monotonic, so a validated `end > start` gives `end_sample >=
        // start_sample`); it is here so the expression stays correct on its
        // own terms rather than depending on a check twenty lines away.
        // `usize::MAX` is likewise unreachable on a 64-bit target and is the
        // honest fallback on a 32-bit one, where the `.min` then applies the
        // real bound.
        //
        let expected_samples = usize::try_from(end_sample.saturating_sub(start_sample))
            .unwrap_or(usize::MAX)
            .min(prealloc_cap(sample_rate));

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
                Err(symphonia::core::errors::Error::DecodeError(msg)) => {
                    trace!("Skipping corrupted audio frame: {msg}");
                    continue;
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn group(start: f64, end: f64) -> DetectionGroup {
        DetectionGroup {
            scientific_name: "Parus major".to_string(),
            common_name: "Great Tit".to_string(),
            start,
            end,
            max_confidence: 1.0,
            detection_count: 1,
        }
    }

    /// A path that cannot exist, so a validation error proves the range is
    /// rejected before any I/O rather than deep in the decode loop, where the
    /// bad value has already been cast to a sample count.
    fn missing_path() -> &'static Path {
        Path::new("/nonexistent/birda-extractor-test-fixture.wav")
    }

    #[test]
    fn test_prealloc_cap_scales_with_the_rate_but_never_past_the_ceiling() {
        // The ordinary case: a full minute of audio, whatever the rate.
        assert_eq!(prealloc_cap(48_000), MAX_CLIP_PREALLOC_SECS * 48_000);
        assert_eq!(
            prealloc_cap(crate::constants::bat::SAMPLE_RATE),
            MAX_CLIP_PREALLOC_SAMPLES,
            "the ceiling is set at the highest rate the tool handles, so it \
             should bind exactly there and not below"
        );

        // The reason the ceiling exists. Nothing validates the rate a
        // container declares, and 60 seconds at `u32::MAX` is 257 billion
        // samples, a terabyte of reservation from a hand-built header.
        assert_eq!(prealloc_cap(u32::MAX), MAX_CLIP_PREALLOC_SAMPLES);
        assert!(
            MAX_CLIP_PREALLOC_SAMPLES < MAX_CLIP_PREALLOC_SECS.saturating_mul(u32::MAX as usize)
        );

        // A rate of zero reserves nothing rather than everything.
        assert_eq!(prealloc_cap(0), 0);
    }

    #[test]
    fn test_extract_clip_rejects_non_finite_range_before_touching_the_file() {
        for (start, end) in [
            (0.0, f64::INFINITY),
            (f64::NAN, 5.0),
            (0.0, f64::NAN),
            (f64::NEG_INFINITY, 5.0),
            (f64::NAN, f64::NAN),
        ] {
            let result = ClipExtractor::new().extract_clip(missing_path(), &group(start, end));
            assert!(
                matches!(result, Err(Error::InvalidTimeRange { .. })),
                "start={start} end={end} was not rejected as an invalid range"
            );
        }
    }

    #[test]
    fn test_extract_clip_rejects_empty_and_inverted_ranges() {
        for (start, end) in [(5.0, 5.0), (5.0, 1.0)] {
            let result = ClipExtractor::new().extract_clip(missing_path(), &group(start, end));
            assert!(
                matches!(result, Err(Error::InvalidTimeRange { .. })),
                "start={start} end={end} was not rejected as an invalid range"
            );
        }
    }

    #[test]
    fn test_a_valid_range_gets_past_validation() {
        // The counterpart to the cases above: a sane range must reach the file
        // open and fail there, not in the range check.
        let result = ClipExtractor::new().extract_clip(missing_path(), &group(1.0, 3.0));
        assert!(
            matches!(result, Err(Error::AudioOpen { .. })),
            "a valid range should have reached the file open and failed there"
        );
    }
}
