//! Audio decoding using symphonia.

use crate::error::{Error, Result};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{CODEC_TYPE_NULL, Decoder, DecoderOptions};
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::FormatReader;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decoded audio data.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Audio samples as mono f32 in range [-1.0, 1.0].
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Duration in seconds.
    pub duration_secs: f32,
}

/// A raw segment of decoded audio (before resampling).
#[derive(Debug, Clone)]
pub struct RawSegment {
    /// Audio samples at source sample rate.
    pub samples: Vec<f32>,
    /// Start position in samples from beginning of file.
    pub start_sample: usize,
}

/// Streams audio segments from a file as they're decoded.
pub struct StreamingDecoder {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
    sample_rate: u32,
    channels: usize,
    duration_secs: Option<f64>,
    /// Buffer for accumulating decoded samples.
    buffer: Vec<f32>,
    /// Total samples emitted so far (for tracking position).
    samples_emitted: usize,
    /// Path for error reporting.
    path: std::path::PathBuf,
    /// Whether we've reached end of stream.
    eof: bool,
}

impl StreamingDecoder {
    /// Open an audio file for streaming decode.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::AudioOpen {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?;

        let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension() {
            hint.with_extension(&ext.to_string_lossy());
        }

        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| Error::AudioOpen {
                path: path.to_path_buf(),
                source: Box::new(e),
            })?;

        let format = probed.format;

        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| Error::NoAudioTracks {
                path: path.to_path_buf(),
            })?;

        let track_id = track.id;
        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| Error::AudioDecode {
                path: path.to_path_buf(),
                source: "missing sample rate".into(),
            })?;
        let channels = track
            .codec_params
            .channels
            .map_or(1, symphonia::core::audio::Channels::count);

        // Try to get duration from metadata
        #[allow(clippy::cast_precision_loss)]
        let duration_secs = track
            .codec_params
            .n_frames
            .map(|frames| frames as f64 / f64::from(sample_rate));

        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| Error::AudioDecode {
                path: path.to_path_buf(),
                source: Box::new(e),
            })?;

        Ok(Self {
            format,
            decoder,
            track_id,
            sample_rate,
            channels,
            duration_secs,
            buffer: Vec::new(),
            samples_emitted: 0,
            path: path.to_path_buf(),
            eof: false,
        })
    }

    /// Estimated total duration in seconds (from metadata), if available.
    pub fn duration_hint(&self) -> Option<f64> {
        self.duration_secs
    }

    /// Source sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Yield the next segment of decoded audio.
    ///
    /// # Arguments
    /// * `segment_samples` - Number of samples per segment
    /// * `overlap_samples` - Number of samples to overlap between segments (must be less than `segment_samples`)
    ///
    /// Returns `None` when the file is exhausted.
    ///
    /// # Errors
    /// Returns an error if `overlap_samples >= segment_samples`.
    pub fn next_segment(
        &mut self,
        segment_samples: usize,
        overlap_samples: usize,
    ) -> Result<Option<RawSegment>> {
        // Validate parameters
        if overlap_samples >= segment_samples {
            return Err(crate::error::Error::Internal {
                message: format!(
                    "overlap_samples ({overlap_samples}) must be less than segment_samples ({segment_samples})"
                ),
            });
        }

        // Keep decoding until we have enough samples or hit EOF
        while self.buffer.len() < segment_samples && !self.eof {
            self.decode_next_packet()?;
        }

        // If we don't have any samples, we're done
        if self.buffer.is_empty() {
            return Ok(None);
        }

        // Build the segment
        let take_samples = segment_samples.min(self.buffer.len());
        let mut samples = self.buffer[..take_samples].to_vec();

        // Zero-pad if needed (for final segment)
        if samples.len() < segment_samples {
            samples.resize(segment_samples, 0.0);
        }

        let start_sample = self.samples_emitted;

        // Advance buffer, keeping overlap
        let advance = take_samples.saturating_sub(overlap_samples);

        if advance > 0 {
            self.buffer.drain(..advance);
            self.samples_emitted += advance;
        } else {
            // Final segment: advance is 0 when take_samples <= overlap_samples,
            // which only happens when buffer has less than a full segment (EOF)
            self.buffer.clear();
            self.samples_emitted += take_samples;
        }

        Ok(Some(RawSegment {
            samples,
            start_sample,
        }))
    }

    /// Decode the next packet and append samples to buffer.
    fn decode_next_packet(&mut self) -> Result<()> {
        let packet = match self.format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                self.eof = true;
                return Ok(());
            }
            Err(e) => {
                return Err(Error::AudioDecode {
                    path: self.path.clone(),
                    source: Box::new(e),
                });
            }
        };

        if packet.track_id() != self.track_id {
            return Ok(());
        }

        let decoded = self
            .decoder
            .decode(&packet)
            .map_err(|e| Error::AudioDecode {
                path: self.path.clone(),
                source: Box::new(e),
            })?;

        append_samples(&decoded, self.channels, &mut self.buffer);
        Ok(())
    }
}

/// Decode an audio file to mono f32 samples.
///
/// Supports WAV, FLAC, MP3, and AAC formats.
pub fn decode_audio_file(path: &Path) -> Result<DecodedAudio> {
    let file = File::open(path).map_err(|e| Error::AudioOpen {
        path: path.to_path_buf(),
        source: Box::new(e),
    })?;

    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

    // Create hint from file extension
    // Use to_string_lossy() to handle non-UTF-8 extensions gracefully
    let mut hint = Hint::new();
    if let Some(ext) = path.extension() {
        hint.with_extension(&ext.to_string_lossy());
    }

    // Probe the file
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| Error::AudioOpen {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?;

    let mut format = probed.format;

    // Find the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| Error::NoAudioTracks {
            path: path.to_path_buf(),
        })?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| Error::AudioDecode {
            path: path.to_path_buf(),
            source: "missing sample rate".into(),
        })?;
    let channels = track
        .codec_params
        .channels
        .map_or(1, symphonia::core::audio::Channels::count);

    // Create decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| Error::AudioDecode {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?;

    let mut samples = Vec::new();

    // Decode all packets
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
                    path: path.to_path_buf(),
                    source: Box::new(e),
                });
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).map_err(|e| Error::AudioDecode {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?;

        append_samples(&decoded, channels, &mut samples);
    }

    #[allow(clippy::cast_precision_loss)]
    let duration_secs = samples.len() as f32 / sample_rate as f32;

    Ok(DecodedAudio {
        samples,
        sample_rate,
        duration_secs,
    })
}

/// Append decoded samples to the output buffer, converting to mono.
fn append_samples(buffer: &AudioBufferRef, channels: usize, output: &mut Vec<f32>) {
    match buffer {
        AudioBufferRef::F32(buf) => {
            if channels == 1 {
                output.extend(buf.chan(0));
            } else {
                // Mix to mono
                let frames = buf.frames();
                for i in 0..frames {
                    let mut sum = 0.0f32;
                    for ch in 0..channels {
                        sum += buf.chan(ch)[i];
                    }
                    #[allow(clippy::cast_precision_loss)]
                    output.push(sum / channels as f32);
                }
            }
        }
        AudioBufferRef::S16(buf) => {
            const I16_NORM: f32 = 32768.0;
            if channels == 1 {
                output.extend(buf.chan(0).iter().map(|&s| f32::from(s) / I16_NORM));
            } else {
                let frames = buf.frames();
                for i in 0..frames {
                    let mut sum = 0.0f32;
                    for ch in 0..channels {
                        sum += f32::from(buf.chan(ch)[i]) / I16_NORM;
                    }
                    #[allow(clippy::cast_precision_loss)]
                    output.push(sum / channels as f32);
                }
            }
        }
        AudioBufferRef::S32(buf) => {
            const I32_NORM: f32 = 2_147_483_648.0;
            if channels == 1 {
                #[allow(clippy::cast_precision_loss)]
                output.extend(buf.chan(0).iter().map(|&s| s as f32 / I32_NORM));
            } else {
                let frames = buf.frames();
                for i in 0..frames {
                    let mut sum = 0.0f32;
                    for ch in 0..channels {
                        #[allow(clippy::cast_precision_loss)]
                        {
                            sum += buf.chan(ch)[i] as f32 / I32_NORM;
                        }
                    }
                    #[allow(clippy::cast_precision_loss)]
                    output.push(sum / channels as f32);
                }
            }
        }
        _ => {
            // Unsupported format, skip
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_segment_construction() {
        // Basic struct construction test
        let segment = RawSegment {
            samples: vec![1.0, 2.0, 3.0],
            start_sample: 0,
        };
        assert_eq!(segment.samples.len(), 3);
        assert_eq!(segment.start_sample, 0);
    }
}
