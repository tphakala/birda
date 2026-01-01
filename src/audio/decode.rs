//! Audio decoding using symphonia.

use crate::error::{Error, Result};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::formats::FormatOptions;
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
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
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
