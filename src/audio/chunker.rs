//! Audio chunking with overlap support.

/// A chunk of audio with its time offset.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Audio samples for this chunk.
    pub samples: Vec<f32>,
    /// Start time in seconds.
    pub start_time: f32,
    /// End time in seconds.
    pub end_time: f32,
}

/// Chunk audio samples with overlap.
///
/// # Arguments
///
/// * `samples` - Audio samples to chunk
/// * `sample_rate` - Sample rate in Hz
/// * `chunk_duration` - Duration of each chunk in seconds
/// * `overlap` - Overlap between chunks in seconds
///
/// # Returns
///
/// Vector of `AudioChunk` with time offsets.
pub fn chunk_audio(
    samples: &[f32],
    sample_rate: u32,
    chunk_duration: f32,
    overlap: f32,
) -> Vec<AudioChunk> {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let chunk_samples = (chunk_duration * sample_rate as f32) as usize;

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let overlap_samples = (overlap * sample_rate as f32) as usize;

    let step = chunk_samples.saturating_sub(overlap_samples);
    if step == 0 {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut pos = 0;

    while pos < samples.len() {
        let end = (pos + chunk_samples).min(samples.len());
        let mut chunk_data = samples[pos..end].to_vec();

        // Zero-pad if needed
        chunk_data.resize(chunk_samples, 0.0);

        #[allow(clippy::cast_precision_loss)]
        let start_time = pos as f32 / sample_rate as f32;

        let end_time = start_time + chunk_duration;

        chunks.push(AudioChunk {
            samples: chunk_data,
            start_time,
            end_time,
        });

        pos += step;
    }

    chunks
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_audio_no_overlap() {
        let samples = vec![0.0; 96_000]; // 2 seconds at 48kHz
        let chunks = chunk_audio(&samples, 48_000, 1.0, 0.0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start_time, 0.0);
        assert_eq!(chunks[1].start_time, 1.0);
    }

    #[test]
    fn test_chunk_audio_with_overlap() {
        let samples = vec![0.0; 144_000]; // 3 seconds at 48kHz
        let chunks = chunk_audio(&samples, 48_000, 1.0, 0.5);
        // With 1s chunks and 0.5s overlap, step is 0.5s
        // Positions: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5
        assert_eq!(chunks.len(), 6);
        assert_eq!(chunks[0].start_time, 0.0);
        assert_eq!(chunks[1].start_time, 0.5);
    }

    #[test]
    fn test_chunk_audio_pads_final_chunk() {
        let samples = vec![0.0; 60_000]; // 1.25 seconds at 48kHz
        let chunks = chunk_audio(&samples, 48_000, 1.0, 0.0);
        assert_eq!(chunks.len(), 2);
        // Second chunk should be padded to 48000 samples
        assert_eq!(chunks[1].samples.len(), 48_000);
    }

    #[test]
    fn test_chunk_audio_empty_input() {
        let samples: Vec<f32> = vec![];
        let chunks = chunk_audio(&samples, 48_000, 1.0, 0.0);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_audio_overlap_equals_duration() {
        let samples = vec![0.0; 96_000];
        let chunks = chunk_audio(&samples, 48_000, 1.0, 1.0);
        // Step would be 0, should return empty
        assert!(chunks.is_empty());
    }
}
