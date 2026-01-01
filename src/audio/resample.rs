//! Audio resampling using rubato.

use crate::error::{Error, Result};
use rubato::{FftFixedInOut, Resampler};

/// Resample audio to the target sample rate.
///
/// Returns the input unchanged if already at the target rate.
pub fn resample(samples: Vec<f32>, from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples);
    }

    let mut resampler = FftFixedInOut::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        1024, // chunk size
        1,    // channels
    )
    .map_err(|e| Error::Resample {
        reason: e.to_string(),
    })?;

    let input_frames_needed = resampler.input_frames_next();
    let mut output = Vec::with_capacity(estimate_output_len(samples.len(), from_rate, to_rate));

    // Process in chunks
    let mut pos = 0;
    while pos + input_frames_needed <= samples.len() {
        let chunk = &samples[pos..pos + input_frames_needed];
        let input = vec![chunk.to_vec()];

        let resampled = resampler
            .process(&input, None)
            .map_err(|e| Error::Resample {
                reason: e.to_string(),
            })?;

        output.extend(&resampled[0]);
        pos += input_frames_needed;
    }

    // Handle remaining samples by padding
    if pos < samples.len() {
        let remaining = samples.len() - pos;
        let mut padded = samples[pos..].to_vec();
        padded.resize(input_frames_needed, 0.0);

        let input = vec![padded];
        let resampled = resampler
            .process(&input, None)
            .map_err(|e| Error::Resample {
                reason: e.to_string(),
            })?;

        // Only take proportional amount of output
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let output_frames =
            (remaining as f64 * f64::from(to_rate) / f64::from(from_rate)).ceil() as usize;
        output.extend(&resampled[0][..output_frames.min(resampled[0].len())]);
    }

    Ok(output)
}

/// Estimate output length after resampling.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn estimate_output_len(input_len: usize, from_rate: u32, to_rate: u32) -> usize {
    ((input_len as f64) * f64::from(to_rate) / f64::from(from_rate)).ceil() as usize + 1024
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_same_rate_returns_input() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = resample(samples.clone(), 48000, 48000);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), samples);
    }

    #[test]
    fn test_resample_downsample() {
        // Create a simple test signal
        #[allow(clippy::cast_precision_loss)]
        let samples: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.001).sin()).collect();
        let result = resample(samples, 48000, 32000);
        assert!(result.is_ok());
        let output = result.unwrap();
        // Output should be roughly 2/3 the length
        assert!(output.len() > 20000);
        assert!(output.len() < 35000);
    }

    #[test]
    fn test_resample_upsample() {
        #[allow(clippy::cast_precision_loss)]
        let samples: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.001).sin()).collect();
        let result = resample(samples, 32000, 48000);
        assert!(result.is_ok());
        let output = result.unwrap();
        // Output should be roughly 1.5x the length
        assert!(output.len() > 45000);
        assert!(output.len() < 55000);
    }
}
