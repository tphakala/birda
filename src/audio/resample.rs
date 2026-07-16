//! Audio resampling using rubato.

use crate::error::{Error, Result};
use audioadapter_buffers::direct::SequentialSlice;
use rubato::{Fft, FixedSync, Resampler};

/// Resample audio to the target sample rate.
///
/// Returns the input unchanged if already at the target rate.
pub fn resample(samples: Vec<f32>, from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples);
    }

    // Create FFT-based synchronous resampler with fixed input/output sizes
    let chunk_size = 1024;
    let sub_chunks = 1;
    let channels = 1;

    let mut resampler = Fft::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size,
        sub_chunks,
        channels,
        FixedSync::Both,
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
        // Wrap as single-channel sequential data
        let input_adapter =
            SequentialSlice::new(chunk, channels, input_frames_needed).map_err(|e| {
                Error::Resample {
                    reason: format!("failed to create input adapter: {e}"),
                }
            })?;

        let resampled =
            resampler
                .process(&input_adapter, 0, None)
                .map_err(|e| Error::Resample {
                    reason: e.to_string(),
                })?;

        // Extract samples from the interleaved output
        let output_data = resampled.take_data();
        output.extend_from_slice(&output_data);
        pos += input_frames_needed;
    }

    // Handle remaining samples by padding
    if pos < samples.len() {
        let remaining = samples.len() - pos;
        let mut padded = samples[pos..].to_vec();
        padded.resize(input_frames_needed, 0.0);

        let input_adapter =
            SequentialSlice::new(&padded, channels, input_frames_needed).map_err(|e| {
                Error::Resample {
                    reason: format!("failed to create input adapter: {e}"),
                }
            })?;

        let resampled =
            resampler
                .process(&input_adapter, 0, None)
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

        let output_data = resampled.take_data();
        let take_count = output_frames.min(output_data.len());
        output.extend_from_slice(&output_data[..take_count]);
    }

    Ok(output)
}

/// Resample a single audio chunk.
///
/// Convenience wrapper for streaming workflows. Delegates to the standard
/// FFT-based [`resample`] function.
pub fn resample_chunk(samples: Vec<f32>, from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples);
    }

    // For streaming, we use the same FFT resampler but on chunk-sized data
    // The existing resample function handles this well
    resample(samples, from_rate, to_rate)
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
    use std::f32::consts::PI;

    /// Sample rate of the test signals, matching the common capture rate.
    const TEST_RATE_HIGH: u32 = 48_000;
    /// Target rate used by the downsample tests, matching the `BirdNET` input rate.
    const TEST_RATE_LOW: u32 = 32_000;
    /// Length of the generated test signals, one second at the higher rate.
    const TEST_SIGNAL_LEN: usize = 48_000;
    /// A tone in the middle of the range most bird vocalisations occupy.
    const BIRD_BAND_HZ: f32 = 6_000.0;
    /// How far the measured tone may sit below the strongest competing band.
    const DOMINANCE_RATIO: f32 = 100.0;

    /// Generate a pure sine wave.
    #[allow(clippy::cast_precision_loss)]
    fn sine(freq_hz: f32, rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / rate as f32).sin())
            .collect()
    }

    /// Power of a single frequency, via the Goertzel algorithm.
    ///
    /// Used instead of comparing samples directly because the resampler delays
    /// its output by half the FFT block, so the signal comes back phase shifted.
    /// Measuring in the frequency domain ignores that shift.
    #[allow(clippy::cast_precision_loss)]
    fn tone_power(samples: &[f32], rate: u32, freq_hz: f32) -> f32 {
        let n = samples.len() as f32;
        let k = (n * freq_hz / rate as f32).round();
        let w = 2.0 * PI * k / n;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for &x in samples {
            let s0 = coeff.mul_add(s1, x) - s2;
            s2 = s1;
            s1 = s0;
        }
        // Goertzel magnitude: s1^2 + s2^2 - coeff*s1*s2.
        (coeff * s1).mul_add(-s2, s1.mul_add(s1, s2 * s2)).max(0.0) / n
    }

    /// Root mean square of a signal.
    #[allow(clippy::cast_precision_loss)]
    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
    }

    /// Drop the head and tail of a resampled signal.
    ///
    /// The resampler ramps up over its delay and the last chunk is zero padded,
    /// so both ends are unrepresentative. The middle is the steady state.
    fn steady_state(samples: &[f32]) -> &[f32] {
        let margin = samples.len() / 8;
        &samples[margin..samples.len() - margin]
    }

    #[test]
    fn test_resample_preserves_tone_frequency() {
        let input = sine(1_000.0, TEST_RATE_HIGH, TEST_SIGNAL_LEN);
        let output = resample(input, TEST_RATE_HIGH, TEST_RATE_LOW).unwrap();
        let body = steady_state(&output);

        let at_tone = tone_power(body, TEST_RATE_LOW, 1_000.0);
        // A resampler that shifted the pitch would move the energy to a
        // different bin, so check the neighbours are quiet rather than just
        // checking the tone is present.
        for other in [500.0, 2_000.0, 4_000.0] {
            let at_other = tone_power(body, TEST_RATE_LOW, other);
            assert!(
                at_tone > at_other * DOMINANCE_RATIO,
                "1 kHz tone did not dominate {other} Hz: {at_tone} vs {at_other}"
            );
        }
    }

    #[test]
    fn test_resample_preserves_bird_band_content() {
        // Guards the anti-aliasing cutoff. 6 kHz is well under the 16 kHz
        // Nyquist limit of the 32 kHz output, so it must survive intact. A
        // resampler configured with too small an FFT block lowers the cutoff
        // and quietly attenuates exactly this band, which is where most bird
        // song lives.
        let input = sine(BIRD_BAND_HZ, TEST_RATE_HIGH, TEST_SIGNAL_LEN);
        let output = resample(input, TEST_RATE_HIGH, TEST_RATE_LOW).unwrap();
        let body = steady_state(&output);

        let at_tone = tone_power(body, TEST_RATE_LOW, BIRD_BAND_HZ);
        for other in [3_000.0, 9_000.0, 12_000.0] {
            let at_other = tone_power(body, TEST_RATE_LOW, other);
            assert!(
                at_tone > at_other * DOMINANCE_RATIO,
                "6 kHz tone did not dominate {other} Hz: {at_tone} vs {at_other}"
            );
        }

        // The tone must also survive at close to full amplitude. Dominance
        // alone would still pass if everything were attenuated together.
        let level = rms(body);
        assert!(
            level > 0.6,
            "6 kHz tone was attenuated by resampling: rms {level}, expected about 0.707"
        );
    }

    #[test]
    fn test_resample_filters_content_above_output_nyquist() {
        // The one test here that can tell a real resampler from a naive one.
        //
        // 20 kHz fits under the 24 kHz Nyquist limit of the 48 kHz input but
        // not under the 16 kHz limit of the 32 kHz output. A resampler with a
        // working anti-aliasing filter removes it. One without folds it back to
        // |20000 - 32000| = 12 kHz, inventing a tone that was never played.
        //
        // In-band tones cannot catch this: they stay dominant either way, which
        // is why the other tests here pass even against nearest-neighbour
        // decimation.
        let input = sine(20_000.0, TEST_RATE_HIGH, TEST_SIGNAL_LEN);
        let output = resample(input, TEST_RATE_HIGH, TEST_RATE_LOW).unwrap();
        let body = steady_state(&output);

        let alias = tone_power(body, TEST_RATE_LOW, 12_000.0);
        assert!(
            alias < 1e-3,
            "20 kHz folded back to 12 kHz, anti-aliasing filter is not working: power {alias}"
        );

        let level = rms(body);
        assert!(
            level < 0.1,
            "content above the output Nyquist limit survived resampling: rms {level}"
        );
    }

    #[test]
    fn test_resample_preserves_amplitude() {
        let input = sine(1_000.0, TEST_RATE_HIGH, TEST_SIGNAL_LEN);
        let input_rms = rms(&input);
        let output = resample(input, TEST_RATE_HIGH, TEST_RATE_LOW).unwrap();
        let output_rms = rms(steady_state(&output));

        assert!(
            (output_rms - input_rms).abs() < 0.05,
            "amplitude changed: {input_rms} in, {output_rms} out"
        );
    }

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
