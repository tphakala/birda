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
    let channels = 1;

    let mut resampler = Fft::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size,
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

        let resampled = resampler
            .process(&input_adapter, None)
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

        let resampled = resampler
            .process(&input_adapter, None)
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
    /// The rate of CD sourced audio, and the usual rate for mp3 files.
    const TEST_RATE_CD: u32 = 44_100;
    /// Length of the generated test signals, one second at the higher rate.
    const TEST_SIGNAL_LEN: usize = 48_000;
    /// A tone in the middle of the range most bird vocalisations occupy.
    const BIRD_BAND_HZ: f32 = 6_000.0;
    /// A reference tone low in the band, used to check pitch is preserved.
    const REFERENCE_TONE_HZ: f32 = 1_000.0;
    /// A tone above the 16 kHz Nyquist limit of the output rate, used to check
    /// the anti-aliasing filter removes what it cannot represent.
    const ABOVE_NYQUIST_HZ: f32 = 20_000.0;
    /// Where a resampler with no anti-aliasing filter folds `ABOVE_NYQUIST_HZ`
    /// back to, as |20000 - 32000|. Nothing should ever be measurable here.
    const ALIAS_IMAGE_HZ: f32 = 12_000.0;
    /// How far the measured tone may sit below the strongest competing band.
    const DOMINANCE_RATIO: f32 = 100.0;
    /// Least of the expected power a tone may retain in its own bin.
    ///
    /// Dominance over other bins is not enough on its own: a tone shifted by a
    /// few Hz leaves its bin while still towering over distant ones, so the
    /// ratio alone accepts a pitch shift. This pins the energy to where it
    /// belongs.
    const MIN_TONE_POWER_FRACTION: f32 = 0.5;
    /// Highest RMS accepted for a signal that should have been filtered away.
    const FILTERED_RMS_CEILING: f32 = 0.1;
    /// Least RMS expected of a tone that must survive resampling intact.
    ///
    /// A full amplitude sine has an RMS of about 0.707.
    const PRESERVED_RMS_FLOOR: f32 = 0.6;
    /// Most the RMS may drift between input and output.
    const RMS_TOLERANCE: f32 = 0.05;
    /// Fraction of a resampled signal trimmed from each end before measuring.
    const STEADY_STATE_MARGIN: usize = 8;
    /// Length of the 44.1 kHz test signals, one second at that rate.
    const TEST_SIGNAL_LEN_CD: usize = 44_100;
    /// Most of the expected power an aliased image may carry.
    ///
    /// A working filter leaves nothing measurable here, so this only needs to
    /// sit far below a real tone.
    const ALIAS_POWER_FRACTION: f32 = 1e-6;

    /// Power the Goertzel filter reports for a full amplitude sine sitting
    /// exactly on the bin being measured. Falls out of the algorithm as n / 4.
    #[allow(clippy::cast_precision_loss)]
    fn expected_tone_power(len: usize) -> f32 {
        len as f32 / 4.0
    }

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
        let margin = samples.len() / STEADY_STATE_MARGIN;
        &samples[margin..samples.len() - margin]
    }

    /// Assert a tone came through at its own frequency and at full strength.
    ///
    /// Both halves are needed. The absolute floor catches a tone that drifted
    /// off its bin, and the dominance check catches energy appearing where it
    /// should not be.
    fn assert_tone_intact(body: &[f32], rate: u32, tone_hz: f32, other_bins: &[f32]) {
        let at_tone = tone_power(body, rate, tone_hz);
        let floor = expected_tone_power(body.len()) * MIN_TONE_POWER_FRACTION;
        assert!(
            at_tone > floor,
            "{tone_hz} Hz tone lost power in its own bin, so the pitch moved: {at_tone}, expected above {floor}"
        );

        for other in other_bins {
            let at_other = tone_power(body, rate, *other);
            assert!(
                at_tone > at_other * DOMINANCE_RATIO,
                "{tone_hz} Hz tone did not dominate {other} Hz: {at_tone} vs {at_other}"
            );
        }
    }

    #[test]
    fn test_resample_preserves_tone_frequency() {
        let input = sine(REFERENCE_TONE_HZ, TEST_RATE_HIGH, TEST_SIGNAL_LEN);
        let output = resample(input, TEST_RATE_HIGH, TEST_RATE_LOW).unwrap();
        assert_tone_intact(
            steady_state(&output),
            TEST_RATE_LOW,
            REFERENCE_TONE_HZ,
            &[500.0, 2_000.0, 4_000.0],
        );
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

        assert_tone_intact(
            body,
            TEST_RATE_LOW,
            BIRD_BAND_HZ,
            &[3_000.0, 9_000.0, 12_000.0],
        );

        // The tone must also survive at close to full amplitude. Dominance
        // alone would still pass if everything were attenuated together.
        let level = rms(body);
        assert!(
            level > PRESERVED_RMS_FLOOR,
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
        let input = sine(ABOVE_NYQUIST_HZ, TEST_RATE_HIGH, TEST_SIGNAL_LEN);
        let output = resample(input, TEST_RATE_HIGH, TEST_RATE_LOW).unwrap();
        let body = steady_state(&output);

        let alias = tone_power(body, TEST_RATE_LOW, ALIAS_IMAGE_HZ);
        let ceiling = expected_tone_power(body.len()) * ALIAS_POWER_FRACTION;
        assert!(
            alias < ceiling,
            "20 kHz folded back to {ALIAS_IMAGE_HZ} Hz, anti-aliasing filter is not working: power {alias}"
        );

        let level = rms(body);
        assert!(
            level < FILTERED_RMS_CEILING,
            "content above the output Nyquist limit survived resampling: rms {level}"
        );
    }

    #[test]
    fn test_resample_from_cd_rate_filters_above_output_nyquist() {
        // 44.1 kHz is what mp3 and CD sourced files arrive at, so this path is
        // as real as the 48 kHz one, and it sizes the FFT completely
        // differently. The block size comes from the greatest common divisor of
        // the two rates: 48 kHz and 32 kHz share 16000, giving 342 FFT chunks
        // for a 1024 frame chunk, while 44.1 kHz and 32 kHz share only 100,
        // giving 3. Passing at one rate pair says little about the other.
        let input = sine(ABOVE_NYQUIST_HZ, TEST_RATE_CD, TEST_SIGNAL_LEN_CD);
        let output = resample(input, TEST_RATE_CD, TEST_RATE_LOW).unwrap();
        let body = steady_state(&output);

        let level = rms(body);
        assert!(
            level < FILTERED_RMS_CEILING,
            "content above the output Nyquist limit survived 44.1 kHz to 32 kHz resampling: rms {level}"
        );
    }

    #[test]
    fn test_resample_from_cd_rate_preserves_bird_band_content() {
        let input = sine(BIRD_BAND_HZ, TEST_RATE_CD, TEST_SIGNAL_LEN_CD);
        let output = resample(input, TEST_RATE_CD, TEST_RATE_LOW).unwrap();
        assert_tone_intact(
            steady_state(&output),
            TEST_RATE_LOW,
            BIRD_BAND_HZ,
            &[3_000.0, 9_000.0, 12_000.0],
        );
    }

    #[test]
    fn test_resample_preserves_amplitude() {
        let input = sine(REFERENCE_TONE_HZ, TEST_RATE_HIGH, TEST_SIGNAL_LEN);
        let input_rms = rms(&input);
        let output = resample(input, TEST_RATE_HIGH, TEST_RATE_LOW).unwrap();
        let output_rms = rms(steady_state(&output));

        assert!(
            (output_rms - input_rms).abs() < RMS_TOLERANCE,
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
