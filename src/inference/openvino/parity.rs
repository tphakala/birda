//! Probe input and output comparison for the startup verification.
//!
//! The engine compares what a compiled shape computes with what ONNX Runtime
//! CPU computes for the same input. This file holds the two pure halves of that
//! check: the deterministic input and the comparison.

use crate::constants::openvino::{PROBE_AMPLITUDE, PROBE_NOISE_AMPLITUDE};

/// Multiplier of the 64-bit linear congruential generator (Knuth's MMIX).
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// Increment of the 64-bit linear congruential generator.
const LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;

/// Base frequencies, in radians per sample, of the tones mixed into a probe.
const TONE_STEPS: [f32; 3] = [0.031, 0.047, 0.113];

/// `TONE_STEPS.len()` as a float, to average the tones.
const TONE_COUNT: f32 = 3.0;

/// Per-row increase of each tone's frequency, so no two rows share a tone.
const TONE_STEP_PER_ROW: f32 = 0.0017;

/// Spreads consecutive row numbers over the generator's state space.
const NOISE_SEED_SALT: u64 = 0x9e37_79b9_7f4a_7c15;

/// A deterministic synthetic segment of `len` samples for batch row `row`.
///
/// Every row differs from every other: a batch that mixed up rows, or computed
/// only one of them, has to show up as a mismatch, which zeros in the other
/// rows would hide. The signal is a mix of three tones plus pseudo-random
/// noise, with a peak well below full scale like a quiet field recording.
#[must_use]
pub fn probe_segment(row: usize, len: usize) -> Vec<f32> {
    let mut state = (row as u64).wrapping_mul(NOISE_SEED_SALT) ^ LCG_INCREMENT;
    #[allow(clippy::cast_precision_loss)]
    let row_shift = (row % 997) as f32 * TONE_STEP_PER_ROW;

    (0..len)
        .map(|n| {
            state = state
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(LCG_INCREMENT);
            // Top 24 bits of the state, mapped to [-1, 1).
            #[allow(clippy::cast_precision_loss)]
            let noise = ((state >> 40) as f32 / 8_388_608.0) - 1.0;
            #[allow(clippy::cast_precision_loss)]
            let t = n as f32;
            let tones: f32 = TONE_STEPS
                .iter()
                .map(|step| (t * (step + row_shift)).sin())
                .sum::<f32>()
                / TONE_COUNT;
            PROBE_NOISE_AMPLITUDE.mul_add(noise, PROBE_AMPLITUDE * tones)
        })
        .collect()
}

/// Why two output buffers cannot be compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthMismatch {
    /// Length of the `OpenVINO` buffer.
    pub actual: usize,
    /// Length of the reference buffer.
    pub reference: usize,
}

/// Largest absolute difference between two equally long buffers.
///
/// A non-finite value on either side makes the result infinite, so a NaN can
/// never pass the tolerance (`f32::max` would otherwise skip it).
///
/// # Errors
///
/// [`LengthMismatch`] when the buffers differ in length.
pub fn max_abs_diff(actual: &[f32], reference: &[f32]) -> Result<f32, LengthMismatch> {
    if actual.len() != reference.len() {
        return Err(LengthMismatch {
            actual: actual.len(),
            reference: reference.len(),
        });
    }
    let mut worst = 0.0_f32;
    for (a, r) in actual.iter().zip(reference) {
        let diff = (a - r).abs();
        if !diff.is_finite() {
            return Ok(f32::INFINITY);
        }
        worst = worst.max(diff);
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_segment_is_deterministic_and_has_the_requested_length() {
        assert_eq!(probe_segment(3, 1_000), probe_segment(3, 1_000));
        assert_eq!(probe_segment(0, 17).len(), 17);
        assert_eq!(probe_segment(0, 0), Vec::<f32>::new());
    }

    #[test]
    fn probe_segments_differ_between_rows_and_stay_quiet() {
        let rows: Vec<Vec<f32>> = (0..4).map(|row| probe_segment(row, 2_000)).collect();
        for (i, a) in rows.iter().enumerate() {
            for b in &rows[i + 1..] {
                assert_ne!(a, b, "every row must carry its own signal");
            }
            let peak = a.iter().fold(0.0_f32, |peak, v| peak.max(v.abs()));
            assert!(
                peak > 0.0 && peak <= PROBE_AMPLITUDE + PROBE_NOISE_AMPLITUDE,
                "peak {peak}"
            );
        }
    }

    #[test]
    fn identical_buffers_have_no_difference() {
        assert_eq!(max_abs_diff(&[0.5, -1.0], &[0.5, -1.0]), Ok(0.0));
    }

    #[test]
    fn the_worst_difference_decides_and_tolerance_is_inclusive_below_it() {
        use crate::constants::openvino::VERIFY_TOLERANCE;

        let reference = [0.0_f32, 1.0, -2.0];
        let half = VERIFY_TOLERANCE / 2.0;
        let just_under = [0.0, 1.0 + half, -2.0];
        let just_over = [0.0, 1.0, -2.0 - (VERIFY_TOLERANCE + VERIFY_TOLERANCE)];
        assert!(max_abs_diff(&just_under, &reference).unwrap() < VERIFY_TOLERANCE);
        assert!(max_abs_diff(&just_over, &reference).unwrap() > VERIFY_TOLERANCE);
    }

    #[test]
    fn nan_and_infinity_never_pass() {
        assert_eq!(
            max_abs_diff(&[f32::NAN, 0.0], &[0.0, 0.0]),
            Ok(f32::INFINITY)
        );
        assert_eq!(
            max_abs_diff(&[0.0, f32::INFINITY], &[0.0, f32::INFINITY]),
            Ok(f32::INFINITY)
        );
    }

    #[test]
    fn a_length_mismatch_is_an_error_not_a_pass() {
        assert_eq!(
            max_abs_diff(&[0.0; 3], &[0.0; 4]),
            Err(LengthMismatch {
                actual: 3,
                reference: 4
            })
        );
    }
}
