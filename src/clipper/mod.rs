//! Audio clip extraction from detection results.
//!
//! This module provides functionality to extract audio segments from
//! detection result files, grouping by species and merging overlapping
//! detections.

pub mod command;
mod extractor;
mod grouper;
mod parser;
mod writer;

pub use extractor::{ClipExtractor, ExtractedClip};
pub use grouper::{DetectionGroup, group_detections};
pub use parser::{ParsedDetection, parse_detection_file};
pub use writer::WavWriter;

use crate::Error;

/// Check that a clip's time range is usable, at every layer that accepts one.
///
/// Shared rather than restated, because the two callers are the command's
/// range guard and the extractor's own entry check, and a rule spelled twice
/// is a rule that can be tightened in one place only. Issue #306 was filed
/// against exactly that shape.
///
/// The finiteness test has to come first and cannot be inferred from the
/// ordering: every comparison against NaN is false, so `end <= start` accepts
/// NaN, and it accepts an infinite `end` too because nothing is greater than
/// infinity. Both then reach a seconds-to-samples cast, which saturates rather
/// than trapping.
///
/// The lower bound is not implied by the ordering either. `--start -100 --end
/// -1` is finite and increasing, and `parse_time` refuses it, but a library
/// caller reaches this with it: `(start - pre).max(0.0)` then clamps to 0.0
/// and the caller silently receives the opening seconds of the file under a
/// name claiming a range it never asked for. Same class of wrong answer #310
/// was filed for, so the rule is enforced here rather than only in clap.
///
/// # Errors
///
/// Returns [`Error::InvalidTimeRange`] if either bound is not finite or
/// negative, or if `end` is not strictly greater than `start`.
pub(crate) fn validate_time_range(start: f64, end: f64) -> Result<(), Error> {
    if !start.is_finite() || !end.is_finite() || start < 0.0 || end <= start {
        return Err(Error::InvalidTimeRange { start, end });
    }

    Ok(())
}
