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
