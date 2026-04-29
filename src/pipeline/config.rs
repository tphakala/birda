//! Configuration types for the processing pipeline.

use crate::config::OutputFormat;
use std::path::Path;

/// Configuration for processing a single audio file.
///
/// Bundles the parameters needed by `process_file` to reduce its argument count.
pub struct ProcessingConfig<'a> {
    /// Path to input audio file.
    pub input_path: &'a Path,
    /// Directory for output files.
    pub output_dir: &'a Path,
    /// Output formats to generate.
    pub formats: &'a [OutputFormat],
    /// Minimum confidence threshold (0.0-1.0).
    pub min_confidence: f32,
    /// Overlap between chunks in seconds.
    pub overlap: f32,
    /// Number of chunks to process in parallel.
    pub batch_size: usize,
    /// Additional columns to include in CSV output.
    pub csv_columns: &'a [String],
    /// Whether to show progress bars.
    pub progress_enabled: bool,
    /// Whether to include UTF-8 BOM in CSV output.
    pub csv_bom_enabled: bool,
    /// Model name for JSON output metadata.
    pub model_name: &'a str,
    /// Optional (lat, lon, week) for JSON output metadata.
    pub range_filter_params: Option<(f64, f64, u8)>,
    /// Optional (lat, lon, `day_of_year`) for BSG SDM.
    pub bsg_params: Option<(f64, f64, Option<u32>)>,
    /// Optional reporter for stdout mode.
    pub reporter: Option<&'a dyn crate::output::ProgressReporter>,
    /// Whether to write both files and stdout.
    pub dual_output_mode: bool,
}
