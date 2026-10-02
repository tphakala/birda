//! Processing pipeline components.

mod config;
mod coordinator;
mod processor;

pub use config::ProcessingConfig;
pub use coordinator::{
    OutputTarget, ProcessCheck, ProcessOptions, collect_input_files, plan_output_targets,
    should_process,
};
pub use processor::{ProcessResult, process_file};
