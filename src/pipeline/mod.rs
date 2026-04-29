//! Processing pipeline components.

mod config;
mod coordinator;
mod processor;

pub use config::ProcessingConfig;
pub use coordinator::{
    ProcessCheck, ProcessOptions, collect_input_files, output_dir_for, output_path_for,
    should_process,
};
pub use processor::{ProcessResult, process_file};
