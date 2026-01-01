//! Processing pipeline components.

mod coordinator;
mod processor;

pub use coordinator::{
    collect_input_files, output_dir_for, output_path_for, should_process, ProcessCheck,
    ProcessOptions,
};
pub use processor::{process_file, ProcessResult};
