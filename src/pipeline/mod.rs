//! Processing pipeline components.

mod coordinator;

pub use coordinator::{
    collect_input_files, output_dir_for, output_path_for, should_process, ProcessCheck,
    ProcessOptions,
};
