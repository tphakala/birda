//! CLI argument parsing and command handling.

mod args;
pub mod clip;
pub mod help;
pub mod species;
// Crate-visible rather than private so a sibling that restates one of these
// rules can be tested against it directly. `clipper::command` re-applies the
// confidence rule at its library boundary, and #306 is what happens when two
// spellings of one rule are only claimed to agree.
pub(crate) mod validators;

pub use args::{AnalyzeArgs, Cli, Command, ConfigAction, ModelsAction, SortOrder};
pub use clip::ClipArgs;
