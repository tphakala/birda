//! Regenerate `registry.json` from the vendored model manifests.
//!
//! Run from the repository root:
//!
//! ```text
//! cargo run --features gen-registry --bin gen-registry
//! ```
//!
//! A maintenance tool, excluded from the default build by the `gen-registry`
//! feature. `tests/registry_generation.rs` asserts the checked-in registry
//! matches what this produces, so forgetting to run it fails CI rather than
//! shipping a stale gallery.

#![allow(clippy::print_stdout)]

fn main() -> std::process::ExitCode {
    let root = env!("CARGO_MANIFEST_DIR");
    match birda::gen_registry::write_registry(root) {
        Ok(()) => {
            println!("Wrote {root}/registry.json");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
