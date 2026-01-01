//! Birda CLI entry point.

#![allow(clippy::print_stdout)]
#![allow(clippy::print_stderr)]

fn main() {
    if let Err(e) = birda::run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
