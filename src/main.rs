//! Birda CLI entry point.

#![allow(clippy::print_stdout)]
#![allow(clippy::print_stderr)]

fn main() {
    if let Err(e) = birda::run() {
        eprintln!("error: {e}");

        // Print the causes too. Most error variants keep the underlying I/O,
        // TOML or serde failure as a `#[source]`, and thiserror does not render
        // a source in the top-level `Display`, so the reason was being dropped:
        // the user saw "failed to write config file 'X'" and never "Permission
        // denied". That was survivable while the causes were few and obvious.
        // Writing the config atomically added several distinct failure points
        // behind that one message, which are indistinguishable without this.
        let mut cause = std::error::Error::source(&e);
        while let Some(source) = cause {
            eprintln!("  caused by: {source}");
            cause = source.source();
        }

        std::process::exit(1);
    }
}
