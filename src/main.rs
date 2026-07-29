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
        //
        // A cause the message above already spells out is skipped rather than
        // repeated. `#[from]` implies `#[source]`, so a variant that also
        // interpolates that source into its own message carries it twice, and
        // walking the chain blindly printed "I/O error: No such file or
        // directory" followed by "caused by: No such file or directory".
        // Filtering here rather than rewriting the variant keeps each error's
        // `Display` intact for the JSON envelope, which reports the message on
        // its own and cannot walk a chain.
        //
        // Two properties, each closing one way of getting this wrong.
        //
        // `ends_with` rather than `contains`, because appending is the shape
        // the duplication actually takes: a variant that renders its own source
        // puts it at the end, as in `#[error("I/O error: {0}")]`. A substring
        // test would also drop a distinct cause that happened to appear
        // anywhere in its parent, for instance inside an interpolated path.
        //
        // Checked against every line already shown, not just the previous one,
        // so the guarantee holds at any depth. `AudioOpen` and `AudioDecode`
        // wrap boxed errors from the decoder libraries, so a chain here can be
        // deeper than the single `#[from]` hop the rest of the enum uses.
        //
        // Where the two pull against each other, this errs toward printing a
        // line twice rather than hiding a cause: a repeated line is noise, a
        // suppressed one loses the reason the command failed.
        let mut shown = vec![e.to_string()];
        let mut cause = std::error::Error::source(&e);
        while let Some(source) = cause {
            let text = source.to_string();
            if !shown.iter().any(|seen| seen.ends_with(&text)) {
                eprintln!("  caused by: {text}");
            }
            shown.push(text);
            cause = source.source();
        }

        std::process::exit(1);
    }
}
