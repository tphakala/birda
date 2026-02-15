# Birda Project Rules

## Overview

Birda is a Rust CLI tool for analyzing audio files using BirdNET and Google Perch AI models. It uses the `birdnet-onnx` crate as its inference library.

## Tech Stack

- **Language:** Rust 1.92, Edition 2024
- **Inference:** birdnet-onnx (local crate at ../rust-birdnet-onnx)
- **Audio Decoding:** symphonia
- **Resampling:** rubato
- **CLI:** clap with derive
- **Config:** toml + serde
- **Async:** tokio
- **Logging:** tracing

## Code Quality Rules

### Strict Static Checking

All code MUST pass these checks before commit:

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

### Linting Configuration

The following lints are enforced in `Cargo.toml`:

```toml
[lints.rust]
unsafe_code = "deny"
missing_docs = "warn"

[lints.clippy]
correctness = { level = "deny", priority = -1 }
pedantic = { level = "warn", priority = -1 }
nursery = { level = "warn", priority = -1 }
cargo = { level = "warn", priority = -1 }

# Restriction lints
unwrap_used = "warn"
expect_used = "warn"
panic = "warn"
todo = "warn"
unimplemented = "warn"
dbg_macro = "warn"

# Allow these where pedantic is too strict
module_name_repetitions = "allow"
similar_names = "allow"
too_many_lines = "allow"
must_use_candidate = "allow"
missing_errors_doc = "allow"
missing_panics_doc = "allow"
```

### No Magic Numbers or Strings

**WRONG:**
```rust
if sample_rate == 48000 {
    chunk_size = 144000;
}
```

**RIGHT:**
```rust
const SAMPLE_RATE_V24: u32 = 48_000;
const CHUNK_SAMPLES_V24: usize = 144_000;

if sample_rate == SAMPLE_RATE_V24 {
    chunk_size = CHUNK_SAMPLES_V24;
}
```

All constants MUST be defined in a dedicated `constants.rs` module or as associated constants on relevant types.

### Input Validation

All external inputs MUST be validated:

1. **CLI arguments:** Use clap's built-in validation (value_parser, range)
2. **Config files:** Validate after parsing, return descriptive errors
3. **Audio files:** Validate format, sample rate, channel count before processing
4. **File paths:** Check existence, permissions, validate against path traversal

Example:
```rust
fn validate_confidence(value: f32) -> Result<f32, ValidationError> {
    if !(0.0..=1.0).contains(&value) {
        return Err(ValidationError::ConfidenceOutOfRange { value });
    }
    Ok(value)
}
```

### Error Handling

**NEVER use:**
- `.unwrap()` - use `.ok_or()` or `?` operator
- `.expect()` - use proper error types
- `panic!()` - return `Result` instead
- `todo!()` / `unimplemented!()` - implement or remove

**ALWAYS:**
- Use `thiserror` for error type definitions
- Provide context with error variants
- Chain errors with `.map_err()` or `?`
- Use meaningful error messages

Example:
```rust
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("unsupported sample rate {rate} Hz, expected {expected} Hz")]
    UnsupportedSampleRate { rate: u32, expected: u32 },

    #[error("failed to open audio file '{path}'")]
    OpenFailed {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}
```

## Security Practices

### Path Handling

- Canonicalize paths before use
- Validate paths don't escape intended directories
- Use `Path::join()` not string concatenation
- Check file permissions before operations

### Lock Files

- Use atomic file creation (`O_CREAT | O_EXCL`)
- Always clean up locks on exit (use RAII guards)
- Store PID/hostname for debugging stale locks

### Input Sanitization

- Validate audio file headers before processing
- Limit resource consumption (max file size, batch size)
- Handle malformed config files gracefully

## Performance Practices

### Memory Management

- Prefer streaming/chunked processing over loading entire files
- Reuse buffers where possible (Vec::clear() + extend)
- Use `Box<[T]>` for fixed-size allocations
- Avoid unnecessary clones

### Async Pipeline

- Use bounded channels to prevent memory exhaustion
- Keep inference thread hot (producer stays ahead)
- Batch GPU operations appropriately

### Profiling

Before optimizing, measure with:
```bash
cargo build --release
perf record ./target/release/birda ...
perf report
```

## Maintainability Practices

### Module Organization

- One concept per module
- Clear public API (`pub` only what's needed)
- Document all public items
- Keep modules under 500 lines

### Testing

- Unit tests in same file (`#[cfg(test)] mod tests`)
- Integration tests in `tests/` directory
- Test error paths, not just happy paths
- Use property-based testing for parsers

### Documentation

- All public items have doc comments
- Include examples in doc comments
- Document panics (if any) and errors
- Keep README.md updated

## Development Tools & Context

### LEANN (Low-storage Vector Index)
LEANN is a local, privacy-focused vector database and RAG system optimized for low storage. It uses AST-aware chunking to maintain semantic code boundaries, making it highly effective for finding relevant logic and gathering context in large or unfamiliar codebases without keyword matching.

#### Commands

- **Index Name:** `birda`
- **Rebuild Index:** `fish -c "leann build birda --docs src tests docs scripts installer README.md Cargo.toml Taskfile.yml action.yml about.toml .github/workflows --use-ast-chunking --force"`
- **Search:** `fish -c "leann search birda '<query>'"` - Fast file/module location (instant)
- **Ask:** `fish -c "leann ask birda '<question>'"` - Comprehensive answers with code context (15-37s)

#### When to Use LEANN

**Prefer LEANN for:**
- Semantic/exploratory searches: "How does audio resampling work?"
- Architecture questions: "What CLI commands are available?"
- Pattern discovery: "What error handling patterns does the codebase use?"
- Context gathering before implementation: "What output formats are supported?"
- Finding code without knowing exact file names or keywords

**Use direct tools (Grep/Glob/Read) for:**
- Exact file path reads when you know the location
- Specific symbol searches when you know the name (class definitions, function names)
- Single file content searches
- Quick syntax checks

#### Effective Query Examples

**Good queries:**
- "How does the audio resampling work? What library is used?"
- "What are the main CLI commands and subcommands?"
- "What output formats are supported and how are they implemented?"
- "Where is configuration loaded from TOML files?"
- "What error handling patterns are used?"

**Less effective:**
- Very specific line-level questions (use Read tool instead)
- Queries about code you've already read in the current session
- File existence checks (use Glob instead)

### Cross-Project Reference: birda-gui
The GUI frontend for Birda is located at `../birda-gui`. 
- When changing CLI output formats (JSON/CSV), ensure compatibility with the GUI.
- Cross-check `birda-gui/src/api/types.ts` when modifying Rust output structures in `src/output/types.rs`.
- LEANN also has an index for `birda-gui` to aid in cross-project navigation.

## Commit Guidelines

- Run `cargo fmt && cargo clippy && cargo test` before commit
- One logical change per commit
- Conventional commit messages: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- Reference issue numbers if applicable

## File Naming

- Use snake_case for all Rust files
- Test files: `tests/integration/<module>_test.rs` or inline `#[cfg(test)]`
- Constants: in dedicated `constants.rs` or associated with types
