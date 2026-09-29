# Birda Project Rules

## Overview

Birda is a Rust CLI tool for analyzing audio files using BirdNET and Google Perch AI models. It uses the `birdnet-onnx` crate as its inference library.

## Tech Stack

- **Language:** Rust, edition 2024. The minimum supported version is `rust-version` in `Cargo.toml`, and the CI MSRV job builds on exactly that toolchain.
- **Inference:** `birdnet-onnx` from crates.io (source: `tphakala/rust-birdnet-onnx`), on top of `ort`. ONNX Runtime is loaded at run time, because both dependencies are declared with `load-dynamic`.
- **Audio decoding:** `symphonia`, pinned to the `feature/rf64-support` branch of the `tphakala/Symphonia` fork
- **Resampling:** `rubato`, fed through `audioadapter-buffers`
- **CLI:** `clap` with derive
- **Config:** `toml` + `serde`
- **Output:** `csv`, `serde_json`, `arrow` + `parquet`
- **Network:** `reqwest` with rustls, for model downloads and self-update
- **Async:** `tokio`, for network work only. Each command that downloads builds its own runtime (`handle_update_command` and the `handle_*_install` functions in `src/lib.rs`).
- **Logging:** `tracing`

## Cargo Features

- `cuda` (default): CUDA support in `birdnet-onnx`.
- `gen-registry`: builds the `gen-registry` maintenance binary that regenerates `registry.json`. It is not part of the shipped CLI.

There is no `load-dynamic` feature: ONNX Runtime is always loaded at run time, because both dependencies request `load-dynamic` unconditionally and `src/inference/runtime.rs` needs `ort::init_from`.

CI builds with `--no-default-features`.

## Code Quality Rules

### Checks

CI (`.github/workflows/ci.yml`) runs these, and all of them must pass:

```bash
cargo fmt --check
cargo clippy --no-default-features --features gen-registry --all-targets -- -D warnings
cargo test --no-default-features --no-fail-fast
cargo test --no-default-features --features gen-registry --no-fail-fast --lib gen_registry
cargo test --no-default-features --features gen-registry --no-fail-fast --test registry_generation
```

The MSRV job also runs `cargo check --locked --no-default-features --all-targets` on the toolchain named by `rust-version`. Locally that is `cargo +<rust-version> check --locked --no-default-features --all-targets`.

`task check` runs the same commands (the MSRV job aside), and the pre-commit hook from `task install-hooks` runs the fmt and clippy lines when a commit stages `.rs` files. `CLIPPY_FLAGS` in `Taskfile.yml` and the clippy line in `hooks/pre-commit` must match the Lint job. Run `task check` before pushing.

### Lints

Lint levels live in `[lints]` in `Cargo.toml`, which is the source of truth. In short: `unsafe_code` is denied; `missing_docs` warns; the clippy `pedantic`, `nursery` and `cargo` groups warn; and so do `unwrap_used`, `expect_used`, `panic`, `todo`, `unimplemented` and `dbg_macro`. CI passes `-D warnings`, so any warning fails the build.

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

### CLI Help Text

clap turns a `///` doc comment on an `#[arg]` field or a subcommand variant into `--help` output (see `src/cli/args.rs` and `src/cli/clip.rs`). Keep the `///` text to what a user needs, and put implementation rationale in a `//` comment above it. After touching any argument, build and read the output of `birda --help` and `birda <command> --help`.

### Error Handling

In production code, **NEVER use:**
- `.unwrap()`: use `.ok_or()` or the `?` operator
- `.expect()`: use proper error types
- `panic!()`: return `Result` instead
- `todo!()` / `unimplemented!()`: implement or remove

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

Test code may unwrap and panic. Unit tests are covered by the `#![cfg_attr(test, allow(...))]` at the top of `src/lib.rs`, and each file in `tests/` opens with its own `#![allow(...)]`; copy that header into a new integration test file.

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

### Inference Pipeline

The pipeline runs on std threads connected by a bounded `sync_channel` (`src/pipeline/processor.rs`), not on tokio.

- Keep channels bounded to prevent memory exhaustion
- Keep inference thread hot (producer stays ahead)
- Batch GPU operations appropriately

### Profiling

Before optimizing, measure. On Linux:
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
- `src/lib.rs` holds the command handlers and is already very large. Put new logic in the module that owns the concept and keep the handlers thin.
- The intended module layering is declared in `.sentrux/rules.toml`: a module may import only from layers more foundational than its own, and cycles are not allowed. CI does not check it.

### Testing

- Unit tests in same file (`#[cfg(test)] mod tests`)
- Integration tests are flat files in `tests/`, one per feature area. Fixtures live in `tests/fixtures/`, which `.gitignore` re-includes so audio fixtures there are tracked.
- Any test that runs the binary or touches config must point `BIRDA_CONFIG_DIR` (`constants::CONFIG_DIR_ENV`) at a temporary directory; `config_dir()` and `data_dir()` both honour it. Setting `HOME` or `XDG_CONFIG_HOME` alone does not redirect the config on Windows (see the note on `CONFIG_DIR_ENV` in `src/constants.rs`), so such a test would write to the developer's real profile.
- Test error paths, not just happy paths
- Use property-based testing for parsers
- Prefer `assert_eq!` on the whole rendered value over `contains`. A substring check passes when the expected text sits inside a different value: `"90.0"` is inside `"-90.0"`. `test_invalid_padding_renders_the_value_and_the_ceiling` in `src/error.rs` shows the pattern. Where a whole-value compare is impractical, scope the search to the part that should hold the value, and assert that the neighbouring values are absent.
- For a test that pins a bug fix or a guard, revert that one line and confirm the test fails on an assertion, not on a compile error. When the same guard exists at several layers (clap parser, library entry point, file parser), a test for one layer can pass on a neighbouring layer's error: revert the guards one at a time, and assert on something only the tested layer produces.

### Documentation

- All public items have doc comments
- Include examples in doc comments
- Document panics (if any) and errors
- User-facing docs are `README.md` and `docs/*.md`. Update them when a flag, config key or output format changes.
- Working notes (plans, designs, research) go in the gitignored `docs/plans/`, `docs/design/`, `docs/research/` or `docs/notes/`, not in tracked docs.

## Generated Files

Don't hand-edit these:

- `registry.json`: generated from `manifests/*.json` and `registry-sources.toml`. After editing either, run `cargo run --features gen-registry --bin gen-registry` and commit the result. The generator bumps `registry_version` on every content change, and CI fails when `registry.json` does not match its output.
- `THIRD_PARTY_LICENSES.txt`: regenerated by `.github/workflows/licenses.yml` with `cargo-about` (`about.toml` selects the licenses, `about.hbs` formats them). It runs on pushes to `main` that change `Cargo.toml`, `Cargo.lock`, `about.toml` or `about.hbs`, and opens a `chore: update third-party licenses` PR.

## Dependencies

- `ort` and `ort-sys` move only together with `ONNXRUNTIME_VERSION` in `.github/workflows/release.yml`, which pins the native runtime and the CUDA and cuDNN set shipped with every release. Because of `load-dynamic`, an ABI mismatch compiles and passes CI, then fails when a released binary loads the library. Dependabot ignores both crates (`.github/dependabot.yml`). Keep them at their current version in routine `cargo update` sweeps (check with `cargo update --dry-run --verbose`), and bump them in a change of their own that updates `release.yml` and runs real inference.
- `rubato` and `audioadapter-buffers` must resolve to the same major version: `src/audio/resample.rs` builds `audioadapter_buffers::direct::SequentialSlice` and hands it to rubato. Bump both in one change; `cargo tree -i audioadapter-buffers` should show a single copy.
- `arrow` and `parquet` must share a major version, since `parquet` depends on the `arrow-*` crates of its own major and `src/output/parquet.rs` passes values between them. Bump both in one change.
- Dependabot opens a separate PR for each crate in a coupled pair. Combine them on one branch and close the other. `cargo tree -d -e normal` lists duplicated crates.

## CI Workflows

- `ci.yml`: lint, test and MSRV (see Checks)
- `release.yml`: release builds; pins the ONNX Runtime, CUDA and cuDNN versions
- `licenses.yml`: see Generated Files
- `action-test.yml`: tests the composite GitHub Action in `action.yml` and its `scripts/`

After editing a workflow, run `actionlint` on it. `task install-hooks` installs a pre-commit hook that does this when `actionlint` is on the PATH. Actions evaluates expressions across the whole file before the shell runs, shell comments inside `run:` blocks included, so never write a bare or empty `${{` delimiter in a comment: it fails the entire workflow.

## Cross-Project Reference: birda-gui

The GUI frontend is a sibling checkout at `../birda-gui` and consumes birda's JSON output.

- The JSON envelope is defined in `src/output/json_envelope.rs` (which carries `SPEC_VERSION`) and `src/output/types.rs`. The GUI mirrors these types in `../birda-gui/shared/types.ts`; cross-check it when changing either Rust file.
- When changing CLI output formats (JSON/CSV), ensure compatibility with the GUI, and update `docs/json-output.md` for any JSON change.

## Commit Guidelines

- Run the checks above before pushing
- One logical change per commit
- Conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `perf:`, `test:`, `docs:`, `build:`, `ci:`, `chore:`, and `deps:` for dependency updates. Dependabot writes `deps:` for cargo and `ci:` for GitHub Actions.
- Reference issue numbers if applicable

## File Naming

- Use snake_case for all Rust files
- Tests: inline `#[cfg(test)] mod tests`, or a flat file in `tests/`
- Constants: in dedicated `constants.rs` or associated with types
