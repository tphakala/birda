# Stdout NDJSON Stream Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `--stdout` flag that outputs detection results and progress as NDJSON stream for GUI integration.

**Architecture:** Extend existing JSON envelope system with new `Detections` event type. Add CLI validation for stdout constraints. Modify processor to emit detections via reporter instead of writing files when stdout mode is active.

**Tech Stack:** Rust 2024, clap (CLI), serde/serde_json (serialization), existing JSON envelope infrastructure

---

## Task 1: Add Detections Event Type

**Files:**
- Modify: `src/output/json_envelope.rs:39-59`
- Test: Inline tests in same file

**Step 1: Write failing test for Detections event type**

Add to `src/output/json_envelope.rs` test module (after existing tests):

```rust
#[test]
fn test_detections_event_serialization() {
    assert_eq!(
        serde_json::to_string(&EventType::Detections).expect("serialize"),
        "\"detections\""
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_detections_event_serialization`
Expected: FAIL with compilation error "no variant named `Detections`"

**Step 3: Add Detections variant to EventType enum**

In `src/output/json_envelope.rs:39-59`, add new variant after `Cancelled`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    /// Analysis pipeline starting.
    PipelineStarted,
    /// Starting to process a file.
    FileStarted,
    /// Periodic progress update.
    Progress,
    /// File processing finished.
    FileCompleted,
    /// All files processed.
    PipelineCompleted,
    /// Final result.
    Result,
    /// Error occurred.
    Error,
    /// Operation cancelled.
    Cancelled,
    /// Detection results for a file.
    Detections,
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test test_detections_event_serialization`
Expected: PASS

**Step 5: Commit**

```bash
git add src/output/json_envelope.rs
git commit -m "feat: add Detections event type to JSON envelope"
```

---

## Task 2: Add DetectionInfo and DetectionsPayload Types

**Files:**
- Modify: `src/output/json_envelope.rs` (add after `CancelledPayload` around line 290)
- Test: Inline tests in same file

**Step 1: Write failing test for DetectionInfo serialization**

Add to `src/output/json_envelope.rs` test module:

```rust
#[test]
fn test_detection_info_serialization() {
    let info = DetectionInfo {
        species: "Parus major_Great Tit".to_string(),
        common_name: "Great Tit".to_string(),
        scientific_name: "Parus major".to_string(),
        confidence: 0.95,
        start_time: 0.0,
        end_time: 3.0,
    };
    let json = serde_json::to_string(&info).expect("serialize");
    let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");
    let expected = serde_json::json!({
        "species": "Parus major_Great Tit",
        "common_name": "Great Tit",
        "scientific_name": "Parus major",
        "confidence": 0.95,
        "start_time": 0.0,
        "end_time": 3.0
    });
    assert_eq!(actual, expected);
}

#[test]
fn test_detections_payload_serialization() {
    let payload = DetectionsPayload {
        file: PathBuf::from("audio.wav"),
        detections: vec![DetectionInfo {
            species: "Parus major_Great Tit".to_string(),
            common_name: "Great Tit".to_string(),
            scientific_name: "Parus major".to_string(),
            confidence: 0.95,
            start_time: 0.0,
            end_time: 3.0,
        }],
    };
    let json = serde_json::to_string(&payload).expect("serialize");
    assert!(json.contains("\"file\":\"audio.wav\""));
    assert!(json.contains("\"detections\""));
    assert!(json.contains("\"Great Tit\""));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test test_detection_info_serialization test_detections_payload_serialization`
Expected: FAIL with compilation error "cannot find type `DetectionInfo`"

**Step 3: Add DetectionInfo and DetectionsPayload structs**

Add after `CancelledPayload` in `src/output/json_envelope.rs`:

```rust
// ============================================================================
// Detections Event Payload
// ============================================================================

/// Payload for `detections` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionsPayload {
    /// Path to source file.
    pub file: PathBuf,
    /// All detections found in the file.
    pub detections: Vec<DetectionInfo>,
}

/// Information about a single detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionInfo {
    /// Full species label (e.g., "Parus major_Great Tit").
    pub species: String,
    /// Common name.
    pub common_name: String,
    /// Scientific name.
    pub scientific_name: String,
    /// Confidence score (0.0-1.0).
    pub confidence: f32,
    /// Start time in seconds.
    pub start_time: f32,
    /// End time in seconds.
    pub end_time: f32,
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test test_detection_info_serialization test_detections_payload_serialization`
Expected: PASS

**Step 5: Commit**

```bash
git add src/output/json_envelope.rs
git commit -m "feat: add DetectionInfo and DetectionsPayload types"
```

---

## Task 3: Export New Types from output Module

**Files:**
- Modify: `src/output/mod.rs:17-25`

**Step 1: Add exports for new types**

In `src/output/mod.rs`, update the `json_envelope` exports (around line 17):

```rust
pub use json_envelope::{
    AvailableModelEntry, AvailableModelsPayload, BatchProgress, CancelReason, CancelledPayload,
    ClipExtractionEntry, ClipExtractionPayload, ConfigPathPayload, ConfigPayload,
    DetectionInfo, DetectionsPayload, DownloadProgress,
    ErrorPayload, ErrorSeverity, EventType, FileCompletedPayload, FileErrorInfo, FileProgress,
    FileStartedPayload, FileStatus, JsonEnvelope, ModelCheckEntry, ModelCheckPayload, ModelDetails,
    ModelEntry, ModelInfoPayload, ModelListPayload, PipelineCompletedPayload,
    PipelineStartedPayload, PipelineStatus, ProgressPayload, ProviderInfo, ProvidersPayload,
    ResultType, SPEC_VERSION, SpeciesEntry, SpeciesListPayload, VersionPayload,
};
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add src/output/mod.rs
git commit -m "feat: export DetectionInfo and DetectionsPayload types"
```

---

## Task 4: Add detections() Method to ProgressReporter Trait

**Files:**
- Modify: `src/output/reporter.rs:19-55`
- Test: Inline tests in same file

**Step 1: Write failing test for detections reporting**

Add to `src/output/reporter.rs` test module:

```rust
#[test]
fn test_json_reporter_emits_detections() {
    use crate::output::{Detection, DetectionInfo, DetectionsPayload};

    let buffer = Arc::new(Mutex::new(Vec::new()));
    let writer = TestWriter {
        buffer: buffer.clone(),
    };

    let reporter = JsonProgressReporter::with_writer(OutputMode::Ndjson, writer);

    let detections = vec![Detection {
        file_path: PathBuf::from("test.wav"),
        start_time: 0.0,
        end_time: 3.0,
        scientific_name: "Parus major".to_string(),
        common_name: "Great Tit".to_string(),
        confidence: 0.95,
        metadata: Default::default(),
    }];

    reporter.detections(Path::new("test.wav"), &detections);

    let output = buffer.lock().expect("lock");
    let output_str = String::from_utf8_lossy(&output);
    assert!(output_str.contains("\"event\":\"detections\""));
    assert!(output_str.contains("\"Great Tit\""));
    assert!(output_str.contains("\"confidence\":0.95"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_json_reporter_emits_detections`
Expected: FAIL with compilation error "no method named `detections`"

**Step 3: Add detections() method to ProgressReporter trait**

In `src/output/reporter.rs`, add method to `ProgressReporter` trait (after `cancelled` method, around line 54):

```rust
/// Report detection results.
fn detections(&self, file: &Path, detections: &[Detection]);
```

**Step 4: Implement detections() for JsonProgressReporter**

In `src/output/reporter.rs`, add implementation in `impl ProgressReporter for JsonProgressReporter` (after `cancelled` method, around line 373):

```rust
fn detections(&self, file: &Path, detections: &[Detection]) {
    use crate::output::{DetectionInfo, DetectionsPayload};

    let detection_infos: Vec<DetectionInfo> = detections
        .iter()
        .map(|d| DetectionInfo {
            species: format!("{}_{}", d.scientific_name, d.common_name),
            common_name: d.common_name.clone(),
            scientific_name: d.scientific_name.clone(),
            confidence: d.confidence,
            start_time: d.start_time,
            end_time: d.end_time,
        })
        .collect();

    self.emit(
        EventType::Detections,
        DetectionsPayload {
            file: file.to_path_buf(),
            detections: detection_infos,
        },
    );
}
```

**Step 5: Implement detections() for NullReporter**

In `src/output/reporter.rs`, add implementation in `impl ProgressReporter for NullReporter` (after `cancelled` method, around line 405):

```rust
fn detections(&self, _file: &Path, _detections: &[Detection]) {}
```

**Step 6: Run test to verify it passes**

Run: `cargo test test_json_reporter_emits_detections`
Expected: PASS

**Step 7: Run all reporter tests**

Run: `cargo test -p birda reporter::`
Expected: All PASS

**Step 8: Commit**

```bash
git add src/output/reporter.rs
git commit -m "feat: add detections() method to ProgressReporter trait"
```

---

## Task 5: Add --stdout Flag to CLI

**Files:**
- Modify: `src/cli/args.rs:169-338`
- Test: Inline tests in same file

**Step 1: Write failing test for stdout flag parsing**

Add to `src/cli/args.rs` test module:

```rust
#[test]
fn test_cli_parse_with_stdout() {
    let cli = Cli::try_parse_from(["birda", "--stdout", "test.wav"]);
    assert!(cli.is_ok());
    let cli = cli.unwrap();
    assert!(cli.analyze.stdout);
}

#[test]
fn test_cli_stdout_flag_exists() {
    let cli = Cli::try_parse_from(["birda", "--stdout", "test.wav"]);
    assert!(cli.is_ok());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test test_cli_parse_with_stdout test_cli_stdout_flag_exists`
Expected: FAIL with compilation error "no field `stdout`"

**Step 3: Add stdout field to AnalyzeArgs**

In `src/cli/args.rs`, add field to `AnalyzeArgs` struct (after `stale_lock_timeout`, around line 337):

```rust
/// Write results to stdout as NDJSON stream (single file only).
#[arg(long)]
pub stdout: bool,
```

**Step 4: Run tests to verify they pass**

Run: `cargo test test_cli_parse_with_stdout test_cli_stdout_flag_exists`
Expected: PASS

**Step 5: Run all CLI tests**

Run: `cargo test -p birda cli::`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/cli/args.rs
git commit -m "feat: add --stdout CLI flag"
```

---

## Task 6: Add Validation for Stdout Constraints

**Files:**
- Modify: `src/main.rs` (add validation before calling run_analysis)
- Test: Manual testing (integration test in later task)

**Step 1: Add stdout validation before analysis**

Find the section in `src/main.rs` where `run_analysis` is called. Add validation before that call:

```rust
// Validate stdout mode constraints
if args.analyze.stdout {
    // Must have exactly one input file
    if args.inputs.len() != 1 {
        return Err(crate::error::Error::Validation {
            message: "--stdout requires exactly one input file".to_string(),
        });
    }

    // Cannot use with --output-dir
    if args.analyze.output_dir.is_some() {
        return Err(crate::error::Error::Validation {
            message: "--stdout cannot be used with --output-dir".to_string(),
        });
    }

    // Cannot use with --combine
    if args.analyze.combine {
        return Err(crate::error::Error::Validation {
            message: "--stdout cannot be used with --combine".to_string(),
        });
    }

    // Cannot use with --format
    if args.analyze.format.is_some() {
        return Err(crate::error::Error::Validation {
            message: "--stdout cannot be used with --format (detections are output in JSON format automatically)".to_string(),
        });
    }
}
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: SUCCESS

**Step 3: Test manually**

Run: `cargo run -- --stdout test1.wav test2.wav`
Expected: Error message "error: --stdout requires exactly one input file"

Run: `cargo run -- --stdout --output-dir /tmp test.wav`
Expected: Error message "error: --stdout cannot be used with --output-dir"

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: add validation for --stdout constraints"
```

---

## Task 7: Auto-Enable NDJSON Mode When Stdout Active

**Files:**
- Modify: `src/main.rs` (after stdout validation, before run_analysis)

**Step 1: Auto-enable NDJSON output mode when stdout is active**

In `src/main.rs`, after the stdout validation added in Task 6:

```rust
// Auto-enable NDJSON mode for stdout
let output_mode = if args.analyze.stdout {
    crate::config::OutputMode::Ndjson
} else {
    args.output_mode.unwrap_or_default()
};
```

**Step 2: Pass output_mode to downstream functions**

Update the code that uses `args.output_mode` to use the `output_mode` variable instead.

**Step 3: Verify compilation**

Run: `cargo check`
Expected: SUCCESS

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: auto-enable NDJSON mode when --stdout is active"
```

---

## Task 8: Modify Processor to Emit Detections Event

**Files:**
- Modify: `src/pipeline/processor.rs:491-502`

**Step 1: Add stdout_mode parameter to process_file**

In `src/pipeline/processor.rs`, update `process_file` function signature (around line 298):

```rust
#[allow(clippy::too_many_arguments)]
pub fn process_file(
    input_path: &Path,
    output_dir: &Path,
    classifier: &BirdClassifier,
    formats: &[OutputFormat],
    min_confidence: f32,
    overlap: f32,
    batch_size: usize,
    csv_columns: &[String],
    progress_enabled: bool,
    csv_bom_enabled: bool,
    model_name: &str,
    range_filter_params: Option<(f64, f64, u8)>,
    reporter: Option<&dyn ProgressReporter>,
) -> Result<ProcessResult> {
```

**Step 2: Replace file writing with detections event emission**

In `src/pipeline/processor.rs`, replace the file writing loop (around line 491-502):

```rust
// Write output files or emit detections event
if let Some(reporter) = reporter {
    // Stdout mode - emit detections event instead of writing files
    reporter.detections(input_path, &detections);
} else {
    // File mode - write output files
    for format in formats {
        write_output(
            input_path,
            output_dir,
            *format,
            &detections,
            csv_columns,
            csv_bom_enabled,
            json_config.as_ref(),
        )?;
    }
}
```

**Step 3: Verify compilation**

Run: `cargo check`
Expected: SUCCESS (may have warnings about unused parameters)

**Step 4: Update all callers of process_file**

Find all places where `process_file` is called and update to pass `reporter` parameter.

**Step 5: Run all tests**

Run: `cargo test`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/pipeline/processor.rs
git commit -m "feat: emit detections event in stdout mode instead of writing files"
```

---

## Task 9: Update Main Analysis Flow

**Files:**
- Modify: `src/main.rs` (run_analysis function or equivalent)

**Step 1: Pass reporter to processor when stdout mode active**

In the main analysis flow, when calling `process_file`:

```rust
let reporter = if args.analyze.stdout {
    Some(reporter.as_ref() as &dyn ProgressReporter)
} else {
    None
};

// Call process_file with reporter
process_file(
    // ... existing parameters ...
    reporter,
)?;
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: wire up reporter to processor for stdout mode"
```

---

## Task 10: Integration Test for Stdout Mode

**Files:**
- Create: `tests/stdout_integration.rs`

**Step 1: Create integration test file**

Create `tests/stdout_integration.rs`:

```rust
//! Integration tests for --stdout mode.

use assert_cmd::Command;
use predicates::prelude::*;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_stdout_requires_single_file() {
    let mut cmd = Command::cargo_bin("birda").expect("binary");
    cmd.arg("--stdout")
        .arg("file1.wav")
        .arg("file2.wav");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--stdout requires exactly one input file"));
}

#[test]
fn test_stdout_conflicts_with_output_dir() {
    let mut cmd = Command::cargo_bin("birda").expect("binary");
    cmd.arg("--stdout")
        .arg("--output-dir")
        .arg("/tmp")
        .arg("test.wav");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--stdout cannot be used with --output-dir"));
}

#[test]
fn test_stdout_conflicts_with_combine() {
    let mut cmd = Command::cargo_bin("birda").expect("binary");
    cmd.arg("--stdout")
        .arg("--combine")
        .arg("test.wav");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--stdout cannot be used with --combine"));
}

#[test]
fn test_stdout_conflicts_with_format() {
    let mut cmd = Command::cargo_bin("birda").expect("binary");
    cmd.arg("--stdout")
        .arg("--format")
        .arg("csv")
        .arg("test.wav");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--stdout cannot be used with --format"));
}

/// Test that stdout mode outputs NDJSON stream.
/// Note: This test requires a valid audio file and model to run.
/// It's marked as ignored and can be run manually with audio files.
#[test]
#[ignore]
fn test_stdout_outputs_ndjson() {
    // Create a small test audio file (silence)
    let mut temp_file = NamedTempFile::new().expect("create temp file");

    // Write minimal WAV header (44 bytes) + some silence
    let wav_header = create_wav_header(48000, 1, 16, 48000); // 1 second
    temp_file.write_all(&wav_header).expect("write header");
    temp_file.write_all(&vec![0u8; 96000]).expect("write samples");
    temp_file.flush().expect("flush");

    let path = temp_file.path();

    let mut cmd = Command::cargo_bin("birda").expect("binary");
    cmd.arg("--stdout")
        .arg(path);

    let output = cmd.output().expect("run command");
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Verify NDJSON structure
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(!lines.is_empty(), "Should output at least one line");

    // Each line should be valid JSON
    for line in lines {
        let parsed: serde_json::Value = serde_json::from_str(line)
            .expect("each line should be valid JSON");
        assert!(parsed.get("event").is_some(), "Should have event field");
        assert!(parsed.get("timestamp").is_some(), "Should have timestamp field");
    }

    // Should contain detections event
    assert!(stdout.contains("\"event\":\"detections\""));
}

/// Create a minimal WAV file header.
fn create_wav_header(sample_rate: u32, channels: u16, bits_per_sample: u16, num_samples: u32) -> Vec<u8> {
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample) / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_size = num_samples * u32::from(channels) * u32::from(bits_per_sample) / 8;
    let file_size = 36 + data_size;

    let mut header = Vec::new();

    // RIFF header
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&file_size.to_le_bytes());
    header.extend_from_slice(b"WAVE");

    // fmt chunk
    header.extend_from_slice(b"fmt ");
    header.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    header.extend_from_slice(&1u16.to_le_bytes()); // audio format (PCM)
    header.extend_from_slice(&channels.to_le_bytes());
    header.extend_from_slice(&sample_rate.to_le_bytes());
    header.extend_from_slice(&byte_rate.to_le_bytes());
    header.extend_from_slice(&block_align.to_le_bytes());
    header.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    header.extend_from_slice(b"data");
    header.extend_from_slice(&data_size.to_le_bytes());

    header
}
```

**Step 2: Add dependencies to Cargo.toml**

In `Cargo.toml`, add to `[dev-dependencies]`:

```toml
assert_cmd = "2.0"
predicates = "3.0"
```

**Step 3: Run non-ignored tests**

Run: `cargo test --test stdout_integration`
Expected: Tests should PASS (excluding ignored test)

**Step 4: Commit**

```bash
git add tests/stdout_integration.rs Cargo.toml
git commit -m "test: add integration tests for --stdout mode"
```

---

## Task 11: Update Documentation

**Files:**
- Modify: `README.md` (if exists, add --stdout usage example)
- Modify: `docs/design/2026-02-05-stdout-ndjson-mode-design.md` (mark as implemented)

**Step 1: Add usage example to README**

If `README.md` exists, add example under CLI usage section:

```markdown
### Stdout Mode (GUI Integration)

Output detection results as NDJSON stream for integration with GUI applications:

```bash
birda --stdout audio.wav
```

Output format: Each line is a JSON object with event type and payload. Progress updates and detection results are streamed in real-time.

Example output:
```json
{"event":"pipeline_started","timestamp":"...","payload":{"total_files":1,"model":"birdnet-v24","min_confidence":0.1}}
{"event":"file_started","timestamp":"...","payload":{"file":"audio.wav","index":0,"estimated_segments":100}}
{"event":"progress","timestamp":"...","payload":{"file":{"percent":10.0,"segments_processed":10}}}
{"event":"detections","timestamp":"...","payload":{"file":"audio.wav","detections":[...]}}
{"event":"file_completed","timestamp":"...","payload":{"file":"audio.wav","status":"processed"}}
```

Constraints:
- Only one input file allowed
- Cannot combine with `--output-dir`, `--combine`, or `--format`
- Progress bars automatically disabled (NDJSON mode)
```

**Step 2: Mark design document as implemented**

Update `docs/design/2026-02-05-stdout-ndjson-mode-design.md`:

```markdown
**Date:** 2026-02-05
**Status:** Implemented
```

**Step 3: Commit**

```bash
git add README.md docs/design/2026-02-05-stdout-ndjson-mode-design.md
git commit -m "docs: add --stdout usage documentation"
```

---

## Task 12: Final Testing and Cleanup

**Files:**
- All files

**Step 1: Run full test suite**

Run: `cargo test`
Expected: All tests PASS

**Step 2: Run clippy**

Run: `cargo clippy -- -D warnings`
Expected: No warnings or errors

**Step 3: Run fmt check**

Run: `cargo fmt --check`
Expected: No formatting issues

**Step 4: Build release**

Run: `cargo build --release`
Expected: SUCCESS

**Step 5: Manual end-to-end test (if audio file and model available)**

Run: `cargo run --release -- --stdout <audio_file.wav>`
Expected: NDJSON stream with pipeline, progress, detections, and completion events

**Step 6: Final commit if any fixes needed**

```bash
git add .
git commit -m "chore: final cleanup for stdout mode"
```

---

## Verification Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Clippy reports no warnings
- [ ] Code is properly formatted
- [ ] All new types are documented
- [ ] CLI help text is clear
- [ ] Validation errors are user-friendly
- [ ] NDJSON output is valid (each line parses as JSON)
- [ ] Events are emitted in correct order
- [ ] Progress updates work correctly
- [ ] Detections event contains all expected fields
- [ ] Stdout mode rejects invalid usage (multiple files, conflicts)

## Notes

- The `reporter` parameter is passed as `Option<&dyn ProgressReporter>` to allow conditional detection emission
- When `reporter.is_some()`, file writing is skipped entirely
- NDJSON mode automatically disables indicatif progress bars
- All validation happens before processing starts (fail fast)
- The DetectionInfo type is separate from the internal Detection type to allow clean serialization
