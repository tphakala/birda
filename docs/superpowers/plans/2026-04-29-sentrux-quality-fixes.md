# Sentrux Quality Signal Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix validated sentrux findings to improve birda's quality signal from 7449 toward 8000 by addressing the equality bottleneck (the two monster functions), removing dead code, and adding a trait default for no-op `write_header()`.

**Architecture:** Four independent changes: (1) extract `from_config` execution provider selection into a helper, (2) introduce a `ProcessingConfig` struct to bundle `process_file`'s 15 parameters, (3) add a default `write_header` on the `OutputWriter` trait, (4) remove one dead test helper. Changes 1-4 are independent and can be done in any order.

**Tech Stack:** Rust 1.92, clippy pedantic+nursery lints, existing test suite.

**Validation commands:**
```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
~/bin/sentrux check .
```

---

### Task 1: Add default `write_header()` to `OutputWriter` trait

**Files:**
- Modify: `src/output/writer.rs:9`
- Modify: `src/output/audacity.rs:26-29`
- Modify: `src/output/json.rs` (its `write_header` impl)
- Modify: `src/output/parquet.rs:122-124`

- [ ] **Step 1: Add default method body to the trait**

In `src/output/writer.rs`, change the trait definition:

```rust
/// Trait for writing detection results.
pub trait OutputWriter {
    /// Write the file header (if applicable).
    ///
    /// Default is a no-op. Override for formats that need a header (CSV, Raven, Kaleidoscope).
    fn write_header(&mut self) -> Result<()> {
        Ok(())
    }

    /// Write a single detection.
    fn write_detection(&mut self, detection: &Detection) -> Result<()>;

    /// Finalize the output (flush, close, etc.).
    fn finalize(&mut self) -> Result<()>;
}
```

- [ ] **Step 2: Remove no-op `write_header` implementations**

Remove the `write_header` method from these three `impl OutputWriter` blocks (they just return `Ok(())`):

- `src/output/audacity.rs` lines 26-29 (entire `fn write_header` block)
- `src/output/json.rs` (its `fn write_header` that returns `Ok(())` with a comment about writing at finalize)
- `src/output/parquet.rs` lines 122-124 (entire `fn write_header` block)

Keep `write_header` in `src/output/csv.rs`, `src/output/kaleidoscope.rs`, and `src/output/raven.rs` since those write actual content.

- [ ] **Step 3: Run validation**

Run: `cargo fmt --check && cargo clippy -- -D warnings && cargo test`
Expected: All pass. Existing tests call `write_header()` on audacity/parquet/json writers and should still work via the default.

- [ ] **Step 4: Commit**

```bash
git add src/output/writer.rs src/output/audacity.rs src/output/json.rs src/output/parquet.rs
git commit -m "refactor: add default no-op write_header to OutputWriter trait

Formats that don't need a header (Audacity, JSON, Parquet) no longer
need to implement write_header. Only CSV, Kaleidoscope, and Raven
override with actual header content."
```

---

### Task 2: Remove dead `create_test_model` helper

**Files:**
- Modify: `src/registry/license.rs:93-124`

- [ ] **Step 1: Verify the function is unused**

Run: `grep -rn 'create_test_model' src/ tests/`
Expected: Only one match at the definition site (`src/registry/license.rs:93`). No call sites.

- [ ] **Step 2: Remove the dead function**

Delete lines 93-124 in `src/registry/license.rs` (the entire `fn create_test_model` function). Also remove the unused imports it pulled in on line 91:

Before:
```rust
use crate::registry::types::{FileInfo, LabelsInfo, LanguageVariant, ModelFiles};
```

After: remove this line entirely (the three tests below construct `LicenseInfo` directly, which is imported via `use super::*`).

- [ ] **Step 3: Run validation**

Run: `cargo fmt --check && cargo clippy -- -D warnings && cargo test`
Expected: All pass. The three remaining tests in this module don't use `create_test_model`.

- [ ] **Step 4: Commit**

```bash
git add src/registry/license.rs
git commit -m "refactor: remove unused create_test_model test helper"
```

---

### Task 3: Extract execution provider selection from `from_config`

The `from_config` function (434 lines) is dominated by a ~220-line `match device` block that selects and configures execution providers. Extract this into a dedicated helper.

**Files:**
- Modify: `src/inference/classifier.rs:166-599`

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `cargo test --lib inference::classifier`
Expected: All pass.

- [ ] **Step 2: Define the return type for the extracted helper**

Add this struct above `from_config` (around line 164):

```rust
/// Result of selecting and configuring an execution provider.
struct ProviderSelection {
    builder: ClassifierBuilder,
    device_name: &'static str,
    status: ExecutionProviderStatus,
}
```

- [ ] **Step 3: Extract the `select_execution_provider` function**

Create a new function that contains the entire `match device { ... }` block from `from_config` (lines 218-441). The function signature:

```rust
/// Select and configure the execution provider based on device setting.
fn select_execution_provider(
    builder: ClassifierBuilder,
    device: InferenceDevice,
    available_providers: &[birdnet_onnx::ExecutionProviderInfo],
) -> Result<ProviderSelection> {
    // GPU provider priority order
    #[allow(unused_mut)]
    let mut gpu_priority = vec![
        (ExecutionProviderInfo::TensorRt, "TensorRT"),
        (ExecutionProviderInfo::Cuda, "CUDA"),
        (ExecutionProviderInfo::DirectMl, "DirectML"),
        (ExecutionProviderInfo::Rocm, "ROCm"),
        (ExecutionProviderInfo::OpenVino, "OpenVINO"),
    ];

    #[cfg(not(target_os = "macos"))]
    gpu_priority.insert(3, (ExecutionProviderInfo::CoreMl, "CoreML"));

    match device {
        // ... move the entire match block here, adjusting return values
        // Each arm returns Ok(ProviderSelection { builder, device_name, status })
        // instead of a tuple (builder, actual_device_msg, ep_status)
    }
}
```

Each match arm currently returns a 3-tuple `(builder, &str, ExecutionProviderStatus)`. Change each to return `Ok(ProviderSelection { builder, device_name: <str>, status: <ep_status> })`.

The `InferenceDevice::Cpu` arm becomes:
```rust
InferenceDevice::Cpu => {
    info!("Requested device: CPU");
    Ok(ProviderSelection {
        builder,
        device_name: "CPU",
        status: ExecutionProviderStatus {
            requested: "cpu".to_string(),
            actual: "CPU".to_string(),
            fallback_reason: None,
        },
    })
}
```

The explicit provider arms (`Cuda`, `TensorRt`, etc.) call `configure_explicit_provider` and wrap with:
```rust
InferenceDevice::Cuda => {
    let (builder, name, status) = configure_explicit_provider(
        builder, available_providers, ExecutionProviderInfo::Cuda, "CUDA",
    )?;
    Ok(ProviderSelection { builder, device_name: name, status })
}
```

- [ ] **Step 4: Update `from_config` to call the extracted function**

Replace the `match device { ... }` block and the `gpu_priority` setup (lines 192-441) with:

```rust
let ProviderSelection {
    builder,
    device_name: actual_device_msg,
    status: ep_status,
} = select_execution_provider(builder, device, &available_providers)?;
```

The rest of `from_config` (lines 443-599) stays unchanged.

- [ ] **Step 5: Run validation**

Run: `cargo fmt --check && cargo clippy -- -D warnings && cargo test`
Expected: All pass. No behavioral change, pure extraction.

- [ ] **Step 6: Run sentrux check**

Run: `~/bin/sentrux check .`
Expected: All rules pass. `from_config` CC should drop significantly since the 14-arm match is now in the helper.

- [ ] **Step 7: Commit**

```bash
git add src/inference/classifier.rs
git commit -m "refactor: extract execution provider selection from from_config

Moves the ~220-line device match block into select_execution_provider(),
reducing from_config from 434 to ~210 lines and cutting its cyclomatic
complexity roughly in half."
```

---

### Task 4: Introduce `ProcessingConfig` struct to reduce `process_file` parameter count

The `process_file` function takes 15 parameters. Most of these are configuration values that travel together. Bundle them into a config struct.

**Files:**
- Create: `src/pipeline/config.rs`
- Modify: `src/pipeline/mod.rs`
- Modify: `src/pipeline/processor.rs:373-389`
- Modify: `src/lib.rs` (all call sites of `process_file`)

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `cargo test`
Expected: All pass.

- [ ] **Step 2: Create `ProcessingConfig` struct**

Create `src/pipeline/config.rs`:

```rust
//! Configuration types for the processing pipeline.

use crate::config::OutputFormat;
use std::path::{Path, PathBuf};

/// Configuration for processing a single audio file.
///
/// Bundles the parameters needed by `process_file` to reduce its argument count.
pub struct ProcessingConfig<'a> {
    /// Path to input audio file.
    pub input_path: &'a Path,
    /// Directory for output files.
    pub output_dir: &'a Path,
    /// Output formats to generate.
    pub formats: &'a [OutputFormat],
    /// Minimum confidence threshold (0.0-1.0).
    pub min_confidence: f32,
    /// Overlap between chunks in seconds.
    pub overlap: f32,
    /// Number of chunks to process in parallel.
    pub batch_size: usize,
    /// Additional columns to include in CSV output.
    pub csv_columns: &'a [String],
    /// Whether to show progress bars.
    pub progress_enabled: bool,
    /// Whether to include UTF-8 BOM in CSV output.
    pub csv_bom_enabled: bool,
    /// Model name for JSON output metadata.
    pub model_name: &'a str,
    /// Optional (lat, lon, week) for JSON output metadata.
    pub range_filter_params: Option<(f64, f64, u8)>,
    /// Optional (lat, lon, day_of_year) for BSG SDM.
    pub bsg_params: Option<(f64, f64, Option<u32>)>,
    /// Optional reporter for stdout mode.
    pub reporter: Option<&'a dyn crate::output::ProgressReporter>,
    /// Whether to write both files and stdout.
    pub dual_output_mode: bool,
}
```

- [ ] **Step 3: Add the module to `src/pipeline/mod.rs`**

Add `mod config;` and `pub use config::ProcessingConfig;` to `src/pipeline/mod.rs`.

- [ ] **Step 4: Update `process_file` signature**

Change `process_file` in `src/pipeline/processor.rs` from 15 parameters to:

```rust
pub fn process_file(
    config: &ProcessingConfig<'_>,
    classifier: &BirdClassifier,
) -> Result<ProcessResult> {
```

Inside the function body, replace all bare parameter references with `config.` prefix. For example:
- `input_path` becomes `config.input_path`
- `min_confidence` becomes `config.min_confidence`
- `formats` becomes `config.formats`
- etc.

The `classifier` stays as a separate parameter because it's a heavyweight object with its own lifecycle, not a configuration value.

- [ ] **Step 5: Update call sites in `src/lib.rs`**

Find all calls to `process_file` in `src/lib.rs`. There should be one in `process_all_files`. Update it to construct a `ProcessingConfig` and pass it:

```rust
use crate::pipeline::ProcessingConfig;

let config = ProcessingConfig {
    input_path: &file_path,
    output_dir: &output_dir,
    formats: &formats,
    min_confidence,
    overlap,
    batch_size,
    csv_columns: &csv_columns,
    progress_enabled,
    csv_bom_enabled,
    model_name: &model_name,
    range_filter_params,
    bsg_params,
    reporter: reporter_ref,
    dual_output_mode,
};

let result = process_file(&config, &classifier)?;
```

- [ ] **Step 6: Run validation**

Run: `cargo fmt --check && cargo clippy -- -D warnings && cargo test`
Expected: All pass. No behavioral change.

- [ ] **Step 7: Run sentrux check and gate**

Run: `~/bin/sentrux check . && ~/bin/sentrux gate .`
Expected: All rules pass. Quality signal should improve (fewer high-param functions, `process_file` CC may drop slightly due to cleaner structure).

- [ ] **Step 8: Commit**

```bash
git add src/pipeline/config.rs src/pipeline/mod.rs src/pipeline/processor.rs src/lib.rs
git commit -m "refactor: introduce ProcessingConfig to reduce process_file params

Bundles 13 of process_file's 15 parameters into a ProcessingConfig
struct. The classifier stays separate as a heavyweight runtime object.
Reduces parameter threading through the pipeline."
```

---

## Verification

After all four tasks, run a final quality check:

```bash
cargo fmt --check && cargo clippy -- -D warnings && cargo test
~/bin/sentrux gate .
```

The quality signal should improve from 7449, primarily from the equality metric (fewer monster functions, more balanced complexity distribution).
