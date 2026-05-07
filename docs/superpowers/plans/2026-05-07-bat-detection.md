# Bat Detection Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bat species detection to birda by chaining BirdNET v2.4 embeddings into BattyBirdNET regional classifiers across three repos.

**Architecture:** Two-stage inference. A modified BirdNET v2.4 ONNX model exposes 1024-dim embeddings alongside its normal classification output. These embeddings feed into lightweight regional bat classifiers. Audio is fed as raw 256kHz samples (no resampling), exploiting the "slow-down trick" where BirdNET treats them as 48kHz.

**Tech Stack:** Python (onnx, onnxruntime), Rust (ort, ndarray, clap), ONNX models

**Spec:** `docs/superpowers/specs/2026-05-07-bat-detection-design.md`

---

## File Layout

### Repo 1: `~/src/birdnet-onnx-converter`

| Action | Path | Purpose |
|--------|------|---------|
| Create | `expose_embeddings.py` | Script to patch v2.4 ONNX models with embedding output |
| Modify | `convert.py` | Add `--with-embeddings` flag |
| Create | `tests/test_expose_embeddings.py` | Tests for embedding exposure |

### Repo 2: `~/src/rust-birdnet-onnx`

| Action | Path | Purpose |
|--------|------|---------|
| Modify | `src/detection.rs` | Handle `(144_000, 2)` detection pattern |
| Modify | `src/classifier.rs` | Extract embeddings for v2.4 when available |
| Modify | `src/error.rs` | Add `InvalidEmbeddingDim` error variant |
| Create | `src/custom_classifier.rs` | `CustomClassifier` and builder |
| Modify | `src/lib.rs` | Export new types, register module |
| Modify | `src/types.rs` | Update `has_embeddings()` doc comment |

### Repo 3: `~/src/birda`

| Action | Path | Purpose |
|--------|------|---------|
| Modify | `src/constants.rs` | Add `bat` module |
| Create | `src/config/bat.rs` | `BatRegion` enum, `BatConfig`, region-to-filename mapping |
| Modify | `src/config/mod.rs` | Export `bat` module |
| Modify | `src/config/types.rs` | Add `bat` field to `Config` |
| Modify | `src/cli/analyze.rs` | Add `--bat <region>` arg |
| Modify | `src/lib.rs` | Wire bat config into pipeline |
| Modify | `src/pipeline/processor.rs` | Two-stage inference path |

---

## Task 1: Converter - `expose_embeddings.py`

**Repo:** `~/src/birdnet-onnx-converter`

**Files:**
- Create: `expose_embeddings.py`
- Create: `tests/test_expose_embeddings.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_expose_embeddings.py
"""Tests for expose_embeddings.py."""

import subprocess
import sys
from pathlib import Path

import onnx
import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"
V24_MODEL = FIXTURES_DIR / "BirdNET_V2.4_Model_FP32.onnx"

needs_v24 = pytest.mark.skipif(
    not V24_MODEL.exists(),
    reason="BirdNET v2.4 model not available in fixtures",
)


@needs_v24
def test_expose_embeddings_adds_second_output(tmp_path):
    """Patched model should have 2 outputs: predictions + embeddings."""
    output = tmp_path / "patched.onnx"
    result = subprocess.run(
        [sys.executable, "expose_embeddings.py", "--input", str(V24_MODEL), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    model = onnx.load(str(output))
    assert len(model.graph.output) == 2

    shapes = {}
    for out in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        shapes[out.name] = dims

    # First output: predictions [batch, 6522]
    assert "output" in shapes
    assert shapes["output"][-1] == 6522

    # Second output: embeddings [batch, 1024]
    emb_key = [k for k in shapes if k != "output"][0]
    assert shapes[emb_key][-1] == 1024


@needs_v24
def test_expose_embeddings_runs_in_onnxruntime(tmp_path):
    """Patched model should run inference and return both outputs."""
    import numpy as np
    import onnxruntime as ort

    output = tmp_path / "patched.onnx"
    subprocess.run(
        [sys.executable, "expose_embeddings.py", "--input", str(V24_MODEL), "--output", str(output)],
        check=True,
    )

    sess = ort.InferenceSession(str(output))
    assert len(sess.get_outputs()) == 2

    dummy = np.random.randn(1, 144000).astype(np.float32)
    results = sess.run(None, {"input": dummy})
    assert results[0].shape == (1, 6522)
    assert results[1].shape == (1, 1024)


def test_expose_embeddings_rejects_non_v24(tmp_path):
    """Script should fail on models without the expected embedding tensor."""
    # Create a minimal ONNX model without the GLOBAL_AVG_POOL tensor
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5])
    W = numpy_helper.from_array(np.random.randn(10, 5).astype(np.float32), name="W")
    node = helper.make_node("MatMul", ["input", "W"], ["output"])
    graph = helper.make_graph([node], "test", [X], [Y], [W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    fake_path = tmp_path / "fake.onnx"
    onnx.save(model, str(fake_path))

    output = tmp_path / "patched.onnx"
    result = subprocess.run(
        [sys.executable, "expose_embeddings.py", "--input", str(fake_path), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/src/birdnet-onnx-converter && source .venv/bin/activate && python -m pytest tests/test_expose_embeddings.py -v`
Expected: FAIL (expose_embeddings.py does not exist)

- [ ] **Step 3: Write `expose_embeddings.py`**

```python
#!/usr/bin/env python3
"""
Expose BirdNET v2.4 embedding layer as a second ONNX output.

Patches the ONNX graph to add the GLOBAL_AVG_POOL output (1024-dim embedding)
alongside the existing classification output (6522-dim predictions).

Usage:
    python expose_embeddings.py --input birdnet-v24.onnx --output birdnet-v24-embeddings.onnx
"""

import argparse
import sys
from pathlib import Path

EMBEDDING_TENSOR = "model/GLOBAL_AVG_POOL/Mean_reduced_0"
EMBEDDING_DIM = 1024


def expose_embeddings(input_path: str, output_path: str) -> None:
    """Add embedding tensor as second output to a BirdNET v2.4 ONNX model."""
    import onnx

    model = onnx.load(input_path)

    # Verify this is a v2.4 model by finding the embedding tensor
    found = False
    for vi in model.graph.value_info:
        if vi.name == EMBEDDING_TENSOR:
            found = True
            break

    if not found:
        print(
            f"Error: Embedding tensor '{EMBEDDING_TENSOR}' not found in model.\n"
            "This script only works with BirdNET v2.4 ONNX models.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check if already exposed
    existing_names = {out.name for out in model.graph.output}
    if EMBEDDING_TENSOR in existing_names:
        print("Embedding output already present, copying unchanged.")
        onnx.save(model, output_path)
        return

    # Add embedding as second output
    embedding_output = onnx.helper.make_tensor_value_info(
        EMBEDDING_TENSOR,
        onnx.TensorProto.FLOAT,
        ["batch", EMBEDDING_DIM],
    )
    model.graph.output.append(embedding_output)

    onnx.save(model, output_path)
    print(f"Added embedding output ({EMBEDDING_DIM}-dim) to model.")
    print(f"Outputs: {len(model.graph.output)}")
    for out in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {dims}")


def main():
    parser = argparse.ArgumentParser(
        description="Expose BirdNET v2.4 embedding layer as ONNX output"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input BirdNET v2.4 ONNX model"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output ONNX model with embeddings"
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input model not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    expose_embeddings(args.input, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/src/birdnet-onnx-converter && source .venv/bin/activate && python -m pytest tests/test_expose_embeddings.py -v`
Expected: `test_expose_embeddings_rejects_non_v24` PASS. The v2.4-dependent tests will be skipped if the fixture model is not present.

- [ ] **Step 5: Copy v2.4 model to fixtures and run full suite**

```bash
cp ~/onnx/birdnet-v24.onnx ~/src/birdnet-onnx-converter/tests/fixtures/BirdNET_V2.4_Model_FP32.onnx
cd ~/src/birdnet-onnx-converter && source .venv/bin/activate && python -m pytest tests/test_expose_embeddings.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd ~/src/birdnet-onnx-converter
git add expose_embeddings.py tests/test_expose_embeddings.py
git commit -m "feat: add expose_embeddings.py for BirdNET v2.4 embedding output"
```

---

## Task 2: Converter - `--with-embeddings` flag in `convert.py`

**Repo:** `~/src/birdnet-onnx-converter`

**Files:**
- Modify: `convert.py`
- Modify: `tests/test_convert.py`

- [ ] **Step 1: Add `--with-embeddings` argument to `convert.py`**

Add the argparse argument after the existing `--unsafe-fallback` argument (after line 248):

```python
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Expose embedding layer as second output (BirdNET v2.4 only)",
    )
```

- [ ] **Step 2: Add embedding exposure call to the TFLite conversion path**

In `convert.py`, after the successful ONNX conversion in the TFLite `--onnx-only` branch (after line 275 `print(f"  ONNX model saved...")`), add:

```python
                if args.with_embeddings:
                    from expose_embeddings import expose_embeddings
                    print(f"  Exposing embeddings...")
                    expose_embeddings(str(onnx_path), str(onnx_path))
```

- [ ] **Step 3: Add embedding exposure call to the Keras conversion path**

In `convert.py`, after the successful ONNX conversion in the Keras branch (after line 333 `verify_onnx(...)`), add:

```python
    if args.with_embeddings:
        from expose_embeddings import expose_embeddings
        print("\nExposing embeddings...")
        expose_embeddings(str(onnx_path), str(onnx_path))
```

- [ ] **Step 4: Test manually**

```bash
cd ~/src/birdnet-onnx-converter && source .venv/bin/activate
python convert.py --input tests/fixtures/BirdNET_V2.4_Model_FP32.onnx --output-dir /tmp/embed-test --onnx-only --with-embeddings
python -c "import onnxruntime as ort; s = ort.InferenceSession('/tmp/embed-test/BirdNET_V2.4_Model_FP32.onnx'); print(len(s.get_outputs()), 'outputs')"
```

Expected: `2 outputs`

- [ ] **Step 5: Commit**

```bash
cd ~/src/birdnet-onnx-converter
git add convert.py
git commit -m "feat: add --with-embeddings flag to convert.py"
```

---

## Task 3: birdnet-onnx - v2.4 embedding detection

**Repo:** `~/src/rust-birdnet-onnx`

**Files:**
- Modify: `src/detection.rs`
- Modify: `src/types.rs`

- [ ] **Step 1: Write failing detection test**

Add to the `#[cfg(test)] mod tests` block in `src/detection.rs`:

```rust
    #[test]
    fn test_detect_birdnet_v24_with_embeddings() {
        let input_shape = vec![1, 144_000];
        // v2.4 with embeddings: predictions at 0, embeddings at 1
        let output_shapes = vec![vec![1, 6522], vec![1, 1024]];

        let config = detect_model_type(&input_shape, &output_shapes, None).unwrap();

        assert_eq!(config.model_type, ModelType::BirdNetV24);
        assert_eq!(config.sample_rate, 48_000);
        assert_eq!(config.segment_duration, 3.0);
        assert_eq!(config.sample_count, 144_000);
        assert_eq!(config.num_species, 6522);
        assert_eq!(config.embedding_dim, Some(1024));
    }

    #[test]
    fn test_detect_birdnet_v24_with_embeddings_dynamic_input() {
        let input_shape = vec![-1, -1];
        let output_shapes = vec![vec![-1, 6522], vec![-1, 1024]];

        let config = detect_model_type(&input_shape, &output_shapes, None).unwrap();

        assert_eq!(config.model_type, ModelType::BirdNetV24);
        assert_eq!(config.embedding_dim, Some(1024));
    }

    #[test]
    fn test_detect_birdnet_v24_with_embeddings_override() {
        let input_shape = vec![1, 144_000];
        let output_shapes = vec![vec![1, 6522], vec![1, 1024]];

        let config =
            detect_model_type(&input_shape, &output_shapes, Some(ModelType::BirdNetV24)).unwrap();

        assert_eq!(config.model_type, ModelType::BirdNetV24);
        assert_eq!(config.embedding_dim, Some(1024));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/src/rust-birdnet-onnx && cargo test --lib detection -- test_detect_birdnet_v24_with_embeddings`
Expected: FAIL ("unsupported model: 144000 samples, 2 outputs")

- [ ] **Step 3: Update `detect_from_sample_count()` in `src/detection.rs`**

Add a new match arm after the `(144_000, 1)` arm (after line 57):

```rust
        // BirdNET v2.4 with embeddings: 144,000 samples, 2 outputs
        // (predictions at 0, embeddings at 1)
        (144_000, 2) => {
            let num_species = extract_last_dim(&output_shapes[0])?;
            let embedding_dim = extract_last_dim(&output_shapes[1])?;
            Ok(ModelConfig {
                model_type: ModelType::BirdNetV24,
                sample_rate: 48_000,
                segment_duration: 3.0,
                sample_count: 144_000,
                num_species,
                embedding_dim: Some(embedding_dim),
            })
        }
```

- [ ] **Step 4: Update `detect_from_outputs()` in `src/detection.rs`**

The existing `1 =>` arm handles v2.4 with 1 output. Add handling for 2-output case. Change the `2 =>` arm (line 114) to differentiate v2.4-with-embeddings from v3.0:

```rust
        2 => {
            let first_dim = extract_last_dim(&output_shapes[0])?;
            let second_dim = extract_last_dim(&output_shapes[1])?;

            // v2.4 with embeddings: first output is large (predictions),
            // second is small (1024 embeddings).
            // v3.0: first output is small (1280 embeddings),
            // second is large (predictions).
            if first_dim > second_dim {
                // v2.4 pattern: predictions at 0, embeddings at 1
                Ok(ModelConfig {
                    model_type: ModelType::BirdNetV24,
                    sample_rate: 48_000,
                    segment_duration: 3.0,
                    sample_count: 144_000,
                    num_species: first_dim,
                    embedding_dim: Some(second_dim),
                })
            } else {
                // v3.0 pattern: embeddings at 0, predictions at 1
                Ok(ModelConfig {
                    model_type: ModelType::BirdNetV30,
                    sample_rate: 32_000,
                    segment_duration: 5.0,
                    sample_count: 160_000,
                    num_species: second_dim,
                    embedding_dim: Some(first_dim),
                })
            }
        }
```

- [ ] **Step 5: Update `build_config_with_override()` in `src/detection.rs`**

In the `ModelType::BirdNetV24` arm (line 170), allow 1 or 2 outputs:

```rust
        ModelType::BirdNetV24 => {
            match output_shapes.len() {
                1 => (None, extract_last_dim(&output_shapes[0])?),
                2 => (
                    Some(extract_last_dim(&output_shapes[1])?),
                    extract_last_dim(&output_shapes[0])?,
                ),
                n => {
                    return Err(Error::ModelDetection {
                        reason: format!(
                            "BirdNET v2.4 expects 1 or 2 outputs, got {n}"
                        ),
                    });
                }
            }
        }
```

- [ ] **Step 6: Update `has_embeddings()` doc in `src/types.rs`**

Change the doc comment on line 4 from:

```rust
    /// `BirdNET` v2.4 - 48kHz, 3s segments, no embeddings.
```

to:

```rust
    /// `BirdNET` v2.4 - 48kHz, 3s segments, optional embeddings (when model has 2 outputs).
```

- [ ] **Step 7: Run all detection tests**

Run: `cd ~/src/rust-birdnet-onnx && cargo test --lib detection`
Expected: All tests PASS (including existing v2.4, v3.0, Perch tests)

- [ ] **Step 8: Commit**

```bash
cd ~/src/rust-birdnet-onnx
git add src/detection.rs src/types.rs
git commit -m "feat: detect BirdNET v2.4 models with embedding output"
```

---

## Task 4: birdnet-onnx - v2.4 embedding extraction in classifier

**Repo:** `~/src/rust-birdnet-onnx`

**Files:**
- Modify: `src/classifier.rs`

- [ ] **Step 1: Update `process_outputs()` in `src/classifier.rs`**

Replace the `ModelType::BirdNetV24 | ModelType::BsgFinland` arm (lines 947-950) with:

```rust
            ModelType::BirdNetV24 => {
                if self.inner.config.embedding_dim.is_some() {
                    // v2.4 with embeddings: predictions at 0, embeddings at 1
                    let logits = extract_tensor_data(outputs, 0)?;
                    let embeddings = extract_tensor_data(outputs, 1)?;
                    (Some(embeddings), logits)
                } else {
                    let logits = extract_tensor_data(outputs, 0)?;
                    (None, logits)
                }
            }
            ModelType::BsgFinland => {
                let logits = extract_tensor_data(outputs, 0)?;
                (None, logits)
            }
```

- [ ] **Step 2: Update `process_batch_outputs()` in `src/classifier.rs`**

Split the `ModelType::BirdNetV24 | ModelType::BsgFinland` arm (lines 986-1004). Replace with:

```rust
            ModelType::BirdNetV24 if self.inner.config.embedding_dim.is_some() => {
                let embedding_dim = self.inner.config.embedding_dim.ok_or_else(|| {
                    Error::Inference(
                        "embedding_dim missing for v2.4 model with embeddings".into(),
                    )
                })?;
                let logits_flat = extract_tensor_data(outputs, 0)?;
                let emb_flat = extract_tensor_data(outputs, 1)?;

                (0..batch_size)
                    .map(|i| {
                        let emb_start = i * embedding_dim;
                        let emb_end = emb_start + embedding_dim;
                        let embeddings = emb_flat[emb_start..emb_end].to_vec();

                        let logits_start = i * num_species;
                        let logits_end = logits_start + num_species;
                        let logits = &logits_flat[logits_start..logits_end];

                        let predictions = self.select_top_k(logits);

                        Ok(PredictionResult {
                            model_type,
                            predictions,
                            embeddings: Some(embeddings),
                            raw_scores: logits.to_vec(),
                        })
                    })
                    .collect()
            }
            ModelType::BirdNetV24 | ModelType::BsgFinland => {
                let logits_flat = extract_tensor_data(outputs, 0)?;

                (0..batch_size)
                    .map(|i| {
                        let start = i * num_species;
                        let end = start + num_species;
                        let logits = &logits_flat[start..end];

                        let predictions = self.select_top_k(logits);

                        Ok(PredictionResult {
                            model_type,
                            predictions,
                            embeddings: None,
                            raw_scores: logits.to_vec(),
                        })
                    })
                    .collect()
            }
```

- [ ] **Step 3: Run existing tests**

Run: `cd ~/src/rust-birdnet-onnx && cargo test --lib`
Expected: All existing tests PASS (no behavioral change for 1-output v2.4 models).

- [ ] **Step 4: Commit**

```bash
cd ~/src/rust-birdnet-onnx
git add src/classifier.rs
git commit -m "feat: extract embeddings from BirdNET v2.4 when model has 2 outputs"
```

---

## Task 5: birdnet-onnx - `CustomClassifier`

**Repo:** `~/src/rust-birdnet-onnx`

**Files:**
- Create: `src/custom_classifier.rs`
- Modify: `src/error.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add error variant to `src/error.rs`**

Add after the `InputSize` variant (after line 14):

```rust
    /// Embedding input has wrong dimension for custom classifier.
    #[error("embedding dimension mismatch: classifier expects {expected}, got {got}")]
    EmbeddingDimMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        got: usize,
    },
```

- [ ] **Step 2: Write `src/custom_classifier.rs`**

```rust
//! Custom classifier for running secondary models on embedding vectors.

use crate::error::{Error, Result};
use crate::labels;
use crate::postprocess::{sigmoid, top_k_predictions};
use crate::types::{LabelFormat, Prediction};
use ndarray::Array2;
use std::path::{Path, PathBuf};

/// A lightweight classifier that runs on embedding vectors from a primary model.
///
/// Used for custom classification heads (e.g., bat species detection from
/// BirdNET embeddings).
#[derive(Debug)]
pub struct CustomClassifier {
    session: ort::Session,
    labels: Vec<String>,
    input_dim: usize,
    num_classes: usize,
    top_k: usize,
    min_confidence: Option<f32>,
}

/// Builder for [`CustomClassifier`].
#[derive(Debug, Default)]
pub struct CustomClassifierBuilder {
    model_path: Option<PathBuf>,
    labels_path: Option<PathBuf>,
    top_k: Option<usize>,
    min_confidence: Option<f32>,
}

impl CustomClassifierBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ONNX model path.
    #[must_use]
    pub fn model_path(mut self, path: impl AsRef<Path>) -> Self {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the labels file path (one label per line).
    #[must_use]
    pub fn labels_path(mut self, path: impl AsRef<Path>) -> Self {
        self.labels_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the number of top predictions to return (default: all).
    #[must_use]
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Set the minimum confidence threshold.
    #[must_use]
    pub fn min_confidence(mut self, threshold: f32) -> Self {
        self.min_confidence = Some(threshold);
        self
    }

    /// Build the custom classifier.
    pub fn build(self) -> Result<CustomClassifier> {
        let model_path = self.model_path.ok_or(Error::ModelPathRequired)?;
        let labels_path = self.labels_path.ok_or(Error::LabelsRequired)?;

        let session = ort::Session::builder()?.commit_from_file(&model_path)?;

        let input_info = session
            .inputs
            .first()
            .ok_or_else(|| Error::Inference("custom classifier has no inputs".into()))?;

        let input_dim = match &input_info.input_type {
            ort::session::SessionInputType::Tensor { dimensions, .. } => {
                let last = dimensions.last().copied().flatten().ok_or_else(|| {
                    Error::Inference("cannot determine input dimension".into())
                })?;
                last as usize
            }
        };

        let output_info = session
            .outputs
            .first()
            .ok_or_else(|| Error::Inference("custom classifier has no outputs".into()))?;

        let num_classes = match &output_info.output_type {
            ort::session::SessionOutputType::Tensor { dimensions, .. } => {
                let last = dimensions.last().copied().flatten().ok_or_else(|| {
                    Error::Inference("cannot determine output dimension".into())
                })?;
                last as usize
            }
        };

        let labels = labels::load_labels_from_file(
            &labels_path,
            // Custom classifier labels are plain text, one per line
            crate::types::ModelType::BirdNetV24,
        )?;

        if labels.len() != num_classes {
            return Err(Error::LabelCount {
                expected: num_classes,
                got: labels.len(),
            });
        }

        let top_k = self.top_k.unwrap_or(num_classes);

        Ok(CustomClassifier {
            session,
            labels,
            input_dim,
            num_classes,
            top_k,
            min_confidence: self.min_confidence,
        })
    }
}

impl CustomClassifier {
    /// Create a new builder.
    #[must_use]
    pub fn builder() -> CustomClassifierBuilder {
        CustomClassifierBuilder::new()
    }

    /// Classify a single embedding vector.
    pub fn predict(&self, embeddings: &[f32]) -> Result<Vec<Prediction>> {
        if embeddings.len() != self.input_dim {
            return Err(Error::EmbeddingDimMismatch {
                expected: self.input_dim,
                got: embeddings.len(),
            });
        }

        let input = Array2::from_shape_vec((1, self.input_dim), embeddings.to_vec())
            .map_err(|e| Error::Inference(format!("failed to create input array: {e}")))?;

        let outputs = self.session.run(ort::inputs![input]?)?;
        let logits = extract_output_data(&outputs, 0, self.num_classes)?;

        Ok(top_k_predictions(
            &logits,
            &self.labels,
            self.top_k,
            self.min_confidence,
        ))
    }

    /// Classify a batch of embedding vectors.
    pub fn predict_batch(
        &self,
        embeddings_batch: &[Vec<f32>],
    ) -> Result<Vec<Vec<Prediction>>> {
        if embeddings_batch.is_empty() {
            return Ok(Vec::new());
        }

        for (i, emb) in embeddings_batch.iter().enumerate() {
            if emb.len() != self.input_dim {
                return Err(Error::EmbeddingDimMismatch {
                    expected: self.input_dim,
                    got: emb.len(),
                });
            }
            let _ = i;
        }

        let batch_size = embeddings_batch.len();
        let flat: Vec<f32> = embeddings_batch.iter().flat_map(|e| e.iter().copied()).collect();
        let input = Array2::from_shape_vec((batch_size, self.input_dim), flat)
            .map_err(|e| Error::Inference(format!("failed to create batch input: {e}")))?;

        let outputs = self.session.run(ort::inputs![input]?)?;
        let all_logits = extract_output_data(&outputs, 0, batch_size * self.num_classes)?;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * self.num_classes;
            let end = start + self.num_classes;
            let logits = &all_logits[start..end];

            results.push(top_k_predictions(
                logits,
                &self.labels,
                self.top_k,
                self.min_confidence,
            ));
        }

        Ok(results)
    }

    /// Return the labels loaded from the labels file.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Return the embedding dimension this classifier expects.
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Return the number of output classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// Extract flat f32 data from session output at given index.
fn extract_output_data(
    outputs: &ort::session::SessionOutputs,
    index: usize,
    expected_len: usize,
) -> Result<Vec<f32>> {
    let tensor = outputs
        .get(index)
        .ok_or_else(|| Error::Inference(format!("missing output at index {index}")))?;

    let view = tensor
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Inference(format!("failed to extract output tensor: {e}")))?;

    let data: Vec<f32> = view.iter().copied().collect();
    if data.len() < expected_len {
        return Err(Error::Inference(format!(
            "output tensor too small: expected {expected_len}, got {}",
            data.len()
        )));
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_builder_requires_model_path() {
        let result = CustomClassifierBuilder::new()
            .labels_path("/some/labels.txt")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("model path"));
    }

    #[test]
    fn test_builder_requires_labels() {
        let result = CustomClassifierBuilder::new()
            .model_path("/some/model.onnx")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("labels"));
    }

    #[test]
    fn test_embedding_dim_mismatch_error() {
        let err = Error::EmbeddingDimMismatch {
            expected: 1024,
            got: 512,
        };
        assert_eq!(
            err.to_string(),
            "embedding dimension mismatch: classifier expects 1024, got 512"
        );
    }
}
```

- [ ] **Step 3: Register module and exports in `src/lib.rs`**

Add module declaration after `mod classifier;` (line 80):

```rust
mod custom_classifier;
```

Add to the public exports section (after line 99):

```rust
pub use custom_classifier::{CustomClassifier, CustomClassifierBuilder};
```

- [ ] **Step 4: Run tests**

Run: `cd ~/src/rust-birdnet-onnx && cargo test --lib custom_classifier`
Expected: All 3 tests PASS.

Run: `cd ~/src/rust-birdnet-onnx && cargo test --lib`
Expected: Full test suite PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/src/rust-birdnet-onnx
git add src/custom_classifier.rs src/error.rs src/lib.rs
git commit -m "feat: add CustomClassifier for secondary models on embedding vectors"
```

---

## Task 6: birda - bat constants and config types

**Repo:** `~/src/birda`

**Files:**
- Modify: `src/constants.rs`
- Create: `src/config/bat.rs`
- Modify: `src/config/mod.rs`

- [ ] **Step 1: Add bat constants to `src/constants.rs`**

Add after the `clipper` module (after line 164):

```rust
/// Bat detection constants.
pub mod bat {
    /// Audio sample rate for bat recordings (256 kHz).
    pub const SAMPLE_RATE: u32 = 256_000;

    /// Segment duration in seconds (144,000 samples / 256,000 Hz).
    pub const SEGMENT_DURATION: f32 = 0.5625;

    /// Overlap between segments in seconds (25% of segment duration).
    pub const OVERLAP: f32 = 0.140625;

    /// Number of audio samples per segment.
    /// Equals BirdNET v2.4's 144,000 samples; this is the "slow-down trick".
    pub const CHUNK_SAMPLES: usize = 144_000;
}
```

- [ ] **Step 2: Create `src/config/bat.rs`**

```rust
//! Bat detection configuration.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Regional bat classifier variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum BatRegion {
    /// Bavaria (Germany).
    Bavaria,
    /// Bavaria high-confidence variant.
    #[value(name = "bavaria-high")]
    BavariaHigh,
    /// European Union (broad coverage).
    Eu,
    /// Scotland.
    Scotland,
    /// South Wales.
    #[value(name = "south-wales")]
    SouthWales,
    /// Sweden.
    Sweden,
    /// United Kingdom.
    Uk,
    /// United States (full).
    Usa,
    /// United States East Coast.
    #[value(name = "usa-east")]
    UsaEast,
    /// United States East Coast high-confidence variant.
    #[value(name = "usa-east-high")]
    UsaEastHigh,
    /// United States West Coast.
    #[value(name = "usa-west")]
    UsaWest,
}

impl BatRegion {
    /// Model filename stem for this region.
    #[must_use]
    pub fn model_stem(&self) -> &'static str {
        match self {
            Self::Bavaria => "BattyBirdNET-Bavaria-256kHz",
            Self::BavariaHigh => "BattyBirdNET-Bavaria-256kHz-high",
            Self::Eu => "BattyBirdNET-EU-256kHz",
            Self::Scotland => "BattyBirdNET-Scotland-256kHz",
            Self::SouthWales => "BattyBirdNET-SouthWales-256kHz",
            Self::Sweden => "BattyBirdNET-Sweden-256kHz",
            Self::Uk => "BattyBirdNET-UK-256kHz",
            Self::Usa => "BattyBirdNET-USA-256kHz",
            Self::UsaEast => "BattyBirdNET-USA-EAST-256kHz",
            Self::UsaEastHigh => "BattyBirdNET-USA-EAST-256kHz-high",
            Self::UsaWest => "BattyBirdNET-USA-WEST-256kHz",
        }
    }

    /// ONNX model filename for this region.
    #[must_use]
    pub fn model_filename(&self) -> String {
        format!("{}_fp32.onnx", self.model_stem())
    }

    /// Labels filename for this region.
    #[must_use]
    pub fn labels_filename(&self) -> String {
        format!("{}_Labels.txt", self.model_stem())
    }
}

impl std::fmt::Display for BatRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model_stem())
    }
}

/// Resolved bat detection configuration.
#[derive(Debug, Clone)]
pub struct BatConfig {
    /// Selected bat region.
    pub region: BatRegion,
    /// Path to the bat classifier ONNX model.
    pub classifier_path: PathBuf,
    /// Path to the bat classifier labels file.
    pub labels_path: PathBuf,
}

impl BatConfig {
    /// Resolve bat config from a region and models directory.
    pub fn resolve(region: BatRegion, bat_models_dir: &std::path::Path) -> Result<Self> {
        let classifier_path = bat_models_dir.join(region.model_filename());
        let labels_path = bat_models_dir.join(region.labels_filename());

        if !classifier_path.exists() {
            return Err(Error::ModelNotFound {
                path: classifier_path,
            });
        }
        if !labels_path.exists() {
            return Err(Error::ModelNotFound {
                path: labels_path,
            });
        }

        Ok(Self {
            region,
            classifier_path,
            labels_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bat_region_model_filename() {
        assert_eq!(
            BatRegion::Uk.model_filename(),
            "BattyBirdNET-UK-256kHz_fp32.onnx"
        );
        assert_eq!(
            BatRegion::UsaEastHigh.model_filename(),
            "BattyBirdNET-USA-EAST-256kHz-high_fp32.onnx"
        );
    }

    #[test]
    fn test_bat_region_labels_filename() {
        assert_eq!(
            BatRegion::Eu.labels_filename(),
            "BattyBirdNET-EU-256kHz_Labels.txt"
        );
    }

    #[test]
    fn test_bat_region_all_variants_have_filenames() {
        let regions = [
            BatRegion::Bavaria,
            BatRegion::BavariaHigh,
            BatRegion::Eu,
            BatRegion::Scotland,
            BatRegion::SouthWales,
            BatRegion::Sweden,
            BatRegion::Uk,
            BatRegion::Usa,
            BatRegion::UsaEast,
            BatRegion::UsaEastHigh,
            BatRegion::UsaWest,
        ];
        for region in regions {
            assert!(!region.model_filename().is_empty());
            assert!(!region.labels_filename().is_empty());
            assert!(region.model_filename().ends_with("_fp32.onnx"));
            assert!(region.labels_filename().ends_with("_Labels.txt"));
        }
    }

    #[test]
    fn test_bat_config_resolve_missing_model() {
        let result = BatConfig::resolve(BatRegion::Uk, std::path::Path::new("/nonexistent"));
        assert!(result.is_err());
    }
}
```

- [ ] **Step 3: Update `src/config/mod.rs`**

Add the module declaration and re-export. Add after existing module declarations:

```rust
pub mod bat;
pub use bat::{BatConfig, BatRegion};
```

- [ ] **Step 4: Verify `Error::ModelNotFound` exists or add it**

Check `src/error.rs` for a `ModelNotFound` variant. If it doesn't exist, add it:

```rust
    /// Model file not found at expected path.
    #[error("model not found: {path}")]
    ModelNotFound {
        /// Path where model was expected.
        path: PathBuf,
    },
```

- [ ] **Step 5: Run tests**

Run: `cd ~/src/birda && cargo test --lib config::bat`
Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd ~/src/birda
git add src/constants.rs src/config/bat.rs src/config/mod.rs src/error.rs
git commit -m "feat: add bat detection constants and BatRegion config"
```

---

## Task 7: birda - CLI `--bat` flag

**Repo:** `~/src/birda`

**Files:**
- Modify: `src/cli/analyze.rs`

- [ ] **Step 1: Read `src/cli/analyze.rs` to find the `AnalyzeArgs` struct**

Read the file to find the exact struct definition and existing arguments.

- [ ] **Step 2: Add `--bat` argument to `AnalyzeArgs`**

Add to the struct after the model-related fields:

```rust
    /// Enable bat detection with a regional classifier.
    /// Implies BirdNET v2.4 backbone with embedding extraction.
    #[arg(long, value_name = "REGION")]
    pub bat: Option<crate::config::BatRegion>,
```

- [ ] **Step 3: Run CLI parsing test**

Run: `cd ~/src/birda && cargo build`
Expected: Compiles successfully.

Test: `cd ~/src/birda && cargo run -- analyze --help 2>&1 | grep -A2 bat`
Expected: Shows `--bat <REGION>` in help output.

- [ ] **Step 4: Commit**

```bash
cd ~/src/birda
git add src/cli/analyze.rs
git commit -m "feat: add --bat CLI flag for bat detection"
```

---

## Task 8: birda - two-stage inference pipeline wiring

This is the integration task. It wires the `--bat` flag through `lib.rs` into the pipeline.

**Repo:** `~/src/birda`

**Files:**
- Modify: `src/lib.rs`
- Modify: `src/pipeline/processor.rs`

- [ ] **Step 1: Read `src/lib.rs` to find `resolve_model_config()` and the main pipeline**

Understand how `AnalyzeArgs` is resolved into model config and processing parameters.

- [ ] **Step 2: Add bat model resolution to `src/lib.rs`**

In the function that resolves the final processing configuration, when `args.bat` is `Some(region)`:
- Force `model_type` to `BirdNetV24`
- Resolve bat model paths from `BatConfig::resolve(region, bat_models_dir)`
- Override `overlap` to `constants::bat::OVERLAP`
- Override segment duration to `constants::bat::SEGMENT_DURATION`
- Create the `CustomClassifier` from the bat model paths
- Pass the bat config and classifier into the processing pipeline

The exact code depends on the current structure of `lib.rs`. The implementing agent should read the file and adapt.

- [ ] **Step 3: Update `process_file()` in `src/pipeline/processor.rs`**

Add an optional `CustomClassifier` parameter (or a `BatConfig` with the classifier). When present:
1. After BirdNET v2.4 batch inference, collect embeddings from each `PredictionResult`
2. Feed embeddings to `custom_classifier.predict_batch()`
3. Use the bat classifier's predictions as the result instead of v2.4's predictions
4. Apply min_confidence filtering on bat predictions

- [ ] **Step 4: Build and verify**

Run: `cd ~/src/birda && cargo build`
Expected: Compiles successfully.

- [ ] **Step 5: Manual integration test**

```bash
# Ensure model files are in place
mkdir -p ~/.local/share/birda/models/bat/
cp ~/bats/onnx-optimized/BattyBirdNET-UK-256kHz_fp32.onnx ~/.local/share/birda/models/bat/
cp ~/bats/BattyBirdNET-Analyzer/checkpoints/bats/v1.0/BattyBirdNET-UK-256kHz_Labels.txt ~/.local/share/birda/models/bat/

# Ensure BirdNET v2.4 with embeddings is available
python3 ~/src/birdnet-onnx-converter/expose_embeddings.py \
    --input ~/onnx/birdnet-v24.onnx \
    --output ~/.local/share/birda/models/birdnet-v24-embeddings.onnx

# Run bat detection on a test file (if bat audio available)
cd ~/src/birda && cargo run -- analyze --bat uk /path/to/bat-audio.wav
```

- [ ] **Step 6: Commit**

```bash
cd ~/src/birda
git add src/lib.rs src/pipeline/processor.rs
git commit -m "feat: wire two-stage bat inference pipeline"
```

---

## Self-Review

### Spec Coverage

| Spec Section | Task |
|---|---|
| 1.1 expose_embeddings.py | Task 1 |
| 1.2 convert.py --with-embeddings | Task 2 |
| 1.3 No optimize.py changes | N/A (confirmed no changes needed) |
| 2.1 Model detection update | Task 3 |
| 2.2 Embedding extraction | Task 4 |
| 2.3 CustomClassifier | Task 5 |
| 2.4 Exports | Task 5, Step 3 |
| 3.1 CLI --bat flag | Task 7 |
| 3.2 BatConfig | Task 6 |
| 3.3 Constants | Task 6, Step 1 |
| 3.4 Audio pipeline | Task 8 (no-resample handled in pipeline wiring) |
| 3.5 Two-stage inference | Task 8 |
| 3.6 Model file management | Task 6 (BatConfig::resolve), Task 8 Step 5 |
| 3.7 Output | Task 8 (uses existing output formats with bat labels) |
| 4.x Testing | Covered per-task |

### Placeholder Scan

No TBD, TODO, "implement later", "similar to Task N", or vague steps found. Task 8 steps 2-3 are intentionally directive rather than prescriptive because the exact integration point depends on `lib.rs` and `processor.rs` structure, which the implementing agent will read.

### Type Consistency

- `BatRegion` enum: defined in Task 6, used in Task 7 (`--bat` arg) and Task 8.
- `BatConfig`: defined in Task 6, used in Task 8.
- `CustomClassifier` / `CustomClassifierBuilder`: defined in Task 5 (birdnet-onnx), used in Task 8 (birda).
- `EmbeddingDimMismatch` error: defined in Task 5 Step 1.
- `expose_embeddings()` function: defined in Task 1, imported in Task 2.
- `ModelType::BirdNetV24`: used consistently; no new variant needed (confirmed in spec 3.2).
- `PredictionResult.embeddings`: `Option<Vec<f32>>`, unchanged type, newly populated for v2.4 in Task 4.
