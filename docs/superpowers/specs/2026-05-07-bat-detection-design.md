# Bat Detection Support - Design Spec

**Goal:** Add bat species detection to birda by chaining BirdNET v2.4 embeddings into BattyBirdNET regional classifiers.

**Architecture:** Two-stage inference pipeline. BirdNET v2.4 extracts 1024-dim embeddings from audio; a lightweight custom classifier maps embeddings to bat species. Audio is fed at 256kHz without resampling, exploiting the "slow-down trick" where BirdNET's 48kHz-trained spectrogram pipeline shifts ultrasonic bat calls into the audible frequency range.

**Repos affected:** birdnet-onnx-converter, rust-birdnet-onnx, birda.

---

## 1. ONNX Converter (`~/src/birdnet-onnx-converter`)

### 1.1 New script: `expose_embeddings.py`

Patches an existing BirdNET v2.4 ONNX model to expose the global average pooling layer (`model/GLOBAL_AVG_POOL/Mean_reduced_0`, shape `[batch, 1024]`) as a second output alongside the original classification output (`[batch, 6522]`).

**CLI:**
```
python expose_embeddings.py --input birdnet-v24.onnx --output birdnet-v24-embeddings.onnx
```

**Implementation:** Load ONNX graph, find the `Mean_reduced_0` value_info, append it to `graph.output`, save. Validate by loading with onnxruntime and checking 2 outputs. Error if the tensor is not found (model is not BirdNET v2.4 or has unexpected architecture).

### 1.2 Update `convert.py`

Add `--with-embeddings` flag. When converting from Keras, this flag applies the same graph surgery after the ONNX conversion step. When converting from TFLite with `--onnx-only`, the flag is applied to the resulting ONNX model.

### 1.3 No changes to `optimize.py`

The optimizer preserves all graph outputs. Models optimized after embedding exposure retain both outputs.

---

## 2. birdnet-onnx (`~/src/rust-birdnet-onnx`)

### 2.1 Model detection update

In `detection.rs`, update `detect_from_sample_count()`:
- Pattern `(144_000, 2)` detected as `BirdNetV24` with `embedding_dim: Some(1024)` (predictions at output 0, embeddings at output 1).
- Pattern `(144_000, 1)` remains unchanged (v2.4 without embeddings).

Update `detect_from_outputs()` similarly: 2-output models with dynamic input default to v2.4-with-embeddings when second output dim is 1024.

Update `build_config_with_override()` to allow `BirdNetV24` with 2 outputs.

### 2.2 Embedding extraction for v2.4

In `classifier.rs`, update `process_outputs()` and `process_batch_outputs()`:
- When `model_type == BirdNetV24` and `embedding_dim.is_some()`, extract predictions from output index 0 and embeddings from output index 1. Note: this is reversed from v3.0's order (v3.0 has embeddings at 0, predictions at 1) because the graph surgery appends the embedding output after the existing prediction output.
- When `embedding_dim.is_none()`, behavior unchanged (single output = predictions).

This means `PredictionResult.embeddings` will be `Some(Vec<f32>)` for v2.4 models with embeddings enabled, `None` otherwise.

### 2.3 Custom classifier (`CustomClassifier`)

New public type for running a secondary classification model on embedding vectors.

**Public API:**

```rust
pub struct CustomClassifier { ... }

pub struct CustomClassifierBuilder {
    model_path: Option<PathBuf>,
    labels_path: Option<PathBuf>,
    // ORT session options inherited from main classifier or set explicitly
}

impl CustomClassifierBuilder {
    pub fn new() -> Self;
    pub fn model_path(self, path: impl AsRef<Path>) -> Self;
    pub fn labels_path(self, path: impl AsRef<Path>) -> Self;
    pub fn build(self) -> Result<CustomClassifier>;
}

impl CustomClassifier {
    /// Classify a single embedding vector.
    pub fn predict(&self, embeddings: &[f32]) -> Result<Vec<Prediction>>;

    /// Classify a batch of embedding vectors.
    pub fn predict_batch(&self, embeddings: &[Vec<f32>]) -> Result<Vec<Vec<Prediction>>>;

    /// Return the labels loaded from the labels file.
    pub fn labels(&self) -> &[String];

    /// Return embedding dimension this classifier expects.
    pub fn input_dim(&self) -> usize;
}
```

**Input validation:** `predict()` checks that `embeddings.len() == input_dim()` (read from ONNX input shape). Mismatches return `Error::InvalidInput`.

**Output processing:** Raw logits from the classifier go through sigmoid activation (same as BirdNET v2.4's flat_sigmoid), then top-k selection. No softmax; BattyBirdNET uses sigmoid like BirdNET.

**Labels format:** One species per line, same format as BirdNET label files. Loaded from the path provided to the builder.

### 2.4 Exports

Add to `lib.rs` public exports: `CustomClassifier`, `CustomClassifierBuilder`.

---

## 3. birda (`~/src/birda`)

### 3.1 CLI: `--bat <region>` flag

New optional argument on `AnalyzeArgs`:

```rust
#[arg(long, value_name = "REGION")]
bat: Option<BatRegion>,
```

**`BatRegion` enum:**
```rust
pub enum BatRegion {
    Bavaria,
    BavariaHigh,
    Eu,
    Scotland,
    SouthWales,
    Sweden,
    Uk,
    Usa,
    UsaEast,
    UsaEastHigh,
    UsaWest,
}
```

Each variant maps to a model filename (e.g., `BatRegion::Uk` -> `BattyBirdNET-UK-256kHz_fp32.onnx` + `BattyBirdNET-UK-256kHz_Labels.txt`).

When `--bat` is provided:
- Overrides `--model-type` to `birdnet-v24` (embedding backbone).
- Sets bat-specific audio parameters (chunk size, overlap).
- Loads the regional bat classifier as a `CustomClassifier`.

### 3.2 Config: `ModelType` extension

No new `ModelType` variant needed. Bat mode reuses `BirdNetV24` as the backbone. The bat classifier is a separate model, not a birda model type.

Add to `config/types.rs`:

```rust
pub struct BatConfig {
    pub region: BatRegion,
    pub classifier_path: PathBuf,
    pub labels_path: PathBuf,
}
```

### 3.3 Constants

In `constants.rs`, add a `bat` module:

```rust
pub mod bat {
    /// Audio sample rate for bat recordings (256 kHz).
    pub const SAMPLE_RATE: u32 = 256_000;

    /// Segment duration in seconds (144,000 samples / 256,000 Hz).
    pub const SEGMENT_DURATION: f32 = 0.5625;

    /// Overlap between segments (25% of segment duration).
    pub const OVERLAP: f32 = 0.140625;

    /// Number of audio samples per segment.
    /// Equal to BirdNET v2.4's 144,000 samples - the "slow-down trick".
    pub const CHUNK_SAMPLES: usize = 144_000;
}
```

### 3.4 Audio pipeline adjustments

When bat mode is active:
- `StreamingDecoder` does NOT resample to 48kHz. The raw samples (at whatever the source rate is) are chunked into 144,000-sample segments.
- If the source audio is not 256kHz, birda logs a warning but proceeds (the user may have recordings at different rates).
- Overlap: 25% (36,000 samples).

### 3.5 Two-stage inference in `process_file()`

When bat config is present, `process_file()` follows this flow:

1. **Decode** audio into 144,000-sample chunks (no resampling).
2. **Batch** chunks into groups of `batch_size`.
3. **Stage 1:** Run BirdNET v2.4 batch inference. Extract `PredictionResult.embeddings` (1024-dim vectors).
4. **Stage 2:** Feed embeddings to `CustomClassifier.predict_batch()`. Get bat species predictions.
5. **Filter** by `min_confidence`.
6. **Write** results using bat classifier labels.

The two stages share the same execution provider (CPU/GPU).

### 3.6 Model file management

Bat classifier ONNX models and their label files are stored in the birda models directory alongside bird models:
- Linux: `~/.local/share/birda/models/bat/`
- macOS: `~/Library/Application Support/birda/models/bat/`
- Windows: `%APPDATA%\birda\models\bat\`

Model files follow BattyBirdNET naming: `BattyBirdNET-<Region>-256kHz_fp32.onnx` + `BattyBirdNET-<Region>-256kHz_Labels.txt`.

The BirdNET v2.4 backbone model with embeddings (`birdnet-v24-embeddings.onnx`) is stored in the standard models directory. If the user's existing v2.4 model lacks embeddings (1 output), birda prints an error explaining they need the embeddings variant.

### 3.7 Output

Bat detection results use the same output formats as bird detections (CSV, Raven, Audacity, JSON, Parquet, Kaleidoscope). The `combined_prefix` defaults to `"BattyBirdNET"` when in bat mode. Species names come from the bat classifier's labels file.

---

## 4. Testing

### 4.1 birdnet-onnx-converter
- Unit test: `expose_embeddings.py` produces a 2-output model from a 1-output v2.4 model.
- Integration test: modified model runs in onnxruntime with correct embedding shape.

### 4.2 birdnet-onnx
- Detection test: `(144_000, 2)` correctly detects v2.4 with embeddings.
- Embedding extraction test: v2.4-with-embeddings model returns `Some(vec)` with 1024 elements.
- CustomClassifier test: loads bat model, accepts 1024-dim input, returns predictions.
- Input validation test: wrong embedding dimension returns error.

### 4.3 birda
- CLI parsing test: `--bat uk` parses to `BatRegion::Uk`.
- Region-to-path mapping test: each region resolves to correct filenames.
- Constants test: `CHUNK_SAMPLES` equals BirdNET v2.4 sample count (144,000).
- Integration test: bat mode pipeline processes a test file end-to-end (requires model files).

---

## 5. Out of Scope

- Real-time/streaming bat detection (birda is currently batch-only).
- Bat model training or fine-tuning.
- Automatic model downloading (models are pre-installed).
- The `CUSTOM-BAT-256kHz` and `CUSTOM-BIRD-48kHz` models (these are custom classifier templates, not regional models).
- 144kHz alternative sample rate (BattyBirdNET supports this but we start with 256kHz only).
