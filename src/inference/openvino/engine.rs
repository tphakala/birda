//! The native `OpenVINO` inference engine.
//!
//! One engine runs one model on one Intel GPU. Model semantics (family
//! detection, labels, activation, top-k) come from `birdnet_onnx`; this file
//! runs the network and hands the flat output buffers to
//! [`PredictionDecoder`].
//!
//! Shapes are static: one compiled model for a single segment and one for the
//! run's batch size. A compiled shape is checked against ONNX Runtime CPU on a
//! deterministic probe before it carries audio.

use super::cache;
use super::device::{OvDevice, ResolveError, resolve};
use super::parity::{max_abs_diff, probe_segment};
use super::runtime::{Diagnosis, Os, RenderNode, Runtime, gather_render_nodes, render_diagnosis};
use crate::config::{ModelConfig as BirdaModelConfig, OpenVinoSettings, openvino_cache_dir};
use crate::constants::openvino as ov;
use crate::error::{Error, Result};
use birdnet_onnx::{
    ClassifierBuilder, InferenceOptions, ModelConfig, PredictionDecoder, PredictionResult,
    detect_model_type, load_labels_from_file,
};
use openvino::{
    CompiledModel, DeviceType, ElementType, InferRequest, Model, PartialShape, PropertyKey,
    RwPropertyKey, Shape, Tensor,
};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;
use tracing::{debug, info, warn};

/// A contiguous run of input segments sent to one compiled shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    /// Index of the first segment.
    pub start: usize,
    /// Segments in the chunk that carry real audio.
    pub valid: usize,
    /// Batch size of the compiled shape that runs it: `1` or the large shape.
    pub shape: usize,
}

/// Split `n` segments over the shapes `1` and `large`.
///
/// Full chunks of `large` first; a remainder of at most half of `large` runs as
/// single segments, since padding it to the large shape would compute more
/// rows than it carries; any other remainder is padded to one `large` chunk.
/// The pipeline pads a partial batch to the batch size itself
/// (`src/pipeline/processor.rs:244`), so a short remainder reaches this function
/// only for a file whose length is known up front and that is shorter than a batch.
/// The chunks cover `0..n` in order, so results map back to inputs by position.
#[must_use]
pub fn plan_chunks(n: usize, large: usize) -> Vec<Chunk> {
    let large = large.max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < n {
        let remaining = n - start;
        if remaining >= large {
            chunks.push(Chunk {
                start,
                valid: large,
                shape: large,
            });
            start += large;
        } else if 2 * remaining <= large {
            chunks.extend((start..n).map(|index| Chunk {
                start: index,
                valid: 1,
                shape: 1,
            }));
            start = n;
        } else {
            chunks.push(Chunk {
                start,
                valid: remaining,
                shape: large,
            });
            start = n;
        }
    }
    chunks
}

/// Which device ran and how, for the log line and the JSON envelope.
#[derive(Debug, Clone)]
pub struct DeviceReport {
    /// The device spec as configured.
    pub requested: String,
    /// The device the spec resolved to.
    pub device: OvDevice,
    /// Runtime build number.
    pub runtime_version: String,
    /// Precision the model is compiled at.
    pub precision: &'static str,
}

/// One compiled batch shape with its reusable request and input tensor.
struct CompiledShape {
    batch: usize,
    // Declaration order is drop order: the request goes before the compiled
    // model it was created from.
    request: InferRequest,
    _compiled: CompiledModel,
    input: Tensor,
}

/// Raw outputs for one row of the probe, from either engine.
struct RowOutput {
    scores: Vec<f32>,
    embeddings: Option<Vec<f32>>,
}

struct State {
    // `shapes` first: compiled models are dropped before the model and the
    // core that created them (fields drop in declaration order).
    shapes: BTreeMap<usize, CompiledShape>,
    model: Model,
    core: openvino::Core,
    /// Batch size of the large shape. `1` until `prepare` fixes it.
    large: usize,
    /// ONNX Runtime outputs for probe rows `0..reference.len()`.
    reference: Vec<RowOutput>,
}

/// Runs one model on one `OpenVINO` GPU.
pub struct OpenVinoEngine {
    decoder: PredictionDecoder,
    report: DeviceReport,
    device: DeviceType<'static>,
    /// Static input shape with the batch dimension left at `-1`.
    input_dims: Vec<i64>,
    model_path: String,
    labels_path: String,
    state: Mutex<State>,
}

fn model_load<E: std::fmt::Display>(context: &str) -> impl FnOnce(E) -> Error + '_ {
    move |error| Error::OpenVinoModelLoad {
        reason: format!("{context}: {error}"),
    }
}

/// Shape of a port with dynamic dimensions as `-1`.
fn port_dims(shape: &PartialShape, what: &str) -> Result<Vec<i64>> {
    if shape.get_rank().is_dynamic() {
        return Err(Error::OpenVinoModelLoad {
            reason: format!("the {what} has a dynamic rank, which birda cannot run"),
        });
    }
    Ok(shape
        .get_dimensions()
        .iter()
        .map(|dim| if dim.is_dynamic() { -1 } else { dim.get_min() })
        .collect())
}

/// Map a resolve failure to the user-facing error, with the render-node
/// diagnosis when no Intel GPU is listed at all.
pub(super) fn resolve_error(error: ResolveError, runtime: &Runtime) -> Error {
    map_resolve_error(
        error,
        Os::current(),
        runtime.devices.iter().any(OvDevice::is_selectable),
        gather_render_nodes,
    )
}

/// The pure part of [`resolve_error`]: the host's OS, whether an Intel GPU is
/// listed and the node gatherer are passed in.
fn map_resolve_error(
    error: ResolveError,
    os: Option<Os>,
    any_selectable: bool,
    gather: impl FnOnce() -> Vec<RenderNode>,
) -> Error {
    match error {
        ResolveError::Unsupported { reason, .. } => Error::OpenVinoUnsupported { reason },
        ResolveError::NotFound { requested, found } => {
            let diagnosis = render_diagnosis(os, any_selectable, gather);
            if let Some(Diagnosis::AccessDenied { node }) = diagnosis {
                return Error::OpenVinoGpuAccessDenied { node };
            }
            Error::OpenVinoDeviceUnavailable {
                requested,
                found: if found.is_empty() {
                    "nothing".to_string()
                } else {
                    found.join("; ")
                },
                hint: diagnosis.map(|d| d.hint()).unwrap_or_default(),
            }
        }
    }
}

impl OpenVinoEngine {
    /// Load the runtime, resolve the device and read the model.
    ///
    /// Nothing is compiled yet: [`Self::prepare`] compiles and verifies the
    /// shapes the run needs.
    ///
    /// # Errors
    ///
    /// The `OpenVino*` errors for a missing runtime or device, and
    /// `OpenVinoModelLoad` when the model cannot be read.
    pub fn new(
        model_config: &BirdaModelConfig,
        settings: &OpenVinoSettings,
        top_k: usize,
        min_confidence: f32,
    ) -> Result<Self> {
        let mut runtime = Runtime::load(settings.runtime_path.as_deref())?;
        let device = resolve(&settings.device, &runtime.devices)
            .map_err(|error| resolve_error(error, &runtime))?
            .clone();
        let target = DeviceType::from(device.name.as_str()).to_owned();

        // The GPU plugin defaults to f16, which changes detections for BirdNET
        // v2.4 and v3.0: set f32 before any compile. Read back after compiling
        // (see `compile_shape`), because the property is only a hint.
        runtime
            .core
            .set_property(
                &target,
                &RwPropertyKey::HintInferencePrecision,
                ov::PRECISION,
            )
            .map_err(model_load("setting the inference precision"))?;
        Self::configure_cache(&mut runtime, &target);

        let model_path = model_config.path.to_string_lossy().to_string();
        let model = runtime
            .core
            .read_model_from_file(&model_path, "")
            .map_err(model_load("reading the model"))?;
        let (config, input_dims) = Self::describe_model(&model)?;
        let labels =
            load_labels_from_file(&model_config.labels, config.model_type).map_err(|e| {
                Error::OpenVinoModelLoad {
                    reason: format!("reading the labels: {e}"),
                }
            })?;
        let decoder =
            PredictionDecoder::new(config, labels, top_k, Some(min_confidence)).map_err(|e| {
                Error::ClassifierBuild {
                    reason: e.to_string(),
                }
            })?;

        let report = DeviceReport {
            requested: settings.device.to_string(),
            device,
            runtime_version: runtime.version.clone(),
            precision: ov::PRECISION,
        };
        info!(
            "OpenVINO {}: device {}, precision {}, requested {}",
            report.runtime_version,
            report.device.describe(),
            report.precision,
            report.requested
        );

        Ok(Self {
            decoder,
            report,
            device: target,
            input_dims,
            model_path,
            labels_path: model_config.labels.to_string_lossy().to_string(),
            state: Mutex::new(State {
                shapes: BTreeMap::new(),
                model,
                core: runtime.core,
                large: 1,
                reference: Vec::new(),
            }),
        })
    }

    /// Point `OpenVINO` at birda's cache directory. A cache that cannot be
    /// set up costs compile time, not correctness, so it only warns.
    fn configure_cache(runtime: &mut Runtime, target: &DeviceType<'_>) {
        let dir = openvino_cache_dir()
            .map(|root| {
                let versioned = root.join(cache::runtime_subdir(&runtime.version));
                (root, versioned)
            })
            .ok()
            .and_then(|(root, versioned)| {
                std::fs::create_dir_all(&versioned).ok()?;
                let removed = cache::prune(&root, ov::CACHE_SIZE_CAP_BYTES);
                if removed > 0 {
                    debug!(
                        "OpenVINO cache: removed {removed} old file(s) to stay under the size cap"
                    );
                }
                versioned.to_str().map(str::to_string)
            });
        let Some(dir) = dir else {
            warn!("OpenVINO compiled-model cache not enabled: no usable cache directory");
            return;
        };
        match runtime
            .core
            .set_property(target, &RwPropertyKey::CacheDir, &dir)
        {
            Ok(()) => debug!("OpenVINO cache directory: {dir}"),
            Err(e) => warn!("OpenVINO compiled-model cache not enabled: {e}"),
        }
    }

    /// Detect the model family from the unreshaped model's ports.
    fn describe_model(model: &Model) -> Result<(ModelConfig, Vec<i64>)> {
        let inputs = model
            .get_inputs_len()
            .map_err(model_load("reading the model inputs"))?;
        if inputs != 1 {
            return Err(Error::OpenVinoModelLoad {
                reason: format!("the model has {inputs} inputs; birda models take one"),
            });
        }
        let input = model
            .get_input_by_index(0)
            .map_err(model_load("reading the model input"))?;
        if input
            .get_element_type()
            .map_err(model_load("reading the input type"))?
            != ElementType::F32
        {
            return Err(Error::OpenVinoModelLoad {
                reason: "the model input is not 32-bit float".to_string(),
            });
        }
        let input_dims = port_dims(
            &input
                .get_partial_shape()
                .map_err(model_load("reading the input shape"))?,
            "input",
        )?;

        let outputs = model
            .get_outputs_len()
            .map_err(model_load("reading the model outputs"))?;
        let mut output_dims = Vec::with_capacity(outputs);
        let mut output_names = Vec::with_capacity(outputs);
        for index in 0..outputs {
            let output = model
                .get_output_by_index(index)
                .map_err(model_load("reading a model output"))?;
            output_dims.push(port_dims(
                &output
                    .get_partial_shape()
                    .map_err(model_load("reading an output shape"))?,
                "output",
            )?);
            output_names.push(output.get_name().ok());
        }
        // Names are passed all-or-nothing: a partial list would misalign roles.
        let names: Vec<String> = output_names.into_iter().flatten().collect();
        let names = if names.len() == outputs {
            names
        } else {
            Vec::new()
        };

        let config = detect_model_type(&input_dims, &output_dims, &names, None).map_err(|e| {
            Error::ClassifierBuild {
                reason: e.to_string(),
            }
        })?;
        Ok((config, input_dims))
    }

    /// The model configuration detected from the model's ports.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        self.decoder.config()
    }

    /// Species labels.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        self.decoder.labels()
    }

    /// Which device ran and how.
    #[must_use]
    pub const fn report(&self) -> &DeviceReport {
        &self.report
    }

    fn lock(&self) -> Result<MutexGuard<'_, State>> {
        self.state.lock().map_err(|_| Error::Inference {
            reason: "the OpenVINO engine failed earlier and cannot be used again".to_string(),
        })
    }

    /// Compile and verify the shapes a run with this batch size needs: one
    /// segment, and `batch_size` on the first call that asks for more than one.
    ///
    /// # Errors
    ///
    /// Any compile or verification failure. The run must not continue.
    pub fn prepare(&self, batch_size: usize) -> Result<()> {
        let mut state = self.lock()?;
        if state.large == 1 && batch_size > 1 && state.shapes.is_empty() {
            state.large = batch_size;
        }
        let large = state.large;
        for shape in [1, large] {
            self.ensure_shape(&mut state, shape)?;
        }
        drop(state);
        Ok(())
    }

    /// Run one segment.
    ///
    /// # Errors
    ///
    /// Inference failure or a segment of the wrong length.
    pub fn predict(&self, segment: &[f32]) -> Result<PredictionResult> {
        self.predict_batch(&[segment])?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Inference {
                reason: "OpenVINO returned no result for a single segment".to_string(),
            })
    }

    /// Run a batch, one result per segment, in input order.
    ///
    /// # Errors
    ///
    /// Inference failure or a segment of the wrong length.
    pub fn predict_batch(&self, segments: &[&[f32]]) -> Result<Vec<PredictionResult>> {
        let samples = self.decoder.config().sample_count;
        if let Some(bad) = segments.iter().find(|segment| segment.len() != samples) {
            return Err(Error::Inference {
                reason: format!(
                    "segment has {} samples, the model takes {samples}",
                    bad.len()
                ),
            });
        }
        let mut state = self.lock()?;
        let mut results = Vec::with_capacity(segments.len());
        for chunk in plan_chunks(segments.len(), state.large) {
            self.ensure_shape(&mut state, chunk.shape)?;
            let rows = &segments[chunk.start..chunk.start + chunk.valid];
            let shape = state
                .shapes
                .get_mut(&chunk.shape)
                .ok_or_else(|| Error::Inference {
                    reason: format!("batch shape {} is not compiled", chunk.shape),
                })?;
            results.extend(self.run(shape, rows)?);
        }
        drop(state);
        Ok(results)
    }

    /// Compile and verify `batch` unless already done.
    fn ensure_shape(&self, state: &mut State, batch: usize) -> Result<()> {
        if state.shapes.contains_key(&batch) {
            return Ok(());
        }
        let started = Instant::now();
        let mut shape = self.compile_shape(state, batch)?;
        info!(
            "OpenVINO: compiled batch {batch} in {:.1}s",
            started.elapsed().as_secs_f64()
        );
        self.verify(state, &mut shape)?;
        state.shapes.insert(batch, shape);
        Ok(())
    }

    fn compile_shape(&self, state: &mut State, batch: usize) -> Result<CompiledShape> {
        let samples = self.decoder.config().sample_count;
        let dims = fixed_dims(&self.input_dims, batch, samples)?;
        let partial = static_shape(&self.input_dims, batch, samples)?;
        state
            .model
            .reshape_single_input(&partial)
            .map_err(model_load("reshaping the model"))?;
        let mut compiled = state
            .core
            .compile_model(&state.model, self.device.to_owned())
            .map_err(|e| Error::OpenVinoModelLoad {
                reason: format!("compiling for {}: {e}", self.report.device.name),
            })?;

        // The precision is a hint the plugin may ignore, and results at f16
        // differ from ONNX Runtime, so check what the compiled model reports.
        let precision = compiled
            .get_property(&PropertyKey::Rw(RwPropertyKey::HintInferencePrecision))
            .map(Cow::into_owned)
            .map_err(model_load("reading the compiled precision"))?;
        if !precision.eq_ignore_ascii_case(ov::PRECISION) {
            return Err(Error::OpenVinoModelLoad {
                reason: format!(
                    "{} compiled the model at {precision} although {} was requested",
                    self.report.device.name,
                    ov::PRECISION
                ),
            });
        }
        // Never run on the CPU plugin by accident, whatever the target was.
        let executing = compiled
            .get_property(&PropertyKey::Other("EXECUTION_DEVICES".into()))
            .map(Cow::into_owned)
            .map_err(model_load("reading the executing device"))?;
        if executing.split_whitespace().any(|d| d.starts_with("CPU")) {
            return Err(Error::OpenVinoModelLoad {
                reason: format!("the model would execute on {executing}, not on a GPU"),
            });
        }

        let request = compiled
            .create_infer_request()
            .map_err(model_load("creating an inference request"))?;
        let input = Tensor::new(
            ElementType::F32,
            &Shape::new(&dims).map_err(model_load("building the input shape"))?,
        )
        .map_err(model_load("allocating the input"))?;
        Ok(CompiledShape {
            batch,
            request,
            _compiled: compiled,
            input,
        })
    }

    /// Run `rows` through a compiled shape; rows beyond `rows.len()` are zeros.
    fn run(&self, shape: &mut CompiledShape, rows: &[&[f32]]) -> Result<Vec<PredictionResult>> {
        let samples = self.decoder.config().sample_count;
        {
            let data = shape
                .input
                .get_data_mut::<f32>()
                .map_err(model_load("filling the input"))?;
            for (slot, row) in data.chunks_exact_mut(samples).zip(rows) {
                slot.copy_from_slice(row);
            }
            data[rows.len() * samples..].fill(0.0);
        }
        shape
            .request
            .set_input_tensor_by_index(0, &shape.input)
            .map_err(model_load("setting the input"))?;
        shape.request.infer().map_err(|e| Error::Inference {
            reason: format!("OpenVINO inference failed: {e}"),
        })?;

        let config = self.decoder.config();
        let scores = shape
            .request
            .get_output_tensor_by_index(config.predictions_index)
            .map_err(model_load("reading the class scores"))?;
        let embeddings = config
            .embeddings_index
            .map(|index| shape.request.get_output_tensor_by_index(index))
            .transpose()
            .map_err(model_load("reading the embeddings"))?;
        for tensor in std::iter::once(&scores).chain(embeddings.as_ref()) {
            if tensor
                .get_element_type()
                .map_err(model_load("reading an output type"))?
                != ElementType::F32
            {
                return Err(Error::Inference {
                    reason: "an OpenVINO output is not 32-bit float".to_string(),
                });
            }
        }
        let scores = scores
            .get_data::<f32>()
            .map_err(model_load("reading the class scores"))?;
        let embeddings = embeddings
            .as_ref()
            .map(Tensor::get_data::<f32>)
            .transpose()
            .map_err(model_load("reading the embeddings"))?;

        let mut results = self
            .decoder
            .decode_batch(shape.batch, scores, embeddings)
            .map_err(|e| Error::Inference {
                reason: e.to_string(),
            })?;
        results.truncate(rows.len());
        Ok(results)
    }

    /// Compare a freshly compiled shape with ONNX Runtime CPU on probe rows.
    ///
    /// Every row carries its own probe segment, so a batch that confuses or
    /// skips rows cannot pass on the strength of identical inputs.
    fn verify(&self, state: &mut State, shape: &mut CompiledShape) -> Result<()> {
        let samples = self.decoder.config().sample_count;
        self.ensure_reference(state, shape.batch.max(state.large))?;
        let probes: Vec<Vec<f32>> = (0..shape.batch)
            .map(|row| probe_segment(row, samples))
            .collect();
        let rows: Vec<&[f32]> = probes.iter().map(Vec::as_slice).collect();
        let actual = self.run(shape, &rows)?;

        let worst = check_against_reference(
            &self.report.device.describe(),
            shape.batch,
            &actual,
            &state.reference,
            ov::VERIFY_TOLERANCE,
        )?;
        debug!(
            "OpenVINO batch {} matches ONNX Runtime to {worst:e} on {} probe row(s)",
            shape.batch, shape.batch
        );
        Ok(())
    }

    /// Compute the ONNX Runtime CPU outputs for probe rows `0..rows`, once.
    fn ensure_reference(&self, state: &mut State, rows: usize) -> Result<()> {
        if state.reference.len() >= rows {
            return Ok(());
        }
        info!("OpenVINO: computing the ONNX Runtime reference for {rows} probe segment(s)");
        let samples = self.decoder.config().sample_count;
        let reference = ClassifierBuilder::new()
            .model_path(self.model_path.clone())
            .labels_path(self.labels_path.clone())
            .top_k(1)
            .build()
            .map_err(|e| Error::ClassifierBuild {
                reason: format!("ONNX Runtime reference for the OpenVINO check: {e}"),
            })?;
        let options = InferenceOptions::default();
        let mut outputs = Vec::with_capacity(rows);
        let mut next = 0;
        while next < rows {
            let end = (next + ov::REFERENCE_CHUNK).min(rows);
            let probes: Vec<Vec<f32>> =
                (next..end).map(|row| probe_segment(row, samples)).collect();
            let slices: Vec<&[f32]> = probes.iter().map(Vec::as_slice).collect();
            let results =
                reference
                    .predict_batch(&slices, &options)
                    .map_err(|e| Error::Inference {
                        reason: format!("ONNX Runtime reference run failed: {e}"),
                    })?;
            outputs.extend(results.into_iter().map(|result| RowOutput {
                scores: result.raw_scores,
                embeddings: result.embeddings,
            }));
            next = end;
        }
        state.reference = outputs;
        Ok(())
    }
}

/// Decide whether a compiled shape reproduced the ONNX Runtime reference.
///
/// `actual` holds one result per batch row and `reference` the ONNX Runtime
/// output for at least the same probe rows. Returns the largest absolute
/// difference when it is within `tolerance`.
///
/// # Errors
///
/// `OpenVinoVerificationFailed` when the difference exceeds the tolerance (a
/// NaN or infinite value counts as infinitely different), and
/// `OpenVinoModelLoad` when the row count or an output width does not line up.
fn check_against_reference(
    device: &str,
    batch: usize,
    actual: &[PredictionResult],
    reference: &[RowOutput],
    tolerance: f32,
) -> Result<f32> {
    if actual.len() != batch || reference.len() < batch {
        return Err(Error::OpenVinoModelLoad {
            reason: format!(
                "the check needs {batch} rows, got {} from OpenVINO and {} from ONNX Runtime",
                actual.len(),
                reference.len()
            ),
        });
    }
    let mismatch = |what: &str| Error::OpenVinoModelLoad {
        reason: format!("the {what} width differs from ONNX Runtime's"),
    };
    let mut worst = 0.0_f32;
    for (result, expected) in actual.iter().zip(reference) {
        worst = worst.max(
            max_abs_diff(&result.raw_scores, &expected.scores)
                .map_err(|_| mismatch("class score"))?,
        );
        if let (Some(a), Some(r)) = (&result.embeddings, &expected.embeddings) {
            worst = worst.max(max_abs_diff(a, r).map_err(|_| mismatch("embedding"))?);
        }
    }
    if worst > tolerance {
        return Err(Error::OpenVinoVerificationFailed {
            device: device.to_string(),
            shape: batch,
            difference: worst,
            tolerance,
        });
    }
    Ok(worst)
}

/// Input dimensions with the batch set and a dynamic sample axis filled in.
fn fixed_dims(dims: &[i64], batch: usize, samples: usize) -> Result<Vec<i64>> {
    let to_i64 = |value: usize| {
        i64::try_from(value).map_err(|_| Error::OpenVinoModelLoad {
            reason: "batch size too large".to_string(),
        })
    };
    let last = dims.len().saturating_sub(1);
    dims.iter()
        .enumerate()
        .map(|(axis, dim)| match (axis, *dim) {
            (0, _) => to_i64(batch),
            (axis, -1) if axis == last => to_i64(samples),
            (_, -1) => Err(Error::OpenVinoModelLoad {
                reason: "the model input has a dynamic axis other than batch and samples"
                    .to_string(),
            }),
            (_, fixed) => Ok(fixed),
        })
        .collect()
}

fn static_shape(dims: &[i64], batch: usize, samples: usize) -> Result<PartialShape> {
    let fixed = fixed_dims(dims, batch, samples)?;
    let rank = i64::try_from(fixed.len()).map_err(|_| Error::OpenVinoModelLoad {
        reason: "input rank too large".to_string(),
    })?;
    PartialShape::new_static(rank, &fixed).map_err(model_load("building a static input shape"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_chunks_pins_the_four_rules_with_literal_cases() {
        let chunk = |start, valid, shape| Chunk {
            start,
            valid,
            shape,
        };
        // n == s: one full chunk.
        assert_eq!(plan_chunks(8, 8), vec![chunk(0, 8, 8)]);
        // 2n <= s: n singles (no padding of a small tail to the large shape).
        assert_eq!(
            plan_chunks(4, 8),
            vec![
                chunk(0, 1, 1),
                chunk(1, 1, 1),
                chunk(2, 1, 1),
                chunk(3, 1, 1)
            ]
        );
        // 2n > s: one padded chunk of s.
        assert_eq!(plan_chunks(5, 8), vec![chunk(0, 5, 8)]);
        // n > s: full chunks, then the remainder by the same rule.
        assert_eq!(
            plan_chunks(19, 8),
            vec![
                chunk(0, 8, 8),
                chunk(8, 8, 8),
                chunk(16, 1, 1),
                chunk(17, 1, 1),
                chunk(18, 1, 1)
            ]
        );
        assert_eq!(plan_chunks(0, 8), Vec::<Chunk>::new());
    }

    #[test]
    fn plan_chunks_covers_every_input_once_in_order_with_only_two_shapes() {
        for large in 1..=33 {
            for n in 0..=3 * large {
                let chunks = plan_chunks(n, large);
                let mut next = 0;
                for chunk in &chunks {
                    assert_eq!(chunk.start, next, "n={n} large={large}: gap or overlap");
                    assert!(
                        chunk.shape == 1 || chunk.shape == large,
                        "n={n} large={large}: shape {}",
                        chunk.shape
                    );
                    assert!(
                        chunk.valid >= 1 && chunk.valid <= chunk.shape,
                        "n={n} large={large}: {chunk:?}"
                    );
                    next += chunk.valid;
                }
                assert_eq!(next, n, "n={n} large={large}: coverage");
            }
        }
    }

    #[test]
    fn a_zero_large_shape_is_treated_as_one() {
        assert_eq!(plan_chunks(2, 0).len(), 2);
    }

    fn result(scores: &[f32], embeddings: Option<&[f32]>) -> PredictionResult {
        PredictionResult {
            model_type: birdnet_onnx::ModelType::BirdNetV24,
            predictions: Vec::new(),
            embeddings: embeddings.map(<[f32]>::to_vec),
            raw_scores: scores.to_vec(),
        }
    }

    fn reference(scores: &[f32], embeddings: Option<&[f32]>) -> RowOutput {
        RowOutput {
            scores: scores.to_vec(),
            embeddings: embeddings.map(<[f32]>::to_vec),
        }
    }

    #[test]
    fn the_check_passes_within_the_tolerance_and_returns_the_worst_difference() {
        let actual = [
            result(&[0.0, 1.0], Some(&[0.5])),
            result(&[0.25, 1.0], Some(&[0.5])),
        ];
        let expected = [
            reference(&[0.0, 1.0], Some(&[0.5])),
            reference(&[0.0, 1.0], Some(&[0.5])),
            // A longer reference (more probe rows than this shape) is fine.
            reference(&[9.0, 9.0], None),
        ];

        assert_eq!(
            check_against_reference("GPU", 2, &actual, &expected, 0.25).unwrap(),
            0.25
        );
    }

    #[test]
    fn the_check_fails_with_the_whole_verification_message() {
        let actual = [result(&[0.0, 1.5], None)];
        let expected = [reference(&[0.0, 1.0], None)];

        let err =
            check_against_reference("GPU (Iris Xe)", 1, &actual, &expected, 0.25).unwrap_err();

        assert_eq!(
            err.to_string(),
            "OpenVINO on GPU (Iris Xe) gives different results from ONNX Runtime for batch shape \
             1: largest difference 5e-1, allowed 2.5e-1. \
             Results from this device would not be trustworthy, so the run was stopped. \
             Use --cpu or another provider, a different OpenVINO device (see `birda openvino devices`), \
             or update the Intel GPU driver and the OpenVINO runtime"
        );
    }

    #[test]
    fn the_check_looks_at_embeddings_nan_and_every_row() {
        let tolerance = 0.1;
        // Only the embeddings differ.
        let embedding_off = check_against_reference(
            "GPU",
            1,
            &[result(&[0.0], Some(&[1.0]))],
            &[reference(&[0.0], Some(&[2.0]))],
            tolerance,
        );
        assert!(matches!(
            embedding_off,
            Err(Error::OpenVinoVerificationFailed { shape: 1, .. })
        ));
        // Only the second row differs: the first row's match must not hide it.
        let second_row_off = check_against_reference(
            "GPU",
            2,
            &[result(&[0.0], None), result(&[5.0], None)],
            &[reference(&[0.0], None), reference(&[0.0], None)],
            tolerance,
        );
        assert!(matches!(
            second_row_off,
            Err(Error::OpenVinoVerificationFailed { .. })
        ));
        // NaN never passes.
        let nan = check_against_reference(
            "GPU",
            1,
            &[result(&[f32::NAN], None)],
            &[reference(&[0.0], None)],
            tolerance,
        );
        assert!(matches!(nan, Err(Error::OpenVinoVerificationFailed { .. })));
    }

    #[test]
    fn the_check_refuses_rows_that_do_not_line_up() {
        let one_row = [result(&[0.0], None)];
        let two_refs = [reference(&[0.0], None), reference(&[0.0], None)];

        // Fewer results than the batch: a zip would silently skip the rest.
        assert_eq!(
            check_against_reference("GPU", 2, &one_row, &two_refs, 0.1)
                .unwrap_err()
                .to_string(),
            "OpenVINO could not load the model: the check needs 2 rows, got 1 from OpenVINO and 2 \
             from ONNX Runtime"
        );
        // A different class-score width.
        assert_eq!(
            check_against_reference(
                "GPU",
                1,
                &[result(&[0.0, 0.0], None)],
                &[reference(&[0.0], None)],
                0.1
            )
            .unwrap_err()
            .to_string(),
            "OpenVINO could not load the model: the class score width differs from ONNX Runtime's"
        );
    }

    fn denied_node() -> Vec<RenderNode> {
        vec![RenderNode {
            path: "/dev/dri/renderD128".to_string(),
            vendor: Some(0x8086),
            openable: false,
        }]
    }

    fn not_found(found: &[&str]) -> ResolveError {
        ResolveError::NotFound {
            requested: "dgpu".to_string(),
            found: found.iter().map(ToString::to_string).collect(),
        }
    }

    #[test]
    fn resolve_errors_map_to_their_user_facing_errors() {
        // Unsupported keeps its reason whatever the host looks like.
        assert_eq!(
            map_resolve_error(
                ResolveError::Unsupported {
                    requested: "npu".to_string(),
                    reason: "the NPU is not supported yet".to_string(),
                },
                Some(Os::Linux),
                false,
                denied_node
            )
            .to_string(),
            "OpenVINO backend: the NPU is not supported yet"
        );
        // A listed GPU: the devices are named and there is no hint.
        assert_eq!(
            map_resolve_error(
                not_found(&["CPU", "GPU (Iris Xe)"]),
                Some(Os::Linux),
                true,
                || panic!("the render nodes must not be read when a GPU is listed")
            )
            .to_string(),
            "OpenVINO device 'dgpu' is not available. OpenVINO found: CPU; GPU (Iris Xe)"
        );
        // Nothing listed on Windows: "nothing", and no Linux advice.
        assert_eq!(
            map_resolve_error(not_found(&[]), Some(Os::Windows), false, denied_node).to_string(),
            "OpenVINO device 'dgpu' is not available. OpenVINO found: nothing"
        );
        // Nothing listed on Linux with an Intel node this user cannot open.
        assert_eq!(
            map_resolve_error(not_found(&["CPU"]), Some(Os::Linux), false, denied_node).to_string(),
            "no permission to use the Intel GPU render node /dev/dri/renderD128. \
             Add your user to the 'render' group (sudo usermod -aG render $USER) and log in again"
        );
        // Nothing listed on Linux with no render node at all.
        assert_eq!(
            map_resolve_error(not_found(&["CPU"]), Some(Os::Linux), false, Vec::new).to_string(),
            "OpenVINO device 'dgpu' is not available. OpenVINO found: CPU\n\
             No GPU render node exists under /dev/dri. Check that the Intel GPU driver is installed \
             and, in a container, that /dev/dri is passed through"
        );
    }

    #[test]
    fn input_dims_get_the_batch_and_a_dynamic_sample_axis_filled() {
        assert_eq!(fixed_dims(&[1, 144_000], 8, 144_000).unwrap(), [8, 144_000]);
        assert_eq!(fixed_dims(&[-1, -1], 4, 160_000).unwrap(), [4, 160_000]);
        assert_eq!(
            fixed_dims(&[-1, 1, 144_000], 2, 144_000).unwrap(),
            [2, 1, 144_000]
        );
        assert!(matches!(
            fixed_dims(&[-1, -1, 144_000], 2, 144_000),
            Err(Error::OpenVinoModelLoad { .. })
        ));
    }
}
