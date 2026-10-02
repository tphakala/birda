//! End-to-end check that inputs sharing a file name do not share an output
//! (issue #414).
//!
//! The run needs a real classifier, so it needs the ONNX Runtime shared library
//! and a model, which CI has neither of. The test skips, saying why, unless the
//! runtime loads and `BIRDA_TEST_MODEL` and `BIRDA_TEST_LABELS` name a
//! `BirdNET` v2.4 model and its labels. Run with `cargo test -- --nocapture` to
//! see the notice. The naming rules themselves are covered without a model by
//! the unit tests in `src/pipeline/coordinator.rs`.
// Integration test crate. `unwrap`, `expect` and `panic` are how a test reports
// failure, not unhandled error paths, so rewriting them into propagated errors
// would only hide which assertion fired. The crate-level deny still governs
// everything birda ships. The one exact float comparison is on a whole number of
// seconds written to a file and read back, not on a computed value.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use assert_cmd::cargo::cargo_bin_cmd;
use birda::constants::CONFIG_DIR_ENV;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Model file for the run (`BirdNET` v2.4).
const MODEL_ENV: &str = "BIRDA_TEST_MODEL";
/// Labels file that goes with the model.
const LABELS_ENV: &str = "BIRDA_TEST_LABELS";
/// Model type passed to birda for the model above.
const MODEL_TYPE: &str = "birdnet-v24";
/// Sample rate of the generated audio, the model's own, so no resampling runs.
const SAMPLE_RATE: u32 = 48_000;
/// A run over a few short files of silence finishes in seconds; this is a
/// ceiling for a hung run, not an expectation.
const RUN_TIMEOUT: Duration = Duration::from_mins(2);

/// The model and labels, or `None` (after saying why) when the run cannot happen.
fn test_model() -> Option<(PathBuf, PathBuf)> {
    if birda::inference::ensure_runtime_available().is_err() {
        eprintln!("skipping: ONNX Runtime not available in this environment");
        return None;
    }
    let (Some(model), Some(labels)) = (
        std::env::var_os(MODEL_ENV).map(PathBuf::from),
        std::env::var_os(LABELS_ENV).map(PathBuf::from),
    ) else {
        eprintln!("skipping: {MODEL_ENV} and {LABELS_ENV} are not set");
        return None;
    };
    Some((model, labels))
}

/// Write `seconds` of silence as a mono 16-bit WAV.
///
/// The length is what tells two outputs apart: each result file records its
/// input's duration, so a file overwritten by another input shows it.
fn write_wav(path: &Path, seconds: u32) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for _ in 0..SAMPLE_RATE * seconds {
        writer.write_sample(0_i16).unwrap();
    }
    writer.finalize().unwrap();
}

/// Run birda over `input` with `-o out`, and return the `file_completed`
/// payloads from its NDJSON stdout.
fn run(model: &Path, labels: &Path, config_dir: &Path, input: &Path, out: &Path) -> Vec<Value> {
    let output = cargo_bin_cmd!("birda")
        .env(CONFIG_DIR_ENV, config_dir)
        .env_remove("BIRDA_OUTPUT_DIR")
        .env_remove("BIRDA_FORMAT")
        .env_remove("BIRDA_MODEL")
        .timeout(RUN_TIMEOUT)
        .args([
            "--output-mode",
            "ndjson",
            "--no-progress",
            "--cpu",
            "-f",
            "json",
        ])
        .arg("--model-path")
        .arg(model)
        .arg("--labels-path")
        .arg(labels)
        .args(["--model-type", MODEL_TYPE])
        .arg("-o")
        .arg(out)
        .arg(input)
        .output()
        .expect("birda should run");
    assert!(
        output.status.success(),
        "birda failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["event"] == "file_completed")
        .map(|event| event["payload"].clone())
        .collect()
}

/// The payload for `file`, which must be the only one.
fn payload_for<'a>(payloads: &'a [Value], file: &Path) -> &'a Value {
    let matching: Vec<&Value> = payloads
        .iter()
        .filter(|p| p["file"] == json!(file))
        .collect();
    assert_eq!(
        matching.len(),
        1,
        "one file_completed for {}",
        file.display()
    );
    matching[0]
}

/// The audio duration a result file recorded.
fn recorded_duration(result_file: &Path) -> f64 {
    let text = std::fs::read_to_string(result_file).unwrap();
    let result: Value = serde_json::from_str(&text).unwrap();
    result["summary"]["audio_duration_seconds"]
        .as_f64()
        .unwrap()
}

#[test]
fn test_same_named_inputs_each_get_their_own_output() {
    let Some((model, labels)) = test_model() else {
        return;
    };
    let work = tempfile::tempdir().unwrap();
    let config_dir = tempfile::tempdir().unwrap();
    let input = work.path().join("in");
    let out = work.path().join("out");
    // The two folders hold an `x.wav` each, and `b` also holds a differently
    // typed `x`: three inputs, one stem. The WAV bytes in `x.flac` decode
    // because the decoder reads the content, not the name.
    // Each path is joined one component at a time, so it is spelled with the
    // platform's separator, as birda spells the paths it reports.
    let a_wav = input.join("a").join("x.wav");
    let b_wav = input.join("b").join("x.wav");
    let b_flac = input.join("b").join("x.flac");
    write_wav(&a_wav, 6);
    write_wav(&b_wav, 9);
    write_wav(&b_flac, 12);
    let expected = [
        (&a_wav, out.join("a").join("x.wav.BirdNET.json"), 6.0),
        (&b_wav, out.join("b").join("x.wav.BirdNET.json"), 9.0),
        (&b_flac, out.join("b").join("x.flac.BirdNET.json"), 12.0),
    ];

    let first = run(&model, &labels, config_dir.path(), &input, &out);

    assert_eq!(first.len(), 3);
    for (file, result_file, seconds) in &expected {
        let payload = payload_for(&first, file);
        assert_eq!(payload["status"], "processed");
        assert_eq!(payload["output_files"], json!({ "json": result_file }));
        assert_eq!(recorded_duration(result_file), *seconds);
    }

    // A second run finds each input's own output and skips it, naming the same
    // files.
    let second = run(&model, &labels, config_dir.path(), &input, &out);

    assert_eq!(second.len(), 3);
    for (file, result_file, _) in &expected {
        let payload = payload_for(&second, file);
        assert_eq!(payload["status"], "skipped");
        assert_eq!(payload["output_files"], json!({ "json": result_file }));
    }
}
