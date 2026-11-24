//! Regional Perch score calibration (#449): `models install` stores a tile's
//! softmax rest logit, and the ONNX Runtime engine applies the stored value.
//!
//! The install test drives the binary against a seeded registry with downloads
//! served from a loopback server, so it runs the real handler and writes a real
//! `config.toml` without ONNX Runtime. The inference test needs the ONNX
//! Runtime shared library, which birda loads dynamically, and skips where it is
//! absent (see `tests/geomodel_range_filter.rs`). Its model is a tiny stand-in
//! built by `tests/fixtures/make_fixture_perch.py`.
// Integration test crate: see `tests/model_manifest.rs` for why these are allowed.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use assert_cmd::cargo::cargo_bin_cmd;
use birda::config::{Config, InferenceDevice, OpenVinoSettings, load_config_file};
use birda::constants::CONFIG_DIR_ENV;
use birda::inference::{BirdClassifier, InferenceOptions};
use serde_json::Value;

const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// The loopback server stops accepting after this long, so a run that never
/// connects cannot leave its thread behind.
const SERVER_DEADLINE: Duration = Duration::from_secs(30);

/// Files the two installs download: a model and a label set each.
const DOWNLOADS: usize = 4;

/// The regional tile the install test uses.
const REGION: &str = "nordic";

/// Answer the next [`DOWNLOADS`] connections, one at a time, with a small
/// `200 OK` body, and return the base URL.
fn serve_downloads() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let deadline = Instant::now() + SERVER_DEADLINE;
        let mut served = 0;
        while served < DOWNLOADS && Instant::now() < deadline {
            let Ok((mut stream, _)) = listener.accept() else {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            // Read the whole request head before replying, so closing the
            // socket does not reset the connection under the client.
            let mut request = Vec::new();
            let mut chunk = [0_u8; 1024];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") && request.len() < 16_384 {
                match stream.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => request.extend_from_slice(&chunk[..n]),
                }
            }
            let body = b"fixture";
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body);
            served += 1;
        }
    });
    base
}

/// Seed `home` with the bundled registry, newer than the bundled one so the
/// cached copy is kept, with the global and `nordic` Perch downloads pointing at
/// `base`. Returns the `nordic` rest logit the seeded registry carries. The
/// geomodel is dropped so the side install never reaches the network.
fn seed_registry(home: &Path, base: &str) -> f64 {
    let bundled = Path::new(env!("CARGO_MANIFEST_DIR")).join("registry.json");
    let mut registry: Value =
        serde_json::from_slice(&std::fs::read(bundled).unwrap()).expect("bundled registry parses");
    registry["registry_version"] = Value::from(9999);
    registry.as_object_mut().unwrap().remove("range_filter");

    let entry = registry["models"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|model| model["id"] == "perch-v2")
        .expect("bundled registry has perch-v2");
    let mut rest_logit = None;
    for (n, variant) in entry["variants"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        let region = variant.get("region").and_then(Value::as_str);
        if region.is_some_and(|region| region != REGION) {
            continue;
        }
        if region == Some(REGION) {
            rest_logit = variant["score_calibration"]["rest_logit"].as_f64();
        }
        for (asset, name) in [("model", "model.onnx"), ("labels", "labels.txt")] {
            variant[asset]["url"] = Value::from(format!("{base}/{n}-{name}"));
            variant[asset].as_object_mut().unwrap().remove("sha256");
        }
    }
    std::fs::write(home.join("registry.json"), registry.to_string()).unwrap();
    rest_logit.expect("the bundled registry carries a rest logit for the nordic tile")
}

/// Run `birda` in an isolated home, with a null stdin (`assert_cmd`'s `output()`).
fn run_in(home: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.current_dir(home)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_DATA_HOME", home.join("data"))
        .env(CONFIG_DIR_ENV, home)
        .env_remove("BIRDA_OUTPUT_MODE")
        .env_remove("HF_ENDPOINT")
        .env_remove("RUST_LOG")
        .timeout(COMMAND_TIMEOUT);
    cmd.args(args);
    cmd.output().expect("birda should run")
}

fn install(home: &Path, args: &[&str]) {
    let mut full = vec!["models", "install", "perch-v2"];
    full.extend_from_slice(args);
    let output = run_in(home, &full);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_install_stores_the_tile_rest_logit_and_leaves_the_global_model_without_one() {
    let home = tempfile::tempdir().unwrap();
    let rest_logit = seed_registry(home.path(), &serve_downloads());

    install(home.path(), &["--region", REGION]);
    install(home.path(), &[]);

    let text = std::fs::read_to_string(home.path().join("config.toml")).unwrap();
    let config: Config = toml::from_str(&text).unwrap();
    #[allow(clippy::cast_possible_truncation)]
    let expected = rest_logit as f32;
    assert_eq!(
        config.models["perch-v2-nordic"]
            .softmax_rest_logit
            .map(f32::to_bits),
        Some(expected.to_bits()),
        "the tile keeps the registry's value, got:\n{text}"
    );
    assert_eq!(
        config.models["perch-v2"].softmax_rest_logit, None,
        "the global model has no rest logit, got:\n{text}"
    );
    assert_eq!(
        text.matches("softmax_rest_logit").count(),
        1,
        "only the tile carries the key, got:\n{text}"
    );
}

/// The warning a regional Perch entry with no stored value gets, as the run prints it.
const UNCALIBRATED_WARNING: &str = "model 'perch-v2-nordic' has no stored score calibration, \
     so its scores are uncalibrated and run higher than the global Perch model's. Reinstall \
     it to fix this: birda models install perch-v2 --region nordic";

/// Stderr of `birda -m perch-v2-nordic` on an input that does not exist, with a
/// `perch-v2-nordic` entry written by `models install` that has `extra_keys` appended.
///
/// The model files are placeholders. The warning is emitted before the run
/// initializes ONNX Runtime, and the run then fails (missing runtime or missing
/// input), so only stderr is read and no model is ever loaded.
fn analyze_stderr(extra_keys: &str) -> String {
    let home = tempfile::tempdir().unwrap();
    let model = home.path().join("perch-v2-nordic.onnx");
    let labels = home.path().join("perch-v2-nordic-labels.txt");
    std::fs::write(&model, b"placeholder").unwrap();
    std::fs::write(&labels, b"placeholder").unwrap();
    std::fs::write(
        home.path().join("config.toml"),
        format!(
            "[models.perch-v2-nordic]\npath = '{}'\nlabels = '{}'\ntype = \"perch-v2\"\n\
             registry_id = \"perch-v2\"\ninstalled_version = \"2\"\ninstalled_build = 1\n\
             region = \"nordic\"\nvariant = \"no-dft-fp32\"\n{extra_keys}",
            model.display(),
            labels.display()
        ),
    )
    .unwrap();
    let output = run_in(
        home.path(),
        &["-m", "perch-v2-nordic", "--cpu", "no-such-input.wav"],
    );
    assert_eq!(output.status.code(), Some(1));
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn test_analyze_warns_once_for_a_regional_perch_entry_without_a_stored_value() {
    let stderr = analyze_stderr("");
    assert_eq!(
        stderr.matches(UNCALIBRATED_WARNING).count(),
        1,
        "one warning per run, got:\n{stderr}"
    );

    let stderr = analyze_stderr("softmax_rest_logit = 9.75\n");
    assert_eq!(
        stderr.matches("no stored score calibration").count(),
        0,
        "a calibrated entry is silent, got:\n{stderr}"
    );
}

/// Whether the ONNX Runtime shared library can be loaded.
///
/// Resolved once per test binary: the probe itself is cheap, but calling it
/// per test would repeat the library search.
fn onnx_runtime_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| birda::inference::ensure_runtime_available().is_ok())
}

/// Skip the calling test when the ONNX Runtime is unavailable.
///
/// Rust has no first-class skip, so this returns from the test after saying
/// why. Run with `cargo test -- --nocapture` to see the notice.
macro_rules! require_onnx_runtime {
    () => {
        if !onnx_runtime_available() {
            eprintln!("skipping: ONNX Runtime not available in this environment");
            return;
        }
    };
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Confidence of each fixture label for a silent segment, in label order.
///
/// The fixture's logits are the constants `[2.0, 1.0, 0.0]` whatever the input,
/// so the segment's content does not matter.
fn fixture_confidences(rest_logit_line: &str) -> Vec<f32> {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    let model = fixture_path("fixture-perch.onnx");
    let labels = fixture_path("fixture-perch-labels.txt");
    std::fs::write(
        &config_path,
        format!(
            "[models.fixture]\npath = '{}'\nlabels = '{}'\ntype = \"perch-v2\"\n{rest_logit_line}",
            model.display(),
            labels.display()
        ),
    )
    .unwrap();
    let config = load_config_file(&config_path).unwrap();
    let classifier = BirdClassifier::from_config(
        &config.models["fixture"],
        InferenceDevice::Cpu,
        &OpenVinoSettings::default(),
        0.0,
        3,
        None,
        None,
    )
    .unwrap();
    let result = classifier
        .predict(&vec![0.0_f32; 160_000], &InferenceOptions::default())
        .unwrap();
    let mut by_label = vec![f32::NAN; 3];
    for prediction in &result.predictions {
        by_label[prediction.index] = prediction.confidence;
    }
    by_label
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() < 1e-6,
            "got {actual:?}, expected {expected:?}"
        );
    }
}

#[test]
fn test_ort_engine_applies_the_stored_rest_logit() {
    require_onnx_runtime!();

    let logits = [2.0_f32, 1.0, 0.0];

    let total: f32 = logits.iter().chain(&[3.0]).map(|z| z.exp()).sum();
    assert_close(
        &fixture_confidences("softmax_rest_logit = 3.0\n"),
        &logits.map(|z| z.exp() / total),
    );

    let total: f32 = logits.iter().map(|z| z.exp()).sum();
    assert_close(&fixture_confidences(""), &logits.map(|z| z.exp() / total));
}

/// sha256 of each committed fixture, as produced by
/// `tests/fixtures/make_fixture_perch.py`.
///
/// The generator is deterministic, but nothing else ties the committed binary to
/// it: an edit to the script without a regenerate, or a regenerate with a
/// different `onnx` version, would leave the two out of step and the suite green.
/// When you deliberately regenerate a fixture, update its hash here in the same
/// commit.
const FIXTURE_HASHES: [(&str, &str); 2] = [
    (
        "fixture-perch.onnx",
        "f776e33dda86051dd51297d119b1732c0177d1355eb3c5668510bffbbfb56b9f",
    ),
    (
        "fixture-perch-labels.txt",
        "0e74a230523741de66202492d810cac99cbb25aaa565c555ebce94b7a72e5019",
    ),
];

#[test]
fn test_fixture_perch_files_match_their_recorded_hashes() {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;

    for (name, expected) in FIXTURE_HASHES {
        let bytes = std::fs::read(fixture_path(name)).unwrap();
        let actual = Sha256::digest(&bytes)
            .iter()
            .fold(String::new(), |mut hex, byte| {
                let _ = write!(hex, "{byte:02x}");
                hex
            });
        assert_eq!(
            actual, expected,
            "{name} changed; if you regenerated it on purpose, update FIXTURE_HASHES"
        );
    }
}
