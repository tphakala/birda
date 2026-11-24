//! `models install` and the documents a model's publisher requires agreement to (#450).
//!
//! These drive the binary against a seeded registry whose `birdnet-v24` entry
//! carries a licence document and an acceptable use document that requires
//! agreement. The model and label downloads are served by a loopback server in
//! this file, so a successful install runs the real handler and writes a real
//! `config.toml`.
// Integration test crate: see `tests/model_manifest.rs` for why these are allowed.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::time::{Duration, Instant};

use assert_cmd::cargo::cargo_bin_cmd;
use birda::config::{AcceptedDocument, Config};
use birda::constants::CONFIG_DIR_ENV;
use serde_json::{Value, json};

const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// The loopback server stops accepting after this long, so a run that never
/// connects cannot leave its thread behind.
const SERVER_DEADLINE: Duration = Duration::from_secs(30);

/// Files an install of the seeded entry downloads: the model and one label set.
const SEEDED_DOWNLOADS: usize = 2;

const AUP_URL: &str = "https://huggingface.co/example/x/resolve/r1/ACCEPTABLE_USE.md";
const AUP_REVISION: &str = "r1";
const AUP_SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// Answer the next [`SEEDED_DOWNLOADS`] connections, one at a time, with a
/// small `200 OK` body, and return the base URL.
fn serve_downloads() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let deadline = Instant::now() + SERVER_DEADLINE;
        let mut served = 0;
        while served < SEEDED_DOWNLOADS && Instant::now() < deadline {
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
/// cached copy is kept, and `birdnet-v24` changed to carry the two documents,
/// one label set, and downloads pointing at `base`. The geomodel is dropped so
/// the side install never reaches the network.
fn seed_registry(home: &Path, base: &str) {
    let bundled = Path::new(env!("CARGO_MANIFEST_DIR")).join("registry.json");
    let mut registry: Value =
        serde_json::from_slice(&std::fs::read(bundled).unwrap()).expect("bundled registry parses");
    registry["registry_version"] = Value::from(9999);
    registry.as_object_mut().unwrap().remove("range_filter");

    let entry = registry["models"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|model| model["id"] == "birdnet-v24")
        .expect("bundled registry has birdnet-v24");
    entry["documents"] = json!([
        {
            "kind": "license",
            "requires_agreement": false,
            "url": "https://example.org/LICENSE.md",
        },
        {
            "kind": "acceptable_use",
            "requires_agreement": true,
            "url": AUP_URL,
            "revision": AUP_REVISION,
            "sha256": AUP_SHA256,
        },
    ]);
    let files = &mut entry["files"];
    files["model"]["url"] = Value::from(format!("{base}/model.onnx"));
    files["model"].as_object_mut().unwrap().remove("sha256");
    let languages = files["labels"]["languages"].as_array_mut().unwrap();
    languages.retain(|language| language["code"] == "en");
    assert_eq!(languages.len(), 1, "the seeded entry keeps one label set");
    languages[0]["url"] = Value::from(format!("{base}/labels.txt"));
    languages[0].as_object_mut().unwrap().remove("sha256");

    std::fs::write(home.join("registry.json"), registry.to_string()).unwrap();
}

/// Run `birda` in an isolated home, with a null stdin (`assert_cmd`'s `output()`).
fn run_in(home: &Path, hf_endpoint: Option<&str>, args: &[&str]) -> std::process::Output {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.current_dir(home)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_DATA_HOME", home.join("data"))
        .env(CONFIG_DIR_ENV, home)
        .env_remove("BIRDA_OUTPUT_MODE")
        .timeout(COMMAND_TIMEOUT);
    match hf_endpoint {
        Some(endpoint) => cmd.env("HF_ENDPOINT", endpoint),
        None => cmd.env_remove("HF_ENDPOINT"),
    };
    cmd.args(args);
    cmd.output().expect("birda should run")
}

/// The refusal text for the given URL of the acceptable use document.
fn refusal_message(url: &str) -> String {
    format!(
        "installing BirdNET v2.4 requires agreement to acceptable_use ({url}); no prompt can \
         be shown without a terminal or under structured output, so read each one and run \
         the install again with --yes to agree"
    )
}

/// Nothing an install writes exists: no entry for the model in `config.toml`,
/// and no model file.
fn assert_nothing_installed(home: &Path) {
    let config = std::fs::read_to_string(home.join("config.toml")).unwrap_or_default();
    assert!(
        !config.contains("birdnet-v24"),
        "a refused install must not write a config entry, got:\n{config}"
    );
    let models = home.join("models");
    let files = std::fs::read_dir(&models).map_or(0, Iterator::count);
    assert_eq!(files, 0, "a refused install must not download anything");
}

#[test]
fn test_install_gates_unagreed_documents_without_a_prompt() {
    // JSON mode with a mirror: one `error` event, naming the mirrored URL.
    let home = tempfile::tempdir().unwrap();
    seed_registry(home.path(), &serve_downloads());
    let mirrored = "https://mirror.example/example/x/resolve/r1/ACCEPTABLE_USE.md";
    let output = run_in(
        home.path(),
        Some("https://mirror.example"),
        &["--output-mode", "json", "models", "install", "birdnet-v24"],
    );
    assert_eq!(output.status.code(), Some(1));
    let envelope: Value = serde_json::from_slice(&output.stdout)
        .expect("stdout is exactly one JSON value, the error event");
    assert_eq!(envelope["event"], "error");
    assert_eq!(
        envelope["payload"],
        json!({
            "code": "agreement_required",
            "severity": "fatal",
            "message": refusal_message(mirrored),
        })
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        format!("error: {}\n", refusal_message(mirrored))
    );
    assert_nothing_installed(home.path());

    // Text mode with no terminal: the same refusal as plain text, nothing on
    // stdout. A gate keyed on structured output instead of on the missing
    // prompt would install here.
    let home = tempfile::tempdir().unwrap();
    seed_registry(home.path(), &serve_downloads());
    let output = run_in(home.path(), None, &["models", "install", "birdnet-v24"]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&output.stdout), "");
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        format!("error: {}\n", refusal_message(AUP_URL))
    );
    assert_nothing_installed(home.path());

    // `--yes` is how birda-gui agrees: the install succeeds and records what
    // was agreed to, the licence document excluded.
    let home = tempfile::tempdir().unwrap();
    let base = serve_downloads();
    seed_registry(home.path(), &base);
    let output = run_in(
        home.path(),
        None,
        &[
            "--output-mode",
            "json",
            "--yes",
            "models",
            "install",
            "birdnet-v24",
        ],
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: Value =
        serde_json::from_slice(&output.stdout).expect("stdout is one JSON value, the result");
    assert_eq!(envelope["event"], "result");
    assert_eq!(envelope["payload"]["result_type"], "model_installed");

    let text = std::fs::read_to_string(home.path().join("config.toml")).unwrap();
    let config: Config = toml::from_str(&text).unwrap();
    assert_eq!(
        config.models["birdnet-v24"].accepted_documents,
        vec![AcceptedDocument {
            kind: "acceptable_use".to_string(),
            revision: Some(AUP_REVISION.to_string()),
            sha256: Some(AUP_SHA256.to_string()),
        }]
    );
}
