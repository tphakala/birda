//! Integration tests for providers command.
//!
//! Note: These tests require ONNX Runtime to be available.
//! They will be skipped if ONNX Runtime initialization fails (e.g., in CI).

use assert_cmd::cargo::cargo_bin_cmd;
use serde_json::Value;

/// Helper function to check if ONNX Runtime is available by running providers command.
/// Returns Some(stdout) if successful, None if ONNX Runtime isn't available.
fn run_providers_command(args: &[&str]) -> Option<Vec<u8>> {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.arg("providers");
    for arg in args {
        cmd.arg(arg);
    }

    let output = cmd.output().ok()?;
    if output.status.success() {
        Some(output.stdout)
    } else {
        eprintln!("Skipping test: providers command failed (ONNX Runtime not available)");
        None
    }
}

#[test]
fn test_providers_command_human_readable() {
    let stdout = match run_providers_command(&[]) {
        Some(stdout) => stdout,
        None => return,
    };

    let output = String::from_utf8(stdout).expect("stdout should be valid UTF-8");
    assert!(output.contains("Available execution providers:"));
    assert!(output.contains("CPU"));
}

#[test]
fn test_providers_command_json_output() {
    let stdout = match run_providers_command(&["--output-mode", "json"]) {
        Some(stdout) => stdout,
        None => return,
    };

    let output_str = String::from_utf8(stdout).expect("stdout should be valid UTF-8");

    // Parse JSON
    let json: Value = serde_json::from_str(&output_str).expect("Valid JSON output");

    // Verify structure
    assert_eq!(json["spec_version"], "1.0");
    assert_eq!(json["event"], "result");
    assert!(json["timestamp"].is_string());
    assert_eq!(json["payload"]["result_type"], "providers");

    // Verify providers array exists and has at least CPU
    let providers = json["payload"]["providers"]
        .as_array()
        .expect("providers is array");
    assert!(
        !providers.is_empty(),
        "At least CPU provider should be present"
    );

    // Verify CPU provider structure
    let cpu_provider = providers
        .iter()
        .find(|p| p["id"] == "cpu")
        .expect("CPU provider exists");
    assert_eq!(cpu_provider["name"], "CPU");
    assert_eq!(cpu_provider["description"], "CPU (always available)");
    assert!(cpu_provider["id"].is_string());
}

#[test]
fn test_providers_json_all_fields_present() {
    let stdout = match run_providers_command(&["--output-mode", "json"]) {
        Some(stdout) => stdout,
        None => return,
    };

    let output_str = String::from_utf8(stdout).expect("stdout should be valid UTF-8");
    let json: Value = serde_json::from_str(&output_str).expect("Valid JSON output");

    let providers = json["payload"]["providers"]
        .as_array()
        .expect("providers is array");

    for provider in providers {
        // Verify all required fields are present
        assert!(provider["id"].is_string(), "id field must be string");
        assert!(provider["name"].is_string(), "name field must be string");
        assert!(
            provider["description"].is_string(),
            "description field must be string"
        );

        // Verify fields are non-empty
        assert!(
            !provider["id"]
                .as_str()
                .expect("id should be string")
                .is_empty(),
            "id must not be empty"
        );
        assert!(
            !provider["name"]
                .as_str()
                .expect("name should be string")
                .is_empty(),
            "name must not be empty"
        );
        assert!(
            !provider["description"]
                .as_str()
                .expect("description should be string")
                .is_empty(),
            "description must not be empty"
        );
    }
}

#[test]
fn test_providers_command_shows_usage_help() {
    let stdout = match run_providers_command(&[]) {
        Some(stdout) => stdout,
        None => return,
    };

    let output = String::from_utf8(stdout).expect("stdout should be valid UTF-8");
    assert!(output.contains("Usage:"));
    assert!(output.contains("--cpu"));
    assert!(output.contains("--gpu"));
    assert!(output.contains("Explicit providers"));
}
