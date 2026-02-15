//! Integration tests for providers command.

use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use serde_json::Value;

#[test]
fn test_providers_command_human_readable() {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.arg("providers");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Available execution providers:"))
        .stdout(predicate::str::contains("CPU"));
}

#[test]
fn test_providers_command_json_output() {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.arg("providers").arg("--output-mode").arg("json");

    let output = cmd.assert().success();
    let stdout = String::from_utf8(output.get_output().stdout.clone()).unwrap();

    // Parse JSON
    let json: Value = serde_json::from_str(&stdout).expect("Valid JSON output");

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
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.arg("providers").arg("--output-mode").arg("json");

    let output = cmd.assert().success();
    let stdout = String::from_utf8(output.get_output().stdout.clone()).unwrap();
    let json: Value = serde_json::from_str(&stdout).unwrap();

    let providers = json["payload"]["providers"].as_array().unwrap();

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
            !provider["id"].as_str().unwrap().is_empty(),
            "id must not be empty"
        );
        assert!(
            !provider["name"].as_str().unwrap().is_empty(),
            "name must not be empty"
        );
        assert!(
            !provider["description"].as_str().unwrap().is_empty(),
            "description must not be empty"
        );
    }
}
