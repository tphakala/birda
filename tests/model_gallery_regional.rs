//! Regional and variant surface of the model gallery.
//!
//! These drive the real binary against the bundled registry, so they exercise
//! clap wiring, the registry contents and the output shape together. None of
//! them download anything: every assertion is about listing, or about rejecting
//! bad input before a single byte moves.

use assert_cmd::Command;
use predicates::prelude::*;

fn birda() -> Command {
    Command::cargo_bin("birda").expect("binary builds")
}

#[test]
fn test_models_regions_lists_tiles_grouped_by_continent() {
    birda()
        .args(["models", "regions", "birdnet-v30"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Europe:"))
        .stdout(predicate::str::contains("nordic"))
        .stdout(predicate::str::contains("Asia:"));
}

#[test]
fn test_models_regions_reports_species_counts_and_sizes() {
    // A user picking a region is choosing between memory footprints, so the
    // listing has to carry both numbers.
    birda()
        .args(["models", "regions", "birdnet-v30"])
        .assert()
        .success()
        .stdout(predicate::str::contains("species"))
        .stdout(predicate::str::contains("MB"));
}

#[test]
fn test_models_regions_rejects_a_model_without_regions() {
    birda()
        .args(["models", "regions", "birdnet-v24"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("no regional variants"));
}

#[test]
fn test_models_regions_rejects_an_unknown_model() {
    birda()
        .args(["models", "regions", "not-a-model"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not-a-model"));
}

#[test]
fn test_models_install_rejects_an_unknown_region_before_downloading() {
    birda()
        .args([
            "models",
            "install",
            "birdnet-v30",
            "--region",
            "atlantis",
            "--yes",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("atlantis"))
        .stderr(predicate::str::contains("nordic"));
}

#[test]
fn test_models_install_rejects_an_unknown_variant_before_downloading() {
    birda()
        .args([
            "models",
            "install",
            "birdnet-v30",
            "--variant",
            "int4",
            "--yes",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("int4"))
        .stderr(predicate::str::contains("fp32"));
}

#[test]
fn test_models_install_rejects_a_region_on_a_model_that_has_none() {
    // Silently ignoring it would install the global model while the user
    // believed they had asked for a regional one.
    birda()
        .args([
            "models",
            "install",
            "birdnet-v24",
            "--region",
            "nordic",
            "--yes",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("no regional variants"));
}

#[test]
fn test_models_info_shows_the_exact_upstream_version_and_build() {
    // "3.0" would be a lie: the published weights are preview3.1, and GA will
    // be different weights under a version string that must not collide.
    birda()
        .args(["models", "info", "birdnet-v30"])
        .assert()
        .success()
        .stdout(predicate::str::contains("3.0-preview3.1"))
        .stdout(predicate::str::contains("build 1"));
}

#[test]
fn test_models_info_reports_the_variants_and_the_regional_count() {
    birda()
        .args(["models", "info", "birdnet-v30"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Variants:"))
        .stdout(predicate::str::contains("Regional models: 39"));
}

#[test]
fn test_models_info_still_works_for_a_legacy_entry() {
    birda()
        .args(["models", "info", "birdnet-v24"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Files:"))
        .stdout(predicate::str::contains("languages available"));
}

#[test]
fn test_models_languages_explains_itself_on_a_variant_entry() {
    // Variant families ship one English labels file per region, so there is no
    // language list. Saying so beats printing an empty one.
    birda()
        .args(["models", "info", "birdnet-v30", "--languages"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("language variants"));
}

#[test]
fn test_models_list_available_shows_every_family() {
    birda()
        .args(["models", "list-available"])
        .assert()
        .success()
        .stdout(predicate::str::contains("birdnet-v24"))
        .stdout(predicate::str::contains("birdnet-v30"))
        .stdout(predicate::str::contains("perch-v2"));
}

#[test]
fn test_perch_publishes_regions_too() {
    birda()
        .args(["models", "regions", "perch-v2"])
        .assert()
        .success()
        .stdout(predicate::str::contains("nordic"));
}
