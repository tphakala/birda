//! Integration test for clip extraction, including the non-finite and
//! out-of-range time inputs of #310, which used to abort the process on an
//! allocation or exit 0 after writing a clip nobody asked for.
//!
//! These drive the real binary, because the hazard is in the seam between the
//! CLI parsers, the range guard and the extractor's allocation, and each layer
//! looks correct on its own.
// Integration test crate. `unwrap`, `expect` and `panic` are how a test reports
// failure, not unhandled error paths, so rewriting them into propagated errors
// would only hide which assertion fired. Every exact float assertion in these
// tests is on a passed-through value (a literal parsed from a file, a
// coordinate round-tripped through JSON, a clip boundary clamped to a whole
// number) rather than a computed one, so exact equality is the assertion the
// test wants. The crate-level deny still governs everything birda ships.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use std::io::Write;
use std::path::PathBuf;
use std::process::Output;
use std::time::Duration;

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use hound::{SampleFormat, WavSpec, WavWriter};
use tempfile::TempDir;

/// Bound every spawn, following `tests/config_validation.rs`.
///
/// `test_clip_survives_a_range_far_beyond_the_file` is why this matters here:
/// it asks for a range of 1e12 seconds, and the regression it guards against
/// is the extractor trying to service that literally. Unbounded, such a
/// regression hangs the suite instead of failing it.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// A `birda` invocation insulated from the developer's shell.
///
/// Two variables would otherwise decide what these assertions see:
///
/// - `--output-mode` is `global = true` and bound to `BIRDA_OUTPUT_MODE`, so
///   it applies to `clip` as well as `analyze`, and it reshapes both streams.
/// - `init_logging` builds its filter with `EnvFilter::try_from_default_env`,
///   so an inherited `RUST_LOG` overrides the built-in level. The skipped-row
///   assertions below match on `warn!` output, and `RUST_LOG=error` makes
///   them match nothing.
///
/// Either one exported in a shell profile changes the result rather than
/// failing outright, which is the harder failure to notice.
fn birda_cmd() -> Command {
    let mut cmd = cargo_bin_cmd!("birda");
    cmd.timeout(COMMAND_TIMEOUT)
        .env_remove("BIRDA_OUTPUT_MODE")
        .env_remove("RUST_LOG");
    cmd
}

/// Create a dummy WAV file with silence for testing.
fn create_test_wav(path: &std::path::Path, duration_secs: u32, sample_rate: u32) {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec).unwrap();

    // Write silence (zeros) for the specified duration
    let num_samples = sample_rate * duration_secs;
    for _ in 0..num_samples {
        writer.write_sample(0i16).unwrap();
    }

    writer.finalize().unwrap();
}

#[test]
fn test_clip_command_extracts_clips() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test WAV file (5 seconds of silence at 48kHz)
    let wav_path = temp_dir.path().join("test.wav");
    create_test_wav(&wav_path, 5, 48000);

    // Create a detection CSV pointing to the WAV
    let csv_path = temp_dir.path().join("test.wav.BirdNET.results.csv");
    let mut csv_file = std::fs::File::create(&csv_path).unwrap();
    writeln!(
        csv_file,
        "Start (s),End (s),Scientific name,Common name,Confidence"
    )
    .unwrap();
    writeln!(csv_file, "0.0,3.0,Parus major,Great Tit,0.85").unwrap();

    let output_dir = temp_dir.path().join("clips");

    let output = birda_cmd()
        .args([
            "clip",
            csv_path.to_str().unwrap(),
            "--output",
            output_dir.to_str().unwrap(),
            "--pre",
            "0",
            "--post",
            "0",
        ])
        .output()
        .expect("failed to execute birda clip");

    // Check that the command succeeded
    assert!(
        output.status.success(),
        "clip command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check that the output directory was created with species subdirectory
    let species_dir = output_dir.join("Parus major");
    assert!(species_dir.exists(), "Species directory should exist");

    // Check that at least one clip was extracted
    let clips: Vec<_> = std::fs::read_dir(&species_dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "wav"))
        .collect();

    assert!(!clips.is_empty(), "Should have extracted at least one clip");
}

#[test]
fn test_clip_help_displays() {
    let output = birda_cmd()
        .args(["clip", "--help"])
        .output()
        .expect("failed to execute birda clip --help");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(stdout.contains("Extract audio clips"));
    assert!(stdout.contains("--confidence"));
    assert!(stdout.contains("--pre"));
    assert!(stdout.contains("--post"));
    assert!(stdout.contains("--base-dir"));
}

/// CSV round-trip test: verify clipper can parse CSV files generated by birda's output writers.
/// This ensures the integration between analyze output and clip input is seamless.
#[test]
fn test_csv_roundtrip_parse() {
    use birda::clipper::parse_detection_file;
    use birda::output::{CsvWriter, Detection, DetectionMetadata, OutputWriter};

    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("roundtrip.BirdNET.results.csv");

    // Create detections using birda's output types
    let detections = vec![
        Detection {
            file_path: PathBuf::from("test.wav"),
            start_time: 0.0,
            end_time: 3.0,
            scientific_name: "Parus major".to_string(),
            common_name: "Great Tit".to_string(),
            confidence: 0.8542,
            metadata: DetectionMetadata::default(),
        },
        Detection {
            file_path: PathBuf::from("test.wav"),
            start_time: 15.0,
            end_time: 18.0,
            scientific_name: "Cyanistes caeruleus".to_string(),
            common_name: "Eurasian Blue Tit".to_string(),
            confidence: 0.7123,
            metadata: DetectionMetadata::default(),
        },
    ];

    // Write using birda's CSV writer (no extra columns, no BOM for simplicity)
    let mut writer = CsvWriter::new(&csv_path, vec![], false).unwrap();
    writer.write_header().unwrap();
    for detection in &detections {
        writer.write_detection(detection).unwrap();
    }
    writer.finalize().unwrap();

    // Parse back using clipper's parser
    let parsed = parse_detection_file(&csv_path).unwrap();

    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].scientific_name, "Parus major");
    assert_eq!(parsed[1].scientific_name, "Cyanistes caeruleus");
    assert!((parsed[0].confidence - 0.8542).abs() < 0.001);
}

/// Run `birda clip` in direct-extraction mode and hand back the finished process.
fn run_direct_clip(wav: &std::path::Path, out: &std::path::Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "clip",
        "--audio",
        wav.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);

    birda_cmd()
        .args(&args)
        .output()
        .expect("failed to execute birda clip")
}

/// A run rejected by clap's `value_parser` must fail cleanly: non-zero exit, a
/// diagnostic naming the offending option, and no panic or allocation abort.
///
/// The `error: invalid value` assertion is what pins each test to the layer it
/// names. Every guard added for #310 sits at a different layer and they all
/// phrase the rejection similarly, so a test matching only on the message text
/// stays green when the parser it is meant to cover is deleted and a
/// downstream guard catches the value instead. That is not hypothetical: it
/// was true of the padding test until this assertion was added. Only the clap
/// parser produces this prefix.
fn assert_rejected_by_clap(output: &Output, expected: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "expected a non-zero exit, got success. stderr: {stderr}"
    );
    assert!(
        stderr.contains("error: invalid value"),
        "the value should have been rejected by the argument parser, not \
         further downstream: {stderr}"
    );
    assert!(
        stderr.contains(expected),
        "stderr did not mention {expected:?}: {stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "the run panicked instead of reporting an error: {stderr}"
    );
    assert!(
        !stderr.contains("memory allocation of"),
        "the run aborted on an allocation instead of reporting an error: {stderr}"
    );
}

/// `--end inf` used to abort the process with `capacity overflow`: the cast
/// from seconds to samples saturates rather than trapping, so the extractor
/// asked `Vec::with_capacity` for `u64::MAX` samples.
#[test]
fn test_clip_rejects_infinite_end() {
    let temp_dir = TempDir::new().unwrap();
    let wav_path = temp_dir.path().join("tone.wav");
    create_test_wav(&wav_path, 5, 48000);

    let output = run_direct_clip(
        &wav_path,
        &temp_dir.path().join("clips"),
        &["--start", "0", "--end", "inf"],
    );

    assert_rejected_by_clap(&output, "finite non-negative");
}

/// `--start nan` used to exit 0 after writing `detection_NaN-5.wav` over a
/// range nobody asked for, because every comparison against NaN is false.
#[test]
fn test_clip_rejects_nan_start() {
    let temp_dir = TempDir::new().unwrap();
    let wav_path = temp_dir.path().join("tone.wav");
    create_test_wav(&wav_path, 5, 48000);
    let output_dir = temp_dir.path().join("clips");

    let output = run_direct_clip(&wav_path, &output_dir, &["--start", "nan", "--end", "5"]);

    assert_rejected_by_clap(&output, "finite non-negative");
    assert!(
        !output_dir.exists() || std::fs::read_dir(&output_dir).unwrap().next().is_none(),
        "a rejected run must not write a clip"
    );
}

/// `--pre nan` was swallowed by `(start - pre).max(0.0)`, which returns the
/// other operand for a NaN receiver, so the padding vanished without a word.
#[test]
fn test_clip_rejects_nan_padding() {
    let temp_dir = TempDir::new().unwrap();
    let wav_path = temp_dir.path().join("tone.wav");
    create_test_wav(&wav_path, 5, 48000);

    for flag in ["--pre", "--post"] {
        let output = run_direct_clip(
            &wav_path,
            &temp_dir.path().join("clips"),
            &["--start", "1", "--end", "3", flag, "nan"],
        );
        assert_rejected_by_clap(&output, "finite non-negative");
    }
}

/// The case the finiteness checks do not cover: `--end 1e12` is finite,
/// non-negative and passes every range test, yet at 48 kHz it asks for 4.8e16
/// samples, and the process used to die with
/// `memory allocation of 192000000000960000 bytes failed`. The reservation is
/// now capped, so the run succeeds and returns what the file actually holds.
#[test]
fn test_clip_survives_a_range_far_beyond_the_file() {
    let temp_dir = TempDir::new().unwrap();
    let wav_path = temp_dir.path().join("tone.wav");
    create_test_wav(&wav_path, 5, 48000);
    let output_dir = temp_dir.path().join("clips");

    let output = run_direct_clip(
        &wav_path,
        &output_dir,
        &["--start", "0", "--end", "1e12", "--pre", "0", "--post", "0"],
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "a huge but finite range must not abort the process: {stderr}"
    );
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(!stderr.contains("memory allocation of"), "{stderr}");

    let clips: Vec<_> = walk_wav_files(&output_dir);
    assert_eq!(
        clips.len(),
        1,
        "expected exactly one clip in {output_dir:?}"
    );

    // The clip is bounded by the file, not by the requested end.
    let reader = hound::WavReader::open(&clips[0]).unwrap();
    assert_eq!(
        reader.duration(),
        5 * 48000,
        "the clip should hold the whole 5s file"
    );
}

/// An infinite end in a detection file reached the same allocation as the CLI
/// route, because `end <= start` is false for infinity too.
#[test]
fn test_clip_skips_non_finite_rows_in_a_detection_file() {
    // Every fixture is three rows with the bad one in the middle, because a
    // single-row fixture cannot see the failure this test exists for: a hard
    // error in the parser discards the whole file, so the two good rows
    // disappear with the bad one and the run still exits 0. `line 3` is what
    // pins each case to the parser; the extractor rejects a non-finite range
    // too, and both layers phrase it the same way.
    for (bad_row, expected) in [
        ("5.0,inf,Parus major,Great Tit,0.85", "line 3"),
        ("nan,8.0,Parus major,Great Tit,0.85", "line 3"),
        ("5.0,8.0,Parus major,Great Tit,nan", "line 3"),
        // An ordinary decimal that overflows f32 to infinity on the way in.
        ("5.0,8.0,Parus major,Great Tit,1e40", "line 3"),
    ] {
        let temp_dir = TempDir::new().unwrap();
        let wav_path = temp_dir.path().join("rec.wav");
        create_test_wav(&wav_path, 20, 48000);

        let csv_path = temp_dir.path().join("rec.wav.BirdNET.results.csv");
        let mut csv_file = std::fs::File::create(&csv_path).unwrap();
        writeln!(
            csv_file,
            "Start (s),End (s),Scientific name,Common name,Confidence"
        )
        .unwrap();
        writeln!(csv_file, "0.0,3.0,Turdus merula,Eurasian Blackbird,0.85").unwrap();
        writeln!(csv_file, "{bad_row}").unwrap();
        writeln!(csv_file, "12.0,15.0,Erithacus rubecula,European Robin,0.91").unwrap();
        drop(csv_file);

        let output_dir = temp_dir.path().join("clips");
        let output = birda_cmd()
            .args([
                "clip",
                csv_path.to_str().unwrap(),
                "--output",
                output_dir.to_str().unwrap(),
                "--pre",
                "0",
                "--post",
                "0",
            ])
            .output()
            .expect("failed to execute birda clip");

        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panicked"), "{bad_row}: {stderr}");
        assert!(
            !stderr.contains("memory allocation of"),
            "{bad_row}: {stderr}"
        );
        assert!(
            stderr.contains(expected),
            "{bad_row}: stderr did not name the offending line: {stderr}"
        );
        assert!(
            stderr.contains("skipping detection"),
            "{bad_row}: the skip should be reported, not silent: {stderr}"
        );

        // Exit 0 is deliberate, not an oversight. `clip` takes many detection
        // files and reports per-file problems as warnings, so one skipped row
        // is not a failed run. What matters is that the good rows survived.
        assert!(
            output.status.success(),
            "{bad_row}: skipping a row should not fail the run: {stderr}"
        );
        assert_eq!(
            walk_wav_files(&output_dir).len(),
            2,
            "{bad_row}: the two valid detections should still produce clips"
        );
    }
}

/// Collect every `.wav` anywhere under `dir`, at any depth.
fn walk_wav_files(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(walk_wav_files(&path));
        } else if path.extension().is_some_and(|e| e == "wav") {
            found.push(path);
        }
    }

    found
}

/// The reservation is a sizing hint, and this is what says so.
///
/// A mutation adding `samples.truncate(expected_samples)` to the extractor,
/// turning the hint into a hard bound, passed the whole suite: every other
/// fixture is short enough to sit under the cap, so nothing ever exercised the
/// buffer growing past it. A clip silently cut to the cap length would have
/// shipped green.
///
/// The fixture is deliberately at 8 kHz. The cap scales with the file's own
/// sample rate, so a low rate keeps "longer than the cap" cheap to generate
/// while still crossing it, and the length is derived from the constant rather
/// than hardcoded so raising the cap cannot quietly retire the test.
#[test]
fn test_a_clip_longer_than_the_preallocation_cap_is_complete() {
    use birda::constants::clipper::MAX_CLIP_PREALLOC_SECS;

    const RATE: u32 = 8_000;
    let over_cap_secs = u32::try_from(MAX_CLIP_PREALLOC_SECS).unwrap() + 5;

    let temp_dir = TempDir::new().unwrap();
    let wav_path = temp_dir.path().join("long.wav");
    create_test_wav(&wav_path, over_cap_secs, RATE);
    let output_dir = temp_dir.path().join("clips");

    let output = run_direct_clip(
        &wav_path,
        &output_dir,
        &[
            "--start",
            "0",
            "--end",
            &over_cap_secs.to_string(),
            "--pre",
            "0",
            "--post",
            "0",
        ],
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");

    let clips = walk_wav_files(&output_dir);
    assert_eq!(clips.len(), 1, "expected one clip in {output_dir:?}");

    let reader = hound::WavReader::open(&clips[0]).unwrap();
    assert_eq!(
        reader.duration(),
        over_cap_secs * RATE,
        "the clip was cut short, so the reservation is bounding the output \
         rather than sizing it"
    );
}

/// The per-row warnings are capped and then summarised, and this is what says
/// so. Deleting both the cap and the summary left the whole suite green, since
/// every other fixture has at most one bad row.
#[test]
fn test_skipped_row_warnings_are_capped_and_then_summarised() {
    use birda::constants::clipper::MAX_SKIPPED_ROW_WARNINGS;

    let bad_rows = MAX_SKIPPED_ROW_WARNINGS + 5;

    let temp_dir = TempDir::new().unwrap();
    let wav_path = temp_dir.path().join("rec.wav");
    create_test_wav(&wav_path, 20, 48000);

    let csv_path = temp_dir.path().join("rec.wav.BirdNET.results.csv");
    let mut csv_file = std::fs::File::create(&csv_path).unwrap();
    writeln!(
        csv_file,
        "Start (s),End (s),Scientific name,Common name,Confidence"
    )
    .unwrap();
    for i in 0..bad_rows {
        writeln!(csv_file, "{i}.0,inf,Parus major,Great Tit,0.85").unwrap();
    }
    writeln!(csv_file, "0.0,3.0,Turdus merula,Eurasian Blackbird,0.85").unwrap();
    drop(csv_file);

    let output_dir = temp_dir.path().join("clips");
    let output = birda_cmd()
        .args([
            "clip",
            csv_path.to_str().unwrap(),
            "--output",
            output_dir.to_str().unwrap(),
            "--pre",
            "0",
            "--post",
            "0",
        ])
        .output()
        .expect("failed to execute birda clip");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let per_row = stderr.matches("skipping detection").count();
    assert_eq!(
        per_row, MAX_SKIPPED_ROW_WARNINGS,
        "expected exactly {MAX_SKIPPED_ROW_WARNINGS} per-row warnings for \
         {bad_rows} bad rows: {stderr}"
    );
    assert!(
        stderr.contains(&format!("skipped {bad_rows} malformed detections")),
        "the suppressed rows should still be counted: {stderr}"
    );

    // The cap bounds the diagnostics, never the skipping.
    assert_eq!(walk_wav_files(&output_dir).len(), 1);
}
