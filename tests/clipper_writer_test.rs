//! Tests for WAV file writer.
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

use birda::clipper::WavWriter;
use tempfile::TempDir;

/// The species, confidence and time range used by the atomicity tests.
///
/// Bound once because the filename is derived from all four, and the tests below
/// depend on two calls resolving to the *same* path: a drifting confidence or
/// time range would silently turn a replacement into two separate files and the
/// assertions would pass without testing anything.
const CLIP: (&str, f32, f64, f64) = ("Test Species", 0.85, 0.0, 1.0);

/// Number of samples in the clip written first.
const FIRST_SAMPLES: usize = 48_000;

/// Number of samples in the clip written second, distinguishable from the first.
const SECOND_SAMPLES: usize = 96_000;

/// The two clips must differ in length, or the in-place test proves nothing: an
/// in-place writer would satisfy every assertion in it. Enforced rather than left
/// as prose, since the sibling precondition (both writes resolving to one path) is
/// asserted at runtime and this one deserves the same.
const _: () = assert!(FIRST_SAMPLES != SECOND_SAMPLES);

/// Sample rate the test clips are written at.
const SAMPLE_RATE: u32 = 48_000;

/// Write a silent clip of `sample_count` samples through the public API.
fn write_silent_clip(writer: &WavWriter, sample_count: usize) -> std::path::PathBuf {
    let samples = vec![0.0_f32; sample_count];
    let (species, confidence, start, end) = CLIP;
    writer
        .write_clip(&samples, SAMPLE_RATE, species, confidence, start, end)
        .unwrap()
}

/// Number of samples the WAV file at `path` actually holds.
///
/// Returned as `usize` to match the sample-count constants, so the assertions
/// read without a cast between the two.
fn samples_in(path: &std::path::Path) -> usize {
    usize::try_from(hound::WavReader::open(path).unwrap().len()).unwrap()
}

#[test]
fn test_write_clip_creates_species_directory() {
    let temp_dir = TempDir::new().unwrap();
    let writer = WavWriter::new(temp_dir.path().to_path_buf());

    // Simple sine wave samples
    let samples: Vec<f32> = (0..48_000u16)
        .map(|i| (f32::from(i) * 0.01).sin())
        .collect();

    let path = writer
        .write_clip(&samples, 48000, "Parus major", 0.85, 10.5, 11.5)
        .unwrap();

    assert!(path.exists());
    assert!(path.parent().unwrap().ends_with("Parus major"));
}

#[test]
fn test_write_clip_filename_format() {
    let temp_dir = TempDir::new().unwrap();
    let writer = WavWriter::new(temp_dir.path().to_path_buf());

    let samples: Vec<f32> = vec![0.0; 48000];

    let path = writer
        .write_clip(&samples, 48000, "Cyanistes caeruleus", 0.9234, 5.0, 8.0)
        .unwrap();

    let filename = path.file_name().unwrap().to_str().unwrap();
    // Format: species_confidence_start-end.wav
    assert!(filename.starts_with("Cyanistes caeruleus_92p_"));
    assert_eq!(path.extension(), Some(std::ffi::OsStr::new("wav")));
}

#[test]
fn test_write_clip_sanitizes_species_name() {
    let temp_dir = TempDir::new().unwrap();
    let writer = WavWriter::new(temp_dir.path().to_path_buf());

    let samples: Vec<f32> = vec![0.0; 48000];

    // Species name with characters that need sanitization
    let path = writer
        .write_clip(
            &samples,
            48000,
            "Species/with:special*chars",
            0.80,
            0.0,
            1.0,
        )
        .unwrap();

    assert!(path.exists());
    // Directory name should be sanitized
    let dir_name = path
        .parent()
        .unwrap()
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    assert!(!dir_name.contains('/'));
    assert!(!dir_name.contains(':'));
    assert!(!dir_name.contains('*'));
}

#[test]
fn test_written_wav_is_valid() {
    let temp_dir = TempDir::new().unwrap();
    let writer = WavWriter::new(temp_dir.path().to_path_buf());

    let samples: Vec<f32> = (0..48_000u16)
        .map(|i| (f32::from(i) * 0.01).sin())
        .collect();

    let path = writer
        .write_clip(&samples, 48000, "Test Species", 0.85, 0.0, 1.0)
        .unwrap();

    // Verify we can read the WAV back
    let reader = hound::WavReader::open(&path).unwrap();
    let spec = reader.spec();

    assert_eq!(spec.sample_rate, 48000);
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.bits_per_sample, 16);
}

#[test]
#[cfg(unix)]
fn test_write_clip_does_not_write_the_clip_in_place() {
    // hound writes a placeholder RIFF header first and patches the RIFF and
    // `data` lengths only in `finalize()`, so writing at the final path leaves a
    // header claiming zero data bytes if anything interrupts the sample loop:
    // structurally valid, silently empty, and indistinguishable from a
    // legitimately empty clip.
    //
    // A hardlink is a second name for the same inode. Writing in place would
    // show the new clip through the link; writing to a temporary and renaming
    // gives the clip path a *different* inode, leaving the old one intact behind
    // the link. Reading the original sample count back through the link is
    // therefore proof the clip was published by rename.
    //
    // Named for what it proves rather than for atomicity, because it proves
    // less: it shows the path acquired a new inode, not that a reader had an
    // uninterrupted view throughout. Simulating the actual crash would need a
    // process the test could kill mid-loop.
    let temp_dir = TempDir::new().unwrap();
    let writer = WavWriter::new(temp_dir.path().to_path_buf());

    let path = write_silent_clip(&writer, FIRST_SAMPLES);
    let link = temp_dir.path().join("first.wav");
    std::fs::hard_link(&path, &link).unwrap();

    let again = write_silent_clip(&writer, SECOND_SAMPLES);
    assert_eq!(
        again, path,
        "both writes must resolve to one path, or this test proves nothing"
    );

    assert_eq!(
        samples_in(&link),
        FIRST_SAMPLES,
        "the first clip must survive behind its own name; seeing the second \
         clip's length here means the clip was written in place"
    );
    assert_eq!(
        samples_in(&path),
        SECOND_SAMPLES,
        "the clip path must carry the new clip"
    );
}

#[test]
fn test_write_clip_leaves_no_temporary_behind() {
    // The temporary is created in the species directory, because rename is only
    // atomic within a filesystem and $TMPDIR is routinely a different one. That
    // puts it where the user is about to go looking for their clips, so a
    // successful extraction must clean up after itself. Clip extraction writes
    // many files into one directory, so a leak here scales with the run.
    let temp_dir = TempDir::new().unwrap();
    let writer = WavWriter::new(temp_dir.path().to_path_buf());

    let path = write_silent_clip(&writer, FIRST_SAMPLES);

    let expected = path.file_name().unwrap();
    let strays: Vec<String> = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .filter(|name| name != expected)
        .map(|name| name.to_string_lossy().into_owned())
        .collect();
    assert!(
        strays.is_empty(),
        "a successful extraction must leave only the clip, found: {strays:?}"
    );
}

#[test]
#[cfg(unix)]
fn test_a_clip_is_not_narrowed_to_its_owner() {
    // Publishing by rename hands the clip path the temporary's inode, and a
    // temporary is created owner-only. Without an explicit mode policy every
    // extracted clip would come out 0600 instead of whatever the umask allows,
    // which breaks a clip directory served by a web server or read by another
    // account. The clip is a file the user asked to be produced, so it has to
    // come out exactly as it did when it was written in place.
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = TempDir::new().unwrap();
    let writer = WavWriter::new(temp_dir.path().to_path_buf());

    let path = write_silent_clip(&writer, FIRST_SAMPLES);

    // Compared against a file `File::create` made under the same umask rather
    // than against a literal 0o644, because the umask this test runs under is
    // not knowable from here: a literal would fail for anyone whose umask is
    // 0o077 or 0o002.
    let reference = temp_dir.path().join("reference");
    drop(std::fs::File::create(&reference).unwrap());

    let mode_of = |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(
        mode_of(&path),
        mode_of(&reference),
        "a clip must keep the mode File::create would have given it"
    );

    // What this cannot see, said out loud rather than left as a silent pass: under
    // a umask that masks the group and world bits away, the two mode policies both
    // yield 0o600, so the assertion above holds whichever one production passes.
    if mode_of(&reference) & 0o066 == 0 {
        eprintln!(
            "skipped the policy distinction: this umask masks both policies to {:o}, \
             so a wrong one would not be detected here",
            mode_of(&reference)
        );
    }
}
