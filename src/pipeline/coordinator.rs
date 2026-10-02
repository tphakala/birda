//! Pipeline coordination for file processing.

use crate::config::OutputFormat;
use crate::constants::{FALLBACK_OUTPUT_NAME, output_extensions};
use crate::error::{Error, Result};
use crate::locking::FileLock;
use crate::output::OutputFiles;
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};
use tracing::{info, warn};

/// Options for processing a single file.
#[derive(Debug, Clone)]
pub struct ProcessOptions {
    /// Output directory (None = same as input).
    pub output_dir: Option<PathBuf>,
    /// Output formats to generate.
    pub formats: Vec<OutputFormat>,
    /// Force reprocessing even if output exists.
    pub force: bool,
    /// Minimum confidence threshold.
    pub min_confidence: f32,
    /// Segment overlap in seconds.
    pub overlap: f32,
    /// Batch size for inference.
    pub batch_size: usize,
    /// Model name.
    pub model_name: String,
}

/// Result of checking whether a file should be processed.
#[derive(Debug)]
pub enum ProcessCheck {
    /// File should be processed.
    Process,
    /// Skip - output already exists.
    SkipExists,
    /// Skip - file is locked by another process.
    SkipLocked,
}

/// Determine the output directory for a file.
pub fn output_dir_for(input: &Path, explicit_output_dir: Option<&Path>) -> PathBuf {
    explicit_output_dir.map_or_else(
        || {
            input
                .parent()
                .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
        },
        Path::to_path_buf,
    )
}

/// Sanitize a filename to prevent path traversal attacks.
///
/// Replaces path separators with underscores.
fn sanitize_filename(filename: &str) -> String {
    filename.replace(['/', '\\'], "_")
}

/// Where the output files for one input go, and what they are called.
///
/// Built only by [`plan_output_targets`], so every name in a run has been
/// checked against every other input in that run. Deriving a name from a single
/// input's stem is what let two inputs write the same file (#414).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputTarget {
    /// Directory the output files are written to.
    dir: PathBuf,
    /// Directory no output path may leave (the `-o` directory, or `dir`).
    root: PathBuf,
    /// Sanitized file name shared by every format, without a format suffix.
    base: String,
}

impl OutputTarget {
    /// Directory the output files are written to.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// File name shared by every format, without the format suffix.
    #[must_use]
    pub fn base(&self) -> &str {
        &self.base
    }

    /// Output file path for a given format.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PathTraversal`] if the path would escape the output
    /// directory.
    pub fn path_for(&self, format: OutputFormat) -> Result<PathBuf> {
        let extension = match format {
            OutputFormat::Csv => output_extensions::CSV,
            OutputFormat::Raven => output_extensions::RAVEN,
            OutputFormat::Audacity => output_extensions::AUDACITY,
            OutputFormat::Kaleidoscope => output_extensions::KALEIDOSCOPE,
            OutputFormat::Json => output_extensions::JSON,
            OutputFormat::Parquet => output_extensions::PARQUET,
        };

        let output_path = self.dir.join(format!("{}{extension}", self.base));

        // Runtime verification: output path must stay within the output directory
        if !output_path.starts_with(&self.root) {
            return Err(Error::PathTraversal {
                output_path,
                output_dir: self.root.clone(),
            });
        }

        Ok(output_path)
    }

    /// Output file paths for every requested format.
    ///
    /// # Errors
    ///
    /// Returns the first [`Error::PathTraversal`] from [`Self::path_for`].
    pub fn paths_for(&self, formats: &[OutputFormat]) -> Result<OutputFiles> {
        formats
            .iter()
            .map(|fmt| self.path_for(*fmt).map(|p| (*fmt, p)))
            .collect()
    }
}

/// Working state for one input while output names are being planned.
struct Candidate {
    /// Sanitized file stem, the name an uncontested file keeps.
    stem: String,
    /// Sanitized full file name (stem and extension), the name a contested file
    /// takes.
    full: String,
    /// Output directory as `output_dir_for` gives it, spelled as the caller did.
    plain_dir: PathBuf,
    /// Canonical form of `plain_dir`, for comparing directories.
    canonical_dir: PathBuf,
    /// Canonical parent folder of the input.
    canonical_parent: PathBuf,
    /// Whether the full file name is used instead of the stem.
    qualified: bool,
    /// Subfolders under `-o` that keep inputs from different folders apart.
    mirror: Vec<OsString>,
}

impl Candidate {
    fn base(&self) -> &str {
        if self.qualified {
            &self.full
        } else {
            &self.stem
        }
    }

    /// Identity of the file this candidate would write: the directory (folder
    /// names compared case-insensitively) and the lowercased name.
    fn key(&self) -> (PathBuf, String) {
        let mut dir = self.canonical_dir.clone();
        for part in &self.mirror {
            dir.push(part.to_string_lossy().to_lowercase());
        }
        (dir, self.base().to_lowercase())
    }

    fn target_dir(&self, explicit_output_dir: Option<&Path>) -> PathBuf {
        match explicit_output_dir {
            Some(out) if !self.mirror.is_empty() => self
                .mirror
                .iter()
                .fold(out.to_path_buf(), |dir, part| dir.join(part)),
            _ => self.plain_dir.clone(),
        }
    }
}

/// Canonical form of a directory, falling back to the path as given when it
/// does not exist (yet). An empty path means the current directory.
fn canonical_dir(dir: &Path) -> PathBuf {
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf())
}

/// The folder names in a path, without the drive prefix and root, so paths on
/// different drives can be compared and joined under another directory.
fn normal_components(path: &Path) -> Vec<OsString> {
    path.components()
        .filter_map(|c| match c {
            Component::Normal(part) => Some(part.to_os_string()),
            _ => None,
        })
        .collect()
}

/// Group candidate indices by their output identity.
fn group_by_key(candidates: &[Candidate]) -> HashMap<(PathBuf, String), Vec<usize>> {
    let mut groups: HashMap<(PathBuf, String), Vec<usize>> = HashMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        groups.entry(candidate.key()).or_default().push(index);
    }
    groups
}

/// Plan the output name of every input in a run.
///
/// Returns one entry per input, in input order. An input that shares no name
/// with another keeps the name it always had: its file stem, in `-o` or next to
/// the input. Inputs whose names would clash (the same stem, compared without
/// regard to case, in one output directory) all switch to their full file name,
/// so `x.wav` and `x.flac` become `x.wav.BirdNET.json` and `x.flac.BirdNET.json`.
/// When `-o` is given and the clashing inputs sit in different folders, each
/// also gets its folder, relative to the closest folder the group shares, under
/// `-o`. The rule is symmetric, so the result does not depend on input order.
///
/// A name that is still shared after that (folders that differ only in case, or
/// the same folder on two drives) is an [`Error::OutputPathCollision`] for every
/// input involved.
#[must_use]
pub fn plan_output_targets(
    files: &[PathBuf],
    explicit_output_dir: Option<&Path>,
) -> Vec<Result<OutputTarget>> {
    let mut candidates: Vec<Candidate> = files
        .iter()
        .map(|file| {
            let plain_dir = output_dir_for(file, explicit_output_dir);
            let name = |part: Option<&std::ffi::OsStr>| {
                part.map_or_else(
                    || FALLBACK_OUTPUT_NAME.to_string(),
                    |s| sanitize_filename(&s.to_string_lossy()),
                )
            };
            Candidate {
                stem: name(file.file_stem()),
                full: name(file.file_name()),
                canonical_dir: canonical_dir(&plain_dir),
                canonical_parent: canonical_dir(file.parent().unwrap_or_else(|| Path::new(""))),
                plain_dir,
                qualified: false,
                mirror: Vec::new(),
            }
        })
        .collect();

    // Inputs that share a name: qualify them, and keep different source folders
    // apart under `-o`.
    for members in group_by_key(&candidates).values().filter(|m| m.len() > 1) {
        let parents: Vec<Vec<OsString>> = members
            .iter()
            .map(|&i| normal_components(&candidates[i].canonical_parent))
            .collect();
        let shared = common_prefix_len(&parents);
        for (&i, parent) in members.iter().zip(&parents) {
            candidates[i].qualified = true;
            if explicit_output_dir.is_some() {
                candidates[i].mirror = parent[shared..].to_vec();
            }
        }
    }

    // A name that stayed plain can still match a qualified one (`x.wav` next to
    // `x.wav.wav` and `x.flac`). Qualify it too, until nothing changes.
    loop {
        let mut changed = false;
        for members in group_by_key(&candidates).values().filter(|m| m.len() > 1) {
            for &i in members {
                if !candidates[i].qualified {
                    candidates[i].qualified = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    let groups = group_by_key(&candidates);
    candidates
        .iter()
        .map(|candidate| {
            let target = OutputTarget {
                dir: candidate.target_dir(explicit_output_dir),
                root: explicit_output_dir
                    .map_or_else(|| candidate.plain_dir.clone(), Path::to_path_buf),
                base: candidate.base().to_string(),
            };
            let clashing = &groups[&candidate.key()];
            if clashing.len() > 1 {
                return Err(Error::OutputPathCollision {
                    output: target.dir.join(&target.base),
                    inputs: clashing.iter().map(|&i| files[i].clone()).collect(),
                });
            }
            Ok(target)
        })
        .collect()
}

/// Length of the leading run of folder names shared by every list.
fn common_prefix_len(lists: &[Vec<OsString>]) -> usize {
    let Some(first) = lists.first() else { return 0 };
    let mut len = first.len();
    for list in &lists[1..] {
        len = len.min(first.iter().zip(list).take_while(|(a, b)| a == b).count());
    }
    len
}

/// Check if a file should be processed.
pub fn should_process(
    input: &Path,
    target: &OutputTarget,
    formats: &[OutputFormat],
    force: bool,
    stdout_mode: bool,
) -> ProcessCheck {
    // Check if locked
    if FileLock::is_locked(input, target.dir()) {
        return ProcessCheck::SkipLocked;
    }

    // Skip file existence check in stdout mode (no files written)
    if stdout_mode {
        return ProcessCheck::Process;
    }

    // Check if all outputs exist (unless force)
    //
    // The list has to be non-empty for the question to mean anything, since
    // `all` over an empty slice is vacuously true (#339).
    //
    // `config::validate` rejects an empty `defaults.formats`, so the analyze
    // path cannot arrive here with one. This guard is for the direct caller:
    // `should_process` is public and takes the slice, so it should not depend on
    // a rule enforced two layers up. Returning `Process` is the honest answer;
    // no output can be found to already exist when none was asked for. It is
    // not free, though: such a caller now decodes and runs inference over every
    // file and still writes nothing, where before it did nothing quickly. Both
    // are silent, which is why the config rule is the one that matters.
    if !force && !formats.is_empty() {
        let all_exist = formats.iter().all(|fmt| {
            target.path_for(*fmt).map_or_else(
                |e| {
                    warn!("Failed to generate output path: {}", e);
                    false
                },
                |p| p.exists(),
            )
        });
        if all_exist {
            return ProcessCheck::SkipExists;
        }
    }

    ProcessCheck::Process
}

/// Collect input files from paths (files and directories).
///
/// A file that several arguments reach (listed twice, or inside a listed
/// directory) is kept once, at its first position. Two entries for one file
/// would otherwise be planned as a name collision. Files are compared by their
/// resolved folder and file name, so a symlink to a file already collected is
/// still its own input.
pub fn collect_input_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for path in paths {
        if path.is_file() {
            if is_audio_file(path) {
                files.push(path.clone());
            }
        } else if path.is_dir() {
            collect_audio_files_recursive(path, &mut files)?;
        } else {
            warn!("Skipping non-existent path: {}", path.display());
        }
    }

    let mut seen = HashSet::new();
    files.retain(|file| {
        // Resolve the folder, not the file: two symlinks to one recording in
        // different folders are two inputs with two output locations.
        let identity = match (file.parent(), file.file_name()) {
            (Some(parent), Some(name)) => canonical_dir(parent).join(name),
            _ => file.clone(),
        };
        let first = seen.insert(identity);
        if !first {
            info!("Skipping duplicate input: {}", file.display());
        }
        first
    });

    Ok(files)
}

/// Recursively collect audio files from a directory.
fn collect_audio_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            collect_audio_files_recursive(&path, files)?;
        } else if is_audio_file(&path) {
            files.push(path);
        }
    }

    Ok(())
}

/// Check if a file is a supported audio format.
fn is_audio_file(path: &Path) -> bool {
    const AUDIO_EXTENSIONS: &[&str] = &["wav", "flac", "mp3", "m4a", "aac"];

    // Compare extension directly as OsStr to handle non-UTF-8 filenames
    path.extension().is_some_and(|ext| {
        AUDIO_EXTENSIONS
            .iter()
            .any(|&audio_ext| ext.eq_ignore_ascii_case(audio_ext))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create an empty file, and its folders, under `root`.
    fn touch(root: &Path, relative: &str) -> PathBuf {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "").unwrap();
        path
    }

    /// Plan a run that must have no unresolved collision.
    fn plan(files: &[PathBuf], out: Option<&Path>) -> Vec<OutputTarget> {
        plan_output_targets(files, out)
            .into_iter()
            .map(|t| t.unwrap())
            .collect()
    }

    /// The JSON output path of a target.
    fn json_path(target: &OutputTarget) -> PathBuf {
        target.path_for(OutputFormat::Json).unwrap()
    }

    #[test]
    fn test_should_process_does_not_skip_when_no_formats_are_requested() {
        // #339, at the layer that has the information. `all` over an empty
        // slice is vacuously true, so this returned `SkipExists` for a file
        // whose outputs did not exist and could not, and the caller logged
        // "Skipping (output exists)" naming a reason that was never true.
        //
        // The input deliberately does not exist: with no format requested there
        // is no output path to test, so nothing about the filesystem can make
        // the answer `SkipExists`. The tempdir is for a unique path, not for
        // the lock check, which is a bare `Path::exists` and answers false for
        // a path under a directory that is not there either.
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("recording.wav");
        let target = plan(std::slice::from_ref(&input), None).remove(0);

        let check = should_process(&input, &target, &[], false, false);

        assert!(
            matches!(check, ProcessCheck::Process),
            "an empty format list must not read as 'every output already exists', got {check:?}"
        );
    }

    #[test]
    fn test_should_process_still_skips_when_the_only_output_exists() {
        // The other half of the pair: the empty-slice guard must not stop the
        // existence check doing its job for a list that has something in it.
        // Without this, deleting the whole `if` block passes the test above.
        let dir = tempfile::tempdir().unwrap();
        let input = touch(dir.path(), "recording.wav");
        let target = plan(std::slice::from_ref(&input), None).remove(0);
        touch(dir.path(), "recording.BirdNET.results.csv");

        let check = should_process(&input, &target, &[OutputFormat::Csv], false, false);

        assert!(
            matches!(check, ProcessCheck::SkipExists),
            "an existing output must still be skipped, got {check:?}"
        );
    }

    #[test]
    fn test_should_process_skips_each_planned_output_on_rerun() {
        // Same names on a second run: each member of a colliding pair finds its
        // own output, so the rerun skips both instead of redoing one of them.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let a = touch(dir.path(), "a/x.wav");
        let b = touch(dir.path(), "b/x.wav");
        let files = [a, b];
        let formats = [OutputFormat::Json];

        let first = plan(&files, Some(&out));
        for target in &first {
            let path = json_path(target);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "").unwrap();
        }
        let second = plan(&files, Some(&out));

        for (input, target) in files.iter().zip(&second) {
            let check = should_process(input, target, &formats, false, false);
            assert!(
                matches!(check, ProcessCheck::SkipExists),
                "{} must find its own output on a rerun, got {check:?}",
                input.display()
            );
        }
    }

    #[test]
    fn test_should_process_ignores_a_flat_output_for_a_qualified_target() {
        // The symptom in #414: a flat `out/x.BirdNET.json`, left by a run that
        // saw only one x.wav, must not make either file skip once both are in
        // the run, because neither's output is that file any more.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let a = touch(dir.path(), "a/x.wav");
        let b = touch(dir.path(), "b/x.wav");
        touch(&out, "x.BirdNET.json");
        let targets = plan(&[a.clone(), b.clone()], Some(&out));

        for (input, target) in [&a, &b].into_iter().zip(&targets) {
            let check = should_process(input, target, &[OutputFormat::Json], false, false);
            assert!(
                matches!(check, ProcessCheck::Process),
                "{} must not be skipped on another file's output, got {check:?}",
                input.display()
            );
        }
    }

    #[test]
    fn test_plan_keeps_the_stem_name_when_nothing_collides() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let a = touch(dir.path(), "a/one.wav");
        let b = touch(dir.path(), "b/two.flac");

        let with_out = plan(&[a.clone(), b.clone()], Some(&out));
        assert_eq!(json_path(&with_out[0]), out.join("one.BirdNET.json"));
        assert_eq!(json_path(&with_out[1]), out.join("two.BirdNET.json"));

        let without_out = plan(&[a, b], None);
        assert_eq!(
            json_path(&without_out[0]),
            dir.path().join("a/one.BirdNET.json")
        );
        assert_eq!(
            json_path(&without_out[1]),
            dir.path().join("b/two.BirdNET.json")
        );
    }

    #[test]
    fn test_plan_qualifies_same_stem_files_in_one_folder_with_their_extension() {
        let dir = tempfile::tempdir().unwrap();
        let wav = touch(dir.path(), "x.wav");
        let flac = touch(dir.path(), "x.flac");
        let other = touch(dir.path(), "y.wav");

        let targets = plan(&[wav, flac, other], None);

        assert_eq!(
            json_path(&targets[0]),
            dir.path().join("x.wav.BirdNET.json")
        );
        assert_eq!(
            json_path(&targets[1]),
            dir.path().join("x.flac.BirdNET.json")
        );
        assert_eq!(json_path(&targets[2]), dir.path().join("y.BirdNET.json"));
    }

    #[test]
    fn test_plan_mirrors_same_named_files_from_different_folders_under_the_output_dir() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let a = touch(dir.path(), "in/a/x.wav");
        let b = touch(dir.path(), "in/b/x.wav");
        let c = touch(dir.path(), "in/b/z.wav");

        let targets = plan(&[a, b, c], Some(&out));

        assert_eq!(json_path(&targets[0]), out.join("a/x.wav.BirdNET.json"));
        assert_eq!(json_path(&targets[1]), out.join("b/x.wav.BirdNET.json"));
        assert_eq!(json_path(&targets[2]), out.join("z.BirdNET.json"));
    }

    #[test]
    fn test_plan_leaves_same_named_files_in_different_folders_alone_without_an_output_dir() {
        let dir = tempfile::tempdir().unwrap();
        let a = touch(dir.path(), "a/x.wav");
        let b = touch(dir.path(), "b/x.wav");

        let targets = plan(&[a, b], None);

        assert_eq!(json_path(&targets[0]), dir.path().join("a/x.BirdNET.json"));
        assert_eq!(json_path(&targets[1]), dir.path().join("b/x.BirdNET.json"));
    }

    #[test]
    fn test_plan_mirrors_explicit_files_from_different_folders() {
        // Explicit files, not a walked tree: the folders are mirrored relative
        // to the closest folder the two share, here the tempdir itself.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let d1 = touch(dir.path(), "d1/x.wav");
        let d2 = touch(dir.path(), "d2/x.wav");

        let targets = plan(&[d1, d2], Some(&out));

        assert_eq!(json_path(&targets[0]), out.join("d1/x.wav.BirdNET.json"));
        assert_eq!(json_path(&targets[1]), out.join("d2/x.wav.BirdNET.json"));
    }

    #[test]
    fn test_plan_mirrors_only_below_the_folder_the_group_shares() {
        // a/x.wav and a/b/x.wav: `a` is shared, so one lands directly in `out`
        // and the other in `out/b`; `a` itself is not repeated.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let top = touch(dir.path(), "a/x.wav");
        let deep = touch(dir.path(), "a/b/x.wav");

        let targets = plan(&[top, deep], Some(&out));

        assert_eq!(json_path(&targets[0]), out.join("x.wav.BirdNET.json"));
        assert_eq!(json_path(&targets[1]), out.join("b/x.wav.BirdNET.json"));
    }

    #[test]
    fn test_plan_treats_stems_that_differ_only_in_case_as_colliding() {
        // An output folder on exFAT or SMB cannot hold both `X.BirdNET.json`
        // and `x.BirdNET.json`, so case does not make the names different.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let upper = touch(dir.path(), "a/X.wav");
        let lower = touch(dir.path(), "b/x.wav");

        let targets = plan(&[upper, lower], Some(&out));

        assert_eq!(json_path(&targets[0]), out.join("a/X.wav.BirdNET.json"));
        assert_eq!(json_path(&targets[1]), out.join("b/x.wav.BirdNET.json"));
    }

    #[test]
    fn test_plan_qualifies_a_plain_name_that_matches_a_qualified_one() {
        // x.wav and x.flac collide and become `x.wav` and `x.flac`. The file
        // `x.wav.wav` has the stem `x.wav`, the name x.wav just took, so it
        // must be qualified too, and so on down the chain.
        let dir = tempfile::tempdir().unwrap();
        let files = [
            touch(dir.path(), "x.wav"),
            touch(dir.path(), "x.flac"),
            touch(dir.path(), "x.wav.wav"),
            touch(dir.path(), "x.wav.wav.wav"),
        ];

        let names: Vec<String> = plan(&files, None)
            .iter()
            .map(|t| t.base().to_string())
            .collect();

        assert_eq!(names, vec!["x.wav", "x.flac", "x.wav.wav", "x.wav.wav.wav"]);
    }

    #[test]
    fn test_plan_is_independent_of_input_order() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let files = vec![
            touch(dir.path(), "a/x.wav"),
            touch(dir.path(), "b/x.wav"),
            touch(dir.path(), "b/x.flac"),
            touch(dir.path(), "b/y.wav"),
        ];
        let mut reversed = files.clone();
        reversed.reverse();

        let forward = plan(&files, Some(&out));
        let backward = plan(&reversed, Some(&out));

        for (index, target) in forward.iter().enumerate() {
            assert_eq!(
                target,
                &backward[files.len() - 1 - index],
                "{} planned differently in reverse order",
                files[index].display()
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_plan_fails_every_member_of_an_unresolvable_collision() {
        // `A/x.wav` and `a/x.wav` mirror to `out/A` and `out/a`, which are one
        // folder on a case-insensitive filesystem. No name tells them apart, so
        // both fail rather than one overwriting the other. Linux only: other
        // filesystems cannot create both folders.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let upper = touch(dir.path(), "A/x.wav");
        let lower = touch(dir.path(), "a/x.wav");
        let fine = touch(dir.path(), "b/y.wav");

        let plans = plan_output_targets(&[upper.clone(), lower.clone(), fine], Some(&out));

        assert_eq!(plans.len(), 3);
        for failed in &plans[..2] {
            match failed {
                Err(Error::OutputPathCollision { inputs, .. }) => {
                    let mut inputs = inputs.clone();
                    inputs.sort();
                    assert_eq!(inputs, vec![upper.clone(), lower.clone()]);
                }
                other => panic!("expected an output path collision, got {other:?}"),
            }
        }
        assert_eq!(
            json_path(plans[2].as_ref().unwrap()),
            out.join("y.BirdNET.json")
        );
    }

    #[test]
    fn test_collect_input_files_drops_a_file_listed_twice() {
        // A directory and a file inside it reach the same file twice. Planned as
        // two inputs it would read as a name collision with itself.
        let dir = tempfile::tempdir().unwrap();
        let file = touch(dir.path(), "sub/x.wav");

        let files = collect_input_files(&[dir.path().to_path_buf(), file.clone()]).unwrap();

        assert_eq!(files, vec![file]);
    }

    #[cfg(unix)]
    #[test]
    fn test_collect_input_files_keeps_symlinks_to_one_file_in_different_folders() {
        let dir = tempfile::tempdir().unwrap();
        let real = touch(dir.path(), "data/x.wav");
        std::fs::create_dir_all(dir.path().join("a")).unwrap();
        std::fs::create_dir_all(dir.path().join("b")).unwrap();
        let first = dir.path().join("a/x.wav");
        let second = dir.path().join("b/x.wav");
        std::os::unix::fs::symlink(&real, &first).unwrap();
        std::os::unix::fs::symlink(&real, &second).unwrap();

        let files = collect_input_files(&[first.clone(), second.clone()]).unwrap();

        assert_eq!(files, vec![first, second]);
    }

    #[test]
    fn test_output_target_path_for_csv() {
        let target = OutputTarget {
            dir: PathBuf::from("/output"),
            root: PathBuf::from("/output"),
            base: "test".to_string(),
        };
        assert_eq!(
            target.path_for(OutputFormat::Csv).unwrap(),
            PathBuf::from("/output/test.BirdNET.results.csv")
        );
    }

    #[test]
    fn test_output_dir_for_with_explicit() {
        let input = Path::new("/data/audio.wav");
        let output = output_dir_for(input, Some(Path::new("/results")));
        assert_eq!(output, PathBuf::from("/results"));
    }

    #[test]
    fn test_output_dir_for_without_explicit() {
        let input = Path::new("/data/audio.wav");
        let output = output_dir_for(input, None);
        assert_eq!(output, PathBuf::from("/data"));
    }

    #[test]
    fn test_is_audio_file() {
        assert!(is_audio_file(Path::new("test.wav")));
        assert!(is_audio_file(Path::new("test.FLAC")));
        assert!(is_audio_file(Path::new("test.mp3")));
        assert!(!is_audio_file(Path::new("test.txt")));
    }

    #[test]
    fn test_is_audio_file_with_unicode() {
        // Test with Finnish/Swedish characters
        assert!(is_audio_file(Path::new("ääni_tiedostö.wav")));
        assert!(is_audio_file(Path::new("räkä.flac")));
        assert!(is_audio_file(Path::new("öljy.mp3")));
        assert!(is_audio_file(Path::new("テスト.wav"))); // Japanese
    }

    #[test]
    fn test_plan_preserves_unicode_names() {
        // Unicode filenames keep their names in the output.
        let dir = tempfile::tempdir().unwrap();
        let input = touch(dir.path(), "ääni_tiedostö.wav");

        let target = plan(&[input], None).remove(0);

        assert_eq!(target.base(), "ääni_tiedostö");
    }

    #[test]
    fn test_sanitize_filename_normal() {
        // Normal filenames should pass through unchanged
        assert_eq!(sanitize_filename("audio_file"), "audio_file");
        assert_eq!(sanitize_filename("recording-2024"), "recording-2024");
        assert_eq!(sanitize_filename("test123"), "test123");
    }

    #[test]
    fn test_sanitize_filename_path_separators() {
        // Path separators should be replaced with underscores
        assert_eq!(sanitize_filename("../etc/passwd"), ".._etc_passwd");
        assert_eq!(sanitize_filename("subdir/file"), "subdir_file");
        assert_eq!(
            sanitize_filename("..\\windows\\system32"),
            ".._windows_system32"
        );
    }

    #[test]
    fn test_sanitize_filename_parent_directory() {
        // Path separators in parent directory references are sanitized
        assert_eq!(sanitize_filename(".."), ".."); // No slashes to replace
        assert_eq!(sanitize_filename("../audio"), ".._audio");
        assert_eq!(sanitize_filename("../../file"), ".._.._file");
    }

    #[test]
    fn test_output_target_path_for_prevents_traversal() {
        // Sanitization alone keeps a name inside the folder; the check is the
        // second line. A target whose directory is outside its root must fail.
        let target = OutputTarget {
            dir: PathBuf::from("/elsewhere"),
            root: PathBuf::from("/safe/output"),
            base: "x".to_string(),
        };

        let err = target.path_for(OutputFormat::Json).unwrap_err();

        assert_eq!(
            err.to_string(),
            "output path '/elsewhere/x.BirdNET.json' escapes output directory '/safe/output'"
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_plan_sanitizes_a_name_that_would_traverse() {
        // A file name cannot hold a separator on Unix, so build the stem the
        // hostile way: a backslash, which sanitisation turns into an underscore.
        let dir = tempfile::tempdir().unwrap();
        let input = touch(dir.path(), "..\\evil.wav");

        let target = plan(&[input], None).remove(0);

        assert_eq!(target.base(), ".._evil");
        assert!(json_path(&target).starts_with(dir.path()));
    }
}
