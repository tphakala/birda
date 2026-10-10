//! Pipeline coordination for file processing.

use crate::config::OutputFormat;
use crate::constants::{FALLBACK_OUTPUT_NAME, output_extensions};
use crate::error::{Error, Result};
use crate::locking::FileLock;
use crate::output::OutputFiles;
use crate::output::ownership::{
    recorded_file_name, recorded_is_exactly_one_of, recorded_matches, recorded_source,
};
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};
use tracing::{info, warn};

/// Result of checking whether a file should be processed.
#[derive(Debug)]
pub enum ProcessCheck {
    /// File should be processed.
    Process,
    /// Skip - output already exists.
    SkipExists,
    /// Skip - file is locked by another process.
    SkipLocked,
    /// An existing output holds results recorded for another input, whose own
    /// results this run does not replace.
    OwnedByOther {
        /// The existing output file.
        output: PathBuf,
        /// The source it records, as written in it.
        recorded: String,
    },
}

/// Determine the output directory for a file.
fn output_dir_for(input: &Path, explicit_output_dir: Option<&Path>) -> PathBuf {
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

/// Sanitized output name from a file stem or file name, or a fallback when the
/// input path has none.
fn output_name(part: Option<&std::ffi::OsStr>) -> String {
    part.map_or_else(
        || FALLBACK_OUTPUT_NAME.to_string(),
        |s| sanitize_filename(&s.to_string_lossy()),
    )
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
    /// The input this target is for: its canonical folder joined with its file
    /// name. Result files record it so a later run can tell whose they are.
    source: PathBuf,
    /// Whether the output files sit next to the input (no `-o` was given).
    beside_input: bool,
}

impl OutputTarget {
    /// Directory the output files are written to.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The input this target is for: its canonical folder joined with its file
    /// name.
    #[must_use]
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// Whether the output files sit next to the input (no `-o` was given).
    #[must_use]
    pub const fn beside_input(&self) -> bool {
        self.beside_input
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
    /// The folder the output lands in, with the mirror folders under
    /// `canonical_dir` resolved through any symlink among them.
    resolved_dir: PathBuf,
}

impl Candidate {
    fn base(&self) -> &str {
        if self.qualified {
            &self.full
        } else {
            &self.stem
        }
    }

    /// Identity of the file this candidate would write: the resolved directory
    /// and the name, both lowercased when `fold_case`.
    fn key(&self, fold_case: bool) -> (PathBuf, String) {
        if fold_case {
            (
                PathBuf::from(self.resolved_dir.to_string_lossy().to_lowercase()),
                self.base().to_lowercase(),
            )
        } else {
            (self.resolved_dir.clone(), self.base().to_string())
        }
    }

    /// `plain_dir`, with the mirror subfolders under it. Only a run with `-o`
    /// sets a mirror, and its `plain_dir` is the `-o` directory.
    fn target_dir(&self) -> PathBuf {
        self.mirror
            .iter()
            .fold(self.plain_dir.clone(), |dir, part| dir.join(part))
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

/// `canonical_dir` results for one pass over a file list. Files mostly share a
/// few folders, so each folder is resolved once rather than once per file.
#[derive(Default)]
struct CanonicalDirs {
    dirs: HashMap<PathBuf, PathBuf>,
    resolved: HashMap<PathBuf, PathBuf>,
}

impl CanonicalDirs {
    fn get(&mut self, dir: &Path) -> PathBuf {
        self.dirs
            .entry(dir.to_path_buf())
            .or_insert_with(|| canonical_dir(dir))
            .clone()
    }

    /// `path` with its deepest existing ancestor canonicalized and the rest
    /// appended as it is, so a symlink among the folders that already exist
    /// shows in the result.
    fn resolve_through_existing(&mut self, path: &Path) -> PathBuf {
        if let Some(done) = self.resolved.get(path) {
            return done.clone();
        }
        let mut missing: Vec<OsString> = Vec::new();
        let mut head = path.to_path_buf();
        let resolved = loop {
            if let Ok(mut found) = std::fs::canonicalize(&head) {
                found.extend(missing.iter().rev());
                break found;
            }
            match (head.file_name(), head.parent()) {
                (Some(name), Some(parent)) => {
                    missing.push(name.to_os_string());
                    head = parent.to_path_buf();
                }
                _ => break path.to_path_buf(),
            }
        };
        self.resolved.insert(path.to_path_buf(), resolved.clone());
        resolved
    }
}

/// The identity of an input: its canonical folder joined with its file name.
fn input_identity(file: &Path, dirs: &mut CanonicalDirs) -> PathBuf {
    let parent = dirs.get(file.parent().unwrap_or_else(|| Path::new("")));
    file.file_name()
        .map_or_else(|| file.to_path_buf(), |name| parent.join(name))
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
fn group_by_key(
    candidates: &[Candidate],
    fold_case: bool,
) -> HashMap<(PathBuf, String), Vec<usize>> {
    let mut groups: HashMap<(PathBuf, String), Vec<usize>> = HashMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        groups
            .entry(candidate.key(fold_case))
            .or_default()
            .push(index);
    }
    groups
}

/// The folders every argument sits below, as folder names. A clashing input's
/// folder below this anchor is its mirror folder under `-o`.
///
/// The anchor is the longest prefix the arguments' folders share (a directory
/// argument itself, a file argument's parent). When one of the arguments is
/// that shared folder (a single argument, or one nested in another), the
/// anchor stops one folder short of it, so that argument keeps its own name in
/// the mirror instead of vanishing from it.
///
/// It depends only on the arguments, so it stays the same across reruns of the
/// same command whatever the inputs are.
fn mirror_anchor(roots: &[PathBuf], dirs: &mut CanonicalDirs) -> Vec<OsString> {
    let bases: Vec<Vec<OsString>> = roots
        .iter()
        .filter_map(|root| {
            if root.is_dir() {
                Some(dirs.get(root))
            } else {
                root.parent().map(|parent| dirs.get(parent))
            }
        })
        .map(|base| normal_components(&base))
        .collect();
    let Some(first) = bases.first() else {
        return Vec::new();
    };
    let shared_len = bases.iter().skip(1).fold(first.len(), |len, other| {
        len.min(first.iter().zip(other).take_while(|(a, b)| a == b).count())
    });
    let mut anchor = first[..shared_len].to_vec();
    if bases.iter().any(|base| base.len() == shared_len) {
        anchor.pop();
    }
    anchor
}

/// Folders under `-o` for a clashing input: its folder below the anchor, or
/// the whole folder when it is not under the anchor (a file reached through a
/// symlinked folder that resolves outside every argument).
fn mirror_for(candidate: &Candidate, anchor: &[OsString]) -> Vec<OsString> {
    let parts = normal_components(&candidate.canonical_parent);
    if parts.starts_with(anchor) {
        parts[anchor.len()..].to_vec()
    } else {
        parts
    }
}

/// Plan the output name of every input in a run.
///
/// Returns one entry per input, in input order. An input that shares no name
/// with another keeps the name it always had: its file stem, in `-o` or next to
/// the input. Inputs whose names would clash (the same stem, compared without
/// regard to case, in one output directory) all switch to their full file name,
/// so `x.wav` and `x.flac` become `x.wav.BirdNET.json` and `x.flac.BirdNET.json`.
/// When `-o` is given, each clashing input also goes into a folder that
/// mirrors its own folder below the anchor, the folder every argument sits
/// below (see [`mirror_anchor`]; with one argument `in`, the anchor is the
/// folder that holds `in`, so the mirror starts with `in`). Mirror folders are
/// the input's canonical folder with a common prefix removed, so two inputs in
/// different folders share one only when the folders differ in case, are the
/// same folder on two drives, or meet through a symlink (all reported below).
/// The rule is symmetric, so the result does
/// not depend on input order, and a clashing input's path depends only on its
/// own location and the arguments, so it does not move when inputs are added
/// under the same arguments.
///
/// A name that is still shared after that (under `-o`, folders that differ only
/// in case, the same folder on two drives, or folders that are one folder
/// through a symlink under `-o`) is an [`Error::OutputPathCollision`] for every
/// input involved. Without `-o`, names that differ only in case are not a
/// clash: the inputs' own folder holds both, so it tells case apart.
///
/// With `writes_files` false (`--stdout`) nothing is written, so there is
/// nothing for two names to collide on: every input keeps its stem and none
/// fails.
#[must_use]
pub fn plan_output_targets(
    files: &[PathBuf],
    roots: &[PathBuf],
    explicit_output_dir: Option<&Path>,
    writes_files: bool,
) -> Vec<Result<OutputTarget>> {
    let mut dirs = CanonicalDirs::default();
    if !writes_files {
        return files
            .iter()
            .map(|file| {
                let dir = output_dir_for(file, explicit_output_dir);
                Ok(OutputTarget {
                    root: dir.clone(),
                    dir,
                    base: output_name(file.file_stem()),
                    source: input_identity(file, &mut dirs),
                    beside_input: explicit_output_dir.is_none(),
                })
            })
            .collect();
    }

    let anchor = mirror_anchor(roots, &mut dirs);
    let mut candidates: Vec<Candidate> = files
        .iter()
        .map(|file| {
            let plain_dir = output_dir_for(file, explicit_output_dir);
            let canonical_parent = dirs.get(file.parent().unwrap_or_else(|| Path::new("")));
            let canonical_dir = dirs.get(&plain_dir);
            Candidate {
                stem: output_name(file.file_stem()),
                full: output_name(file.file_name()),
                resolved_dir: canonical_dir.clone(),
                canonical_dir,
                canonical_parent,
                plain_dir,
                qualified: false,
                mirror: Vec::new(),
            }
        })
        .collect();

    // Inputs that share a name: qualify them, and under `-o` put each in a
    // folder that mirrors its own folder below the anchor. The path depends only
    // on where the input is and which arguments were given, never on which other
    // inputs clash, so a rerun that finds more recordings cannot give one of
    // them the name another input's earlier output already has.
    for members in group_by_key(&candidates, true)
        .values()
        .filter(|m| m.len() > 1)
    {
        for &i in members {
            candidates[i].qualified = true;
            if explicit_output_dir.is_some() {
                let mirror = mirror_for(&candidates[i], &anchor);
                let mut folder = candidates[i].canonical_dir.clone();
                folder.extend(&mirror);
                candidates[i].resolved_dir = dirs.resolve_through_existing(&folder);
                candidates[i].mirror = mirror;
            }
        }
    }

    // A name that stayed plain can still match a qualified one (`x.wav` next to
    // `x.wav.wav` and `x.flac`). Qualify it too, until nothing changes. This
    // only happens without `-o`: under `-o` a qualified name sits in its
    // argument's folder and a plain one directly in `-o`, so they never meet.
    let groups = loop {
        let groups = group_by_key(&candidates, true);
        let mut changed = false;
        for members in groups.values().filter(|m| m.len() > 1) {
            for &i in members {
                if !candidates[i].qualified {
                    candidates[i].qualified = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break groups;
        }
    };

    // Names are qualified without regard to case, since `-o` can point at a
    // filesystem that ignores it. Without `-o` every output sits in its input's
    // own folder: two inputs there whose names differ only in case both exist,
    // so that folder tells case apart, and only an exact match is a clash.
    let fold_case = explicit_output_dir.is_some();
    let groups = if fold_case {
        groups
    } else {
        group_by_key(&candidates, false)
    };

    candidates
        .iter()
        .zip(files)
        .map(|(candidate, file)| {
            let target = OutputTarget {
                dir: candidate.target_dir(),
                root: candidate.plain_dir.clone(),
                base: candidate.base().to_string(),
                source: input_identity(file, &mut dirs),
                beside_input: explicit_output_dir.is_none(),
            };
            let clashing = &groups[&candidate.key(fold_case)];
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

/// Check if a file should be processed.
///
/// An input is skipped only when every requested output exists and none of
/// them records a different input as its source. `run_sources` holds the
/// identity ([`OutputTarget::source`]) of every input planned in this run: an
/// output that records one of those is a stale copy, since that input has its
/// own planned output, and is overwritten when the recorded path names that
/// input exactly. Beside its input (no `-o`), an output that names no input of
/// the run counts as the input's own when the recorded file name is the input's,
/// since the folder may have moved. An output recording any other input is
/// [`ProcessCheck::OwnedByOther`] unless `force` is set.
pub fn should_process(
    input: &Path,
    target: &OutputTarget,
    formats: &[OutputFormat],
    force: bool,
    stdout_mode: bool,
    run_sources: &[PathBuf],
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
    if force || formats.is_empty() {
        return ProcessCheck::Process;
    }

    let mut outputs: Vec<(OutputFormat, PathBuf)> = Vec::with_capacity(formats.len());
    for fmt in formats {
        match target.path_for(*fmt) {
            Ok(path) => outputs.push((*fmt, path)),
            Err(e) => warn!("Failed to generate output path: {}", e),
        }
    }
    // A folder at an output path is not an output, so `is_file`, not `exists`.
    let existing: Vec<&(OutputFormat, PathBuf)> =
        outputs.iter().filter(|(_, path)| path.is_file()).collect();

    // Whose is each existing output? Read what it records about its source.
    // An output that records nothing (no detections, or a format that holds no
    // source) cannot be told apart and counts as this input's, as it always did.
    let lexical = std::path::absolute(input).ok();
    let mut identities: Vec<&Path> = vec![target.source()];
    identities.extend(lexical.as_deref());
    let mut replaces_stale_copy = false;
    for (fmt, path) in &existing {
        let Some(recorded) = recorded_source(path, *fmt) else {
            continue;
        };
        if recorded_matches(&recorded, &identities) {
            continue;
        }
        if recorded_is_exactly_one_of(&recorded, run_sources) {
            replaces_stale_copy = true;
            continue;
        }
        // Beside its input, an output was written for the file of that name, so
        // when it names no input of this run only the name says whose it is:
        // the folder may have moved since.
        if target.beside_input() && recorded_matches(recorded_file_name(&recorded), &identities) {
            continue;
        }
        return ProcessCheck::OwnedByOther {
            output: path.clone(),
            recorded,
        };
    }
    if replaces_stale_copy {
        return ProcessCheck::Process;
    }

    if outputs.len() == formats.len() && existing.len() == formats.len() {
        return ProcessCheck::SkipExists;
    }

    ProcessCheck::Process
}

/// Collect input files from paths (files and directories).
///
/// A file that several arguments reach (listed twice, or inside a listed
/// directory) is kept once, at its first position. Two entries for one file
/// would otherwise be planned as a name collision. A regular file is compared
/// by its canonical path, which also catches a second spelling of one file on a
/// case-insensitive filesystem. A symlinked file is not followed: it is compared
/// by its resolved folder and its own name, so a link to a file already
/// collected is still its own input.
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

    let mut dirs = CanonicalDirs::default();
    let mut seen = HashSet::new();
    files.retain(|file| {
        // For a symlink, resolve the folder, not the file: two symlinks to one
        // recording in different folders are two inputs with two output
        // locations.
        let identity = match (file.parent(), file.file_name()) {
            (Some(parent), Some(name)) => dirs.get(parent).join(name),
            _ => file.clone(),
        };
        let resolved = if file.is_symlink() {
            None
        } else {
            std::fs::canonicalize(file).ok()
        };
        let first = seen.insert(input_key(identity, resolved));
        if !first {
            info!("Skipping duplicate input: {}", file.display());
        }
        first
    });

    Ok(files)
}

/// Identity `collect_input_files` compares inputs by: the canonical path of a
/// regular file (`resolved`), or else the resolved folder joined with the name
/// as given (`identity`), which is what a symlink gets.
fn input_key(identity: PathBuf, resolved: Option<PathBuf>) -> PathBuf {
    resolved.unwrap_or(identity)
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
        plan_output_targets(files, files, out, true)
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

        let check = should_process(&input, &target, &[], false, false, &[]);

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

        let check = should_process(&input, &target, &[OutputFormat::Csv], false, false, &[]);

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
            let check = should_process(input, target, &formats, false, false, &[]);
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
            let check = should_process(input, target, &[OutputFormat::Json], false, false, &[]);
            assert!(
                matches!(check, ProcessCheck::Process),
                "{} must not be skipped on another file's output, got {check:?}",
                input.display()
            );
        }
    }

    /// Write the result file of `format` at `path` the way the real writers do,
    /// for an input named `recorded` in the file (and, for JSON, `source_path`).
    fn write_result(path: &Path, format: OutputFormat, recorded: &str, source_path: Option<&Path>) {
        use crate::output::{CsvWriter, Detection, JsonResultWriter, OutputWriter, RavenWriter};
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let detection = Detection::from_label(
            "Passer domesticus_House Sparrow",
            0.9,
            0.0,
            3.0,
            PathBuf::from(recorded),
        );
        let mut writer: Box<dyn OutputWriter> = match format {
            OutputFormat::Csv => Box::new(CsvWriter::new(path, Vec::new(), true).unwrap()),
            OutputFormat::Raven => Box::new(RavenWriter::new(path).unwrap()),
            OutputFormat::Json => {
                let name = Path::new(recorded).file_name().unwrap().to_string_lossy();
                let mut json =
                    JsonResultWriter::new(path, &name, 10.0, "m", 0.1, 0.0, None, None, None)
                        .unwrap();
                if let Some(source) = source_path {
                    json = json.with_source_path(&source.to_string_lossy());
                }
                Box::new(json)
            }
            other => panic!("no result writer for {other:?}"),
        };
        writer.write_header().unwrap();
        writer.write_detection(&detection).unwrap();
        writer.finalize().unwrap();
    }

    /// Plan one input on its own, as a run over just that file would.
    fn plan_alone(file: &Path, out: Option<&Path>) -> OutputTarget {
        plan(std::slice::from_ref(&file.to_path_buf()), out).remove(0)
    }

    #[test]
    fn test_should_process_skips_its_own_outputs_on_rerun() {
        let dir = tempfile::tempdir().unwrap();
        let input = touch(dir.path(), "in/x.wav");
        let target = plan_alone(&input, None);
        write_result(
            &target.path_for(OutputFormat::Json).unwrap(),
            OutputFormat::Json,
            "x.wav",
            Some(target.source()),
        );
        // The CLI records the input path as the user typed it.
        write_result(
            &target.path_for(OutputFormat::Csv).unwrap(),
            OutputFormat::Csv,
            "in/x.wav",
            None,
        );
        write_result(
            &target.path_for(OutputFormat::Raven).unwrap(),
            OutputFormat::Raven,
            "./x.wav",
            None,
        );

        let check = should_process(
            &input,
            &target,
            &[OutputFormat::Json, OutputFormat::Csv, OutputFormat::Raven],
            false,
            false,
            &[],
        );

        assert!(
            matches!(check, ProcessCheck::SkipExists),
            "an input must find its own outputs on a rerun, got {check:?}"
        );
    }

    #[test]
    fn test_should_process_does_not_skip_an_output_recorded_for_another_input() {
        // The sequence from #416 without `-o`. Run 1 sees `x.wav.wav` and
        // `x.flac`, which do not clash, so `x.wav.wav` writes
        // `x.wav.BirdNET.json`. Run 2 adds `x.wav`, which clashes with `x.flac`
        // and so takes exactly that name: the file is there, but it is not
        // `x.wav`'s result.
        let dir = tempfile::tempdir().unwrap();
        let double = touch(dir.path(), "x.wav.wav");
        let flac = touch(dir.path(), "x.flac");
        let wav = touch(dir.path(), "x.wav");
        let first = plan(&[double.clone(), flac.clone()], None);
        for (input, target) in [&double, &flac].into_iter().zip(&first) {
            let source = target.source().to_path_buf();
            write_result(
                &json_path(target),
                OutputFormat::Json,
                &input.to_string_lossy(),
                Some(&source),
            );
        }
        let files = [double, flac, wav.clone()];
        let second = plan(&files, None);
        let run_sources: Vec<PathBuf> = second.iter().map(|t| t.source().to_path_buf()).collect();

        let check = should_process(
            &wav,
            &second[2],
            &[OutputFormat::Json],
            false,
            false,
            &run_sources,
        );

        assert_eq!(
            json_path(&second[2]),
            json_path(&first[0]),
            "the fixture needs x.wav to plan the file x.wav.wav wrote"
        );
        assert!(
            matches!(check, ProcessCheck::Process),
            "x.wav must not be skipped on x.wav.wav's results, got {check:?}"
        );
    }

    #[test]
    fn test_should_process_overwrites_a_csv_or_raven_copy_recorded_for_an_input_in_the_run() {
        // The default format is CSV, which records the input path as typed. An
        // absolute argument (as a GUI passes) names its owner exactly, so an
        // owner that is part of this run has its own output and this copy is
        // stale.
        let dir = tempfile::tempdir().unwrap();
        let double = touch(dir.path(), "x.wav.wav");
        let flac = touch(dir.path(), "x.flac");
        let wav = touch(dir.path(), "x.wav");
        let files = [double, flac, wav.clone()];
        let plans = plan(&files, None);
        let run_sources: Vec<PathBuf> = plans.iter().map(|t| t.source().to_path_buf()).collect();
        for (format, recorded) in [
            (OutputFormat::Csv, run_sources[0].to_string_lossy()),
            (OutputFormat::Raven, run_sources[0].to_string_lossy()),
        ] {
            let path = plans[2].path_for(format).unwrap();
            write_result(&path, format, &recorded, None);

            let check = should_process(&wav, &plans[2], &[format], false, false, &run_sources);
            let alone = should_process(&wav, &plans[2], &[format], false, false, &[]);

            assert!(
                matches!(check, ProcessCheck::Process),
                "{format:?}: an owner in the run is overwritten, got {check:?}"
            );
            assert!(
                matches!(&alone, ProcessCheck::OwnedByOther { recorded: r, .. } if *r == *recorded),
                "{format:?}: an owner outside the run is refused, got {alone:?}"
            );
        }
    }

    #[test]
    fn test_should_process_overwrites_a_copy_recorded_with_a_relative_path_for_an_input_in_the_run()
    {
        // The value the CLI writes when the owner was typed relative to the
        // current directory, here with `..` steps up to the temp folder.
        let dir = tempfile::tempdir().unwrap();
        let double = touch(dir.path(), "x.wav.wav");
        let flac = touch(dir.path(), "x.flac");
        let wav = touch(dir.path(), "x.wav");
        let plans = plan(&[double, flac, wav.clone()], None);
        let run_sources: Vec<PathBuf> = plans.iter().map(|t| t.source().to_path_buf()).collect();
        let cwd = std::env::current_dir().unwrap();
        let mut relative = PathBuf::new();
        for _ in normal_components(&cwd) {
            relative.push("..");
        }
        relative.extend(normal_components(&run_sources[0]));
        let recorded = relative.to_string_lossy().into_owned();
        write_result(
            &plans[2].path_for(OutputFormat::Csv).unwrap(),
            OutputFormat::Csv,
            &recorded,
            None,
        );

        let check = should_process(
            &wav,
            &plans[2],
            &[OutputFormat::Csv],
            false,
            false,
            &run_sources,
        );

        assert!(matches!(check, ProcessCheck::Process), "got {check:?}");
    }

    #[test]
    fn test_should_process_refuses_to_overwrite_an_output_whose_owner_is_not_in_the_run() {
        // `card1` was analyzed into `out`; a later run over `card2` plans the
        // same `out/REC0001.BirdNET.json` (a single argument never clashes with
        // itself) but must not replace card1's results.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let first = touch(dir.path(), "card1/REC0001.WAV");
        let second = touch(dir.path(), "card2/REC0001.WAV");
        let first_target = plan_alone(&first, Some(&out));
        let second_target = plan_alone(&second, Some(&out));
        let output = json_path(&first_target);
        assert_eq!(output, json_path(&second_target));
        write_result(
            &output,
            OutputFormat::Json,
            "REC0001.WAV",
            Some(first_target.source()),
        );
        let run_sources = [second_target.source().to_path_buf()];

        let check = should_process(
            &second,
            &second_target,
            &[OutputFormat::Json],
            false,
            false,
            &run_sources,
        );
        let forced = should_process(
            &second,
            &second_target,
            &[OutputFormat::Json],
            true,
            false,
            &run_sources,
        );

        match check {
            ProcessCheck::OwnedByOther {
                output: found,
                recorded,
            } => {
                assert_eq!(found, output);
                assert_eq!(recorded, first_target.source().to_string_lossy());
            }
            other => panic!("expected OwnedByOther, got {other:?}"),
        }
        assert!(
            matches!(forced, ProcessCheck::Process),
            "--force overwrites, got {forced:?}"
        );
    }

    #[test]
    fn test_should_process_recognises_outputs_written_without_source_path() {
        // Files from earlier versions: JSON that names only the file, and CSV
        // whose `File` is the path as the user typed it, relative to wherever
        // they ran.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let first = touch(dir.path(), "in/a/x.wav");
        let second = touch(dir.path(), "in/b/x.wav");
        let first_target = plan_alone(&first, Some(&out));
        let second_target = plan_alone(&second, Some(&out));
        let csv = first_target.path_for(OutputFormat::Csv).unwrap();
        assert_eq!(csv, second_target.path_for(OutputFormat::Csv).unwrap());
        write_result(&csv, OutputFormat::Csv, "in/a/x.wav", None);
        let json = first_target.path_for(OutputFormat::Json).unwrap();
        std::fs::write(
            &json,
            r#"{"source_file":"x.wav","analysis_date":"2024-01-01T00:00:00Z","model":"m","settings":{"min_confidence":0.1,"overlap":0.0},"detections":[],"summary":{"total_detections":0,"unique_species":0,"audio_duration_seconds":1.0}}"#,
        )
        .unwrap();

        let own = should_process(
            &first,
            &first_target,
            &[OutputFormat::Csv],
            false,
            false,
            &[],
        );
        let other = should_process(
            &second,
            &second_target,
            &[OutputFormat::Csv],
            false,
            false,
            &[],
        );
        let legacy_json = should_process(
            &first,
            &first_target,
            &[OutputFormat::Json],
            false,
            false,
            &[],
        );

        assert!(matches!(own, ProcessCheck::SkipExists), "got {own:?}");
        assert!(
            matches!(&other, ProcessCheck::OwnedByOther { recorded, .. } if recorded == "in/a/x.wav"),
            "got {other:?}"
        );
        assert!(
            matches!(legacy_json, ProcessCheck::SkipExists),
            "got {legacy_json:?}"
        );
    }

    #[test]
    fn test_should_process_does_not_let_a_same_named_input_claim_a_relative_owner() {
        // The CSV was written for some `audio.wav` run from its own folder, so
        // it records only `audio.wav`. `audio.flac` finds it at its path. The
        // run also holds an unrelated `/deep/audio.wav`: its name ends the same
        // way, but the CSV is not shown to be its result, so `audio.flac` must
        // not overwrite it on that account.
        let dir = tempfile::tempdir().unwrap();
        let flac = touch(dir.path(), "new/audio.flac");
        let unrelated = touch(dir.path(), "deep/audio.wav");
        let targets = plan(&[flac.clone(), unrelated], None);
        let run_sources: Vec<PathBuf> = targets.iter().map(|t| t.source().to_path_buf()).collect();
        write_result(
            &targets[0].path_for(OutputFormat::Csv).unwrap(),
            OutputFormat::Csv,
            "audio.wav",
            None,
        );

        let check = should_process(
            &flac,
            &targets[0],
            &[OutputFormat::Csv],
            false,
            false,
            &run_sources,
        );

        assert!(
            matches!(&check, ProcessCheck::OwnedByOther { recorded, .. } if recorded == "audio.wav"),
            "got {check:?}"
        );
    }

    #[test]
    fn test_should_process_checks_ownership_when_only_some_outputs_exist() {
        // Adding a format on a rerun must not overwrite another input's CSV
        // either, so the check does not wait for every output to exist.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let first = touch(dir.path(), "card1/REC0001.WAV");
        let second = touch(dir.path(), "card2/REC0001.WAV");
        let first_target = plan_alone(&first, Some(&out));
        let second_target = plan_alone(&second, Some(&out));
        write_result(
            &first_target.path_for(OutputFormat::Csv).unwrap(),
            OutputFormat::Csv,
            &first.to_string_lossy(),
            None,
        );

        let check = should_process(
            &second,
            &second_target,
            &[OutputFormat::Csv, OutputFormat::Json],
            false,
            false,
            &[],
        );

        assert!(
            matches!(check, ProcessCheck::OwnedByOther { .. }),
            "got {check:?}"
        );
    }

    #[test]
    fn test_should_process_treats_a_folder_at_the_output_path_as_missing() {
        let dir = tempfile::tempdir().unwrap();
        let input = touch(dir.path(), "recording.wav");
        let target = plan_alone(&input, None);
        std::fs::create_dir(json_path(&target)).unwrap();

        let check = should_process(&input, &target, &[OutputFormat::Json], false, false, &[]);

        assert!(matches!(check, ProcessCheck::Process), "got {check:?}");
    }

    #[test]
    fn test_should_process_skips_outputs_beside_the_input_after_its_folder_moved() {
        // Without `-o` the output sits next to its input, so a copy recorded
        // for the same file name in an old folder is that input's own result:
        // the folder moved. A different name in the same place is not.
        let dir = tempfile::tempdir().unwrap();
        let input = touch(dir.path(), "in/x.wav");
        let target = plan_alone(&input, None);
        let output = json_path(&target);

        write_result(
            &output,
            OutputFormat::Json,
            "x.wav",
            Some(Path::new("/old/place/in/x.wav")),
        );
        let moved = should_process(&input, &target, &[OutputFormat::Json], false, false, &[]);
        write_result(
            &output,
            OutputFormat::Json,
            "x.wav.wav",
            Some(Path::new("/old/place/in/x.wav.wav")),
        );
        let renamed = should_process(&input, &target, &[OutputFormat::Json], false, false, &[]);

        // The copy records a same-named input that is part of this run: it has
        // its own output, so this one is the stale copy and is overwritten.
        let other = touch(dir.path(), "other/x.wav");
        let other_target = plan_alone(&other, None);
        write_result(
            &output,
            OutputFormat::Json,
            "x.wav",
            Some(other_target.source()),
        );
        let run_sources = [
            target.source().to_path_buf(),
            other_target.source().to_path_buf(),
        ];
        let stale = should_process(
            &input,
            &target,
            &[OutputFormat::Json],
            false,
            false,
            &run_sources,
        );

        assert!(matches!(moved, ProcessCheck::SkipExists), "got {moved:?}");
        assert!(matches!(stale, ProcessCheck::Process), "got {stale:?}");
        assert!(
            matches!(&renamed, ProcessCheck::OwnedByOther { recorded, .. } if recorded == "/old/place/in/x.wav.wav"),
            "got {renamed:?}"
        );
    }

    #[test]
    fn test_should_process_refuses_a_same_named_output_under_an_explicit_output_dir() {
        // With `-o` many folders write into one place, so the same file name
        // recorded for another folder is another input's result.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let input = touch(dir.path(), "in/x.wav");
        let target = plan_alone(&input, Some(&out));
        write_result(
            &json_path(&target),
            OutputFormat::Json,
            "x.wav",
            Some(Path::new("/elsewhere/x.wav")),
        );

        let check = should_process(&input, &target, &[OutputFormat::Json], false, false, &[]);

        assert!(
            matches!(&check, ProcessCheck::OwnedByOther { recorded, .. } if recorded == "/elsewhere/x.wav"),
            "got {check:?}"
        );
    }

    #[test]
    fn test_should_process_does_not_take_an_absolute_value_that_is_only_the_tail_of_the_input() {
        // The JSON records an absolute path that ends the way this input's path
        // does but starts elsewhere: it names a different file.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let input = touch(dir.path(), "in/x.wav");
        let target = plan_alone(&input, Some(&out));
        let tail = target
            .source()
            .strip_prefix(Path::new("/").join(target.source().components().nth(1).unwrap()))
            .unwrap()
            .to_path_buf();
        let recorded = Path::new("/").join(tail);
        write_result(
            &json_path(&target),
            OutputFormat::Json,
            "x.wav",
            Some(&recorded),
        );

        let check = should_process(&input, &target, &[OutputFormat::Json], false, false, &[]);

        assert!(
            matches!(&check, ProcessCheck::OwnedByOther { .. }),
            "got {check:?}"
        );
    }

    #[test]
    fn test_should_process_refuses_a_same_named_output_when_the_output_dir_is_the_input_folder() {
        // `-o` naming the input's own folder is still an explicit output
        // directory: other folders' inputs can write there too.
        let dir = tempfile::tempdir().unwrap();
        let input = touch(dir.path(), "in/x.wav");
        let target = plan_alone(&input, input.parent());
        write_result(
            &json_path(&target),
            OutputFormat::Json,
            "x.wav",
            Some(Path::new("/elsewhere/x.wav")),
        );

        let check = should_process(&input, &target, &[OutputFormat::Json], false, false, &[]);

        assert!(
            matches!(&check, ProcessCheck::OwnedByOther { recorded, .. } if recorded == "/elsewhere/x.wav"),
            "got {check:?}"
        );
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

    /// Plan `files` as found under the directory argument `root`.
    fn plan_under(root: &Path, files: &[PathBuf], out: &Path) -> Vec<PathBuf> {
        plan_output_targets(files, &[root.to_path_buf()], Some(out), true)
            .iter()
            .map(|t| json_path(t.as_ref().unwrap()))
            .collect()
    }

    #[test]
    fn test_plan_mirrors_folders_below_the_directory_argument() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in");
        let out = dir.path().join("out");
        let top = touch(&input, "a/x.wav");
        let deep = touch(&input, "a/b/x.wav");

        let names = plan_under(&input, &[top, deep], &out);

        assert_eq!(
            names,
            vec![
                out.join("in").join("a").join("x.wav.BirdNET.json"),
                out.join("in")
                    .join("a")
                    .join("b")
                    .join("x.wav.BirdNET.json"),
            ]
        );
    }

    #[test]
    fn test_plan_keeps_mirrored_names_when_a_recording_is_added() {
        // A rerun with one more recording must not hand the new one a name an
        // earlier output already has: that output would be taken as its result.
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in");
        let out = dir.path().join("out");
        let top = touch(&input, "a/x.wav");
        let deep = touch(&input, "a/b/x.wav");
        let first = plan_under(&input, &[top.clone(), deep.clone()], &out);
        let added = touch(&input, "b/x.wav");

        let second = plan_under(&input, &[top, deep, added], &out);

        assert_eq!(second[..2], first[..]);
        assert!(
            !first.contains(&second[2]),
            "{} is reused",
            second[2].display()
        );
    }

    #[test]
    fn test_plan_keeps_mirrored_names_when_several_arguments_find_more() {
        // Expanding a run under two arguments must not move an output either:
        // the moved-from path would then be taken as a new input's result.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let one = dir.path().join("one");
        let two = dir.path().join("two");
        let roots = [one.clone(), two.clone()];
        let deep = touch(&one, "one/a/x.wav");
        let other = touch(&two, "a/x.wav");
        let names = |files: &[PathBuf]| -> Vec<PathBuf> {
            plan_output_targets(files, &roots, Some(&out), true)
                .iter()
                .map(|t| json_path(t.as_ref().unwrap()))
                .collect()
        };
        let first = names(&[deep.clone(), other.clone()]);
        let shallow = touch(&one, "a/x.wav");
        let nested = touch(&two, "one/a/x.wav");

        let second = names(&[deep, other, shallow, nested]);

        assert_eq!(second[..2], first[..]);
        for added in &second[2..] {
            assert!(!first.contains(added), "{} is reused", added.display());
        }
    }

    #[test]
    fn test_plan_tells_apart_arguments_whose_folder_names_match() {
        // Recorder folders are often named by date, so two stations give two
        // arguments that both end in `2024-05-01`.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let first_day = dir.path().join("stationA").join("2024-05-01");
        let second_day = dir.path().join("stationB").join("2024-05-01");
        let first = touch(&first_day, "x.wav");
        let second = touch(&second_day, "x.wav");

        let plans =
            plan_output_targets(&[first, second], &[first_day, second_day], Some(&out), true);

        let names: Vec<Option<PathBuf>> = plans
            .iter()
            .map(|p| p.as_ref().ok().map(json_path))
            .collect();
        assert_eq!(
            names,
            vec![
                Some(
                    out.join("stationA")
                        .join("2024-05-01")
                        .join("x.wav.BirdNET.json")
                ),
                Some(
                    out.join("stationB")
                        .join("2024-05-01")
                        .join("x.wav.BirdNET.json")
                ),
            ]
        );
    }

    #[test]
    fn test_plan_keeps_arguments_apart_when_one_label_ends_where_another_begins() {
        // `p/2024` and `r/p` used to meet: the first was labelled `p/2024` to
        // tell it from `q/2024`, the second `p`, and its folder `2024` below the
        // argument made the same mirror folder `p/2024`.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let first = dir.path().join("p").join("2024");
        let second = dir.path().join("q").join("2024");
        let third = dir.path().join("r").join("p");
        let in_first = touch(&first, "x.wav");
        touch(&first, "x.flac");
        let in_second = touch(&second, "y.wav");
        let in_third = touch(&third, "2024/x.wav");
        let files = [in_first, in_second, in_third];

        let plans = plan_output_targets(&files, &[first, second, third], Some(&out), true);

        let names: Vec<Option<PathBuf>> = plans
            .iter()
            .map(|p| p.as_ref().ok().map(json_path))
            .collect();
        assert_eq!(
            names,
            vec![
                Some(out.join("p").join("2024").join("x.wav.BirdNET.json")),
                Some(out.join("y.BirdNET.json")),
                Some(
                    out.join("r")
                        .join("p")
                        .join("2024")
                        .join("x.wav.BirdNET.json")
                ),
            ]
        );
    }

    #[test]
    fn test_plan_gives_the_outer_of_nested_same_name_arguments_its_own_folder() {
        // `rec` and `rec/a/rec` share the folder name. The outer one used to get
        // an empty label, so its clashing outputs landed directly in `out`.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let outer = dir.path().join("rec");
        let inner = outer.join("a").join("rec");
        let in_outer = touch(&outer, "x.wav");
        let in_inner = touch(&inner, "x.wav");

        let plans = plan_output_targets(&[in_outer, in_inner], &[outer, inner], Some(&out), true);

        let names: Vec<Option<PathBuf>> = plans
            .iter()
            .map(|p| p.as_ref().ok().map(json_path))
            .collect();
        assert_eq!(
            names,
            vec![
                Some(out.join("rec").join("x.wav.BirdNET.json")),
                Some(
                    out.join("rec")
                        .join("a")
                        .join("rec")
                        .join("x.wav.BirdNET.json")
                ),
            ]
        );
    }

    #[test]
    fn test_plan_mirror_does_not_depend_on_other_inputs() {
        // The path of a clashing input depends on the input and the arguments
        // only: adding, removing or reordering other recordings must not move it.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let one = dir.path().join("one");
        let two = dir.path().join("two");
        let roots = [one.clone(), two.clone()];
        let a = touch(&one, "a/x.wav");
        let b = touch(&two, "a/x.wav");
        let c = touch(&one, "b/x.wav");
        let d = touch(&two, "a/x.flac");
        let lone = touch(&two, "z/other.wav");
        let sequences: Vec<Vec<PathBuf>> = vec![
            vec![a.clone(), b.clone()],
            vec![a.clone(), b.clone(), c.clone()],
            vec![b.clone(), a.clone(), c.clone(), d.clone()],
            vec![c, lone.clone(), a.clone()],
            vec![d, b.clone(), lone],
        ];
        let mut seen: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();

        for files in &sequences {
            let plans = plan_output_targets(files, &roots, Some(&out), true);
            for (file, plan) in files.iter().zip(&plans) {
                // Only inputs that clash get a mirror; compare those.
                let path = json_path(plan.as_ref().unwrap());
                if path.starts_with(out.join("one")) || path.starts_with(out.join("two")) {
                    seen.entry(file.clone()).or_default().push(path);
                }
            }
        }

        for (file, paths) in &seen {
            assert!(
                paths.windows(2).all(|w| w[0] == w[1]),
                "{} moved between runs: {paths:?}",
                file.display()
            );
        }
        assert!(seen.contains_key(&a) && seen.contains_key(&b));
    }

    #[cfg(unix)]
    #[test]
    fn test_plan_fails_inputs_whose_mirror_folders_are_one_folder_through_a_symlink() {
        // `out/in/b` is a link to `out/in/a`, so the two mirrored outputs would
        // be one file. The lexical mirror folders differ, so only resolving the
        // link shows it.
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in");
        let out = dir.path().join("out");
        let first = touch(&input, "a/x.wav");
        let second = touch(&input, "b/x.wav");
        std::fs::create_dir_all(out.join("in").join("a")).unwrap();
        std::os::unix::fs::symlink(out.join("in").join("a"), out.join("in").join("b")).unwrap();

        let plans = plan_output_targets(&[first, second], &[input], Some(&out), true);

        assert!(
            plans
                .iter()
                .all(|p| matches!(p, Err(Error::OutputPathCollision { .. }))),
            "both inputs must fail as a collision, got {plans:?}"
        );
    }

    #[test]
    fn test_plan_keeps_a_plain_name_that_only_matches_a_mirrored_one() {
        // Under `-o` the qualified `x.wav` goes into the argument's folder, so
        // the plain name `x.wav.wav` gets directly in `out` does not meet it.
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in");
        let out = dir.path().join("out");
        let wav = touch(&input, "x.wav");
        let flac = touch(&input, "x.flac");
        let double = touch(&input, "x.wav.wav");

        let names = plan_under(&input, &[wav, flac, double], &out);

        assert_eq!(
            names,
            vec![
                out.join("in").join("x.wav.BirdNET.json"),
                out.join("in").join("x.flac.BirdNET.json"),
                out.join("x.wav.BirdNET.json"),
            ]
        );
    }

    #[test]
    fn test_plan_puts_inputs_from_different_arguments_under_their_names() {
        // Two directory arguments with the same layout meet at `a/x.wav`, so
        // each also gets its argument's folder name.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let one = dir.path().join("one");
        let two = dir.path().join("two");
        let first = touch(&one, "a/x.wav");
        let second = touch(&two, "a/x.wav");

        let plans = plan_output_targets(&[first, second], &[one, two], Some(&out), true);

        let names: Vec<Option<PathBuf>> = plans
            .iter()
            .map(|p| p.as_ref().ok().map(json_path))
            .collect();
        assert_eq!(
            names,
            vec![
                Some(out.join("one").join("a").join("x.wav.BirdNET.json")),
                Some(out.join("two").join("a").join("x.wav.BirdNET.json")),
            ]
        );
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
    fn test_plan_keeps_names_that_differ_only_in_case_apart_without_an_output_dir() {
        // Both files exist in one folder, so it tells case apart, and the
        // outputs written next to them can too.
        let dir = tempfile::tempdir().unwrap();
        let upper = touch(dir.path(), "X.wav");
        let lower = touch(dir.path(), "x.wav");
        // Only meaningful where both names are separate files.
        if std::fs::read_dir(dir.path()).unwrap().count() != 2 {
            return;
        }

        let plans =
            plan_output_targets(&[upper.clone(), lower.clone()], &[upper, lower], None, true);

        let names: Vec<Option<PathBuf>> = plans
            .iter()
            .map(|p| p.as_ref().ok().map(json_path))
            .collect();
        assert_eq!(
            names,
            vec![
                Some(dir.path().join("X.wav.BirdNET.json")),
                Some(dir.path().join("x.wav.BirdNET.json")),
            ]
        );
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

        let names: Vec<String> = plan(&files, None).iter().map(|t| t.base.clone()).collect();

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

        let plans = plan_output_targets(
            &[upper.clone(), lower.clone(), fine.clone()],
            &[upper.clone(), lower.clone(), fine],
            Some(&out),
            true,
        );

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
    fn test_input_key_uses_the_canonical_path_of_a_regular_file() {
        // On a case-insensitive filesystem `x.wav` and `X.wav` name one file:
        // the spellings differ, but canonicalizing either gives the same path.
        assert_eq!(
            input_key(PathBuf::from("/d/X.wav"), Some(PathBuf::from("/d/x.wav"))),
            PathBuf::from("/d/x.wav")
        );
    }

    #[test]
    fn test_input_key_keeps_the_name_of_a_symlink() {
        // A symlink is not followed, so two links to one recording stay two
        // inputs.
        assert_eq!(
            input_key(PathBuf::from("/a/x.wav"), None),
            PathBuf::from("/a/x.wav")
        );
    }

    #[cfg(windows)]
    #[test]
    fn test_collect_input_files_drops_a_file_listed_in_two_cases() {
        let dir = tempfile::tempdir().unwrap();
        let file = touch(dir.path(), "x.wav");

        let files = collect_input_files(&[file.clone(), dir.path().join("X.wav")]).unwrap();

        assert_eq!(files, vec![file]);
    }

    #[test]
    fn test_output_target_path_for_csv() {
        let target = OutputTarget {
            dir: PathBuf::from("/output"),
            root: PathBuf::from("/output"),
            base: "test".to_string(),
            source: PathBuf::from("/in/test.wav"),
            beside_input: false,
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

        assert_eq!(target.base, "ääni_tiedostö");
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
            source: PathBuf::from("/in/x.wav"),
            beside_input: false,
        };

        let err = target.path_for(OutputFormat::Json).unwrap_err();

        // Built with `join`, so the separator is the platform's own.
        let escaped = Path::new("/elsewhere").join("x.BirdNET.json");
        assert_eq!(
            err.to_string(),
            format!(
                "output path '{}' escapes output directory '/safe/output'",
                escaped.display()
            )
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

        assert_eq!(target.base, ".._evil");
        assert!(json_path(&target).starts_with(dir.path()));
    }
}
