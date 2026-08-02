//! Durable file writes.
//!
//! The rule, rather than a list that goes stale: anything that replaces a file
//! the user would miss if it were lost halfway through goes through
//! [`write_atomic`] or [`write_atomic_with`]. A caller that performs a rename of
//! its own, because it streams or extracts the file rather than writing it in
//! one pass, uses [`sync_parent_directory`] to finish the job.
//!
//! Deliberately NOT here: the analysis-result writers in `crate::output`. They
//! write straight to their final path, so an interrupted run leaves a truncated
//! results file. Whether they should be published atomically is a product
//! question rather than an oversight, because atomic publication means the file
//! does not appear until the run ends, and a long analysis is exactly the case
//! where watching it matters.
//!
//! The sequence, and what each step is for:
//!
//! 1. Write the new contents to a temporary **in the target's directory**.
//!    Rename is only atomic within a filesystem and fails outright with EXDEV
//!    across one, and `$TMPDIR` is routinely a different filesystem from the
//!    user's home directory, so the temporary cannot live there.
//! 2. `fsync` the temporary. Without this the rename can reach the disk while
//!    the data behind it has not, leaving the target pointing at a file of
//!    zeros: the outcome the rename is here to prevent, reached by another
//!    route.
//! 3. Rename it over the target. This is the step that makes the replacement
//!    atomic: a concurrent reader sees either the whole old file or the whole
//!    new one, never a truncated one.
//! 4. `fsync` the target's directory, so the rename itself survives a crash.
//!
//! What this does not do is serialise concurrent writers. The config and
//! registry callers are lock-free load-mutate-save, so two overlapping saves
//! still lose one of the two edits. The rename makes each write whole, not the
//! pair of them ordered. (The clip and species-list callers build their contents
//! from memory and never read the file they replace, so they have no such pair.)
//!
//! On the 500-line module guideline: the production half is under it, and the
//! overrun is the co-located `#[cfg(test)]` suite that the same guidelines require
//! to live here. Splitting the tests out to satisfy one rule would break the other,
//! so the file is deliberately over budget.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The mode to create a file with, where the platform has one.
///
/// Only consulted when the target does not exist yet. A file being *replaced*
/// keeps the mode it already had, whatever this says; see [`write_atomic_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewFileMode {
    /// Readable and writable only by its owner.
    ///
    /// For files in the user's config directory, which are per-user by
    /// definition and may hold something worth not sharing.
    OwnerOnly,
    /// Whatever the process umask allows, as `File::create` would have given.
    ///
    /// For files the user asked to be produced, such as an extracted clip. A
    /// clip directory can be served by a web server or read by another account,
    /// so narrowing these to owner-only would break a working setup.
    Umask,
}

impl NewFileMode {
    /// The permission bits this module carries across a replacement.
    ///
    /// Deliberately not `0o7777`: setuid, setgid and the sticky bit are dropped
    /// rather than preserved, for the reasons in [`write_atomic_with`].
    #[cfg(unix)]
    const PERMISSION_BITS: u32 = 0o777;

    /// Readable and writable by the owner alone.
    ///
    /// Both the mode requested for a private file and the ceiling a temporary is
    /// held at while it is being filled.
    #[cfg(unix)]
    const OWNER_RW: u32 = 0o600;

    /// The mode `File::create` requests, before the kernel applies the umask.
    ///
    /// `0o666` rather than `0o644`, because this is the mode *requested* and the
    /// kernel masks it. Asking for `0o644` outright would ignore a umask of
    /// `0o002` that the user set precisely to get group-writable output.
    #[cfg(unix)]
    const CREATE_REQUEST: u32 = 0o666;

    /// The mode to pass to `open`, before the kernel applies the umask.
    #[cfg(unix)]
    const fn requested_bits(self) -> u32 {
        match self {
            Self::Umask => Self::CREATE_REQUEST,
            Self::OwnerOnly => Self::OWNER_RW,
        }
    }
}

/// Replace `path` with `contents`, atomically and durably.
///
/// See [`write_atomic_with`], which this is the byte-slice case of.
pub fn write_atomic(path: &Path, contents: &[u8], mode: NewFileMode) -> std::io::Result<()> {
    write_atomic_with(path, mode, |file| {
        // `Write` is implemented for `&File`, so the shared handle needs a
        // mutable binding of its own rather than a mutable file.
        let mut sink = file;
        sink.write_all(contents)
    })
}

/// Replace `path` with whatever `fill` writes to the temporary, atomically.
///
/// The temporary is created beside `path`, handed to `fill`, then flushed and
/// renamed over `path`. It is removed on every return path, so a failed write
/// leaves no partial file at `path` and no temporary beside it. Two caveats on
/// that: the Ctrl+C handler installed in `run()` ends the process with
/// `std::process::exit`, which runs no destructors, so interrupting a write can
/// leave one temporary behind; and the parent directories created below are not
/// removed if a later step fails.
///
/// `fill` must flush any buffered writer it wraps around the handle, and
/// propagate that flush's error. The handle is fsynced, not flushed, so a
/// `BufWriter` left to flush in its `Drop` discards the error and publishes a
/// truncated file: the outcome this function exists to prevent, reached by
/// another route.
///
/// Missing parent directories of `path` are created. Their own directory entries
/// are NOT made durable, though: the fsync below is of the directory the file
/// lands in, which says nothing about that directory's entry in its parent. So the
/// first ever write into a brand new tree can return success and still lose both
/// the file and the directories leading to it in a crash. Nothing that already
/// existed is destroyed and the write can simply be repeated, which is why this is
/// documented rather than fixed.
///
/// # What `path` resolves to
///
/// A rename replaces the *name* it is given, where writing a file follows it. Two
/// consequences are handled here so that routing an existing writer through this
/// function does not change where its bytes land:
///
/// - a symlink whose target exists is followed, and the target is replaced. A
///   *dangling* symlink is not followed, because resolving one means creating the
///   directories leading to a target that may not be the caller's own path;
///   `config::file::resolve_link` does that deliberately for the config file,
///   where the link is always one the user placed at their own config path.
/// - a target that exists and is not a regular file is written IN PLACE, with no
///   temporary and no rename. A device or a FIFO has no contents to replace,
///   `--output /dev/stdout` is a reasonable thing to ask for, and renaming over a
///   device node would destroy it. Not every such target can be opened for
///   writing: a directory gives EISDIR, and a Unix socket is refused with an errno
///   that differs by platform (ENXIO on Linux, EOPNOTSUPP on macOS). All of them
///   surface as the caller's write error, as they did for a plain write.
///   Note this branch has no temporary, so a failure part way through leaves the
///   node partly written; there is nothing else it could do.
///
/// # The mode of the published file
///
/// Replacing a file by rename gives the target the temporary's inode, so the
/// published file would take the temporary's mode rather than the mode it
/// replaced. The replaced file's mode is therefore applied to the temporary, which
/// is what stops a write silently narrowing a file the user or their umask had
/// widened. `mode` applies only when there is no file to take a mode from.
///
/// While `fill` runs, the temporary is no more permissive than owner-only, and
/// the published mode is applied afterwards. Both halves matter and in opposite
/// directions: writing at the final mode first would expose a config file's
/// contents to anyone the mode allows for the whole duration of the write, and
/// applying the final mode after the `fsync` would leave that metadata change
/// unflushed, so a crash could publish the file still at the private mode. The
/// narrowing half is best effort; see [`restrict_while_writing`] for why it has
/// to be.
///
/// Only the permission bits are carried across. setuid, setgid and the sticky bit
/// are dropped, because none of them is meaningful on a config file, a registry, a
/// clip or a species list, and a temporary is a different inode whose group comes
/// from its directory rather than from the file being replaced, so carrying such a
/// bit would mean preserving a permission whose meaning did not survive the copy.
/// That argument stands on its own; no claim is made here about how either platform
/// would respond to the attempt, because two previous attempts to state one were
/// wrong in opposite directions.
///
/// POSIX ACLs and extended attributes set on the replaced file are lost, because
/// the published file is a new inode; a default ACL on the containing directory is
/// inherited as it would be for any new file.
///
/// # Errors
///
/// Returns whatever `fill` returned, or the first I/O failure in the sequence:
/// creating the directory, creating the temporary, reading either mode, applying
/// the published one (narrowing cannot fail, by design), flushing
/// or renaming. `E` is the caller's error type so it can name the file it was
/// asked to write; the `From<std::io::Error>` bound is what lets this function's
/// own failures reach it.
pub fn write_atomic_with<E, F>(path: &Path, mode: NewFileMode, fill: F) -> Result<(), E>
where
    F: FnOnce(&File) -> Result<(), E>,
    E: From<std::io::Error>,
{
    let target = resolve_existing_link(path);
    let target = target.as_path();

    if let Some(irreplaceable) = open_in_place(target)? {
        // No temporary, so no atomicity: a device or a FIFO has no contents to
        // replace, and the caller asked for this exact node rather than for a
        // file at its path.
        return fill(&irreplaceable);
    }

    let dir = parent_dir(target);
    // Which ancestors are about to be created. `create_dir_all` fsyncs nothing, so
    // each directory it makes needs its own parent flushed below; otherwise a crash
    // right after a first write on a fresh install (the first `config set`, or the
    // first clip into a new output tree) can lose the directory entry along with
    // the file, having reported success. Recorded before the call because after it
    // they all exist.
    let created_dirs = missing_ancestors(dir);
    std::fs::create_dir_all(dir)?;

    // Worth knowing when the next line fails: creating the temporary needs write
    // and execute on the DIRECTORY, where writing the file in place needed only
    // the write bit on the file itself, so a file the user can plainly write can
    // now fail to be written. The `io::Error` names the temporary, and so the
    // directory with it.
    let temp = new_temp_in(dir, mode)?;

    // Both modes are read once, here, because `fill` runs in between and the
    // target may be replaced by the time it returns.
    let created = current_mode(temp.as_file())?;
    let published = existing_mode(target)?.unwrap_or(created);

    restrict_while_writing(temp.as_file(), created, published);

    fill(temp.as_file())?;

    apply_published_mode(temp.as_file(), published)?;

    // Untested, and untestable from here: deleting this line, or the directory
    // fsync below, or either `sync_parent_directory` call site, leaves the whole
    // suite green, because the difference only shows up across a crash. All of
    // them are reasoned, not covered.
    temp.as_file().sync_all()?;

    // Drops the temporary on failure, so a rejected write leaves nothing behind.
    temp.persist(target).map_err(|e| e.error)?;

    // The leaf directory holds the file's new entry.
    sync_directory(dir);
    // Each directory `create_dir_all` created holds a new entry in ITS parent, so
    // flush those too. The shallowest one's parent is the deepest pre-existing
    // ancestor, which is what actually makes the new tree survive a crash.
    for created in &created_dirs {
        sync_parent_directory(created);
    }

    Ok(())
}

/// The ancestors of `dir`, from deepest to shallowest, that do not exist yet.
///
/// Used to record what [`write_atomic_with`] is about to create so each new
/// directory can be made durable afterwards. Best effort by nature: a peer
/// creating one of these between this check and the create only means its own
/// durability is that peer's responsibility, not a correctness bug here.
fn missing_ancestors(dir: &Path) -> Vec<PathBuf> {
    let mut missing = Vec::new();
    for ancestor in dir.ancestors() {
        if ancestor.as_os_str().is_empty() || ancestor.exists() {
            break;
        }
        missing.push(ancestor.to_path_buf());
    }
    missing
}

/// `path` with a resolvable symlink followed, or `path` itself.
///
/// `canonicalize` requires every component including the last to exist, so it
/// fails for a dangling link and for a file that does not exist yet. Both fall
/// back to `path`, which is the right answer for the second and a deliberate
/// choice for the first; see [`write_atomic_with`].
fn resolve_existing_link(path: &Path) -> std::path::PathBuf {
    let is_link = std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_symlink());
    if !is_link {
        return path.to_path_buf();
    }

    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// An open handle to `target` when it exists and cannot be replaced by a rename.
///
/// `Ok(None)` means the ordinary case: `target` is absent, or is a regular file.
///
/// No `truncate`: only a regular file has a length to truncate, `O_TRUNC` on a
/// FIFO is undefined by POSIX, and passing it would mean a failed in-place write
/// could leave a regular file truncated if the target's type changed between the
/// `metadata` call and the `open`.
///
/// Not every non-regular target can actually be opened for writing. A directory
/// gives EISDIR, and a Unix socket is refused with a platform-dependent errno
/// (ENXIO on Linux, EOPNOTSUPP on macOS); both surface as the caller's write error,
/// which is what a plain write did too.
fn open_in_place(target: &Path) -> std::io::Result<Option<File>> {
    match std::fs::metadata(target) {
        Ok(meta) if !meta.is_file() => std::fs::OpenOptions::new()
            .write(true)
            .open(target)
            .map(Some),
        // Anything unstattable is treated as absent, which lands on the ordinary
        // path where creating the temporary reports the real reason.
        _ => Ok(None),
    }
}

/// The directory a file lives in.
///
/// `parent` is empty for a bare relative filename like `config.toml`, which is a
/// valid path to write to and not the same thing as having no parent.
fn parent_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

/// Create a uniquely named temporary in `dir` with the requested mode.
///
/// The uniqueness is `tempfile`'s random suffix rather than anything this module
/// computes, and it is load-bearing rather than incidental: two concurrent birda
/// processes write into the same config directory, and clip extraction and the
/// species-list export both write into a directory the user may be running a
/// second birda against. With a shared temporary name their writes interleave and
/// the loser's rename fails with ENOENT, losing its file entirely.
#[cfg(unix)]
fn new_temp_in(dir: &Path, mode: NewFileMode) -> std::io::Result<tempfile::NamedTempFile> {
    use std::os::unix::fs::PermissionsExt;

    // `permissions` reaches `open`'s mode argument rather than a later `chmod`,
    // so the kernel still applies the umask and `Umask` reproduces exactly what
    // `File::create` would have produced.
    tempfile::Builder::new()
        .permissions(std::fs::Permissions::from_mode(mode.requested_bits()))
        .tempfile_in(dir)
}

/// Create a uniquely named temporary in `dir`, on platforms without file modes.
#[cfg(not(unix))]
fn new_temp_in(dir: &Path, _mode: NewFileMode) -> std::io::Result<tempfile::NamedTempFile> {
    tempfile::Builder::new().tempfile_in(dir)
}

/// The permission bits `file` currently carries.
#[cfg(unix)]
fn current_mode(file: &File) -> std::io::Result<u32> {
    use std::os::unix::fs::PermissionsExt;

    Ok(file.metadata()?.permissions().mode() & NewFileMode::PERMISSION_BITS)
}

/// The permission bits of the file at `target`, if there is one.
///
/// `0o777`, so setuid, setgid and the sticky bit are dropped rather than carried.
/// See [`write_atomic_with`] for why that is deliberate.
///
/// A `NotFound` target is the ordinary case of creating a file for the first
/// time. Any other `metadata` failure is propagated rather than folded into it:
/// with two mode policies, guessing "there was nothing there" can silently widen
/// a file the user had restricted, which is the direction that matters. A symlink
/// cycle at `target` therefore surfaces as ELOOP, which is what a plain write did
/// too.
#[cfg(unix)]
fn existing_mode(target: &Path) -> std::io::Result<Option<u32>> {
    use std::os::unix::fs::PermissionsExt;

    match std::fs::metadata(target) {
        Ok(existing) => Ok(Some(
            existing.permissions().mode() & NewFileMode::PERMISSION_BITS,
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Narrow the temporary to owner-only while `fill` writes into it.
///
/// Narrowing only, never widening: the published mode is applied afterwards, so
/// the contents are never readable by anyone the final file would not admit, and
/// a published mode *narrower* than owner-only is left alone. The handle is
/// already open, so restricting it cannot stop `fill` writing.
///
/// Best effort, and that is the important part. **Do not make this fallible.**
/// Linux vfat and exFAT reject a `chmod` whose bits differ from the ones the mount
/// options dictate, and those options routinely give a temporary a mode wider than
/// owner-only, so this call is attempted and rejected rather than skipped.
/// Propagating that failed clip extraction to an SD card entirely, which is how a
/// field recorder's card is formatted. macOS msdosfs ignores the same call
/// silently, measured, so no test on a Mac can reproduce it.
///
/// Ignoring the failure costs nothing that matters: this is hardening rather than
/// correctness. The published mode is applied afterwards either way by
/// [`apply_published_mode`], which does propagate, so a swallowed `chmod` can only
/// widen the exposure window *during* `fill`, never the published file. On the very
/// filesystems that reject it, the audience during the write is the audience the
/// final mode admits anyway, because the temporary and the file it replaces both
/// take their mode from the mount.
///
/// The skip below fires when the temporary is already no wider, which is always the
/// ordinary config write (created 0o600, published 0o600) and is also a umask-mode
/// write under a umask that masks the group and other bits away. It is a umask-mode
/// write under a permissive umask that reaches the `chmod`, which is the case the
/// paragraph above is about.
#[cfg(unix)]
fn restrict_while_writing(temp: &File, created: u32, published: u32) {
    let while_writing = published & NewFileMode::OWNER_RW;
    if created & !while_writing == 0 {
        return;
    }

    drop(apply_mode(temp, while_writing));
}

/// Give the temporary the mode its published form must have.
///
/// Skipped when it already has it, which is what keeps this from failing on a
/// filesystem that refuses `chmod`: there, the temporary and the file it replaces
/// both take their mode from the mount, so they already agree.
#[cfg(unix)]
fn apply_published_mode(temp: &File, published: u32) -> std::io::Result<()> {
    if current_mode(temp)? == published {
        return Ok(());
    }

    apply_mode(temp, published)
}

/// Set the temporary's permission bits.
#[cfg(unix)]
fn apply_mode(temp: &File, bits: u32) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    temp.set_permissions(std::fs::Permissions::from_mode(bits))
}

/// Platforms without Unix permission bits have nothing to read.
///
/// Windows carries an ACL rather than a mode, and `MoveFileEx` does NOT carry
/// the replaced file's security descriptor across; the temporary's own ACL
/// survives the move. So the hazard exists here too and is simply not handled:
/// an explicit ACL set on the target is lost when it is replaced.
///
/// Accepted rather than fixed, because the temporary is created in the target's
/// own directory and inherits that directory's inheritable ACEs, which is
/// exactly what a file freshly created there would get. That is the right answer
/// for a per-user config directory or an output directory, and the wrong one
/// only for a file someone has deliberately re-ACLed, which would need
/// `ReplaceFile` rather than `MoveFileEx`.
#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
fn current_mode(_file: &File) -> std::io::Result<u32> {
    Ok(0)
}

/// Platforms without Unix permission bits have nothing to read.
#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
fn existing_mode(_target: &Path) -> std::io::Result<Option<u32>> {
    Ok(None)
}

/// No-op on platforms without Unix permission bits.
#[cfg(not(unix))]
fn restrict_while_writing(_temp: &File, _created: u32, _published: u32) {}

/// No-op on platforms without Unix permission bits.
#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
fn apply_published_mode(_temp: &File, _published: u32) -> std::io::Result<()> {
    Ok(())
}

/// Flush the directory entry a rename onto `path` has just created.
///
/// Call this after any `rename` that publishes a file, not only the ones this
/// module performs: a rename is atomic with respect to a reader, but the *record*
/// of it can still be lost in a crash, and the old name comes back.
///
/// What that costs depends entirely on whether the file's own data was fsynced
/// first, which is why this is only ever half of the job:
///
/// - **With** the file fsync, losing the rename costs only work. A downloaded
///   model reverts to absent, `is_installed` correctly reports it missing, and
///   the next run downloads it again. Annoying for a large model, never wrong.
/// - **Without** it, the rename can reach the disk while the data behind it has
///   not, and the published name points at a file of zeros. That is worse than
///   losing the rename, so a caller that renames a file it never fsynced is
///   better off calling neither than calling only this.
///
/// Failures are deliberately ignored, and the direction of the tradeoff is why:
/// the rename has already happened, and reporting a durability failure for a
/// write that completed would cost the caller a result to buy durability that
/// was never required for correctness. Some filesystems also reject `fsync` on
/// a directory outright, so the failure is expected rather than exceptional.
pub fn sync_parent_directory(path: &Path) {
    sync_directory(parent_dir(path));
}

/// Flush a directory's own entries to disk, best effort.
#[cfg(unix)]
fn sync_directory(dir: &Path) {
    if let Ok(handle) = File::open(dir) {
        drop(handle.sync_all());
    }
}

/// No-op on platforms where a directory is not an openable file.
#[cfg(not(unix))]
fn sync_directory(_dir: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every entry in `dir` that the caller did not expect to be there.
    ///
    /// A leaked temporary sits next to the real file, which is both litter and,
    /// on a failure path, evidence that a partially written file survived.
    /// Stated as "anything unexpected" rather than "anything matching the temp
    /// prefix", so it still catches a leak if `tempfile` changes how it names
    /// them.
    fn strays_in(dir: &Path, expected: &[&str]) -> Vec<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| !expected.contains(&name.as_str()))
            .collect()
    }

    #[test]
    fn test_write_atomic_creates_the_file_and_its_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("deeper").join("file.txt");

        write_atomic(&path, b"contents", NewFileMode::OwnerOnly).unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"contents");
    }

    #[test]
    fn test_missing_ancestors_lists_only_the_directories_to_be_created() {
        let root = tempfile::tempdir().unwrap();
        // root exists; a/b/c does not.
        let target = root.path().join("a").join("b").join("c");

        let missing = missing_ancestors(&target);

        assert_eq!(
            missing,
            vec![
                root.path().join("a").join("b").join("c"),
                root.path().join("a").join("b"),
                root.path().join("a"),
            ],
            "deepest-to-shallowest, stopping at the first existing ancestor"
        );
    }

    #[test]
    fn test_missing_ancestors_is_empty_when_the_directory_exists() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            missing_ancestors(dir.path()).is_empty(),
            "an existing directory needs no creation and so no parent flush"
        );
    }

    #[test]
    fn test_write_atomic_replaces_existing_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");

        write_atomic(&path, b"first", NewFileMode::OwnerOnly).unwrap();
        write_atomic(&path, b"second", NewFileMode::OwnerOnly).unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"second");
    }

    #[test]
    fn test_parent_dir_of_a_bare_relative_filename_is_the_current_directory() {
        // `Path::parent` is `Some("")` for a bare filename, not `None`, so a
        // naive `parent().unwrap_or(".")` would try to create a directory named
        // "" and fail.
        //
        // Asserted on `parent_dir` directly rather than by writing through a
        // relative path, because a relative path resolves against the process
        // working directory, which is shared by all 450-odd tests in this binary,
        // and no test here calls `set_current_dir`. Named for what it checks, so
        // it does not read as an end-to-end guarantee it cannot give.
        assert_eq!(parent_dir(Path::new("bare.txt")), Path::new("."));
        assert_eq!(parent_dir(Path::new("dir/bare.txt")), Path::new("dir"));
    }

    #[test]
    #[cfg(unix)]
    fn test_write_atomic_does_not_truncate_the_target_in_place() {
        // A hardlink is a second name for the same inode. Truncating and
        // rewriting in place would show the new contents through the link; a
        // write to a temporary followed by a rename gives the path a *different*
        // inode, leaving the old one intact behind the link. Reading the old
        // contents back through the link is therefore proof the target was never
        // truncated.
        //
        // Named for what it proves rather than for atomicity, because it proves
        // less than that: it shows the path acquired a new inode, not that a
        // reader had an uninterrupted view throughout. Unlinking the target
        // immediately before the rename, which opens a window where the file
        // does not exist at all, leaves this green.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");

        write_atomic(&path, b"first", NewFileMode::OwnerOnly).unwrap();
        let link = dir.path().join("link.txt");
        std::fs::hard_link(&path, &link).unwrap();

        write_atomic(&path, b"second", NewFileMode::OwnerOnly).unwrap();

        assert_eq!(
            std::fs::read(&link).unwrap(),
            b"first",
            "the previous file must survive behind its own name; seeing 'second' \
             here means the target was truncated in place rather than replaced"
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"second");
    }

    #[test]
    fn test_a_successful_write_leaves_no_temporary_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");

        write_atomic(&path, b"contents", NewFileMode::OwnerOnly).unwrap();

        let strays = strays_in(dir.path(), &["file.txt"]);
        assert!(
            strays.is_empty(),
            "a successful write must not litter the directory, found: {strays:?}"
        );
    }

    #[test]
    fn test_a_failed_fill_leaves_the_directory_as_it_was() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");

        let result: Result<(), std::io::Error> =
            write_atomic_with(&path, NewFileMode::OwnerOnly, |_| {
                Err(std::io::Error::other("no"))
            });

        assert!(result.is_err());
        assert!(!path.exists(), "a failed write must create no file");
        let strays = strays_in(dir.path(), &[]);
        assert!(
            strays.is_empty(),
            "a failed write must leave no partial file behind, found: {strays:?}"
        );
    }

    #[test]
    fn test_a_failed_fill_leaves_an_existing_target_untouched() {
        // The regression that matters most for the config file and the registry:
        // a write that gives up halfway must not have destroyed what was there.
        //
        // The fill returns `FillError::Refused`, which the helper's own failures
        // cannot produce, so the assertions below cannot be satisfied by the fill
        // never running. Without that, a pre-flight guard rejecting the path
        // before the fill would leave this green while proving nothing about a
        // partial write.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        write_atomic(&path, b"original", NewFileMode::OwnerOnly).unwrap();

        let result: Result<(), FillError> =
            write_atomic_with(&path, NewFileMode::OwnerOnly, |file| {
                let mut sink = file;
                sink.write_all(b"partial")?;
                Err(FillError::Refused)
            });

        assert_eq!(
            result,
            Err(FillError::Refused),
            "the fill must have run and its own error must be what came back"
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"original",
            "the existing file must survive a failed write untouched"
        );
    }

    /// An error type that is not an `io::Error`, for the pass-through test.
    #[derive(Debug, PartialEq, Eq)]
    enum FillError {
        /// Returned by a fill closure that fails on its own terms.
        Refused,
        /// Converted from the helper's own I/O failures.
        Io,
    }

    impl From<std::io::Error> for FillError {
        fn from(_: std::io::Error) -> Self {
            Self::Io
        }
    }

    #[test]
    fn test_the_fill_error_reaches_the_caller_unchanged() {
        // The reason `write_atomic_with` is generic over the fill's error rather
        // than taking an `io::Result` closure. The clip writer's fill returns
        // `hound::Error`, which is not an I/O error at all for a bad sample, and
        // forcing it through `io::Error::other` would nest it inside a variant
        // that claims otherwise.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");

        let result: Result<(), FillError> =
            write_atomic_with(&path, NewFileMode::OwnerOnly, |_| Err(FillError::Refused));

        assert_eq!(result, Err(FillError::Refused));
    }

    #[test]
    #[cfg(unix)]
    fn test_a_new_file_gets_the_requested_mode() {
        // The half of the mode rule that `NewFileMode` exists for. A clip must
        // come out as `File::create` would have left it, because a clip
        // directory can be served or read by another account; a config file must
        // not, because it is per-user.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let private = dir.path().join("private.txt");
        let shared = dir.path().join("shared.txt");

        write_atomic(&private, b"x", NewFileMode::OwnerOnly).unwrap();
        write_atomic(&shared, b"x", NewFileMode::Umask).unwrap();

        let mode_of = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode_of(&private) & 0o077,
            0,
            "an OwnerOnly file must not be group or world accessible"
        );

        // Compared against a file `File::create` made in the same directory
        // rather than against the literal 0o644, because the umask this test
        // runs under is not knowable from here. Asserting a literal would fail
        // for anyone with a umask of 0o077 or 0o002.
        let reference = dir.path().join("reference.txt");
        drop(File::create(&reference).unwrap());
        assert_eq!(
            mode_of(&shared),
            mode_of(&reference),
            "a Umask file must match what File::create would have produced"
        );

        // The positive control, and the reason this test is not the whole story.
        // Under a umask of 0o077 the kernel masks 0o666 and 0o600 to the same
        // 0o600, so the two policies become indistinguishable and every assertion
        // above holds no matter which one production passes. Measured: setting
        // `Umask => 0o600` in `requested_bits` leaves this test green under that
        // umask. Say so out loud rather than reporting a pass that proves nothing.
        if mode_of(&reference) & 0o066 == 0 {
            eprintln!(
                "skipped the OwnerOnly-vs-Umask distinction: this umask masks both \
                 policies to {:o}, so a wrong policy would not be detected here",
                mode_of(&reference)
            );
            return;
        }
        assert_ne!(
            mode_of(&shared),
            mode_of(&private),
            "the two policies must produce different modes, or this test cannot \
             tell a caller passing the wrong one"
        );
    }

    #[test]
    fn test_concurrent_writers_to_one_path_all_succeed() {
        // The unique temporary name is documented as load-bearing, so it needs a
        // test rather than a promise. With a fixed name the writers share one
        // temporary, their writes interleave, and the loser's rename fails with
        // ENOENT, losing its file entirely.
        //
        // Each thread writes DISTINCT contents, because identical payloads cannot
        // detect shared state: interleaved writes of the same bytes still produce
        // those bytes.
        const WRITERS: usize = 8;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("contended.txt");

        let payloads: Vec<String> = (0..WRITERS).map(|i| format!("writer-{i}")).collect();

        std::thread::scope(|scope| {
            for payload in &payloads {
                let path = path.as_path();
                scope.spawn(move || {
                    write_atomic(path, payload.as_bytes(), NewFileMode::OwnerOnly)
                        .expect("a contended write must not fail");
                });
            }
        });

        // The winner is whichever renamed last, which is deliberately not asserted.
        // What must hold is that the file is exactly one writer's payload and not a
        // blend of several, and that nobody left a temporary behind.
        let published = std::fs::read_to_string(&path).unwrap();
        assert!(
            payloads.contains(&published),
            "the published file must be exactly one writer's payload, got {published:?}"
        );
        let strays = strays_in(dir.path(), &["contended.txt"]);
        assert!(
            strays.is_empty(),
            "no writer may leave a temporary behind, found: {strays:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_a_symlink_whose_target_exists_is_written_through() {
        // `fs::write` and `File::create` both follow a symlink; a rename replaces
        // it. Three of the callers routed through this helper were converted from
        // one of those, so following an existing link is what stops the change
        // silently redirecting their writes to a new local file and leaving the
        // real one stale.
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.txt");
        let link = dir.path().join("link.txt");
        write_atomic(&real, b"first", NewFileMode::OwnerOnly).unwrap();
        std::os::unix::fs::symlink(&real, &link).unwrap();

        write_atomic(&link, b"second", NewFileMode::OwnerOnly).unwrap();

        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the link must survive the write rather than be replaced by a file"
        );
        assert_eq!(
            std::fs::read(&real).unwrap(),
            b"second",
            "the write must land on the file the link points at"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_a_target_that_is_not_a_regular_file_is_written_in_place() {
        // `--output /dev/stdout` is a reasonable thing to ask for and worked
        // before these paths were made atomic. A device or FIFO has no contents to
        // replace, cannot be atomically replaced, and renaming over one destroys
        // it: as root that would turn /dev/null into a regular file.
        //
        // Uses a FIFO rather than a device node, because creating a device needs
        // root while `mkfifo` does not, and both take the same branch.
        //
        // The read end is opened NON-BLOCKING and up front, by this thread, rather
        // than by a reader thread that blocks in `open`. That is not tidiness: with
        // a blocking reader, removing the guard under test means no writer ever
        // opens the FIFO, so the reader never returns and the test WEDGES instead
        // of failing. A mutation that can only hang is not evidence, and it turns
        // a future regression into a CI timeout. Here the guard's absence shows up
        // as an empty read plus a target that stopped being a FIFO, both of which
        // are ordinary failed assertions.
        //
        // KEEP THE PAYLOAD SMALL. The write end is blocking and nothing drains the
        // pipe while the write is in progress, so a payload larger than the pipe's
        // capacity would block the writer and reintroduce exactly the wedge the
        // paragraph above is about. Sixteen bytes is far below the capacity of every
        // platform this builds for.
        use std::io::Read;
        use std::os::unix::fs::{FileTypeExt, OpenOptionsExt};

        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("sink.fifo");

        // `mkfifo` is in coreutils, so it is present on every developer machine and
        // on the CI image, but not in a minimal container. Skip rather than fail
        // there: a missing shell utility says nothing about whether this code is
        // correct, and a red suite for that reason trains people to ignore red.
        // Distinguished from `mkfifo` running and FAILING, which is a real problem
        // and still panics.
        let Ok(status) = std::process::Command::new("mkfifo").arg(&fifo).status() else {
            eprintln!("skipped: mkfifo is unavailable, the in-place write path is not exercised");
            return;
        };
        assert!(status.success(), "mkfifo failed");

        // O_NONBLOCK so this open returns immediately with no writer attached, and
        // so the read below cannot block either.
        let mut read_end = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&fifo)
            .expect("opening a FIFO read end non-blocking must not block");

        write_atomic(&fifo, b"through the pipe", NewFileMode::Umask).unwrap();

        let mut received = Vec::new();
        // A non-blocking read on an empty FIFO whose writer has closed returns 0
        // rather than EAGAIN, so a missing writer reads as empty rather than
        // erroring. Either way this returns.
        drop(read_end.read_to_end(&mut received));
        assert_eq!(
            String::from_utf8_lossy(&received),
            "through the pipe",
            "the write must have gone through the FIFO rather than around it"
        );

        assert!(
            std::fs::metadata(&fifo).unwrap().file_type().is_fifo(),
            "the FIFO must still be a FIFO, not a regular file that replaced it"
        );
        let strays = strays_in(dir.path(), &["sink.fifo"]);
        assert!(
            strays.is_empty(),
            "the in-place path must create no temporary, found: {strays:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_the_temporary_is_never_wider_than_the_file_it_replaces() {
        // Observed from INSIDE the fill closure, because that is the whole
        // duration of the exposure and it is invisible from outside: by the time
        // the write returns, the published mode has been applied and the
        // temporary is gone.
        //
        // The case that bites is a `Umask` caller replacing a file the user
        // deliberately restricted: the temporary would be created at the umask
        // default and only narrowed afterwards, so the new contents would sit in a
        // world-readable file for as long as the write took. A clip can be several
        // megabytes.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("restricted.txt");
        write_atomic(&path, b"first", NewFileMode::Umask).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        // The temporary is created at the umask-masked `Umask` mode, so under a
        // umask that already masks the group and world bits away it starts at
        // 0o600 and the narrowing step has nothing to do: deleting that step would
        // leave this green. Say so rather than reporting a pass that proves
        // nothing. Checked against a reference file, since the umask is not
        // readable from safe Rust.
        let reference = dir.path().join("reference");
        drop(File::create(&reference).unwrap());
        let reference_mode = std::fs::metadata(&reference).unwrap().permissions().mode();
        if reference_mode & 0o066 == 0 {
            eprintln!(
                "skipped: this umask creates temporaries at {:o} already, so the \
                 narrowing step is not exercised",
                reference_mode & 0o777
            );
            return;
        }

        let observed = std::cell::Cell::new(0o777);
        write_atomic_with(&path, NewFileMode::Umask, |file| {
            observed.set(file.metadata()?.permissions().mode() & 0o777);
            let mut sink = file;
            sink.write_all(b"second")
        })
        .unwrap();

        assert_eq!(
            observed.get() & 0o077,
            0,
            "while the contents were being written the temporary was mode {:o}, \
             wider than the 0o600 file it replaces",
            observed.get()
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600,
            "the published file must still carry the mode it had"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_replacing_a_file_keeps_the_mode_it_had() {
        // The other half, and the one that has to hold for every caller rather
        // than per policy: publishing by rename hands the target the
        // temporary's inode, so without this a rewrite would narrow a file the
        // user or their umask had widened. `NewFileMode` is deliberately the
        // *wrong* answer here, to prove the existing mode wins over it.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");

        write_atomic(&path, b"first", NewFileMode::OwnerOnly).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();

        write_atomic(&path, b"second", NewFileMode::OwnerOnly).unwrap();

        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640,
            "a rewrite must keep the mode the file already had"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_write_atomic_works_when_the_target_is_not_on_the_temp_filesystem() {
        // Pins the invariant every other test here is blind to: the temporary
        // must be created in the TARGET's directory, not in $TMPDIR. Rename is
        // only atomic within a filesystem and fails outright with EXDEV across
        // one, so a temporary in $TMPDIR breaks for the ordinary desktop layout
        // of /tmp on tmpfs and ~/.config on disk.
        //
        // Every other test here writes into `tempfile::tempdir()`, which is
        // itself under $TMPDIR, so `persist` never crosses a device and swapping
        // `tempfile_in(dir)` for `tempfile()` would leave them all green.
        //
        // Two environments make this inert: /dev/shm missing, and $TMPDIR
        // pointed at /dev/shm so both ends share a device. It skips rather than
        // fails in both, because a red build over an environment quirk is worse
        // than a test that stops proving something, and `TMPDIR=/dev/shm` is a
        // real technique for speeding up tempfile-heavy suites.
        use std::os::unix::fs::MetadataExt;

        let Ok(dir) = tempfile::tempdir_in(Path::new("/dev/shm")) else {
            eprintln!("skipped: /dev/shm is unusable, cross-filesystem write not exercised");
            return;
        };

        let device_of = |p: &Path| std::fs::metadata(p).map(|m| m.dev()).ok();
        if device_of(dir.path()) == device_of(&std::env::temp_dir()) {
            eprintln!(
                "skipped: /dev/shm and $TMPDIR share a device, cross-filesystem write not exercised"
            );
            return;
        }

        let path = dir.path().join("file.txt");
        write_atomic(&path, b"contents", NewFileMode::OwnerOnly)
            .expect("a write must not depend on $TMPDIR sharing a filesystem");

        assert_eq!(std::fs::read(&path).unwrap(), b"contents");
    }
}
