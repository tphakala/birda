//! Durable file writes.
//!
//! Four places in the tree replace a file the user would miss if it were lost
//! halfway through: the config file, the model registry, a downloaded model and
//! an extracted clip. They all want the same sequence, which is why it lives
//! here rather than being derived a fourth time.
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
//! What this does not do is serialise concurrent writers. Every caller is a
//! lock-free load-mutate-save, so two overlapping saves still lose one of the
//! two edits. The rename makes each write whole, not the pair of them ordered.

use std::fs::File;
use std::io::Write;
use std::path::Path;

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
    /// The mode to pass to `open`, before the kernel applies the umask.
    #[cfg(unix)]
    const fn requested_bits(self) -> u32 {
        match self {
            // 0o666 rather than 0o644: this is the mode *requested*, which the
            // kernel then masks, and it is what `File::create` asks for. Asking
            // for 0o644 outright would ignore a umask of 0o002 that the user set
            // precisely to get group-writable output.
            Self::Umask => 0o666,
            Self::OwnerOnly => 0o600,
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
/// leaves neither a partial file at `path` nor litter beside it. Not on every
/// path out, though: the Ctrl+C handler installed in `run()` ends the process
/// with `std::process::exit`, which runs no destructors, so interrupting a write
/// can leave one temporary behind.
///
/// Missing parent directories of `path` are created.
///
/// # The mode of the published file
///
/// Replacing a file by rename gives the target the temporary's inode, so the
/// published file would take the temporary's mode rather than the mode it
/// replaced. An existing target's mode is therefore copied onto the temporary
/// first, which is what stops a write silently narrowing a file the user or
/// their umask had widened. `mode` applies only when there is nothing to copy.
///
/// The copy happens before the flush, not after. `fsync` persists the inode's
/// metadata as well as its data, so widening the permissions afterwards would
/// leave that one change unflushed, and a crash could publish the file still at
/// the temporary's private mode.
///
/// # Errors
///
/// Returns whatever `fill` returned, or the first I/O failure among creating the
/// directory, creating the temporary, flushing it and renaming it. `E` is the
/// caller's error type so it can name the file it was asked to write; the
/// `From<std::io::Error>` bound is what lets this function's own failures reach
/// it.
pub fn write_atomic_with<E, F>(path: &Path, mode: NewFileMode, fill: F) -> Result<(), E>
where
    F: FnOnce(&File) -> Result<(), E>,
    E: From<std::io::Error>,
{
    let dir = parent_dir(path);
    std::fs::create_dir_all(dir)?;

    // Worth knowing when the next line fails: creating the temporary needs write
    // and execute on the DIRECTORY, which writing the file in place never did,
    // so a file the user can plainly write can now fail to be written. The
    // `io::Error` names the temporary, and so the directory with it.
    let temp = new_temp_in(dir, mode)?;

    fill(temp.as_file())?;

    copy_existing_mode(path, temp.as_file())?;

    // Untested, and untestable from here: deleting this line, or the directory
    // fsync below, leaves the whole suite green, because the difference only
    // shows up across a crash. Both are reasoned, not covered.
    temp.as_file().sync_all()?;

    // Drops the temporary on failure, so a rejected write leaves nothing behind.
    temp.persist(path).map_err(|e| e.error)?;

    sync_directory(dir);

    Ok(())
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
/// The name has to be unique rather than fixed: clip extraction writes many
/// files into one directory and two concurrent processes write into the user's
/// config directory, and with a shared temporary name their writes interleave
/// and the loser's rename fails with ENOENT, losing its file entirely.
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

/// Give the temporary the mode of the file it is about to replace.
///
/// A missing target is the ordinary case of creating a file for the first time
/// and leaves the temporary's own mode in place. An unreadable one is treated
/// the same way rather than failing the write: refusing to write a file whose
/// old permissions could not be read would trade a lost setting for a cosmetic
/// difference.
#[cfg(unix)]
fn copy_existing_mode(target: &Path, temp: &File) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let Ok(existing) = std::fs::metadata(target) else {
        return Ok(());
    };

    // `0o7777`, not `0o777`: the mask exists to strip the file-type bits, and a
    // narrower one would silently drop setuid, setgid and the sticky bit from a
    // mode the user chose. None of the three is meaningful on the files written
    // here, which is the point: dropping them would be an unannounced change to
    // something this function claims to preserve.
    let bits = existing.permissions().mode() & 0o7777;
    temp.set_permissions(std::fs::Permissions::from_mode(bits))
}

/// No-op on platforms without Unix permission bits.
///
/// Windows carries an ACL rather than a mode, and `MoveFileEx` does NOT carry
/// the replaced file's security descriptor across; the temporary's own ACL
/// survives the move. So the hazard above exists here too and is simply not
/// handled: an explicit ACL set on the target is lost when it is replaced.
///
/// Accepted rather than fixed, because the temporary is created in the target's
/// own directory and inherits that directory's inheritable ACEs, which is
/// exactly what a file freshly created there would get. That is the right answer
/// for a per-user config directory or an output directory, and the wrong one
/// only for a file someone has deliberately re-ACLed, which would need
/// `ReplaceFile` rather than `MoveFileEx`.
#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
fn copy_existing_mode(_target: &Path, _temp: &File) -> std::io::Result<()> {
    Ok(())
}

/// Flush the directory entry a rename onto `path` has just created.
///
/// Call this after any `rename` that publishes a file, not only the ones this
/// module performs: a rename is atomic with respect to a reader, but the
/// *record* of it can still be lost in a crash. For a config file that means
/// the old contents come back, which is survivable. For a downloaded model it
/// means a directory entry pointing at nothing, while the existence check that
/// decides whether to download it again has already been satisfied.
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
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
    fn test_write_atomic_replaces_existing_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");

        write_atomic(&path, b"first", NewFileMode::OwnerOnly).unwrap();
        write_atomic(&path, b"second", NewFileMode::OwnerOnly).unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"second");
    }

    #[test]
    fn test_write_atomic_accepts_a_bare_relative_filename() {
        // `Path::parent` is `Some("")` for a bare filename, not `None`, so a
        // naive `parent().unwrap_or(".")` would try to create a directory named
        // "" and fail. Run inside a temporary directory rather than the crate
        // root, since the test writes a real file.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bare.txt");
        let relative = Path::new("bare.txt");

        // `write_atomic` resolves relative paths against the process working
        // directory, which is shared by every test in the binary, so the call is
        // made with an absolute path and the *parent* handling is asserted
        // directly instead of by chdir-ing.
        assert_eq!(parent_dir(relative), Path::new("."));
        write_atomic(&path, b"ok", NewFileMode::OwnerOnly).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"ok");
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
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        write_atomic(&path, b"original", NewFileMode::OwnerOnly).unwrap();

        let result: Result<(), std::io::Error> =
            write_atomic_with(&path, NewFileMode::OwnerOnly, |file| {
                let mut sink = file;
                sink.write_all(b"partial")?;
                Err(std::io::Error::other("gave up after writing"))
            });

        assert!(result.is_err());
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
