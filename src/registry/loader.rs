//! Registry loading and bootstrapping.

use super::types::{ModelEntry, Registry};
use crate::error::{Error, Result};
use std::path::PathBuf;

/// Load registry from user config or bundled default.
///
/// Three outcomes, because the on-disk copy is a cache that can fail in two
/// materially different ways:
///
/// - It is absent, older than the bundled registry, or not parseable by this
///   binary: the bundled registry is written over it and returned.
/// - It could not be READ (a permission or I/O failure, which says nothing about
///   the contents): the bundled registry is returned and the file is left exactly
///   as it is, so an intact registry survives a transient failure. Note the
///   returned registry is then NOT what is on disk, and no version upgrade is
///   persisted until the file becomes readable again.
/// - Otherwise the user's copy is returned as is, including when it is NEWER than
///   the bundled one.
pub fn load_registry() -> Result<Registry> {
    let registry_path = registry_file_path()?;
    let bundled_registry = load_bundled_registry()?;

    Ok(load_registry_from(&registry_path, bundled_registry))
}

/// Resolve the registry to use, given where the user's copy lives.
///
/// Split out from [`load_registry`] so it can be tested against a temporary
/// directory. [`load_registry`] resolves its path through
/// `crate::config::config_dir()`, so driving it directly in a test reads and
/// writes the developer's real config directory.
///
/// Infallible on purpose: every failure below is already a degradation to the
/// bundled registry, which is in hand before this is called, so there is nothing
/// left that could justify aborting the command that asked for it.
fn load_registry_from(registry_path: &std::path::Path, bundled_registry: Registry) -> Registry {
    // `try_exists` rather than `exists`, which folds every stat failure into
    // "absent" and would hand an intact registry to the branch that REPLACES it.
    // That is the same two-way collapse the error arms below exist to undo, three
    // lines earlier in the same function, so it gets the same three outcomes:
    // present, definitively absent, or could not determine and therefore do not
    // act.
    //
    // Only the third outcome changes. `Ok(true)` and `Ok(false)` behave exactly as
    // before, including for a dangling symlink, which reports `Ok(false)` and is
    // still bootstrapped over (the pre-existing behaviour `write_registry_file`
    // documents). The reachable permission case was already self-protecting: an
    // EACCES that breaks the stat breaks the subsequent write too, so the old code
    // was saved by the write failing rather than by the check. What it did not
    // cover is a failure that breaks stat on the inode but not create-and-rename in
    // its directory, such as an ESTALE after an NFS server restart.
    match registry_path.try_exists() {
        Ok(true) => {}
        Ok(false) => return bootstrap_registry(registry_path, &bundled_registry),
        Err(e) => {
            tracing::warn!(
                "Could not determine whether the model registry at {} exists: {e}. \
                 Continuing with the bundled registry; nothing on disk is touched.",
                registry_path.display()
            );
            return bundled_registry;
        }
    }

    let user_registry = match load_from_file(registry_path) {
        Ok(registry) => registry,
        // The bytes on disk are not a registry this binary can use, so replacing
        // them with the bundled copy is the repair rather than a loss. This is the
        // behaviour that has always been here.
        //
        // "Not usable by this binary" rather than "definitively corrupt", because
        // this arm is wider than malformed JSON. Nothing here sets
        // `deny_unknown_fields`, so a registry from a NEWER birda that ADDED a
        // field parses fine and is kept; what lands here is one that REMOVED or
        // renamed a field this binary still requires, on a downgrade. That is not
        // corruption either, and it is replaced all the same. For where bytes in
        // another text encoding land, and the one case that escapes this arm, see
        // `load_from_file`.
        Err(e @ Error::RegistryParse { .. }) => {
            tracing::warn!(
                "{e}{}. Replacing it with the bundled registry.",
                cause_of(&e)
            );
            persist_registry(registry_path, &bundled_registry);
            return bundled_registry;
        }
        // Could not be READ, which says nothing about the contents: EACCES, EIO,
        // EMFILE, a flaky network mount, fd exhaustion during a parallel batch.
        // The file is very possibly intact, so it is used from the bundled copy
        // in memory and left untouched on disk. Overwriting here was "could not
        // determine, so assume the destructive answer", and it silently discarded
        // hand-maintained registries that birda-gui also reads.
        //
        // The destructive branch is the one that has to name its variant, so an
        // error variant added to `load_from_file` later defaults to this arm.
        Err(e) => {
            tracing::warn!(
                "{e}{}. Continuing with the bundled registry; the file on disk is \
                 left as it is and will be read again on the next run.",
                cause_of(&e)
            );
            return bundled_registry;
        }
    };

    // Compare versions - if bundled is newer, replace user's registry
    if bundled_registry.registry_version > user_registry.registry_version {
        tracing::info!(
            "Updating registry from version {} to {}",
            user_registry.registry_version,
            bundled_registry.registry_version
        );
        persist_registry(registry_path, &bundled_registry);
        bundled_registry
    } else {
        user_registry
    }
}

/// Write the registry to disk, warning rather than failing if it cannot be.
///
/// Persisting the registry is a cache side effect: `load_registry` already holds a
/// usable one in memory by the time this runs, so a failure to write it must not
/// abort the command that asked for it. Propagating it took down `species` and
/// every `models` subcommand, for a file the caller no longer needs. (`analyze`
/// already survived, because it matches on the error and continues without range
/// filtering, which is a different degradation and a worse one than not writing a
/// cache.)
///
/// Not a hypothetical. Replacing the file by rename needs write and execute on
/// the DIRECTORY, where the previous plain write needed only the write bit on the
/// file itself, so two ordinary layouts started failing here: a read-only config
/// directory holding a writable registry.json, and a registry.json bind-mounted
/// into a container as a FILE, where the rename fails with EBUSY. Both would
/// otherwise fail every invocation, and the version-bump branch above retries on
/// every single run because the file on disk never gets updated.
fn persist_registry(path: &std::path::Path, registry: &Registry) {
    if let Err(e) = write_registry_file(path, registry) {
        // Through `cause_of` like the two arms above, and this is the site that
        // needs it most: the doc below names a read-only config directory and a
        // bind-mounted registry.json as real layouts, and EROFS, EBUSY and EACCES
        // want three different responses from the user. Without the cause the
        // message named the path twice and the reason not at all.
        tracing::warn!(
            "{e}{}. Continuing with the bundled registry; this will be retried on \
             the next run.",
            cause_of(&e)
        );
    }
}

/// The cause behind an error, rendered for a log line, or empty if it has none.
///
/// `thiserror` puts only the `#[error(...)]` string into `Display`, so `{e}` on
/// these two variants names the path and stops. The `#[source]` is the part that
/// tells the read arm's cases apart, and they want opposite responses: "Permission
/// denied" is a chmod, "Input/output error" is a failing disk, "Too many open
/// files" is a transient batch-run condition that will clear on its own. `main`
/// walks this same chain for errors it returns (see its `caused by:` lines); these
/// are warnings and never reach it, so they have to walk it themselves.
fn cause_of(e: &Error) -> String {
    std::error::Error::source(e).map_or_else(String::new, |source| format!(": {source}"))
}

/// Get path to registry file in user config.
fn registry_file_path() -> Result<PathBuf> {
    Ok(crate::config::config_dir()?.join("registry.json"))
}

/// Load registry from existing file.
///
/// Bytes rather than `read_to_string`, and `from_slice` rather than `from_str`,
/// because which of the two errors this returns is now a behavioural decision
/// rather than a label: [`load_registry_from`] repairs a file it can only fail to
/// PARSE and preserves one it could only fail to READ.
///
/// `read_to_string` breaks that split. It validates UTF-8 and reports a failure as
/// `io::ErrorKind::InvalidData`, so bytes that are definitively not a registry
/// arrived as a transport error and the file was never repaired: a warning on
/// every run, no version upgrade, and no route back short of deleting the file by
/// hand. The reachable sources are ordinary rather than exotic. A registry.json
/// saved from Notepad as "Unicode" or written by PowerShell 5.1's `>` is UTF-16;
/// one round-tripped through a cp1252 editor mangles the `ä` this project's own
/// bundled registry carries; and a pre-atomic-write birda killed mid-write could
/// truncate one mid-codepoint.
///
/// `from_slice` validates UTF-8 in every string it materialises, so all of those
/// surface as the parse errors they are, and `RegistryRead` is left meaning a
/// failure to obtain the bytes, which is what the caller's non-destructive arm
/// assumes. Measured limit, stated rather than glossed: a string in a field the
/// deserializer IGNORES is skipped without a UTF-8 check, so bad bytes there are
/// accepted instead of repaired. That is the harmless direction (nothing is
/// destroyed), but a field promoted from ignored to used would inherit it.
fn load_from_file(path: &std::path::Path) -> Result<Registry> {
    let content = std::fs::read(path).map_err(|e| Error::RegistryRead {
        path: path.to_path_buf(),
        source: e,
    })?;

    serde_json::from_slice(&content).map_err(|e| Error::RegistryParse {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Load bundled registry from binary.
fn load_bundled_registry() -> Result<Registry> {
    const BUNDLED_REGISTRY: &str = include_str!("../../registry.json");

    serde_json::from_str(BUNDLED_REGISTRY).map_err(|e| Error::RegistryParse {
        path: PathBuf::from("bundled://registry.json"),
        source: e,
    })
}

/// Write registry to file, atomically.
///
/// Not a plain `fs::write`, which truncates the file and then writes it, leaving
/// registry.json empty in between. That is not a rare path: this function runs
/// whenever the file is absent, fails to parse, or is older than the bundled
/// version, and the last of those happens on the first launch after any upgrade
/// that bumps the bundled registry.
///
/// It partly self-heals, which caps the severity but does not remove it: a
/// zero-length file fails to parse on the next run and takes the re-bootstrap
/// branch, but any user-local edits are gone by then with no message, and the
/// recovery write could lose the same race again. It also has an external
/// consumer: birda-gui reads the file straight off disk and calls `JSON.parse`,
/// so a truncated registry surfaces there as a parse error rather than as
/// anything actionable, and model-language selection breaks.
///
/// Parent directories are created by the helper, so a missing config directory
/// now fails as a `RegistryWrite` naming the file rather than as a bare
/// `Error::Io`. Callers route through [`persist_registry`] rather than calling
/// this directly, because a failure to persist must not abort the command.
///
/// One behaviour change from the plain write, and it is not compensated for here:
/// a `registry.json` that is a **symlink whose target exists** is followed, but a
/// *dangling* one is replaced by a regular file rather than being written through.
/// Only `config.toml` resolves a dangling link, via `config::file::resolve_link`.
fn write_registry_file(path: &std::path::Path, registry: &Registry) -> Result<()> {
    let content = serde_json::to_string_pretty(registry)
        .map_err(|e| Error::RegistrySerialize { source: e })?;

    // `Umask` rather than `OwnerOnly`, which is what the `fs::write` this replaced
    // produced. registry.json holds the model catalogue, which ships bundled in
    // the binary and is not private, and it has an external reader: birda-gui
    // reads it straight off disk. Creating it 0600 would break a shared install
    // on first run and never recover, because nothing widens it afterwards.
    // config.toml is the file in this directory that is created private, and it
    // is private because it can name paths the user would rather not advertise.
    crate::utils::fs::write_atomic(
        path,
        content.as_bytes(),
        crate::utils::fs::NewFileMode::Umask,
    )
    .map_err(|e| Error::RegistryWrite {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Bootstrap registry from bundled default.
fn bootstrap_registry(dest: &std::path::Path, registry: &Registry) -> Registry {
    // Infallible, and that is the point: the bundled registry this persists is
    // already in hand, so a first run on a read-only config directory should work
    // rather than refuse to start. See [`persist_registry`].
    persist_registry(dest, registry);
    registry.clone()
}

/// Find model entry by ID.
pub fn find_model<'a>(registry: &'a Registry, id: &str) -> Option<&'a ModelEntry> {
    registry.models.iter().find(|m| m.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::types::{FileInfo, LabelsInfo, LanguageVariant, LicenseInfo, ModelFiles};

    /// A registry carrying nothing but its version, for the write-path tests.
    ///
    /// The version is the payload: it is what makes two writes distinguishable
    /// when the point of the test is which of them a given name resolves to.
    fn versioned_registry(version: u32) -> Registry {
        Registry {
            schema_version: "1.0".into(),
            registry_version: version,
            range_filter: None,
            models: vec![],
        }
    }

    /// The `registry_version` recorded in the file at `path`.
    ///
    /// For the repair tests the interesting failure IS the file failing to load
    /// back, so the panic has to carry which of the two failures it was. `{e}`
    /// gives that plus the path (once), and `cause_of` gives the errno or the
    /// serde detail. A bare `unwrap` reported the variant but no cause; panicking
    /// without `{e}` reported the cause but lost the classification this whole
    /// module exists to draw.
    ///
    /// `#[track_caller]` because the panic location is otherwise this helper
    /// rather than the assertion that called it, which is what makes an `unwrap`
    /// in a shared test helper hard to place.
    #[track_caller]
    fn version_at(path: &std::path::Path) -> u32 {
        match load_from_file(path) {
            Ok(registry) => registry.registry_version,
            Err(e) => panic!("{e}{}", super::cause_of(&e)),
        }
    }

    #[test]
    fn test_write_registry_file_creates_the_config_directory() {
        // The bootstrap case: registry.json is written into a config directory
        // that does not exist yet on a fresh install. This function used to
        // create it itself and now leans on the write helper, so the coverage
        // has to move rather than disappear.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing").join("registry.json");

        write_registry_file(&path, &versioned_registry(7)).unwrap();

        assert_eq!(version_at(&path), 7);
    }

    #[test]
    #[cfg(unix)]
    fn test_write_registry_file_does_not_truncate_the_target_in_place() {
        // #311. `fs::write` truncates the file and then writes it, so
        // registry.json is empty in between, and this runs on the first launch
        // after any upgrade that bumps the bundled registry version.
        //
        // A hardlink is a second name for the same inode: truncate-and-rewrite
        // shows the new contents through the link, write-then-rename leaves the
        // old inode intact behind it. Reading the old version back through the
        // link is proof the file was replaced rather than rewritten.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");

        write_registry_file(&path, &versioned_registry(1)).unwrap();
        let link = dir.path().join("registry.json.link");
        std::fs::hard_link(&path, &link).unwrap();

        write_registry_file(&path, &versioned_registry(2)).unwrap();

        assert_eq!(
            version_at(&link),
            1,
            "the previous registry must survive behind its own name; seeing \
             version 2 here means the file was truncated in place"
        );
        assert_eq!(
            version_at(&path),
            2,
            "the registry path must carry the new contents"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_a_new_registry_is_not_narrowed_to_its_owner() {
        // The `fs::write` this replaced created registry.json at whatever the
        // umask allowed, usually 0644. Publishing by rename hands the path the
        // temporary's inode, and a temporary is owner-only, so without an explicit
        // policy a fresh install would get a 0600 registry and never recover:
        // nothing widens it afterwards, and birda-gui reads this file straight off
        // disk, so a shared install would break on first run.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");

        write_registry_file(&path, &versioned_registry(1)).unwrap();

        // Compared against a file `File::create` made under the same umask rather
        // than against a literal 0o644, which would fail for anyone whose umask is
        // 0o077 or 0o002.
        let reference = dir.path().join("reference");
        drop(std::fs::File::create(&reference).unwrap());

        let mode_of =
            |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode_of(&path),
            mode_of(&reference),
            "the registry must keep the mode File::create would have given it"
        );

        // What this cannot see, said out loud rather than left as a silent pass:
        // under a umask that masks the group and world bits away, `Umask` and
        // `OwnerOnly` both yield 0o600, so the assertion above holds whichever
        // policy production passes and the regression it guards could return
        // undetected.
        if mode_of(&reference) & 0o066 == 0 {
            eprintln!(
                "skipped the policy distinction: this umask masks both policies to \
                 {:o}, so a wrong one would not be detected here",
                mode_of(&reference)
            );
        }
    }

    #[test]
    fn test_write_registry_file_leaves_no_temporary_behind() {
        // The temporary has to be created beside the target, because rename is
        // only atomic within a filesystem and $TMPDIR is routinely a different
        // one. That puts it in the user's config directory, so a successful
        // write must clean up after itself.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");

        write_registry_file(&path, &versioned_registry(1)).unwrap();

        let strays: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name != "registry.json")
            .collect();
        assert!(
            strays.is_empty(),
            "a successful write must not litter the config directory, found: {strays:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_a_registry_that_could_not_be_read_is_left_alone() {
        // #323. Every failure to LOAD used to be treated as grounds to overwrite
        // the file with the bundled copy, and `load_from_file` returns two
        // semantically different errors. An unreadable file (EACCES, EIO, a flaky
        // network mount, fd exhaustion during a parallel batch) may be perfectly
        // intact, and replacing it destroys any user-local edits with no message.
        // birda-gui reads this file straight off disk, so a hand-maintained
        // registry disappearing is a user-visible outcome, not a cache miss.
        //
        // Mode 0000 blocks the read but not the rename: rename needs write and
        // execute on the DIRECTORY, not on the file. So the buggy version really
        // does replace this file, and this assertion really does distinguish them.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        let sentinel = serde_json::to_string_pretty(&versioned_registry(1)).unwrap();
        std::fs::write(&path, &sentinel).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();

        if std::fs::read(&path).is_ok() {
            // A silent skip: `cargo test` hides a passing test's output, so this
            // reports ok while asserting nothing. Tolerated rather than made
            // fatal, because the only discriminator available in here is ambient.
            // Gating on `CI` was tried and reverted: act, GitLab, Drone and any
            // `docker run -e CI=true` all set it AND run as root, so it reddened
            // a contributor's suite over an environment fact that says nothing
            // about their change, while never firing on this project's own
            // ubuntu-latest runner, which is not root. Making the skip loud needs
            // a marker this repo owns, set in ci.yml's test job.
            eprintln!(
                "skipped: this process can read a mode-0000 file (running as root, \
                 or a filesystem that ignores modes), so the read-failure arm \
                 cannot be reached here"
            );
            return;
        }

        let loaded = load_registry_from(&path, versioned_registry(99));

        assert_eq!(
            loaded.registry_version, 99,
            "the caller must still get a usable registry, from the bundled copy"
        );

        // Widen it again so the assertion below can read it back.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            sentinel,
            "an intact but unreadable registry must survive byte for byte; \
             rewriting it destroys user-local edits for an error that says \
             nothing about the file's contents"
        );
    }

    #[test]
    fn test_a_registry_whose_bytes_are_not_utf8_is_repaired() {
        // The bytes being invalid UTF-8 is a verdict on the CONTENTS, so it has to
        // reach the repair arm. It did not while `load_from_file` used
        // `read_to_string`, which reports that as `io::ErrorKind::InvalidData` and
        // therefore as `RegistryRead`: the file was left alone forever, warning on
        // every run, never upgraded, with no route back except deleting it by hand.
        //
        // Deliberately NOT `#[cfg(unix)]`. The reachable ways to produce this file
        // are mostly Windows ones (Notepad's "Unicode" is UTF-16, so is PowerShell
        // 5.1's `>`), and CI runs tests on Linux only, so gating this would leave
        // the platform that triggers it with no coverage at all.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");

        // Valid JSON shape; one byte inside a string value is not valid UTF-8, so
        // the ONLY reason this fails to load is the encoding.
        let mut bytes = br#"{"schema_version":"1.0","registry_version":1,"models":[]}"#.to_vec();
        bytes[20] = 0xFF;
        assert!(
            String::from_utf8(bytes.clone()).is_err(),
            "the fixture must actually be invalid UTF-8, or this test proves nothing"
        );
        std::fs::write(&path, &bytes).unwrap();

        let loaded = load_registry_from(&path, versioned_registry(42));

        assert_eq!(loaded.registry_version, 42);
        assert_eq!(
            version_at(&path),
            42,
            "bytes that are not UTF-8 cannot be a registry, so the file must be \
             repaired rather than preserved as possibly-intact"
        );
    }

    #[test]
    fn test_a_corrupt_registry_is_still_replaced() {
        // The other half of #323, pinned so the fix cannot overshoot. A file that
        // does not parse holds nothing worth keeping, and leaving it would wedge
        // every run behind a permanent fallback that never repairs itself.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        std::fs::write(&path, b"{ this is not json").unwrap();

        let loaded = load_registry_from(&path, versioned_registry(42));

        assert_eq!(loaded.registry_version, 42);
        assert_eq!(
            version_at(&path),
            42,
            "a registry that cannot be parsed must be repaired on disk, not just \
             in memory"
        );
    }

    #[test]
    fn test_a_newer_bundled_registry_replaces_the_cached_one() {
        // The first launch after any upgrade that bumps the bundled registry
        // takes this branch, so it is the most-travelled write path in the file.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        write_registry_file(&path, &versioned_registry(1)).unwrap();

        let loaded = load_registry_from(&path, versioned_registry(2));

        assert_eq!(loaded.registry_version, 2);
        assert_eq!(version_at(&path), 2, "the cached copy must be updated too");
    }

    #[test]
    fn test_cause_of_renders_the_source_rather_than_the_error_itself() {
        // The three warnings in this file are the user's only account of what
        // happened to their registry, and `thiserror` puts only the `#[error(..)]`
        // string into Display, so without this the errno never reaches them:
        // "Permission denied" wants a chmod and "Input/output error" wants a
        // different disk. Asserted on the function, not on the emitted log line,
        // because the crate carries no log-capture dev-dependency.
        let source = || std::io::Error::from_raw_os_error(13);
        let denied = Error::RegistryRead {
            path: std::path::PathBuf::from("/tmp/registry.json"),
            source: source(),
        };

        // Compared against a freshly built io::Error rather than a literal, so
        // the assertion does not depend on the platform's wording for EACCES.
        assert_eq!(
            cause_of(&denied),
            format!(": {}", source()),
            "the cause must be the SOURCE, not the error's own Display, which \
             names only the path"
        );

        // The empty branch, which nothing else reaches: an error with no
        // `#[source]` must not leave a bare colon dangling on the message.
        assert_eq!(
            cause_of(&Error::Internal {
                message: "no source behind this one".into()
            }),
            ""
        );
    }

    #[test]
    fn test_a_missing_registry_is_bootstrapped_onto_disk() {
        // The first-run branch, and the only path through the resolver the other
        // tests never reach. A bootstrap that returned the right registry without
        // ever writing it would have passed every one of them.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        assert!(!path.exists(), "the fixture must start with no registry");

        let loaded = load_registry_from(&path, versioned_registry(7));

        assert_eq!(loaded.registry_version, 7);
        assert_eq!(
            version_at(&path),
            7,
            "a first run must leave the registry cached on disk, not only in memory"
        );
    }

    #[test]
    fn test_an_equal_version_is_not_an_upgrade_and_rewrites_nothing() {
        // Two ways to break this are invisible to the newer/older pair, because
        // neither of those lands on the boundary: widening `>` to `>=`, and a keep
        // path that rewrites the file anyway. Both would rewrite the user's
        // registry.json on EVERY run at steady state, which is the state nearly
        // every install is in, discarding local edits each time.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        write_registry_file(&path, &versioned_registry(5)).unwrap();

        // A byte the resolver has no reason to touch. The registry round-trips
        // exactly, so a rewrite would be byte-identical and undetectable without
        // one: trailing whitespace survives a read but not a re-serialize.
        let marked = format!("{}\n\n", std::fs::read_to_string(&path).unwrap());
        std::fs::write(&path, &marked).unwrap();
        let before = std::fs::read(&path).unwrap();

        let loaded = load_registry_from(&path, versioned_registry(5));

        assert_eq!(
            loaded.registry_version, 5,
            "the cached copy must be returned"
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "an equal version is not an upgrade, so the file must be left byte for \
             byte alone"
        );
    }

    #[test]
    fn test_a_registry_newer_than_the_bundled_one_is_kept() {
        // The downgrade guard: a user whose registry.json is ahead of the binary
        // (a newer birda ran, or they edited it) must not have it rolled back.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        write_registry_file(&path, &versioned_registry(9)).unwrap();

        let loaded = load_registry_from(&path, versioned_registry(2));

        assert_eq!(loaded.registry_version, 9);
        assert_eq!(version_at(&path), 9, "the user's copy must survive");
    }

    #[test]
    fn test_find_model_by_id() {
        let registry = Registry {
            schema_version: "1.0".into(),
            registry_version: 0,
            range_filter: None,
            models: vec![
                ModelEntry {
                    id: "test-1".into(),
                    name: "Test Model 1".into(),
                    description: "First test model".into(),
                    vendor: "Test Vendor".into(),
                    version: "1.0".into(),
                    model_type: "birdnet-v24".into(),
                    license: LicenseInfo {
                        r#type: "MIT".into(),
                        url: "https://example.com".into(),
                        commercial_use: true,
                        attribution_required: false,
                        share_alike: false,
                    },
                    files: Some(ModelFiles {
                        model: FileInfo {
                            url: "https://example.com/model.onnx".into(),
                            filename: "model.onnx".into(),
                            sha256: None,
                            size_bytes: None,
                        },
                        labels: LabelsInfo {
                            default_language: "en".into(),
                            languages: vec![LanguageVariant {
                                code: "en".into(),
                                name: "English".into(),
                                url: "https://example.com/labels.txt".into(),
                                filename: "labels.txt".into(),
                            }],
                        },
                        bsg_calibration: None,
                        bsg_migration: None,
                        bsg_distribution_maps: None,
                    }),
                    build: None,
                    default_variant: None,
                    selection: std::collections::BTreeMap::new(),
                    variants: Vec::new(),
                    recommended: false,
                },
                ModelEntry {
                    id: "test-2".into(),
                    name: "Test Model 2".into(),
                    description: "Second test model".into(),
                    vendor: "Test Vendor".into(),
                    version: "2.0".into(),
                    model_type: "perch-v2".into(),
                    license: LicenseInfo {
                        r#type: "Apache-2.0".into(),
                        url: "https://example.com".into(),
                        commercial_use: true,
                        attribution_required: true,
                        share_alike: false,
                    },
                    files: Some(ModelFiles {
                        model: FileInfo {
                            url: "https://example.com/model2.onnx".into(),
                            filename: "model2.onnx".into(),
                            sha256: None,
                            size_bytes: None,
                        },
                        labels: LabelsInfo {
                            default_language: "en".into(),
                            languages: vec![LanguageVariant {
                                code: "en".into(),
                                name: "English".into(),
                                url: "https://example.com/labels2.txt".into(),
                                filename: "labels2.txt".into(),
                            }],
                        },
                        bsg_calibration: None,
                        bsg_migration: None,
                        bsg_distribution_maps: None,
                    }),
                    build: None,
                    default_variant: None,
                    selection: std::collections::BTreeMap::new(),
                    variants: Vec::new(),
                    recommended: true,
                },
            ],
        };

        // Test finding existing models
        assert!(find_model(&registry, "test-1").is_some());
        assert!(find_model(&registry, "test-2").is_some());

        let model1 = find_model(&registry, "test-1").unwrap();
        assert_eq!(model1.name, "Test Model 1");
        assert_eq!(model1.version, "1.0");

        let model2 = find_model(&registry, "test-2").unwrap();
        assert_eq!(model2.name, "Test Model 2");
        assert!(model2.recommended);

        // Test finding non-existent model
        assert!(find_model(&registry, "missing").is_none());
    }

    #[test]
    fn test_bundled_registry_parses() {
        // This test verifies that the bundled registry.json is valid
        const BUNDLED_REGISTRY: &str = include_str!("../../registry.json");

        let result = serde_json::from_str::<Registry>(BUNDLED_REGISTRY);
        assert!(result.is_ok(), "Bundled registry should parse successfully");

        let registry = result.unwrap();
        assert_eq!(registry.schema_version, "2.0");
        assert!(
            !registry.models.is_empty(),
            "Registry should contain models"
        );

        // Verify we have expected models
        assert!(find_model(&registry, "birdnet-v24").is_some());
        assert!(find_model(&registry, "birdnet-v30").is_some());
        assert!(find_model(&registry, "perch-v2").is_some());
        assert!(find_model(&registry, "bsg-fi-v44").is_some());
    }

    #[test]
    fn test_bundled_registry_defines_the_geomodel_range_filter() {
        const BUNDLED_REGISTRY: &str = include_str!("../../registry.json");

        let registry = serde_json::from_str::<Registry>(BUNDLED_REGISTRY).unwrap();
        let range_filter = registry
            .range_filter
            .expect("bundled registry must define range_filter");

        assert_eq!(range_filter.version, "3.0.2");
        assert_eq!(range_filter.species_count, 12012);
        assert!(
            range_filter.name.contains("BirdNET"),
            "geomodel display name must credit BirdNET"
        );
        assert!(
            range_filter.model.sha256.is_some(),
            "geomodel model file must be checksum verified"
        );
        assert!(
            range_filter.labels.sha256.is_some(),
            "geomodel labels file must be checksum verified"
        );
        assert!(
            registry.registry_version >= 4,
            "registry_version must be bumped so existing installs pick up the geomodel"
        );
    }
}
