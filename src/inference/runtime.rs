//! ONNX Runtime startup checks.

use crate::constants::onnx_runtime;
use crate::error::{Error, Result};
use ort::init_from;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Condvar, Mutex, PoisonError};
use std::time::Duration;

/// Ensure ONNX Runtime is discoverable before inference-related code touches `ort`.
///
/// This prevents the downstream `ort` loader from blocking indefinitely when the
/// dynamic library is missing at runtime, and turns a load that never finishes
/// into an error (see `initialize_runtime`).
pub fn ensure_runtime_available() -> Result<()> {
    let executable_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    if let Some(path) = resolve_runtime_library_path(executable_dir.as_deref())? {
        initialize_runtime(&path)?;
        return Ok(());
    }

    Err(missing_runtime_error(executable_dir.as_deref()))
}

fn resolve_runtime_library_path(executable_dir: Option<&Path>) -> Result<Option<PathBuf>> {
    if let Some(path) = env_override_path()? {
        return Ok(Some(path));
    }

    Ok(resolve_without_override(
        executable_dir,
        birdnet_onnx::find_ort_library,
        find_runtime_in_default_search_paths,
    ))
}

/// Pick the runtime library when `ORT_DYLIB_PATH` is not set.
///
/// The directory that holds the birda executable comes first, because the
/// release archives ship the library next to the binary. Only then do the
/// `lib/` and working-directory locations of `birdnet_onnx::find_ort_library`
/// and the system search paths apply. Searching the system directories first
/// picked an unrelated copy (for example Homebrew's on macOS), which a signed
/// release binary may not be allowed to load.
fn resolve_without_override(
    executable_dir: Option<&Path>,
    find_bundled_elsewhere: impl FnOnce() -> Option<PathBuf>,
    find_in_search_paths: impl FnOnce() -> Option<PathBuf>,
) -> Option<PathBuf> {
    executable_dir
        .and_then(find_runtime_in_directory)
        .or_else(find_bundled_elsewhere)
        .or_else(find_in_search_paths)
}

fn env_override_path() -> Result<Option<PathBuf>> {
    std::env::var_os(onnx_runtime::DYLIB_PATH_ENV).map_or_else(
        || Ok(None),
        |value| {
            let path = PathBuf::from(&value);
            if path.is_file() {
                Ok(Some(path))
            } else {
                Err(invalid_env_override_error(&value))
            }
        },
    )
}

fn find_runtime_in_default_search_paths() -> Option<PathBuf> {
    search_path_directories()
        .into_iter()
        .chain(common_search_directories())
        .find_map(|dir| find_runtime_in_directory(&dir))
}

fn search_path_directories() -> Vec<PathBuf> {
    std::env::var_os(onnx_runtime::SEARCH_PATH_ENV)
        .map(|value| std::env::split_paths(&value).collect())
        .unwrap_or_default()
}

fn common_search_directories() -> impl Iterator<Item = PathBuf> {
    onnx_runtime::COMMON_SEARCH_DIRS
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>()
        .into_iter()
}

fn find_runtime_in_directory(directory: &Path) -> Option<PathBuf> {
    // Check exact library name first (fast path, avoids read_dir on large dirs).
    let exact = directory.join(onnx_runtime::LIBRARY_FILE_NAME);
    if exact.is_file() {
        return Some(exact);
    }

    // On Linux, also look for versioned variants (e.g. libonnxruntime.so.1.22.0).
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir(directory) {
            return entries.flatten().find_map(|entry| {
                let path = entry.path();
                (is_versioned_library_name(&entry.file_name().to_string_lossy()) && path.is_file())
                    .then_some(path)
            });
        }
    }

    None
}

/// Whether `name` is the library file name followed by a dotted numeric
/// version, such as `libonnxruntime.so.1.22.0`. Names that only start with the
/// library file name, such as `libonnxruntime.so.backup`, do not match.
#[cfg(any(target_os = "linux", test))]
fn is_versioned_library_name(name: &str) -> bool {
    name.strip_prefix(onnx_runtime::LIBRARY_FILE_NAME)
        .and_then(|rest| rest.strip_prefix('.'))
        .is_some_and(|version| {
            version
                .split('.')
                .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
        })
}

fn initialize_runtime(path: &Path) -> Result<()> {
    // `ort` 2.0.0-rc.12 builds its load error inside the initializer of the
    // lock that holds the loaded library, and building that error waits on the
    // same lock (`load_dylib_from_path` in ort's `src/lib.rs`), so a library
    // that fails to load blocks forever instead of returning the error. Load on
    // a separate thread and give up after a time limit; the blocked thread ends
    // with the process. A later call waits for that thread instead of starting
    // another load that would block on the same lock.
    let owned_path = path.to_path_buf();
    let outcome = run_with_time_limit(
        &RUNTIME_LOAD,
        move || {
            init_from(&owned_path)
                .map(|builder| {
                    let _already_initialized = builder.commit();
                })
                .map_err(|e| e.to_string())
        },
        Duration::from_secs(onnx_runtime::LOAD_TIME_LIMIT_SECS),
    );

    match outcome {
        TimeLimited::Finished(Ok(())) => Ok(()),
        TimeLimited::Finished(Err(e)) => Err(load_failed_error(path, &e)),
        TimeLimited::Panicked => Err(load_failed_error(path, "the loader panicked")),
        TimeLimited::TimedOut => Err(load_timed_out_error(path)),
        TimeLimited::EarlierWorkRunning => Err(earlier_load_running_error(path)),
    }
}

/// How `run_with_time_limit` ended.
#[derive(Debug, PartialEq, Eq)]
enum TimeLimited<T> {
    /// The work returned this value within the limit.
    Finished(T),
    /// The work was still running when the limit ran out.
    TimedOut,
    /// The work panicked.
    Panicked,
    /// Earlier work in the same slot was still running when the limit ran out,
    /// so this work was never started.
    EarlierWorkRunning,
}

/// Lets one piece of work run at a time. Work keeps the slot until it returns,
/// even after its caller stopped waiting for it.
struct WorkSlot {
    busy: Mutex<bool>,
    idle: Condvar,
}

impl WorkSlot {
    const fn new() -> Self {
        Self {
            busy: Mutex::new(false),
            idle: Condvar::new(),
        }
    }
}

/// The slot that runtime loads started by `initialize_runtime` run in.
static RUNTIME_LOAD: WorkSlot = WorkSlot::new();

/// Frees the slot when the work returns or panics.
struct SlotGuard(&'static WorkSlot);

impl Drop for SlotGuard {
    fn drop(&mut self) {
        *self.0.busy.lock().unwrap_or_else(PoisonError::into_inner) = false;
        self.0.idle.notify_all();
    }
}

/// Run `work` on a new thread once `slot` is free, and return its result.
///
/// Waits up to `limit` for earlier work in the slot to finish, then up to
/// `limit` for `work` itself.
fn run_with_time_limit<T: Send + 'static>(
    slot: &'static WorkSlot,
    work: impl FnOnce() -> T + Send + 'static,
    limit: Duration,
) -> TimeLimited<T> {
    let (mut busy, _) = slot
        .idle
        .wait_timeout_while(
            slot.busy.lock().unwrap_or_else(PoisonError::into_inner),
            limit,
            |busy| *busy,
        )
        .unwrap_or_else(PoisonError::into_inner);
    if *busy {
        return TimeLimited::EarlierWorkRunning;
    }
    *busy = true;
    drop(busy);
    let guard = SlotGuard(slot);
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let result = work();
        // Free the slot before sending, so a caller that receives the result
        // can start new work straight away.
        drop(guard);
        // The receiver is gone when the caller already gave up; nothing to do.
        let _ = sender.send(result);
    });
    match receiver.recv_timeout(limit) {
        Ok(result) => TimeLimited::Finished(result),
        Err(RecvTimeoutError::Timeout) => TimeLimited::TimedOut,
        // The thread drops the sender without sending only when `work` panics.
        Err(RecvTimeoutError::Disconnected) => TimeLimited::Panicked,
    }
}

fn load_failed_error(path: &Path, reason: &str) -> Error {
    Error::RuntimeInitialization {
        reason: format!(
            "failed to load ONNX Runtime from '{}': {reason}",
            path.display()
        ),
    }
}

fn load_timed_out_error(path: &Path) -> Error {
    Error::RuntimeInitialization {
        reason: format!(
            "loading ONNX Runtime from '{}' did not finish within {} seconds, which means the library could not be loaded. Use the {} that ships with birda, or set {} to its full path.",
            path.display(),
            onnx_runtime::LOAD_TIME_LIMIT_SECS,
            onnx_runtime::LIBRARY_FILE_NAME,
            onnx_runtime::DYLIB_PATH_ENV
        ),
    }
}

fn earlier_load_running_error(path: &Path) -> Error {
    Error::RuntimeInitialization {
        reason: format!(
            "an earlier attempt to load ONNX Runtime was still running after {} seconds, so '{}' was not tried. The library could not be loaded. Use the {} that ships with birda, or set {} to its full path.",
            onnx_runtime::LOAD_TIME_LIMIT_SECS,
            path.display(),
            onnx_runtime::LIBRARY_FILE_NAME,
            onnx_runtime::DYLIB_PATH_ENV
        ),
    }
}

fn invalid_env_override_error(value: &OsString) -> Error {
    let display = PathBuf::from(value).display().to_string();
    Error::RuntimeInitialization {
        reason: format!(
            "{} points to '{}', but that file does not exist. Set it to the full path of {}.",
            onnx_runtime::DYLIB_PATH_ENV,
            display,
            onnx_runtime::LIBRARY_FILE_NAME
        ),
    }
}

fn missing_runtime_error(executable_dir: Option<&Path>) -> Error {
    let searched_dirs = onnx_runtime::COMMON_SEARCH_DIRS.join(", ");
    // `birdnet_onnx::find_ort_library` (birdnet-onnx 2.0.0-rc.16,
    // `src/runtime.rs`) returns `None` when the executable path is unknown, so
    // without it only the search paths were checked.
    let bundled = executable_dir.map_or_else(String::new, |dir| {
        format!(
            "the directory that holds birda ('{}'), the bundled locations near it and in the working directory, ",
            dir.display()
        )
    });
    Error::RuntimeInitialization {
        reason: format!(
            "could not locate {}. Install ONNX Runtime or set {} to the full path of the library. Also checked {bundled}{} and common library directories ({})",
            onnx_runtime::LIBRARY_FILE_NAME,
            onnx_runtime::DYLIB_PATH_ENV,
            onnx_runtime::SEARCH_PATH_ENV,
            searched_dirs
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch_library(dir: &Path) -> PathBuf {
        let library_path = dir.join(onnx_runtime::LIBRARY_FILE_NAME);
        std::fs::write(&library_path, []).expect("library file should be created");
        library_path
    }

    #[test]
    fn test_resolve_prefers_the_library_next_to_the_executable() {
        let exe_dir = tempfile::tempdir().expect("temp dir should be created");
        let bundled = touch_library(exe_dir.path());
        let elsewhere = PathBuf::from("/elsewhere/lib/libonnxruntime");
        let system = PathBuf::from("/opt/homebrew/lib/libonnxruntime");

        let chosen = resolve_without_override(
            Some(exe_dir.path()),
            || Some(elsewhere.clone()),
            || Some(system.clone()),
        );

        assert_eq!(chosen, Some(bundled));
    }

    #[test]
    fn test_resolve_falls_back_in_order_when_nothing_is_next_to_the_executable() {
        let exe_dir = tempfile::tempdir().expect("temp dir should be created");
        let elsewhere = PathBuf::from("/elsewhere/lib/libonnxruntime");
        let system = PathBuf::from("/opt/homebrew/lib/libonnxruntime");

        assert_eq!(
            resolve_without_override(
                Some(exe_dir.path()),
                || Some(elsewhere.clone()),
                || Some(system.clone()),
            ),
            Some(elsewhere)
        );
        assert_eq!(
            resolve_without_override(Some(exe_dir.path()), || None, || Some(system.clone())),
            Some(system.clone())
        );
        assert_eq!(
            resolve_without_override(None, || None, || Some(system.clone())),
            Some(system)
        );
        assert_eq!(resolve_without_override(None, || None, || None), None);
    }

    #[test]
    fn test_run_with_time_limit_returns_the_result_of_work_that_finishes() {
        static SLOT: WorkSlot = WorkSlot::new();
        assert_eq!(
            run_with_time_limit(&SLOT, || 42, Duration::from_secs(10)),
            TimeLimited::Finished(42)
        );
        // The slot is free again once the work has returned.
        assert_eq!(
            run_with_time_limit(&SLOT, || 43, Duration::from_millis(50)),
            TimeLimited::Finished(43)
        );
    }

    #[test]
    fn test_run_with_time_limit_gives_up_on_work_that_does_not_finish() {
        static SLOT: WorkSlot = WorkSlot::new();
        let (release, blocked) = mpsc::channel::<()>();
        let outcome = run_with_time_limit(
            &SLOT,
            move || {
                // Blocks until `release` is dropped below.
                let _ = blocked.recv();
                1
            },
            Duration::from_millis(50),
        );
        assert_eq!(outcome, TimeLimited::TimedOut);

        // While the first work is still blocked, new work is not started. It
        // would return 2 at once if it ran.
        assert_eq!(
            run_with_time_limit(&SLOT, || 2, Duration::from_millis(50)),
            TimeLimited::EarlierWorkRunning
        );

        // A caller that is waiting for the slot runs its work once the earlier
        // work finishes.
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            drop(release);
        });
        assert_eq!(
            run_with_time_limit(&SLOT, || 3, Duration::from_secs(10)),
            TimeLimited::Finished(3)
        );
    }

    #[test]
    fn test_run_with_time_limit_reports_a_panic_and_frees_the_slot() {
        static SLOT: WorkSlot = WorkSlot::new();
        assert_eq!(
            run_with_time_limit(
                &SLOT,
                || -> i32 { panic!("work panicked on purpose") },
                Duration::from_secs(10),
            ),
            TimeLimited::Panicked
        );
        assert_eq!(
            run_with_time_limit(&SLOT, || 4, Duration::from_secs(2)),
            TimeLimited::Finished(4)
        );
    }

    #[test]
    fn test_earlier_load_running_error_names_the_limit_path_and_override() {
        let error = earlier_load_running_error(Path::new("/opt/homebrew/lib/libonnxruntime.dylib"));
        assert_eq!(
            error.to_string(),
            Error::RuntimeInitialization {
                reason: format!(
                    "an earlier attempt to load ONNX Runtime was still running after {} seconds, so '/opt/homebrew/lib/libonnxruntime.dylib' was not tried. The library could not be loaded. Use the {} that ships with birda, or set {} to its full path.",
                    onnx_runtime::LOAD_TIME_LIMIT_SECS,
                    onnx_runtime::LIBRARY_FILE_NAME,
                    onnx_runtime::DYLIB_PATH_ENV
                ),
            }
            .to_string()
        );
    }

    #[test]
    fn test_missing_runtime_error_lists_the_executable_directory_when_known() {
        let searched_dirs = onnx_runtime::COMMON_SEARCH_DIRS.join(", ");
        assert_eq!(
            missing_runtime_error(Some(Path::new("/opt/birda"))).to_string(),
            Error::RuntimeInitialization {
                reason: format!(
                    "could not locate {}. Install ONNX Runtime or set {} to the full path of the library. Also checked the directory that holds birda ('/opt/birda'), the bundled locations near it and in the working directory, {} and common library directories ({searched_dirs})",
                    onnx_runtime::LIBRARY_FILE_NAME,
                    onnx_runtime::DYLIB_PATH_ENV,
                    onnx_runtime::SEARCH_PATH_ENV,
                ),
            }
            .to_string()
        );
        assert_eq!(
            missing_runtime_error(None).to_string(),
            Error::RuntimeInitialization {
                reason: format!(
                    "could not locate {}. Install ONNX Runtime or set {} to the full path of the library. Also checked {} and common library directories ({searched_dirs})",
                    onnx_runtime::LIBRARY_FILE_NAME,
                    onnx_runtime::DYLIB_PATH_ENV,
                    onnx_runtime::SEARCH_PATH_ENV,
                ),
            }
            .to_string()
        );
    }

    #[test]
    fn test_load_timed_out_error_names_the_path_limit_and_override() {
        let error = load_timed_out_error(Path::new("/opt/homebrew/lib/libonnxruntime.dylib"));
        assert_eq!(
            error.to_string(),
            Error::RuntimeInitialization {
                reason: format!(
                    "loading ONNX Runtime from '/opt/homebrew/lib/libonnxruntime.dylib' did not finish within {} seconds, which means the library could not be loaded. Use the {} that ships with birda, or set {} to its full path.",
                    onnx_runtime::LOAD_TIME_LIMIT_SECS,
                    onnx_runtime::LIBRARY_FILE_NAME,
                    onnx_runtime::DYLIB_PATH_ENV
                ),
            }
            .to_string()
        );
    }

    #[test]
    fn test_find_runtime_in_directory_matches_exact_file() {
        let temp_dir = tempfile::tempdir().expect("temp dir should be created");
        let library_path = temp_dir.path().join(onnx_runtime::LIBRARY_FILE_NAME);
        std::fs::write(&library_path, []).expect("library file should be created");

        assert_eq!(
            find_runtime_in_directory(temp_dir.path()),
            Some(library_path)
        );
    }

    #[test]
    fn test_is_versioned_library_name_accepts_only_a_numeric_version_suffix() {
        let name = onnx_runtime::LIBRARY_FILE_NAME;
        for accepted in [format!("{name}.1"), format!("{name}.1.22.0")] {
            assert!(is_versioned_library_name(&accepted), "{accepted}");
        }
        for rejected in [
            name.to_owned(),
            format!("{name}."),
            format!("{name}.backup"),
            format!("{name}.1.22.0.bak"),
            format!("{name}.1..0"),
            format!("{name}1.22.0"),
            format!("x{name}.1.22.0"),
        ] {
            assert!(!is_versioned_library_name(&rejected), "{rejected}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_find_runtime_in_directory_skips_names_that_only_share_the_prefix() {
        let temp_dir = tempfile::tempdir().expect("temp dir should be created");
        std::fs::write(
            temp_dir
                .path()
                .join(format!("{}.backup", onnx_runtime::LIBRARY_FILE_NAME)),
            [],
        )
        .expect("decoy file should be created");
        std::fs::create_dir(
            temp_dir
                .path()
                .join(format!("{}.1.21.0", onnx_runtime::LIBRARY_FILE_NAME)),
        )
        .expect("decoy directory should be created");

        assert_eq!(find_runtime_in_directory(temp_dir.path()), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_find_runtime_in_directory_matches_versioned_linux_library() {
        let temp_dir = tempfile::tempdir().expect("temp dir should be created");
        let library_path = temp_dir
            .path()
            .join(format!("{}.1.22.0", onnx_runtime::LIBRARY_FILE_NAME));
        std::fs::write(&library_path, []).expect("library file should be created");

        assert_eq!(
            find_runtime_in_directory(temp_dir.path()),
            Some(library_path)
        );
    }
}
