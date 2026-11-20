//! ONNX Runtime startup checks.

use crate::constants::onnx_runtime;
use crate::error::{Error, Result};
use ort::init_from;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

/// Ensure ONNX Runtime is discoverable before inference-related code touches `ort`.
///
/// This prevents the downstream `ort` loader from blocking indefinitely when the
/// dynamic library is missing at runtime, and turns a load that never finishes
/// into an error (see `initialize_runtime`).
pub fn ensure_runtime_available() -> Result<()> {
    if let Some(path) = resolve_runtime_library_path()? {
        initialize_runtime(&path)?;
        return Ok(());
    }

    Err(missing_runtime_error())
}

fn resolve_runtime_library_path() -> Result<Option<PathBuf>> {
    if let Some(path) = env_override_path()? {
        return Ok(Some(path));
    }

    let executable_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    Ok(resolve_without_override(
        executable_dir.as_deref(),
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
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(onnx_runtime::LIBRARY_FILE_NAME)
                    .then_some(entry.path())
            });
        }
    }

    None
}

fn initialize_runtime(path: &Path) -> Result<()> {
    // `ort` 2.0.0-rc.12 builds its load error inside the initializer of the
    // lock that holds the loaded library, and building that error waits on the
    // same lock (`load_dylib_from_path` in ort's `src/lib.rs`), so a library
    // that fails to load blocks forever instead of returning the error. Load on
    // a separate thread and give up after a time limit; the blocked thread ends
    // with the process. While it is blocked, later calls fail at once instead
    // of starting another load that would block on the same lock.
    let owned_path = path.to_path_buf();
    let outcome = run_with_time_limit(
        &LOAD_IN_PROGRESS,
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
        Some(Ok(())) => Ok(()),
        Some(Err(e)) => Err(Error::RuntimeInitialization {
            reason: format!("failed to load ONNX Runtime from '{}': {e}", path.display()),
        }),
        None => Err(load_timed_out_error(path)),
    }
}

/// Set while a runtime load started by `initialize_runtime` is still running,
/// including one whose caller already gave up waiting.
static LOAD_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Clears the in-progress flag when the work finishes or panics.
struct InProgressGuard(&'static AtomicBool);

impl Drop for InProgressGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Run `work` on a new thread and return its result, or `None` when it has not
/// finished within `limit`. Returns `None` at once, without running `work`,
/// while earlier work started with the same `in_progress` flag is still running.
fn run_with_time_limit<T: Send + 'static>(
    in_progress: &'static AtomicBool,
    work: impl FnOnce() -> T + Send + 'static,
    limit: Duration,
) -> Option<T> {
    if in_progress.swap(true, Ordering::SeqCst) {
        return None;
    }
    let guard = InProgressGuard(in_progress);
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let result = work();
        // Clear the flag before sending, so a caller that receives the result
        // can start new work straight away.
        drop(guard);
        // The receiver is gone when the caller already gave up; nothing to do.
        let _ = sender.send(result);
    });
    receiver.recv_timeout(limit).ok()
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

fn missing_runtime_error() -> Error {
    let searched_dirs = onnx_runtime::COMMON_SEARCH_DIRS.join(", ");
    Error::RuntimeInitialization {
        reason: format!(
            "could not locate {}. Install ONNX Runtime or set {} to the full path of the library. Also checked {} and common library directories ({})",
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

    /// Wait for `flag` to clear, failing the test if it stays set.
    fn wait_until_clear(flag: &AtomicBool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while flag.load(Ordering::SeqCst) {
            assert!(
                std::time::Instant::now() < deadline,
                "the in-progress flag was never cleared"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn test_run_with_time_limit_returns_the_result_of_work_that_finishes() {
        static IN_PROGRESS: AtomicBool = AtomicBool::new(false);
        assert_eq!(
            run_with_time_limit(&IN_PROGRESS, || 42, Duration::from_secs(10)),
            Some(42)
        );
        assert!(!IN_PROGRESS.load(Ordering::SeqCst));
        assert_eq!(
            run_with_time_limit(&IN_PROGRESS, || 43, Duration::from_secs(10)),
            Some(43)
        );
    }

    #[test]
    fn test_run_with_time_limit_gives_up_on_work_that_does_not_finish() {
        static IN_PROGRESS: AtomicBool = AtomicBool::new(false);
        let (release, blocked) = mpsc::channel::<()>();
        let outcome = run_with_time_limit(
            &IN_PROGRESS,
            move || {
                // Blocks until `release` is dropped below.
                let _ = blocked.recv();
                1
            },
            Duration::from_millis(50),
        );
        assert_eq!(outcome, None);

        // While the first work is still blocked, new work is not started. It
        // would return 2 well within the limit if it ran.
        assert_eq!(
            run_with_time_limit(&IN_PROGRESS, || 2, Duration::from_secs(10)),
            None
        );

        // Once the first work finishes, new work runs again.
        drop(release);
        wait_until_clear(&IN_PROGRESS);
        assert_eq!(
            run_with_time_limit(&IN_PROGRESS, || 3, Duration::from_secs(10)),
            Some(3)
        );
    }

    #[test]
    fn test_run_with_time_limit_clears_the_flag_when_work_panics() {
        static IN_PROGRESS: AtomicBool = AtomicBool::new(false);
        assert_eq!(
            run_with_time_limit(
                &IN_PROGRESS,
                || -> i32 { panic!("work panicked on purpose") },
                Duration::from_secs(10),
            ),
            None
        );
        wait_until_clear(&IN_PROGRESS);
        assert_eq!(
            run_with_time_limit(&IN_PROGRESS, || 4, Duration::from_secs(10)),
            Some(4)
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
