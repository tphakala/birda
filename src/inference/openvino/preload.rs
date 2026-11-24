//! Loading the libraries the `OpenVINO` C API library imports.
//!
//! `openvino-sys` opens the C API library with `libloading::Library::new`
//! (`openvino-sys` 0.11.1 `src/linking/runtime.rs:108`), and the system loader
//! then looks for that library's own imports, the core library and oneTBB,
//! on the loader path (`LD_LIBRARY_PATH`, `PATH`). A runtime found through
//! `INTEL_OPENVINO_DIR`, an install root or a Store package is not on that
//! path. Opening the two dependencies first, by absolute path, makes the loader
//! satisfy the import from the already loaded module instead of searching.
//!
//! Selecting the files is pure over a directory listing; opening them is the
//! only code here that loads a library.

use crate::constants::openvino as ov;
use crate::utils::text::strip_prefix_ignore_ascii_case;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tracing::debug;

/// Lists a directory's entry names; `None` when it is not a readable directory.
/// The same shape as `runtime::ListDir`.
pub type ListNames<'a> = &'a dyn Fn(&Path) -> Option<Vec<String>>;

/// The file and directory names of one OS's install layout, as plain data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// File name of the C API library.
    pub c_api: &'static str,
    /// File name of the core library the C API library imports.
    pub core: &'static str,
    /// File name of the oneTBB library the core library imports.
    pub tbb: &'static str,
    /// Install-root subdirectory holding the C API and core libraries, with `/`
    /// separators.
    pub lib_dir: &'static str,
    /// Install-root subdirectory holding oneTBB, with `/` separators.
    pub tbb_dir: &'static str,
    /// Whether file and directory names match without regard to ASCII case.
    pub ignore_case: bool,
}

impl Layout {
    /// The Linux install layout.
    pub const LINUX: Self = Self {
        c_api: ov::LINUX_LIBRARY_FILE,
        core: ov::LINUX_CORE_LIBRARY_FILE,
        tbb: ov::LINUX_TBB_LIBRARY_FILE,
        lib_dir: ov::LINUX_INSTALL_LIB_DIR,
        tbb_dir: ov::LINUX_TBB_DIR,
        ignore_case: false,
    };

    /// The Windows install layout.
    pub const WINDOWS: Self = Self {
        c_api: ov::WINDOWS_LIBRARY_FILE,
        core: ov::WINDOWS_CORE_LIBRARY_FILE,
        tbb: ov::WINDOWS_TBB_LIBRARY_FILE,
        lib_dir: ov::WINDOWS_INSTALL_LIB_DIR,
        tbb_dir: ov::WINDOWS_TBB_DIR,
        ignore_case: true,
    };

    /// Whether two file or directory names are the same in this layout.
    fn same(&self, a: &str, b: &str) -> bool {
        if self.ignore_case {
            a.eq_ignore_ascii_case(b)
        } else {
            a == b
        }
    }

    /// The listed name that matches `wanted`: the exact spelling, else, when
    /// this layout ignores case, the first spelling that differs only in case.
    pub fn find<'a>(&self, names: &'a [String], wanted: &str) -> Option<&'a String> {
        names
            .iter()
            .find(|name| *name == wanted)
            .or_else(|| names.iter().find(|name| self.same(name, wanted)))
    }

    /// The core library's file name for a located C API file name: the C API
    /// prefix is replaced and the rest (a version suffix) kept, so the core
    /// stays at the version of the C API library the search picked. `None`
    /// when the file is not named like the C API library.
    fn core_name(&self, c_api_file: &str) -> Option<String> {
        let suffix = if self.ignore_case {
            strip_prefix_ignore_ascii_case(c_api_file, self.c_api)?
        } else {
            c_api_file.strip_prefix(self.c_api)?
        };
        Some(format!("{}{suffix}", self.core))
    }

    /// The install root when `lib_dir_path` ends with this layout's library
    /// directory.
    fn install_root<'a>(&self, lib_dir_path: &'a Path) -> Option<&'a Path> {
        let wanted: Vec<&str> = self.lib_dir.split('/').collect();
        let actual: Vec<_> = lib_dir_path.components().rev().take(wanted.len()).collect();
        if actual.len() != wanted.len() {
            return None;
        }
        let matches = actual
            .iter()
            .zip(wanted.iter().rev())
            .all(|(have, want)| self.same(&have.as_os_str().to_string_lossy(), want));
        if matches {
            lib_dir_path.ancestors().nth(wanted.len())
        } else {
            None
        }
    }
}

/// The files to open before `library`, in the order to open them: oneTBB, then
/// the core library.
///
/// Each file comes from the first of these directories that lists it: the
/// library's own directory, then, when that is an install's library directory,
/// the install's oneTBB directory. The names are exact, so the allocator
/// replacement, the NUMA helper and the debug variants that sit beside oneTBB
/// are never opened.
///
/// # Errors
///
/// All or nothing: when either file is not found, or the core library's name
/// cannot be derived from the located file, nothing is selected and the text
/// says what is missing. Preloading one of the two would mix it with whatever
/// the loader finds for the other.
pub fn dependency_files(
    library: &Path,
    layout: &Layout,
    list_dir: ListNames<'_>,
) -> Result<Vec<PathBuf>, String> {
    let (Some(dir), Some(file_name)) = (
        library.parent(),
        library.file_name().map(|name| name.to_string_lossy()),
    ) else {
        return Err(format!(
            "{} has no directory or file name",
            library.display()
        ));
    };
    let mut tried: Vec<PathBuf> = vec![dir.to_path_buf()];
    let mut dirs: Vec<(PathBuf, Vec<String>)> = Vec::new();
    if let Some(names) = list_dir(dir) {
        dirs.push((dir.to_path_buf(), names));
    }
    if let Some(root) = layout.install_root(dir) {
        let tbb_dir = root.join(layout.tbb_dir);
        tried.push(tbb_dir.clone());
        if let Some(names) = list_dir(&tbb_dir) {
            dirs.push((tbb_dir, names));
        }
    }
    let from_first_dir = |wanted: &str| {
        dirs.iter()
            .find_map(|(dir, names)| layout.find(names, wanted).map(|listed| dir.join(listed)))
            .ok_or_else(|| {
                let places: Vec<String> =
                    tried.iter().map(|dir| dir.display().to_string()).collect();
                format!("{wanted} not found in {}", places.join(", "))
            })
    };
    let core = layout
        .core_name(&file_name)
        .ok_or_else(|| format!("{file_name} is not named like {}", layout.c_api))?;
    Ok(vec![from_first_dir(layout.tbb)?, from_first_dir(&core)?])
}

/// Open each file with `open`, continuing past a failure. Returns what opened.
///
/// A dependency that cannot be opened is not fatal here: the library that
/// needs it is loaded next and its own error names what is missing.
fn open_each<T, E: std::fmt::Display>(
    files: &[PathBuf],
    mut open: impl FnMut(&Path) -> Result<T, E>,
) -> Vec<T> {
    files
        .iter()
        .filter_map(|file| match open(file) {
            Ok(handle) => {
                debug!("preloaded {}", file.display());
                Some(handle)
            }
            Err(error) => {
                debug!("could not preload {}: {error}", file.display());
                None
            }
        })
        .collect()
}

/// Handles of the preloaded libraries. They are never dropped: unloading the
/// `OpenVINO` core after its initialisers ran, while the process may still use
/// it, is not safe.
static PRELOADED: Mutex<Vec<libloading::Library>> = Mutex::new(Vec::new());

#[allow(unsafe_code)]
fn open_dependency(path: &Path) -> Result<libloading::Library, libloading::Error> {
    // SAFETY: the file is the oneTBB or the `OpenVINO` core library of the
    // install whose C API library this process loads next through
    // `openvino-sys`, which runs the same initialisers through the same
    // `Library::new` call (`openvino-sys` 0.11.1 `src/linking/runtime.rs:108`).
    // No symbol is taken from the handle, and the handle is never dropped
    // (see `PRELOADED`).
    unsafe { libloading::Library::new(path) }
}

/// Open `files` and keep them loaded for the life of the process. Returns
/// whether any of them opened.
///
/// Each file opens on its own: one that fails is logged and the others still
/// open, and a handle that did open cannot be taken back.
pub fn preload(files: &[PathBuf]) -> bool {
    let opened = open_each(files, open_dependency);
    if opened.is_empty() {
        return false;
    }
    PRELOADED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .extend(opened);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    type Selection = Result<Vec<PathBuf>, String>;

    fn files(layout: &Layout, library: &Path, fs: &[(&Path, &[&str])]) -> Selection {
        let fs: HashMap<PathBuf, Vec<String>> = fs
            .iter()
            .map(|(dir, names)| {
                (
                    dir.to_path_buf(),
                    names.iter().map(ToString::to_string).collect(),
                )
            })
            .collect();
        dependency_files(library, layout, &|dir| fs.get(dir).cloned())
    }

    fn ok(paths: &[&Path]) -> Vec<PathBuf> {
        paths.iter().map(|path| path.to_path_buf()).collect()
    }

    fn missing(name: &str, dirs: &[&Path]) -> Selection {
        let dirs: Vec<String> = dirs.iter().map(|dir| dir.display().to_string()).collect();
        Err(format!("{name} not found in {}", dirs.join(", ")))
    }

    #[test]
    fn dependencies_are_tbb_then_core_or_nothing() {
        let root = Path::new("/ov/openvino_2026.3");
        let lib = root.join("runtime/lib/intel64");
        let tbb = root.join("runtime/3rdparty/tbb/lib");
        let archive: [(&Path, &[&str]); 2] = [
            (
                &lib,
                &[
                    "libopenvino_c.so",
                    "libopenvino.so",
                    "libopenvino.so.2630",
                    "libopenvino_c.so.2630",
                    "LIBTBB.SO.12",
                ],
            ),
            (
                &tbb,
                &[
                    "libtbb.so",
                    "libtbb.so.12",
                    "libtbbmalloc_proxy.so.2",
                    "libtbbbind_2_5.so.3",
                ],
            ),
        ];

        // a. The unversioned C API link gives the unversioned core link. The
        // upper-case TBB name in the library directory is not taken on Linux.
        assert_eq!(
            files(&Layout::LINUX, &lib.join("libopenvino_c.so"), &archive),
            Ok(ok(&[
                &tbb.join("libtbb.so.12"),
                &lib.join("libopenvino.so")
            ]))
        );
        // b. A versioned C API file keeps its version on the core.
        assert_eq!(
            files(&Layout::LINUX, &lib.join("libopenvino_c.so.2630"), &archive),
            Ok(ok(&[
                &tbb.join("libtbb.so.12"),
                &lib.join("libopenvino.so.2630")
            ]))
        );
        // c. A flat directory holding all three: the sibling is not needed.
        let flat = Path::new("/opt/birda");
        assert_eq!(
            files(
                &Layout::LINUX,
                &flat.join("libopenvino_c.so"),
                &[(
                    flat,
                    &["libopenvino_c.so", "libopenvino.so", "libtbb.so.12"]
                )]
            ),
            Ok(ok(&[
                &flat.join("libtbb.so.12"),
                &flat.join("libopenvino.so")
            ]))
        );
        // d. The library's own directory comes before the install's TBB
        // directory, so a copy placed beside the library is the one used.
        assert_eq!(
            files(
                &Layout::LINUX,
                &lib.join("libopenvino_c.so"),
                &[
                    (
                        &lib,
                        &["libopenvino_c.so", "libopenvino.so", "libtbb.so.12"]
                    ),
                    (&tbb, &["libtbb.so.12"]),
                ]
            ),
            Ok(ok(&[
                &lib.join("libtbb.so.12"),
                &lib.join("libopenvino.so")
            ]))
        );
        // e. No TBB beside the library and no readable TBB directory: nothing is
        // preloaded, and the error names the file and the places looked at.
        assert_eq!(
            files(
                &Layout::LINUX,
                &lib.join("libopenvino_c.so"),
                &[(&lib, &["libopenvino_c.so", "libopenvino.so"])]
            ),
            missing("libtbb.so.12", &[&lib, &tbb])
        );
        // The core missing while TBB is present also preloads nothing.
        assert_eq!(
            files(
                &Layout::LINUX,
                &lib.join("libopenvino_c.so"),
                &[(&lib, &["libopenvino_c.so"]), (&tbb, &["libtbb.so.12"])]
            ),
            missing("libopenvino.so", &[&lib, &tbb])
        );
        // f. A file not named like the C API library has no core to derive.
        let custom = Path::new("/opt/ov/lib");
        assert_eq!(
            files(
                &Layout::LINUX,
                &custom.join("custom.so"),
                &[(custom, &["custom.so", "libopenvino.so", "libtbb.so.12"])]
            ),
            Err("custom.so is not named like libopenvino_c.so".to_string())
        );
    }

    #[test]
    fn windows_dependencies_match_names_without_regard_to_case() {
        let root = Path::new("/ov/OpenVINO_2026.3");
        let lib = root.join("Runtime/Bin/Intel64/release");
        let tbb = root.join("runtime/3rdparty/tbb/bin");
        let fs: [(&Path, &[&str]); 2] = [
            (&lib, &["OpenVINO_C.dll", "OPENVINO.DLL"]),
            (
                &tbb,
                &[
                    "tbb12_debug.dll",
                    "TBB12.dll",
                    "tbbmalloc_proxy.dll",
                    "tbbmalloc.dll",
                ],
            ),
        ];
        assert_eq!(
            files(&Layout::WINDOWS, &lib.join("OpenVINO_C.dll"), &fs),
            Ok(ok(&[&tbb.join("TBB12.dll"), &lib.join("OPENVINO.DLL")]))
        );
        // Two spellings of one file: the exact name wins over the earlier
        // listed variant.
        let flat = Path::new("/opt/birda");
        assert_eq!(
            files(
                &Layout::WINDOWS,
                &flat.join("openvino_c.dll"),
                &[(
                    flat,
                    &[
                        "openvino_c.dll",
                        "OPENVINO.DLL",
                        "openvino.dll",
                        "TBB12.DLL",
                        "tbb12.dll"
                    ]
                )]
            ),
            Ok(ok(&[&flat.join("tbb12.dll"), &flat.join("openvino.dll")]))
        );
        // The same names are not found by the Linux layout, which matches
        // exactly.
        assert_eq!(
            files(
                &Layout::LINUX,
                &flat.join("libopenvino_c.so"),
                &[(
                    flat,
                    &["libopenvino_c.so", "LIBOPENVINO.SO", "LIBTBB.SO.12"]
                )]
            ),
            missing("libtbb.so.12", &[flat])
        );
    }

    #[test]
    fn a_failed_preload_does_not_stop_the_next() {
        let paths = [
            PathBuf::from("/a/libtbb.so.12"),
            PathBuf::from("/a/libopenvino.so"),
        ];
        let mut tried = Vec::new();
        let opened = open_each(&paths, |path| {
            tried.push(path.to_path_buf());
            if path == paths[0] {
                Err("refused")
            } else {
                Ok(path.to_path_buf())
            }
        });
        assert_eq!(tried, paths);
        assert_eq!(opened, [paths[1].clone()]);
    }
}
