//! Finding, loading and enumerating the user's `OpenVINO` runtime.
//!
//! birda ships no `OpenVINO` files. The `openvino` crate is built with its
//! `runtime-linking` feature, so the C API library (`libopenvino_c`) is opened
//! at run time. This file decides where to look, loads the library from the
//! first place that has it, and lists the devices the runtime reports.
//!
//! The library's own imports, oneTBB and the core library, are opened first by
//! absolute path (see [`super::preload`]), so a runtime found through
//! `INTEL_OPENVINO_DIR`, an install root or a Store package loads without its
//! directories being on the loader path.
//!
//! The path logic, the load-error classification and the render-node diagnosis
//! are pure functions, tested without a runtime.

use super::device::{
    OvDevice, is_intel_architecture, parse_capabilities, parse_gpu_class, parse_memory_bytes,
    parse_pci_info, parse_uuid,
};
use super::{msix, preload};
use crate::constants::openvino as ov;
use crate::error::{Error, Result};
use crate::utils::text::strip_prefix_ignore_ascii_case;
use openvino::{Core, DeviceType, PropertyKey};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tracing::debug;

/// Operating systems the runtime search knows the layout of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    /// Linux.
    Linux,
    /// Windows.
    Windows,
}

impl Os {
    /// The OS this binary runs on, or `None` where `OpenVINO` is unsupported.
    #[must_use]
    pub const fn current() -> Option<Self> {
        if cfg!(target_os = "linux") {
            Some(Self::Linux)
        } else if cfg!(target_os = "windows") {
            Some(Self::Windows)
        } else {
            None
        }
    }

    /// File name of the C API library.
    #[must_use]
    pub const fn library_file(self) -> &'static str {
        self.layout().c_api
    }

    /// Whether a bare library name may be handed to the system loader: only
    /// on Linux, where `dlopen` does not search the working directory.
    const fn has_loader_fallback(self) -> bool {
        matches!(self, Self::Linux)
    }

    const fn install_lib_dir(self) -> &'static str {
        self.layout().lib_dir
    }

    /// The names and directories of the install layout, which include the
    /// libraries `open_library` opens before the C API library.
    const fn layout(self) -> preload::Layout {
        match self {
            Self::Linux => preload::Layout::LINUX,
            Self::Windows => preload::Layout::WINDOWS,
        }
    }

    const fn system_lib_dirs(self) -> &'static [&'static str] {
        match self {
            Self::Linux => ov::LINUX_SYSTEM_LIB_DIRS,
            Self::Windows => &[],
        }
    }
}

/// The process environment, as far as the search reads it.
#[derive(Debug, Clone)]
pub struct SearchEnv {
    /// Target OS.
    pub os: Os,
    /// Directory of the running executable.
    pub exe_dir: Option<PathBuf>,
    /// Value of `INTEL_OPENVINO_DIR`.
    pub install_dir: Option<PathBuf>,
    /// Entries of the loader path variable (`LD_LIBRARY_PATH`, `PATH`).
    pub loader_path: Vec<PathBuf>,
    /// `%ProgramFiles%`; read on every OS, used only for Windows.
    pub program_files: PathBuf,
    /// `%ProgramFiles(x86)%`; read on every OS, used only for Windows.
    pub program_files_x86: PathBuf,
}

impl SearchEnv {
    /// Read the search inputs from this process.
    #[must_use]
    pub fn from_process(os: Os) -> Self {
        let loader_var = match os {
            Os::Linux => "LD_LIBRARY_PATH",
            Os::Windows => "PATH",
        };
        let env_path = |name: &str, default: &str| {
            std::env::var_os(name)
                .filter(|value| !value.is_empty())
                .map_or_else(|| PathBuf::from(default), PathBuf::from)
        };
        Self {
            os,
            exe_dir: std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(Path::to_path_buf)),
            install_dir: std::env::var_os(ov::INSTALL_DIR_ENV)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from),
            loader_path: std::env::var_os(loader_var)
                .map(|value| std::env::split_paths(&value).collect())
                .unwrap_or_default(),
            program_files: env_path(ov::PROGRAM_FILES_ENV, ov::DEFAULT_PROGRAM_FILES),
            program_files_x86: env_path(ov::PROGRAM_FILES_X86_ENV, ov::DEFAULT_PROGRAM_FILES_X86),
        }
    }

    /// The directories that hold install roots, in search order.
    ///
    /// On Windows, `<ProgramFiles>\Intel` then `<ProgramFiles(x86)>\Intel`,
    /// with the second dropped when it is the first spelled in another case.
    fn install_parents(&self) -> Vec<PathBuf> {
        match self.os {
            Os::Linux => ov::LINUX_INSTALL_PARENTS
                .iter()
                .map(PathBuf::from)
                .collect(),
            Os::Windows => {
                let first = self.program_files.join(ov::WINDOWS_INSTALL_VENDOR_DIR);
                let second = self.program_files_x86.join(ov::WINDOWS_INSTALL_VENDOR_DIR);
                if same_path_ignoring_case(&first, &second) {
                    vec![first]
                } else {
                    vec![first, second]
                }
            }
        }
    }

    /// The directory that holds Microsoft Store packages.
    fn windows_apps(&self) -> PathBuf {
        self.program_files.join(ov::WINDOWS_APPS_DIR)
    }
}

/// Whether two paths spell the same text apart from ASCII case.
fn same_path_ignoring_case(a: &Path, b: &Path) -> bool {
    a.to_string_lossy()
        .eq_ignore_ascii_case(&b.to_string_lossy())
}

/// Lists a directory's entry names; `None` when it is not a readable directory.
pub type ListDir<'a> = &'a dyn Fn(&Path) -> Option<Vec<String>>;

/// [`ListDir`] over the real filesystem.
#[must_use]
pub fn list_dir_fs(dir: &Path) -> Option<Vec<String>> {
    let entries = std::fs::read_dir(dir).ok()?;
    Some(
        entries
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect(),
    )
}

/// Result of searching for the library file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Located {
    /// The library file to load.
    File(PathBuf),
    /// Nothing found in the searched places, one line per place. The caller
    /// may still try the operating system's own loader path.
    NotFound {
        /// The places that were looked at.
        searched: Vec<String>,
    },
}

/// Why the configured path was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathProblem {
    /// What is wrong with the configured path.
    pub reason: String,
}

/// Version part of a versioned library name such as `libopenvino_c.so.2025.4.1`.
fn version_key(name: &str, base: &str) -> Option<Vec<u64>> {
    let suffix = name.strip_prefix(base)?.strip_prefix('.')?;
    suffix.split('.').map(|part| part.parse().ok()).collect()
}

/// Pick the library from a directory listing: the exact name, else on Linux
/// the highest versioned one. On Windows the exact name matches without regard
/// to ASCII case and the listed spelling is returned.
fn pick_library(names: &[String], os: Os) -> Option<String> {
    let exact = os.library_file();
    if let Some(name) = os.layout().find(names, exact) {
        return Some(name.clone());
    }
    if os != Os::Linux {
        return None;
    }
    names
        .iter()
        .filter_map(|name| version_key(name, exact).map(|key| (key, name)))
        .max()
        .map(|(_, name)| name.clone())
}

/// Look for the library in one directory.
fn find_in(dir: &Path, os: Os, list_dir: ListDir<'_>) -> Option<PathBuf> {
    let names = list_dir(dir)?;
    pick_library(&names, os).map(|name| dir.join(name))
}

/// Version of an install root's directory name: `openvino_2025.4` is
/// `[2025, 4]`, a bare `openvino` is `[]` (a symlink or an unversioned install),
/// and a name whose suffix is not numeric, such as an archive directory with a
/// platform in its name, has none.
fn root_version(name: &str) -> Option<Vec<u64>> {
    let rest = strip_prefix_ignore_ascii_case(name, ov::INSTALL_ROOT_PREFIX)?;
    if rest.is_empty() {
        return Some(Vec::new());
    }
    let suffix = rest.strip_prefix('_').or_else(|| rest.strip_prefix('-'))?;
    suffix.split('.').map(|part| part.parse().ok()).collect()
}

/// Compare two versions, a missing component counting as 0.
fn compare_versions(a: &[u64], b: &[u64]) -> std::cmp::Ordering {
    (0..a.len().max(b.len()))
        .map(|i| {
            (
                a.get(i).copied().unwrap_or(0),
                b.get(i).copied().unwrap_or(0),
            )
        })
        .map(|(x, y)| x.cmp(&y))
        .find(|order| order.is_ne())
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// Install-root directory names, newest version first.
///
/// Compared by the parsed version, so `openvino_2025.10` ranks above
/// `openvino_2025.4`. A bare `openvino` counts as version 0, and names without
/// a parsable version come last. Ties are broken by name, descending.
fn order_install_roots(mut names: Vec<String>) -> Vec<String> {
    names.sort_by(|a, b| match (root_version(a), root_version(b)) {
        (Some(va), Some(vb)) => compare_versions(&vb, &va).then_with(|| b.cmp(a)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => b.cmp(a),
    });
    names
}

/// Install roots found directly under `parent`, newest version first.
fn install_roots(parent: &Path, list_dir: ListDir<'_>) -> Vec<PathBuf> {
    let names: Vec<String> = list_dir(parent)
        .unwrap_or_default()
        .into_iter()
        .filter(|name| strip_prefix_ignore_ascii_case(name, ov::INSTALL_ROOT_PREFIX).is_some())
        .collect();
    order_install_roots(names)
        .into_iter()
        .map(|name| parent.join(name))
        .collect()
}

/// Resolve `inference.openvino_path`: a library file, a directory holding it,
/// or an install root.
///
/// A relative path is made absolute against the working directory first, so a
/// bare file name names the file there and the loader never searches for it.
fn resolve_explicit(
    path: &Path,
    os: Os,
    list_dir: ListDir<'_>,
) -> std::result::Result<PathBuf, PathProblem> {
    let absolute = std::path::absolute(path).map_err(|e| PathProblem {
        reason: format!("it cannot be made absolute: {e}"),
    })?;
    let path = absolute.as_path();
    if list_dir(path).is_some() {
        let candidates = [path.to_path_buf(), path.join(os.install_lib_dir())];
        return candidates
            .iter()
            .find_map(|dir| find_in(dir, os, list_dir))
            .ok_or_else(|| PathProblem {
                reason: format!(
                    "no {} in that directory or in its {}",
                    os.library_file(),
                    os.install_lib_dir()
                ),
            });
    }
    let file_name = path.file_name().map(|name| name.to_string_lossy());
    let listed = path
        .parent()
        .and_then(list_dir)
        .zip(file_name)
        .is_some_and(|(names, file)| os.layout().find(&names, &file).is_some());
    if listed {
        Ok(path.to_path_buf())
    } else {
        Err(PathProblem {
            reason: "it does not exist".to_string(),
        })
    }
}

/// The places searched so far and what is needed to look in one more.
struct Search<'a> {
    os: Os,
    list_dir: ListDir<'a>,
    /// One line per place looked at, in order.
    searched: Vec<String>,
}

impl Search<'_> {
    /// Look for the library in `dir`, and record the place as `label`.
    fn check(&mut self, label: &str, dir: &Path) -> Option<PathBuf> {
        let found = find_in(dir, self.os, self.list_dir);
        self.searched.push(format!("  {label}: {}", dir.display()));
        found
    }

    /// Look in the library directory of each Store package folder in `roots`
    /// that is not in `tried`, adding each to `tried`.
    fn check_packages(&mut self, roots: Vec<PathBuf>, tried: &mut Vec<PathBuf>) -> Option<PathBuf> {
        for root in roots {
            if tried
                .iter()
                .any(|seen| same_path_ignoring_case(seen, &root))
            {
                continue;
            }
            let dir = root.join(self.os.install_lib_dir());
            tried.push(root);
            if let Some(file) = self.check("Microsoft Store package", &dir) {
                return Some(file);
            }
        }
        None
    }
}

/// Find the library in a Microsoft Store package.
///
/// The `WindowsApps` listing is tried first. The registry is queried only when
/// that produced no package holding the library, because the query starts a
/// process and a standard user usually cannot list `WindowsApps` at all. A
/// package is chosen only if its library directory lists the library, so a
/// stale registry entry for an uninstalled package is skipped.
fn find_in_store_packages(
    search: &mut Search<'_>,
    env: &SearchEnv,
    registry: msix::RegistryQuery<'_>,
) -> Option<PathBuf> {
    let apps = env.windows_apps();
    let mut tried = Vec::new();
    let listed = (search.list_dir)(&apps)
        .map(|names| msix::from_listing(&apps, &names))
        .unwrap_or_default();
    if let Some(file) = search.check_packages(listed, &mut tried) {
        return Some(file);
    }
    let registered = registry()
        .map(|output| msix::from_registry(&output))
        .unwrap_or_default();
    if let Some(file) = search.check_packages(registered, &mut tried) {
        return Some(file);
    }
    if tried.is_empty() {
        search.searched.push(format!(
            "  Microsoft Store packages: none found in {} or the package registry",
            apps.display()
        ));
    }
    None
}

/// Find the `OpenVINO` C API library.
///
/// An explicit path is the only place searched when it is given: a wrong
/// configured path must not silently fall through to some other runtime.
/// Otherwise, in order:
///
/// 1. the directory of the running executable, holding the library directly;
/// 2. `INTEL_OPENVINO_DIR`;
/// 3. the loader path variable;
/// 4. the system library directories (Linux);
/// 5. the install roots under `/opt/intel` (or `Program Files\Intel` and
///    `Program Files (x86)\Intel`), newest version first;
/// 6. Microsoft Store packages (Windows), newest first.
///
/// `registry` runs the package registry query for step 6 and is called at most
/// once, and only on Windows.
///
/// # Errors
///
/// [`PathProblem`] when `explicit` does not lead to the library.
pub fn locate(
    explicit: Option<&Path>,
    env: &SearchEnv,
    list_dir: ListDir<'_>,
    registry: msix::RegistryQuery<'_>,
) -> std::result::Result<Located, PathProblem> {
    let os = env.os;
    if let Some(path) = explicit {
        return resolve_explicit(path, os, list_dir).map(Located::File);
    }

    let mut search = Search {
        os,
        list_dir,
        searched: Vec::new(),
    };

    if let Some(dir) = &env.exe_dir
        && let Some(file) = search.check("executable directory", dir)
    {
        return Ok(Located::File(file));
    }
    if let Some(root) = &env.install_dir {
        let dir = root.join(os.install_lib_dir());
        if let Some(file) = search.check(ov::INSTALL_DIR_ENV, &dir) {
            return Ok(Located::File(file));
        }
    }
    for dir in &env.loader_path {
        if let Some(file) = search.check("loader path", dir) {
            return Ok(Located::File(file));
        }
    }
    for dir in os.system_lib_dirs() {
        if let Some(file) = search.check("system library directory", Path::new(dir)) {
            return Ok(Located::File(file));
        }
    }
    for parent in env.install_parents() {
        let roots = install_roots(&parent, list_dir);
        if roots.is_empty() {
            search.searched.push(format!(
                "  install directory: {} (none)",
                parent
                    .join(format!("{}*", ov::INSTALL_ROOT_PREFIX))
                    .display()
            ));
        }
        for root in roots {
            if let Some(file) = search.check("install directory", &root.join(os.install_lib_dir()))
            {
                return Ok(Located::File(file));
            }
        }
    }
    if os == Os::Windows
        && let Some(file) = find_in_store_packages(&mut search, env, registry)
    {
        return Ok(Located::File(file));
    }
    Ok(Located::NotFound {
        searched: search.searched,
    })
}

/// How a library load failed, as far as the message tells.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadFailure {
    /// The runtime is older than the bindings support; carries its version.
    TooOld(String),
    /// The loader refused the file or one of its dependencies.
    CouldNotOpen(String),
    /// Anything else; the original text.
    Other(String),
}

/// Classify an error string from `openvino_sys::library::load_from` or
/// `load`.
///
/// The three shapes come from `openvino-sys` 0.11.1: the version check at
/// `src/lib.rs:86`, the loader failure at `src/linking/runtime.rs:110`, and the
/// finder failure at `src/linking/runtime.rs:81`, which `Other` carries as is.
#[must_use]
pub fn classify_load_error(message: &str) -> LoadFailure {
    if message.starts_with("OpenVINO version is too old") {
        let found = message.rsplit(": ").next().unwrap_or(message);
        return LoadFailure::TooOld(found.trim().to_string());
    }
    if message.contains("could not be opened") {
        return LoadFailure::CouldNotOpen(message.to_string());
    }
    LoadFailure::Other(message.to_string())
}

impl LoadFailure {
    fn into_error(self) -> Error {
        match self {
            Self::TooOld(found) => Error::OpenVinoRuntimeTooOld {
                found,
                minimum: ov::MIN_RUNTIME_VERSION,
            },
            Self::CouldNotOpen(reason) | Self::Other(reason) => {
                Error::OpenVinoRuntimeLoadFailed { reason }
            }
        }
    }
}

/// The error for a platform without an `OpenVINO` GPU backend.
#[must_use]
pub fn unsupported_platform() -> Error {
    Error::OpenVinoUnsupported {
        reason: "this platform is not supported; the OpenVINO backend runs on Linux and \
                 Windows, where OpenVINO has a GPU plugin"
            .to_string(),
    }
}

/// A DRM render node and what is known about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderNode {
    /// Device path, e.g. `/dev/dri/renderD128`.
    pub path: String,
    /// PCI vendor id from sysfs, when readable.
    pub vendor: Option<u32>,
    /// What happened when this process opened the node read-write.
    pub access: NodeAccess,
}

/// The outcome of opening a render node read-write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeAccess {
    /// The node opened.
    Open,
    /// The open was refused for lack of permission.
    Denied,
    /// The open failed for another reason, such as the node having gone.
    Failed,
}

/// What the render nodes say about why no GPU is usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Diagnosis {
    /// There is no render node at all: no GPU driver, or no `/dev/dri` (a
    /// container without the device).
    NoRenderNodes,
    /// An Intel render node exists but this user cannot open it.
    AccessDenied {
        /// Path of the node.
        node: String,
    },
    /// An Intel render node can be opened but `OpenVINO` found no GPU: the
    /// `OpenCL` compute runtime is probably missing.
    ComputeRuntimeMissing,
    /// Render nodes exist, and every one has a known vendor other than Intel.
    NoIntelGpu,
}

/// Decide what the render nodes imply for a host where `OpenVINO` found no
/// usable Intel GPU. Pure over the node records.
///
/// `None` when the nodes do not support a diagnosis: a node whose vendor could
/// not be read may be an Intel GPU, and an Intel node that failed to open for a
/// reason other than permissions says nothing about the user's access.
#[must_use]
pub fn diagnose(nodes: &[RenderNode]) -> Option<Diagnosis> {
    let intel: Vec<&RenderNode> = nodes
        .iter()
        .filter(|node| node.vendor == Some(ov::INTEL_PCI_VENDOR_ID))
        .collect();
    if nodes.is_empty() {
        Some(Diagnosis::NoRenderNodes)
    } else if intel.is_empty() {
        nodes
            .iter()
            .all(|node| node.vendor.is_some())
            .then_some(Diagnosis::NoIntelGpu)
    } else if let Some(node) = intel.iter().find(|node| node.access == NodeAccess::Denied) {
        Some(Diagnosis::AccessDenied {
            node: node.path.clone(),
        })
    } else if intel.iter().any(|node| node.access == NodeAccess::Open) {
        Some(Diagnosis::ComputeRuntimeMissing)
    } else {
        None
    }
}

/// The diagnosis to attach to a "device not available" error, if any.
///
/// Only Linux has render nodes to read, and only a host where `OpenVINO` lists
/// no Intel GPU at all needs one. `gather` is called only then, because
/// gathering opens each node.
#[must_use]
pub fn render_diagnosis(
    os: Option<Os>,
    any_selectable: bool,
    gather: impl FnOnce() -> Vec<RenderNode>,
) -> Option<Diagnosis> {
    (os == Some(Os::Linux) && !any_selectable).then(|| diagnose(&gather()))?
}

impl Diagnosis {
    /// Text appended to a device-unavailable error, starting with a newline.
    #[must_use]
    pub fn hint(&self) -> String {
        match self {
            Self::NoRenderNodes => format!(
                "\nNo GPU render node exists under {}. Check that the Intel GPU driver is \
                 installed and, in a container, that /dev/dri is passed through",
                ov::DRM_DEVICE_DIR
            ),
            Self::AccessDenied { node } => format!(
                "\nThe Intel GPU render node {node} cannot be opened by this user; \
                 add the user to the 'render' group and log in again"
            ),
            Self::ComputeRuntimeMissing => "\nAn Intel GPU render node is present and can be \
                 opened, but OpenVINO lists no GPU. Install the Intel compute runtime \
                 (intel-opencl-icd; GPUs older than Tiger Lake need intel-opencl-icd-legacy1)"
                .to_string(),
            Self::NoIntelGpu => {
                "\nThe render nodes present do not belong to an Intel GPU".to_string()
            }
        }
    }
}

/// Read the render nodes of this host. Empty outside Linux.
#[must_use]
pub fn gather_render_nodes() -> Vec<RenderNode> {
    if Os::current() != Some(Os::Linux) {
        return Vec::new();
    }
    let Some(names) = list_dir_fs(Path::new(ov::DRM_DEVICE_DIR)) else {
        return Vec::new();
    };
    let mut nodes: Vec<RenderNode> = names
        .into_iter()
        .filter(|name| name.starts_with(ov::RENDER_NODE_PREFIX))
        .map(|name| {
            let path = Path::new(ov::DRM_DEVICE_DIR).join(&name);
            let vendor = std::fs::read_to_string(
                Path::new(ov::DRM_SYSFS_DIR)
                    .join(&name)
                    .join("device/vendor"),
            )
            .ok()
            .and_then(|text| u32::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok());
            let access = match std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
            {
                Ok(_) => NodeAccess::Open,
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => NodeAccess::Denied,
                Err(_) => NodeAccess::Failed,
            };
            RenderNode {
                path: path.display().to_string(),
                vendor,
                access,
            }
        })
        .collect();
    nodes.sort_by(|a, b| a.path.cmp(&b.path));
    nodes
}

/// A loaded `OpenVINO` runtime and the devices it reported.
pub struct Runtime {
    /// The core handle, used to read properties and compile models.
    pub core: Core,
    /// Runtime build number, e.g. `2025.4.1-19314-...`.
    pub version: String,
    /// The library file that was loaded, when it was loaded by path.
    pub library: Option<PathBuf>,
    /// Every device the runtime listed, in its own order.
    pub devices: Vec<OvDevice>,
}

impl Runtime {
    /// Locate, load and enumerate the runtime.
    ///
    /// # Errors
    ///
    /// The `OpenVino*` errors for a missing, unloadable, too old or
    /// unsupported-platform runtime.
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        let Some(os) = Os::current() else {
            return Err(unsupported_platform());
        };
        let env = SearchEnv::from_process(os);
        let located = locate(
            explicit,
            &env,
            &list_dir_fs,
            &msix::query_package_repository,
        )
        .map_err(|problem| Error::OpenVinoPathInvalid {
            path: explicit.map(Path::to_path_buf).unwrap_or_default(),
            reason: problem.reason,
        })?;

        let library = open_located(located, os, &mut |target, origin| {
            open_library(target, origin, os)
        })?;
        // A library that passed the version check can still lack a function
        // the core or the device listing calls, which panics in the binding
        // like a missing version function does.
        let shown = library.clone();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| create_runtime(library)))
            .unwrap_or_else(|_| {
                Err(Error::OpenVinoRuntimeLoadFailed {
                    reason: format!(
                        "{} lacks a function of the OpenVINO C API that birda calls",
                        shown.as_deref().map_or_else(
                            || os.library_file().to_string(),
                            |path| path.display().to_string()
                        )
                    ),
                })
            })
    }
}

/// Where the path handed to the opener came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    /// A file the search or the user chose.
    Located,
    /// A bare library name left to the system loader.
    LoaderSearch,
}

/// Open the runtime library for a search result, through `open`.
///
/// Returns the file that was opened, or `None` when the system loader found
/// the bare library name. `open` is a parameter so the decisions here can be
/// tested without a runtime:
///
/// * a located file is opened as it is, and a failure is final on every OS: a
///   path the search or the user chose never falls back to another runtime.
///   Only a located file has its dependencies preloaded (see
///   [`Origin::Located`]);
/// * with nothing located, only Linux asks the system loader for the bare name
///   (see [`Os::has_loader_fallback`]); Windows reports the places searched.
fn open_located(
    located: Located,
    os: Os,
    open: &mut dyn FnMut(&Path, Origin) -> std::result::Result<(), LoadFailure>,
) -> Result<Option<PathBuf>> {
    match located {
        Located::File(file) => {
            open(&file, Origin::Located).map_err(LoadFailure::into_error)?;
            Ok(Some(file))
        }
        Located::NotFound { searched } => {
            // The Linux loader has its own search path (ld.so.conf.d, RUNPATH)
            // that the directory search cannot see. On Windows a bare name
            // would also be searched for in the working directory.
            if !os.has_loader_fallback() {
                return Err(runtime_not_found(&searched, os, None));
            }
            open(Path::new(os.library_file()), Origin::LoaderSearch)
                .map_err(|failure| fallback_error(&searched, os, failure))?;
            Ok(None)
        }
    }
}

/// The error for a runtime that no searched place has.
///
/// `loader_reason` is what the system loader said when it was asked for the
/// bare library name, when it was asked.
fn runtime_not_found(searched: &[String], os: Os, loader_reason: Option<&str>) -> Error {
    let mut lines = searched.to_vec();
    if let Some(reason) = loader_reason {
        lines.push(format!(
            "  the system loader path: {} ({reason})",
            os.library_file()
        ));
    }
    Error::OpenVinoRuntimeNotFound {
        searched: lines.join("\n"),
        library: os.library_file(),
    }
}

/// Map a failed load of the bare library name through the system loader.
///
/// A loader reason that starts with the library's own name means the library
/// is not there, which is "not found". A reason that names another file means
/// the library was found and one of its dependencies was not, which is a load
/// failure that carries the loader's message. The loader prints a missing file
/// as `<file>: cannot open shared object file` on glibc; the dependency case
/// is NOT MEASURED, the missing-library case is the shape in the test below.
fn fallback_error(searched: &[String], os: Os, failure: LoadFailure) -> Error {
    match failure {
        LoadFailure::CouldNotOpen(message) => {
            let reason = message
                .split_once("could not be opened: ")
                .map_or(message.as_str(), |(_, reason)| reason);
            if reason.starts_with(os.library_file()) {
                runtime_not_found(searched, os, Some(reason))
            } else {
                LoadFailure::CouldNotOpen(message).into_error()
            }
        }
        // `load_from` never reports "library not found" in any other shape, so
        // these mean the library opened and then failed (version query, age).
        other => other.into_error(),
    }
}

/// The library this process handed to `openvino-sys`, once one was opened.
///
/// `openvino-sys` 0.11.1 keeps the first library it opens for the life of the
/// process and turns every later `load_from` into a no-op
/// (`src/linking/runtime.rs:97-100`), so a different path asked for later would
/// be reported while the first library does the work.
static LOADED_LIBRARY: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Open the runtime library at `target` through `openvino-sys`.
///
/// `openvino-sys` 0.11.1 reads the version through a function it looked up in
/// the library, and panics when that function is missing (`src/lib.rs:92-100`
/// and `src/linking/runtime.rs:128-134`): any shared library that is not the
/// `OpenVINO` C API does that. The panic is caught and reported as a load
/// failure. The binding's own panic message is still printed first.
///
/// For a [`Origin::Located`] target, the oneTBB and core libraries beside it
/// are opened first (see [`preload`]), but only while no runtime is held by
/// this process: no file of a second install is loaded once any file of a first
/// one is. The rules are in [`guarded_load`].
fn open_library(target: &Path, origin: Origin, os: Os) -> std::result::Result<(), LoadFailure> {
    let mut loaded = LOADED_LIBRARY
        .lock()
        .map_err(|_| LoadFailure::Other("an earlier OpenVINO runtime load failed".to_string()))?;
    guarded_load(
        &mut loaded,
        target,
        origin,
        &mut |path| {
            preload::dependency_files(path, &os.layout(), &list_dir_fs)
                .map(|files| preload::preload(&files))
        },
        &mut |path| {
            let outcome = std::panic::catch_unwind(|| openvino_sys::library::load_from(path));
            match outcome {
                Ok(Ok(())) => None,
                Ok(Err(message)) => Some(classify_load_error(&message)),
                Err(_) => Some(LoadFailure::Other(format!(
                    "{} is not the OpenVINO C API library: it lacks the functions birda calls",
                    path.display()
                ))),
            }
        },
    )
}

/// Load `target` under the one-runtime rule, through two injected steps.
///
/// `preload` opens the dependencies of a path and returns whether any opened,
/// or why none were selected. `load` runs the real load and returns how it
/// failed. A located target is made absolute first, so a relative
/// `INTEL_OPENVINO_DIR` or `PATH` entry never reaches the loader as a relative
/// path, and that one path is used for the preload, the load and the record.
fn guarded_load(
    loaded: &mut Option<PathBuf>,
    target: &Path,
    origin: Origin,
    preload: &mut dyn FnMut(&Path) -> std::result::Result<bool, String>,
    load: &mut dyn FnMut(&Path) -> Option<LoadFailure>,
) -> std::result::Result<(), LoadFailure> {
    let target = match origin {
        Origin::Located => std::path::absolute(target).unwrap_or_else(|_| target.to_path_buf()),
        Origin::LoaderSearch => target.to_path_buf(),
    };
    let target = target.as_path();
    let wanted = admit(loaded.as_deref(), target, origin)?;
    let (preloaded, skipped) = if wanted {
        match preload(target) {
            Ok(opened) => (opened, None),
            Err(reason) => {
                debug!(
                    "not preloading the dependencies of {}: {reason}",
                    target.display()
                );
                (false, Some(reason))
            }
        }
    } else {
        (false, None)
    };
    let failure = load(target);
    record(loaded, target, preloaded, failure.as_ref());
    with_skipped_preload(failure, skipped.as_deref()).map_or(Ok(()), Err)
}

/// Whether to preload for `target`, or why it may not be loaded at all.
///
/// A runtime other than the one already held is refused. The dependencies are
/// preloaded only for a located file, and only while no runtime is held.
fn admit(
    previous: Option<&Path>,
    target: &Path,
    origin: Origin,
) -> std::result::Result<bool, LoadFailure> {
    if let Some(previous) = previous
        && previous != target
    {
        return Err(LoadFailure::Other(format!(
            "this process already loaded the OpenVINO runtime {}, and a process can load only \
             one; {} was not loaded",
            previous.display(),
            target.display()
        )));
    }
    Ok(previous.is_none() && origin == Origin::Located)
}

/// Note that this process now holds files of the runtime at `target`.
///
/// A library the loader refused is not held by the binding, but dependencies
/// preloaded for it stay resident and are never unloaded, so the target is
/// recorded and no other runtime is loaded afterwards.
fn record(
    loaded: &mut Option<PathBuf>,
    target: &Path,
    preloaded: bool,
    failure: Option<&LoadFailure>,
) {
    if preloaded || !matches!(failure, Some(LoadFailure::CouldNotOpen(_))) {
        *loaded = Some(target.to_path_buf());
    }
}

/// Add why nothing was preloaded to a refused load, so the error names the
/// dependency that is missing instead of only the file the loader reported.
fn with_skipped_preload(
    failure: Option<LoadFailure>,
    skipped: Option<&str>,
) -> Option<LoadFailure> {
    match (failure, skipped) {
        (Some(LoadFailure::CouldNotOpen(message)), Some(reason)) => {
            Some(LoadFailure::CouldNotOpen(format!(
                "{message} (birda did not preload the runtime's dependencies: {reason})"
            )))
        }
        (failure, _) => failure,
    }
}

fn create_runtime(library: Option<PathBuf>) -> Result<Runtime> {
    let core = Core::new().map_err(|e| Error::OpenVinoRuntimeLoadFailed {
        reason: e.to_string(),
    })?;
    let version = openvino::version().build_number;
    let devices = enumerate(&core)?;
    Ok(Runtime {
        core,
        version,
        library,
        devices,
    })
}

/// Read one property as a string, `None` when the device lacks it.
fn read(core: &Core, device: &DeviceType<'_>, key: &str) -> Option<String> {
    core.get_property(device, &PropertyKey::Other(key.to_string().into()))
        .ok()
        .filter(|value| !value.is_empty())
}

/// List the devices and read the properties birda uses.
///
/// Properties are read only from GPUs: the CPU device is never selectable, so
/// it is listed by name alone. A property that cannot be read is left unset.
fn enumerate(core: &Core) -> Result<Vec<OvDevice>> {
    let names: Vec<String> = core
        .available_devices()
        .map_err(|e| Error::OpenVinoDeviceListing {
            reason: e.to_string(),
        })?
        .iter()
        .map(ToString::to_string)
        .collect();

    Ok(names
        .iter()
        .map(|name| {
            let mut device = OvDevice::named(name);
            if !matches!(device.kind, super::device::DeviceKind::Gpu(_)) {
                return device;
            }
            let id = DeviceType::from(name.as_str());
            device.full_name = core
                .get_property(&id, &PropertyKey::DeviceFullName)
                .ok()
                .filter(|value| !value.is_empty());
            device.class = read(core, &id, "DEVICE_TYPE")
                .map_or(super::device::GpuClass::Unknown, |value| {
                    parse_gpu_class(&value)
                });
            device.architecture = read(core, &id, "DEVICE_ARCHITECTURE");
            device.intel = device
                .architecture
                .as_deref()
                .is_some_and(is_intel_architecture);
            device.uuid = read(core, &id, "DEVICE_UUID").and_then(|value| parse_uuid(&value));
            device.pci =
                read(core, &id, "DEVICE_PCI_INFO").and_then(|value| parse_pci_info(&value));
            device.memory_bytes = read(core, &id, "GPU_DEVICE_TOTAL_MEM_SIZE")
                .and_then(|value| parse_memory_bytes(&value));
            device.capabilities = core
                .get_property(&id, &PropertyKey::DeviceCapabilities)
                .map(|value| parse_capabilities(&value))
                .unwrap_or_default();
            device
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn fake_fs(entries: &[(&str, &[&str])]) -> HashMap<PathBuf, Vec<String>> {
        entries
            .iter()
            .map(|(dir, names)| {
                (
                    PathBuf::from(dir),
                    names.iter().map(ToString::to_string).collect(),
                )
            })
            .collect()
    }

    fn env_for(os: Os) -> SearchEnv {
        SearchEnv {
            os,
            exe_dir: None,
            install_dir: None,
            loader_path: Vec::new(),
            program_files: PathBuf::from(ov::DEFAULT_PROGRAM_FILES),
            program_files_x86: PathBuf::from(ov::DEFAULT_PROGRAM_FILES_X86),
        }
    }

    fn linux_env() -> SearchEnv {
        env_for(Os::Linux)
    }

    fn locate_in(
        explicit: Option<&str>,
        env: &SearchEnv,
        fs: &HashMap<PathBuf, Vec<String>>,
    ) -> std::result::Result<Located, PathProblem> {
        locate(
            explicit.map(Path::new),
            env,
            &|dir| fs.get(dir).cloned(),
            &|| None,
        )
    }

    fn file(path: &str) -> Located {
        Located::File(PathBuf::from(path))
    }

    // Unix layout only: the fixtures use drive-less absolute paths, and on
    // Windows `std::path::absolute` adds the current drive to the explicit
    // path before the lookup, so the keys would not match.
    #[cfg(unix)]
    #[test]
    fn an_explicit_file_or_directory_is_used_and_nothing_else_is_searched() {
        let fs = fake_fs(&[
            (
                "/opt/ov/lib",
                &["libopenvino_c.so", "libopenvino.so.2025.4.1"],
            ),
            ("/opt/ov/root", &["runtime"]),
            ("/opt/ov/root/runtime/lib/intel64", &["libopenvino_c.so"]),
            ("/usr/lib", &["libopenvino_c.so"]),
        ]);
        let env = linux_env();
        assert_eq!(
            locate_in(Some("/opt/ov/lib/libopenvino_c.so"), &env, &fs),
            Ok(file("/opt/ov/lib/libopenvino_c.so"))
        );
        assert_eq!(
            locate_in(Some("/opt/ov/lib"), &env, &fs),
            Ok(file("/opt/ov/lib/libopenvino_c.so"))
        );
        // An install root resolves through its runtime/lib/intel64 directory.
        assert_eq!(
            locate_in(Some("/opt/ov/root"), &env, &fs),
            Ok(file("/opt/ov/root/runtime/lib/intel64/libopenvino_c.so"))
        );
        // A configured path that does not exist stops the search even though
        // /usr/lib has a runtime.
        assert_eq!(
            locate_in(Some("/nonexistent"), &env, &fs),
            Err(PathProblem {
                reason: "it does not exist".to_string()
            })
        );
        // An existing file is passed on whatever its name; the loader judges it.
        assert_eq!(
            locate_in(Some("/opt/ov/lib/libopenvino.so.2025.4.1"), &env, &fs),
            Ok(file("/opt/ov/lib/libopenvino.so.2025.4.1"))
        );
    }

    #[test]
    fn a_bare_file_name_names_the_file_in_the_working_directory() {
        let cwd = std::env::current_dir().unwrap();
        let fs = HashMap::from([(cwd.clone(), vec!["libopenvino_c.so".to_string()])]);
        let expected = Ok(Located::File(cwd.join("libopenvino_c.so")));
        assert_eq!(
            locate_in(Some("libopenvino_c.so"), &linux_env(), &fs),
            expected
        );
        assert_eq!(
            locate_in(Some("./libopenvino_c.so"), &linux_env(), &fs),
            expected
        );
    }

    // Loads a real shared library into this test process. No other unit test
    // opens a library, so the process-wide state it leaves behind is unused.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn a_library_that_is_not_openvino_fails_to_load_and_stays_the_only_one() {
        let not_openvino = if cfg!(target_os = "macos") {
            "/usr/lib/libSystem.B.dylib"
        } else {
            "libc.so.6"
        };
        assert_eq!(
            open_library(Path::new(not_openvino), Origin::LoaderSearch, Os::Linux),
            Err(LoadFailure::Other(format!(
                "{not_openvino} is not the OpenVINO C API library: it lacks the functions \
                 birda calls"
            )))
        );
        assert_eq!(
            open_library(
                Path::new("/opt/other/libopenvino_c.so"),
                Origin::Located,
                Os::Linux
            ),
            Err(LoadFailure::Other(format!(
                "this process already loaded the OpenVINO runtime {not_openvino}, and a process \
                 can load only one; /opt/other/libopenvino_c.so was not loaded"
            )))
        );
    }

    // Unix layout only: the fixtures use drive-less absolute paths, and on
    // Windows `std::path::absolute` adds the current drive to the explicit
    // path before the lookup, so the keys would not match.
    #[cfg(unix)]
    #[test]
    fn a_directory_without_the_library_names_what_is_missing() {
        let fs = fake_fs(&[("/opt/empty", &["README"])]);
        assert_eq!(
            locate_in(Some("/opt/empty"), &linux_env(), &fs),
            Err(PathProblem {
                reason: "no libopenvino_c.so in that directory or in its runtime/lib/intel64"
                    .to_string()
            })
        );
    }

    #[test]
    fn the_default_search_order_is_exe_dir_install_dir_loader_path_system_dirs_then_roots() {
        let fs = fake_fs(&[
            ("/opt/birda", &["libopenvino_c.so"]),
            ("/home/u/ov/runtime/lib/intel64", &["libopenvino_c.so"]),
            ("/custom/lib", &["libopenvino_c.so"]),
            ("/usr/lib", &["libopenvino_c.so"]),
            ("/opt/intel", &["openvino_2025", "openvino_2026", "other"]),
            (
                "/opt/intel/openvino_2026/runtime/lib/intel64",
                &["libopenvino_c.so"],
            ),
            (
                "/opt/intel/openvino_2025/runtime/lib/intel64",
                &["libopenvino_c.so"],
            ),
        ]);
        let mut env = linux_env();
        env.exe_dir = Some(PathBuf::from("/opt/birda"));
        env.install_dir = Some(PathBuf::from("/home/u/ov"));
        env.loader_path = vec![PathBuf::from("/empty"), PathBuf::from("/custom/lib")];
        assert_eq!(
            locate_in(None, &env, &fs),
            Ok(file("/opt/birda/libopenvino_c.so"))
        );
        env.exe_dir = None;
        assert_eq!(
            locate_in(None, &env, &fs),
            Ok(file("/home/u/ov/runtime/lib/intel64/libopenvino_c.so"))
        );
        env.install_dir = None;
        assert_eq!(
            locate_in(None, &env, &fs),
            Ok(file("/custom/lib/libopenvino_c.so"))
        );
        env.loader_path.clear();
        assert_eq!(
            locate_in(None, &env, &fs),
            Ok(file("/usr/lib/libopenvino_c.so"))
        );

        // With no system copy the newest install root under /opt/intel wins.
        let fs = fake_fs(&[
            ("/opt/intel", &["openvino_2025", "openvino_2026", "other"]),
            (
                "/opt/intel/openvino_2026/runtime/lib/intel64",
                &["libopenvino_c.so"],
            ),
            (
                "/opt/intel/openvino_2025/runtime/lib/intel64",
                &["libopenvino_c.so"],
            ),
        ]);
        assert_eq!(
            locate_in(None, &env, &fs),
            Ok(file(
                "/opt/intel/openvino_2026/runtime/lib/intel64/libopenvino_c.so"
            ))
        );
    }

    fn strings(names: &[&str]) -> Vec<String> {
        names.iter().map(ToString::to_string).collect()
    }

    fn windows_env(program_files: &str, program_files_x86: &str) -> SearchEnv {
        SearchEnv {
            program_files: PathBuf::from(program_files),
            program_files_x86: PathBuf::from(program_files_x86),
            ..env_for(Os::Windows)
        }
    }

    #[test]
    fn the_windows_layout_uses_its_own_library_name_and_directories() {
        // A relocated Program Files, and an x86 one that is the same place in
        // another case, so the second parent is not searched again.
        let env = windows_env("D:\\Apps", "d:\\apps");
        let parent = PathBuf::from("D:\\Apps").join("Intel");
        let newest = parent.join("openvino_2026.2");
        let lib_dir = newest.join(ov::WINDOWS_INSTALL_LIB_DIR);
        let fs: HashMap<PathBuf, Vec<String>> = HashMap::from([
            (parent, strings(&["openvino_2026.2", "openvino_2025.4"])),
            (
                lib_dir.clone(),
                strings(&["openvino_c.dll", "openvino.dll"]),
            ),
        ]);
        let locate_win = |env: &SearchEnv| locate(None, env, &|dir| fs.get(dir).cloned(), &|| None);
        assert_eq!(
            locate_win(&env),
            Ok(Located::File(lib_dir.join("openvino_c.dll")))
        );

        // Nothing holds the library: each install directory is listed once,
        // and the x86 parent, which holds no roots, is the same place as the
        // first.
        let fs_empty: HashMap<PathBuf, Vec<String>> = HashMap::from([(
            PathBuf::from("D:\\Apps").join("Intel"),
            strings(&["openvino_2026.2"]),
        )]);
        let Ok(Located::NotFound { searched }) =
            locate(None, &env, &|dir| fs_empty.get(dir).cloned(), &|| None)
        else {
            panic!("nothing is installed");
        };
        let install_lines: Vec<&String> = searched
            .iter()
            .filter(|line| line.starts_with("  install directory"))
            .collect();
        assert_eq!(
            install_lines,
            [&format!(
                "  install directory: {}",
                PathBuf::from("D:\\Apps")
                    .join("Intel")
                    .join("openvino_2026.2")
                    .join(ov::WINDOWS_INSTALL_LIB_DIR)
                    .display()
            )]
        );
    }

    #[test]
    fn a_parent_without_install_roots_is_listed_as_empty() {
        let env = windows_env("D:\\Apps", "E:\\Apps32");
        let Ok(Located::NotFound { searched }) = locate(None, &env, &|_| None, &|| None) else {
            panic!("nothing is installed");
        };
        let none_line = |root: &str| {
            format!(
                "  install directory: {} (none)",
                PathBuf::from(root)
                    .join("Intel")
                    .join("openvino*")
                    .display()
            )
        };
        assert_eq!(searched[0], none_line("D:\\Apps"));
        assert_eq!(searched[1], none_line("E:\\Apps32"));
    }

    #[test]
    fn windows_names_match_without_regard_to_case() {
        let env = windows_env("D:\\Apps", "D:\\Apps");
        let parent = PathBuf::from("D:\\Apps").join("Intel");
        let lib_dir = |root: &str| parent.join(root).join(ov::WINDOWS_INSTALL_LIB_DIR);
        let fs: HashMap<PathBuf, Vec<String>> = HashMap::from([
            (
                parent.clone(),
                strings(&["OpenVINO_2026.2", "openvino_2025.4"]),
            ),
            (lib_dir("OpenVINO_2026.2"), strings(&["OPENVINO_C.DLL"])),
            (lib_dir("openvino_2025.4"), strings(&["openvino_c.dll"])),
        ]);
        // Each of the install-root filter, its version parse and the library
        // name match must fold case, or the older root is picked.
        assert_eq!(
            locate(None, &env, &|dir| fs.get(dir).cloned(), &|| None),
            Ok(Located::File(
                lib_dir("OpenVINO_2026.2").join("OPENVINO_C.DLL")
            ))
        );
        // The folding is Windows only: a Linux library name is case sensitive.
        assert_eq!(
            pick_library(&strings(&["LIBOPENVINO_C.SO"]), Os::Linux),
            None
        );
    }

    #[test]
    fn a_configured_windows_file_matches_without_regard_to_case() {
        let cwd = std::env::current_dir().unwrap();
        let fs = HashMap::from([(cwd.clone(), strings(&["openvino_c.dll"]))]);
        // The configured spelling is kept; NTFS opens the file either way.
        assert_eq!(
            locate_in(Some("OPENVINO_C.DLL"), &env_for(Os::Windows), &fs),
            Ok(Located::File(cwd.join("OPENVINO_C.DLL")))
        );
        // Linux file names are case sensitive.
        assert_eq!(
            locate_in(Some("OPENVINO_C.DLL"), &env_for(Os::Linux), &fs),
            Err(PathProblem {
                reason: "it does not exist".to_string()
            })
        );
    }

    #[test]
    fn a_listing_with_two_spellings_gives_the_exact_one() {
        assert_eq!(
            pick_library(&strings(&["OPENVINO_C.DLL", "openvino_c.dll"]), Os::Windows),
            Some("openvino_c.dll".to_string())
        );
    }

    #[test]
    fn a_readable_listing_without_a_usable_package_still_asks_the_registry() {
        const PHOTOS: &str = "Microsoft.Photos_2030.1.0.0_x64__8wekyb3d8bbwe";
        const OLD: &str = "Intel.OpenVINOToolkit.X_2026.9.0.0_x64__8wekyb3d8bbwe";
        const NEW: &str = "Intel.OpenVINOToolkit.X_2026.10.0.0_x64__8wekyb3d8bbwe";
        let env = windows_env("D:\\Apps", "D:\\Apps");
        let apps = PathBuf::from("D:\\Apps").join(ov::WINDOWS_APPS_DIR);
        let lib = |root: &Path| root.join(ov::WINDOWS_INSTALL_LIB_DIR);
        let entry = |name: &str, folder: &str| {
            format!(
                "{}\\{name}\r\n    Path    REG_SZ    {folder}\r\n\r\n",
                ov::PACKAGE_REPOSITORY_KEY
            )
        };

        // A package on another drive: the listing is readable but holds no
        // OpenVINO package, so the registry is what finds the install.
        let elsewhere = PathBuf::from("E:\\Store").join(OLD);
        let registry = entry(OLD, &elsewhere.display().to_string());
        let fs: HashMap<PathBuf, Vec<String>> = HashMap::from([
            (apps.clone(), strings(&[PHOTOS])),
            (lib(&elsewhere), strings(&["openvino_c.dll"])),
        ]);
        assert_eq!(
            locate(None, &env, &|dir| fs.get(dir).cloned(), &|| Some(
                registry.clone()
            )),
            Ok(Located::File(lib(&elsewhere).join("openvino_c.dll")))
        );

        // A stale listed package that the registry also names, in another
        // case, is checked once; the registry's other package follows it.
        let listed = apps.join(OLD);
        let gone = PathBuf::from("D:\\Gone").join(NEW);
        let registry = format!(
            "{}{}",
            entry(OLD, &listed.display().to_string().to_lowercase()),
            entry(NEW, &gone.display().to_string())
        );
        let fs: HashMap<PathBuf, Vec<String>> = HashMap::from([(apps, strings(&[OLD]))]);
        let Ok(Located::NotFound { searched }) =
            locate(None, &env, &|dir| fs.get(dir).cloned(), &|| {
                Some(registry.clone())
            })
        else {
            panic!("nothing is installed");
        };
        let store_lines: Vec<&String> = searched
            .iter()
            .filter(|line| line.starts_with("  Microsoft Store package"))
            .collect();
        assert_eq!(
            store_lines,
            [
                &format!("  Microsoft Store package: {}", lib(&listed).display()),
                &format!("  Microsoft Store package: {}", lib(&gone).display()),
            ]
        );
    }

    #[test]
    fn store_packages_come_last_and_fall_back_to_the_registry() {
        const OLD: &str = "Intel.OpenVINOToolkit.X_2026.9.0.0_x64__8wekyb3d8bbwe";
        const NEW: &str = "Intel.OpenVINOToolkit.X_2026.10.0.0_x64__8wekyb3d8bbwe";
        let env = windows_env("D:\\Apps", "D:\\Apps");
        let apps = PathBuf::from("D:\\Apps").join(ov::WINDOWS_APPS_DIR);
        let lib = |root: &Path| root.join(ov::WINDOWS_INSTALL_LIB_DIR);
        let panics = || -> Option<String> { panic!("the registry must not be queried") };

        // a. The newest package by version, not by string, wins from the
        // listing, and the registry is not asked.
        let fs: HashMap<PathBuf, Vec<String>> = HashMap::from([
            (apps.clone(), strings(&[OLD, NEW])),
            (lib(&apps.join(OLD)), strings(&["openvino_c.dll"])),
            (lib(&apps.join(NEW)), strings(&["openvino_c.dll"])),
        ]);
        assert_eq!(
            locate(None, &env, &|dir| fs.get(dir).cloned(), &panics),
            Ok(Located::File(lib(&apps.join(NEW)).join("openvino_c.dll")))
        );

        // b. A listing the user may not read: the registry names a newer
        // package whose folder is gone and an older one that is installed.
        let registry = format!(
            "{key}\\{NEW}\r\n    Path    REG_SZ    D:\\Gone\\{NEW}\r\n\r\n\
             {key}\\{OLD}\r\n    Path    REG_SZ    D:\\Store\\{OLD}\r\n",
            key = ov::PACKAGE_REPOSITORY_KEY
        );
        let installed = lib(Path::new(&format!("D:\\Store\\{OLD}")));
        let fs: HashMap<PathBuf, Vec<String>> =
            HashMap::from([(installed.clone(), strings(&["openvino_c.dll"]))]);
        assert_eq!(
            locate(None, &env, &|dir| fs.get(dir).cloned(), &|| Some(
                registry.clone()
            )),
            Ok(Located::File(installed.join("openvino_c.dll")))
        );
        let none_installed: HashMap<PathBuf, Vec<String>> = HashMap::new();
        let Ok(Located::NotFound { searched }) =
            locate(None, &env, &|dir| none_installed.get(dir).cloned(), &|| {
                Some(registry.clone())
            })
        else {
            panic!("nothing is installed");
        };
        let store_lines: Vec<&String> = searched
            .iter()
            .filter(|line| line.starts_with("  Microsoft Store package"))
            .collect();
        assert_eq!(
            store_lines,
            [
                &format!(
                    "  Microsoft Store package: {}",
                    lib(Path::new(&format!("D:\\Gone\\{NEW}"))).display()
                ),
                &format!(
                    "  Microsoft Store package: {}",
                    lib(Path::new(&format!("D:\\Store\\{OLD}"))).display()
                ),
            ]
        );

        // c. Nothing anywhere: one line says so.
        let Ok(Located::NotFound { searched }) = locate(None, &env, &|_| None, &|| None) else {
            panic!("nothing is installed");
        };
        assert_eq!(
            searched.last(),
            Some(&format!(
                "  Microsoft Store packages: none found in {} or the package registry",
                apps.display()
            ))
        );

        // d. An install root under Program Files\Intel beats a Store package.
        let intel = PathBuf::from("D:\\Apps").join("Intel");
        let root_lib = lib(&intel.join("openvino_2026.2"));
        let fs: HashMap<PathBuf, Vec<String>> = HashMap::from([
            (intel, strings(&["openvino_2026.2"])),
            (root_lib.clone(), strings(&["openvino_c.dll"])),
            (apps.clone(), strings(&[NEW])),
            (lib(&apps.join(NEW)), strings(&["openvino_c.dll"])),
        ]);
        assert_eq!(
            locate(None, &env, &|dir| fs.get(dir).cloned(), &panics),
            Ok(Located::File(root_lib.join("openvino_c.dll")))
        );
    }

    #[test]
    fn a_versioned_library_is_picked_only_when_the_plain_name_is_absent() {
        let versioned = |names: &[&str]| {
            pick_library(
                &names.iter().map(ToString::to_string).collect::<Vec<_>>(),
                Os::Linux,
            )
        };
        assert_eq!(
            versioned(&[
                "libopenvino_c.so.2025.4.1",
                "libopenvino_c.so.2025.10.0",
                "libopenvino_c.so.2025.9.9",
                "libopenvino_c.so.bak",
                "libopenvino_c.so.2025.x",
                "libopenvino.so.2026.0.0",
            ]),
            Some("libopenvino_c.so.2025.10.0".to_string())
        );
        assert_eq!(
            versioned(&["libopenvino_c.so.2025.4.1", "libopenvino_c.so"]),
            Some("libopenvino_c.so".to_string())
        );
        assert_eq!(versioned(&["libopenvino.so"]), None);
        // Windows has no versioned names.
        assert_eq!(
            pick_library(&["openvino_c.dll.1".to_string()], Os::Windows),
            None
        );
    }

    #[test]
    fn not_found_lists_every_place_searched() {
        let fs = fake_fs(&[("/usr/lib", &["libm.so"])]);
        let mut env = linux_env();
        env.exe_dir = Some(PathBuf::from("/opt/birda"));
        env.install_dir = Some(PathBuf::from("/home/u/ov"));
        env.loader_path = vec![PathBuf::from("/custom/lib")];
        let Ok(Located::NotFound { searched }) = locate_in(None, &env, &fs) else {
            panic!("nothing is installed");
        };
        let under_root = Path::new("/home/u/ov").join(ov::LINUX_INSTALL_LIB_DIR);
        assert_eq!(searched[0], "  executable directory: /opt/birda");
        assert_eq!(
            searched[1],
            format!("  INTEL_OPENVINO_DIR: {}", under_root.display())
        );
        assert_eq!(searched[2], "  loader path: /custom/lib");
        assert_eq!(searched[3], "  system library directory: /usr/lib");
        let empty_parent = Path::new("/opt/intel").join("openvino*");
        assert_eq!(
            searched.last(),
            Some(&format!(
                "  install directory: {} (none)",
                empty_parent.display()
            ))
        );
        // Executable directory, INTEL_OPENVINO_DIR, loader path, the system
        // directories, then the one empty install parent.
        assert_eq!(searched.len(), 3 + ov::LINUX_SYSTEM_LIB_DIRS.len() + 1);
    }

    #[test]
    fn load_errors_are_classified_from_the_literal_binding_messages() {
        assert_eq!(
            classify_load_error(
                "OpenVINO version is too old (see https://github.com/intel/openvino-rs/issues/167): 2024.6.0-17404"
            ),
            LoadFailure::TooOld("2024.6.0-17404".to_string())
        );
        let opened = "the shared library at /x/libopenvino_c.so could not be opened: \
                      libopenvino.so.2620: cannot open shared object file";
        assert_eq!(
            classify_load_error(opened),
            LoadFailure::CouldNotOpen(opened.to_string())
        );
        assert_eq!(
            classify_load_error("Unable to find the `openvino_c` library to load"),
            LoadFailure::Other("Unable to find the `openvino_c` library to load".to_string())
        );
    }

    #[test]
    fn fallback_error_tells_a_missing_runtime_from_a_failed_dependency() {
        let searched = vec!["  system library directory: /usr/lib".to_string()];
        let opened = |reason: &str| {
            LoadFailure::CouldNotOpen(format!(
                "the shared library at libopenvino_c.so could not be opened: {reason}"
            ))
        };

        // The library itself is absent: not found, with the loader's reason.
        assert_eq!(
            fallback_error(
                &searched,
                Os::Linux,
                opened(
                    "libopenvino_c.so: cannot open shared object file: No such file or directory"
                )
            )
            .to_string(),
            "OpenVINO runtime not found. Searched:\n\
             \x20 system library directory: /usr/lib\n\
             \x20 the system loader path: libopenvino_c.so (libopenvino_c.so: cannot open shared object file: No such file or directory)\n\
             Install OpenVINO 2025.1 or newer (see docs/openvino.md), set \
             INTEL_OPENVINO_DIR to its install root, or set inference.openvino_path \
             in the config file to the directory holding libopenvino_c.so"
        );

        // A dependency is absent: the runtime was found, so this is a load failure.
        assert_eq!(
            fallback_error(
                &searched,
                Os::Linux,
                opened("libtbb.so.12: cannot open shared object file: No such file or directory")
            )
            .to_string(),
            "could not load the OpenVINO runtime: the shared library at libopenvino_c.so could not be \
             opened: libtbb.so.12: cannot open shared object file: No such file or directory\n\
             birda loads the TBB and core libraries of a runtime it located from that \
             install. On Linux, a runtime found only through the system loader needs \
             its TBB directory on LD_LIBRARY_PATH or in ld.so.conf.d (then run \
             ldconfig). Check that the OpenVINO install is complete \
             (see docs/openvino.md)"
        );

        // A runtime that opened and then failed its version query is a load
        // failure, not a missing one.
        assert!(matches!(
            fallback_error(
                &searched,
                Os::Linux,
                LoadFailure::Other("failed to get OpenVINO version: GENERAL_ERROR".to_string())
            ),
            Error::OpenVinoRuntimeLoadFailed { .. }
        ));

        // A too old runtime stays too old.
        assert!(matches!(
            fallback_error(
                &searched,
                Os::Linux,
                LoadFailure::TooOld("2024.6".to_string())
            ),
            Error::OpenVinoRuntimeTooOld { .. }
        ));
    }

    #[test]
    fn install_roots_are_ordered_by_parsed_version() {
        let ordered = |names: &[&str]| {
            order_install_roots(names.iter().map(ToString::to_string).collect::<Vec<_>>())
        };
        // 2025.10 is newer than 2025.4, which a string sort gets backwards.
        assert_eq!(
            ordered(&["openvino_2025.4", "openvino_2025.10"]),
            ["openvino_2025.10", "openvino_2025.4"]
        );
        // A later year beats a higher minor of the year before.
        assert_eq!(
            ordered(&["openvino_2025.10", "openvino_2026.2"]),
            ["openvino_2026.2", "openvino_2025.10"]
        );
        // A bare name is version 0: below every numbered root, above an
        // unparsable one. Missing components count as 0.
        assert_eq!(
            ordered(&[
                "openvino_toolkit_ubuntu22_2026.2.0",
                "openvino",
                "openvino_2025",
                "openvino_2025.0.1",
                "openvino_2025.0",
            ]),
            [
                "openvino_2025.0.1",
                "openvino_2025.0",
                "openvino_2025",
                "openvino",
                "openvino_toolkit_ubuntu22_2026.2.0",
            ]
        );
        // Equal versions (2026 and 2026.0) fall back to the name, descending;
        // unparsable names also go by name, descending.
        assert_eq!(
            ordered(&["openvino_2026", "openvino_2026.0"]),
            ["openvino_2026.0", "openvino_2026"]
        );
        assert_eq!(
            ordered(&["openvino_a", "openvino_b"]),
            ["openvino_b", "openvino_a"]
        );
    }

    #[test]
    fn the_install_root_search_takes_the_newest_version_not_the_last_name() {
        let fs = fake_fs(&[
            ("/opt/intel", &["openvino_2025.4", "openvino_2025.10"]),
            (
                "/opt/intel/openvino_2025.10/runtime/lib/intel64",
                &["libopenvino_c.so"],
            ),
            (
                "/opt/intel/openvino_2025.4/runtime/lib/intel64",
                &["libopenvino_c.so"],
            ),
        ]);
        assert_eq!(
            locate_in(None, &linux_env(), &fs),
            Ok(file(
                "/opt/intel/openvino_2025.10/runtime/lib/intel64/libopenvino_c.so"
            ))
        );
    }

    /// Records every path handed to the loader, with where it came from, and
    /// answers from a script.
    fn scripted_open(
        calls: &mut Vec<(String, Origin)>,
        fail_with: Option<LoadFailure>,
    ) -> impl FnMut(&Path, Origin) -> std::result::Result<(), LoadFailure> + '_ {
        move |path, origin| {
            calls.push((path.display().to_string(), origin));
            fail_with.clone().map_or(Ok(()), Err)
        }
    }

    fn nothing_found() -> Located {
        Located::NotFound {
            searched: vec!["  system library directory: /usr/lib".to_string()],
        }
    }

    #[test]
    fn nothing_located_asks_the_loader_for_the_bare_name_on_linux_only() {
        let mut calls = Vec::new();
        let opened = open_located(
            nothing_found(),
            Os::Linux,
            &mut scripted_open(&mut calls, None),
        );
        assert_eq!(opened.unwrap(), None);
        assert_eq!(
            calls,
            [("libopenvino_c.so".to_string(), Origin::LoaderSearch)]
        );

        let mut calls = Vec::new();
        let err = open_located(
            nothing_found(),
            Os::Windows,
            &mut scripted_open(&mut calls, None),
        )
        .unwrap_err();
        assert_eq!(calls, [], "Windows must not load a bare name");
        assert_eq!(
            err.to_string(),
            "OpenVINO runtime not found. Searched:\n\
             \x20 system library directory: /usr/lib\n\
             Install OpenVINO 2025.1 or newer (see docs/openvino.md), set \
             INTEL_OPENVINO_DIR to its install root, or set inference.openvino_path \
             in the config file to the directory holding openvino_c.dll"
        );
    }

    #[test]
    fn a_located_file_that_fails_to_load_never_falls_back_to_the_bare_name() {
        let failure = LoadFailure::CouldNotOpen(
            "the shared library at /opt/ov/lib/libopenvino_c.so could not be opened: \
             libtbb.so.12: cannot open shared object file"
                .to_string(),
        );
        for (os, name) in [
            (Os::Linux, "/opt/ov/lib/libopenvino_c.so"),
            (Os::Windows, "C:\\ov\\openvino_c.dll"),
        ] {
            let mut calls = Vec::new();
            let err = open_located(
                Located::File(PathBuf::from(name)),
                os,
                &mut scripted_open(&mut calls, Some(failure.clone())),
            )
            .unwrap_err();
            assert_eq!(
                calls,
                [(name.to_string(), Origin::Located)],
                "only the located file is tried on {os:?}"
            );
            assert!(
                matches!(err, Error::OpenVinoRuntimeLoadFailed { .. }),
                "{os:?}: {err}"
            );
        }
    }

    #[test]
    fn a_located_file_that_loads_is_the_library_in_use() {
        let mut calls = Vec::new();
        let opened = open_located(
            Located::File(PathBuf::from("/usr/lib/libopenvino_c.so")),
            Os::Linux,
            &mut scripted_open(&mut calls, None),
        );
        assert_eq!(
            opened.unwrap(),
            Some(PathBuf::from("/usr/lib/libopenvino_c.so"))
        );
        assert_eq!(
            calls,
            [("/usr/lib/libopenvino_c.so".to_string(), Origin::Located)]
        );
    }

    fn other_install_refusal(previous: &str, target: &str) -> LoadFailure {
        LoadFailure::Other(format!(
            "this process already loaded the OpenVINO runtime {previous}, and a process can load \
             only one; {target} was not loaded"
        ))
    }

    #[test]
    fn the_load_guard_preloads_only_a_first_located_load_and_records_a_preloaded_failure() {
        let t = Path::new("/opt/ov/libopenvino_c.so");
        let other = Path::new("/opt/other/libopenvino_c.so");
        // admit: preload only a located file while no runtime is held, and
        // refuse a different runtime once one is.
        assert_eq!(admit(None, t, Origin::LoaderSearch), Ok(false));
        assert_eq!(admit(None, t, Origin::Located), Ok(true));
        assert_eq!(admit(Some(t), t, Origin::Located), Ok(false));
        assert_eq!(
            admit(Some(other), t, Origin::LoaderSearch),
            Err(other_install_refusal(
                "/opt/other/libopenvino_c.so",
                "/opt/ov/libopenvino_c.so"
            ))
        );
        // record: a preloaded dependency stays resident after a refused open.
        let refused = LoadFailure::CouldNotOpen("x".to_string());
        let mut loaded = None;
        record(&mut loaded, t, true, Some(&refused));
        assert_eq!(loaded.as_deref(), Some(t));
        let mut loaded = None;
        record(&mut loaded, t, false, Some(&refused));
        assert_eq!(loaded, None);
        let mut loaded = None;
        record(&mut loaded, t, false, None);
        assert_eq!(loaded.as_deref(), Some(t));
        let mut loaded = None;
        record(
            &mut loaded,
            t,
            false,
            Some(&LoadFailure::Other("x".to_string())),
        );
        assert_eq!(loaded.as_deref(), Some(t));
    }

    /// Runs `guarded_load` over fakes. Returns the result, what the preload and
    /// the load were handed, and the recorded target.
    #[allow(clippy::type_complexity)]
    fn guarded(
        previous: Option<&str>,
        target: &str,
        origin: Origin,
        preload: std::result::Result<bool, &str>,
        load: Option<&LoadFailure>,
    ) -> (
        std::result::Result<(), LoadFailure>,
        Vec<PathBuf>,
        Vec<PathBuf>,
        Option<PathBuf>,
    ) {
        let mut loaded = previous.map(PathBuf::from);
        let (mut preloads, mut loads) = (Vec::new(), Vec::new());
        let result = guarded_load(
            &mut loaded,
            Path::new(target),
            origin,
            &mut |path| {
                preloads.push(path.to_path_buf());
                preload.map_err(str::to_string)
            },
            &mut |path| {
                loads.push(path.to_path_buf());
                load.cloned()
            },
        );
        (result, preloads, loads, loaded)
    }

    #[test]
    fn a_located_load_uses_one_absolute_path_for_the_preload_the_load_and_the_record() {
        let absolute = std::env::current_dir()
            .unwrap()
            .join("rel")
            .join("libopenvino_c.so");
        let refused = LoadFailure::CouldNotOpen("could not be opened: x".to_string());
        let (result, preloads, loads, loaded) = guarded(
            None,
            "rel/libopenvino_c.so",
            Origin::Located,
            Ok(true),
            Some(&refused),
        );
        assert_eq!(result, Err(refused));
        assert_eq!(preloads, std::slice::from_ref(&absolute));
        assert_eq!(loads, std::slice::from_ref(&absolute));
        assert_eq!(loaded, Some(absolute));

        // A bare name for the system loader stays as it is and preloads nothing.
        let (result, preloads, loads, loaded) = guarded(
            None,
            "libopenvino_c.so",
            Origin::LoaderSearch,
            Err("the preload must not run"),
            None,
        );
        assert_eq!(result, Ok(()));
        assert_eq!(preloads, Vec::<PathBuf>::new());
        assert_eq!(loads, [PathBuf::from("libopenvino_c.so")]);
        assert_eq!(loaded, Some(PathBuf::from("libopenvino_c.so")));
    }

    #[test]
    fn a_second_runtime_is_refused_before_anything_is_preloaded_or_loaded() {
        // Absolute on every OS, because a located target is made absolute.
        let cwd = std::env::current_dir().unwrap();
        let held = cwd.join("other").join("libopenvino_c.so");
        let wanted = cwd.join("ov").join("libopenvino_c.so");
        let (result, preloads, loads, loaded) = guarded(
            Some(held.to_str().unwrap()),
            wanted.to_str().unwrap(),
            Origin::Located,
            Ok(true),
            None,
        );
        assert_eq!(
            result,
            Err(other_install_refusal(
                &held.display().to_string(),
                &wanted.display().to_string()
            ))
        );
        assert_eq!(preloads, Vec::<PathBuf>::new());
        assert_eq!(loads, Vec::<PathBuf>::new());
        assert_eq!(loaded, Some(held));
    }

    #[test]
    fn a_skipped_preload_is_named_in_the_error_of_a_refused_load() {
        let cwd = std::env::current_dir().unwrap();
        let library = cwd.join("ov").join("libopenvino_c.so");
        let shown = library.display();
        let refused = LoadFailure::CouldNotOpen(format!(
            "the shared library at {shown} could not be opened: libopenvino.so.2630: \
             cannot open shared object file"
        ));
        let (result, _, loads, loaded) = guarded(
            None,
            library.to_str().unwrap(),
            Origin::Located,
            Err("libtbb.so.12 not found in ov"),
            Some(&refused),
        );
        assert_eq!(
            result,
            Err(LoadFailure::CouldNotOpen(format!(
                "the shared library at {shown} could not be opened: \
                 libopenvino.so.2630: cannot open shared object file (birda did not preload the \
                 runtime's dependencies: libtbb.so.12 not found in ov)"
            )))
        );
        // Nothing was preloaded and the loader refused the file: nothing is held.
        assert_eq!(loads, std::slice::from_ref(&library));
        assert_eq!(loaded, None);

        // A load that succeeds despite the skip says nothing about it.
        let (result, ..) = guarded(
            None,
            library.to_str().unwrap(),
            Origin::Located,
            Err("libtbb.so.12 not found in ov"),
            None,
        );
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn only_linux_hands_a_bare_library_name_to_the_loader() {
        assert!(Os::Linux.has_loader_fallback());
        assert!(!Os::Windows.has_loader_fallback());
    }

    #[test]
    fn a_not_found_error_without_a_loader_reason_lists_only_the_searched_places() {
        let searched = vec!["  loader path: C:\\openvino".to_string()];
        assert_eq!(
            runtime_not_found(&searched, Os::Windows, None).to_string(),
            "OpenVINO runtime not found. Searched:\n\
             \x20 loader path: C:\\openvino\n\
             Install OpenVINO 2025.1 or newer (see docs/openvino.md), set \
             INTEL_OPENVINO_DIR to its install root, or set inference.openvino_path \
             in the config file to the directory holding openvino_c.dll"
        );
    }

    #[test]
    fn load_failures_map_to_distinct_errors() {
        assert!(matches!(
            LoadFailure::TooOld("2024.6".to_string()).into_error(),
            Error::OpenVinoRuntimeTooOld {
                minimum: "2025.1",
                ..
            }
        ));
        assert!(matches!(
            LoadFailure::CouldNotOpen("x".to_string()).into_error(),
            Error::OpenVinoRuntimeLoadFailed { .. }
        ));
    }

    fn node(path: &str, vendor: Option<u32>, access: NodeAccess) -> RenderNode {
        RenderNode {
            path: path.to_string(),
            vendor,
            access,
        }
    }

    #[test]
    fn render_nodes_tell_the_failure_causes_apart() {
        use NodeAccess::{Denied, Failed, Open};
        let intel = Some(0x8086);
        let nvidia = Some(0x10de);
        assert_eq!(diagnose(&[]), Some(Diagnosis::NoRenderNodes));
        assert_eq!(
            diagnose(&[node("/dev/dri/renderD128", intel, Denied)]),
            Some(Diagnosis::AccessDenied {
                node: "/dev/dri/renderD128".to_string()
            })
        );
        assert_eq!(
            diagnose(&[node("/dev/dri/renderD128", intel, Open)]),
            Some(Diagnosis::ComputeRuntimeMissing)
        );
        assert_eq!(
            diagnose(&[node("/dev/dri/renderD128", nvidia, Open)]),
            Some(Diagnosis::NoIntelGpu)
        );
        // A closed non-Intel node does not mask an openable Intel one.
        assert_eq!(
            diagnose(&[
                node("/dev/dri/renderD128", nvidia, Denied),
                node("/dev/dri/renderD129", intel, Open),
            ]),
            Some(Diagnosis::ComputeRuntimeMissing)
        );
        // An unreadable vendor is neither assumed Intel nor assumed not Intel.
        assert_eq!(
            diagnose(&[
                node("/dev/dri/renderD128", nvidia, Open),
                node("/dev/dri/renderD129", None, Denied),
            ]),
            None
        );
        // An Intel node that failed to open for a reason other than
        // permissions earns no advice about the render group.
        assert_eq!(
            diagnose(&[node("/dev/dri/renderD128", intel, Failed)]),
            None
        );
    }

    #[test]
    fn render_diagnosis_runs_only_on_linux_when_no_gpu_is_listed() {
        let denied = || {
            vec![node(
                "/dev/dri/renderD128",
                Some(0x8086),
                NodeAccess::Denied,
            )]
        };
        let expected = Some(Diagnosis::AccessDenied {
            node: "/dev/dri/renderD128".to_string(),
        });

        assert_eq!(render_diagnosis(Some(Os::Linux), false, denied), expected);
        // Windows and unsupported platforms have no render nodes to read.
        assert_eq!(render_diagnosis(Some(Os::Windows), false, denied), None);
        assert_eq!(render_diagnosis(None, false, denied), None);
        // A listed Intel GPU means the nodes are not consulted at all.
        assert_eq!(
            render_diagnosis(Some(Os::Linux), true, || {
                panic!("the render nodes must not be gathered when a GPU is listed")
            }),
            None
        );
    }

    #[test]
    fn hints_name_the_node_and_the_fix() {
        let hint = Diagnosis::AccessDenied {
            node: "/dev/dri/renderD128".to_string(),
        }
        .hint();
        assert_eq!(
            hint,
            "\nThe Intel GPU render node /dev/dri/renderD128 cannot be opened by this user; \
             add the user to the 'render' group and log in again"
        );
        assert!(
            Diagnosis::ComputeRuntimeMissing
                .hint()
                .contains("intel-opencl-icd-legacy1")
        );
    }
}
