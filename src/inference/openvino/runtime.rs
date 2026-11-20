//! Finding, loading and enumerating the user's `OpenVINO` runtime.
//!
//! birda ships no `OpenVINO` files. The `openvino` crate is built with its
//! `runtime-linking` feature, so the C API library (`libopenvino_c`) is opened
//! at run time. This file decides where to look, loads the library from the
//! first place that has it, and lists the devices the runtime reports.
//!
//! The path logic, the load-error classification and the render-node diagnosis
//! are pure functions, tested without a runtime.

use super::device::{
    OvDevice, is_intel_architecture, parse_capabilities, parse_gpu_class, parse_memory_bytes,
    parse_pci_info, parse_uuid,
};
use crate::constants::openvino as ov;
use crate::error::{Error, Result};
use openvino::{Core, DeviceType, PropertyKey};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

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
        match self {
            Self::Linux => ov::LINUX_LIBRARY_FILE,
            Self::Windows => ov::WINDOWS_LIBRARY_FILE,
        }
    }

    /// Whether a bare library name may be handed to the system loader: only
    /// on Linux, where `dlopen` does not search the working directory.
    const fn has_loader_fallback(self) -> bool {
        matches!(self, Self::Linux)
    }

    const fn install_lib_dir(self) -> &'static str {
        match self {
            Self::Linux => ov::LINUX_INSTALL_LIB_DIR,
            Self::Windows => ov::WINDOWS_INSTALL_LIB_DIR,
        }
    }

    const fn install_parents(self) -> &'static [&'static str] {
        match self {
            Self::Linux => ov::LINUX_INSTALL_PARENTS,
            Self::Windows => ov::WINDOWS_INSTALL_PARENTS,
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
    /// Value of `INTEL_OPENVINO_DIR`.
    pub install_dir: Option<PathBuf>,
    /// Entries of the loader path variable (`LD_LIBRARY_PATH`, `PATH`).
    pub loader_path: Vec<PathBuf>,
}

impl SearchEnv {
    /// Read the search inputs from this process.
    #[must_use]
    pub fn from_process(os: Os) -> Self {
        let loader_var = match os {
            Os::Linux => "LD_LIBRARY_PATH",
            Os::Windows => "PATH",
        };
        Self {
            os,
            install_dir: std::env::var_os(ov::INSTALL_DIR_ENV)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from),
            loader_path: std::env::var_os(loader_var)
                .map(|value| std::env::split_paths(&value).collect())
                .unwrap_or_default(),
        }
    }
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
/// the highest versioned one.
fn pick_library(names: &[String], os: Os) -> Option<String> {
    let exact = os.library_file();
    if names.iter().any(|name| name == exact) {
        return Some(exact.to_string());
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
    let rest = name.strip_prefix(ov::INSTALL_ROOT_PREFIX)?;
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

/// Install roots found under the OS's usual parents, newest version first.
fn install_roots(os: Os, list_dir: ListDir<'_>) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for parent in os.install_parents() {
        let parent = Path::new(parent);
        let names: Vec<String> = list_dir(parent)
            .unwrap_or_default()
            .into_iter()
            .filter(|name| name.starts_with(ov::INSTALL_ROOT_PREFIX))
            .collect();
        roots.extend(
            order_install_roots(names)
                .into_iter()
                .map(|name| parent.join(name)),
        );
    }
    roots
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
        .is_some_and(|(names, file)| names.iter().any(|name| *name == *file));
    if listed {
        Ok(path.to_path_buf())
    } else {
        Err(PathProblem {
            reason: "it does not exist".to_string(),
        })
    }
}

/// Find the `OpenVINO` C API library.
///
/// An explicit path is the only place searched when it is given: a wrong
/// configured path must not silently fall through to some other runtime.
/// Otherwise, in order: `INTEL_OPENVINO_DIR`, the loader path variable, the
/// system library directories, and the install roots under `/opt/intel` (or
/// `Program Files\Intel`), highest name first.
///
/// # Errors
///
/// [`PathProblem`] when `explicit` does not lead to the library.
pub fn locate(
    explicit: Option<&Path>,
    env: &SearchEnv,
    list_dir: ListDir<'_>,
) -> std::result::Result<Located, PathProblem> {
    let os = env.os;
    if let Some(path) = explicit {
        return resolve_explicit(path, os, list_dir).map(Located::File);
    }

    let mut searched = Vec::new();
    let mut check = |label: String, dir: PathBuf| -> Option<PathBuf> {
        let found = find_in(&dir, os, list_dir);
        searched.push(format!("  {label}: {}", dir.display()));
        found
    };

    if let Some(root) = &env.install_dir {
        let dir = root.join(os.install_lib_dir());
        if let Some(file) = check(ov::INSTALL_DIR_ENV.to_string(), dir) {
            return Ok(Located::File(file));
        }
    }
    for dir in &env.loader_path {
        if let Some(file) = check("loader path".to_string(), dir.clone()) {
            return Ok(Located::File(file));
        }
    }
    for dir in os.system_lib_dirs() {
        if let Some(file) = check("system library directory".to_string(), PathBuf::from(dir)) {
            return Ok(Located::File(file));
        }
    }
    for root in install_roots(os, list_dir) {
        let dir = root.join(os.install_lib_dir());
        if let Some(file) = check("install directory".to_string(), dir) {
            return Ok(Located::File(file));
        }
    }
    Ok(Located::NotFound { searched })
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
        let located =
            locate(explicit, &env, &list_dir_fs).map_err(|problem| Error::OpenVinoPathInvalid {
                path: explicit.map(Path::to_path_buf).unwrap_or_default(),
                reason: problem.reason,
            })?;

        let library = open_located(located, os, &mut open_library)?;
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

/// Open the runtime library for a search result, through `open`.
///
/// Returns the file that was opened, or `None` when the system loader found
/// the bare library name. `open` is a parameter so the decisions here can be
/// tested without a runtime:
///
/// * a located file is opened as it is, and a failure is final on every OS: a
///   path the search or the user chose never falls back to another runtime;
/// * with nothing located, only Linux asks the system loader for the bare name
///   (see [`Os::has_loader_fallback`]); Windows reports the places searched.
fn open_located(
    located: Located,
    os: Os,
    open: &mut dyn FnMut(&Path) -> std::result::Result<(), LoadFailure>,
) -> Result<Option<PathBuf>> {
    match located {
        Located::File(file) => {
            open(&file).map_err(LoadFailure::into_error)?;
            Ok(Some(file))
        }
        Located::NotFound { searched } => {
            // The Linux loader has its own search path (ld.so.conf.d, RUNPATH)
            // that the directory search cannot see. On Windows a bare name
            // would also be searched for in the working directory.
            if !os.has_loader_fallback() {
                return Err(runtime_not_found(&searched, os, None));
            }
            open(Path::new(os.library_file()))
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

/// Open the `openvino_c` library and check its version.
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
fn open_library(target: &Path) -> std::result::Result<(), LoadFailure> {
    let mut loaded = LOADED_LIBRARY
        .lock()
        .map_err(|_| LoadFailure::Other("an earlier OpenVINO runtime load failed".to_string()))?;
    if let Some(previous) = loaded.as_deref()
        && previous != target
    {
        return Err(LoadFailure::Other(format!(
            "this process already loaded the OpenVINO runtime {}, and a process can load only \
             one; {} was not loaded",
            previous.display(),
            target.display()
        )));
    }
    let outcome = std::panic::catch_unwind(|| openvino_sys::library::load_from(target));
    let failure = match outcome {
        Ok(Ok(())) => None,
        Ok(Err(message)) => Some(classify_load_error(&message)),
        Err(_) => Some(LoadFailure::Other(format!(
            "{} is not the OpenVINO C API library: it lacks the functions birda calls",
            target.display()
        ))),
    };
    // Only a library the loader refused is not held by the binding.
    if !matches!(failure, Some(LoadFailure::CouldNotOpen(_))) {
        *loaded = Some(target.to_path_buf());
    }
    drop(loaded);
    failure.map_or(Ok(()), Err)
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

    fn linux_env() -> SearchEnv {
        SearchEnv {
            os: Os::Linux,
            install_dir: None,
            loader_path: Vec::new(),
        }
    }

    fn locate_in(
        explicit: Option<&str>,
        env: &SearchEnv,
        fs: &HashMap<PathBuf, Vec<String>>,
    ) -> std::result::Result<Located, PathProblem> {
        locate(explicit.map(Path::new), env, &|dir| fs.get(dir).cloned())
    }

    fn file(path: &str) -> Located {
        Located::File(PathBuf::from(path))
    }

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
            open_library(Path::new(not_openvino)),
            Err(LoadFailure::Other(format!(
                "{not_openvino} is not the OpenVINO C API library: it lacks the functions \
                 birda calls"
            )))
        );
        assert_eq!(
            open_library(Path::new("/opt/other/libopenvino_c.so")),
            Err(LoadFailure::Other(format!(
                "this process already loaded the OpenVINO runtime {not_openvino}, and a process \
                 can load only one; /opt/other/libopenvino_c.so was not loaded"
            )))
        );
    }

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
    fn the_default_search_follows_install_dir_loader_path_system_dirs_then_roots() {
        let fs = fake_fs(&[
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
        env.install_dir = Some(PathBuf::from("/home/u/ov"));
        env.loader_path = vec![PathBuf::from("/empty"), PathBuf::from("/custom/lib")];
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

    #[test]
    fn the_windows_layout_uses_its_own_library_name_and_directories() {
        let parent = PathBuf::from(ov::WINDOWS_INSTALL_PARENTS[0]);
        let newest = parent.join("openvino_2026.2");
        let lib_dir = newest.join(ov::WINDOWS_INSTALL_LIB_DIR);
        let fs: HashMap<PathBuf, Vec<String>> = HashMap::from([
            (
                parent,
                vec!["openvino_2026.2".to_string(), "openvino_2025.4".to_string()],
            ),
            (
                lib_dir.clone(),
                vec!["openvino_c.dll".to_string(), "openvino.dll".to_string()],
            ),
        ]);
        let env = SearchEnv {
            os: Os::Windows,
            install_dir: None,
            loader_path: Vec::new(),
        };
        assert_eq!(
            locate(None, &env, &|dir| fs.get(dir).cloned()),
            Ok(Located::File(lib_dir.join("openvino_c.dll")))
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
        env.install_dir = Some(PathBuf::from("/home/u/ov"));
        env.loader_path = vec![PathBuf::from("/custom/lib")];
        let Ok(Located::NotFound { searched }) = locate_in(None, &env, &fs) else {
            panic!("nothing is installed");
        };
        let under_root = Path::new("/home/u/ov").join(ov::LINUX_INSTALL_LIB_DIR);
        assert_eq!(
            searched[0],
            format!("  INTEL_OPENVINO_DIR: {}", under_root.display())
        );
        assert_eq!(searched[1], "  loader path: /custom/lib");
        assert_eq!(searched[2], "  system library directory: /usr/lib");
        assert_eq!(searched.len(), 2 + ov::LINUX_SYSTEM_LIB_DIRS.len());
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
             Install OpenVINO 2025.1 or newer (see docs/openvino.md) or set \
             inference.openvino_path in the config file to the directory holding libopenvino_c.so"
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
             The runtime and its TBB library directories must be on the loader path \
             (LD_LIBRARY_PATH, or PATH on Windows), as OpenVINO's setupvars script does \
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

    /// Records every path handed to the loader and answers from a script.
    fn scripted_open(
        calls: &mut Vec<String>,
        fail_with: Option<LoadFailure>,
    ) -> impl FnMut(&Path) -> std::result::Result<(), LoadFailure> + '_ {
        move |path| {
            calls.push(path.display().to_string());
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
        assert_eq!(calls, ["libopenvino_c.so"]);

        let mut calls = Vec::new();
        let err = open_located(
            nothing_found(),
            Os::Windows,
            &mut scripted_open(&mut calls, None),
        )
        .unwrap_err();
        assert_eq!(
            calls,
            Vec::<String>::new(),
            "Windows must not load a bare name"
        );
        assert_eq!(
            err.to_string(),
            "OpenVINO runtime not found. Searched:\n\
             \x20 system library directory: /usr/lib\n\
             Install OpenVINO 2025.1 or newer (see docs/openvino.md) or set \
             inference.openvino_path in the config file to the directory holding openvino_c.dll"
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
            assert_eq!(calls, [name], "only the located file is tried on {os:?}");
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
        assert_eq!(calls, ["/usr/lib/libopenvino_c.so"]);
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
             Install OpenVINO 2025.1 or newer (see docs/openvino.md) or set \
             inference.openvino_path in the config file to the directory holding openvino_c.dll"
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
