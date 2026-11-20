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
    OvDevice, classify, is_intel_architecture, parse_capabilities, parse_gpu_class,
    parse_memory_bytes, parse_pci_info,
};
use crate::constants::openvino as ov;
use crate::error::{Error, Result};
use openvino::{Core, DeviceType, PropertyKey};
use std::path::{Path, PathBuf};

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

/// Install roots found under the OS's usual parents, highest name first.
fn install_roots(os: Os, list_dir: ListDir<'_>) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for parent in os.install_parents() {
        let parent = Path::new(parent);
        let mut names: Vec<String> = list_dir(parent)
            .unwrap_or_default()
            .into_iter()
            .filter(|name| name.starts_with(ov::INSTALL_ROOT_PREFIX))
            .collect();
        names.sort_unstable_by(|a, b| b.cmp(a));
        roots.extend(names.into_iter().map(|name| parent.join(name)));
    }
    roots
}

/// Resolve `inference.openvino_path`: a library file, a directory holding it,
/// or an install root.
fn resolve_explicit(
    path: &Path,
    os: Os,
    list_dir: ListDir<'_>,
) -> std::result::Result<PathBuf, PathProblem> {
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

/// A DRM render node and what is known about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderNode {
    /// Device path, e.g. `/dev/dri/renderD128`.
    pub path: String,
    /// PCI vendor id from sysfs, when readable.
    pub vendor: Option<u32>,
    /// Whether this process can open the node read-write.
    pub openable: bool,
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
    /// Render nodes exist, none of them Intel's.
    NoIntelGpu,
}

/// Decide what the render nodes imply for a host where `OpenVINO` found no
/// usable Intel GPU. Pure over the node records.
#[must_use]
pub fn diagnose(nodes: &[RenderNode]) -> Diagnosis {
    let intel: Vec<&RenderNode> = nodes
        .iter()
        .filter(|node| node.vendor == Some(ov::INTEL_PCI_VENDOR_ID))
        .collect();
    if nodes.is_empty() {
        Diagnosis::NoRenderNodes
    } else if intel.is_empty() {
        Diagnosis::NoIntelGpu
    } else if let Some(node) = intel.iter().find(|node| !node.openable) {
        Diagnosis::AccessDenied {
            node: node.path.clone(),
        }
    } else {
        Diagnosis::ComputeRuntimeMissing
    }
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
            let openable = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .is_ok();
            RenderNode {
                path: path.display().to_string(),
                vendor,
                openable,
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
            return Err(Error::OpenVinoUnsupported {
                reason: "this platform is not supported; the OpenVINO backend runs on Linux \
                         and Windows, where OpenVINO has a GPU plugin"
                    .to_string(),
            });
        };
        let env = SearchEnv::from_process(os);
        let located =
            locate(explicit, &env, &list_dir_fs).map_err(|problem| Error::OpenVinoPathInvalid {
                path: explicit.map(Path::to_path_buf).unwrap_or_default(),
                reason: problem.reason,
            })?;

        let (library, load_target) = match located {
            Located::File(file) => (Some(file.clone()), file),
            // The OS loader has its own search path (ld.so.conf.d, RUNPATH)
            // that the directory search cannot see.
            Located::NotFound { searched } => {
                return load_core(Path::new(os.library_file()), None).map_err(|e| match e {
                    Error::OpenVinoRuntimeLoadFailed { .. } => Error::OpenVinoRuntimeNotFound {
                        searched: format!(
                            "{}\n  the system loader path: {}",
                            searched.join("\n"),
                            os.library_file()
                        ),
                        library: os.library_file(),
                    },
                    other => other,
                });
            }
        };
        load_core(&load_target, library)
    }

    /// The devices birda may run on, in listing order.
    #[must_use]
    pub fn selectable(&self) -> Vec<&OvDevice> {
        self.devices
            .iter()
            .filter(|device| device.is_selectable())
            .collect()
    }
}

fn load_core(target: &Path, library: Option<PathBuf>) -> Result<Runtime> {
    openvino_sys::library::load_from(target)
        .map_err(|message| classify_load_error(&message).into_error())?;
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
        .map_err(|e| Error::OpenVinoModelLoad {
            reason: format!("listing devices failed: {e}"),
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
            device.uuid = read(core, &id, "DEVICE_UUID").map(|value| value.to_ascii_lowercase());
            device.pci =
                read(core, &id, "DEVICE_PCI_INFO").and_then(|value| parse_pci_info(&value));
            device.memory_bytes = read(core, &id, "GPU_DEVICE_TOTAL_MEM_SIZE")
                .and_then(|value| parse_memory_bytes(&value));
            device.capabilities = core
                .get_property(&id, &PropertyKey::DeviceCapabilities)
                .map(|value| parse_capabilities(&value))
                .unwrap_or_default();
            device.kind = classify(name);
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

    fn node(path: &str, vendor: Option<u32>, openable: bool) -> RenderNode {
        RenderNode {
            path: path.to_string(),
            vendor,
            openable,
        }
    }

    #[test]
    fn render_nodes_tell_the_failure_causes_apart() {
        assert_eq!(diagnose(&[]), Diagnosis::NoRenderNodes);
        assert_eq!(
            diagnose(&[node("/dev/dri/renderD128", Some(0x8086), false)]),
            Diagnosis::AccessDenied {
                node: "/dev/dri/renderD128".to_string()
            }
        );
        assert_eq!(
            diagnose(&[node("/dev/dri/renderD128", Some(0x8086), true)]),
            Diagnosis::ComputeRuntimeMissing
        );
        assert_eq!(
            diagnose(&[node("/dev/dri/renderD128", Some(0x10de), true)]),
            Diagnosis::NoIntelGpu
        );
        // A closed non-Intel node does not mask an openable Intel one.
        assert_eq!(
            diagnose(&[
                node("/dev/dri/renderD128", Some(0x10de), false),
                node("/dev/dri/renderD129", Some(0x8086), true),
            ]),
            Diagnosis::ComputeRuntimeMissing
        );
        // An unreadable vendor is not assumed to be Intel.
        assert_eq!(
            diagnose(&[node("/dev/dri/renderD128", None, false)]),
            Diagnosis::NoIntelGpu
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
