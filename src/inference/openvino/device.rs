//! `OpenVINO` device records and device selection.
//!
//! Nothing in this file calls `OpenVINO`. Records are built by
//! `runtime::enumerate` from property strings, and [`resolve`] is a pure
//! function over them, so a selection rule can be tested with a synthetic
//! device list.

use crate::config::{OpenVinoDeviceSpec, PciAddress};
use crate::constants::openvino::INTEL_VENDOR_MARKER;

/// What an `OpenVINO` device name denotes.
///
/// Classified by name first: the CPU device reports its `DEVICE_TYPE` property
/// as `integrated` too (read on two Intel hosts, `OpenVINO` 2025.4.1 and
/// 2026.2), so the type property cannot tell it from a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// The `CPU` device. Listed, never selected.
    Cpu,
    /// `GPU`, or `GPU.N`. A device listed as bare `GPU` is index 0.
    Gpu(u32),
    /// A tile of a multi-tile GPU (`GPU.N.M`). Listed, not selectable.
    GpuTile,
    /// The `NPU` device. Listed, not supported.
    Npu,
    /// Anything else (`AUTO`, `HETERO`, a plugin birda does not know).
    Other,
}

impl DeviceKind {
    /// Lowercase name used in listings and the JSON envelope.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu(_) => "gpu",
            Self::GpuTile => "gpu_tile",
            Self::Npu => "npu",
            Self::Other => "other",
        }
    }
}

/// Class of a GPU, from its `DEVICE_TYPE` property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuClass {
    /// `integrated`.
    Integrated,
    /// `discrete`.
    Discrete,
    /// Missing or an unrecognized value.
    Unknown,
}

impl GpuClass {
    /// Lowercase name used in listings and the JSON envelope.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Integrated => "integrated",
            Self::Discrete => "discrete",
            Self::Unknown => "unknown",
        }
    }
}

/// One device `OpenVINO` reported, with the properties birda uses.
///
/// Every property is optional because a failed property read is not an error:
/// it only makes the listing and the pins less specific.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OvDevice {
    /// `OpenVINO`'s name for the device: `CPU`, `GPU`, `GPU.1`.
    pub name: String,
    /// What the name denotes.
    pub kind: DeviceKind,
    /// GPU class; `Unknown` for anything that is not a GPU.
    pub class: GpuClass,
    /// Whether `DEVICE_ARCHITECTURE` names Intel as the vendor.
    pub intel: bool,
    /// `FULL_DEVICE_NAME`.
    pub full_name: Option<String>,
    /// `DEVICE_ARCHITECTURE`, verbatim.
    pub architecture: Option<String>,
    /// `DEVICE_UUID`, lowercase.
    pub uuid: Option<String>,
    /// `DEVICE_PCI_INFO`.
    pub pci: Option<PciAddress>,
    /// `GPU_DEVICE_TOTAL_MEM_SIZE`.
    pub memory_bytes: Option<u64>,
    /// `OPTIMIZATION_CAPABILITIES`, split into words.
    pub capabilities: Vec<String>,
}

impl OvDevice {
    /// A device with only a name; every property unknown.
    #[must_use]
    pub fn named(name: &str) -> Self {
        Self {
            name: name.to_string(),
            kind: classify(name),
            class: GpuClass::Unknown,
            intel: false,
            full_name: None,
            architecture: None,
            uuid: None,
            pci: None,
            memory_bytes: None,
            capabilities: Vec::new(),
        }
    }

    /// Whether birda may run on this device: an Intel GPU, not a tile.
    #[must_use]
    pub fn is_selectable(&self) -> bool {
        matches!(self.kind, DeviceKind::Gpu(_)) && self.intel
    }

    /// One line for messages: name, marketing name, class, PCI address.
    #[must_use]
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(full_name) = &self.full_name {
            parts.push(full_name.clone());
        }
        if matches!(self.kind, DeviceKind::Gpu(_)) {
            parts.push(self.class.as_str().to_string());
        }
        if let Some(pci) = self.pci {
            parts.push(format!("pci {pci}"));
        }
        if parts.is_empty() {
            self.name.clone()
        } else {
            format!("{} ({})", self.name, parts.join(", "))
        }
    }
}

/// Classify a device by name.
#[must_use]
pub fn classify(name: &str) -> DeviceKind {
    if name == "CPU" {
        return DeviceKind::Cpu;
    }
    if name == "NPU" {
        return DeviceKind::Npu;
    }
    if name == "GPU" {
        return DeviceKind::Gpu(0);
    }
    if let Some(rest) = name.strip_prefix("GPU.") {
        let numeric = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
        return match rest.split_once('.') {
            None if numeric(rest) => rest.parse().map_or(DeviceKind::Other, DeviceKind::Gpu),
            Some((index, tile)) if numeric(index) && numeric(tile) => DeviceKind::GpuTile,
            _ => DeviceKind::Other,
        };
    }
    DeviceKind::Other
}

/// Parse `DEVICE_TYPE`.
#[must_use]
pub fn parse_gpu_class(value: &str) -> GpuClass {
    match value.trim().to_ascii_lowercase().as_str() {
        "integrated" => GpuClass::Integrated,
        "discrete" => GpuClass::Discrete,
        _ => GpuClass::Unknown,
    }
}

/// Whether a `DEVICE_ARCHITECTURE` value names Intel as the vendor.
#[must_use]
pub fn is_intel_architecture(value: &str) -> bool {
    value
        .split_whitespace()
        .any(|word| word.eq_ignore_ascii_case(INTEL_VENDOR_MARKER))
}

/// Parse `DEVICE_PCI_INFO`, which `OpenVINO` prints as
/// `{domain: D bus: B device: 0xN function: F}`.
///
/// Domain, bus and function are decimal and the device number is hexadecimal
/// with a `0x` prefix: `operator<<` for `PCIInfo` at
/// `src/inference/include/openvino/runtime/properties.hpp:1262-1265` in the
/// `OpenVINO` 2026.2.0 sources.
///
/// The all-zero address is treated as not reported: on x86 PCI it belongs to
/// the host bridge, not to a GPU, so offering it as a pin could only match the
/// wrong device.
#[must_use]
pub fn parse_pci_info(value: &str) -> Option<PciAddress> {
    let inner = value.trim().strip_prefix('{')?.strip_suffix('}')?;
    let mut words = inner.split_whitespace();
    let mut field = |label: &str| {
        (words.next()? == label).then_some(())?;
        words.next()
    };
    let domain = field("domain:")?.parse::<u16>().ok()?;
    let bus = field("bus:")?.parse::<u8>().ok()?;
    let device = u8::from_str_radix(field("device:")?.strip_prefix("0x")?, 16).ok()?;
    let function = field("function:")?.parse::<u8>().ok()?;
    words.next().is_none().then_some(())?;
    if (domain, bus, device, function) == (0, 0, 0, 0) {
        return None;
    }
    // Reject what a PCI address cannot hold rather than pin a wrong device.
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}")
        .parse()
        .ok()
}

/// Parse `DEVICE_UUID` into the lowercase form `uuid:` pins compare against.
///
/// An all-zero value is treated as not reported: it identifies no particular
/// GPU, so a pin made from it would match any other GPU that reports zeros.
#[must_use]
pub fn parse_uuid(value: &str) -> Option<String> {
    let uuid = value.trim().to_ascii_lowercase();
    (!uuid.is_empty() && !uuid.chars().all(|c| c == '0')).then_some(uuid)
}

/// Parse `GPU_DEVICE_TOTAL_MEM_SIZE`.
#[must_use]
pub fn parse_memory_bytes(value: &str) -> Option<u64> {
    value.trim().parse().ok()
}

/// Split `OPTIMIZATION_CAPABILITIES` into words.
#[must_use]
pub fn parse_capabilities(value: &str) -> Vec<String> {
    value
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect()
}

/// Why no device was resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// Nothing matches; `found` describes every device `OpenVINO` listed.
    NotFound {
        /// The spec as the user wrote it.
        requested: String,
        /// One description per listed device.
        found: Vec<String>,
    },
    /// The spec matches a device birda refuses to use.
    Unsupported {
        /// The spec as the user wrote it.
        requested: String,
        /// Why that device is refused.
        reason: String,
    },
}

/// The Intel GPUs in enumeration order, with their position in the list.
fn intel_gpus(devices: &[OvDevice]) -> impl Iterator<Item = &OvDevice> {
    devices.iter().filter(|device| device.is_selectable())
}

fn describe_all(devices: &[OvDevice]) -> Vec<String> {
    devices.iter().map(OvDevice::describe).collect()
}

/// The best Intel GPU: discrete, then integrated, then unknown class.
fn best_gpu(devices: &[OvDevice]) -> Option<&OvDevice> {
    [GpuClass::Discrete, GpuClass::Integrated, GpuClass::Unknown]
        .into_iter()
        .find_map(|class| intel_gpus(devices).find(|device| device.class == class))
}

/// Resolve a spec against the devices `OpenVINO` listed.
///
/// A pure function of its arguments: pins (`pci:`, `uuid:`) pick the same
/// device whatever else is installed, while index selectors are relative to
/// the list by nature.
///
/// # Errors
///
/// [`ResolveError::NotFound`] when no device matches, with every listed device
/// described, and [`ResolveError::Unsupported`] when the spec names the NPU or
/// a device that is not an Intel GPU.
pub fn resolve<'a>(
    spec: &OpenVinoDeviceSpec,
    devices: &'a [OvDevice],
) -> Result<&'a OvDevice, ResolveError> {
    let requested = spec.to_string();
    let not_found = || ResolveError::NotFound {
        requested: requested.clone(),
        found: describe_all(devices),
    };
    let non_intel = |device: &OvDevice| ResolveError::Unsupported {
        requested: requested.clone(),
        reason: format!(
            "{} is not an Intel GPU, and the OpenVINO backend only supports Intel GPUs",
            device.describe()
        ),
    };
    // The device a pin names, whatever its vendor, so that a non-Intel match
    // is reported as unsupported and not as missing.
    let pinned = |matches: &dyn Fn(&OvDevice) -> bool| -> Result<&'a OvDevice, ResolveError> {
        let device = devices
            .iter()
            .filter(|device| matches!(device.kind, DeviceKind::Gpu(_)))
            .find(|device| matches(device))
            .ok_or_else(not_found)?;
        if device.intel {
            Ok(device)
        } else {
            Err(non_intel(device))
        }
    };
    let nth_of_class = |class: GpuClass, index: Option<u32>| {
        intel_gpus(devices)
            .filter(|device| device.class == class)
            .nth(index.unwrap_or(0) as usize)
            .ok_or_else(not_found)
    };

    match spec {
        OpenVinoDeviceSpec::Auto | OpenVinoDeviceSpec::Gpu(None) => {
            best_gpu(devices).ok_or_else(not_found)
        }
        OpenVinoDeviceSpec::Gpu(Some(index)) => {
            pinned(&|device| device.kind == DeviceKind::Gpu(*index))
        }
        OpenVinoDeviceSpec::Igpu(index) => nth_of_class(GpuClass::Integrated, *index),
        OpenVinoDeviceSpec::Dgpu(index) => nth_of_class(GpuClass::Discrete, *index),
        OpenVinoDeviceSpec::Pci(address) => pinned(&|device| device.pci == Some(*address)),
        OpenVinoDeviceSpec::Uuid(uuid) => {
            pinned(&|device| device.uuid.as_deref() == Some(uuid.as_str()))
        }
        OpenVinoDeviceSpec::Npu => Err(ResolveError::Unsupported {
            requested,
            reason: "the NPU is not supported yet; the OpenVINO backend runs on Intel GPUs only"
                .to_string(),
        }),
    }
}

/// Every spec that resolves to `device` within `devices`, most general first.
///
/// Built by trying candidates through [`resolve`], so the listing cannot
/// drift from what the resolver does.
#[must_use]
pub fn selectors_for(device: &OvDevice, devices: &[OvDevice]) -> Vec<String> {
    if !device.is_selectable() {
        return Vec::new();
    }
    let mut candidates = vec![OpenVinoDeviceSpec::Gpu(None)];
    let class_position = intel_gpus(devices)
        .filter(|other| other.class == device.class)
        .position(|other| other.name == device.name);
    if let Some(position) = class_position {
        let index = u32::try_from(position).ok();
        let make: Option<fn(Option<u32>) -> OpenVinoDeviceSpec> = match device.class {
            GpuClass::Integrated => Some(OpenVinoDeviceSpec::Igpu),
            GpuClass::Discrete => Some(OpenVinoDeviceSpec::Dgpu),
            GpuClass::Unknown => None,
        };
        if let Some(make) = make {
            candidates.push(make(if position == 0 { None } else { index }));
        }
    }
    if let DeviceKind::Gpu(index) = device.kind {
        candidates.push(OpenVinoDeviceSpec::Gpu(Some(index)));
    }
    if let Some(pci) = device.pci {
        candidates.push(OpenVinoDeviceSpec::Pci(pci));
    }
    if let Some(uuid) = &device.uuid {
        candidates.push(OpenVinoDeviceSpec::Uuid(uuid.clone()));
    }
    candidates
        .into_iter()
        .filter(|spec| resolve(spec, devices).is_ok_and(|found| found.name == device.name))
        .map(|spec| spec.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 32-digit UUID made of one repeated byte.
    fn uuid(byte: &str) -> String {
        byte.repeat(16)
    }

    fn gpu(name: &str, class: GpuClass, pci_device: u8, uuid_byte: &str) -> OvDevice {
        OvDevice {
            full_name: Some(format!("Intel GPU at {pci_device:#x}")),
            class,
            intel: true,
            pci: Some(PciAddress {
                domain: 0,
                bus: 0,
                device: pci_device,
                function: 0,
            }),
            uuid: Some(uuid(uuid_byte)),
            ..OvDevice::named(name)
        }
    }

    fn cpu() -> OvDevice {
        // The CPU device reports DEVICE_TYPE integrated as well; class is set
        // to prove the name decides before the type does.
        OvDevice {
            class: GpuClass::Integrated,
            intel: true,
            ..OvDevice::named("CPU")
        }
    }

    fn non_intel(name: &str) -> OvDevice {
        OvDevice {
            full_name: Some("NVIDIA GeForce".to_string()),
            class: GpuClass::Discrete,
            ..OvDevice::named(name)
        }
    }

    /// iGPU plus two dGPUs, the layout `OpenVINO` documents: integrated first.
    fn igpu_and_two_dgpus() -> Vec<OvDevice> {
        vec![
            cpu(),
            gpu("GPU.0", GpuClass::Integrated, 0x02, "aa"),
            gpu("GPU.1", GpuClass::Discrete, 0x03, "bb"),
            gpu("GPU.2", GpuClass::Discrete, 0x04, "cc"),
        ]
    }

    fn resolved_name(spec: &str, devices: &[OvDevice]) -> Result<String, ResolveError> {
        resolve(&spec.parse().unwrap(), devices).map(|device| device.name.clone())
    }

    #[test]
    fn resolve_picks_by_class_index_pin_and_vendor() {
        let layout = igpu_and_two_dgpus();
        // auto and gpu prefer the first discrete GPU; igpu is the integrated one.
        assert_eq!(resolved_name("auto", &layout), Ok("GPU.1".to_string()));
        assert_eq!(resolved_name("gpu", &layout), Ok("GPU.1".to_string()));
        assert_eq!(resolved_name("igpu", &layout), Ok("GPU.0".to_string()));
        assert_eq!(resolved_name("dgpu", &layout), Ok("GPU.1".to_string()));
        assert_eq!(resolved_name("dgpu.1", &layout), Ok("GPU.2".to_string()));
        assert_eq!(resolved_name("gpu.1", &layout), Ok("GPU.1".to_string()));
        assert_eq!(
            resolved_name("pci:0000:00:04.0", &layout),
            Ok("GPU.2".to_string())
        );
        assert_eq!(
            resolved_name(&format!("uuid:{}", uuid("cc")), &layout),
            Ok("GPU.2".to_string())
        );

        // Two identical dGPUs without an iGPU: the pins tell them apart.
        let twins = vec![
            cpu(),
            gpu("GPU.0", GpuClass::Discrete, 0x03, "bb"),
            gpu("GPU.1", GpuClass::Discrete, 0x04, "cc"),
        ];
        assert_eq!(resolved_name("dgpu", &twins), Ok("GPU.0".to_string()));
        assert_eq!(
            resolved_name("pci:0000:00:04.0", &twins),
            Ok("GPU.1".to_string())
        );

        // A single GPU is listed as bare GPU and is index 0 of every selector.
        let single = vec![cpu(), gpu("GPU", GpuClass::Integrated, 0x02, "aa")];
        for spec in ["gpu", "igpu", "gpu.0", "igpu.0", "auto"] {
            assert_eq!(
                resolved_name(spec, &single),
                Ok("GPU".to_string()),
                "{spec}"
            );
        }

        // A GPU of unknown class is the last resort for auto and gpu.
        let unknown = vec![gpu("GPU", GpuClass::Unknown, 0x02, "aa")];
        assert_eq!(resolved_name("auto", &unknown), Ok("GPU".to_string()));
        assert_eq!(
            resolved_name("igpu", &unknown),
            Err(ResolveError::NotFound {
                requested: "igpu".to_string(),
                found: vec!["GPU (Intel GPU at 0x2, unknown, pci 0000:00:02.0)".to_string()],
            })
        );
    }

    #[test]
    fn resolve_errors_list_what_was_found() {
        let layout = igpu_and_two_dgpus();
        let found = describe_all(&layout);
        assert_eq!(
            found,
            vec![
                "CPU".to_string(),
                "GPU.0 (Intel GPU at 0x2, integrated, pci 0000:00:02.0)".to_string(),
                "GPU.1 (Intel GPU at 0x3, discrete, pci 0000:00:03.0)".to_string(),
                "GPU.2 (Intel GPU at 0x4, discrete, pci 0000:00:04.0)".to_string(),
            ]
        );
        assert_eq!(
            resolved_name("dgpu.2", &layout),
            Err(ResolveError::NotFound {
                requested: "dgpu.2".to_string(),
                found: found.clone(),
            })
        );
        assert_eq!(
            resolved_name("gpu.3", &layout),
            Err(ResolveError::NotFound {
                requested: "gpu.3".to_string(),
                found,
            })
        );
        assert_eq!(
            resolved_name("npu", &layout),
            Err(ResolveError::Unsupported {
                requested: "npu".to_string(),
                reason:
                    "the NPU is not supported yet; the OpenVINO backend runs on Intel GPUs only"
                        .to_string(),
            })
        );
    }

    #[test]
    fn a_non_intel_gpu_is_skipped_by_auto_and_refused_by_name() {
        let devices = vec![
            cpu(),
            gpu("GPU.0", GpuClass::Integrated, 0x02, "aa"),
            non_intel("GPU.1"),
        ];
        assert_eq!(resolved_name("auto", &devices), Ok("GPU.0".to_string()));
        assert_eq!(resolved_name("gpu", &devices), Ok("GPU.0".to_string()));
        // The discrete NVIDIA card is not an Intel dGPU, so dgpu finds nothing.
        assert!(matches!(
            resolved_name("dgpu", &devices),
            Err(ResolveError::NotFound { .. })
        ));
        // Asked for by name it is refused as unsupported, not reported missing.
        assert_eq!(
            resolved_name("gpu.1", &devices),
            Err(ResolveError::Unsupported {
                requested: "gpu.1".to_string(),
                reason: "GPU.1 (NVIDIA GeForce, discrete) is not an Intel GPU, and the \
                         OpenVINO backend only supports Intel GPUs"
                    .to_string(),
            })
        );
    }

    #[test]
    fn auto_never_resolves_to_the_cpu_device() {
        // The CPU reports itself as integrated; only its name keeps it out.
        let only_cpu = vec![cpu()];
        assert_eq!(
            resolved_name("auto", &only_cpu),
            Err(ResolveError::NotFound {
                requested: "auto".to_string(),
                found: vec!["CPU".to_string()],
            })
        );
        assert_eq!(
            resolved_name("igpu", &only_cpu),
            Err(ResolveError::NotFound {
                requested: "igpu".to_string(),
                found: vec!["CPU".to_string()],
            })
        );
        assert_eq!(
            resolved_name("gpu", &[]),
            Err(ResolveError::NotFound {
                requested: "gpu".to_string(),
                found: Vec::new(),
            })
        );
    }

    #[test]
    fn pins_survive_changes_to_the_device_list() {
        // Start with iGPU + dGPU A + dGPU B, then remove and reorder devices.
        let igpu = gpu("GPU.0", GpuClass::Integrated, 0x02, "aa");
        let a = gpu("GPU.1", GpuClass::Discrete, 0x03, "bb");
        let b = gpu("GPU.2", GpuClass::Discrete, 0x04, "cc");
        let rename = |device: &OvDevice, name: &str| OvDevice {
            name: name.to_string(),
            kind: classify(name),
            ..device.clone()
        };
        let listings = [
            vec![cpu(), igpu, a.clone(), b.clone()],
            // iGPU disabled in firmware: names shift down.
            vec![cpu(), rename(&a, "GPU.0"), rename(&b, "GPU.1")],
            // Then A is removed: B is the only GPU and is listed as bare GPU.
            vec![cpu(), rename(&b, "GPU")],
            // Both dGPUs back, in the other order.
            vec![cpu(), rename(&b, "GPU.0"), rename(&a, "GPU.1")],
        ];
        for listing in &listings {
            let by_pci = resolve(&"pci:0000:00:04.0".parse().unwrap(), listing).unwrap();
            let by_uuid = resolve(&OpenVinoDeviceSpec::Uuid(uuid("cc")), listing).unwrap();
            assert_eq!(by_pci.uuid, Some(uuid("cc")), "pci pin in {listing:?}");
            assert_eq!(by_uuid.uuid, Some(uuid("cc")), "uuid pin in {listing:?}");
            assert_eq!(
                resolve(&OpenVinoDeviceSpec::Dgpu(None), listing)
                    .unwrap()
                    .class,
                GpuClass::Discrete
            );
        }
        // Index selectors are relative: gpu.2 stops existing when the list shrinks.
        assert!(resolved_name("gpu.2", &listings[0]).is_ok());
        assert!(matches!(
            resolved_name("gpu.2", &listings[1]),
            Err(ResolveError::NotFound { .. })
        ));
        assert!(matches!(
            resolved_name("gpu.2", &listings[2]),
            Err(ResolveError::NotFound { .. })
        ));
    }

    #[test]
    fn every_reported_selector_resolves_back_to_its_device() {
        let layouts = [
            igpu_and_two_dgpus(),
            vec![cpu(), gpu("GPU", GpuClass::Integrated, 0x02, "aa")],
            vec![
                cpu(),
                gpu("GPU.0", GpuClass::Discrete, 0x03, "bb"),
                gpu("GPU.1", GpuClass::Discrete, 0x04, "cc"),
            ],
        ];
        for layout in &layouts {
            for device in layout.iter().filter(|device| device.is_selectable()) {
                let selectors = selectors_for(device, layout);
                assert!(!selectors.is_empty(), "{} has selectors", device.name);
                for selector in selectors {
                    let spec: OpenVinoDeviceSpec = selector.parse().unwrap();
                    assert_eq!(
                        resolve(&spec, layout).unwrap().name,
                        device.name,
                        "{selector} must resolve back"
                    );
                }
            }
        }
        // The single-GPU host from the probe: the whole list, whole value.
        let single = vec![cpu(), gpu("GPU", GpuClass::Integrated, 0x02, "aa")];
        assert_eq!(
            selectors_for(&single[1], &single),
            vec![
                "gpu".to_string(),
                "igpu".to_string(),
                "gpu.0".to_string(),
                "pci:0000:00:02.0".to_string(),
                format!("uuid:{}", uuid("aa")),
            ]
        );
        // The second dGPU is only reachable by index or pin, not by plain dgpu.
        let layout = igpu_and_two_dgpus();
        assert_eq!(
            selectors_for(&layout[3], &layout),
            vec![
                "dgpu.1".to_string(),
                "gpu.2".to_string(),
                "pci:0000:00:04.0".to_string(),
                format!("uuid:{}", uuid("cc")),
            ]
        );
        // Never selectable devices have none.
        assert_eq!(selectors_for(&layout[0], &layout), Vec::<String>::new());
    }

    #[test]
    fn names_are_classified_before_properties() {
        assert_eq!(classify("CPU"), DeviceKind::Cpu);
        assert_eq!(classify("GPU"), DeviceKind::Gpu(0));
        assert_eq!(classify("GPU.3"), DeviceKind::Gpu(3));
        assert_eq!(classify("GPU.0.1"), DeviceKind::GpuTile);
        assert_eq!(classify("NPU"), DeviceKind::Npu);
        assert_eq!(classify("GPU."), DeviceKind::Other);
        assert_eq!(classify("GPU.x"), DeviceKind::Other);
        assert_eq!(classify("AUTO"), DeviceKind::Other);
        // A tile is listed but never selected, even when it says it is Intel.
        let tile = OvDevice {
            intel: true,
            class: GpuClass::Discrete,
            ..OvDevice::named("GPU.0.1")
        };
        assert!(!tile.is_selectable());
        assert!(resolve(&OpenVinoDeviceSpec::Auto, &[tile]).is_err());
    }

    #[test]
    fn property_strings_from_the_test_hosts_parse() {
        // Values read through the openvino crate on an Iris Xe host.
        assert_eq!(
            parse_pci_info("{domain: 0 bus: 0 device: 0x2 function: 0}")
                .unwrap()
                .to_string(),
            "0000:00:02.0"
        );
        assert_eq!(
            parse_pci_info("{domain: 1 bus: 26 device: 0x1f function: 7}")
                .unwrap()
                .to_string(),
            "0001:1a:1f.7"
        );
        assert!(is_intel_architecture("GPU: vendor=0x8086 arch=v12.3.0"));
        assert!(!is_intel_architecture("GPU: vendor=0x10de arch=sm_89"));
        assert_eq!(parse_gpu_class("integrated"), GpuClass::Integrated);
        assert_eq!(parse_gpu_class("DISCRETE"), GpuClass::Discrete);
        assert_eq!(parse_gpu_class("virtual"), GpuClass::Unknown);
        assert_eq!(parse_memory_bytes("30269562880"), Some(30_269_562_880));
        assert_eq!(parse_memory_bytes("30.3 GB"), None);
        assert_eq!(
            parse_capabilities("FP32 BIN FP16 INT8 GPU_USM_MEMORY EXPORT_IMPORT"),
            [
                "FP32",
                "BIN",
                "FP16",
                "INT8",
                "GPU_USM_MEMORY",
                "EXPORT_IMPORT"
            ]
        );
        assert_eq!(parse_capabilities("FP32, BIN"), ["FP32", "BIN"]);
    }

    #[test]
    fn a_uuid_is_lowercased_and_an_all_zero_one_is_none() {
        assert_eq!(
            parse_uuid("8680A6460C0000000002000000000000").as_deref(),
            Some("8680a6460c0000000002000000000000")
        );
        assert_eq!(parse_uuid("00000000000000000000000000000000"), None);
        assert_eq!(parse_uuid(""), None);
    }

    #[test]
    fn malformed_pci_info_is_none() {
        for value in [
            "",
            "domain: 0 bus: 0 device: 0x2 function: 0",
            "{domain: 0 bus: 0 device: 2 function: 0}",
            "{domain: 0 bus: 0 device: 0x20 function: 0}",
            "{domain: 0 bus: 0 device: 0x2 function: 8}",
            "{domain: 0 bus: 300 device: 0x2 function: 0}",
            "{domain: 0 bus: 0 device: 0x2 function: 0 extra}",
            "{bus: 0 domain: 0 device: 0x2 function: 0}",
            // All zero: the host bridge's address, never a GPU's.
            "{domain: 0 bus: 0 device: 0x0 function: 0}",
        ] {
            assert_eq!(parse_pci_info(value), None, "{value:?}");
        }
    }
}
