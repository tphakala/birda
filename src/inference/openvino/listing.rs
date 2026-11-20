//! What `birda openvino devices` reports.

use super::device::{DeviceKind, OvDevice, selectors_for};
use super::engine::resolve_error;
use super::runtime::Runtime;
use crate::config::OpenVinoSettings;
use crate::error::Result;
use std::fmt::Write;
use std::path::PathBuf;

/// One device in the listing.
#[derive(Debug, Clone)]
pub struct ListedDevice {
    /// The device record.
    pub device: OvDevice,
    /// Why birda will not run on it, or `None` when it is selectable.
    pub refusal: Option<String>,
    /// The specs that resolve to this device.
    pub selectors: Vec<String>,
}

/// The runtime, the configured spec and every listed device.
#[derive(Debug, Clone)]
pub struct DeviceListing {
    /// Runtime build number.
    pub runtime_version: String,
    /// The library that was loaded, when loaded by path.
    pub library: Option<PathBuf>,
    /// The configured spec.
    pub configured: String,
    /// The device the spec resolves to, or the reason it does not.
    pub resolved: std::result::Result<String, String>,
    /// All devices, in the runtime's order.
    pub devices: Vec<ListedDevice>,
}

fn refusal(device: &OvDevice) -> Option<String> {
    match device.kind {
        DeviceKind::Gpu(_) if device.intel => None,
        DeviceKind::Gpu(_) => Some("not an Intel GPU".to_string()),
        DeviceKind::GpuTile => Some("GPU tiles cannot be selected".to_string()),
        DeviceKind::Cpu => Some("CPU inference runs on ONNX Runtime; use --cpu".to_string()),
        DeviceKind::Npu => Some("the NPU is not supported yet".to_string()),
        DeviceKind::Other => Some("not a GPU".to_string()),
    }
}

/// Load the runtime and describe its devices.
///
/// # Errors
///
/// The `OpenVino*` errors when the runtime cannot be located or loaded.
pub fn list(settings: &OpenVinoSettings) -> Result<DeviceListing> {
    let runtime = Runtime::load(settings.runtime_path.as_deref())?;
    let resolved = super::device::resolve(&settings.device, &runtime.devices)
        .map(OvDevice::describe)
        .map_err(|error| resolve_error(error, &runtime).to_string());
    let devices = runtime
        .devices
        .iter()
        .map(|device| ListedDevice {
            device: device.clone(),
            refusal: refusal(device),
            selectors: selectors_for(device, &runtime.devices),
        })
        .collect();
    Ok(DeviceListing {
        runtime_version: runtime.version.clone(),
        library: runtime.library,
        configured: settings.device.to_string(),
        resolved,
        devices,
    })
}

/// Format a byte count with a binary unit, one decimal.
fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    #[allow(clippy::cast_precision_loss)]
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

/// The listing as text for a terminal.
#[must_use]
pub fn render_human(listing: &DeviceListing) -> String {
    let mut out = String::new();
    let library = listing
        .library
        .as_ref()
        .map_or_else(String::new, |path| format!(" ({})", path.display()));
    let _ = writeln!(
        out,
        "OpenVINO runtime: {}{library}",
        listing.runtime_version
    );
    match &listing.resolved {
        Ok(device) => {
            let _ = writeln!(out, "Configured device: {} -> {device}", listing.configured);
        }
        Err(reason) => {
            let _ = writeln!(
                out,
                "Configured device: {} -> not available: {reason}",
                listing.configured
            );
        }
    }
    let _ = writeln!(out);
    if listing.devices.is_empty() {
        let _ = writeln!(out, "OpenVINO lists no devices.");
    }
    for listed in &listing.devices {
        let device = &listed.device;
        let _ = writeln!(
            out,
            "{}  {}",
            device.name,
            device.full_name.as_deref().unwrap_or("")
        );
        if let Some(reason) = &listed.refusal {
            let _ = writeln!(out, "    not selectable: {reason}");
            continue;
        }
        let _ = writeln!(out, "    class: {}", device.class.as_str());
        if let Some(pci) = device.pci {
            let _ = writeln!(out, "    pci: {pci}");
        }
        if let Some(uuid) = &device.uuid {
            let _ = writeln!(out, "    uuid: {uuid}");
        }
        if let Some(bytes) = device.memory_bytes {
            let _ = writeln!(out, "    memory: {}", human_bytes(bytes));
        }
        if let Some(architecture) = &device.architecture {
            let _ = writeln!(out, "    architecture: {architecture}");
        }
        if !device.capabilities.is_empty() {
            let _ = writeln!(out, "    capabilities: {}", device.capabilities.join(" "));
        }
        let _ = writeln!(
            out,
            "    select with --openvino-device: {}",
            listed.selectors.join(" | ")
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PciAddress;
    use crate::inference::openvino::device::GpuClass;

    fn iris_xe_listing() -> DeviceListing {
        let gpu = OvDevice {
            full_name: Some("Intel(R) Iris(R) Xe Graphics (iGPU)".to_string()),
            class: GpuClass::Integrated,
            intel: true,
            architecture: Some("GPU: vendor=0x8086 arch=v12.3.0".to_string()),
            uuid: Some("8680a6460c0000000002000000000000".to_string()),
            pci: Some(PciAddress {
                domain: 0,
                bus: 0,
                device: 2,
                function: 0,
            }),
            memory_bytes: Some(30_269_562_880),
            capabilities: vec!["FP32".to_string(), "FP16".to_string()],
            ..OvDevice::named("GPU")
        };
        let cpu = OvDevice::named("CPU");
        let devices = vec![cpu, gpu];
        DeviceListing {
            runtime_version: "2025.4.1-19314".to_string(),
            library: Some(PathBuf::from("/usr/lib/libopenvino_c.so")),
            configured: "auto".to_string(),
            resolved: Ok(devices[1].describe()),
            devices: devices
                .iter()
                .map(|device| ListedDevice {
                    device: device.clone(),
                    refusal: refusal(device),
                    selectors: selectors_for(device, &devices),
                })
                .collect(),
        }
    }

    #[test]
    fn the_human_listing_names_selectors_and_refuses_the_cpu() {
        assert_eq!(
            render_human(&iris_xe_listing()),
            "OpenVINO runtime: 2025.4.1-19314 (/usr/lib/libopenvino_c.so)\n\
             Configured device: auto -> GPU (Intel(R) Iris(R) Xe Graphics (iGPU), integrated, pci 0000:00:02.0)\n\
             \n\
             CPU  \n\
             \x20   not selectable: CPU inference runs on ONNX Runtime; use --cpu\n\
             GPU  Intel(R) Iris(R) Xe Graphics (iGPU)\n\
             \x20   class: integrated\n\
             \x20   pci: 0000:00:02.0\n\
             \x20   uuid: 8680a6460c0000000002000000000000\n\
             \x20   memory: 28.2 GiB\n\
             \x20   architecture: GPU: vendor=0x8086 arch=v12.3.0\n\
             \x20   capabilities: FP32 FP16\n\
             \x20   select with --openvino-device: gpu | igpu | gpu.0 | pci:0000:00:02.0 | \
             uuid:8680a6460c0000000002000000000000\n"
        );
    }

    #[test]
    fn an_unresolvable_spec_is_reported_not_hidden() {
        let mut listing = iris_xe_listing();
        listing.configured = "dgpu".to_string();
        listing.resolved = Err("OpenVINO device 'dgpu' is not available".to_string());
        let text = render_human(&listing);
        assert_eq!(
            text.lines().nth(1),
            Some(
                "Configured device: dgpu -> not available: OpenVINO device 'dgpu' is not available"
            )
        );
    }

    #[test]
    fn non_gpu_devices_are_refused_with_a_reason() {
        let tile = OvDevice::named("GPU.0.1");
        let npu = OvDevice::named("NPU");
        let nvidia = OvDevice {
            kind: DeviceKind::Gpu(1),
            ..OvDevice::named("GPU.1")
        };
        assert_eq!(
            refusal(&tile).as_deref(),
            Some("GPU tiles cannot be selected")
        );
        assert_eq!(
            refusal(&npu).as_deref(),
            Some("the NPU is not supported yet")
        );
        assert_eq!(refusal(&nvidia).as_deref(), Some("not an Intel GPU"));
        assert_eq!(
            refusal(&OvDevice::named("AUTO")).as_deref(),
            Some("not a GPU")
        );
    }
}
