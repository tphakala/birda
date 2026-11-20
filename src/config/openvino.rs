//! Device selection for the native `OpenVINO` backend.
//!
//! This lives in `config` rather than next to the backend because the spec is
//! read from `config.toml`, and `config` must not depend on `inference`.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

/// Number of hex digits in a device UUID.
const UUID_HEX_DIGITS: usize = 32;

/// Highest device number in a PCI address.
const PCI_DEVICE_MAX: u8 = 0x1f;

/// Highest function number in a PCI address.
const PCI_FUNCTION_MAX: u8 = 7;

/// Why a device spec was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ParseDeviceSpecError(String);

/// A PCI address, as `lspci -D` prints it: `DDDD:BB:DD.F`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PciAddress {
    /// PCI domain.
    pub domain: u16,
    /// PCI bus.
    pub bus: u8,
    /// PCI device (0 to 0x1f).
    pub device: u8,
    /// PCI function (0 to 7).
    pub function: u8,
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

impl FromStr for PciAddress {
    type Err = ParseDeviceSpecError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let invalid = || {
            ParseDeviceSpecError(format!(
                "'{s}' is not a PCI address; write it as DDDD:BB:DD.F, the way `lspci -D` prints it"
            ))
        };
        let (domain, rest) = s.split_once(':').ok_or_else(invalid)?;
        let (bus, rest) = rest.split_once(':').ok_or_else(invalid)?;
        let (device, function) = rest.split_once('.').ok_or_else(invalid)?;

        if domain.len() != 4 || bus.len() != 2 || device.len() != 2 || function.len() != 1 {
            return Err(invalid());
        }
        let hex = |digits: &str| u32::from_str_radix(digits, 16).map_err(|_| invalid());
        let domain = u16::try_from(hex(domain)?).map_err(|_| invalid())?;
        let bus = u8::try_from(hex(bus)?).map_err(|_| invalid())?;
        let device = u8::try_from(hex(device)?).map_err(|_| invalid())?;
        let function = u8::try_from(hex(function)?).map_err(|_| invalid())?;
        if device > PCI_DEVICE_MAX || function > PCI_FUNCTION_MAX {
            return Err(invalid());
        }
        Ok(Self {
            domain,
            bus,
            device,
            function,
        })
    }
}

/// Which `OpenVINO` device the native backend runs on.
///
/// Only Intel GPUs are ever selected. There is deliberately no spec for the
/// `OpenVINO` CPU device: CPU inference stays on ONNX Runtime.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub enum OpenVinoDeviceSpec {
    /// First Intel discrete GPU, else first Intel integrated GPU, else first
    /// Intel GPU of unknown class.
    #[default]
    Auto,
    /// The same choice as [`Self::Auto`] when no index is given; with an index,
    /// the device `OpenVINO` itself calls `GPU.N`.
    Gpu(Option<u32>),
    /// The Nth Intel integrated GPU (the first when no index is given).
    Igpu(Option<u32>),
    /// The Nth Intel discrete GPU (the first when no index is given).
    Dgpu(Option<u32>),
    /// The GPU at a PCI address.
    Pci(PciAddress),
    /// The GPU with a device UUID (32 lowercase hex digits).
    Uuid(String),
    /// Reserved: parses, and resolution reports that the NPU is unsupported.
    Npu,
}

impl OpenVinoDeviceSpec {
    /// Whether this is the default spec (used to skip it when serializing).
    #[must_use]
    pub fn is_auto(&self) -> bool {
        *self == Self::Auto
    }
}

/// Parse `<prefix>` or `<prefix>.<N>`; `None` when `s` is not that form.
fn indexed(s: &str, prefix: &str) -> Option<Result<Option<u32>, ParseDeviceSpecError>> {
    if s == prefix {
        return Some(Ok(None));
    }
    let index = s.strip_prefix(prefix)?.strip_prefix('.')?;
    // `u32::from_str` accepts a leading '+', which is not part of the grammar.
    let valid = !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit());
    Some(match index.parse::<u32>() {
        Ok(n) if valid => Ok(Some(n)),
        _ => Err(ParseDeviceSpecError(format!(
            "'{s}': the device number after '{prefix}.' must be a whole number"
        ))),
    })
}

impl FromStr for OpenVinoDeviceSpec {
    type Err = ParseDeviceSpecError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "auto" => return Ok(Self::Auto),
            "npu" => return Ok(Self::Npu),
            _ => {}
        }
        if let Some(address) = lower.strip_prefix("pci:") {
            return address.parse().map(Self::Pci);
        }
        if let Some(uuid) = lower.strip_prefix("uuid:") {
            if uuid.len() == UUID_HEX_DIGITS && uuid.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Ok(Self::Uuid(uuid.to_string()));
            }
            return Err(ParseDeviceSpecError(format!(
                "'{s}': a device UUID is {UUID_HEX_DIGITS} hex digits"
            )));
        }
        if let Some(parsed) = indexed(&lower, "gpu") {
            return parsed.map(Self::Gpu);
        }
        if let Some(parsed) = indexed(&lower, "igpu") {
            return parsed.map(Self::Igpu);
        }
        if let Some(parsed) = indexed(&lower, "dgpu") {
            return parsed.map(Self::Dgpu);
        }
        Err(ParseDeviceSpecError(format!(
            "'{s}' is not an OpenVINO device. Use auto, gpu, igpu, dgpu, gpu.N, igpu.N, \
             dgpu.N, pci:DDDD:BB:DD.F or uuid:<32 hex digits>; \
             `birda openvino devices` lists what is available"
        )))
    }
}

impl fmt::Display for OpenVinoDeviceSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let indexed = |f: &mut fmt::Formatter<'_>, name: &str, index: Option<u32>| match index {
            Some(n) => write!(f, "{name}.{n}"),
            None => f.write_str(name),
        };
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Gpu(index) => indexed(f, "gpu", *index),
            Self::Igpu(index) => indexed(f, "igpu", *index),
            Self::Dgpu(index) => indexed(f, "dgpu", *index),
            Self::Pci(address) => write!(f, "pci:{address}"),
            Self::Uuid(uuid) => write!(f, "uuid:{uuid}"),
            Self::Npu => f.write_str("npu"),
        }
    }
}

impl Serialize for OpenVinoDeviceSpec {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for OpenVinoDeviceSpec {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// Everything the native backend needs from the user's configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenVinoSettings {
    /// Which device to run on.
    pub device: OpenVinoDeviceSpec,
    /// Explicit runtime location (library file or directory), if configured.
    pub runtime_path: Option<PathBuf>,
}

/// A warning for an installed model variant that failed the `OpenVINO`
/// startup check on every Intel GPU tested, with the command that installs the
/// variant to use instead.
///
/// `None` for any other variant, and for a model that was not installed from
/// the registry, since there is no install command to give for it.
#[must_use]
pub fn unverified_variant_warning(model: &super::ModelConfig, model_name: &str) -> Option<String> {
    use crate::constants::openvino::{RECOMMENDED_VARIANT, UNVERIFIED_VARIANT};
    if model.variant.as_deref() != Some(UNVERIFIED_VARIANT) {
        return None;
    }
    let id = model.registry_id.as_deref()?;
    let region = model
        .region
        .as_deref()
        .map_or_else(String::new, |region| format!(" --region {region}"));
    Some(format!(
        "model '{model_name}' is the {UNVERIFIED_VARIANT} variant, which failed the OpenVINO \
         startup check on the Intel GPUs tested. If the check fails, install the \
         {RECOMMENDED_VARIANT} variant: birda models install {id} --variant \
         {RECOMMENDED_VARIANT}{region}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_fp16_install_gets_the_command_for_the_fp32_one() {
        let mut model = crate::config::ModelConfig {
            path: PathBuf::from("model.onnx"),
            labels: PathBuf::from("labels.txt"),
            model_type: crate::config::ModelType::BirdnetV30,
            meta_model: None,
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
            registry_id: Some("birdnet-v30".to_string()),
            installed_version: None,
            installed_build: None,
            region: Some("nordic".to_string()),
            variant: Some("fp16".to_string()),
        };
        assert_eq!(
            unverified_variant_warning(&model, "birdnet-v30-nordic").as_deref(),
            Some(
                "model 'birdnet-v30-nordic' is the fp16 variant, which failed the OpenVINO \
                 startup check on the Intel GPUs tested. If the check fails, install the fp32 \
                 variant: birda models install birdnet-v30 --variant fp32 --region nordic"
            )
        );
        model.variant = Some("fp32".to_string());
        assert_eq!(
            unverified_variant_warning(&model, "birdnet-v30-nordic"),
            None
        );
        model.variant = Some("fp16".to_string());
        model.registry_id = None;
        assert_eq!(
            unverified_variant_warning(&model, "birdnet-v30-nordic"),
            None
        );
    }

    #[test]
    fn accepted_specs_round_trip_in_lowercase() {
        let accepted = [
            ("auto", OpenVinoDeviceSpec::Auto),
            ("GPU", OpenVinoDeviceSpec::Gpu(None)),
            ("gpu.2", OpenVinoDeviceSpec::Gpu(Some(2))),
            ("igpu", OpenVinoDeviceSpec::Igpu(None)),
            ("iGPU.0", OpenVinoDeviceSpec::Igpu(Some(0))),
            ("dgpu.1", OpenVinoDeviceSpec::Dgpu(Some(1))),
            ("npu", OpenVinoDeviceSpec::Npu),
            (
                "pci:0000:00:02.0",
                OpenVinoDeviceSpec::Pci(PciAddress {
                    domain: 0,
                    bus: 0,
                    device: 2,
                    function: 0,
                }),
            ),
            (
                "PCI:0001:0A:1F.7",
                OpenVinoDeviceSpec::Pci(PciAddress {
                    domain: 1,
                    bus: 0x0a,
                    device: 0x1f,
                    function: 7,
                }),
            ),
            (
                "uuid:8680A6460C0000000002000000000000",
                OpenVinoDeviceSpec::Uuid("8680a6460c0000000002000000000000".to_string()),
            ),
        ];
        for (text, expected) in accepted {
            let parsed: OpenVinoDeviceSpec = text.parse().unwrap();
            assert_eq!(parsed, expected, "parsing {text}");
            assert_eq!(
                parsed.to_string(),
                text.to_ascii_lowercase(),
                "display {text}"
            );
        }
    }

    #[test]
    fn rejected_specs_name_the_input() {
        let generic = |input: &str| {
            format!(
                "'{input}' is not an OpenVINO device. Use auto, gpu, igpu, dgpu, gpu.N, igpu.N, \
                 dgpu.N, pci:DDDD:BB:DD.F or uuid:<32 hex digits>; \
                 `birda openvino devices` lists what is available"
            )
        };
        let index = |input: &str, prefix: &str| {
            format!("'{input}': the device number after '{prefix}.' must be a whole number")
        };
        let pci = |input: &str| {
            format!(
                "'{input}' is not a PCI address; write it as DDDD:BB:DD.F, the way `lspci -D` \
                 prints it"
            )
        };
        let uuid = |input: &str| format!("'{input}': a device UUID is 32 hex digits");
        let short_uuid = "uuid:8680a6460c000000000200000000000";
        let bad_uuid = "uuid:8680a6460c000000000200000000000g";

        let rejected = [
            ("", generic("")),
            ("cpu", generic("cpu")),
            ("gpu0", generic("gpu0")),
            ("gpu.", index("gpu.", "gpu")),
            ("gpu.x", index("gpu.x", "gpu")),
            ("gpu.+1", index("gpu.+1", "gpu")),
            ("gpu.-1", index("gpu.-1", "gpu")),
            ("igpu.1.2", index("igpu.1.2", "igpu")),
            ("pci:zz", pci("zz")),
            ("pci:0000:00:02", pci("0000:00:02")),
            ("pci:0000:00:20.0", pci("0000:00:20.0")),
            ("pci:0000:00:02.8", pci("0000:00:02.8")),
            ("pci:000:00:02.0", pci("000:00:02.0")),
            ("uuid:", uuid("uuid:")),
            (short_uuid, uuid(short_uuid)),
            (bad_uuid, uuid(bad_uuid)),
        ];
        for (text, message) in rejected {
            assert_eq!(
                text.parse::<OpenVinoDeviceSpec>().unwrap_err().to_string(),
                message,
                "rejecting {text:?}"
            );
        }
    }

    #[test]
    fn spec_serializes_as_a_plain_string_and_rejects_a_bad_one() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct Holder {
            spec: OpenVinoDeviceSpec,
        }

        let holder: Holder = toml::from_str("spec = \"DGPU.1\"").unwrap();
        assert_eq!(holder.spec, OpenVinoDeviceSpec::Dgpu(Some(1)));
        assert_eq!(toml::to_string(&holder).unwrap(), "spec = \"dgpu.1\"\n");

        let err = toml::from_str::<Holder>("spec = \"bogus\"").unwrap_err();
        assert!(
            err.to_string()
                .contains("'bogus' is not an OpenVINO device")
        );
    }
}
