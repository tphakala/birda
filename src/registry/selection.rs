//! Choosing which model variant to download.
//!
//! `models install` runs without the ONNX runtime (`command_requires_runtime`
//! in `src/lib.rs` puts every `models` subcommand on the no-runtime side), so
//! selection cannot ask ORT what it is able to execute. It does not need to:
//! the CUDA and `TensorRT` probes are filesystem library searches.
//!
//! Auto-selection deliberately refuses to pick a narrow variant on a weak
//! signal. Choosing `int8-arm` merely because the host is aarch64 would strand
//! a user who later runs with `OpenVINO`, which the Perch manifest marks
//! unsupported for that file, so a host with nothing better to go on gets the
//! family default, which every backend supports.

use super::types::{ModelEntry, ModelVariant};
use crate::config::InferenceDevice;
use crate::error::{Error, Result};

/// Manifest key for a CUDA host.
const KEY_CUDA: &str = "cuda";
/// Manifest key for a `TensorRT` host.
const KEY_TENSORRT: &str = "tensorrt";
/// Manifest architecture name for x86-64, hyphenated where Rust uses an underscore.
const ARCH_KEY_X86_64: &str = "x86-64";
/// Rust's name for the same architecture.
const RUST_ARCH_X86_64: &str = "x86_64";

/// Why a variant was chosen, reported to the user before downloading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionReason {
    /// The user named it with `--variant`.
    Explicit,
    /// Derived from the configured inference device.
    ConfiguredDevice,
    /// Derived from libraries detected on this host.
    DetectedLibrary,
    /// Derived from the CPU architecture.
    ArchDefault,
    /// The family default, supported on every backend.
    FamilyDefault,
}

impl SelectionReason {
    /// Short explanation shown next to the chosen variant.
    #[must_use]
    pub const fn describe(self) -> &'static str {
        match self {
            Self::Explicit => "requested with --variant",
            Self::ConfiguredDevice => "matches the configured inference device",
            Self::DetectedLibrary => "matches the GPU libraries found on this system",
            Self::ArchDefault => "best for this CPU architecture",
            Self::FamilyDefault => "default, supported on every backend",
        }
    }
}

/// The variant to download and why.
#[derive(Debug)]
pub struct VariantChoice<'a> {
    /// The chosen variant.
    pub variant: &'a ModelVariant,
    /// Why it was chosen.
    pub reason: SelectionReason,
}

/// Host capabilities consulted during selection.
///
/// A trait so the precedence rules can be tested without a GPU.
pub trait HardwareProbe {
    /// Whether CUDA runtime libraries are present.
    fn cuda_available(&self) -> bool;
    /// Whether `TensorRT` libraries are present.
    fn tensorrt_available(&self) -> bool;
    /// Target architecture name, as in `std::env::consts::ARCH`.
    fn arch(&self) -> &str;
}

/// The real host.
#[derive(Debug, Clone, Copy)]
pub struct SystemProbe;

impl HardwareProbe for SystemProbe {
    fn cuda_available(&self) -> bool {
        crate::inference::is_cuda_available()
    }

    fn tensorrt_available(&self) -> bool {
        crate::inference::is_tensorrt_available()
    }

    fn arch(&self) -> &str {
        std::env::consts::ARCH
    }
}

/// Translate `std::env::consts::ARCH` into the manifest's architecture spelling.
fn arch_key(arch: &str) -> &str {
    if arch == RUST_ARCH_X86_64 {
        ARCH_KEY_X86_64
    } else {
        arch
    }
}

/// Manifest key implied by an explicitly configured inference device.
///
/// Exhaustive by design, with no wildcard arm: a device added to
/// [`InferenceDevice`] must be classified here rather than silently inheriting
/// "no opinion". Devices with no manifest vocabulary return `None` and fall
/// through to library detection, which is the honest answer. The manifests
/// describe backends the publisher benchmarked, and inventing a key for the
/// rest would select a variant on no evidence.
fn device_key(device: InferenceDevice, probe: &dyn HardwareProbe) -> Option<String> {
    let arch = arch_key(probe.arch());
    match device {
        InferenceDevice::Auto
        | InferenceDevice::Cpu
        | InferenceDevice::Gpu
        | InferenceDevice::DirectMl
        | InferenceDevice::Rocm
        | InferenceDevice::OneDnn
        | InferenceDevice::Qnn
        | InferenceDevice::Acl
        | InferenceDevice::ArmNn => None,
        InferenceDevice::Cuda => Some(KEY_CUDA.to_string()),
        InferenceDevice::TensorRt => Some(KEY_TENSORRT.to_string()),
        InferenceDevice::OpenVino => Some(format!("{arch}/openvino-cpu")),
        InferenceDevice::CoreMl => Some(format!("{arch}/coreml")),
        InferenceDevice::Xnnpack => Some(format!("{arch}/xnnpack")),
    }
}

/// Manifest keys implied by libraries found on this host, most capable first.
///
/// Both are returned rather than only the best one. A single key would let
/// `TensorRT` shadow CUDA: on a host with both, a family whose manifest maps
/// `cuda` but not `tensorrt` would skip straight past a perfectly good CUDA
/// variant to the architecture default.
fn detected_keys(probe: &dyn HardwareProbe) -> Vec<String> {
    let mut keys = Vec::new();
    if probe.tensorrt_available() {
        keys.push(KEY_TENSORRT.to_string());
    }
    if probe.cuda_available() {
        keys.push(KEY_CUDA.to_string());
    }
    keys
}

/// Pick the variant to install.
///
/// Precedence, first match wins: `requested`, the configured device, detected
/// libraries, the architecture key, the family default.
///
/// Auto-selection degrades rather than fails. A region may publish fewer
/// variants than the global model, and a user who asked for a region asked for
/// a region, not for fp16, so a key resolving to a variant that region does not
/// have falls through to the next rung. An explicit `--variant` is the
/// opposite: the user named it, so giving them a different one silently would
/// be wrong, and it errors.
pub fn select_variant<'a>(
    entry: &'a ModelEntry,
    region: Option<&str>,
    requested: Option<&str>,
    device: InferenceDevice,
    probe: &dyn HardwareProbe,
) -> Result<VariantChoice<'a>> {
    let available = entry.variant_ids_for(region);
    if available.is_empty() {
        // Two different failures share this branch, and they need different
        // words. A named region that does not exist is a user typo, answered
        // with the list of regions. No global variant at all is a broken
        // registry, and reporting it as "no region 'global'" would send the
        // user hunting for a region name that was never the problem.
        let Some(region) = region else {
            return Err(Error::VariantNotFound {
                model_id: entry.id.clone(),
                variant: "global".to_string(),
                available: "none, this model publishes regional variants only".to_string(),
            });
        };

        return Err(Error::RegionNotFound {
            model_id: entry.id.clone(),
            region: region.to_string(),
            available: entry
                .regions()
                .iter()
                .filter_map(|v| v.region.clone())
                .collect::<Vec<_>>()
                .join(", "),
        });
    }

    if let Some(id) = requested {
        let variant = entry
            .find_variant(region, id)
            .ok_or_else(|| Error::VariantNotFound {
                model_id: entry.id.clone(),
                variant: id.to_string(),
                available: available.join(", "),
            })?;
        return Ok(VariantChoice {
            variant,
            reason: SelectionReason::Explicit,
        });
    }

    // A configured device is an instruction, not a hint, so it is the only rung
    // consulted when it is set. Falling through to the detected-library or
    // architecture rungs would answer a different question than the user asked:
    // on aarch64 with OpenVINO configured, the manifest key `aarch64/openvino-cpu`
    // can miss (Perch publishes `aarch64-a76/openvino`), and the architecture
    // rung would then hand back `int8-arm`, which that same manifest marks
    // unsupported on OpenVINO. Better to fall straight through to the family
    // default, which every backend supports.
    let mut candidates: Vec<(String, SelectionReason)> = Vec::new();
    if let Some(key) = device_key(device, probe) {
        candidates.push((key, SelectionReason::ConfiguredDevice));
    } else {
        candidates.extend(
            detected_keys(probe)
                .into_iter()
                .map(|key| (key, SelectionReason::DetectedLibrary)),
        );
        candidates.push((
            format!("{}/onnxruntime", arch_key(probe.arch())),
            SelectionReason::ArchDefault,
        ));
    }

    for (key, reason) in candidates {
        if let Some(variant) = entry
            .selection
            .get(&key)
            .and_then(|id| entry.find_variant(region, id))
        {
            return Ok(VariantChoice { variant, reason });
        }
    }

    let default_id = entry
        .default_variant
        .as_deref()
        .ok_or_else(|| Error::VariantNotFound {
            model_id: entry.id.clone(),
            variant: "default".to_string(),
            available: available.join(", "),
        })?;
    let variant = entry
        .find_variant(region, default_id)
        .ok_or_else(|| Error::VariantNotFound {
            model_id: entry.id.clone(),
            variant: default_id.to_string(),
            available: available.join(", "),
        })?;
    Ok(VariantChoice {
        variant,
        reason: SelectionReason::FamilyDefault,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Test setup code - panics are acceptable
mod tests {
    use super::*;
    use crate::registry::types::{FileInfo, LicenseInfo};

    struct FakeProbe {
        cuda: bool,
        tensorrt: bool,
        arch: &'static str,
    }

    impl HardwareProbe for FakeProbe {
        fn cuda_available(&self) -> bool {
            self.cuda
        }
        fn tensorrt_available(&self) -> bool {
            self.tensorrt
        }
        fn arch(&self) -> &str {
            self.arch
        }
    }

    const fn cpu_probe() -> FakeProbe {
        FakeProbe {
            cuda: false,
            tensorrt: false,
            arch: "x86_64",
        }
    }

    fn file(name: &str) -> FileInfo {
        FileInfo {
            url: format!("https://huggingface.co/x/{name}"),
            filename: name.to_string(),
            sha256: Some("abc".to_string()),
            size_bytes: Some(1),
        }
    }

    fn variant(id: &str, region: Option<&str>) -> ModelVariant {
        ModelVariant {
            id: id.to_string(),
            region: region.map(str::to_string),
            region_name: region.map(str::to_string),
            group: None,
            group_name: None,
            group_order: 0,
            classes: Some(10),
            model: file(&format!("{id}.onnx")),
            labels: file("labels.txt"),
        }
    }

    fn entry(
        default_variant: &str,
        selection: &[(&str, &str)],
        variants: Vec<ModelVariant>,
    ) -> ModelEntry {
        ModelEntry {
            id: "test".to_string(),
            name: "Test".to_string(),
            description: "d".to_string(),
            vendor: "v".to_string(),
            version: "1.0".to_string(),
            model_type: "birdnet-v30".to_string(),
            license: LicenseInfo {
                r#type: "CC".to_string(),
                url: "https://x".to_string(),
                commercial_use: true,
                attribution_required: false,
                share_alike: false,
            },
            files: None,
            build: Some(1),
            default_variant: Some(default_variant.to_string()),
            selection: selection
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
            variants,
            recommended: false,
        }
    }

    fn birdnet_entry() -> ModelEntry {
        entry(
            "fp32",
            &[
                ("cuda", "fp16"),
                ("tensorrt", "fp16"),
                ("x86-64/openvino-gpu", "fp16"),
            ],
            vec![
                variant("fp32", None),
                variant("fp16", None),
                variant("fp32", Some("nordic")),
                variant("fp16", Some("nordic")),
            ],
        )
    }

    #[test]
    fn test_explicit_variant_wins_over_every_other_signal() {
        let e = birdnet_entry();
        let probe = FakeProbe {
            cuda: true,
            tensorrt: true,
            arch: "x86_64",
        };
        let choice = select_variant(&e, None, Some("fp32"), InferenceDevice::Cuda, &probe).unwrap();
        assert_eq!(choice.variant.id, "fp32");
        assert_eq!(choice.reason, SelectionReason::Explicit);
    }

    #[test]
    fn test_unknown_explicit_variant_errors_and_names_the_valid_ids() {
        let e = birdnet_entry();
        let err = select_variant(&e, None, Some("int4"), InferenceDevice::Auto, &cpu_probe())
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("int4"), "got: {message}");
        assert!(message.contains("fp32"), "got: {message}");
    }

    #[test]
    fn test_configured_cuda_device_selects_fp16() {
        let e = birdnet_entry();
        let choice = select_variant(&e, None, None, InferenceDevice::Cuda, &cpu_probe()).unwrap();
        assert_eq!(choice.variant.id, "fp16");
        assert_eq!(choice.reason, SelectionReason::ConfiguredDevice);
    }

    #[test]
    fn test_detected_cuda_library_selects_fp16_under_auto() {
        let e = birdnet_entry();
        let probe = FakeProbe {
            cuda: true,
            tensorrt: false,
            arch: "x86_64",
        };
        let choice = select_variant(&e, None, None, InferenceDevice::Auto, &probe).unwrap();
        assert_eq!(choice.variant.id, "fp16");
        assert_eq!(choice.reason, SelectionReason::DetectedLibrary);
    }

    #[test]
    fn test_tensorrt_does_not_shadow_cuda() {
        // A host with both libraries, and a family whose manifest maps `cuda`
        // but not `tensorrt`. Returning only the most capable detected key
        // would skip the perfectly good CUDA variant and fall through to the
        // architecture default.
        let e = entry(
            "fp32",
            &[("cuda", "fp16")],
            vec![variant("fp32", None), variant("fp16", None)],
        );
        let probe = FakeProbe {
            cuda: true,
            tensorrt: true,
            arch: "x86_64",
        };
        let choice = select_variant(&e, None, None, InferenceDevice::Auto, &probe).unwrap();
        assert_eq!(choice.variant.id, "fp16");
        assert_eq!(choice.reason, SelectionReason::DetectedLibrary);
    }

    #[test]
    fn test_plain_cpu_host_falls_back_to_the_family_default() {
        let e = birdnet_entry();
        let choice = select_variant(&e, None, None, InferenceDevice::Auto, &cpu_probe()).unwrap();
        assert_eq!(choice.variant.id, "fp32");
        assert_eq!(choice.reason, SelectionReason::FamilyDefault);
    }

    #[test]
    fn test_arch_key_is_consulted_before_the_family_default() {
        let e = entry(
            "no-dft-fp32",
            &[("aarch64/onnxruntime", "int8-arm")],
            vec![variant("no-dft-fp32", None), variant("int8-arm", None)],
        );
        let probe = FakeProbe {
            cuda: false,
            tensorrt: false,
            arch: "aarch64",
        };
        let choice = select_variant(&e, None, None, InferenceDevice::Auto, &probe).unwrap();
        assert_eq!(choice.variant.id, "int8-arm");
        assert_eq!(choice.reason, SelectionReason::ArchDefault);
    }

    #[test]
    fn test_x86_64_is_spelled_the_manifest_way_in_the_arch_key() {
        // Rust says x86_64, the manifests say x86-64. Getting this wrong makes
        // every architecture key miss on the most common desktop platform.
        let e = entry(
            "fp32",
            &[("x86-64/onnxruntime", "fp16")],
            vec![variant("fp32", None), variant("fp16", None)],
        );
        let choice = select_variant(&e, None, None, InferenceDevice::Auto, &cpu_probe()).unwrap();
        assert_eq!(choice.variant.id, "fp16");
        assert_eq!(choice.reason, SelectionReason::ArchDefault);
    }

    #[test]
    fn test_auto_falls_back_when_the_selected_variant_is_missing_for_the_region() {
        // A region may publish fewer variants than the global model. Auto must
        // degrade to the family default rather than fail: the user asked for a
        // region, not for fp16.
        let e = entry(
            "fp32",
            &[("cuda", "fp16")],
            vec![
                variant("fp32", None),
                variant("fp16", None),
                variant("fp32", Some("nordic")),
            ],
        );
        let probe = FakeProbe {
            cuda: true,
            tensorrt: false,
            arch: "x86_64",
        };
        let choice =
            select_variant(&e, Some("nordic"), None, InferenceDevice::Auto, &probe).unwrap();
        assert_eq!(choice.variant.id, "fp32");
        assert_eq!(choice.reason, SelectionReason::FamilyDefault);
    }

    #[test]
    fn test_explicit_variant_missing_for_the_region_is_an_error() {
        // Explicit is different from auto: the user named a variant, so quietly
        // handing them another one would be wrong.
        let e = entry(
            "fp32",
            &[],
            vec![variant("fp32", None), variant("fp32", Some("nordic"))],
        );
        let err = select_variant(
            &e,
            Some("nordic"),
            Some("fp16"),
            InferenceDevice::Auto,
            &cpu_probe(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("fp16"));
    }

    #[test]
    fn test_unknown_region_errors_and_names_valid_regions() {
        let e = birdnet_entry();
        let err = select_variant(
            &e,
            Some("atlantis"),
            None,
            InferenceDevice::Auto,
            &cpu_probe(),
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("atlantis"), "got: {message}");
        assert!(message.contains("nordic"), "got: {message}");
    }

    #[test]
    fn test_a_configured_device_never_falls_through_to_the_architecture_rung() {
        // The Perch shape on an aarch64 host with OpenVINO configured. The
        // configured key misses, because the manifest publishes
        // aarch64-a76/openvino. Falling through to aarch64/onnxruntime would
        // hand back int8-arm, which the same manifest marks unsupported on
        // OpenVINO, so the install would succeed and then fail at inference.
        let e = entry(
            "no-dft-fp32",
            &[
                ("aarch64-a76/openvino", "no-dft-fp32"),
                ("aarch64/onnxruntime", "int8-arm"),
            ],
            vec![variant("no-dft-fp32", None), variant("int8-arm", None)],
        );
        let probe = FakeProbe {
            cuda: false,
            tensorrt: false,
            arch: "aarch64",
        };

        let choice = select_variant(&e, None, None, InferenceDevice::OpenVino, &probe).unwrap();

        assert_eq!(choice.variant.id, "no-dft-fp32");
        assert_eq!(choice.reason, SelectionReason::FamilyDefault);
    }

    #[test]
    fn test_a_configured_device_is_not_overridden_by_detected_libraries() {
        // Someone who configured OpenVINO does not want the CUDA variant just
        // because the CUDA libraries happen to be installed.
        let e = entry(
            "fp32",
            &[("cuda", "fp16")],
            vec![variant("fp32", None), variant("fp16", None)],
        );
        let probe = FakeProbe {
            cuda: true,
            tensorrt: false,
            arch: "x86_64",
        };

        let choice = select_variant(&e, None, None, InferenceDevice::OpenVino, &probe).unwrap();

        assert_eq!(choice.variant.id, "fp32");
        assert_eq!(choice.reason, SelectionReason::FamilyDefault);
    }

    #[test]
    fn test_a_device_with_no_manifest_vocabulary_falls_through_to_detection() {
        // ROCm has no manifest key. It must not suppress the CUDA libraries
        // this host actually has.
        let e = birdnet_entry();
        let probe = FakeProbe {
            cuda: true,
            tensorrt: false,
            arch: "x86_64",
        };
        let choice = select_variant(&e, None, None, InferenceDevice::Rocm, &probe).unwrap();
        assert_eq!(choice.variant.id, "fp16");
        assert_eq!(choice.reason, SelectionReason::DetectedLibrary);
    }

    #[test]
    fn test_a_missing_global_variant_is_not_reported_as_a_missing_region() {
        // A registry with regional variants but no global one is broken, not a
        // user typo. Saying "has no region 'global'" would send someone hunting
        // for a region name that was never the problem.
        let e = entry("fp32", &[], vec![variant("fp32", Some("nordic"))]);

        let err = select_variant(&e, None, None, InferenceDevice::Auto, &cpu_probe()).unwrap_err();
        let message = err.to_string();

        assert!(
            !message.contains("has no region"),
            "must not blame a region: {message}"
        );
        assert!(message.contains("global"), "got: {message}");
    }

    #[test]
    fn test_an_entry_without_a_default_variant_errors_rather_than_guessing() {
        let mut e = birdnet_entry();
        e.default_variant = None;
        e.selection.clear();
        let err = select_variant(&e, None, None, InferenceDevice::Auto, &cpu_probe()).unwrap_err();
        assert!(err.to_string().contains("default"));
    }
}
