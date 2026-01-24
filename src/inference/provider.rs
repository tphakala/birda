//! Execution provider metadata.

use birdnet_onnx::ExecutionProviderInfo;

/// Metadata for an execution provider.
pub struct ProviderMetadata {
    /// CLI flag identifier (e.g., "cuda", "tensorrt").
    pub id: &'static str,
    /// Display name (e.g., "CUDA", "`TensorRT`").
    pub name: &'static str,
    /// Full description for human output (e.g., "CUDA (NVIDIA GPU acceleration)").
    pub description: &'static str,
}

/// Get metadata for an execution provider.
#[must_use]
pub fn provider_metadata(provider: ExecutionProviderInfo) -> ProviderMetadata {
    match provider {
        ExecutionProviderInfo::Cpu => ProviderMetadata {
            id: "cpu",
            name: "CPU",
            description: "CPU (always available)",
        },
        ExecutionProviderInfo::Cuda => ProviderMetadata {
            id: "cuda",
            name: "CUDA",
            description: "CUDA (NVIDIA GPU acceleration)",
        },
        ExecutionProviderInfo::TensorRt => ProviderMetadata {
            id: "tensorrt",
            name: "TensorRT",
            description: "TensorRT (NVIDIA optimized inference)",
        },
        ExecutionProviderInfo::DirectMl => ProviderMetadata {
            id: "directml",
            name: "DirectML",
            description: "DirectML (Windows GPU acceleration)",
        },
        ExecutionProviderInfo::CoreMl => ProviderMetadata {
            id: "coreml",
            name: "CoreML",
            description: "CoreML (Apple GPU/Neural Engine)",
        },
        ExecutionProviderInfo::Rocm => ProviderMetadata {
            id: "rocm",
            name: "ROCm",
            description: "ROCm (AMD GPU acceleration)",
        },
        ExecutionProviderInfo::OpenVino => ProviderMetadata {
            id: "openvino",
            name: "OpenVINO",
            description: "OpenVINO (Intel optimization)",
        },
        ExecutionProviderInfo::OneDnn => ProviderMetadata {
            id: "onednn",
            name: "oneDNN",
            description: "oneDNN (Intel CPU optimization)",
        },
        ExecutionProviderInfo::Qnn => ProviderMetadata {
            id: "qnn",
            name: "QNN",
            description: "QNN (Qualcomm Neural Network)",
        },
        ExecutionProviderInfo::Acl => ProviderMetadata {
            id: "acl",
            name: "ACL",
            description: "ACL (Arm Compute Library)",
        },
        ExecutionProviderInfo::ArmNn => ProviderMetadata {
            id: "armnn",
            name: "ArmNN",
            description: "ArmNN (Arm Neural Network)",
        },
        ExecutionProviderInfo::Xnnpack => ProviderMetadata {
            id: "xnnpack",
            name: "XNNPACK",
            description: "XNNPACK (optimized CPU for ARM/x86)",
        },
        _ => ProviderMetadata {
            id: "unknown",
            name: "Unknown",
            description: "Unknown provider",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_metadata_returns_expected_values() {
        // Data-driven test: (provider, expected_id, expected_name, description_keyword)
        let test_cases = [
            (ExecutionProviderInfo::Cpu, "cpu", "CPU", "CPU"),
            (ExecutionProviderInfo::Cuda, "cuda", "CUDA", "NVIDIA"),
            (
                ExecutionProviderInfo::TensorRt,
                "tensorrt",
                "TensorRT",
                "NVIDIA",
            ),
            (
                ExecutionProviderInfo::DirectMl,
                "directml",
                "DirectML",
                "Windows",
            ),
            (ExecutionProviderInfo::CoreMl, "coreml", "CoreML", "Apple"),
            (ExecutionProviderInfo::Rocm, "rocm", "ROCm", "AMD"),
            (
                ExecutionProviderInfo::OpenVino,
                "openvino",
                "OpenVINO",
                "Intel",
            ),
            (ExecutionProviderInfo::OneDnn, "onednn", "oneDNN", "Intel"),
            (ExecutionProviderInfo::Qnn, "qnn", "QNN", "Qualcomm"),
            (ExecutionProviderInfo::Acl, "acl", "ACL", "Arm"),
            (ExecutionProviderInfo::ArmNn, "armnn", "ArmNN", "Arm"),
            (ExecutionProviderInfo::Xnnpack, "xnnpack", "XNNPACK", "CPU"),
        ];

        for (provider, expected_id, expected_name, desc_keyword) in test_cases {
            let meta = provider_metadata(provider);
            assert_eq!(meta.id, expected_id, "ID mismatch for {provider:?}");
            assert_eq!(meta.name, expected_name, "Name mismatch for {provider:?}");
            assert!(
                meta.description.contains(desc_keyword),
                "Description for {provider:?} should contain '{desc_keyword}'"
            );
        }
    }
}
