//! Build script for birda.
//!
//! Embeds library version expectations at compile time so the update command
//! can detect ABI-breaking changes without a local manifest file.

fn main() {
    // ONNX Runtime version this binary was built against.
    // Set by the release workflow; defaults to "unknown" for dev builds.
    let ort_version =
        std::env::var("ONNXRUNTIME_VERSION").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=BIRDA_ONNXRUNTIME_VERSION={ort_version}");

    // CUDA toolkit version expected by this build.
    let cuda_version =
        std::env::var("CUDA_TOOLKIT_VERSION").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=BIRDA_CUDA_TOOLKIT_VERSION={cuda_version}");

    // cuDNN version expected by this build.
    let cudnn_version = std::env::var("CUDNN_VERSION").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=BIRDA_CUDNN_VERSION={cudnn_version}");

    // Re-run if these env vars change.
    println!("cargo:rerun-if-env-changed=ONNXRUNTIME_VERSION");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_VERSION");
    println!("cargo:rerun-if-env-changed=CUDNN_VERSION");
}
