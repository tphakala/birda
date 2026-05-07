# Design: `birda update` Command

## Overview

Add a self-update command to birda that checks for new releases on GitHub and replaces the binary in-place. Only the binary is updated; CUDA and ONNX Runtime libraries are not modified. Helpful warnings are printed when library version requirements change between releases.

## Constraints

- Stable releases only (no pre-release channel)
- Binary-only update (not CUDA libs or ONNX Runtime .so/.dll)
- Block update if ONNX Runtime version changed (ABI break risk)
- Warn if CUDA/cuDNN version changed (external dependency)
- Keep previous binary as `birda.old` for rollback (Linux/macOS only; not available on Windows)
- SHA256 checksum verification on all downloads
- No GitHub API usage; use direct download URLs to avoid rate limits

## Manifest Enhancement

The release `manifest.json` is extended with a `bin` section containing binary-only archives and SHA256 checksums for each platform/variant.

```json
{
  "version": "${VERSION}",
  "min_gui_version": "1.1.0",
  "assets": {
    "bin": {
      "linux-x64": {
        "file": "birda-linux-x64-bin-${TAG}.tar.gz",
        "sha256": "${SHA256_LINUX_BIN}"
      },
      "linux-x64-cuda": {
        "file": "birda-linux-x64-cuda-bin-${TAG}.tar.gz",
        "sha256": "${SHA256_LINUX_CUDA_BIN}"
      },
      "windows-x64": {
        "file": "birda-windows-x64-bin-${TAG}.zip",
        "sha256": "${SHA256_WINDOWS_BIN}"
      },
      "windows-x64-cuda": {
        "file": "birda-windows-x64-cuda-bin-${TAG}.zip",
        "sha256": "${SHA256_WINDOWS_CUDA_BIN}"
      },
      "macos-arm64": {
        "file": "birda-macos-arm64-bin-${TAG}.tar.gz",
        "sha256": "${SHA256_MACOS_BIN}"
      }
    },
    "embed": {
      "linux-x64": "birda-linux-x64-embed-${TAG}.tar.gz",
      "windows-x64": "birda-windows-x64-embed-${TAG}.zip",
      "macos-arm64": "birda-macos-arm64-embed-${TAG}.tar.gz"
    },
    "cuda_libs": {
      "linux-x64": "birda-cuda-libs-linux-x64-${TAG}.tar.gz",
      "windows-x64": "birda-cuda-libs-windows-x64-${TAG}.zip"
    }
  },
  "dependencies": {
    "onnxruntime": "${ONNXRUNTIME_VERSION}"
  },
  "cuda": {
    "cuda_toolkit": "${CUDA_TOOLKIT_VERSION}",
    "cudnn": "${CUDNN_VERSION}"
  }
}
```

Existing `embed` and `cuda_libs` sections are unchanged for backward compatibility with birda-gui.

## Build-time Version Embedding

A `build.rs` bakes library version expectations into the binary at compile time.

```rust
// build.rs
fn main() {
    let ort_version = std::env::var("ONNXRUNTIME_VERSION")
        .unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=BIRDA_ONNXRUNTIME_VERSION={ort_version}");

    let cuda_version = std::env::var("CUDA_TOOLKIT_VERSION")
        .unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=BIRDA_CUDA_TOOLKIT_VERSION={cuda_version}");

    let cudnn_version = std::env::var("CUDNN_VERSION")
        .unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=BIRDA_CUDNN_VERSION={cudnn_version}");

    println!("cargo:rerun-if-env-changed=ONNXRUNTIME_VERSION");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_VERSION");
    println!("cargo:rerun-if-env-changed=CUDNN_VERSION");
}
```

These env vars are already defined at the top of `release.yml` and are in scope during builds. No workflow changes needed for the build step.

Accessed in code via:
- `env!("BIRDA_ONNXRUNTIME_VERSION")`
- `env!("BIRDA_CUDA_TOOLKIT_VERSION")`
- `env!("BIRDA_CUDNN_VERSION")`

Variant detection uses the existing cargo feature: `cfg!(feature = "cuda")`. No extra mechanism needed since CPU and embed binaries are identical.

## CLI Interface

New subcommand:

```rust
/// Check for and install updates.
Update {
    /// Only check for updates, don't install.
    #[arg(long)]
    check: bool,
}
```

- `birda update` -- check + install if newer version available
- `birda update --check` -- print version info only, no changes

### Output Examples

**Up to date:**
```
birda is up to date (v1.8.0)
```

**Update available (--check):**
```
Update available: v1.8.0 -> v1.9.0
Run 'birda update' to install.
```

**Successful update:**
```
Downloading birda v1.9.0...
[progress bar] 100% (14.2MB/14.2MB)
Verifying checksum... ok
Previous version saved as /usr/local/bin/birda.old
Updated birda v1.8.0 -> v1.9.0
```

**Blocked -- ONNX Runtime version changed:**
```
Update available: v1.8.0 -> v2.0.0

WARNING: This release requires updated libraries:
  ONNX Runtime: 1.24.2 -> 1.25.0

Binary-only update would leave birda non-functional.
Please download the full package from:
  https://github.com/tphakala/birda/releases/tag/v2.0.0
```

**CUDA version changed (proceed with warning):**
```
Downloading birda v1.9.0...
[progress bar] 100% (14.2MB/14.2MB)
Verifying checksum... ok
Previous version saved as /usr/local/bin/birda.old
Updated birda v1.8.0 -> v1.9.0

Note: CUDA toolkit requirement changed (12.9 -> 13.0).
If you use GPU acceleration, you may need to update your CUDA installation.
```

## Update Flow

```
birda update [--check]
  |
  +- 1. Guard: if running from a cargo target/ directory, refuse to update
  |     (prevents overwriting dev builds with release binaries)
  |
  +- 2. Fetch manifest.json from GitHub latest release
  |     URL: https://github.com/tphakala/birda/releases/latest/download/manifest.json
  |
  +- 3. Parse manifest, compare manifest.version vs CARGO_PKG_VERSION (semver)
  |     If current >= remote: "up to date", exit
  |
  +- 4. If --check: print update info, exit
  |
  +- 5. Compare library versions:
  |     - manifest.dependencies.onnxruntime vs env!("BIRDA_ONNXRUNTIME_VERSION")
  |       If major.minor changed: block update, print full-package download URL
  |       (applies to ALL build variants -- load-dynamic is always used)
  |     - manifest.cuda.cuda_toolkit vs env!("BIRDA_CUDA_TOOLKIT_VERSION")
  |       If changed AND cuda build: warn
  |     - manifest.cuda.cudnn vs env!("BIRDA_CUDNN_VERSION")
  |       If changed AND cuda build: warn
  |     - If env versions are "unknown" (dev build): skip all checks
  |
  +- 6. Select asset key based on:
  |     - target_os: linux / windows / macos
  |     - target_arch: x86_64 / aarch64
  |     - feature: cuda or not
  |     Example: "linux-x64-cuda" or "macos-arm64"
  |
  +- 7. Check write permissions on parent directory of current binary
  |     (rename requires directory write, not file write)
  |     No permission: abort with instructions (e.g. run as admin/sudo)
  |
  +- 8. Download binary-only archive
  |     URL: https://github.com/tphakala/birda/releases/latest/download/{asset.file}
  |     Reuse existing download_file() with progress bar
  |
  +- 9. Verify SHA256 checksum
  |     Mismatch: abort, delete temp file
  |
  +- 10. Extract archive into same directory as current binary
  |      (avoids EXDEV cross-device rename failures)
  |      Extract as .birda-update-new.tmp
  |      .tar.gz: flate2 + tar
  |      .zip: zip crate
  |      Archives are flat (contain only the binary, no directory nesting)
  |
  +- 11. Set executable permissions (Unix only)
  |      chmod 0o755 on .birda-update-new.tmp
  |
  +- 12. Self-replace:
  |      Linux/macOS:
  |        rename current -> birda.old
  |        rename .birda-update-new.tmp -> current path
  |        If second rename fails: restore birda.old -> current (rollback)
  |      Windows:
  |        use self_replace crate (handles locked binary)
  |        Note: rollback via birda.old is not available on Windows
  |              (self_replace renames original to random temp name)
  |
  +- 13. Print result + any library version warnings
  |      Linux/macOS: "Previous version saved as birda.old"
  |      Windows: omit rollback message
```

### Library Version Comparison Logic

The ONNX Runtime check applies to **all build variants** (CPU, CUDA, embed) because all use `load-dynamic` for the `ort` crate. An ABI-breaking ONNX Runtime change will crash any variant. The check compares **major.minor** only (patch changes are ABI-compatible). If the running binary reports "unknown" (dev build), skip all library checks and proceed.

CUDA/cuDNN checks only apply when `cfg!(feature = "cuda")` is true. For non-CUDA builds these warnings are suppressed.

## New Dependencies

| Crate | Purpose |
|-------|---------|
| `semver` | Version parsing and comparison |
| `sha2` | SHA256 checksum verification |
| `flate2` | gzip decompression |
| `tar` | tar archive extraction |
| `zip` | zip archive extraction (Windows) |
| `self_replace` | Windows binary replacement |

## Module Structure

```
src/
  update/
    mod.rs         -- public API: check_for_update(), perform_update()
    manifest.rs    -- Manifest struct, fetch from GitHub, parse JSON
    platform.rs    -- asset key selection based on target_os/target_arch/features
    replace.rs     -- self-replacement logic (platform-specific)
    checksum.rs    -- SHA256 verification
```

## Release Workflow Changes

1. **Binary-only archives** -- after existing packaging in the release job, create small archives containing just the birda binary for each platform/variant.

2. **SHA256 computation** -- compute hashes of binary-only archives and substitute into the manifest template.

3. **Manifest template** -- update `manifest.template.json` with the new `bin` section.

No changes to existing packages (embed, cuda, cuda_libs). Backward compatible.

## Error Handling

All errors use the existing `Error` enum in `src/error.rs` with new variants:

- `UpdateFetchFailed` -- failed to download manifest or binary
- `UpdateChecksumMismatch` -- SHA256 doesn't match
- `UpdateReplaceFailed` -- binary replacement failed
- `UpdateBlocked` -- ONNX Runtime version mismatch
- `UpdatePermissionDenied` -- no write access to binary path

## Platform-specific Considerations

### Unix (Linux/macOS)
- Set executable permissions (0o755) after extraction
- Rename-based replacement avoids EXDEV by extracting into same directory
- Rollback: if second rename fails, restore `.old` back to original path
- `birda.old` kept for manual rollback

### Windows
- Use `self_replace` crate to handle locked-binary replacement
- No `.old` rollback file available (self_replace uses OS temp cleanup)
- Adjust success messaging to omit rollback instructions

### Dev Build Guard
- If `std::env::current_exe()` resolves to a path containing `/target/` (or `\target\`), refuse to update. This prevents accidentally overwriting a locally-compiled binary with a release build.

## Security

- SHA256 checksum verification on all downloaded archives
- HTTPS-only downloads (reqwest with rustls)
- Permission checks on parent directory before modifying files
- Temp files cleaned up on failure (both on checksum mismatch and extraction errors)
- Archives must be flat (no path traversal in archive entries)
