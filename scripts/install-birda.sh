#!/usr/bin/env bash
set -euo pipefail

# Validate platform - this action only supports Linux x86_64
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "::error::This action only supports Linux x86_64 runners (ubuntu-latest)"
    exit 1
fi

# Version comes from github.action_ref (the tag used to reference the action)
# e.g., "v1.3.0" when used as tphakala/birda@v1.3.0
VERSION="${INPUT_VERSION:-}"

# Handle non-tag refs (main branch, PR refs, etc.) by falling back to latest
if [[ -z "$VERSION" || "$VERSION" == "main" || "$VERSION" == "master" || "$VERSION" =~ ^refs/ ]]; then
    echo "Resolving latest release version..."
    VERSION=$(curl -fsSL https://api.github.com/repos/tphakala/birda/releases/latest | jq -r '.tag_name')
    if [[ -z "$VERSION" ]]; then
        echo "::error::Failed to resolve latest version from GitHub API"
        exit 1
    fi
fi

echo "Installing Birda ${VERSION}..."

# Construct download URL (CPU-only Linux x86_64 build)
URL="https://github.com/tphakala/birda/releases/download/${VERSION}/birda-linux-x64-${VERSION}.tar.gz"

# Setup install directory
INSTALL_DIR="${RUNNER_TEMP}/birda"
mkdir -p "${INSTALL_DIR}"

# Download and extract
echo "Downloading from ${URL}..."
if ! curl -fsSL "${URL}" | tar -xz -C "${INSTALL_DIR}"; then
    echo "::error::Failed to download birda ${VERSION}. Check that the release exists."
    exit 1
fi

# Add to PATH for subsequent steps
echo "${INSTALL_DIR}" >> "${GITHUB_PATH}"

# Export library path for downstream steps (used locally, not globally)
echo "BIRDA_LIB_PATH=${INSTALL_DIR}" >> "${GITHUB_ENV}"

# Verify installation (use local LD_LIBRARY_PATH to avoid global pollution)
LD_LIBRARY_PATH="${INSTALL_DIR}:${LD_LIBRARY_PATH:-}" "${INSTALL_DIR}/birda" --version

echo "Birda ${VERSION} installed successfully"
