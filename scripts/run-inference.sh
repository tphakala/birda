#!/usr/bin/env bash
set -euo pipefail

# Validate required inputs
if [[ -z "${INPUT_MODEL:-}" ]]; then
    echo "::error::Input 'model' is required"
    exit 1
fi

if [[ -z "${INPUT_MODEL_TYPE:-}" ]]; then
    echo "::error::Input 'model-type' is required (birdnet-v24, birdnet-v30, or perch-v2)"
    exit 1
fi

if [[ -z "${INPUT_AUDIO:-}" ]]; then
    echo "::error::Input 'audio' is required"
    exit 1
fi

# Validate input files exist
if [[ ! -f "${INPUT_MODEL}" ]]; then
    echo "::error::Model file not found: ${INPUT_MODEL}"
    exit 1
fi

if [[ ! -f "${INPUT_AUDIO}" ]]; then
    echo "::error::Audio file not found: ${INPUT_AUDIO}"
    exit 1
fi

# Use temp directory for raw output to avoid naming conflicts
# Clean any stale results from previous runs in same job
TEMP_OUT="${RUNNER_TEMP}/birda-output"
rm -rf "${TEMP_OUT}"
mkdir -p "${TEMP_OUT}"

# Build command arguments (ad-hoc model: requires both --model-type and --model-path)
ARGS=(
    "${INPUT_AUDIO}"
    "--model-type" "${INPUT_MODEL_TYPE}"
    "--model-path" "${INPUT_MODEL}"
    "--min-confidence" "${INPUT_CONFIDENCE}"
    "--format" "${INPUT_FORMAT}"
    "--output-dir" "${TEMP_OUT}"
)

# Add optional labels path
if [[ -n "${INPUT_LABELS:-}" ]]; then
    if [[ ! -f "${INPUT_LABELS}" ]]; then
        echo "::error::Labels file not found: ${INPUT_LABELS}"
        exit 1
    fi
    ARGS+=("--labels-path" "${INPUT_LABELS}")
fi

# Run birda (set LD_LIBRARY_PATH locally to avoid polluting global environment)
echo "Running: birda ${ARGS[*]}"
LD_LIBRARY_PATH="${BIRDA_LIB_PATH}:${LD_LIBRARY_PATH:-}" birda "${ARGS[@]}"

# Find the generated output file
# Birda generates files like: {input_stem}.BirdNET.results.{format}
GENERATED_FILE=$(find "${TEMP_OUT}" -type f -name "*.BirdNET.results.*" | head -n 1)

if [[ -z "${GENERATED_FILE}" ]]; then
    echo "::error::No output file was generated"
    exit 1
fi

echo "Generated file: ${GENERATED_FILE}"

# Handle output file placement
if [[ -n "${INPUT_OUTPUT:-}" ]]; then
    # Validate output path - prevent path traversal attacks
    if [[ "${INPUT_OUTPUT}" == /* || "${INPUT_OUTPUT}" == *..* ]]; then
        echo "::error::Invalid output path (absolute paths and '..' not allowed): ${INPUT_OUTPUT}"
        exit 1
    fi
    # User specified output path - ensure parent directory exists
    # Handle trailing-slash paths as directories (dirname "results/" returns ".")
    if [[ "${INPUT_OUTPUT}" == */ ]]; then
        mkdir -p "${INPUT_OUTPUT}"
    else
        OUTPUT_DIR=$(dirname "${INPUT_OUTPUT}")
        if [[ -n "${OUTPUT_DIR}" && "${OUTPUT_DIR}" != "." ]]; then
            mkdir -p "${OUTPUT_DIR}"
        fi
    fi
    # Move/rename to requested location
    mv "${GENERATED_FILE}" "${INPUT_OUTPUT}"
    # Handle case where INPUT_OUTPUT is a directory
    if [[ -d "${INPUT_OUTPUT}" ]]; then
        FINAL_OUTPUT="$(cd "${INPUT_OUTPUT}" && pwd)/$(basename "${GENERATED_FILE}")"
    else
        FINAL_OUTPUT="$(cd "$(dirname "${INPUT_OUTPUT}")" && pwd)/$(basename "${INPUT_OUTPUT}")"
    fi
else
    # No output specified - move to current directory with original name
    FILENAME=$(basename "${GENERATED_FILE}")
    mv "${GENERATED_FILE}" "./${FILENAME}"
    FINAL_OUTPUT="$(pwd)/${FILENAME}"
fi

echo "Output file: ${FINAL_OUTPUT}"

# Set output for subsequent steps (use heredoc to prevent injection)
{
    echo "results<<EOF"
    echo "${FINAL_OUTPUT}"
    echo "EOF"
} >> "${GITHUB_OUTPUT}"
