#!/usr/bin/env bash
# Create or recreate the local UniClean conda environment.
set -euo pipefail

UNICLEAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="${UNICLEAN_ROOT}/.venv"
REQ_FILE="${UNICLEAN_ROOT}/uniclean_cleaners/requirements.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "Missing requirements file: ${REQ_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ -d "${ENV_PATH}" ]]; then
  echo "Environment already exists at ${ENV_PATH}"
else
  echo "Creating conda env at ${ENV_PATH} (Python 3.10)..."
  conda create -y -p "${ENV_PATH}" python=3.10 pip
fi

echo "Installing OpenJDK 11 (required by PySpark 3.1)..."
conda install -y -p "${ENV_PATH}" openjdk=11

echo "Installing Python dependencies..."
conda run -p "${ENV_PATH}" pip install -r "${REQ_FILE}"

echo ""
echo "Done. Activate with:"
echo "  source ${UNICLEAN_ROOT}/activate_uniclean.sh"
