#!/usr/bin/env bash
# Activate the local UniClean conda environment (Python 3.10 + OpenJDK 11 for PySpark 3.1).
#
# Usage:
#   source baselines/UniClean/activate_uniclean.sh
#
# Then run Quintet experiments, e.g.:
#   cd uniclean_cleaners
#   python run_quintet3.py --lake_dir ../../../datasets/unrelated_tables/Quintet

_UNICLEAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_UNICLEAN_ENV="${_UNICLEAN_ROOT}/.venv"

if [[ ! -d "${_UNICLEAN_ENV}" ]]; then
  echo "UniClean env not found at ${_UNICLEAN_ENV}" >&2
  echo "Create it with: bash ${_UNICLEAN_ROOT}/setup_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${_UNICLEAN_ENV}"

export JAVA_HOME="${_UNICLEAN_ENV}"
export PATH="${JAVA_HOME}/bin:${PATH}"

echo "UniClean env active: Python $(python --version 2>&1 | cut -d' ' -f2), Java $(java -version 2>&1 | head -1)"
