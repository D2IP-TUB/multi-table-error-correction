#!/usr/bin/env bash
# Export ALITE fd_output clusters into per-cluster dirty/clean directories.
#
# Usage:
#   ./scripts/export_alite_merged_tables.sh open_data_uk
#   ./scripts/export_alite_merged_tables.sh mit_dwh --cluster cluster_0007

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$REPO_ROOT/env/bin/python" ]]; then
    PYTHON="${PYTHON:-$REPO_ROOT/env/bin/python}"
else
    PYTHON="${PYTHON:-python3}"
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <open_data_uk|mit_dwh> [--cluster NAME] [--fd-output PATH] [--out-dir PATH]" >&2
    exit 1
fi

exec "$PYTHON" "$REPO_ROOT/codes/export_alite_merged_tables.py" "$@"
