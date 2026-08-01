#!/usr/bin/env bash
# Run the full ALITE pipeline on open_data_uk or mit_dwh.
#
# Usage (from repo root or anywhere):
#   ./scripts/run_alite_corpus.sh mit_dwh
#   ./scripts/run_alite_corpus.sh open_data_uk --skip-viz
#   ./scripts/run_alite_corpus.sh mit_dwh --through discover   # clusters only (no FD)
#
# Steps: index → discover → fd → backfill → viz

set -euo pipefail

CORPORA=(open_data_uk mit_dwh)
STEPS=(index discover fd backfill viz)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CONFIG="$REPO_ROOT/blend_merge_tables/config.py"

CORPUS=""
FROM_STEP="index"
THROUGH_STEP="viz"
SKIP_VIZ=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Run the full ALITE pipeline on open_data_uk or mit_dwh.

Usage (from repo root or anywhere):
  ./scripts/run_alite_corpus.sh mit_dwh
  ./scripts/run_alite_corpus.sh open_data_uk --skip-viz
  ./scripts/run_alite_corpus.sh mit_dwh --through discover

Steps: index → discover → fd → backfill → viz
EOF
    echo
    echo "Options:"
    echo "  --from-step STEP   Start at STEP (${STEPS[*]})"
    echo "  --through STEP     Stop after STEP (e.g. --through discover = clusters only)"
    echo "  --skip-viz         Skip visualization (step viz)"
    echo "  --dry-run          Print commands without running them"
    echo "  -h, --help         Show this help"
    echo
    echo "Environment:"
    echo "  PYTHON             Python interpreter (default: python3)"
}

step_index() {
    local step="$1"
    local i
    for i in "${!STEPS[@]}"; do
        if [[ "${STEPS[$i]}" == "$step" ]]; then
            echo "$i"
            return 0
        fi
    done
    echo "Unknown step: $step (expected one of: ${STEPS[*]})" >&2
    exit 1
}

should_run() {
    local step="$1"
    [[ "$(step_index "$step")" -ge "$(step_index "$FROM_STEP")" ]] &&
    [[ "$(step_index "$step")" -le "$(step_index "$THROUGH_STEP")" ]]
}

run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "+ $*"
    else
        echo ">> $*"
        "$@"
    fi
}

set_corpus_in_config() {
    local corpus="$1"
    run "$PYTHON" - "$CONFIG" "$corpus" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
corpus = sys.argv[2]
text = path.read_text(encoding="utf-8")
new_text, n = re.subn(
    r"^CORPUS = ['\"][^'\"]+['\"]",
    f"CORPUS = '{corpus}'",
    text,
    count=1,
    flags=re.MULTILINE,
)
if n != 1:
    raise SystemExit(f"Could not update CORPUS in {path}")
path.write_text(new_text, encoding="utf-8")
print(f"Set CORPUS = '{corpus}' in {path}")
PY
}

validate_corpus() {
    local corpus="$1"
    local found=0
    for c in "${CORPORA[@]}"; do
        if [[ "$c" == "$corpus" ]]; then
            found=1
            break
        fi
    done
    if [[ "$found" -eq 0 ]]; then
        echo "Invalid corpus: $corpus (expected: ${CORPORA[*]})" >&2
        exit 1
    fi
}

validate_isolated_data() {
    local corpus="$1"
    local isolated
    case "$corpus" in
        open_data_uk) isolated="$REPO_ROOT/datasets/tables/uk_open_data/isolated" ;;
        mit_dwh)      isolated="$REPO_ROOT/datasets/tables/mit_dwh/isolated" ;;
    esac
    if [[ ! -d "$isolated" ]] || [[ -z "$(ls -A "$isolated" 2>/dev/null || true)" ]]; then
        echo "No isolated tables found at: $isolated" >&2
        exit 1
    fi
    echo "Isolated tables: $isolated"
}

# --- parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --from-step)
            FROM_STEP="${2:?--from-step requires an argument}"
            shift 2
            ;;
        --through)
            THROUGH_STEP="${2:?--through requires an argument}"
            shift 2
            ;;
        --skip-viz)
            SKIP_VIZ=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        open_data_uk|mit_dwh)
            if [[ -n "$CORPUS" ]]; then
                echo "Only one corpus may be specified." >&2
                exit 1
            fi
            CORPUS="$1"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$CORPUS" ]]; then
    echo "Missing corpus argument." >&2
    usage >&2
    exit 1
fi

step_index "$FROM_STEP" >/dev/null
step_index "$THROUGH_STEP" >/dev/null
if [[ "$(step_index "$FROM_STEP")" -gt "$(step_index "$THROUGH_STEP")" ]]; then
    echo "--from-step ($FROM_STEP) cannot be after --through ($THROUGH_STEP)" >&2
    exit 1
fi
validate_corpus "$CORPUS"

echo "Repo:       $REPO_ROOT"
echo "Corpus:     $CORPUS"
echo "From step:  $FROM_STEP"
echo "Through:    $THROUGH_STEP"
echo "Python:     $PYTHON"
echo

if should_run index; then
    validate_isolated_data "$CORPUS"
    set_corpus_in_config "$CORPUS"
    run "$PYTHON" "$REPO_ROOT/blend_merge_tables/index_tables.py"
fi

if should_run discover; then
    if ! should_run index; then
        set_corpus_in_config "$CORPUS"
    fi
    run "$PYTHON" "$REPO_ROOT/blend_merge_tables/discover_clusters.py"
fi

if should_run fd; then
    run "$PYTHON" "$REPO_ROOT/codes/alite_fd.py" --corpus "$CORPUS"
fi

if should_run backfill; then
    run "$PYTHON" "$REPO_ROOT/codes/backfill_error_types.py" --corpus "$CORPUS"
fi

if should_run viz && [[ "$SKIP_VIZ" -eq 0 ]]; then
    run "$PYTHON" "$REPO_ROOT/codes/visualize_fd_clusters.py" --corpus "$CORPUS"
fi

echo
if [[ "$THROUGH_STEP" == "discover" ]]; then
    echo "Done. Clusters: $REPO_ROOT/results/alite/$CORPUS/clusters_for_alite/"
else
    echo "Done. Results: $REPO_ROOT/results/alite/$CORPUS/"
fi
