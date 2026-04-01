#!/usr/bin/env bash
#
# Run Uniclean on all *_without_duplicates lakes in Final_Datasets in parallel.
#
# Resource budget (96 cores / 1.5 TB RAM, 5 parallel lakes):
#   - 16 Spark cores per lake  (5 × 16 = 80, leaving 16 for OS)
#   - 200g driver memory each  (5 × 200g = 1000g, well within 1.5 TB)
#   - 1 hour timeout per table
#
# Usage:
#   bash run_all_final_lakes.sh            # run cleaning + evaluation
#   bash run_all_final_lakes.sh --skip     # evaluation only (skip cleaning)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASETS_DIR="/home/fatemeh/Uniclean-bench-Result/datasets_and_rules/Final_Datasets"
LOG_DIR="${SCRIPT_DIR}/logs/final_lakes_$(date +%Y%m%d_%H%M%S)"

CORES_PER_LAKE=16
MEMORY_PER_LAKE="50g"
TIMEOUT=3600
SINGLE_MAX=10000

SKIP_FLAG=""
if [[ "${1:-}" == "--skip" ]]; then
    SKIP_FLAG="--skip_cleaning"
fi

mkdir -p "$LOG_DIR"

LAKES=(
    # "disjoint_without_duplicates"
    # "flattened_partial_overlap_25_without_duplicates"
    # "flattened_partial_overlap_50_without_duplicates"
    # "flattened_partial_overlap_75_without_duplicates"
    "maximal_overlap_without_duplicates"
)

echo "=========================================="
echo " Launching ${#LAKES[@]} lakes in parallel"
echo " Cores/lake: ${CORES_PER_LAKE}"
echo " Memory/lake: ${MEMORY_PER_LAKE}"
echo " Timeout/table: ${TIMEOUT}s"
echo " Logs: ${LOG_DIR}"
echo "=========================================="

PIDS=()

for lake in "${LAKES[@]}"; do
    lake_dir="${DATASETS_DIR}/${lake}"
    log_file="${LOG_DIR}/${lake}.log"

    if [[ ! -d "$lake_dir" ]]; then
        echo "SKIP: $lake_dir does not exist"
        continue
    fi

    echo "Starting: ${lake} -> ${log_file}"

    python3 "${SCRIPT_DIR}/run_final_lake.py" \
        --lake_dir "$lake_dir" \
        --driver_memory "$MEMORY_PER_LAKE" \
        --spark_master "local[${CORES_PER_LAKE}]" \
        --timeout "$TIMEOUT" \
        --single_max "$SINGLE_MAX" \
        $SKIP_FLAG \
        > "$log_file" 2>&1 &

    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
done

echo ""
echo "All ${#PIDS[@]} lakes launched. Waiting for completion..."
echo "Monitor with:  tail -f ${LOG_DIR}/*.log"
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    lake=${LAKES[$i]}
    if wait "$pid"; then
        echo "DONE: ${lake} (PID ${pid}) — SUCCESS"
    else
        echo "DONE: ${lake} (PID ${pid}) — FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo " All lakes finished. Failed: ${FAILED}/${#LAKES[@]}"
echo " Logs: ${LOG_DIR}"
echo " Results in each lake's uniclean_results/ directory"
echo "=========================================="
