#!/usr/bin/env bash
# Run Horizon for all merged datasets under Real_Lake_Default_Datasets except "tables"
# Saves aggregated and majority-voting-by-error-type results per merged dataset
set -e
cd "$(dirname "$0")"

DATASET_ROOT="/home/fatemeh/data/horizon-code/ablations_withFD/mit_dwh_with_validation"
SKIP_DIR_NAME="tables"
OUTPUT_ROOT="results/horizon_ablation_collections_with_validation"

mkdir -p "$OUTPUT_ROOT"

# Loop through each top-level collection directory, then each merged dataset root.
# Layout A: .../collection/merged/0,1,...  (find directories named "merged")
# Layout B: .../merged_join_*/0,1,...      (no "merged" dir; collection folder is the root)
for subdir in "$DATASET_ROOT"/*/; do
    [[ -d "$subdir" ]] || continue
    subdir_name=$(basename "$subdir")

    if [[ "$subdir_name" == "$SKIP_DIR_NAME" ]]; then
        echo "Skipping dataset: $subdir_name"
        continue
    fi

    mapfile -t merged_dirs < <(find "$subdir" -type d -name merged | sort)
    if [[ ${#merged_dirs[@]} -eq 0 ]]; then
        merged_dirs=("${subdir%/}")
    fi

    found_any=0
    for merged_dir in "${merged_dirs[@]}"; do
        found_any=1
        rel_path="${merged_dir#${DATASET_ROOT}/}"
        rel_path="${rel_path%/}"
        dataset_tag="${rel_path//\//_}"
        dataset_out_dir="$OUTPUT_ROOT/$dataset_tag"
        standard_out_dir="$dataset_out_dir/standard"
        by_type_out_dir="$dataset_out_dir/majority_voting_by_error_type"
        mkdir -p "$standard_out_dir" "$by_type_out_dir"

        echo "============================================================"
        echo "Processing merged dataset: $rel_path"
        echo "============================================================"

        # Auto-select mode:
        # - run+eval when fds.txt exists
        # - eval-only when repaired files exist but fds.txt is absent
        fds_count=$(find "$merged_dir" -type f -name fds.txt | wc -l)
        repaired_count=$(find "$merged_dir" -type f -name clean.csv.a2.clean | wc -l)

        if [[ "$fds_count" -eq 0 && "$repaired_count" -gt 0 ]]; then
            echo "Mode: eval-only (no fds.txt found; using existing repaired files)"
            python run_horizon_all.py "$merged_dir" \
                --skip_horizon \
                --results_file "$standard_out_dir/horizon_results.json"
        else
            echo "Mode: run+eval"
            python run_horizon_all.py "$merged_dir" \
                --results_file "$standard_out_dir/horizon_results.json"
        fi

        # Move standard aggregate outputs into organized directories
        if [[ -f "horizon_eval_aggregate.csv" ]]; then
            mv "horizon_eval_aggregate.csv" "$standard_out_dir/horizon_aggregate.csv"
        fi

        # Run per-error-type majority-voting evaluation (includes OVERALL row)
        python evaluate_horizon_majority_voting_by_error_type.py \
            "$merged_dir" \
            "$merged_dir" \
            --csv "$by_type_out_dir/horizon_aggregate_majority_voting_by_error_type.csv" \
            --json "$by_type_out_dir/horizon_results_majority_voting_by_error_type.json"

        echo "Saved outputs to: $dataset_out_dir"

        echo ""
    done

    if [[ "$found_any" -eq 0 ]]; then
        echo "No merged datasets found under: $subdir_name"
        echo ""
    fi
done

echo "Done processing all datasets in $DATASET_ROOT (excluding $SKIP_DIR_NAME)."
