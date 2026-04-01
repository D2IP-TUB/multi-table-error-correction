#!/usr/bin/env python3
from __future__ import annotations
"""
Lake-level evaluation script for HoloClean repairs.

Discovers all repaired CSV files under an output directory, pairs each one
with the corresponding dirty/clean data from the input data lake, runs the
per-table evaluation (using utils.py), and aggregates results across all
tables into a lake-wide summary.

Aggregation strategy (mirrors utils.aggregate_lake_results):
  1. Total errors in the lake = sum of n_all_errors counted once per table
     (errors are constant across iterations).
  2. For each iteration, sum TP and corrections-made across all tables, then
     compute precision / recall / F1 for that iteration.
  3. Average the per-iteration metrics to get the final lake score.

Usage:
    python evaluate_lake.py \
        --output-dir  outputs/2026-02-20/08-10-15/dcHoloCleaner-with_init/HoloClean \
        --input-dir   datasets-holo/open_data_uk_filtered_final \
        [--save-csv   results/lake_evaluation.csv]
"""

import argparse
import os
import re
import sys
import logging
import traceback

import pandas as pd

# Make sure utils and evaluate_repair are importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import read_csv, get_dataframes_difference, evaluate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Pattern: repaired_holoclean_<table>_seed<seed>_iter<iter>.csv
_FILENAME_RE = re.compile(
    r"^repaired_holoclean_(?P<table>.+?)_seed(?P<seed>\d+)_iter(?P<iter>\d+)\.csv$"
)


def discover_repaired_files(output_dir: str) -> list[dict]:
    """Return a list of dicts with keys: path, table, seed, iter."""
    entries = []
    for fname in sorted(os.listdir(output_dir)):
        m = _FILENAME_RE.match(fname)
        if m:
            entries.append(
                {
                    "path": os.path.join(output_dir, fname),
                    "table": m.group("table"),
                    "seed": int(m.group("seed")),
                    "iter": int(m.group("iter")),
                }
            )
    return entries


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def _sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '::' and other problematic characters in column names."""
    df.columns = [re.sub(r"::", "_", col) for col in df.columns]
    return df


def evaluate_single(
    dirty_path: str,
    clean_path: str,
    repaired_path: str,
    table_name: str,
) -> dict | None:
    """
    Evaluate one repaired file.  Returns a dict of metrics, or None on error.
    """
    try:
        dirty_df = read_csv(dirty_path, data_type="str")
        clean_df = read_csv(clean_path, data_type="str")
        repaired_df = read_csv(repaired_path, data_type="str")

        # Normalise column names
        clean_df.columns = ["index_col" if c == "index" else c for c in clean_df.columns]
        dirty_df.columns = clean_df.columns
        dirty_df = _sanitize_column_names(dirty_df)
        clean_df = _sanitize_column_names(clean_df)

        # Drop HoloClean's internal _tid_ column if present
        if "_tid_" in repaired_df.columns:
            repaired_df = repaired_df.drop(columns=["_tid_"])
        repaired_df = _sanitize_column_names(repaired_df)

        # Shape guard: repaired_df must match dirty_df exactly
        if dirty_df.shape != repaired_df.shape:
            logger.warning(
                "[%s] Shape mismatch: dirty %s vs repaired %s – skipping",
                table_name,
                dirty_df.shape,
                repaired_df.shape,
            )
            return None

        detections = get_dataframes_difference(dirty_df, clean_df)
        if len(detections) == 0:
            logger.warning("[%s] No errors detected (dirty == clean) – skipping", table_name)
            return None

        results = evaluate(detections, dirty_df, clean_df, repaired_df)
        return results

    except Exception as exc:
        logger.warning("[%s] Evaluation failed: %s", table_name, exc)
        logger.debug(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Main aggregation logic
# ---------------------------------------------------------------------------

def count_lake_errors(input_dir: str) -> int:
    """
    Count total errors across ALL tables in the input lake (dirty vs clean),
    regardless of whether a repaired file exists.  This is the correct
    denominator for lake-wide recall.
    """
    total = 0
    for table_name in sorted(os.listdir(input_dir)):
        table_dir = os.path.join(input_dir, table_name)
        if not os.path.isdir(table_dir):
            continue
        dirty_path = os.path.join(table_dir, "dirty.csv")
        clean_path = os.path.join(table_dir, "clean.csv")
        if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
            continue
        try:
            dirty_df = _sanitize_column_names(read_csv(dirty_path, data_type="str"))
            clean_df = _sanitize_column_names(read_csv(clean_path, data_type="str"))
            clean_df.columns = ["index_col" if c == "index" else c for c in clean_df.columns]
            dirty_df.columns = clean_df.columns
            n_errors = len(get_dataframes_difference(dirty_df, clean_df))
            total += n_errors
        except Exception as exc:
            logger.warning("[%s] Could not count errors for lake total: %s", table_name, exc)
    return total


def run_lake_evaluation(output_dir: str, input_dir: str, save_csv: str | None = None):
    entries = discover_repaired_files(output_dir)
    if not entries:
        logger.error("No repaired files found in %s", output_dir)
        sys.exit(1)

    logger.info("Found %d repaired file(s) across the lake.", len(entries))

    # Count total errors across the entire lake (all tables, not just processed ones)
    logger.info("Counting total errors across all tables in the lake...")
    total_errors_lake = count_lake_errors(input_dir)
    logger.info("Total errors in lake (all tables): %d", total_errors_lake)

    # Group by table name
    tables: dict[str, list[dict]] = {}
    for e in entries:
        tables.setdefault(e["table"], []).append(e)

    logger.info("Covering %d unique table(s).", len(tables))

    per_table_rows = []   # one row per (table, iter) – raw metrics
    skipped_tables = []

    # iter_accumulators: iter_num -> {"tp": int, "corrected": int}
    iter_accumulators: dict[int, dict] = {}
    tables_evaluated = 0
    iters_evaluated = 0

    print("\n" + "=" * 80)
    print(f"{'TABLE':<30} {'ITER':>4} {'ERRORS':>8} {'CORRECTED':>10} "
          f"{'CORRECT':>9} {'PREC':>8} {'REC':>8} {'F1':>8}")
    print("=" * 80)

    for table_name in sorted(tables.keys()):
        table_dir = os.path.join(input_dir, table_name)
        # If the folder doesn't exist, try stripping the SQL-safe 't_' prefix
        # (run_baselines.py adds 't_' to numeric folder names for SQL safety)
        if not os.path.isdir(table_dir) and table_name.startswith("t_"):
            table_dir = os.path.join(input_dir, table_name[2:])
        dirty_path = os.path.join(table_dir, "dirty.csv")
        clean_path = os.path.join(table_dir, "clean.csv")

        if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
            logger.warning("[%s] dirty.csv or clean.csv not found – skipping", table_name)
            skipped_tables.append(table_name)
            continue

        table_iter_results = []
        for entry in sorted(tables[table_name], key=lambda x: x["iter"]):
            res = evaluate_single(dirty_path, clean_path, entry["path"], table_name)
            if res is None:
                skipped_tables.append(f"{table_name}_iter{entry['iter']}")
                print(f"  {table_name:<28} iter{entry['iter']:>2}  SKIPPED")
                continue

            table_iter_results.append(res)
            iters_evaluated += 1

            per_table_rows.append(
                {
                    "table": table_name,
                    "seed": entry["seed"],
                    "iter": entry["iter"],
                    **res,
                }
            )

            it = entry["iter"]
            if it not in iter_accumulators:
                iter_accumulators[it] = {"tp": 0, "corrected": 0}
            iter_accumulators[it]["tp"]        += res["n_truely_corrected_errors"]
            iter_accumulators[it]["corrected"] += res["n_all_corrected_errors"]

            print(
                f"  {table_name:<28} {entry['iter']:>4}  "
                f"{res['n_all_errors']:>8}  {res['n_all_corrected_errors']:>8}  "
                f"{res['n_truely_corrected_errors']:>8}  "
                f"{_fmt(res['precision']):>8}  "
                f"{_fmt(res['recall']):>8}  "
                f"{_fmt(res['f1_score']):>8}"
            )

        if not table_iter_results:
            skipped_tables.append(table_name)
            continue

        tables_evaluated += 1

    # -----------------------------------------------------------------------
    # Lake-wide aggregation (same logic as utils.aggregate_lake_results)
    # -----------------------------------------------------------------------
    # total_errors_lake already computed upfront via count_lake_errors()

    # Per-iteration lake metrics
    iter_precisions, iter_recalls, iter_f1s = [], [], []
    for it in sorted(iter_accumulators.keys()):
        acc = iter_accumulators[it]
        tp_it  = acc["tp"]
        cor_it = acc["corrected"]
        prec_it = tp_it / cor_it if cor_it > 0 else -1.0
        rec_it  = tp_it / total_errors_lake if total_errors_lake > 0 else -1.0
        f1_it   = (
            2 * prec_it * rec_it / (prec_it + rec_it)
            if (prec_it + rec_it) > 0 else -1.0
        )
        iter_precisions.append(prec_it)
        iter_recalls.append(rec_it)
        iter_f1s.append(f1_it)

    def _mean_valid(vals):
        v = [x for x in vals if x >= 0]
        return sum(v) / len(v) if v else -1.0

    avg_precision = _mean_valid(iter_precisions)
    avg_recall    = _mean_valid(iter_recalls)
    avg_f1        = _mean_valid(iter_f1s)
    std_f1        = (
        (sum((x - avg_f1) ** 2 for x in iter_f1s if x >= 0) / len([x for x in iter_f1s if x >= 0])) ** 0.5
        if len([x for x in iter_f1s if x >= 0]) > 1 else 0.0
    )

    print("\n" + "=" * 80)
    print("LAKE-WIDE SUMMARY")
    print("=" * 80)
    print(f"  Tables evaluated:       {tables_evaluated}")
    print(f"  Iterations evaluated:   {iters_evaluated}")
    print(f"  Skipped:                {len(skipped_tables)}")
    print(f"  Total errors in lake:   {total_errors_lake}")
    print()
    if len(iter_precisions) > 1:
        print("  Per-iteration lake metrics:")
        for i, (p, r, f) in enumerate(zip(iter_precisions, iter_recalls, iter_f1s)):
            print(f"    iter {i}: Precision={_fmt(p)}  Recall={_fmt(r)}  F1={_fmt(f)}")
        print()
    print(f"  Precision: {_fmt(avg_precision)}")
    print(f"  Recall:    {_fmt(avg_recall)}")
    print(f"  F1:        {_fmt(avg_f1)}")
    if len(iter_precisions) > 1:
        print(f"  F1 std:    {_fmt(std_f1)}")

    if skipped_tables:
        print(f"\n  Skipped: {skipped_tables}")

    # Optionally save results to CSV
    if save_csv and per_table_rows:
        # Per-iteration detail file
        detail_df = pd.DataFrame(per_table_rows)
        detail_df.to_csv(save_csv, index=False)
        logger.info("Per-iteration results saved to: %s", save_csv)

        # Per-table summary file (same path with _summary suffix)
        base, ext = os.path.splitext(save_csv)
        summary_path = f"{base}_summary{ext}"
        summary_df = pd.DataFrame(per_table_rows)
        # Add aggregate row
        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    [
                        {
                            "table": "*** LAKE TOTAL ***",
                            "n_iters": iters_evaluated,
                            "n_all_errors": total_errors_lake,
                            "precision": avg_precision,
                            "recall": avg_recall,
                            "f1_score": avg_f1,
                            "f1_score_std": std_f1,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        summary_df.to_csv(summary_path, index=False)
        logger.info("Per-table summary saved to:    %s", summary_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.4f}" if v >= 0 else "  N/A "


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HoloClean repairs across an entire data lake."
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory containing repaired_holoclean_*.csv files.",
    )
    parser.add_argument(
        "--input-dir", "-i", required=True,
        help="Root of the input data lake (contains table_XX/ sub-directories).",
    )
    parser.add_argument(
        "--save-csv", "-s", default=None,
        help="Optional path to save per-iteration CSV results (summary CSV is auto-named).",
    )
    args = parser.parse_args()

    run_lake_evaluation(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
