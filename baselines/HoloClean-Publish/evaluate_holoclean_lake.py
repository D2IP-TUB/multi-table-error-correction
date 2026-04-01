#!/usr/bin/env python3
"""
Re-evaluate HoloClean results lake-wide using error_map.csv files from the
isolated sandbox directory.

Aggregation strategy (mirrors evaluate_lake.py):
  1. For each iteration, SUM tp and corrections-attempted across all tables.
  2. Compute precision / recall / F1 from those sums (not by averaging per-table metrics).
  3. Use lake-wide ec_tpfn (ALL tables, including skipped ones) as the recall denominator.
  4. Average the per-iteration P/R/F1 to get the final lake score.

The same logic applies per error type.

Usage:
    python evaluate_holoclean_lake.py \
        --results-dir  outputs/2026-03-02/16-24-03/dcHoloCleaner-with_init/HoloClean \
        --isolated-dir datasets/tables/uk_open_data/isolated \
        --name-dict    open_data_93_res_dict_no.pkl \
        [--output-csv  results/holoclean_open_data_uk_93_with_types.csv]
"""

import argparse
import json
import os
import pickle
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import read_csv, get_dataframes_difference
from evaluate_repair import load_error_map, evaluate_per_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v):
    return f"{v:.4f}" if v >= 0 else "  N/A "


def _mean_valid(vals):
    v = [x for x in vals if x >= 0]
    return sum(v) / len(v) if v else -1.0


def _std_valid(vals, mean):
    v = [x for x in vals if x >= 0]
    if len(v) < 2:
        return 0.0
    return (sum((x - mean) ** 2 for x in v) / len(v)) ** 0.5


def build_table_to_uk_map(name_dict_path):
    """Return {table_N: UK_CSV_basename} from the pickle dict."""
    with open(name_dict_path, "rb") as f:
        d = pickle.load(f)
    return {
        v.rstrip("/").split("/")[-1]: k.rstrip("/").split("/")[-1]
        for k, v in d.items()
    }


def parse_table_name(filename):
    m = re.search(r"(table_\d+)", filename)
    return m.group(1) if m else None


def parse_iter(filename):
    m = re.search(r"iter(\d+)", filename)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Lake-wide ec_tpfn (covers ALL tables, not just evaluated ones)
# ---------------------------------------------------------------------------

def get_lake_ec_tpfn(isolated_dir, dirty_file="dirty.csv", clean_file="clean.csv"):
    """
    Count actual errors (dirty != clean, raw strings) per error_type across
    every table in isolated_dir.  Used as the recall denominator.
    Returns (ec_tpfn_by_type, total_errors).
    """
    ec_tpfn_by_type = {}
    total = 0

    for table in sorted(os.listdir(isolated_dir)):
        tdir = os.path.join(isolated_dir, table)
        if not os.path.isdir(tdir):
            continue
        emap_path  = os.path.join(tdir, "error_map.csv")
        dirty_path = os.path.join(tdir, dirty_file)
        clean_path = os.path.join(tdir, clean_file)
        if not (os.path.exists(emap_path) and os.path.exists(dirty_path)
                and os.path.exists(clean_path)):
            continue

        try:
            dirty_df = pd.read_csv(dirty_path, dtype=str, keep_default_na=False, encoding="latin1")
            clean_df = pd.read_csv(clean_path, dtype=str, keep_default_na=False, encoding="latin1")
            dirty_df.columns = clean_df.columns
        except Exception:
            continue

        col_to_idx = {name: idx for idx, name in enumerate(dirty_df.columns)}

        emap_df = pd.read_csv(emap_path, keep_default_na=False, dtype=str, encoding="latin1")
        if emap_df.empty or "error_type" not in emap_df.columns:
            continue

        for _, row in emap_df.iterrows():
            try:
                r = int(row["row_number"])
            except (ValueError, TypeError):
                continue
            col_name = str(row.get("column_name", "")).strip()
            c = col_to_idx.get(col_name)
            if c is None:
                continue
            et = str(row.get("error_type", "")).strip()
            if not et:
                continue
            try:
                if dirty_df.iloc[r, c] == clean_df.iloc[r, c]:
                    continue   # not an actual error
            except IndexError:
                continue
            ec_tpfn_by_type[et] = ec_tpfn_by_type.get(et, 0) + 1
            total += 1

    return ec_tpfn_by_type, total


# ---------------------------------------------------------------------------
# Per-table evaluation
# ---------------------------------------------------------------------------

def evaluate_one(repaired_path, dirty_path, clean_path, error_map_path):
    """
    Evaluate a single repaired table.
    Returns (overall_dict, by_type_dict) where counts are raw integers.
    """
    dirty_df    = read_csv(dirty_path)
    clean_df    = read_csv(clean_path)
    repaired_df = read_csv(repaired_path)

    if "_tid_" in repaired_df.columns:
        repaired_df = repaired_df.drop(columns=["_tid_"])
    repaired_df.columns = dirty_df.columns

    detections = get_dataframes_difference(dirty_df, clean_df)
    n_errors   = len(detections)
    n_attempted = 0
    n_correct   = 0
    for (r, c) in detections:
        dirty_val    = dirty_df.iloc[r, c]
        clean_val    = clean_df.iloc[r, c]
        repaired_val = repaired_df.iloc[r, c]
        if dirty_val != repaired_val:
            n_attempted += 1
        if clean_val == repaired_val or (len(str(clean_val)) == 0 and len(str(repaired_val)) == 0):
            n_correct += 1

    overall = {
        "n_all_errors":              n_errors,
        "n_all_corrected_errors":    n_attempted,
        "n_truely_corrected_errors": n_correct,
    }

    by_type = {}
    if os.path.exists(error_map_path):
        error_map = load_error_map(error_map_path, dirty_df)
        if error_map:
            by_type = evaluate_per_type(dirty_df, clean_df, repaired_df, error_map)

    return overall, by_type


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate HoloClean lake results with per-type breakdown"
    )
    parser.add_argument(
        "--results-dir", "-r",
        default="/home/fatemeh/LakeCorrectionBench/HoloClean/outputs/2026-03-02/16-24-03/dcHoloCleaner-with_init/HoloClean",
        help="Directory containing repaired_holoclean_table_N_*.csv files",
    )
    parser.add_argument(
        "--isolated-dir", "-i",
        default="/home/fatemeh/LakeCorrectionBench/datasets/tables/uk_open_data/isolated",
        help="Isolated sandbox with UK_CSV.../dirty.csv, clean.csv, error_map.csv",
    )
    parser.add_argument(
        "--name-dict", "-n",
        default="/home/fatemeh/LakeCorrectionBench/open_data_93_res_dict_no.pkl",
        help="Pickle dict mapping UK_CSV paths -> table_N paths",
    )
    parser.add_argument(
        "--output-csv", "-o",
        default=None,
        help="Output CSV path (default: <results-dir>/holoclean_per_type_results.csv)",
    )
    parser.add_argument("--dirty-file", default="dirty.csv")
    parser.add_argument("--clean-file",  default="clean.csv")
    args = parser.parse_args()

    results_dir  = args.results_dir
    isolated_dir = args.isolated_dir
    output_csv   = args.output_csv or os.path.join(results_dir, "holoclean_per_type_results.csv")

    print(f"Results dir : {results_dir}")
    print(f"Isolated dir: {isolated_dir}")
    print(f"Name dict   : {args.name_dict}")

    table_to_uk = build_table_to_uk_map(args.name_dict)
    print(f"Name dict loaded: {len(table_to_uk)} entries")

    # ------------------------------------------------------------------
    # Lake-wide ec_tpfn from ALL tables (recall denominator)
    # ------------------------------------------------------------------
    print("\nCounting lake-wide errors across all isolated tables...")
    lake_ec_tpfn_by_type, lake_total_errors = get_lake_ec_tpfn(
        isolated_dir, args.dirty_file, args.clean_file
    )
    print(f"  Total lake errors: {lake_total_errors:,}")
    for et, cnt in sorted(lake_ec_tpfn_by_type.items()):
        print(f"  {et}: {cnt:,}")

    # ------------------------------------------------------------------
    # Discover and evaluate repaired files
    # ------------------------------------------------------------------
    repaired_files = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith("repaired_holoclean_") and f.endswith(".csv")
    )
    print(f"\nFound {len(repaired_files)} repaired CSV files.\n")

    print("=" * 80)
    print(f"{'TABLE':<35} {'ITER':>4} {'ERRORS':>8} {'CORRECTED':>10} "
          f"{'CORRECT':>9} {'PREC':>8} {'REC':>8} {'F1':>8}")
    print("=" * 80)

    per_table_rows = []   # raw counts per (table, iter) for CSV output
    all_error_types = set()

    # iter_accumulators[it] = {"tp": int, "corrected": int,
    #                          "by_type": {et: {"tp": int, "corrected": int}}}
    iter_accumulators = {}

    for fname in repaired_files:
        table_name = parse_table_name(fname)
        if table_name is None:
            print(f"  [SKIP] Cannot parse table name: {fname}")
            continue

        uk_name = table_to_uk.get(table_name)
        if uk_name is None:
            print(f"  [SKIP] {table_name} not in name dict")
            continue

        iso_dir = os.path.join(isolated_dir, uk_name)
        if not os.path.isdir(iso_dir):
            print(f"  [SKIP] Isolated dir not found: {iso_dir}")
            continue

        repaired_path  = os.path.join(results_dir, fname)
        dirty_path     = os.path.join(iso_dir, args.dirty_file)
        clean_path     = os.path.join(iso_dir, args.clean_file)
        error_map_path = os.path.join(iso_dir, "error_map.csv")

        if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
            print(f"  [SKIP] dirty/clean not found for {uk_name}")
            continue

        it = parse_iter(fname)

        try:
            overall, by_type = evaluate_one(
                repaired_path, dirty_path, clean_path, error_map_path
            )
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")
            continue

        tp       = overall["n_truely_corrected_errors"]
        n_corr   = overall["n_all_corrected_errors"]
        n_errors = overall["n_all_errors"]
        prec = tp / n_corr   if n_corr   > 0 else -1.0
        rec  = tp / n_errors if n_errors > 0 else -1.0   # per-table recall (informational)
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else -1.0

        print(
            f"  {uk_name:<33} {it:>4}  {n_errors:>8}  {n_corr:>8}  {tp:>8}  "
            f"{_fmt(prec):>8}  {_fmt(rec):>8}  {_fmt(f1):>8}"
        )

        # Accumulate into iter bucket
        if it not in iter_accumulators:
            iter_accumulators[it] = {"tp": 0, "corrected": 0, "by_type": {}}
        iter_accumulators[it]["tp"]        += tp
        iter_accumulators[it]["corrected"] += n_corr

        for et, m in by_type.items():
            all_error_types.add(et)
            if et not in iter_accumulators[it]["by_type"]:
                iter_accumulators[it]["by_type"][et] = {"tp": 0, "corrected": 0}
            iter_accumulators[it]["by_type"][et]["tp"]        += m["n_truely_corrected_errors"]
            iter_accumulators[it]["by_type"][et]["corrected"] += m["n_all_corrected_errors"]

        # Row for CSV
        row = {
            "file":       fname,
            "table_name": table_name,
            "uk_name":    uk_name,
            "iteration":  it,
            "n_all_errors":              n_errors,
            "n_all_corrected_errors":    n_corr,
            "n_truely_corrected_errors": tp,
            "by_error_type": json.dumps(by_type),
        }
        for et, m in by_type.items():
            row[f"{et}_tp"]      = m["n_truely_corrected_errors"]
            row[f"{et}_ec_tpfp"] = m["n_all_corrected_errors"]
            row[f"{et}_ec_tpfn"] = m["n_all_errors"]
        per_table_rows.append(row)

    if not per_table_rows:
        print("\nNo results to aggregate.")
        return

    per_table_df = pd.DataFrame(per_table_rows)
    per_table_df.to_csv(output_csv, index=False)
    print(f"\nPer-table results saved -> {output_csv}")

    # ------------------------------------------------------------------
    # Lake-wide aggregation — following evaluate_lake.py exactly
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("LAKE-WIDE AGGREGATED RESULTS")
    print("=" * 65)

    iter_precisions, iter_recalls, iter_f1s = [], [], []
    for it in sorted(iter_accumulators.keys()):
        acc    = iter_accumulators[it]
        tp_it  = acc["tp"]
        cor_it = acc["corrected"]
        prec_it = tp_it / cor_it            if cor_it            > 0 else -1.0
        rec_it  = tp_it / lake_total_errors if lake_total_errors > 0 else -1.0
        f1_it   = 2 * prec_it * rec_it / (prec_it + rec_it) if (prec_it + rec_it) > 0 else -1.0
        iter_precisions.append(prec_it)
        iter_recalls.append(rec_it)
        iter_f1s.append(f1_it)
        print(f"  iter {it}: tp={tp_it:,}  corrected={cor_it:,}  "
              f"P={_fmt(prec_it)}  R={_fmt(rec_it)}  F1={_fmt(f1_it)}")

    precision = _mean_valid(iter_precisions)
    recall    = _mean_valid(iter_recalls)
    f1        = _mean_valid(iter_f1s)
    std_f1    = _std_valid(iter_f1s, f1)

    n_tables = len(per_table_df["uk_name"].unique())
    print(f"\n{'Tables evaluated:':<25} {n_tables}")
    print(f"{'Iterations:':<25} {len(iter_accumulators)}")
    print(f"{'Total lake errors:':<25} {lake_total_errors:,}")
    print(f"{'Precision:':<25} {_fmt(precision)}")
    print(f"{'Recall:':<25} {_fmt(recall)}")
    print(f"{'F1:':<25} {_fmt(f1)}")
    if len(iter_f1s) > 1:
        print(f"{'F1 std:':<25} {_fmt(std_f1)}")

    # ------------------------------------------------------------------
    # Per-error-type aggregation — same accumulator pattern
    # ------------------------------------------------------------------
    if all_error_types:
        print("\n" + "=" * 65)
        print("PER-ERROR-TYPE AGGREGATED RESULTS")
        print("=" * 65)
        print(
            f"\n{'Error Type':<25} {'ec_tpfn':>8} {'Precision':>10} "
            f"{'Recall':>8} {'F1':>8}"
            + (f" {'F1 std':>8}" if len(iter_accumulators) > 1 else "")
        )
        print("-" * (70 if len(iter_accumulators) > 1 else 62))

        type_summary_rows = []
        for et in sorted(all_error_types):
            lake_ec_tpfn_et = lake_ec_tpfn_by_type.get(et, 0)

            et_precisions, et_recalls, et_f1s = [], [], []
            for it in sorted(iter_accumulators.keys()):
                bt = iter_accumulators[it]["by_type"].get(et, {"tp": 0, "corrected": 0})
                tp_et  = bt["tp"]
                cor_et = bt["corrected"]
                p = tp_et / cor_et          if cor_et          > 0 else -1.0
                r = tp_et / lake_ec_tpfn_et if lake_ec_tpfn_et > 0 else -1.0
                f = 2 * p * r / (p + r)     if (p + r)         > 0 else -1.0
                et_precisions.append(p)
                et_recalls.append(r)
                et_f1s.append(f)

            avg_p  = _mean_valid(et_precisions)
            avg_r  = _mean_valid(et_recalls)
            avg_f1 = _mean_valid(et_f1s)
            s_f1   = _std_valid(et_f1s, avg_f1)

            line = (f"{et:<25} {lake_ec_tpfn_et:>8} {_fmt(avg_p):>10} "
                    f"{_fmt(avg_r):>8} {_fmt(avg_f1):>8}")
            if len(iter_accumulators) > 1:
                line += f" {_fmt(s_f1):>8}"
            print(line)

            type_summary_rows.append({
                "error_type":   et,
                "ec_tpfn":      lake_ec_tpfn_et,
                "precision":    avg_p,
                "recall":       avg_r,
                "f1":           avg_f1,
                "f1_std":       s_f1,
            })

        type_csv = output_csv.replace(".csv", "_type_summary.csv")
        pd.DataFrame(type_summary_rows).to_csv(type_csv, index=False)
        print(f"\nPer-type summary saved -> {type_csv}")

    # ------------------------------------------------------------------
    # JSON lake summary
    # ------------------------------------------------------------------
    lake_summary = {
        "tables_in_lake":            sum(1 for t in os.listdir(isolated_dir)
                                         if os.path.isdir(os.path.join(isolated_dir, t))),
        "tables_evaluated":          n_tables,
        "iterations":                sorted(iter_accumulators.keys()),
        "lake_total_errors":         lake_total_errors,
        "lake_ec_tpfn_by_type":      {k: int(v) for k, v in lake_ec_tpfn_by_type.items()},
        "precision":                 precision,
        "recall":                    recall,
        "f1":                        f1,
        "f1_std":                    std_f1,
        "per_iteration": [
            {
                "iter":      it,
                "precision": iter_precisions[i],
                "recall":    iter_recalls[i],
                "f1":        iter_f1s[i],
            }
            for i, it in enumerate(sorted(iter_accumulators.keys()))
        ],
    }
    summary_json = output_csv.replace(".csv", "_lake_summary.json")
    with open(summary_json, "w") as f:
        json.dump(lake_summary, f, indent=2)
    print(f"Lake summary saved      -> {summary_json}")


if __name__ == "__main__":
    main()
