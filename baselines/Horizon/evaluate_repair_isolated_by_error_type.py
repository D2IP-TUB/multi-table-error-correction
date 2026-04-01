#!/usr/bin/env python3
"""
Evaluate Horizon repair on isolated tables with per-error-type metrics.

For isolated-table sandboxes only (e.g. tables/uk_open_data/isolated or
tables/mit_dwh/isolated). Does not use or modify merged tables or merged
error maps (e.g. merged_cell_source_map.csv).

Analogous to Baran's get_res.py for isolated tables: scans a sandbox directory
where each subdir is one table with dirty.csv, clean.csv, repaired file, and
optionally error_map.csv (per-table) or error_map_all_tables.csv (lake-level
fallback). Uses the same evaluation logic as evaluate_repair.py
(utils.read_csv, get_dataframes_difference, evaluate) and breaks down results
by error type when an error map is available.

Error maps: per-table error_map.csv (columns row_number, column_name, error_type)
or sandbox-level error_map_all_tables.csv (columns table_id, row_number,
column_name, error_type) used as fallback when a table has no error_map.csv.

When Horizon results live in a different directory (e.g. OpenData/open_data_uk_93),
pass --repaired-dir so dirty/clean/error maps are read from sandbox_path and
repaired files from repaired_dir (same table subdir names in both).

Outputs:
  - Per-table CSV: dataset, precision, recall, f1_score, tp, ec_tpfp, ec_tpfn, per_error_type_counts
  - Aggregated CSV: precision, recall, f1_score, ec_tpfp, ec_tpfn, per_error_type_metrics (JSON)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from utils import read_csv, get_dataframes_difference, evaluate


# Default repaired filename used by Horizon
REPAIRED_SUFFIX = "clean.csv.a2.clean"

# Skip these subdirs when scanning sandbox (isolated tables only; no merged dirs)
SKIP_PREFIXES = ("union_summary",)
SKIP_SUFFIXES = (".json",)

# Lake-level error map used as fallback when a table has no per-table error_map.csv
ERROR_MAP_ALL_TABLES = "error_map_all_tables.csv"


def _build_error_map_from_df(df, col_name_to_idx):
    """Build {(row, col): error_type} from a DataFrame with row_number, column_name, error_type."""
    error_map = {}
    for _, row in df.iterrows():
        try:
            r = int(row["row_number"])
        except Exception:
            continue
        col_name = str(row["column_name"]).strip()
        c = col_name_to_idx.get(col_name)
        if c is None:
            continue
        et = str(row.get("error_type", "")).strip()
        if et:
            error_map[(r, c)] = et
    return error_map


def load_error_type_map_for_table(sandbox_path, table_name, dirty_file_name="dirty.csv"):
    """
    Load an error-type map for one table from isolated-table error maps.

    Prefers per-table error_map.csv (columns: row_number, column_name, error_type).
    If absent, falls back to sandbox-level error_map_all_tables.csv (columns:
    table_id, row_number, column_name, error_type) filtered by table_name.

    Uses the dirty CSV header to resolve column name -> column index.
    Returns dict {(row, col): error_type} or None if no map is available.
    """
    table_dir = os.path.join(sandbox_path, str(table_name))
    dirty_path = os.path.join(table_dir, dirty_file_name)
    if not os.path.exists(dirty_path):
        return None
    dirty_cols = list(pd.read_csv(dirty_path, nrows=0, encoding="latin1").columns)
    col_name_to_idx = {name: idx for idx, name in enumerate(dirty_cols)}

    # 1) Per-table error_map.csv (isolated layout, e.g. tables/uk_open_data/isolated/<table>/error_map.csv)
    emap_path = os.path.join(table_dir, "error_map.csv")
    if os.path.exists(emap_path):
        df = pd.read_csv(emap_path, keep_default_na=False, dtype=str, encoding="latin1")
        if not df.empty and {"row_number", "column_name", "error_type"}.issubset(df.columns):
            return _build_error_map_from_df(df, col_name_to_idx)

    # 2) Fallback: lake-level error_map_all_tables.csv (table_id, row_number, column_name, error_type)
    all_path = os.path.join(sandbox_path, ERROR_MAP_ALL_TABLES)
    if os.path.exists(all_path):
        df = pd.read_csv(all_path, keep_default_na=False, dtype=str, encoding="latin1")
        if not df.empty and "table_id" in df.columns and {"row_number", "column_name", "error_type"}.issubset(df.columns):
            df = df[df["table_id"].astype(str).str.strip() == str(table_name).strip()]
            if not df.empty:
                return _build_error_map_from_df(df, col_name_to_idx)

    return None


def get_total_errors_by_error_type(base_path, dirty_file_name="dirty.csv"):
    """
    Count total errors per error type over the lake using isolated error maps only.

    Uses per-table error_map.csv where present; for tables without it, counts from
    error_map_all_tables.csv (filtered by table_id) so merged tables are not used.
    Returns (ec_tpfn_by_type, total_errors_all_types).
    """
    ec_tpfn_by_type = {}
    total = 0
    tables_with_per_table_map = set()
    for name in os.listdir(base_path):
        if any(name.startswith(p) for p in SKIP_PREFIXES) or any(name.endswith(s) for s in SKIP_SUFFIXES):
            continue
        p = os.path.join(base_path, name)
        if not os.path.isdir(p):
            continue
        emap_path = os.path.join(p, "error_map.csv")
        if os.path.exists(emap_path):
            df = pd.read_csv(emap_path, keep_default_na=False, dtype=str, encoding="latin1")
            if not df.empty and "error_type" in df.columns:
                tables_with_per_table_map.add(name)
                for _, row in df.iterrows():
                    et = str(row.get("error_type", "")).strip()
                    if not et:
                        continue
                    ec_tpfn_by_type[et] = ec_tpfn_by_type.get(et, 0) + 1
                    total += 1

    # Add counts from error_map_all_tables.csv only for tables that have no per-table error_map.csv
    all_path = os.path.join(base_path, ERROR_MAP_ALL_TABLES)
    if os.path.exists(all_path):
        df = pd.read_csv(all_path, keep_default_na=False, dtype=str, encoding="latin1")
        if not df.empty and "table_id" in df.columns and "error_type" in df.columns:
            for table_id, group in df.groupby("table_id", dropna=False):
                tid = str(table_id).strip()
                if tid in tables_with_per_table_map:
                    continue
                for _, row in group.iterrows():
                    et = str(row.get("error_type", "")).strip()
                    if not et:
                        continue
                    ec_tpfn_by_type[et] = ec_tpfn_by_type.get(et, 0) + 1
                    total += 1

    return ec_tpfn_by_type, total


def evaluate_one_table(
    table_dir,
    dirty_file="dirty.csv",
    clean_file="clean.csv",
    repaired_file=REPAIRED_SUFFIX,
    repaired_dir=None,
    table_name=None,
):
    """
    Run Horizon-style evaluation for one table.

    dirty/clean are always read from table_dir. Repaired is read from
    repaired_dir/table_name/ when repaired_dir is set, else from table_dir.
    table_name is required when repaired_dir is set (same subdir name in both trees).

    Returns (results_dict, detections, dirty_df, clean_df, repaired_df) or (None, ...) if files missing.
    """
    dirty_path = os.path.join(table_dir, dirty_file)
    clean_path = os.path.join(table_dir, clean_file)
    if repaired_dir and table_name:
        repaired_path = os.path.join(repaired_dir, table_name, repaired_file)
    else:
        repaired_path = os.path.join(table_dir, repaired_file)
    for p in (dirty_path, clean_path, repaired_path):
        if not os.path.exists(p):
            return None, None, None, None, None

    dirty_df = read_csv(dirty_path)
    clean_df = read_csv(clean_path)
    repaired_df = read_csv(repaired_path)
    if "_tid_" in repaired_df.columns:
        repaired_df = repaired_df.drop(columns=["_tid_"])

    if dirty_df.shape != clean_df.shape or dirty_df.shape != repaired_df.shape:
        return None, None, None, None, None

    detections = get_dataframes_difference(dirty_df, clean_df)
    results = evaluate(detections, dirty_df, clean_df, repaired_df)

    # Map to Baran-like names: ec_tpfp = corrections attempted, ec_tpfn = total errors, tp = true positives
    n_all_errors = results["n_all_errors"]
    n_all_corrected = results["n_all_corrected_errors"]
    n_tp = results["n_truely_corrected_errors"]
    out = {
        "precision": results["precision"],
        "recall": results["recall"],
        "f1_score": results["f1_score"],
        "tp": n_tp,
        "ec_tpfp": n_all_corrected,
        "ec_tpfn": n_all_errors,
    }
    return out, detections, dirty_df, clean_df, repaired_df


def get_results_df_isolated(
    sandbox_path,
    dirty_file="dirty.csv",
    clean_file="clean.csv",
    repaired_file=REPAIRED_SUFFIX,
    repaired_dir=None,
):
    """
    Build per-table results for all isolated tables.

    dirty/clean/error maps are read from sandbox_path. Repaired files are read
    from repaired_dir when set (same table subdir names), else from sandbox_path.
    When error_map is available, compute per_error_type_counts (tp, fp, total per type).
    Returns DataFrame.
    """
    datasets = []
    for name in os.listdir(sandbox_path):
        if any(name.startswith(p) for p in SKIP_PREFIXES) or any(name.endswith(s) for s in SKIP_SUFFIXES):
            continue
        table_dir = os.path.join(sandbox_path, name)
        if not os.path.isdir(table_dir):
            continue
        datasets.append(name)

    # If using a separate repaired dir, restrict to tables that exist there too
    if repaired_dir:
        repaired_dir = str(Path(repaired_dir).resolve())
        if not os.path.isdir(repaired_dir):
            raise FileNotFoundError(f"repaired_dir is not a directory: {repaired_dir}")
        existing = {d for d in os.listdir(repaired_dir) if os.path.isdir(os.path.join(repaired_dir, d))}
        datasets = [d for d in datasets if d in existing]

    results_dict = {
        "dataset": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "tp": [],
        "ec_tpfp": [],
        "ec_tpfn": [],
        "per_error_type_counts": [],
    }

    error_type_cache = {}
    for ds in datasets:
        error_type_cache[ds] = load_error_type_map_for_table(sandbox_path, ds, dirty_file)

    for dataset in datasets:
        table_dir = os.path.join(sandbox_path, dataset)
        res, detections, dirty_df, clean_df, repaired_df = evaluate_one_table(
            table_dir,
            dirty_file,
            clean_file,
            repaired_file,
            repaired_dir=repaired_dir,
            table_name=dataset if repaired_dir else None,
        )
        if res is None:
            continue

        # Per-error-type counts: for each (row,col) in detections that was corrected, get error_type and TP/FP
        per_type_counts = {}
        error_map = error_type_cache.get(dataset)
        if error_map is not None and dirty_df is not None and clean_df is not None and repaired_df is not None:
            n_rep_rows, n_rep_cols = len(repaired_df), len(repaired_df.columns)
            for (r, c) in detections.keys():
                if r >= n_rep_rows or c >= n_rep_cols:
                    continue
                et = error_map.get((r, c))
                if not et:
                    continue
                dirty_val = dirty_df.iloc[r, c]
                clean_val = clean_df.iloc[r, c]
                repaired_val = repaired_df.iloc[r, c]
                if dirty_val == repaired_val:
                    continue  # no correction attempted for this cell
                stats = per_type_counts.setdefault(et, {"tp": 0, "fp": 0, "total": 0})
                stats["total"] += 1
                if clean_val == repaired_val or (len(str(clean_val)) == 0 and len(str(repaired_val)) == 0):
                    stats["tp"] += 1
                else:
                    stats["fp"] += 1

        results_dict["dataset"].append(dataset)
        results_dict["precision"].append(res["precision"])
        results_dict["recall"].append(res["recall"])
        results_dict["f1_score"].append(res["f1_score"])
        results_dict["tp"].append(res["tp"])
        results_dict["ec_tpfp"].append(res["ec_tpfp"])
        results_dict["ec_tpfn"].append(res["ec_tpfn"])
        results_dict["per_error_type_counts"].append(json.dumps(per_type_counts))

    result_df = pd.DataFrame.from_dict(results_dict)
    return result_df


def get_total_results_isolated(result_df, ec_tpfn_by_type=None):
    """
    Aggregate per-table results into lake-wide totals and per-error-type metrics.
    result_df must have columns: dataset, precision, recall, f1_score, tp, ec_tpfp, ec_tpfn, per_error_type_counts.
    """
    total_results = {
        "precision": [],
        "recall": [],
        "f1_score": [],
        "ec_tpfp": [],
        "ec_tpfn": [],
        "tp": [],
        "n_tables": [],
        "per_error_type_metrics": [],
    }

    # Single row: one "budget" (Horizon has no labeling budget)
    tp = result_df["tp"].sum()
    ec_tpfp = result_df["ec_tpfp"].sum()
    ec_tpfn = result_df["ec_tpfn"].sum()
    n_tables = len(result_df)

    if ec_tpfp == 0:
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
    else:
        precision = tp / ec_tpfp
        recall = tp / ec_tpfn if ec_tpfn else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Aggregate per-error-type from each table's per_error_type_counts
    per_type_agg = {}
    for _, row in result_df.iterrows():
        per_type_json = row.get("per_error_type_counts")
        if not isinstance(per_type_json, str) or not per_type_json:
            continue
        try:
            per_type = json.loads(per_type_json)
        except Exception:
            continue
        for et, stats in per_type.items():
            if not isinstance(stats, dict):
                continue
            tp_et = int(stats.get("tp", 0))
            total_et = int(stats.get("total", 0))
            if total_et <= 0 and tp_et <= 0:
                continue
            agg = per_type_agg.setdefault(et, {"tp": 0, "ec_tpfp": 0})
            agg["tp"] += tp_et
            agg["ec_tpfp"] += total_et

    per_error_type_metrics = {}
    if ec_tpfn_by_type is not None:
        for et, agg in per_type_agg.items():
            tp_et = agg["tp"]
            ec_tpfp_et = agg["ec_tpfp"]
            denom = ec_tpfn_by_type.get(et, 0)
            p_et = tp_et / ec_tpfp_et if ec_tpfp_et > 0 else 0.0
            r_et = tp_et / denom if denom > 0 else 0.0
            f_et = 2 * p_et * r_et / (p_et + r_et) if (p_et + r_et) > 0 else 0.0
            per_error_type_metrics[et] = {
                "precision": p_et,
                "recall": r_et,
                "f1": f_et,
                "tp": tp_et,
                "ec_tpfp": ec_tpfp_et,
                "ec_tpfn": denom,
            }

    total_results["precision"].append(precision)
    total_results["recall"].append(recall)
    total_results["f1_score"].append(f1_score)
    total_results["ec_tpfp"].append(ec_tpfp)
    total_results["ec_tpfn"].append(ec_tpfn)
    total_results["tp"].append(tp)
    total_results["n_tables"].append(n_tables)
    total_results["per_error_type_metrics"].append(json.dumps(per_error_type_metrics))

    return pd.DataFrame.from_dict(total_results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Horizon repair on isolated tables with per-error-type metrics"
    )
    parser.add_argument(
        "sandbox_path",
        type=str,
        help="Root directory with table subdirs containing dirty.csv, clean.csv, and optionally error_map.csv (error types and ground truth).",
    )
    parser.add_argument(
        "--repaired-dir",
        type=str,
        default=None,
        help="Root directory where Horizon repaired outputs live (same table subdir names). If not set, repaired file is read from sandbox_path.",
    )
    parser.add_argument("--dirty", default="dirty.csv", help="Dirty CSV filename (default: dirty.csv)")
    parser.add_argument("--clean", default="clean.csv", help="Clean CSV filename (default: clean.csv)")
    parser.add_argument(
        "--repaired",
        default=REPAIRED_SUFFIX,
        help=f"Repaired CSV filename (default: {REPAIRED_SUFFIX})",
    )
    parser.add_argument(
        "--per-table-csv",
        type=str,
        default=None,
        help="Path to write per-table results CSV (default: <sandbox_path>/horizon_results_per_table.csv)",
    )
    parser.add_argument(
        "--aggregate-csv",
        type=str,
        default=None,
        help="Path to write aggregated results CSV (default: <sandbox_path>/horizon_aggregate_by_error_type.csv)",
    )
    args = parser.parse_args()

    sandbox_path = str(Path(args.sandbox_path).resolve())
    if not os.path.isdir(sandbox_path):
        sys.stderr.write(f"Not a directory: {sandbox_path}\n")
        sys.exit(1)

    repaired_dir = str(Path(args.repaired_dir).resolve()) if args.repaired_dir else None
    result_df = get_results_df_isolated(
        sandbox_path,
        dirty_file=args.dirty,
        clean_file=args.clean,
        repaired_file=args.repaired,
        repaired_dir=repaired_dir,
    )
    if result_df.empty:
        sys.stderr.write("No tables evaluated (missing dirty/clean/repaired or shape mismatch).\n")
        sys.exit(1)

    per_table_csv = args.per_table_csv or os.path.join(sandbox_path, "horizon_results_per_table.csv")
    result_df.to_csv(per_table_csv, index=False)
    print(f"Per-table results written to {per_table_csv} ({len(result_df)} tables)")

    ec_tpfn_by_type, _ = get_total_errors_by_error_type(sandbox_path, args.dirty)
    total_df = get_total_results_isolated(result_df, ec_tpfn_by_type)
    aggregate_csv = args.aggregate_csv or os.path.join(sandbox_path, "horizon_aggregate_by_error_type.csv")
    total_df.to_csv(aggregate_csv, index=False)
    print(f"Aggregate results written to {aggregate_csv}")

    print("\nAggregate metrics:")
    row = total_df.iloc[0]
    print(f"  Precision: {row['precision']:.4f}")
    print(f"  Recall:    {row['recall']:.4f}")
    print(f"  F1 Score:  {row['f1_score']:.4f}")
    print(f"  TP:        {row['tp']:.0f}  EC_TPFP: {row['ec_tpfp']:.0f}  EC_TPFN: {row['ec_tpfn']:.0f}")
    print(f"  Tables:    {row['n_tables']:.0f}")

    per_type_str = row.get("per_error_type_metrics")
    if isinstance(per_type_str, str) and per_type_str:
        try:
            per_type = json.loads(per_type_str)
            if per_type:
                print("\nPer-error-type metrics:")
                for et, m in sorted(per_type.items()):
                    print(f"  {et}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} (tp={m['tp']:.0f} tpfp={m['ec_tpfp']:.0f} tpfn={m['ec_tpfn']:.0f})")
        except json.JSONDecodeError:
            pass


if __name__ == "__main__":
    main()
