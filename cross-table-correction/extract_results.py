#!/usr/bin/env python3
"""
Extract metrics from aggregated_results.json files and produce:
  1. aggregated_metrics_by_budget.csv  – overall P/R/F1 averaged across iterations
  2. error_type_metrics_by_budget.csv  – per-error-type P/R/F1 (requires sandbox_path
     with error_map.csv files and cell_analysis/ CSVs in each output directory)

Directory naming convention:
    output_{dataset}_{iteration}_{labeling_budget}_{threshold}
"""
import json
import csv
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_output_dir_name(dir_name: str) -> Tuple[str, int]:
    """
    Return (labeling_budget_str, iteration_int) from an output directory name.
    Format: output_{dataset}_{iteration}_{budget}_{threshold}
    Parsed from the right to tolerate underscores in dataset names.
    """
    parts = dir_name.split("_")
    if len(parts) >= 5:
        try:
            labeling_budget = parts[-2]
            iteration = int(parts[-3])
            return labeling_budget, iteration
        except (ValueError, IndexError):
            pass
    return "", 0


def load_table_id_to_names(output_dir: Path) -> Dict[str, Dict]:
    """
    Load table_id_to_names.pickle from an output directory.
    Returns {table_id: {'table_name': str, 'columns': [{'column_name', 'column_index'}, ...]}}
    or empty dict if not found.
    """
    pickle_path = output_dir / "table_id_to_names.pickle"
    if not pickle_path.exists():
        return {}
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def load_error_map(sandbox_path: Path, table_name: str) -> Dict[Tuple[int, str], str]:
    """
    Load error_map.csv for a table from the sandbox directory.
    Returns {(row_number, column_name): error_type}.
    """
    emap_path = sandbox_path / table_name / "error_map.csv"
    if not emap_path.exists():
        return {}
    try:
        df = pd.read_csv(emap_path, keep_default_na=False, dtype=str, encoding="latin1")
    except Exception:
        return {}
    required = {"row_number", "column_name", "error_type"}
    if df.empty or not required.issubset(df.columns):
        return {}
    error_map = {}
    for _, row in df.iterrows():
        try:
            r = int(row["row_number"])
        except Exception:
            continue
        col_name = str(row["column_name"]).strip()
        et = str(row.get("error_type", "")).strip()
        if et:
            error_map[(r, col_name)] = et
    return error_map


def get_total_errors_by_error_type(sandbox_path: Path) -> Dict[str, int]:
    """
    Count total erroneous cells per error type across all tables in sandbox_path,
    using each table's error_map.csv.
    Returns {error_type: count}.
    """
    ec_tpfn_by_type: Dict[str, int] = {}
    for entry in sandbox_path.iterdir():
        if not entry.is_dir():
            continue
        emap_path = entry / "error_map.csv"
        if not emap_path.exists():
            continue
        try:
            df = pd.read_csv(emap_path, keep_default_na=False, dtype=str, encoding="latin1")
        except Exception:
            continue
        if df.empty or "error_type" not in df.columns:
            continue
        for _, row in df.iterrows():
            et = str(row.get("error_type", "")).strip()
            if et:
                ec_tpfn_by_type[et] = ec_tpfn_by_type.get(et, 0) + 1
    return ec_tpfn_by_type


def compute_per_error_type_counts(
    output_dir: Path,
    sandbox_path: Optional[Path],
    id_to_info: Dict[str, Dict],
) -> Dict[str, Dict[str, int]]:
    """
    Using cell_analysis/*.csv in output_dir, map each corrected / missed cell to an
    error type via the sandbox's error_map.csv.

    Returns {error_type: {'tp': int, 'fp': int, 'fn': int}}.
    correction_status values:
        CORRECT_CORRECTION   → TP
        INCORRECT_CORRECTION → FP (wrong correction applied)
        MISSED_ERROR         → FN (error left uncorrected)
    """
    if sandbox_path is None or not sandbox_path.exists():
        return {}

    cell_analysis_dir = output_dir / "cell_analysis"
    if not cell_analysis_dir.exists():
        return {}

    # Build column-index → column-name lookup per table_id
    col_lookup: Dict[str, Dict[int, str]] = {}
    for tid, info in id_to_info.items():
        cols = info.get("columns", [])
        col_lookup[tid] = {c["column_index"]: c["column_name"] for c in cols}

    # Cache error maps (table_name → error_map dict)
    error_map_cache: Dict[str, Dict[Tuple[int, str], str]] = {}

    per_type: Dict[str, Dict[str, int]] = {}

    for csv_file in cell_analysis_dir.glob("cell_analysis_*.csv"):
        try:
            df = pd.read_csv(csv_file, keep_default_na=False, dtype=str)
        except Exception:
            continue

        required_cols = {"table_id", "row_idx", "column_idx", "correction_status", "is_error"}
        if not required_cols.issubset(df.columns):
            continue

        for _, row in df.iterrows():
            if str(row.get("is_error", "")).strip().lower() != "true":
                continue

            table_id = str(row["table_id"]).strip()
            try:
                row_idx = int(row["row_idx"])
                col_idx = int(row["column_idx"])
            except (ValueError, TypeError):
                continue

            status = str(row.get("correction_status", "")).strip()

            # Resolve table_name and column_name
            info = id_to_info.get(table_id)
            if info is None:
                continue
            table_name = info["table_name"]
            col_name = col_lookup.get(table_id, {}).get(col_idx)
            if col_name is None:
                continue

            # Load (and cache) error map for this table
            if table_name not in error_map_cache:
                error_map_cache[table_name] = load_error_map(sandbox_path, table_name)
            emap = error_map_cache[table_name]

            et = emap.get((row_idx, col_name))
            if not et:
                continue

            stats = per_type.setdefault(et, {"tp": 0, "fp": 0, "fn": 0})
            if status == "CORRECT_CORRECTION":
                stats["tp"] += 1
            elif status == "INCORRECT_CORRECTION":
                stats["fp"] += 1
            elif status == "MISSED_ERROR":
                stats["fn"] += 1

    return per_type


# ---------------------------------------------------------------------------
# Metric extraction from aggregated_results.json
# ---------------------------------------------------------------------------

def extract_aggregate_metrics(json_data: Dict[str, Any]) -> Dict[str, Any]:
    agg = json_data.get("aggregate_stats", {})
    times = json_data.get("execution_times", {})
    return {
        "precision": agg.get("overall_precision", ""),
        "recall": agg.get("overall_recall", ""),
        "f1_score": agg.get("overall_f1", ""),
        "f1_score_std": agg.get("std_zone_f1", ""),
        "ec_tpfp": agg.get("total_corrected", ""),
        "ec_tpfn": agg.get("total_error_cells", ""),
        "tp": agg.get("total_correct_corrections", ""),
        "execution_time": times.get("total_zone_processing", ""),
        "n_labeled_cells": agg.get("total_manual_samples_correct", ""),
        "n_labeled_tuples": "",
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_results_directory(
    results_dir: str,
    sandbox_path: Optional[str] = None,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    Iterate through all output_* directories.
    Returns:
        grouped_agg   – {labeling_budget: [metrics_dict, ...]}
        grouped_types – {labeling_budget: [per_type_counts_dict, ...]}
    """
    results_path = Path(results_dir)
    sandbox = Path(sandbox_path) if sandbox_path else None

    grouped_agg: Dict[str, List[Dict]] = defaultdict(list)
    grouped_types: Dict[str, List[Dict]] = defaultdict(list)

    output_dirs = sorted(
        [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("output_")]
    )

    for out_dir in output_dirs:
        json_file = out_dir / "aggregated_results.json"
        if not json_file.exists():
            print(f"  [skip] No aggregated_results.json in {out_dir.name}")
            continue

        try:
            with open(json_file) as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"  [error] {out_dir.name}: {e}")
            continue

        budget, iteration = parse_output_dir_name(out_dir.name)
        metrics = extract_aggregate_metrics(json_data)
        metrics["labeling_budget"] = budget
        metrics["iteration"] = iteration
        grouped_agg[budget].append(metrics)
        print(f"  [ok] {out_dir.name}  budget={budget}  iter={iteration}")

        # Per-error-type analysis
        if sandbox is not None:
            id_to_info = load_table_id_to_names(out_dir)
            if id_to_info:
                per_type = compute_per_error_type_counts(out_dir, sandbox, id_to_info)
                if per_type:
                    grouped_types[budget].append(per_type)
                else:
                    print(f"    [warn] No error-type data for {out_dir.name}")
            else:
                print(f"    [warn] table_id_to_names.pickle missing in {out_dir.name}")

    return grouped_agg, grouped_types


def average_aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    numeric = [
        "precision", "recall", "f1_score", "f1_score_std",
        "ec_tpfp", "ec_tpfn", "tp", "execution_time",
        "n_labeled_cells", "n_labeled_tuples",
    ]
    averaged = {}
    for field in numeric:
        values = []
        for m in metrics_list:
            val = m.get(field, "")
            if val != "":
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass
        averaged[field] = sum(values) / len(values) if values else ""
    return averaged


def aggregate_error_type_metrics(
    per_type_list: List[Dict[str, Dict[str, int]]],
    ec_tpfn_by_type: Optional[Dict[str, int]],
) -> Dict[str, Dict[str, float]]:
    """
    Average per-error-type P/R/F1 across repetitions.
    ec_tpfn_by_type: lake-wide ground-truth count per error type (denominator for recall).
    Returns {error_type: {precision, recall, f1, tp, fp, fn, ec_tpfn}}.
    """
    # Accumulate across reps
    agg: Dict[str, Dict[str, List]] = {}
    for rep_counts in per_type_list:
        for et, stats in rep_counts.items():
            tp = stats.get("tp", 0)
            fp = stats.get("fp", 0)
            fn_local = stats.get("fn", 0)

            # Use lake-wide total for recall if available, else use local fn
            if ec_tpfn_by_type is not None:
                denom_recall = ec_tpfn_by_type.get(et, 0)
            else:
                denom_recall = tp + fn_local

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / denom_recall if denom_recall > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            bucket = agg.setdefault(et, {"precision": [], "recall": [], "f1": [],
                                          "tp": [], "fp": [], "fn": []})
            bucket["precision"].append(prec)
            bucket["recall"].append(rec)
            bucket["f1"].append(f1)
            bucket["tp"].append(tp)
            bucket["fp"].append(fp)
            bucket["fn"].append(fn_local)

    result = {}
    for et, vals in agg.items():
        result[et] = {
            "precision": float(np.mean(vals["precision"])),
            "recall": float(np.mean(vals["recall"])),
            "f1": float(np.mean(vals["f1"])),
            "tp": float(np.mean(vals["tp"])),
            "fp": float(np.mean(vals["fp"])),
            "fn": float(np.mean(vals["fn"])),
            "ec_tpfn": float(ec_tpfn_by_type.get(et, 0)) if ec_tpfn_by_type else float(np.mean(vals["tp"]) + np.mean(vals["fn"])),
        }
    return result


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_aggregate_csv(grouped_agg: Dict[str, List[Dict]], output_file: str):
    fieldnames = [
        "labeling_budget", "precision", "recall", "f1_score", "f1_score_std",
        "ec_tpfp", "ec_tpfn", "tp", "execution_time",
        "n_labeled_cells", "n_labeled_tuples",
    ]
    rows = []
    for budget in sorted(grouped_agg, key=lambda x: int(x) if x.isdigit() else 0):
        averaged = average_aggregate_metrics(grouped_agg[budget])
        averaged["labeling_budget"] = budget
        rows.append(averaged)
        n = len(grouped_agg[budget])
        print(f"  Averaged {n} iteration(s) for budget {budget}")

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {output_file}  ({len(rows)} rows)")


def save_error_type_csv(
    grouped_types: Dict[str, List[Dict]],
    ec_tpfn_by_type: Optional[Dict[str, int]],
    output_file: str,
):
    """
    Save a long-format CSV: one row per (labeling_budget, error_type).
    """
    fieldnames = [
        "labeling_budget", "error_type",
        "precision", "recall", "f1",
        "tp", "fp", "fn", "ec_tpfn",
    ]
    rows = []
    for budget in sorted(grouped_types, key=lambda x: int(x) if x.isdigit() else 0):
        et_metrics = aggregate_error_type_metrics(grouped_types[budget], ec_tpfn_by_type)
        for et, vals in sorted(et_metrics.items()):
            rows.append({
                "labeling_budget": budget,
                "error_type": et,
                **vals,
            })

    if not rows:
        print("[warn] No error-type data to save.")
        return

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {output_file}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # ---- Configure paths here ----
    _root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_root, "results_feature_ablation_clustering_based")

    # Path to the sandbox that contains per-table subdirs with error_map.csv.
    # Set to None to skip error-type analysis.
    sandbox_path = os.path.join(_root, "datasets", "tables", "uk_open_data", "isolated")

    output_agg_csv = os.path.join(results_dir, "aggregated_metrics_by_budget.csv")
    output_et_csv  = os.path.join(results_dir, "error_type_metrics_by_budget.csv")
    # ------------------------------

    print(f"Results directory : {results_dir}")
    if sandbox_path:
        print(f"Sandbox (error maps): {sandbox_path}")
    print("-" * 60)

    grouped_agg, grouped_types = process_results_directory(results_dir, sandbox_path)

    if not grouped_agg:
        print("No results found.")
        return

    print("\nAggregating metrics across iterations:")
    print("-" * 60)
    save_aggregate_csv(grouped_agg, output_agg_csv)

    if grouped_types:
        sandbox = Path(sandbox_path) if sandbox_path else None
        ec_tpfn_by_type = get_total_errors_by_error_type(sandbox) if sandbox else None
        if ec_tpfn_by_type:
            print(f"\nError types found in sandbox: {sorted(ec_tpfn_by_type.keys())}")
        print("\nSaving per-error-type metrics:")
        print("-" * 60)
        save_error_type_csv(grouped_types, ec_tpfn_by_type, output_et_csv)
    else:
        print("\n[info] No per-error-type data collected (check sandbox path and cell_analysis dirs).")


if __name__ == "__main__":
    main()
