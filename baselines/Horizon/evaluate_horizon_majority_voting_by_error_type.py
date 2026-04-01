#!/usr/bin/env python3
"""
Evaluate Horizon repair with provenance-based majority voting, broken down by error type.

Uses per-table merged_cell_source_map.csv files to classify each error cell.
Provide either:
    - directory containing table folders with merged_cell_source_map.csv
    - one table map path like .../<table_id>/merged_cell_source_map.csv (auto-resolved to parent root)

(e.g. 'FD_VIOLATION', 'RANDOM_TYPO', 'UNKNOWN').

For each error type, we:
    1. Count unique error provenances (TP+FN) -- each source cell counted once.
    2. Collect corrected values (and corresponding clean values) per provenance.
    3. Assign one repaired value per provenance by highest-frequency vote.
    4. Mark provenance TP/FP by comparing voted repaired value vs clean value.
    5. Compute precision, recall, F1.

Expects per blend table: dirty.csv, clean.csv, clean.csv.a2.clean (repaired), provenance.csv.
Provenance format: "table_name § col_id § row_id" per cell.
"""

import csv
import json
import logging
import os
import argparse
from collections import defaultdict
from collections import Counter

from utils import read_csv

logger = logging.getLogger(__name__)

PROVENANCE_SEP = " § "
REPAIRED_SUFFIX = "clean.csv.a2.clean"
MERGED_CELL_SOURCE_MAP = "merged_cell_source_map.csv"


def _normalize_val(v):
    return str(v).strip() if v is not None else ""


def _eq(a, b):
    a, b = _normalize_val(a), _normalize_val(b)
    if len(a) == 0 and len(b) == 0:
        return True
    return a == b


def _to_int(value):
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


class ErrorTypeLookup:
    """Lookup wrapper for per-table merged_cell_source_map CSV maps."""

    def __init__(self, source_path):
        self.source_path = source_path
        self._table_cache = {}

    def get(self, table_id, col_name, row_idx):
        row_idx = _to_int(row_idx)
        if table_id is None or row_idx is None:
            return "UNKNOWN"

        table_map = self._load_table_csv_map(table_id)
        return table_map.get((col_name, row_idx), "UNKNOWN")

    def describe(self):
        return f"per-table CSV maps under {self.source_path}"

    def _load_table_csv_map(self, table_id):
        if table_id in self._table_cache:
            return self._table_cache[table_id]

        csv_path = self._find_table_csv_path(table_id)
        if csv_path is None:
            self._table_cache[table_id] = {}
            return self._table_cache[table_id]

        df = read_csv(csv_path)
        required_cols = {"column_name", "row_number", "error_type"}
        if not required_cols.issubset(set(df.columns)):
            logger.warning("Missing required columns in %s; expected %s", csv_path, sorted(required_cols))
            self._table_cache[table_id] = {}
            return self._table_cache[table_id]

        table_map = {}
        for _, row in df.iterrows():
            col_name = _normalize_val(row.get("column_name"))
            row_num = _to_int(row.get("row_number"))
            error_type = _normalize_val(row.get("error_type")) or "UNKNOWN"
            if col_name and row_num is not None:
                table_map[(col_name, row_num)] = error_type

        self._table_cache[table_id] = table_map
        return table_map

    def _find_table_csv_path(self, table_id):
        table_id_str = str(table_id)
        candidates = [
            os.path.join(self.source_path, table_id_str, MERGED_CELL_SOURCE_MAP),
            os.path.join(self.source_path, "merged", table_id_str, MERGED_CELL_SOURCE_MAP),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        logger.debug("No merged cell source map found for table_id=%s under %s", table_id_str, self.source_path)
        return None


def load_error_lookup(error_map_path):
    """
    Load error-type lookup from per-table merged_cell_source_map CSV files.

    Accepted inputs:
    - Directory containing table folders, each with merged_cell_source_map.csv
    - Path to a single merged_cell_source_map.csv (parent root is inferred)
    """
    if os.path.isfile(error_map_path):
        csv_root = os.path.dirname(os.path.dirname(error_map_path))
        if os.path.basename(os.path.dirname(error_map_path)).isdigit():
            csv_root = os.path.dirname(csv_root)
        return ErrorTypeLookup(source_path=csv_root)
    if os.path.isdir(error_map_path):
        return ErrorTypeLookup(source_path=error_map_path)
    return None


def load_provenance_for_table(table_dir):
    prov_path = os.path.join(table_dir, "provenance.csv")
    if not os.path.exists(prov_path):
        return None
    df = read_csv(prov_path)
    return [df.iloc[r].tolist() for r in range(len(df))]


def row_col_to_provenance(provenance_matrix, row, col):
    if provenance_matrix is None or row < 0 or col < 0:
        return None
    if row >= len(provenance_matrix) or col >= len(provenance_matrix[0]):
        return None
    p = _normalize_val(provenance_matrix[row][col])
    if not p or PROVENANCE_SEP not in p:
        return None
    return p.split("|")[0].strip()


def get_table_id_from_dir(table_dir):
    basename = os.path.basename(os.path.normpath(table_dir))
    if basename.isdigit():
        return int(basename)
    return None


def get_unique_errors_by_provenance_and_type(dataset_root, error_lookup):
    """
    Count unique error cells by provenance, grouped by error type.
    Returns dict: error_type -> set of provenance strings.
    """
    type_prov_sets = defaultdict(set)
    tables = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
        and not d.startswith("union_summary")
        and not d.endswith(".json")
    ]
    for table in sorted(tables):
        table_dir = os.path.join(dataset_root, table)
        dirty_path = os.path.join(table_dir, "dirty.csv")
        clean_path = os.path.join(table_dir, "clean.csv")
        prov_path = os.path.join(table_dir, "provenance.csv")
        if not all(os.path.exists(p) for p in [dirty_path, clean_path, prov_path]):
            continue
        dirty_df = read_csv(dirty_path)
        clean_df = read_csv(clean_path)
        prov_df = read_csv(prov_path)
        if dirty_df.shape != clean_df.shape or dirty_df.shape != prov_df.shape:
            continue
        table_id = get_table_id_from_dir(table_dir)
        for r in range(len(dirty_df)):
            for c in range(len(dirty_df.columns)):
                if _normalize_val(dirty_df.iloc[r, c]) == _normalize_val(clean_df.iloc[r, c]):
                    continue
                prov = _normalize_val(prov_df.iloc[r, c])
                if not prov or PROVENANCE_SEP not in prov:
                    continue
                for p in prov.split("|"):
                    p = p.strip()
                    if not p:
                        continue
                    col_name = dirty_df.columns[c]
                    error_type = error_lookup.get(table_id, col_name, r) if table_id is not None else "UNKNOWN"
                    type_prov_sets[error_type].add(p)
    return dict(type_prov_sets)


def find_tables_with_provenance(dataset_root, repaired_name=REPAIRED_SUFFIX):
    for d in os.listdir(dataset_root):
        if d.startswith("union_summary") or d.endswith(".json"):
            continue
        table_dir = os.path.join(dataset_root, d)
        if not os.path.isdir(table_dir):
            continue
        required = ["dirty.csv", "clean.csv", "provenance.csv", repaired_name]
        if all(os.path.exists(os.path.join(table_dir, f)) for f in required):
            yield table_dir


def evaluate_one_table_with_provenance_and_type(table_dir, error_lookup, repaired_name=REPAIRED_SUFFIX):
    dirty_path = os.path.join(table_dir, "dirty.csv")
    clean_path = os.path.join(table_dir, "clean.csv")
    repaired_path = os.path.join(table_dir, repaired_name)
    if not all(os.path.exists(p) for p in [dirty_path, clean_path, repaired_path]):
        return None
    table_id = get_table_id_from_dir(table_dir)
    dirty_df = read_csv(dirty_path)
    clean_df = read_csv(clean_path)
    repaired_df = read_csv(repaired_path)

    if "_tid_" in repaired_df.columns:
        repaired_df = repaired_df.drop(columns=["_tid_"])
    if dirty_df.shape != repaired_df.shape or dirty_df.shape != clean_df.shape:
        logger.debug("Shape mismatch in %s", table_dir)
        return None

    prov_matrix = load_provenance_for_table(table_dir)
    if prov_matrix is None:
        return None

    n_rows, n_cols = len(dirty_df), len(dirty_df.columns)
    type_provenance_values = defaultdict(lambda: defaultdict(lambda: {
        "repaired_values": [],
        "clean_values": [],
    }))

    for r in range(n_rows):
        for c in range(n_cols):
            if _normalize_val(dirty_df.iloc[r, c]) == _normalize_val(clean_df.iloc[r, c]):
                continue
            if _normalize_val(dirty_df.iloc[r, c]) == _normalize_val(repaired_df.iloc[r, c]):
                continue

            prov = row_col_to_provenance(prov_matrix, r, c)
            if prov is None:
                continue

            col_name = dirty_df.columns[c]
            error_type = "UNKNOWN"
            if table_id is not None:
                error_type = error_lookup.get(table_id, col_name, r)

            type_provenance_values[error_type][prov]["repaired_values"].append(_normalize_val(repaired_df.iloc[r, c]))
            type_provenance_values[error_type][prov]["clean_values"].append(_normalize_val(clean_df.iloc[r, c]))

    return dict(type_provenance_values)


def _most_frequent(values):
    if not values:
        return None
    counts = Counter(values)
    max_count = max(counts.values())
    winners = [val for val, cnt in counts.items() if cnt == max_count]
    return sorted(winners)[0]


def majority_vote_per_provenance(prov_values):
    tp_count = fp_count = 0
    for values in prov_values.values():
        predicted_value = _most_frequent(values.get("repaired_values", []))
        clean_value = _most_frequent(values.get("clean_values", []))
        if predicted_value is None or clean_value is None:
            continue
        if _eq(predicted_value, clean_value):
            tp_count += 1
        else:
            fp_count += 1
    return tp_count, fp_count


def compute_metrics(tp, tpfp, tpfn):
    precision = tp / tpfp if tpfp > 0 else -1
    recall = tp / tpfn if tpfn > 0 else -1
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else -1
    return precision, recall, f1


def run_majority_voting_by_error_type(
    dataset_root,
    error_map_path,
    repaired_name=REPAIRED_SUFFIX,
    out_csv_path=None,
    out_json_path=None,
    table_dirs=None,
):
    """
    Full pipeline: majority-voting evaluation broken down by error type.

    1. Load error map.
    2. Count unique error provenances per error type (TP+FN).
    3. Collect repaired and clean values per provenance per error type from all tables.
    4. Assign one repaired value per provenance by highest-frequency vote.
    5. Decide TP/FP by comparing voted repaired value vs clean value.
    6. Compute and report per-type and overall metrics.

    Returns (results_dict, per_table_list).
    """
    logger.info("Starting majority-voting evaluation by error type: dataset_root=%s", dataset_root)

    error_lookup = load_error_lookup(error_map_path)
    if error_lookup is None:
        raise FileNotFoundError(f"Error map path not found or invalid: {error_map_path}")
    logger.info("Loaded error lookup: %s", error_lookup.describe())

    type_prov_sets = get_unique_errors_by_provenance_and_type(dataset_root, error_lookup)
    ec_tpfn_by_type = {etype: len(pset) for etype, pset in type_prov_sets.items()}
    logger.info("Unique errors by type (TP+FN): %s", ec_tpfn_by_type)

    if table_dirs is not None:
        tables_list = sorted(table_dirs)
    else:
        tables_list = sorted(find_tables_with_provenance(dataset_root, repaired_name))
    logger.info("Collecting provenance value sets from %d tables...", len(tables_list))

    lake_type_values = defaultdict(lambda: defaultdict(lambda: {
        "repaired_values": [],
        "clean_values": [],
    }))
    per_table = []

    for i, table_dir in enumerate(tables_list):
        if not os.path.isdir(table_dir):
            continue
        type_prov_values = evaluate_one_table_with_provenance_and_type(table_dir, error_lookup, repaired_name)
        if type_prov_values is None:
            continue

        n_corrected = 0
        for etype, prov_values in type_prov_values.items():
            for prov, values in prov_values.items():
                lake_type_values[etype][prov]["repaired_values"].extend(values["repaired_values"])
                lake_type_values[etype][prov]["clean_values"].extend(values["clean_values"])
                n_corrected += len(values["repaired_values"])

        per_table.append({
            "table": table_dir,
            "n_corrected_cells": n_corrected,
        })

        if (i + 1) % 10 == 0 or i == 0:
            counts = {k: len(v) for k, v in lake_type_values.items()}
            logger.info("  Processed %d/%d tables, unique corrected provenances by type: %s",
                        i + 1, len(tables_list), counts)

    logger.info("Processed %d tables total", len(per_table))

    results_by_type = {}
    total_tp = 0
    total_tpfp = 0
    total_tpfn = 0

    all_types = sorted(set(list(ec_tpfn_by_type.keys()) + list(lake_type_values.keys())))

    for etype in all_types:
        tpfn = ec_tpfn_by_type.get(etype, 0)
        prov_values = lake_type_values.get(etype, {})
        tp, fp = majority_vote_per_provenance(prov_values)
        tpfp = tp + fp
        precision, recall, f1 = compute_metrics(tp, tpfp, tpfn)

        results_by_type[etype] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tp": tp,
            "tpfp": tpfp,
            "tpfn": tpfn,
        }
        total_tp += tp
        total_tpfp += tpfp
        total_tpfn += tpfn

        logger.info("  %s: P=%.4f R=%.4f F1=%.4f (TP=%d, TP+FP=%d, TP+FN=%d)",
                     etype, precision, recall, f1, tp, tpfp, tpfn)

    overall_precision, overall_recall, overall_f1 = compute_metrics(total_tp, total_tpfp, total_tpfn)
    results_by_type["OVERALL"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1_score": overall_f1,
        "tp": total_tp,
        "tpfp": total_tpfp,
        "tpfn": total_tpfn,
    }
    logger.info("  OVERALL: P=%.4f R=%.4f F1=%.4f (TP=%d, TP+FP=%d, TP+FN=%d)",
                overall_precision, overall_recall, overall_f1, total_tp, total_tpfp, total_tpfn)

    if out_csv_path:
        with open(out_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["error_type", "precision", "recall", "f1_score", "tp", "tpfp", "tpfn"])
            w.writeheader()
            for etype in all_types + ["OVERALL"]:
                row = {"error_type": etype}
                row.update(results_by_type[etype])
                w.writerow(row)
        logger.info("Wrote CSV to %s", out_csv_path)

    if out_json_path:
        with open(out_json_path, "w") as f:
            json.dump({"by_error_type": results_by_type, "per_table": per_table}, f, indent=2)
        logger.info("Wrote JSON to %s", out_json_path)

    return results_by_type, per_table


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Horizon majority-voting evaluation broken down by error type"
    )
    parser.add_argument(
        "dataset_root",
        help="Root directory containing blend tables (dirty, clean, repaired, provenance)",
    )
    parser.add_argument(
        "error_map",
        help=(
            "Path to per-table error maps: directory containing merged_cell_source_map.csv "
            "files, or one table file .../<table_id>/merged_cell_source_map.csv"
        ),
    )
    parser.add_argument("--repaired", default=REPAIRED_SUFFIX, help="Repaired filename (default: clean.csv.a2.clean)")
    parser.add_argument("--csv", default=None, help="Output CSV path for per-type metrics")
    parser.add_argument("--json", default=None, help="Output JSON path for full results")
    args = parser.parse_args()

    results, _ = run_majority_voting_by_error_type(
        args.dataset_root,
        args.error_map,
        repaired_name=args.repaired,
        out_csv_path=args.csv,
        out_json_path=args.json,
    )

    print("\n=== MAJORITY VOTING RESULTS BY ERROR TYPE ===")
    for etype in sorted(k for k in results if k != "OVERALL"):
        m = results[etype]
        print(f"  {etype:15s}: P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1_score']:.4f}  "
              f"(TP={m['tp']}, TP+FP={m['tpfp']}, TP+FN={m['tpfn']})")
    m = results["OVERALL"]
    print(f"  {'OVERALL':15s}: P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1_score']:.4f}  "
          f"(TP={m['tp']}, TP+FP={m['tpfp']}, TP+FN={m['tpfn']})")
