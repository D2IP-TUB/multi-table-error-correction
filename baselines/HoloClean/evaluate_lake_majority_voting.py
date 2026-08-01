#!/usr/bin/env python3
from __future__ import annotations
"""
Lake-level evaluation script with MAJORITY VOTING for joined tables.

When the lake has joins, the same source error cell can appear in multiple rows
and get corrected multiple times. This script:
  1. Tracks each (row, col) by its provenance (source_table § col § row)
  2. For each unique provenance that was corrected 2+ times, collects all 
     proposed correction values
  3. Picks the value with the highest frequency (majority vote)
  4. Compares the winning value to ground truth: if it matches -> TP, else -> FP
  
This avoids double-counting errors in evaluation.

Usage:
    python evaluate_lake_majority_voting.py \
        --output-dir  outputs/2026-02-20/08-10-15/dcHoloCleaner-with_init/HoloClean \
        --input-dir   datasets/Real_Lake_Default_Datasets/merged_strings_default_set_union/uk_open_data/merged \
        [--save-csv   results/lake_evaluation_majority_voting.csv]
"""

import argparse
import os
import re
import sys
import logging
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Make sure utils is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import read_csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename / metadata parsing
# ---------------------------------------------------------------------------

# Pattern: repaired_holoclean_<table>_seed<seed>_iter<iter>.csv
_FILENAME_RE = re.compile(
    r"^repaired_holoclean_(?P<table>.+?)_seed(?P<seed>\d+)_iter(?P<iter>\d+)\.csv$"
)

# One hour in seconds (for execution_time threshold)
_ONE_HOUR_SECONDS = 3600.0


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


def load_execution_time_map(output_dir: str) -> dict[str, float]:
    """
    Load per-table execution_time (in seconds) from lake_aggregated_results_per_table.csv.

    We expect the CSV to live either in `output_dir` or its parent directory.
    Returns: dict mapping dataset name (e.g. "table_33") -> max execution_time.
    """
    candidates = [
        os.path.join(output_dir, "lake_aggregated_results_per_table.csv"),
        os.path.join(os.path.dirname(output_dir), "lake_aggregated_results_per_table.csv"),
    ]
    csv_path = None
    for p in candidates:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        logger.info("No lake_aggregated_results_per_table.csv found – execution_time threshold not applied.")
        return {}

    try:
        df = pd.read_csv(csv_path)
        if "dataset" not in df.columns or "execution_time" not in df.columns:
            logger.warning(
                "lake_aggregated_results_per_table.csv missing required columns – execution_time threshold not applied."
            )
            return {}
        exec_map: dict[str, float] = {}
        for _, row in df.iterrows():
            dataset = str(row["dataset"]).strip()
            try:
                t = float(row["execution_time"])
            except Exception:
                continue
            if not dataset:
                continue
            prev = exec_map.get(dataset)
            exec_map[dataset] = max(prev, t) if prev is not None else t
        logger.info("Loaded execution_time for %d dataset(s) from %s", len(exec_map), csv_path)
        return exec_map
    except Exception as exc:
        logger.warning("Failed to load execution times from %s: %s", csv_path, exc)
        return {}


# ---------------------------------------------------------------------------
# Provenance loading
# ---------------------------------------------------------------------------

def load_provenance_map(table_dir: str) -> dict[tuple[int, int], str]:
    """
    Load provenance mapping from provenance.csv or merged_cell_source_map.csv.
    
    Returns:
        dict mapping (row_idx, col_idx) -> "source_table § col § row"
    """
    provenance_csv = os.path.join(table_dir, "provenance.csv")
    merged_map_csv = os.path.join(table_dir, "merged_cell_source_map.csv")
    
    provenance_map = {}
    
    # Try provenance.csv first (preferred format with § separator)
    if os.path.exists(provenance_csv):
        try:
            # Use latin-1 to avoid decode errors on '§' and other bytes.
            with open(provenance_csv, 'r', encoding='latin-1') as f:
                lines = f.readlines()
            
            # Skip header line(s) - first line is column mapping
            if len(lines) < 2:
                logger.warning("provenance.csv has insufficient data")
                return provenance_map
            
            # Parse data rows
            for row_idx, line in enumerate(lines[1:]):  # Skip header
                # Split by comma (columns are separated by commas)
                # Each cell contains "source_table § col § row" or empty
                cells = line.strip().split(',')
                
                for col_idx, cell_value in enumerate(cells):
                    cell_value = cell_value.strip()
                    if cell_value and '§' in cell_value:
                        # Accept both § (section sign) and alternatives
                        provenance_map[(row_idx, col_idx)] = cell_value
                        
            logger.info(f"Loaded {len(provenance_map)} provenance entries from provenance.csv")
            return provenance_map
            
        except Exception as e:
            logger.warning(f"Failed to parse provenance.csv: {e}")
    
    # Fall back to merged_cell_source_map.csv
    if os.path.exists(merged_map_csv):
        try:
            df = pd.read_csv(merged_map_csv)
            
            # Expected columns: cell_id, table_id, column_id, row_number, column_name, 
            # source_table, source_row, source_column, error_type
            for _, row in df.iterrows():
                row_idx = int(row['row_number'])
                col_idx = int(row['column_id'])
                source_table = str(row['source_table'])
                source_row = str(row['source_row'])
                source_col = str(row['source_column'])
                
                # Build provenance string in same format: source_table § col § row
                provenance_str = f"{source_table} § {source_col} § {source_row}"
                provenance_map[(row_idx, col_idx)] = provenance_str
            
            logger.info(f"Loaded {len(provenance_map)} provenance entries from merged_cell_source_map.csv")
            return provenance_map
            
        except Exception as e:
            logger.warning(f"Failed to parse merged_cell_source_map.csv: {e}")
    
    logger.warning("No provenance information found")
    return provenance_map


def _find_isolated_error_map(input_dir: str) -> str | None:
    """
    Auto-discover the isolated error map CSV from the merged input_dir path.

    Convention: input_dir ends with .../{dataset_name}/merged, and the isolated
    error map lives at {repo_root}/datasets/tables/{dataset_name}/isolated/error_map_all_tables.csv,
    where {repo_root} is an ancestor directory that contains a "datasets/" subtree.

    Returns the absolute path if found, else None.
    """
    p = Path(input_dir).resolve()
    # dataset name is the component just above 'merged'
    dataset_name = p.parent.name if p.name == "merged" else p.name
    # Walk up through all ancestors looking for a root that has
    # datasets/tables/{dataset_name}/isolated/error_map_all_tables.csv
    for ancestor in p.parents:
        candidate = ancestor / "datasets" / "tables" / dataset_name / "isolated" / "error_map_all_tables.csv"
        if candidate.exists():
            logger.info("Auto-discovered isolated error map: %s", candidate)
            return str(candidate)
        # Stop at the filesystem root
        if ancestor == ancestor.parent:
            break
    return None


def load_error_type_counts_from_isolated(error_map_path: str) -> dict[str, int]:
    """
    Load ground-truth per-type unique error counts from the isolated (pre-join)
    error map (e.g. datasets/tables/mit_dwh/isolated/error_map_all_tables.csv).

    Each row in that file represents one unique error cell in a source table,
    so a simple groupby on error_type gives the true denominator for lake-wide
    per-type recall (unaffected by join duplication in merged tables).

    Keys are returned in upper-case so they match values from merged_cell_source_map.csv.
    Returns: dict mapping error_type (upper-cased) -> count of unique error cells.
    """
    if not error_map_path or not os.path.exists(error_map_path):
        logger.warning("Isolated error map not found: %s", error_map_path)
        return {}
    try:
        df = pd.read_csv(error_map_path, dtype=str)
        if "error_type" not in df.columns:
            logger.warning("'error_type' column missing in %s", error_map_path)
            return {}
        counts = df["error_type"].str.upper().value_counts().to_dict()
        counts = {str(k): int(v) for k, v in counts.items()}
        logger.info(
            "Loaded isolated error type counts from %s: %s",
            error_map_path,
            counts,
        )
        return counts
    except Exception as exc:
        logger.warning("Failed to load isolated error map %s: %s", error_map_path, exc)
        return {}


def load_error_type_map(table_dir: str) -> dict[tuple[int, int], str]:
    """
    Load (row_idx, col_idx) -> error_type from merged_cell_source_map.csv.
    Returns empty dict if file or column is missing. Empty/missing error_type
    is stored as "" (caller should normalize to "unknown" when used).
    """
    merged_map_csv = os.path.join(table_dir, "merged_cell_source_map.csv")
    error_type_map = {}
    if not os.path.exists(merged_map_csv):
        return error_type_map
    try:
        df = pd.read_csv(merged_map_csv)
        if "error_type" not in df.columns or "row_number" not in df.columns or "column_id" not in df.columns:
            return error_type_map
        for _, row in df.iterrows():
            row_idx = int(row["row_number"])
            col_idx = int(row["column_id"])
            et = str(row["error_type"]).strip() if pd.notna(row.get("error_type")) else ""
            error_type_map[(row_idx, col_idx)] = et
        logger.info("Loaded %d error-type entries from merged_cell_source_map.csv", len(error_type_map))
    except Exception as e:
        logger.warning("Failed to load error types from merged_cell_source_map.csv: %s", e)
    return error_type_map


# ---------------------------------------------------------------------------
# Majority voting evaluation
# ---------------------------------------------------------------------------

def _sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '::' and other problematic characters in column names."""
    df.columns = [re.sub(r"::", "_", col) for col in df.columns]
    return df


def get_dataframes_difference(dirty_df: pd.DataFrame, clean_df: pd.DataFrame) -> list[tuple[int, int]]:
    """Return list of (row, col) tuples where dirty != clean."""
    detections = []
    for i in range(len(dirty_df)):
        for j in range(len(dirty_df.columns)):
            dirty_val = str(dirty_df.iloc[i, j]).strip()
            clean_val = str(clean_df.iloc[i, j]).strip()
            if dirty_val != clean_val:
                detections.append((i, j))
    return detections


def evaluate_with_majority_voting(
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    repaired_df: pd.DataFrame,
    provenance_map: dict[tuple[int, int], str],
    error_type_map: dict[tuple[int, int], str] | None = None,
) -> dict:
    """
    Evaluate repairs using majority voting for cells with the same provenance.
    
    Algorithm:
      1. Identify all errors (dirty != clean)
      2. For each error, get its provenance
      3. Group errors by provenance
      4. For provenances that appear 2+ times:
         - Collect all proposed corrections (repaired values)
         - Apply majority voting to pick the winning value
         - Compare winning value to ground truth
      5. For provenances that appear only once:
         - Directly compare repaired value to ground truth
    
    If error_type_map is provided (from merged_cell_source_map.csv), metrics
    are also computed per error type (by_error_type).
    
    Returns:
        dict with keys: n_unique_errors, n_truely_corrected_errors, 
                       n_all_corrected_errors, precision, recall, f1_score,
                       n_duplicate_errors, n_majority_voted, by_error_type
    """
    def _error_type(row_idx: int, col_idx: int) -> str:
        et = (error_type_map or {}).get((row_idx, col_idx), "").strip()
        return et if et else "unknown"

    # Step 1: Get all errors
    detections = get_dataframes_difference(dirty_df, clean_df)
    
    if len(detections) == 0:
        return {
            "n_unique_errors": 0,
            "n_truely_corrected_errors": 0,
            "n_all_corrected_errors": 0,
            "n_duplicate_errors": 0,
            "n_majority_voted": 0,
            "precision": -1,
            "recall": -1,
            "f1_score": -1,
            "by_error_type": {},
        }
    
    # Per-error-type accumulators: type -> { n_unique, n_tp, n_attempted }
    # NOTE: n_unique is interpreted *cell-wise* (number of erroneous cells for
    #       this type), not provenance-wise. This makes per-type UNIQ_ERR
    #       directly comparable to cell-level error counts across systems.
    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_unique": 0, "n_tp": 0, "n_attempted": 0}
    )
    
    # Step 2: Group errors by provenance
    # provenance_to_cells: provenance_str -> list of (row_idx, col_idx)
    provenance_to_cells = defaultdict(list)
    errors_without_provenance = []
    
    for row_idx, col_idx in detections:
        prov = provenance_map.get((row_idx, col_idx))
        if prov:
            provenance_to_cells[prov].append((row_idx, col_idx))
        else:
            errors_without_provenance.append((row_idx, col_idx))
    
    # Step 3: Process each unique provenance
    n_truely_corrected = 0
    n_attempted_corrections = 0
    n_duplicate_errors = 0
    n_majority_voted = 0
    
    for prov, cells in provenance_to_cells.items():
        # Get ground truth (should be same for all cells with same provenance)
        ground_truth = str(clean_df.iloc[cells[0][0], cells[0][1]]).strip()
        dirty_val = str(dirty_df.iloc[cells[0][0], cells[0][1]]).strip()
        err_type = _error_type(cells[0][0], cells[0][1])
        # Count provenance-wise (one unique source error = one count),
        # regardless of how many merged rows share the same provenance.
        by_type[err_type]["n_unique"] += 1
        
        if len(cells) == 1:
            # Single occurrence - no majority voting needed
            row_idx, col_idx = cells[0]
            repaired_val = str(repaired_df.iloc[row_idx, col_idx]).strip()
            
            # Check if correction was attempted
            if dirty_val != repaired_val:
                n_attempted_corrections += 1
                by_type[err_type]["n_attempted"] += 1
                # Check if correction is correct
                if repaired_val == ground_truth or (len(repaired_val) == 0 and len(ground_truth) == 0):
                    n_truely_corrected += 1
                    by_type[err_type]["n_tp"] += 1
        else:
            # Multiple occurrences - apply majority voting
            n_duplicate_errors += len(cells) - 1  # Count duplicates
            n_majority_voted += 1
            
            # Collect all proposed corrections
            proposed_values = []
            correction_attempted = False
            
            for row_idx, col_idx in cells:
                repaired_val = str(repaired_df.iloc[row_idx, col_idx]).strip()
                proposed_values.append(repaired_val)
                
                if dirty_val != repaired_val:
                    correction_attempted = True
            
            # Apply majority voting (tie-break: lexicographically smallest value)
            if correction_attempted:
                n_attempted_corrections += 1
                by_type[err_type]["n_attempted"] += 1
                
                value_counts = Counter(proposed_values)
                max_count = value_counts.most_common(1)[0][1]
                tied_winners = [v for v, c in value_counts.items() if c == max_count]
                winning_value = min(tied_winners)
                
                # Compare winning value to ground truth
                if winning_value == ground_truth or (len(winning_value) == 0 and len(ground_truth) == 0):
                    n_truely_corrected += 1
                    by_type[err_type]["n_tp"] += 1
    
    # Process errors without provenance (fallback to standard evaluation)
    for row_idx, col_idx in errors_without_provenance:
        err_type = _error_type(row_idx, col_idx)
        by_type[err_type]["n_unique"] += 1
        dirty_val = str(dirty_df.iloc[row_idx, col_idx]).strip()
        clean_val = str(clean_df.iloc[row_idx, col_idx]).strip()
        repaired_val = str(repaired_df.iloc[row_idx, col_idx]).strip()
        
        if dirty_val != repaired_val:
            n_attempted_corrections += 1
            by_type[err_type]["n_attempted"] += 1
            if repaired_val == clean_val or (len(repaired_val) == 0 and len(clean_val) == 0):
                n_truely_corrected += 1
                by_type[err_type]["n_tp"] += 1
    
    # Calculate unique errors (count each provenance only once)
    n_unique_errors = len(provenance_to_cells) + len(errors_without_provenance)
    
    # Calculate metrics
    precision = n_truely_corrected / n_attempted_corrections if n_attempted_corrections > 0 else -1
    recall = n_truely_corrected / n_unique_errors if n_unique_errors > 0 else -1
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else -1
    
    # Build by_error_type with precision/recall/f1 per type
    by_error_type = {}
    for et, acc in by_type.items():
        if acc["n_unique"] == 0:
            continue
        p = acc["n_tp"] / acc["n_attempted"] if acc["n_attempted"] > 0 else -1.0
        r = acc["n_tp"] / acc["n_unique"] if acc["n_unique"] > 0 else -1.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else -1.0
        by_error_type[et] = {
            "n_unique_errors": acc["n_unique"],
            "n_truely_corrected_errors": acc["n_tp"],
            "n_all_corrected_errors": acc["n_attempted"],
            "precision": p,
            "recall": r,
            "f1_score": f,
        }
    
    return {
        "n_unique_errors": n_unique_errors,
        "n_total_error_cells": len(detections),
        "n_duplicate_errors": n_duplicate_errors,
        "n_majority_voted": n_majority_voted,
        "n_truely_corrected_errors": n_truely_corrected,
        "n_all_corrected_errors": n_attempted_corrections,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "by_error_type": by_error_type,
    }


def evaluate_single(
    dirty_path: str,
    clean_path: str,
    repaired_path: str,
    provenance_map: dict[tuple[int, int], str],
    table_name: str,
    error_type_map: dict[tuple[int, int], str] | None = None,
) -> dict | None:
    """
    Evaluate one repaired file with majority voting.
    Returns a dict of metrics, or None on error.
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

        results = evaluate_with_majority_voting(
            dirty_df, clean_df, repaired_df, provenance_map, error_type_map
        )
        
        if results["n_unique_errors"] == 0:
            logger.warning("[%s] No errors detected (dirty == clean) – skipping", table_name)
            return None

        return results

    except Exception as exc:
        logger.warning("[%s] Evaluation failed: %s", table_name, exc)
        logger.debug(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Main aggregation logic
# ---------------------------------------------------------------------------

def count_lake_errors(input_dir: str) -> tuple[int, int]:
    """
    Count total unique errors across ALL tables in the input lake.
    Returns (total_unique_errors, total_error_cells).
    """
    total_unique = 0
    total_cells = 0
    
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
            
            provenance_map = load_provenance_map(table_dir)
            detections = get_dataframes_difference(dirty_df, clean_df)
            
            # Count unique errors by provenance
            provenance_set = set()
            errors_without_prov = 0
            
            for row_idx, col_idx in detections:
                prov = provenance_map.get((row_idx, col_idx))
                if prov:
                    provenance_set.add(prov)
                else:
                    errors_without_prov += 1
            
            n_unique = len(provenance_set) + errors_without_prov
            total_unique += n_unique
            total_cells += len(detections)
            
        except Exception as exc:
            logger.warning("[%s] Could not count errors for lake total: %s", table_name, exc)
    
    return total_unique, total_cells


def run_lake_evaluation(
    output_dir: str,
    input_dir: str,
    save_csv: str | None = None,
    error_map_path: str | None = None,
):
    entries = discover_repaired_files(output_dir)
    if not entries:
        logger.error("No repaired files found in %s", output_dir)
        sys.exit(1)

    # Load ground-truth per-type error counts from the isolated error map.
    # These are used as the denominator for per-type recall at lake level,
    # avoiding double-counting that occurs when the same source error appears
    # in multiple merged rows (due to joins) or in multiple merged tables.
    isolated_error_type_counts: dict[str, int] = {}
    resolved_map_path = error_map_path
    if resolved_map_path and not os.path.exists(resolved_map_path):
        # Resolve relative paths against the input_dir so the caller does not
        # have to worry about the working directory.
        resolved_map_path = str(Path(input_dir) / resolved_map_path)
    if not resolved_map_path or not os.path.exists(resolved_map_path):
        # Fall back to auto-discovery based on the input_dir convention.
        resolved_map_path = _find_isolated_error_map(input_dir)
    if resolved_map_path:
        isolated_error_type_counts = load_error_type_counts_from_isolated(resolved_map_path)
    else:
        logger.warning(
            "No isolated error map found (pass --error-map or place it at "
            "datasets/tables/{dataset}/isolated/error_map_all_tables.csv). "
            "Per-type UNIQ_ERR will be derived from merged-table provenance counts."
        )

    # Load per-table execution times (seconds); used to skip tables that ran > 1h
    execution_time_map = load_execution_time_map(output_dir)

    logger.info("Found %d repaired file(s) across the lake.", len(entries))

    # Count total errors across the entire lake
    logger.info("Counting total errors across all tables in the lake...")
    total_unique_errors, total_error_cells = count_lake_errors(input_dir)
    logger.info("Total unique errors in lake: %d (total error cells: %d)", 
                total_unique_errors, total_error_cells)

    # Group by table name
    tables: dict[str, list[dict]] = {}
    for e in entries:
        tables.setdefault(e["table"], []).append(e)

    logger.info("Covering %d unique table(s).", len(tables))

    per_table_rows = []
    skipped_tables = []

    # iter_accumulators: iter_num -> {"tp": int, "corrected": int}
    iter_accumulators: dict[int, dict] = {}
    tables_evaluated = 0
    iters_evaluated = 0

    print("\n" + "=" * 110)
    print(f"{'TABLE':<25} {'ITER':>4} {'UNIQ_ERR':>9} {'TOT_CELLS':>10} "
          f"{'DUP':>5} {'MV':>4} {'CORR':>6} {'TP':>6} {'PREC':>8} {'REC':>8} {'F1':>8}")
    print("=" * 110)

    for table_name in sorted(tables.keys()):
        table_dir = os.path.join(input_dir, table_name)
        # If the folder doesn't exist, try stripping potential prefixes
        if not os.path.isdir(table_dir) and table_name.startswith("t_"):
            table_dir = os.path.join(input_dir, table_name[2:])

        # Map to dataset name as used in lake_aggregated_results_per_table.csv
        dataset_name = table_name[2:] if table_name.startswith("t_") else table_name
        table_exec_time = execution_time_map.get(dataset_name)
        
        dirty_path = os.path.join(table_dir, "dirty.csv")
        clean_path = os.path.join(table_dir, "clean.csv")

        if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
            logger.warning("[%s] dirty.csv or clean.csv not found – skipping", table_name)
            skipped_tables.append(table_name)
            continue

        # Load provenance for this table
        provenance_map = load_provenance_map(table_dir)
        
        if not provenance_map:
            logger.warning("[%s] No provenance information found – skipping", table_name)
            skipped_tables.append(table_name)
            continue

        # Load error types if available (for effectiveness per error type)
        error_type_map = load_error_type_map(table_dir)

        # If execution_time > 1 hour, treat as if HoloClean returned no repairs:
        # we still count errors for recall, but assume no corrections (precision undefined).
        no_output_results = None
        if table_exec_time is not None and table_exec_time > _ONE_HOUR_SECONDS:
            try:
                dirty_df_no = _sanitize_column_names(read_csv(dirty_path, data_type="str"))
                clean_df_no = _sanitize_column_names(read_csv(clean_path, data_type="str"))
                clean_df_no.columns = ["index_col" if c == "index" else c for c in clean_df_no.columns]
                dirty_df_no.columns = clean_df_no.columns
                no_output_results = evaluate_with_majority_voting(
                    dirty_df_no, clean_df_no, dirty_df_no.copy(), provenance_map, error_type_map
                )
                logger.warning(
                    "[%s] execution_time=%.2fs > 1h – assuming no HoloClean output for all iterations.",
                    table_name,
                    table_exec_time,
                )
            except Exception as exc:
                logger.warning("[%s] Failed to construct no-output results: %s", table_name, exc)

        table_iter_results = []
        for entry in sorted(tables[table_name], key=lambda x: x["iter"]):
            if no_output_results is not None:
                res = no_output_results
            else:
                res = evaluate_single(
                    dirty_path, clean_path, entry["path"],
                    provenance_map, table_name, error_type_map
                )
            
            if res is None:
                skipped_tables.append(f"{table_name}_iter{entry['iter']}")
                print(f"  {table_name:<23} iter{entry['iter']:>2}  SKIPPED")
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
                f"  {table_name:<23} {entry['iter']:>4}  "
                f"{res['n_unique_errors']:>9}  {res['n_total_error_cells']:>10}  "
                f"{res['n_duplicate_errors']:>5}  {res['n_majority_voted']:>4}  "
                f"{res['n_all_corrected_errors']:>6}  {res['n_truely_corrected_errors']:>6}  "
                f"{_fmt(res['precision']):>8}  "
                f"{_fmt(res['recall']):>8}  "
                f"{_fmt(res['f1_score']):>8}"
            )

        if not table_iter_results:
            skipped_tables.append(table_name)
            continue

        tables_evaluated += 1

    # -----------------------------------------------------------------------
    # Lake-wide aggregation with majority voting
    # -----------------------------------------------------------------------

    # Per-iteration lake metrics
    iter_precisions, iter_recalls, iter_f1s = [], [], []
    for it in sorted(iter_accumulators.keys()):
        acc = iter_accumulators[it]
        tp_it  = acc["tp"]
        cor_it = acc["corrected"]
        prec_it = tp_it / cor_it if cor_it > 0 else -1.0
        rec_it  = tp_it / total_unique_errors if total_unique_errors > 0 else -1.0
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

    print("\n" + "=" * 110)
    print("LAKE-WIDE SUMMARY (with Majority Voting)")
    print("=" * 110)
    print(f"  Tables evaluated:             {tables_evaluated}")
    print(f"  Iterations evaluated:         {iters_evaluated}")
    print(f"  Skipped:                      {len(skipped_tables)}")
    print(f"  Total unique errors in lake:  {total_unique_errors}")
    print(f"  Total error cells in lake:    {total_error_cells}")
    print(f"  Duplicate error cells:        {total_error_cells - total_unique_errors}")
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

    # -----------------------------------------------------------------------
    # Effectiveness per error type (lake-wide aggregate)
    # -----------------------------------------------------------------------
    lake_by_type: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"n_unique_errors": 0, "n_truely_corrected_errors": 0, "n_all_corrected_errors": 0}
    )
    for row in per_table_rows:
        for et, metrics in row.get("by_error_type", {}).items():
            lake_by_type[et]["n_unique_errors"] += metrics["n_unique_errors"]
            lake_by_type[et]["n_truely_corrected_errors"] += metrics["n_truely_corrected_errors"]
            lake_by_type[et]["n_all_corrected_errors"] += metrics["n_all_corrected_errors"]

    effectiveness_rows: list[dict] = []
    if lake_by_type:
        # Compute precision/recall/f1 per type.
        # Use isolated error type counts as the recall denominator when available,
        # because summing n_unique_errors across merged tables double-counts source
        # errors that appear in multiple merged rows / tables (due to joins).
        effectiveness_rows = []
        for et in sorted(lake_by_type.keys()):
            acc = lake_by_type[et]
            n_unique_merged = acc["n_unique_errors"]
            n_tp = acc["n_truely_corrected_errors"]
            n_corr = acc["n_all_corrected_errors"]
            # Prefer ground-truth count from the isolated error map.
            # Keys in isolated_error_type_counts are upper-cased; normalize et for lookup.
            if isolated_error_type_counts:
                n_unique = isolated_error_type_counts.get(et.upper(), n_unique_merged)
            else:
                n_unique = n_unique_merged
            prec = n_tp / n_corr if n_corr > 0 else -1.0
            rec = n_tp / n_unique if n_unique > 0 else -1.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else -1.0
            effectiveness_rows.append({
                "error_type": et,
                "n_unique_errors": n_unique,
                "n_truely_corrected_errors": n_tp,
                "n_all_corrected_errors": n_corr,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
            })

        print("\n" + "=" * 110)
        print("EFFECTIVENESS PER ERROR TYPE (lake-wide)")
        print("=" * 110)
        print(f"  {'ERROR_TYPE':<25} {'UNIQ_ERR':>10} {'CORR':>8} {'TP':>8} {'PREC':>10} {'REC':>10} {'F1':>10}")
        print("-" * 110)
        for r in effectiveness_rows:
            print(
                f"  {r['error_type']:<25} {r['n_unique_errors']:>10} {r['n_all_corrected_errors']:>8} "
                f"{r['n_truely_corrected_errors']:>8} {_fmt(r['precision']):>10} {_fmt(r['recall']):>10} {_fmt(r['f1_score']):>10}"
            )
        print("=" * 110)

    # Optionally save results to CSV
    if save_csv and per_table_rows:
        # Per-iteration detail file
        detail_df = pd.DataFrame(per_table_rows)
        # Flatten by_error_type for detail CSV (optional: keep as-is or drop for simplicity)
        if "by_error_type" in detail_df.columns:
            detail_df = detail_df.drop(columns=["by_error_type"])
        detail_df.to_csv(save_csv, index=False)
        logger.info("Per-iteration results saved to: %s", save_csv)

        # Per-table summary file
        base, ext = os.path.splitext(save_csv)
        summary_path = f"{base}_summary{ext}"
        summary_df = pd.DataFrame(per_table_rows)
        if "by_error_type" in summary_df.columns:
            summary_df = summary_df.drop(columns=["by_error_type"])
        # Add aggregate row
        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    [
                        {
                            "table": "*** LAKE TOTAL ***",
                            "n_iters": iters_evaluated,
                            "n_unique_errors": total_unique_errors,
                            "n_total_error_cells": total_error_cells,
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

        # Effectiveness per error type CSV
        if effectiveness_rows:
            by_type_path = f"{base}_by_error_type{ext}"
            pd.DataFrame(effectiveness_rows).to_csv(by_type_path, index=False)
            logger.info("Effectiveness per error type: %s", by_type_path)


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
        description="Evaluate HoloClean repairs with majority voting for joined lake tables."
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory containing repaired_holoclean_*.csv files.",
    )
    parser.add_argument(
        "--input-dir", "-i", required=True,
        help="Root of the input data lake (contains merged table sub-directories with provenance).",
    )
    parser.add_argument(
        "--save-csv", "-s",
        help="Optional path to save per-table results as CSV.",
    )
    parser.add_argument(
        "--error-map", "-e",
        help=(
            "Path to the isolated error map CSV (e.g. datasets/tables/mit_dwh/isolated/"
            "error_map_all_tables.csv). When provided, its per-type error counts are used "
            "as the recall denominator for lake-wide per-type metrics, which avoids "
            "double-counting caused by join duplication in merged tables."
        ),
    )
    args = parser.parse_args()

    run_lake_evaluation(args.output_dir, args.input_dir, args.save_csv, args.error_map)


if __name__ == "__main__":
    main()
