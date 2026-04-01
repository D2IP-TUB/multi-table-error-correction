#!/usr/bin/env python3
"""
Script to evaluate data repair quality by comparing repaired data against 
ground truth (clean) and identifying errors from dirty data.

Uses the same conventions as utils.py for reading and comparing data.

Metrics calculated:
- Precision: Correct repairs / Total repairs made
- Recall: Correct repairs / Total errors in dirty data
- F1 Score: Harmonic mean of precision and recall
"""

import argparse
import os
import sys
from utils import read_csv, get_dataframes_difference, evaluate

import pandas as pd


def load_error_map(error_map_path: str, dirty_df: "pd.DataFrame"):
    """
    Load error_map.csv and return a dict {(row, col): error_type}.

    error_map.csv must have at least: row_number, column_name, error_type.
    row_number is 0-based (data rows, excluding the header).
    """
    if not os.path.exists(error_map_path):
        return None

    df = pd.read_csv(error_map_path, keep_default_na=False, dtype=str, encoding="latin1")
    if df.empty:
        return None

    required = {"row_number", "column_name", "error_type"}
    if not required.issubset(df.columns):
        sys.stderr.write(
            f"Warning: error_map.csv is missing required columns {required - set(df.columns)}\n"
        )
        return None

    col_name_to_idx = {name: idx for idx, name in enumerate(dirty_df.columns)}

    error_map = {}
    for _, row in df.iterrows():
        try:
            r = int(row["row_number"])
        except (ValueError, TypeError):
            continue
        col_name = str(row["column_name"]).strip()
        c = col_name_to_idx.get(col_name)
        if c is None:
            continue
        et = str(row.get("error_type", "")).strip()
        if et:
            error_map[(r, c)] = et

    return error_map


def evaluate_per_type(dirty_df: pd.DataFrame, clean_df: pd.DataFrame,
                      repaired_df: pd.DataFrame, error_map: dict):
    """
    Compute precision/recall/F1 broken down by error type.

    error_map: {(row, col): error_type}  — loaded from error_map.csv.

    For each error type:
      - ec_tpfn  = cells where dirty != clean  (actual errors, not just error_map entries)
      - ec_tpfp  = attempted corrections of that type (denominator for precision)
      - tp        = correctly repaired errors of that type
    """
    # Group error cells by error type
    type_to_cells = {}
    for (r, c), et in error_map.items():
        type_to_cells.setdefault(et, []).append((r, c))

    results_by_type = {}

    for et, cells in type_to_cells.items():
        n_errors = 0               # ec_tpfn: only cells that actually differ
        n_attempted = 0            # ec_tpfp
        n_truly_corrected = 0      # tp

        for (r, c) in cells:
            try:
                dirty_val    = dirty_df.iloc[r, c]
                clean_val    = clean_df.iloc[r, c]
                repaired_val = repaired_df.iloc[r, c]
            except IndexError:
                continue

            # Only treat as an actual error if dirty != clean (same logic as count_errors.py)
            if dirty_val == clean_val:
                continue

            n_errors += 1

            was_attempted = dirty_val != repaired_val
            is_correct = (clean_val == repaired_val or
                          (len(str(clean_val)) == 0 and len(str(repaired_val)) == 0))

            if was_attempted:
                n_attempted += 1
            if is_correct:
                n_truly_corrected += 1

        precision = n_truly_corrected / n_attempted if n_attempted > 0 else -1
        recall = n_truly_corrected / n_errors if n_errors > 0 else -1
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else -1

        results_by_type[et] = {
            "n_all_errors": n_errors,
            "n_all_corrected_errors": n_attempted,
            "n_truely_corrected_errors": n_truly_corrected,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    return results_by_type


def evaluate_repair(dirty_path: str, clean_path: str, repaired_path: str,
                    error_map_path: str = None):
    """
    Evaluate repair quality using utils.py conventions.
    
    Args:
        dirty_path: Path to dirty data CSV
        clean_path: Path to clean (ground truth) data CSV
        repaired_path: Path to repaired data CSV
        error_map_path: (Optional) Path to error_map.csv for per-type analysis
    
    Returns:
        Dictionary with evaluation metrics (and 'by_error_type' if error_map provided)
    """
    print("=" * 60)
    print("DATA REPAIR EVALUATION")
    print("=" * 60)
    
    # Load data using utils.read_csv
    print("\nLoading data...")
    dirty_df = read_csv(dirty_path)
    clean_df = read_csv(clean_path)
    repaired_df = read_csv(repaired_path)
    
    # Drop _tid_ column if present (HoloClean adds this)
    if '_tid_' in repaired_df.columns:
        repaired_df = repaired_df.drop(columns=['_tid_'])
    
    print(f"  Dirty:    {len(dirty_df)} rows, {len(dirty_df.columns)} columns")
    print(f"  Clean:    {len(clean_df)} rows, {len(clean_df.columns)} columns")
    print(f"  Repaired: {len(repaired_df)} rows, {len(repaired_df.columns)} columns")
    
    # Check shapes match
    if dirty_df.shape != clean_df.shape:
        sys.stderr.write("Warning: Dirty and clean datasets do not have equal sizes!\n")
    if dirty_df.shape != repaired_df.shape:
        sys.stderr.write("Warning: Dirty and repaired datasets do not have equal sizes!\n")
    
    # Find errors (cells that differ between dirty and clean)
    print("\nIdentifying ground truth errors (dirty vs clean)...")
    detections = get_dataframes_difference(dirty_df, clean_df)
    print(f"  Total errors found: {len(detections)}")
    
    # Overall evaluation
    print("\nEvaluating repairs...")
    results = evaluate(detections, dirty_df, clean_df, repaired_df)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<35} {'Value':<15}")
    print("-" * 50)
    print(f"{'Total errors in dirty data:':<35} {results['n_all_errors']:<15}")
    print(f"{'Total corrections attempted:':<35} {results['n_all_corrected_errors']:<15}")
    print(f"{'Correctly repaired errors:':<35} {results['n_truely_corrected_errors']:<15}")
    
    print(f"\n{'Precision:':<35} {results['precision']:.4f}")
    print(f"{'Recall:':<35} {results['recall']:.4f}")
    print(f"{'F1 Score:':<35} {results['f1_score']:.4f}")
    
    # ------------------------------------------------------------------ #
    # Per-error-type analysis
    # ------------------------------------------------------------------ #
    # Auto-detect error_map.csv next to dirty.csv if not explicitly given
    if error_map_path is None:
        auto_path = os.path.join(os.path.dirname(dirty_path), "error_map.csv")
        if os.path.exists(auto_path):
            error_map_path = auto_path

    results["by_error_type"] = {}
    if error_map_path is not None:
        print(f"\nLoading error map from: {error_map_path}")
        error_map = load_error_map(error_map_path, dirty_df)
        if error_map:
            print(f"  {len(error_map)} cells with error-type annotations found.")
            by_type = evaluate_per_type(dirty_df, clean_df, repaired_df, error_map)
            results["by_error_type"] = by_type

            print("\n" + "=" * 60)
            print("PER-ERROR-TYPE RESULTS")
            print("=" * 60)
            print(
                f"\n{'Error Type':<25} {'#Errors':>8} {'#Attempted':>11} "
                f"{'#Correct':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
            )
            print("-" * 83)
            for et, m in sorted(by_type.items()):
                prec = f"{m['precision']:.4f}" if m['precision'] >= 0 else "N/A"
                rec  = f"{m['recall']:.4f}"    if m['recall']    >= 0 else "N/A"
                f1   = f"{m['f1_score']:.4f}"  if m['f1_score']  >= 0 else "N/A"
                print(
                    f"{et:<25} {m['n_all_errors']:>8} {m['n_all_corrected_errors']:>11} "
                    f"{m['n_truely_corrected_errors']:>9} {prec:>10} {rec:>8} {f1:>8}"
                )
        else:
            print("  Warning: could not parse error_map.csv — skipping per-type analysis.")
    else:
        print("\n(No error_map.csv found — skipping per-type analysis.)")

    # ------------------------------------------------------------------ #
    # Sample errors and repairs
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SAMPLE ERRORS AND REPAIRS")
    print("=" * 60)
    
    sample_correct = []
    sample_incorrect = []
    sample_missed = []
    
    for error in detections.keys():
        row_idx, col_idx = error
        dirty_val = dirty_df.iloc[row_idx, col_idx]
        clean_val = clean_df.iloc[row_idx, col_idx]
        repaired_val = repaired_df.iloc[row_idx, col_idx]
        col_name = dirty_df.columns[col_idx]
        
        is_correct = clean_val == repaired_val or (len(str(clean_val)) == 0 and len(str(repaired_val)) == 0)
        was_repaired = dirty_val != repaired_val
        
        if is_correct and was_repaired:
            if len(sample_correct) < 5:
                sample_correct.append((row_idx, col_name, dirty_val, repaired_val, clean_val))
        elif was_repaired and not is_correct:
            if len(sample_incorrect) < 5:
                sample_incorrect.append((row_idx, col_name, dirty_val, repaired_val, clean_val))
        elif not is_correct:
            if len(sample_missed) < 5:
                sample_missed.append((row_idx, col_name, dirty_val, clean_val))
        
        if len(sample_correct) >= 5 and len(sample_incorrect) >= 5 and len(sample_missed) >= 5:
            break
    
    print("\n--- Sample Correct Repairs ---")
    for row_idx, col_name, dirty_val, repaired_val, clean_val in sample_correct:
        print(f"  [row {row_idx}, {col_name}]: '{dirty_val}' -> '{repaired_val}' (correct: '{clean_val}')")
    
    print("\n--- Sample Incorrect Repairs ---")
    for row_idx, col_name, dirty_val, repaired_val, clean_val in sample_incorrect:
        print(f"  [row {row_idx}, {col_name}]: '{dirty_val}' -> '{repaired_val}' (should be: '{clean_val}')")
    
    print("\n--- Sample Missed Errors (not attempted) ---")
    for row_idx, col_name, dirty_val, clean_val in sample_missed:
        print(f"  [row {row_idx}, {col_name}]: '{dirty_val}' (should be: '{clean_val}')")
    
    # ------------------------------------------------------------------ #
    # Per-column analysis
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("PER-COLUMN ANALYSIS")
    print("=" * 60)
    
    columns = dirty_df.columns.tolist()
    print(f"\n{'Column':<20} {'Errors':<10} {'Corrected':<12} {'Correct':<10} {'Precision':<12} {'Recall':<12}")
    print("-" * 76)
    
    for col_idx, col_name in enumerate(columns):
        col_errors = {e for e in detections.keys() if e[1] == col_idx}
        
        n_col_errors = len(col_errors)
        n_col_truly_corrected = 0
        n_col_all_corrected = 0
        
        for error in col_errors:
            row_idx = error[0]
            dirty_val = dirty_df.iloc[row_idx, col_idx]
            clean_val = clean_df.iloc[row_idx, col_idx]
            repaired_val = repaired_df.iloc[row_idx, col_idx]
            
            if clean_val == repaired_val or (len(str(clean_val)) == 0 and len(str(repaired_val)) == 0):
                n_col_truly_corrected += 1
            if dirty_val != repaired_val:
                n_col_all_corrected += 1
        
        col_precision = n_col_truly_corrected / n_col_all_corrected if n_col_all_corrected != 0 else -1
        col_recall = n_col_truly_corrected / n_col_errors if n_col_errors != 0 else -1
        
        if n_col_errors > 0:
            print(f"{col_name:<20} {n_col_errors:<10} {n_col_all_corrected:<12} {n_col_truly_corrected:<10} {col_precision:<12.4f} {col_recall:<12.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate data repair quality')
    parser.add_argument('--dirty',     '-d', required=True,  help='Path to dirty data CSV')
    parser.add_argument('--clean',     '-c', required=True,  help='Path to clean (ground truth) data CSV')
    parser.add_argument('--repaired',  '-r', required=True,  help='Path to repaired data CSV')
    parser.add_argument('--error-map', '-e', default=None,
                        help='Path to error_map.csv for per-type analysis '
                             '(auto-detected from --dirty directory if omitted)')
    
    args = parser.parse_args()
    
    results = evaluate_repair(args.dirty, args.clean, args.repaired, args.error_map)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nPrecision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")

    if results.get("by_error_type"):
        print("\nPer-error-type F1:")
        for et, m in sorted(results["by_error_type"].items()):
            f1 = f"{m['f1_score']:.4f}" if m['f1_score'] >= 0 else "N/A"
            print(f"  {et}: {f1}")


if __name__ == '__main__':
    main()
