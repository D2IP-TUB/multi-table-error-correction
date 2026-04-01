"""
Run main.py on every table in a data-lake directory, then use
evaluate_result.py functions to aggregate cell-level TP/FP/errors
across all tables for lake-level precision, recall, and F1.

Usage:
    python run_lake.py --lake_dir /path/to/flattened_partitioned_base
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import pandas as pd

# Reuse evaluate_result.py functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluate_result import normalize_value, format_empty_data


def parse_args():
    p = argparse.ArgumentParser(description="Run Uniclean on all tables in a data lake and aggregate results.")
    p.add_argument('--lake_dir', type=str, required=True,
                   help="Root directory of the lake (contains one sub-dir per table).")
    p.add_argument('--output_dir', type=str, default=None,
                   help="Where to write the aggregated evaluation. Defaults to <lake_dir>/uniclean_results/.")
    p.add_argument('--single_max', type=int, default=10000)
    p.add_argument('--timeout', type=int, default=3600,
                   help="Per-table timeout in seconds (default: 3600 = 1 hour). "
                        "The cleaning process is killed if it exceeds this limit.")
    p.add_argument('--driver_memory', type=str, default='48g',
                   help="Spark driver memory (default: 48g). Passed to main.py.")
    p.add_argument('--skip_cleaning', action='store_true',
                   help="Skip the cleaning step; only aggregate existing results.")
    return p.parse_args()


def _table_size_mb(table_dir):
    """Return the size of dirty.csv in megabytes."""
    try:
        return os.path.getsize(os.path.join(table_dir, 'dirty.csv')) / (1024 * 1024)
    except Exception:
        return 0.0


def discover_table_dirs(lake_dir):
    """Discover valid table directories and return them sorted smallest-to-largest
    by dirty.csv file size (MB)."""
    dirs = []
    for name in sorted(os.listdir(lake_dir)):
        full = os.path.join(lake_dir, name)
        if not os.path.isdir(full):
            continue
        if (os.path.isfile(os.path.join(full, 'dirty.csv'))
                and os.path.isfile(os.path.join(full, 'clean.csv'))
                and os.path.isfile(os.path.join(full, 'holo_constraints.txt'))):
            dirs.append(full)
    # Sort by file size (smallest table first)
    dirs.sort(key=_table_size_mb)
    return dirs


def ensure_index_column(csv_path):
    """If the CSV does not have an 'index' column, add one (0, 1, 2, ...)
    and rewrite the file.  Uses the csv module so no type conversion happens."""
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if 'index' in header:
            return False  # already has index
        rows = list(reader)

    header.insert(0, 'index')
    for i, row in enumerate(rows):
        row.insert(0, str(i))

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return True


def align_dirty_columns_to_clean(table_dir):
    """Rename dirty.csv column headers to match clean.csv column headers.

    The dirty files have type annotations in column names (e.g. 'objectid(long)')
    while clean files and holo_constraints use plain names (e.g. 'objectid').
    This rewrites dirty.csv's header row to match clean.csv's, using the csv
    module so no data values are converted."""
    clean_path = os.path.join(table_dir, 'clean.csv')
    dirty_path = os.path.join(table_dir, 'dirty.csv')

    with open(clean_path, 'r', newline='') as f:
        clean_header = next(csv.reader(f))
    with open(dirty_path, 'r', newline='') as f:
        reader = csv.reader(f)
        dirty_header = next(reader)
        if dirty_header == clean_header:
            return False  # already aligned
        rows = list(reader)

    if len(dirty_header) != len(clean_header):
        print(f"  WARNING: column count mismatch in {os.path.basename(table_dir)} "
              f"(dirty={len(dirty_header)}, clean={len(clean_header)}), skipping rename")
        return False

    with open(dirty_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(clean_header)
        writer.writerows(rows)
    return True


def read_csv_like_holoclean(path):
    """Read CSV with same settings as LakeCorrectionBench/HoloClean to ensure consistent error counts."""
    return pd.read_csv(path, sep=",", header="infer", encoding="latin-1", dtype=str, keep_default_na=False)


def compute_raw_counts(clean_df, dirty_df, cleaned_df):
    """Compute cell-level TP, FP, errors.
    
    - Total errors: raw string comparison (no normalization) to match LakeCorrectionBench
    - TP/FP: normalized comparison (semantic equivalence for repair evaluation)
    """
    index_col = 'index'
    
    # Ensure all dataframes have the index column
    for df in [clean_df, dirty_df, cleaned_df]:
        if index_col not in df.columns:
            df[index_col] = range(len(df))
    
    # Set index
    clean_df_raw = clean_df.set_index(index_col)
    dirty_df_raw = dirty_df.set_index(index_col)
    cleaned_df = cleaned_df.set_index(index_col)
    
    # Count total errors using RAW string comparison (no normalization)
    errors_total = 0
    for col in clean_df_raw.columns:
        errors_total += int((dirty_df_raw[col] != clean_df_raw[col]).sum())
    
    # For TP/FP, use normalized values (semantic equivalence)
    clean_df_norm = clean_df_raw.copy()
    dirty_df_norm = dirty_df_raw.copy()
    cleaned_df_norm = cleaned_df.copy()
    
    for col in clean_df_norm.columns:
        clean_df_norm[col] = clean_df_norm[col].apply(normalize_value)
        dirty_df_norm[col] = dirty_df_norm[col].apply(normalize_value)
        cleaned_df_norm[col] = cleaned_df_norm[col].apply(normalize_value)
    
    # Count TP/FP using normalized values
    tp_total, fp_total = 0, 0
    
    for col in clean_df_norm.columns:
        # True positives: correctly repaired (changed from dirty and now matches clean)
        tp_total += int(((cleaned_df_norm[col] == clean_df_norm[col]) & (dirty_df_norm[col] != cleaned_df_norm[col])).sum())
        # False positives: incorrectly changed (changed from dirty but doesn't match clean)
        fp_total += int(((cleaned_df_norm[col] != clean_df_norm[col]) & (dirty_df_norm[col] != cleaned_df_norm[col])).sum())
    
    return tp_total, fp_total, errors_total


def main():
    args = parse_args()
    lake_dir = args.lake_dir
    output_dir = args.output_dir or os.path.join(lake_dir, 'uniclean_results')
    os.makedirs(output_dir, exist_ok=True)

    table_dirs = discover_table_dirs(lake_dir)
    print(f"Discovered {len(table_dirs)} table(s) in {lake_dir}")

    # ------------------------------------------------------------------
    # Phase 0: Preprocess — align column names then ensure index column
    # ------------------------------------------------------------------
    print("Aligning dirty.csv columns to clean.csv ...")
    renamed_count = 0
    for tdir in table_dirs:
        if align_dirty_columns_to_clean(tdir):
            renamed_count += 1
    if renamed_count:
        print(f"  Renamed headers in {renamed_count} dirty.csv file(s).")
    else:
        print("  All dirty.csv headers already match clean.csv.")

    print("Checking index columns ...")
    indexed_count = 0
    for tdir in table_dirs:
        for fname in ('dirty.csv', 'clean.csv'):
            if ensure_index_column(os.path.join(tdir, fname)):
                indexed_count += 1
    if indexed_count:
        print(f"  Added index column to {indexed_count} file(s).")
    else:
        print("  All files already have an index column.")

    main_py = os.path.join(os.path.dirname(__file__), 'main.py')

    # ------------------------------------------------------------------
    # Phase 1: Clean every table by calling main.py
    # ------------------------------------------------------------------
    if not args.skip_cleaning:
        for i, tdir in enumerate(table_dirs):
            tname = os.path.basename(tdir)
            size_mb = _table_size_mb(tdir)
            log_file = os.path.join(output_dir, f'{tname}.log')
            print(f"[{i+1}/{len(table_dirs)}] Cleaning: {tname}  ({size_mb:.2f} MB, timeout={args.timeout}s)")

            cmd = [
                sys.executable, main_py,
                '--dataset_dir', tdir,
                '--table_name', tname,
                '--single_max', str(args.single_max),
                '--driver_memory', args.driver_memory,
            ]
            try:
                with open(log_file, 'w') as lf:
                    ret = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                         cwd=os.path.dirname(main_py),
                                         timeout=args.timeout)
                if ret.returncode != 0:
                    print(f"  -> FAILED (exit {ret.returncode}), see {log_file}")
                else:
                    print(f"  -> OK")
            except subprocess.TimeoutExpired:
                print(f"  -> TIMEOUT after {args.timeout}s — killed, moving to next table")
                with open(log_file, 'a') as lf:
                    lf.write(f"\n\n=== KILLED: exceeded {args.timeout}s timeout ===\n")

    # ------------------------------------------------------------------
    # Phase 2: Aggregate evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AGGREGATED EVALUATION")
    print("=" * 70)

    lake_tp, lake_fp, lake_errors = 0, 0, 0
    lake_rows = 0
    tables_ok, tables_skipped, tables_failed = 0, 0, 0
    per_table_rows = []

    for tdir in table_dirs:
        tname = os.path.basename(tdir)
        cleaned_csv = os.path.join(tdir, 'result', tname, f'{tname}Cleaned.csv')

        if not os.path.isfile(cleaned_csv):
            # No cleaned result (timeout or skipped), but still count the errors
            try:
                clean_df = read_csv_like_holoclean(os.path.join(tdir, 'clean.csv'))
                dirty_df = read_csv_like_holoclean(os.path.join(tdir, 'dirty.csv'))
                # Use dirty_df as "cleaned" to compute total errors (TP=0, FP=0)
                _, _, errors = compute_raw_counts(clean_df, dirty_df, dirty_df)
                
                lake_errors += errors
                lake_rows += len(clean_df)
                tables_skipped += 1
                
                per_table_rows.append({'table': tname, 'status': 'no_result',
                                       'TP': 0, 'FP': 0, 'errors': errors,
                                       'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
            except Exception as e:
                tables_failed += 1
                per_table_rows.append({'table': tname, 'status': f'load_error: {e}',
                                       'TP': 0, 'FP': 0, 'errors': 0,
                                       'precision': None, 'recall': None, 'f1': None})
            continue

        try:
            format_empty_data(cleaned_csv, cleaned_csv)
            clean_df = read_csv_like_holoclean(os.path.join(tdir, 'clean.csv'))
            dirty_df = read_csv_like_holoclean(os.path.join(tdir, 'dirty.csv'))
            cleaned_df = read_csv_like_holoclean(cleaned_csv)

            tp, fp, errors = compute_raw_counts(clean_df, dirty_df, cleaned_df)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / errors if errors > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            lake_tp += tp
            lake_fp += fp
            lake_errors += errors
            lake_rows += len(clean_df)
            tables_ok += 1

            per_table_rows.append({'table': tname, 'status': 'ok',
                                   'TP': tp, 'FP': fp, 'errors': errors,
                                   'precision': prec, 'recall': rec, 'f1': f1})
        except Exception as e:
            # If evaluation fails, try to at least count the errors
            try:
                clean_df = read_csv_like_holoclean(os.path.join(tdir, 'clean.csv'))
                dirty_df = read_csv_like_holoclean(os.path.join(tdir, 'dirty.csv'))
                _, _, errors = compute_raw_counts(clean_df, dirty_df, dirty_df)
                
                lake_errors += errors
                lake_rows += len(clean_df)
                tables_failed += 1
                
                per_table_rows.append({'table': tname, 'status': f'eval_error_but_counted: {str(e)[:50]}',
                                       'TP': 0, 'FP': 0, 'errors': errors,
                                       'precision': None, 'recall': None, 'f1': None})
            except Exception as e2:
                tables_failed += 1
                per_table_rows.append({'table': tname, 'status': f'eval_error: {e}',
                                       'TP': 0, 'FP': 0, 'errors': 0,
                                       'precision': None, 'recall': None, 'f1': None})

    # Lake-level metrics
    lake_precision = lake_tp / (lake_tp + lake_fp) if (lake_tp + lake_fp) > 0 else 0.0
    lake_recall = lake_tp / lake_errors if lake_errors > 0 else 0.0
    lake_f1 = 2 * lake_precision * lake_recall / (lake_precision + lake_recall) if (lake_precision + lake_recall) > 0 else 0.0

    summary = (
        f"\nLake directory : {lake_dir}\n"
        f"Tables total   : {len(table_dirs)}\n"
        f"Tables cleaned : {tables_ok}\n"
        f"Tables skipped : {tables_skipped}\n"
        f"Tables failed  : {tables_failed}\n"
        f"\n--- Lake-Level Aggregated Metrics ---\n"
        f"Total TP       : {lake_tp}\n"
        f"Total FP       : {lake_fp}\n"
        f"Total errors   : {lake_errors}\n"
        f"Precision      : {lake_precision:.6f}\n"
        f"Recall         : {lake_recall:.6f}\n"
        f"F1             : {lake_f1:.6f}\n"
        f"Total rows     : {lake_rows}\n"
    )
    print(summary)

    # Save outputs
    pd.DataFrame(per_table_rows).to_csv(os.path.join(output_dir, 'per_table_results.csv'), index=False)
    with open(os.path.join(output_dir, 'lake_evaluation.txt'), 'w') as f:
        f.write(summary)
    with open(os.path.join(output_dir, 'lake_evaluation.json'), 'w') as f:
        json.dump({'total_tp': lake_tp, 'total_fp': lake_fp, 'total_errors': lake_errors,
                   'precision': lake_precision, 'recall': lake_recall, 'f1': lake_f1,
                   'tables_cleaned': tables_ok, 'tables_skipped': tables_skipped,
                   'tables_failed': tables_failed, 'total_rows': lake_rows}, f, indent=2)

    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
