"""
Run main_quintet3.py on every table in the Quintet_3 dataset directory, then
aggregate cell-level TP/FP/errors for lake-level precision, recall, and F1.

Unlike run_final_lake.py, this script does NOT use holo_constraints.txt.
Instead, FD rules are hard-coded in main_quintet3.py.

Each table directory is expected to contain:
    dirty.csv, clean.csv

Usage:
    # run cleaning + evaluation
    python run_quintet3.py

    # point to a custom location
    python run_quintet3.py --lake_dir /other/path/to/Quintet_3

    # skip cleaning, only re-aggregate existing results
    python run_quintet3.py --skip_cleaning
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluate_result import normalize_value, format_empty_data

_DEFAULT_LAKE_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'datasets_and_rules', 'Quintet_3'
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Uniclean on all Quintet-3 tables using paper-defined FD rules."
    )
    p.add_argument('--lake_dir', type=str, default=_DEFAULT_LAKE_DIR,
                   help="Root directory of the Quintet_3 dataset. "
                        f"Default: {_DEFAULT_LAKE_DIR}")
    p.add_argument('--output_dir', type=str, default=None,
                   help="Where to write aggregated evaluation. "
                        "Defaults to <lake_dir>/uni_clean_results/.")
    p.add_argument('--single_max', type=int, default=10000)
    p.add_argument('--timeout', type=int, default=3600,
                   help="Per-table timeout in seconds (default: 3600).")
    p.add_argument('--driver_memory', type=str, default='48g',
                   help="Spark driver memory (default: 48g).")
    p.add_argument('--spark_master', type=str, default=None,
                   help="Spark master URL, e.g. 'local[16]'.")
    p.add_argument('--skip_cleaning', action='store_true',
                   help="Skip cleaning; only aggregate existing results.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Known Quintet-3 table names (must match keys in main_quintet3.QUINTET3_CLEANERS)
# ---------------------------------------------------------------------------
KNOWN_TABLES = {'hospital', 'flights', 'beers', 'rayyan', 'movies_1'}


def _table_size_mb(table_dir):
    try:
        return os.path.getsize(os.path.join(table_dir, 'dirty.csv')) / (1024 * 1024)
    except Exception:
        return 0.0


def discover_table_dirs(lake_dir):
    """Return valid Quintet-3 table directories sorted smallest-to-largest.

    A directory is valid if it contains dirty.csv and clean.csv and its
    name matches a known Quintet-3 table (holo_constraints.txt is NOT
    required — rules are hard-coded instead).
    """
    dirs = []
    for name in sorted(os.listdir(lake_dir)):
        full = os.path.join(lake_dir, name)
        if not os.path.isdir(full):
            continue
        if name.lower() not in KNOWN_TABLES:
            continue
        if all(os.path.isfile(os.path.join(full, f)) for f in ('dirty.csv', 'clean.csv')):
            dirs.append(full)
    dirs.sort(key=_table_size_mb)
    return dirs


def ensure_index_column(csv_path):
    """Add a 0-based 'index' column if missing. Returns True if added."""
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if 'index' in header:
            return False
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
    """Rewrite dirty.csv header to match clean.csv (strip type annotations)."""
    clean_path = os.path.join(table_dir, 'clean.csv')
    dirty_path = os.path.join(table_dir, 'dirty.csv')

    with open(clean_path, 'r', newline='') as f:
        clean_header = next(csv.reader(f))
    with open(dirty_path, 'r', newline='') as f:
        reader = csv.reader(f)
        dirty_header = next(reader)
        if dirty_header == clean_header:
            return False
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
    return pd.read_csv(
        path, sep=",", header="infer", encoding="latin-1",
        dtype=str, keep_default_na=False,
    )


def compute_raw_counts(clean_df, dirty_df, cleaned_df):
    """Cell-level TP, FP, total errors."""
    index_col = 'index'
    for df in [clean_df, dirty_df, cleaned_df]:
        if index_col not in df.columns:
            df[index_col] = range(len(df))

    clean_r   = clean_df.set_index(index_col)
    dirty_r   = dirty_df.set_index(index_col)
    cleaned_r = cleaned_df.set_index(index_col)

    errors_total = 0
    for col in clean_r.columns:
        errors_total += int((dirty_r[col] != clean_r[col]).sum())

    clean_n   = clean_r.copy()
    dirty_n   = dirty_r.copy()
    cleaned_n = cleaned_r.copy()
    for col in clean_n.columns:
        clean_n[col]   = clean_n[col].apply(normalize_value)
        dirty_n[col]   = dirty_n[col].apply(normalize_value)
        cleaned_n[col] = cleaned_n[col].apply(normalize_value)

    tp_total, fp_total = 0, 0
    for col in clean_n.columns:
        tp_total += int(((cleaned_n[col] == clean_n[col]) & (dirty_n[col] != cleaned_n[col])).sum())
        fp_total += int(((cleaned_n[col] != clean_n[col]) & (dirty_n[col] != cleaned_n[col])).sum())

    return tp_total, fp_total, errors_total


def _prf(tp, fp, errors):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / errors if errors > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def main():
    args     = parse_args()
    lake_dir = os.path.realpath(args.lake_dir)
    output_dir = args.output_dir or os.path.join(lake_dir, 'uni_clean_results')
    os.makedirs(output_dir, exist_ok=True)

    table_dirs = discover_table_dirs(lake_dir)
    print(f"Discovered {len(table_dirs)} table(s) in {lake_dir}")

    # ---- Phase 0: Preprocess ----
    print("Aligning dirty.csv columns to clean.csv ...")
    renamed = sum(1 for t in table_dirs if align_dirty_columns_to_clean(t))
    print(f"  Renamed headers in {renamed} file(s)." if renamed
          else "  All headers already aligned.")

    print("Checking index columns ...")
    indexed = 0
    for tdir in table_dirs:
        for fname in ('dirty.csv', 'clean.csv'):
            if ensure_index_column(os.path.join(tdir, fname)):
                indexed += 1
    print(f"  Added index column to {indexed} file(s)." if indexed
          else "  All files already have an index column.")

    main_py = os.path.join(os.path.dirname(__file__), 'main_quintet3.py')

    # ---- Phase 1: Clean every table ----
    if not args.skip_cleaning:
        for i, tdir in enumerate(table_dirs):
            tname    = os.path.basename(tdir)
            size_mb  = _table_size_mb(tdir)
            log_file = os.path.join(output_dir, f'{tname}.log')
            print(f"[{i+1}/{len(table_dirs)}] Cleaning: {tname}  "
                  f"({size_mb:.2f} MB, timeout={args.timeout}s)")

            cmd = [
                sys.executable, main_py,
                '--dataset_dir', tdir,
                '--table_name',  tname,
                '--single_max',  str(args.single_max),
                '--driver_memory', args.driver_memory,
            ]
            if args.spark_master:
                cmd += ['--spark_master', args.spark_master]

            try:
                with open(log_file, 'w') as lf:
                    ret = subprocess.run(
                        cmd, stdout=lf, stderr=subprocess.STDOUT,
                        cwd=os.path.dirname(main_py), timeout=args.timeout,
                    )
                if ret.returncode != 0:
                    print(f"  -> FAILED (exit {ret.returncode}), see {log_file}")
                else:
                    print(f"  -> OK")
            except subprocess.TimeoutExpired:
                print(f"  -> TIMEOUT after {args.timeout}s — killed")
                with open(log_file, 'a') as lf:
                    lf.write(f"\n\n=== KILLED: exceeded {args.timeout}s timeout ===\n")

    # ---- Phase 2: Aggregate evaluation ----
    print("\n" + "=" * 70)
    print("AGGREGATED EVALUATION")
    print("=" * 70)

    lake_tp, lake_fp, lake_errors = 0, 0, 0
    lake_rows = 0
    tables_ok, tables_skipped, tables_failed = 0, 0, 0
    per_table_rows = []

    for tdir in table_dirs:
        tname       = os.path.basename(tdir)
        cleaned_csv = os.path.join(tdir, 'result', tname, f'{tname}Cleaned.csv')

        if not os.path.isfile(cleaned_csv):
            try:
                clean_df = read_csv_like_holoclean(os.path.join(tdir, 'clean.csv'))
                dirty_df = read_csv_like_holoclean(os.path.join(tdir, 'dirty.csv'))
                _, _, errors = compute_raw_counts(clean_df, dirty_df, dirty_df)

                lake_errors += errors
                lake_rows   += len(clean_df)
                tables_skipped += 1

                per_table_rows.append({
                    'table': tname, 'status': 'no_result',
                    'TP': 0, 'FP': 0, 'errors': errors,
                    'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                })
            except Exception as e:
                tables_failed += 1
                per_table_rows.append({
                    'table': tname, 'status': f'load_error: {e}',
                    'TP': 0, 'FP': 0, 'errors': 0,
                    'precision': None, 'recall': None, 'f1': None,
                })
            continue

        try:
            format_empty_data(cleaned_csv, cleaned_csv)
            clean_df   = read_csv_like_holoclean(os.path.join(tdir, 'clean.csv'))
            dirty_df   = read_csv_like_holoclean(os.path.join(tdir, 'dirty.csv'))
            cleaned_df = read_csv_like_holoclean(cleaned_csv)

            tp, fp, errors = compute_raw_counts(clean_df, dirty_df, cleaned_df)
            prec, rec, f1  = _prf(tp, fp, errors)

            lake_tp     += tp
            lake_fp     += fp
            lake_errors += errors
            lake_rows   += len(clean_df)
            tables_ok   += 1

            per_table_rows.append({
                'table': tname, 'status': 'ok',
                'TP': tp, 'FP': fp, 'errors': errors,
                'precision': prec, 'recall': rec, 'f1': f1,
            })
        except Exception as e:
            try:
                clean_df = read_csv_like_holoclean(os.path.join(tdir, 'clean.csv'))
                dirty_df = read_csv_like_holoclean(os.path.join(tdir, 'dirty.csv'))
                _, _, errors = compute_raw_counts(clean_df, dirty_df, dirty_df)

                lake_errors += errors
                lake_rows   += len(clean_df)
                tables_failed += 1

                per_table_rows.append({
                    'table': tname,
                    'status': f'eval_error_but_counted: {str(e)[:50]}',
                    'TP': 0, 'FP': 0, 'errors': errors,
                    'precision': None, 'recall': None, 'f1': None,
                })
            except Exception:
                tables_failed += 1
                per_table_rows.append({
                    'table': tname, 'status': f'eval_error: {e}',
                    'TP': 0, 'FP': 0, 'errors': 0,
                    'precision': None, 'recall': None, 'f1': None,
                })

    lake_prec, lake_rec, lake_f1 = _prf(lake_tp, lake_fp, lake_errors)

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
        f"Precision      : {lake_prec:.6f}\n"
        f"Recall         : {lake_rec:.6f}\n"
        f"F1             : {lake_f1:.6f}\n"
        f"Total rows     : {lake_rows}\n"
    )
    print(summary)

    # ---- Save outputs ----
    pd.DataFrame(per_table_rows).to_csv(
        os.path.join(output_dir, 'per_table_results.csv'), index=False
    )

    with open(os.path.join(output_dir, 'lake_evaluation.txt'), 'w') as f:
        f.write(summary)

    with open(os.path.join(output_dir, 'lake_evaluation.json'), 'w') as f:
        json.dump({
            'total_tp': lake_tp, 'total_fp': lake_fp,
            'total_errors': lake_errors,
            'precision': lake_prec, 'recall': lake_rec, 'f1': lake_f1,
            'tables_cleaned': tables_ok, 'tables_skipped': tables_skipped,
            'tables_failed': tables_failed, 'total_rows': lake_rows,
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
