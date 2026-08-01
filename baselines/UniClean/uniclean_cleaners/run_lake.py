"""
Run main.py on every table in a data-lake directory, then aggregate metrics.

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

sys.path.insert(0, os.path.dirname(__file__))
from quintet_eval import (
    aggregate_lake_metrics,
    count_ground_truth_errors,
    ensure_index_column,
    evaluate_quintet_table,
    format_lake_summary,
    lake_evaluation_json,
    metrics_result_row,
    skipped_table_metrics,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run UniClean on all tables in a data lake and aggregate results.")
    p.add_argument('--lake_dir', type=str, required=True,
                   help="Root directory of the lake (contains one sub-dir per table).")
    p.add_argument('--output_dir', type=str, default=None,
                   help="Where to write the aggregated evaluation. Defaults to <lake_dir>/uniclean_results/.")
    p.add_argument('--single_max', type=int, default=10000)
    p.add_argument('--timeout', type=int, default=3600,
                   help="Per-table timeout in seconds (default: 3600).")
    p.add_argument('--driver_memory', type=str, default='48g',
                   help="Spark driver memory (default: 48g). Passed to main.py.")
    p.add_argument('--spark_master', type=str, default=None,
                   help="Spark master URL, e.g. 'local[16]'. Passed to main.py.")
    p.add_argument('--missing_token', type=str, default='empty',
                   help="Missing-value token for evaluation (default: empty).")
    p.add_argument('--skip_cleaning', action='store_true',
                   help="Skip the cleaning step; only aggregate existing results.")
    return p.parse_args()


def _table_size_mb(table_dir):
    try:
        return os.path.getsize(os.path.join(table_dir, 'dirty.csv')) / (1024 * 1024)
    except Exception:
        return 0.0


def discover_table_dirs(lake_dir):
    """Return table directories sorted by dirty.csv size."""
    dirs = []
    for name in sorted(os.listdir(lake_dir)):
        full = os.path.join(lake_dir, name)
        if not os.path.isdir(full):
            continue
        if (os.path.isfile(os.path.join(full, 'dirty.csv'))
                and os.path.isfile(os.path.join(full, 'clean.csv'))
                and os.path.isfile(os.path.join(full, 'holo_constraints.txt'))):
            dirs.append(full)
    dirs.sort(key=_table_size_mb)
    return dirs


def align_dirty_columns_to_clean(table_dir):
    """Rewrite dirty.csv header to match clean.csv."""
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


def main():
    args = parse_args()
    lake_dir = args.lake_dir
    output_dir = args.output_dir or os.path.join(lake_dir, 'uniclean_results')
    os.makedirs(output_dir, exist_ok=True)

    table_dirs = discover_table_dirs(lake_dir)
    print(f"Discovered {len(table_dirs)} table(s) in {lake_dir}")

    print("Aligning dirty.csv columns to clean.csv ...")
    renamed_count = sum(1 for tdir in table_dirs if align_dirty_columns_to_clean(tdir))
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
                '--missing_token', args.missing_token,
            ]
            if args.spark_master:
                cmd += ['--spark_master', args.spark_master]
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

    print("\n" + "=" * 70)
    print("AGGREGATED EVALUATION")
    print("=" * 70)

    lake_rows = 0
    tables_ok, tables_skipped, tables_failed = 0, 0, 0
    per_table_rows = []

    for tdir in table_dirs:
        tname = os.path.basename(tdir)
        cleaned_csv = os.path.join(tdir, 'result', tname, f'{tname}Cleaned.csv')
        clean_path = os.path.join(tdir, 'clean.csv')
        dirty_path = os.path.join(tdir, 'dirty.csv')

        if not os.path.isfile(cleaned_csv):
            try:
                errors = count_ground_truth_errors(clean_path, dirty_path, args.missing_token)
                lake_rows += len(pd.read_csv(clean_path, dtype=str, keep_default_na=False))
                tables_skipped += 1
                per_table_rows.append(metrics_result_row(tname, 'no_result', skipped_table_metrics(errors)))
            except Exception as e:
                tables_failed += 1
                per_table_rows.append(metrics_result_row(tname, f'load_error: {e}', {}))
            continue

        try:
            metrics = evaluate_quintet_table(
                clean_path, dirty_path, cleaned_csv,
                missing_token=args.missing_token,
            )
            lake_rows += len(pd.read_csv(clean_path, dtype=str, keep_default_na=False))
            tables_ok += 1
            per_table_rows.append(metrics_result_row(tname, 'ok', metrics))
        except Exception as e:
            try:
                errors = count_ground_truth_errors(clean_path, dirty_path, args.missing_token)
                lake_rows += len(pd.read_csv(clean_path, dtype=str, keep_default_na=False))
                tables_failed += 1
                per_table_rows.append(
                    metrics_result_row(
                        tname, f'eval_error_but_counted: {str(e)[:50]}', skipped_table_metrics(errors)
                    )
                )
            except Exception:
                tables_failed += 1
                per_table_rows.append(metrics_result_row(tname, f'eval_error: {e}', {}))

    lake = aggregate_lake_metrics(per_table_rows)
    summary = format_lake_summary(
        lake_dir, table_dirs, tables_ok, tables_skipped, tables_failed, lake, lake_rows,
    )
    print(summary)

    pd.DataFrame(per_table_rows).to_csv(os.path.join(output_dir, 'per_table_results.csv'), index=False)
    with open(os.path.join(output_dir, 'lake_evaluation.txt'), 'w') as f:
        f.write(summary)
    with open(os.path.join(output_dir, 'lake_evaluation.json'), 'w') as f:
        json.dump(lake_evaluation_json(tables_ok, tables_skipped, tables_failed, lake_rows, lake), f, indent=2)

    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
