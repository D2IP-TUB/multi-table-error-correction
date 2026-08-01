"""
Run main_quintet3.py on every Quintet-3 table and aggregate metrics.

Usage:
    python run_quintet3.py --mode all
    python run_quintet3.py --mode fd --lake_dir /path/to/Quintet_3
    python run_quintet3.py --mode all --skip_cleaning
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

_DEFAULT_LAKE_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'datasets_and_rules', 'Quintet_3'
)
VALID_MODES = ('all', 'fd')


def parse_args():
    p = argparse.ArgumentParser(
        description="Run UniClean on all Quintet-3 tables."
    )
    p.add_argument('--lake_dir', type=str, default=_DEFAULT_LAKE_DIR,
                   help="Root directory of the Quintet_3 dataset. "
                        f"Default: {_DEFAULT_LAKE_DIR}")
    p.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=list(VALID_MODES),
        help=(
            "Cleaner mode passed to main_quintet3.py: "
            "'all' = UniClean-ALL, 'fd' = UniClean-FD (AttrRelation only)."
        ),
    )
    p.add_argument('--output_dir', type=str, default=None,
                   help="Where to write aggregated evaluation. "
                        "Defaults to <lake_dir>/uni_clean_results_<mode>/.")
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


# Known Quintet-3 table names (must match main_quintet3.OFFICIAL_CLEANERS keys).
KNOWN_TABLES = {'hospital', 'flights', 'beers', 'rayyan', 'movies_1', 'movies'}

TABLE_MISSING_TOKEN = {
    'hospital': 'empty',
    'flights': 'empty',
    'beers': 'empty',
    'rayyan': 'empty',
    'movies_1': 'empty',
    'movies': 'empty',
}


def _table_size_mb(table_dir):
    try:
        return os.path.getsize(os.path.join(table_dir, 'dirty.csv')) / (1024 * 1024)
    except Exception:
        return 0.0


def discover_table_dirs(lake_dir):
    """Return Quintet-3 table directories sorted by dirty.csv size."""
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


def _aggregate_lake_metrics(per_table_metrics: list) -> dict:
    return aggregate_lake_metrics(per_table_metrics)


def _metrics_row(tname: str, status: str, metrics: dict) -> dict:
    return metrics_result_row(tname, status, metrics)


def _cleaned_csv_path(tdir, tname, mode):
    """Prefer result/<mode>/<table>/…; fall back to legacy result/<table>/… for mode=all."""
    modern = os.path.join(tdir, 'result', mode, tname, f'{tname}Cleaned.csv')
    if os.path.isfile(modern):
        return modern
    if mode == 'all':
        legacy = os.path.join(tdir, 'result', tname, f'{tname}Cleaned.csv')
        if os.path.isfile(legacy):
            return legacy
    return modern


def main():
    args     = parse_args()
    lake_dir = os.path.realpath(args.lake_dir)
    mode     = args.mode
    output_dir = args.output_dir or os.path.join(lake_dir, f'uni_clean_results_{mode}')
    os.makedirs(output_dir, exist_ok=True)

    table_dirs = discover_table_dirs(lake_dir)
    print(f"Discovered {len(table_dirs)} table(s) in {lake_dir}")
    print(f"Mode: {mode} ({'UniClean-ALL' if mode == 'all' else 'UniClean-FD'})")

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
                  f"({size_mb:.2f} MB, mode={mode}, timeout={args.timeout}s)")

            cmd = [
                sys.executable, main_py,
                '--dataset_dir', tdir,
                '--table_name',  tname,
                '--mode', mode,
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

    # Phase 2: aggregate evaluation
    print("\n" + "=" * 70)
    print(f"AGGREGATED EVALUATION  (mode={mode})")
    print("=" * 70)

    lake_rows = 0
    tables_ok, tables_skipped, tables_failed = 0, 0, 0
    per_table_rows = []

    for tdir in table_dirs:
        tname = os.path.basename(tdir)
        cleaned_csv = _cleaned_csv_path(tdir, tname, mode)
        clean_path = os.path.join(tdir, 'clean.csv')
        dirty_path = os.path.join(tdir, 'dirty.csv')
        missing_token = TABLE_MISSING_TOKEN.get(tname.lower(), 'empty')

        if not os.path.isfile(cleaned_csv):
            try:
                errors = count_ground_truth_errors(clean_path, dirty_path, missing_token)
                lake_rows += len(pd.read_csv(clean_path, dtype=str, keep_default_na=False))
                tables_skipped += 1
                row = _metrics_row(tname, 'no_result', skipped_table_metrics(errors))
                per_table_rows.append(row)
            except Exception as e:
                tables_failed += 1
                per_table_rows.append(_metrics_row(tname, f'load_error: {e}', {}))
            continue

        try:
            metrics = evaluate_quintet_table(
                clean_path, dirty_path, cleaned_csv,
                missing_token=missing_token,
            )
            lake_rows += len(pd.read_csv(clean_path, dtype=str, keep_default_na=False))
            tables_ok += 1
            row = _metrics_row(tname, 'ok', metrics)
            per_table_rows.append(row)
        except Exception as e:
            try:
                errors = count_ground_truth_errors(clean_path, dirty_path, missing_token)
                lake_rows += len(pd.read_csv(clean_path, dtype=str, keep_default_na=False))
                tables_failed += 1
                row = _metrics_row(tname, f'eval_error_but_counted: {str(e)[:50]}', {'errors': errors})
                per_table_rows.append(row)
            except Exception:
                tables_failed += 1
                per_table_rows.append(_metrics_row(tname, f'eval_error: {e}', {}))

    lake = _aggregate_lake_metrics(per_table_rows)

    summary = format_lake_summary(
        lake_dir, table_dirs, tables_ok, tables_skipped, tables_failed, lake, lake_rows,
    )
    print(summary)

    # ---- Save outputs ----
    pd.DataFrame(per_table_rows).to_csv(
        os.path.join(output_dir, 'per_table_results.csv'), index=False
    )

    with open(os.path.join(output_dir, 'lake_evaluation.txt'), 'w') as f:
        f.write(f"mode={mode}\n")
        f.write(summary)

    with open(os.path.join(output_dir, 'lake_evaluation.json'), 'w') as f:
        payload = lake_evaluation_json(tables_ok, tables_skipped, tables_failed, lake_rows, lake)
        payload['mode'] = mode
        json.dump(payload, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
