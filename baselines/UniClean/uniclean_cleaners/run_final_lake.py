"""
Run main.py on every table in a Final_Datasets lake directory, then
aggregate cell-level TP/FP/errors across all tables for lake-level
precision, recall, and F1 — with per-partition and per-source-variant
breakdowns derived from lineage.csv.

Each table directory is expected to contain:
    dirty.csv, clean.csv, holo_constraints.txt, lineage.csv

The lineage.csv has columns:
    row_idx, source_table, source_variant, source_row_idx, partition

Usage:
    python run_final_lake.py --lake_dir /path/to/flattened_partial_overlap_50_without_duplicates
    python run_final_lake.py --lake_dir /path/to/maximal_overlap_with_duplicates --skip_cleaning
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Uniclean on all tables in a Final_Datasets lake and aggregate results."
    )
    p.add_argument('--lake_dir', type=str, required=True,
                   help="Root directory of the lake (contains one sub-dir per table).")
    p.add_argument('--output_dir', type=str, default=None,
                   help="Where to write aggregated evaluation. "
                        "Defaults to <lake_dir>/uniclean_results/.")
    p.add_argument('--single_max', type=int, default=10000)
    p.add_argument('--timeout', type=int, default=3600,
                   help="Per-table timeout in seconds (default: 3600).")
    p.add_argument('--driver_memory', type=str, default='48g',
                   help="Spark driver memory (default: 48g).")
    p.add_argument('--spark_master', type=str, default=None,
                   help="Spark master URL, e.g. 'local[16]' to limit cores per table.")
    p.add_argument('--skip_cleaning', action='store_true',
                   help="Skip cleaning; only aggregate existing results.")
    return p.parse_args()


def _table_size_mb(table_dir):
    try:
        return os.path.getsize(os.path.join(table_dir, 'dirty.csv')) / (1024 * 1024)
    except Exception:
        return 0.0


def discover_table_dirs(lake_dir):
    """Return valid table directories sorted smallest-to-largest by dirty.csv size.

    lineage.csv is optional — tables without it will still be cleaned and
    evaluated, but won't contribute to the per-partition / per-variant
    breakdowns.
    """
    dirs = []
    for name in sorted(os.listdir(lake_dir)):
        full = os.path.join(lake_dir, name)
        if not os.path.isdir(full):
            continue
        required = ['dirty.csv', 'clean.csv', 'holo_constraints.txt']
        if all(os.path.isfile(os.path.join(full, f)) for f in required):
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


_DGOV_VARIANT_RE = re.compile(r'^DGov_(FD|NO|Typo)_')


def load_lineage(table_dir):
    """Load lineage.csv if it exists; otherwise infer from the directory name.

    For DGov_{FD|NO|Typo}_* tables without lineage.csv, we synthesise a
    lineage DataFrame with partition='all' and the variant extracted from
    the name.  For other tables without lineage.csv we return None.
    """
    lineage_path = os.path.join(table_dir, 'lineage.csv')
    if os.path.isfile(lineage_path):
        return pd.read_csv(lineage_path, dtype={'row_idx': int, 'source_row_idx': int})

    tname = os.path.basename(table_dir)
    m = _DGOV_VARIANT_RE.match(tname)
    if m:
        variant = m.group(1)
        source_table = _DGOV_VARIANT_RE.sub('', tname)
        dirty_path = os.path.join(table_dir, 'dirty.csv')
        n_rows = sum(1 for _ in open(dirty_path)) - 1  # minus header
        return pd.DataFrame({
            'row_idx': range(n_rows),
            'source_table': source_table,
            'source_variant': variant,
            'source_row_idx': range(n_rows),
            'partition': 'all',
        })

    return None


def compute_raw_counts(clean_df, dirty_df, cleaned_df):
    """Cell-level TP, FP, total errors (same logic as run_lake.py)."""
    index_col = 'index'
    for df in [clean_df, dirty_df, cleaned_df]:
        if index_col not in df.columns:
            df[index_col] = range(len(df))

    clean_r = clean_df.set_index(index_col)
    dirty_r = dirty_df.set_index(index_col)
    cleaned_r = cleaned_df.set_index(index_col)

    errors_total = 0
    for col in clean_r.columns:
        errors_total += int((dirty_r[col] != clean_r[col]).sum())

    clean_n = clean_r.copy()
    dirty_n = dirty_r.copy()
    cleaned_n = cleaned_r.copy()
    for col in clean_n.columns:
        clean_n[col] = clean_n[col].apply(normalize_value)
        dirty_n[col] = dirty_n[col].apply(normalize_value)
        cleaned_n[col] = cleaned_n[col].apply(normalize_value)

    tp_total, fp_total = 0, 0
    for col in clean_n.columns:
        tp_total += int(((cleaned_n[col] == clean_n[col]) & (dirty_n[col] != cleaned_n[col])).sum())
        fp_total += int(((cleaned_n[col] != clean_n[col]) & (dirty_n[col] != cleaned_n[col])).sum())

    return tp_total, fp_total, errors_total


def compute_lineage_counts(clean_df, dirty_df, cleaned_df, lineage_df):
    """Compute TP/FP/errors broken down by partition and source_variant.

    Returns two dicts:
        by_partition:  {partition: {"tp": int, "fp": int, "errors": int}}
        by_variant:    {source_variant: {"tp": int, "fp": int, "errors": int}}
    """
    index_col = 'index'
    for df in [clean_df, dirty_df, cleaned_df]:
        if index_col not in df.columns:
            df[index_col] = range(len(df))

    clean_r = clean_df.set_index(index_col)
    dirty_r = dirty_df.set_index(index_col)
    cleaned_r = cleaned_df.set_index(index_col)

    clean_n = clean_r.copy()
    dirty_n = dirty_r.copy()
    cleaned_n = cleaned_r.copy()
    for col in clean_n.columns:
        clean_n[col] = clean_n[col].apply(normalize_value)
        dirty_n[col] = dirty_n[col].apply(normalize_value)
        cleaned_n[col] = cleaned_n[col].apply(normalize_value)

    data_cols = list(clean_r.columns)

    def _zero():
        return {"tp": 0, "fp": 0, "errors": 0}

    by_partition = defaultdict(_zero)
    by_variant = defaultdict(_zero)

    for _, lin_row in lineage_df.iterrows():
        row_idx = int(lin_row['row_idx'])
        partition = str(lin_row['partition'])
        variant = str(lin_row['source_variant'])

        if row_idx not in clean_r.index:
            continue

        for col in data_cols:
            dirty_raw = dirty_r.at[row_idx, col]
            clean_raw = clean_r.at[row_idx, col]
            is_error = (dirty_raw != clean_raw)

            dirty_norm = dirty_n.at[row_idx, col]
            clean_norm = clean_n.at[row_idx, col]
            cleaned_norm = cleaned_n.at[row_idx, col]

            changed = (dirty_norm != cleaned_norm)
            correct = (cleaned_norm == clean_norm)

            if is_error:
                by_partition[partition]["errors"] += 1
                by_variant[variant]["errors"] += 1
            if changed and correct:
                by_partition[partition]["tp"] += 1
                by_variant[variant]["tp"] += 1
            if changed and not correct:
                by_partition[partition]["fp"] += 1
                by_variant[variant]["fp"] += 1

    return dict(by_partition), dict(by_variant)


def _accumulate(target, source):
    for key, counts in source.items():
        if key not in target:
            target[key] = {"tp": 0, "fp": 0, "errors": 0}
        target[key]["tp"] += counts["tp"]
        target[key]["fp"] += counts["fp"]
        target[key]["errors"] += counts["errors"]


def _prf(tp, fp, errors):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / errors if errors > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _print_breakdown(title, breakdown):
    """Print a TP/FP/errors breakdown table."""
    print(f"\n{'=' * 85}")
    print(title)
    print('=' * 85)
    print(f"  {'Category':<30} {'errors':>9} {'TP':>9} {'FP':>9} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-' * 83}")
    for key in sorted(breakdown.keys()):
        m = breakdown[key]
        prec, rec, f1 = _prf(m["tp"], m["fp"], m["errors"])
        print(f"  {key:<30} {m['errors']:>9} {m['tp']:>9} {m['fp']:>9} "
              f"{prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")
    print('=' * 85)


def main():
    args = parse_args()
    lake_dir = args.lake_dir
    output_dir = args.output_dir or os.path.join(lake_dir, 'uniclean_results')
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

    main_py = os.path.join(os.path.dirname(__file__), 'main.py')

    # ---- Phase 1: Clean every table ----
    if not args.skip_cleaning:
        for i, tdir in enumerate(table_dirs):
            tname = os.path.basename(tdir)
            size_mb = _table_size_mb(tdir)
            log_file = os.path.join(output_dir, f'{tname}.log')
            print(f"[{i+1}/{len(table_dirs)}] Cleaning: {tname}  "
                  f"({size_mb:.2f} MB, timeout={args.timeout}s)")

            cmd = [
                sys.executable, main_py,
                '--dataset_dir', tdir,
                '--table_name', tname,
                '--single_max', str(args.single_max),
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

    lake_by_partition = {}
    lake_by_variant = {}

    for tdir in table_dirs:
        tname = os.path.basename(tdir)
        cleaned_csv = os.path.join(tdir, 'result', tname, f'{tname}Cleaned.csv')

        try:
            lineage_df = load_lineage(tdir)
        except Exception as e:
            print(f"  WARNING: could not load lineage for {tname}: {e}")
            lineage_df = None

        if not os.path.isfile(cleaned_csv):
            try:
                clean_df = read_csv_like_holoclean(os.path.join(tdir, 'clean.csv'))
                dirty_df = read_csv_like_holoclean(os.path.join(tdir, 'dirty.csv'))
                _, _, errors = compute_raw_counts(clean_df, dirty_df, dirty_df)

                lake_errors += errors
                lake_rows += len(clean_df)
                tables_skipped += 1

                if lineage_df is not None:
                    bp, bv = compute_lineage_counts(clean_df, dirty_df, dirty_df, lineage_df)
                    _accumulate(lake_by_partition, bp)
                    _accumulate(lake_by_variant, bv)

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
            clean_df = read_csv_like_holoclean(os.path.join(tdir, 'clean.csv'))
            dirty_df = read_csv_like_holoclean(os.path.join(tdir, 'dirty.csv'))
            cleaned_df = read_csv_like_holoclean(cleaned_csv)

            tp, fp, errors = compute_raw_counts(clean_df, dirty_df, cleaned_df)
            prec, rec, f1 = _prf(tp, fp, errors)

            lake_tp += tp
            lake_fp += fp
            lake_errors += errors
            lake_rows += len(clean_df)
            tables_ok += 1

            if lineage_df is not None:
                bp, bv = compute_lineage_counts(clean_df, dirty_df, cleaned_df, lineage_df)
                _accumulate(lake_by_partition, bp)
                _accumulate(lake_by_variant, bv)

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
                lake_rows += len(clean_df)
                tables_failed += 1

                if lineage_df is not None:
                    bp, bv = compute_lineage_counts(clean_df, dirty_df, dirty_df, lineage_df)
                    _accumulate(lake_by_partition, bp)
                    _accumulate(lake_by_variant, bv)

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

    if lake_by_partition:
        _print_breakdown("PER-PARTITION LAKE-WIDE RESULTS", lake_by_partition)
    else:
        print("\n(No lineage partition data available.)")

    if lake_by_variant:
        _print_breakdown("PER-SOURCE-VARIANT LAKE-WIDE RESULTS", lake_by_variant)
    else:
        print("\n(No lineage source-variant data available.)")

    # ---- Save outputs ----
    pd.DataFrame(per_table_rows).to_csv(
        os.path.join(output_dir, 'per_table_results.csv'), index=False
    )

    with open(os.path.join(output_dir, 'lake_evaluation.txt'), 'w') as f:
        f.write(summary)

    def _breakdown_to_list(breakdown):
        rows = []
        for key in sorted(breakdown.keys()):
            m = breakdown[key]
            prec, rec, f1 = _prf(m["tp"], m["fp"], m["errors"])
            rows.append({
                "category": key, "errors": m["errors"],
                "tp": m["tp"], "fp": m["fp"],
                "precision": prec, "recall": rec, "f1": f1,
            })
        return rows

    partition_rows = _breakdown_to_list(lake_by_partition) if lake_by_partition else []
    variant_rows = _breakdown_to_list(lake_by_variant) if lake_by_variant else []

    with open(os.path.join(output_dir, 'lake_evaluation.json'), 'w') as f:
        json.dump({
            'total_tp': lake_tp, 'total_fp': lake_fp,
            'total_errors': lake_errors,
            'precision': lake_prec, 'recall': lake_rec, 'f1': lake_f1,
            'tables_cleaned': tables_ok, 'tables_skipped': tables_skipped,
            'tables_failed': tables_failed, 'total_rows': lake_rows,
            'per_partition': partition_rows,
            'per_source_variant': variant_rows,
        }, f, indent=2)

    if partition_rows:
        pd.DataFrame(partition_rows).to_csv(
            os.path.join(output_dir, 'per_partition_results.csv'), index=False
        )
    if variant_rows:
        pd.DataFrame(variant_rows).to_csv(
            os.path.join(output_dir, 'per_variant_results.csv'), index=False
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
