#!/usr/bin/env python3
"""
Evaluate Uniclean results per error type (FD, NO, Typo) for a data lake,
using row-level lineage when available.

Mirrors the logic of the Horizon per-error-type evaluation script so that
results are directly comparable.

For each table directory:
  1. If lineage.csv exists, use it to determine the error type of each ROW,
     then bucket cell-level errors/corrections into that row's error type.
  2. Otherwise, fall back to extracting the error type from the folder name
     (DGov_FD_*, DGov_NO_*, DGov_Typo_*).

Uniclean result path convention:
    <table_dir>/result/<table_name>/<table_name>Cleaned.csv

Outputs:
  - uniclean_by_error_type.csv            (summary per error type + overall)
  - uniclean_by_error_type_per_table.csv  (per-table breakdown)
"""
import os
import sys
import csv
import argparse
from collections import defaultdict
import pandas as pd


def read_csv(path):
    """Read CSV with LakeCorrectionBench-compatible settings."""
    return pd.read_csv(
        path, sep=",", header="infer", encoding="latin-1",
        dtype=str, keep_default_na=False,
    )


def get_error_type_from_folder(folder_name):
    """Extract error type from folder name."""
    name = folder_name.upper()
    if '_FD_' in name or name.startswith('DGOV_FD'):
        return 'FD'
    if '_NO_' in name or name.startswith('DGOV_NO'):
        return 'NO'
    if '_TYPO_' in name or name.startswith('DGOV_TYPO'):
        return 'TYPO'
    return None


def load_lineage(table_dir):
    """
    Load lineage.csv and return a dict mapping row_idx -> error type.
    Returns None if lineage.csv does not exist.
    """
    lineage_path = os.path.join(table_dir, 'lineage.csv')
    if not os.path.exists(lineage_path):
        return None

    row_to_type = {}
    with open(lineage_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_idx = int(row['row_idx'])
            variant = row.get('source_variant', '').strip().upper()
            if variant in ('FD', 'NO', 'TYPO'):
                row_to_type[row_idx] = variant
            else:
                row_to_type[row_idx] = 'UNKNOWN'
    return row_to_type


def find_tables(root):
    """Yield (dirpath, dirname) for table directories with dirty.csv and clean.csv."""
    for entry in sorted(os.listdir(root)):
        dirpath = os.path.join(root, entry)
        if not os.path.isdir(dirpath):
            continue
        files = set(os.listdir(dirpath))
        if 'dirty.csv' in files and 'clean.csv' in files:
            yield dirpath, entry


def find_repaired_csv(table_dir, dirname):
    """Locate the Uniclean cleaned CSV.

    Convention: <table_dir>/result/<dirname>/<dirname>Cleaned.csv
    """
    return os.path.join(table_dir, 'result', dirname, f'{dirname}Cleaned.csv')


def format_empty_data(df):
    """Normalise missing-value representations to empty string for comparison."""
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype(str)
    df.replace(['nan', 'null', '__NULL__', 'empty', 'None'], '', inplace=True)
    return df


def count_gt_errors(table_dir, dirname):
    """
    Count ground-truth errors (dirty != clean) per error type.
    Always succeeds as long as dirty.csv and clean.csv exist.

    Returns dict of {error_type: error_count} and (n_rows, n_cols),
    or None if files can't be read.
    """
    dirty_path = os.path.join(table_dir, 'dirty.csv')
    clean_path = os.path.join(table_dir, 'clean.csv')

    try:
        dirty_df = read_csv(dirty_path)
        clean_df = read_csv(clean_path)
    except Exception:
        return None

    if len(dirty_df.columns) == len(clean_df.columns):
        dirty_df.columns = clean_df.columns

    drop_cols = {'index', '_tid_'}
    for col in drop_cols:
        if col in dirty_df.columns:
            dirty_df = dirty_df.drop(columns=[col])
        if col in clean_df.columns:
            clean_df = clean_df.drop(columns=[col])

    min_rows = min(len(dirty_df), len(clean_df))
    min_cols = min(len(dirty_df.columns), len(clean_df.columns))
    if min_rows == 0 or min_cols == 0:
        return None

    dirty_df = dirty_df.iloc[:min_rows, :min_cols].reset_index(drop=True)
    clean_df = clean_df.iloc[:min_rows, :min_cols].reset_index(drop=True)

    row_to_type = load_lineage(table_dir)
    if row_to_type is None:
        folder_type = get_error_type_from_folder(dirname) or 'UNKNOWN'
        row_to_type = {i: folder_type for i in range(min_rows)}

    type_errors = defaultdict(int)
    for i in range(min_rows):
        error_type = row_to_type.get(i, 'UNKNOWN')
        for j in range(min_cols):
            if str(dirty_df.iloc[i, j]) != str(clean_df.iloc[i, j]):
                type_errors[error_type] += 1

    return dict(type_errors), min_rows, min_cols


def evaluate_table(table_dir, dirname):
    """
    Evaluate a single table with per-error-type granularity.

    Metric definitions:
      tpfn = total error cells in ground truth (dirty != clean)
      tpfp = error cells where a change was attempted (dirty != repaired)
      tp   = error cells changed AND repaired matches clean

    Only error cells that were actually changed contribute to TP/FP.
    Error counting uses raw string comparison (matching count_errors.py).

    Returns (dict of {error_type: counts}, n_rows, n_cols, has_repaired) or None.
    """
    dirty_path = os.path.join(table_dir, 'dirty.csv')
    clean_path = os.path.join(table_dir, 'clean.csv')
    repaired_path = find_repaired_csv(table_dir, dirname)

    # Always count GT errors first
    gt_result = count_gt_errors(table_dir, dirname)
    if gt_result is None:
        return None
    gt_errors, n_rows, n_cols = gt_result

    type_counts = defaultdict(lambda: {'tp': 0, 'tpfp': 0, 'tpfn': 0})

    # Seed tpfn from ground truth
    for error_type, count in gt_errors.items():
        type_counts[error_type]['tpfn'] = count

    has_repaired = os.path.exists(repaired_path)
    if not has_repaired:
        return dict(type_counts), n_rows, n_cols, False

    try:
        dirty_df = read_csv(dirty_path)
        clean_df = read_csv(clean_path)
        repaired_df = read_csv(repaired_path)
        repaired_df = format_empty_data(repaired_df)
    except Exception:
        return dict(type_counts), n_rows, n_cols, False

    if len(dirty_df.columns) == len(clean_df.columns):
        dirty_df.columns = clean_df.columns

    drop_cols = {'index', '_tid_'}
    for col in drop_cols:
        if col in dirty_df.columns:
            dirty_df = dirty_df.drop(columns=[col])
        if col in clean_df.columns:
            clean_df = clean_df.drop(columns=[col])
        if col in repaired_df.columns:
            repaired_df = repaired_df.drop(columns=[col])

    min_rows = min(len(dirty_df), len(clean_df), len(repaired_df))
    min_cols = min(len(dirty_df.columns), len(clean_df.columns), len(repaired_df.columns))
    if min_rows == 0 or min_cols == 0:
        return dict(type_counts), n_rows, n_cols, False

    dirty_df = dirty_df.iloc[:min_rows, :min_cols].reset_index(drop=True)
    clean_df = clean_df.iloc[:min_rows, :min_cols].reset_index(drop=True)
    repaired_df = repaired_df.iloc[:min_rows, :min_cols].reset_index(drop=True)
    repaired_df.columns = clean_df.columns

    row_to_type = load_lineage(table_dir)
    if row_to_type is None:
        folder_type = get_error_type_from_folder(dirname) or 'UNKNOWN'
        row_to_type = {i: folder_type for i in range(min_rows)}

    for i in range(min_rows):
        error_type = row_to_type.get(i, 'UNKNOWN')
        for j in range(min_cols):
            dirty_val = str(dirty_df.iloc[i, j])
            clean_val = str(clean_df.iloc[i, j])
            repaired_val = str(repaired_df.iloc[i, j])

            is_error = dirty_val != clean_val
            was_changed = dirty_val != repaired_val

            if is_error and was_changed:
                type_counts[error_type]['tpfp'] += 1
                if clean_val == repaired_val:
                    type_counts[error_type]['tp'] += 1

    return dict(type_counts), n_rows, n_cols, True


def compute_metrics(tp, tpfp, tpfn):
    precision = tp / tpfp if tpfp > 0 else -1
    recall = tp / tpfn if tpfn > 0 else -1
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else -1
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Uniclean per error type using row-level lineage.'
    )
    parser.add_argument(
        'lake_dir',
        help='Path to the lake directory containing table folders'
    )
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: <lake_dir>/uniclean_by_error_type.csv)')
    args = parser.parse_args()

    lake_dir = args.lake_dir
    if not os.path.isdir(lake_dir):
        print(f"Error: '{lake_dir}' is not a directory.")
        sys.exit(1)

    output_csv = args.output or os.path.join(lake_dir, 'uniclean_by_error_type.csv')

    type_totals = defaultdict(lambda: {'tp': 0, 'tpfp': 0, 'tpfn': 0, 'tables': 0})
    per_table_rows = []
    tables_with_result = 0
    tables_without_result = 0
    tables_failed = 0

    tables = list(find_tables(lake_dir))
    print(f"Found {len(tables)} table directories in {os.path.basename(lake_dir)}\n")

    for table_dir, dirname in tables:
        result = evaluate_table(table_dir, dirname)
        if result is None:
            tables_failed += 1
            continue

        type_counts, n_rows, n_cols, has_repaired = result
        if has_repaired:
            tables_with_result += 1
        else:
            tables_without_result += 1

        table_total_tp = 0
        table_total_tpfp = 0
        table_total_tpfn = 0

        for error_type, counts in type_counts.items():
            type_totals[error_type]['tp'] += counts['tp']
            type_totals[error_type]['tpfp'] += counts['tpfp']
            type_totals[error_type]['tpfn'] += counts['tpfn']
            type_totals[error_type]['tables'] += 1
            table_total_tp += counts['tp']
            table_total_tpfp += counts['tpfp']
            table_total_tpfn += counts['tpfn']

        p, r, f = compute_metrics(table_total_tp, table_total_tpfp, table_total_tpfn)
        types_present = ', '.join(sorted(type_counts.keys()))
        status = 'ok' if has_repaired else 'no_result'
        per_table_rows.append({
            'table': dirname,
            'status': status,
            'error_types': types_present,
            'tp': table_total_tp,
            'tpfp': table_total_tpfp,
            'tpfn': table_total_tpfn,
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1': round(f, 4),
            'rows': n_rows,
            'cols': n_cols,
        })

    print(f"Tables with Uniclean result: {tables_with_result}")
    print(f"Tables without result (errors still counted): {tables_without_result}")
    if tables_failed:
        print(f"Tables failed to load: {tables_failed}")
    print()

    print("=" * 72)
    print(f"{'Error Type':<12} {'Tables':>7} {'TP':>8} {'TP+FP':>8} {'TP+FN':>8}   {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 72)

    grand_tp, grand_tpfp, grand_tpfn = 0, 0, 0
    summary_rows = []

    for error_type in ['FD', 'NO', 'TYPO', 'UNKNOWN']:
        if error_type not in type_totals:
            continue
        t = type_totals[error_type]
        p, r, f = compute_metrics(t['tp'], t['tpfp'], t['tpfn'])
        print(f"{error_type:<12} {t['tables']:>7}  {t['tp']:>7} {t['tpfp']:>8} {t['tpfn']:>8}   {p:>8.4f} {r:>8.4f} {f:>8.4f}")

        grand_tp += t['tp']
        grand_tpfp += t['tpfp']
        grand_tpfn += t['tpfn']

        summary_rows.append({
            'error_type': error_type,
            'tables': t['tables'],
            'tp': t['tp'],
            'tpfp': t['tpfp'],
            'tpfn': t['tpfn'],
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1': round(f, 4),
        })

    print("-" * 72)
    overall_p, overall_r, overall_f = compute_metrics(grand_tp, grand_tpfp, grand_tpfn)
    total_tables = len(tables) - tables_failed
    print(f"{'OVERALL':<12} {total_tables:>7}  {grand_tp:>7} {grand_tpfp:>8} {grand_tpfn:>8}   {overall_p:>8.4f} {overall_r:>8.4f} {overall_f:>8.4f}")
    print("=" * 72)

    summary_rows.append({
        'error_type': 'OVERALL',
        'tables': total_tables,
        'tp': grand_tp,
        'tpfp': grand_tpfp,
        'tpfn': grand_tpfn,
        'precision': round(overall_p, 4),
        'recall': round(overall_r, 4),
        'f1': round(overall_f, 4),
    })

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'error_type', 'tables', 'tp', 'tpfp', 'tpfn', 'precision', 'recall', 'f1'
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to: {output_csv}")

    per_table_csv = output_csv.replace('.csv', '_per_table.csv')
    with open(per_table_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'table', 'status', 'error_types', 'tp', 'tpfp', 'tpfn', 'precision', 'recall', 'f1', 'rows', 'cols'
        ])
        writer.writeheader()
        writer.writerows(per_table_rows)
    print(f"Per-table results saved to: {per_table_csv}")


if __name__ == '__main__':
    main()
