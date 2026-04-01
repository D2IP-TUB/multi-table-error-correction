#!/usr/bin/env python3
"""
Evaluate Horizon results per error type (FD, NO, Typo) for a data lake,
using row-level lineage when available.

For each table directory:
  1. If lineage.csv exists, use it to determine the error type of each ROW,
     then bucket cell-level errors/corrections into that row's error type.
  2. Otherwise, fall back to extracting the error type from the folder name
     (DGov_FD_*, DGov_NO_*, DGov_Typo_*).

Outputs:
  - horizon_by_error_type.csv          (summary per error type + overall)
  - horizon_by_error_type_per_table.csv (per-table breakdown)
"""
import os
import sys
import csv
import json
import argparse
from collections import defaultdict
import pandas as pd
from utils import read_csv


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


def evaluate_table(table_dir, dirname):
    """
    Evaluate a single table with per-error-type granularity.

    Returns dict: {error_type: {'tp': ..., 'tpfp': ..., 'tpfn': ...}, ...}
    or None if repaired file is missing.
    """
    dirty_path = os.path.join(table_dir, 'dirty.csv')
    clean_path = os.path.join(table_dir, 'clean.csv')
    repaired_path = os.path.join(table_dir, 'clean.csv.a2.clean')

    if not os.path.exists(repaired_path):
        return None

    dirty_df = read_csv(dirty_path)
    clean_df = read_csv(clean_path)
    repaired_df = read_csv(repaired_path)

    if '_tid_' in repaired_df.columns:
        repaired_df = repaired_df.drop(columns=['_tid_'])

    min_rows = min(len(dirty_df), len(clean_df), len(repaired_df))
    min_cols = min(len(dirty_df.columns), len(clean_df.columns), len(repaired_df.columns))
    if min_rows == 0 or min_cols == 0:
        return None

    dirty_df = dirty_df.iloc[:min_rows, :min_cols].reset_index(drop=True)
    clean_df = clean_df.iloc[:min_rows, :min_cols].reset_index(drop=True)
    repaired_df = repaired_df.iloc[:min_rows, :min_cols].reset_index(drop=True)

    row_to_type = load_lineage(table_dir)

    if row_to_type is None:
        folder_type = get_error_type_from_folder(dirname)
        if folder_type is None:
            folder_type = 'UNKNOWN'
        row_to_type = {i: folder_type for i in range(min_rows)}

    type_counts = defaultdict(lambda: {'tp': 0, 'tpfp': 0, 'tpfn': 0})

    for i in range(min_rows):
        error_type = row_to_type.get(i, 'UNKNOWN')
        for j in range(min_cols):
            dirty_val = str(dirty_df.iloc[i, j]) if pd.notna(dirty_df.iloc[i, j]) else ''
            clean_val = str(clean_df.iloc[i, j]) if pd.notna(clean_df.iloc[i, j]) else ''
            repaired_val = str(repaired_df.iloc[i, j]) if pd.notna(repaired_df.iloc[i, j]) else ''

            is_error = dirty_val != clean_val
            was_changed = dirty_val != repaired_val

            if is_error:
                type_counts[error_type]['tpfn'] += 1
                # Only count corrections attempted in error cells
                if was_changed:
                    type_counts[error_type]['tpfp'] += 1
                # Count truly corrected errors
                if clean_val == repaired_val or (clean_val == '' and repaired_val == ''):
                    type_counts[error_type]['tp'] += 1

    return dict(type_counts), min_rows, min_cols


def compute_metrics(tp, tpfp, tpfn):
    precision = tp / tpfp if tpfp > 0 else -1
    recall = tp / tpfn if tpfn > 0 else -1
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else -1
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Horizon per error type using row-level lineage.'
    )
    parser.add_argument(
        'lake_dir',
        nargs='?',
        default='/home/fatemeh/data/horizon-code/Final_Datasets/maximal_overlap_with_duplicates',
        help='Path to the lake directory containing table folders'
    )
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: <lake_dir>/horizon_by_error_type.csv)')
    args = parser.parse_args()

    lake_dir = args.lake_dir
    if not os.path.isdir(lake_dir):
        print(f"Error: '{lake_dir}' is not a directory.")
        sys.exit(1)

    output_csv = args.output or os.path.join(lake_dir, 'horizon_by_error_type.csv')

    type_totals = defaultdict(lambda: {'tp': 0, 'tpfp': 0, 'tpfn': 0, 'tables': 0})
    per_table_rows = []
    skipped = 0

    tables = list(find_tables(lake_dir))
    print(f"Found {len(tables)} table directories in {os.path.basename(lake_dir)}\n")

    for table_dir, dirname in tables:
        result = evaluate_table(table_dir, dirname)
        if result is None:
            skipped += 1
            continue

        type_counts, n_rows, n_cols = result

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
        per_table_rows.append({
            'table': dirname,
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

    if skipped:
        print(f"Skipped {skipped} tables (missing repaired file)\n")

    # Print per-error-type results
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
    total_tables = len(tables) - skipped
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

    # Save summary CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'error_type', 'tables', 'tp', 'tpfp', 'tpfn', 'precision', 'recall', 'f1'
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to: {output_csv}")

    # Save per-table CSV
    per_table_csv = output_csv.replace('.csv', '_per_table.csv')
    with open(per_table_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'table', 'error_types', 'tp', 'tpfp', 'tpfn', 'precision', 'recall', 'f1', 'rows', 'cols'
        ])
        writer.writeheader()
        writer.writerows(per_table_rows)
    print(f"Per-table results saved to: {per_table_csv}")


if __name__ == '__main__':
    main()
