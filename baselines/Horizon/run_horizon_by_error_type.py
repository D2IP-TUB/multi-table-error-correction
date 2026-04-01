#!/usr/bin/env python3
"""
Evaluate Horizon results per error type (FD, NO, Typo) for a data lake.

For each table directory in the lake, extracts the error type from the folder
name (DGov_FD_*, DGov_NO_*, DGov_Typo_*) or from manifest.json if available,
then computes precision/recall/F1 aggregated per error type.

Optionally runs Horizon before evaluation (skipped by default with --skip_horizon).
"""
import os
import sys
import csv
import json
import argparse
import subprocess
from collections import defaultdict
import pandas as pd
from utils import read_csv


def get_error_type(folder_name, manifest_lookup=None):
    """Extract error type from folder name or manifest lookup."""
    if manifest_lookup and folder_name in manifest_lookup:
        return manifest_lookup[folder_name].upper()

    name = folder_name.upper()
    if '_FD_' in name or name.startswith('DGOV_FD'):
        return 'FD'
    if '_NO_' in name or name.startswith('DGOV_NO'):
        return 'NO'
    if '_TYPO_' in name or name.startswith('DGOV_TYPO'):
        return 'TYPO'
    return 'UNKNOWN'


def load_manifest(lake_dir):
    """Load manifest.json and build a dir->variant lookup."""
    manifest_path = os.path.join(lake_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {
        entry['dir']: entry.get('variant', '').upper()
        for entry in manifest.get('partitions', [])
    }


def find_tables(root):
    """Yield (dirpath, dirname) for table directories with dirty.csv and clean.csv."""
    for entry in sorted(os.listdir(root)):
        dirpath = os.path.join(root, entry)
        if not os.path.isdir(dirpath):
            continue
        files = set(os.listdir(dirpath))
        if 'dirty.csv' in files and 'clean.csv' in files:
            yield dirpath, entry


def holo_to_fd_line(holo_line, colname_to_idx):
    if not holo_line.strip():
        return None
    parts = holo_line.strip().split('&')
    eqs = [p for p in parts if p.startswith('EQ(')]
    iqs = [p for p in parts if p.startswith('IQ(')]
    if not eqs or not iqs:
        return None

    def extract_attr(expr):
        inside = expr[expr.find('(') + 1:expr.find(')')]
        left, _ = inside.split(',')
        return left.split('.', 1)[1]

    lhs_names = [extract_attr(eq) for eq in eqs]
    rhs_names = [extract_attr(iq) for iq in iqs]
    try:
        lhs = ', '.join(str(colname_to_idx[n]) for n in lhs_names)
        rhs = ', '.join(str(colname_to_idx[n]) for n in rhs_names)
    except KeyError:
        return None
    return f"{lhs} -> {rhs}"


def convert_holo_to_fds(table_dir):
    """Convert holo_constraints.txt to fds.txt if needed."""
    holo_path = os.path.join(table_dir, 'holo_constraints.txt')
    fds_path = os.path.join(table_dir, 'fds.txt')
    clean_path = os.path.join(table_dir, 'clean.csv')
    if os.path.exists(holo_path) and os.path.exists(clean_path) and not os.path.exists(fds_path):
        with open(clean_path) as f:
            header = f.readline().strip().split(',')
        colname_to_idx = {name: idx for idx, name in enumerate(header)}
        with open(holo_path) as f:
            lines = f.readlines()
        fd_lines = [holo_to_fd_line(l, colname_to_idx) for l in lines]
        fd_lines = [l for l in fd_lines if l]
        with open(fds_path, 'w') as f:
            for l in fd_lines:
                f.write(l + '\n')


def run_horizon(table_dir, java_cp, algo=2):
    """Run the Horizon Java tool on a table directory. Returns (repaired_path, runtime_ms) or (None, 0)."""
    dirty = os.path.join(table_dir, 'dirty.csv')
    clean = os.path.join(table_dir, 'clean.csv')
    fds = os.path.join(table_dir, 'fds.txt')
    if not os.path.exists(fds):
        convert_holo_to_fds(table_dir)
    if not os.path.exists(fds):
        print(f"  Skipping {table_dir}: no fds.txt")
        return None, 0
    import shlex
    # Java from java_env conda environment (needs LD_LIBRARY_PATH for libjli.so)
    java_home = '/home/fatemeh/.conda/envs/java_env/lib/jvm'
    java_bin = os.path.join(java_home, 'bin', 'java')
    java_lib = os.path.join(java_home, 'lib')
    cmd_parts = [java_bin, '-cp', java_cp, 'Graph', dirty, clean, fds, str(algo)]
    shell_cmd = f"LD_LIBRARY_PATH={shlex.quote(java_lib)} {' '.join(shlex.quote(p) for p in cmd_parts)}"
    # Run from the project root so relative classpath (src:commons-collections4-4.3.jar) resolves correctly
    project_root = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(shell_cmd, check=True, cwd=project_root, capture_output=True, text=True, shell=True)
    # Parse runtime from Java stdout ("Elapsed time: <ms>")
    runtime_ms = 0
    for line in proc.stdout.splitlines():
        if line.startswith('Elapsed time:'):
            try:
                runtime_ms = int(line.split(':')[1].strip())
            except ValueError:
                pass
    return os.path.join(table_dir, f'clean.csv.a{algo}.clean'), runtime_ms


def evaluate_table(table_dir):
    """
    Evaluate a single table. Returns dict with tp, tpfp, tpfn counts,
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

    tp = 0
    tpfp = 0
    tpfn = 0

    for i in range(min_rows):
        for j in range(min_cols):
            dirty_val = str(dirty_df.iloc[i, j]) if pd.notna(dirty_df.iloc[i, j]) else ''
            clean_val = str(clean_df.iloc[i, j]) if pd.notna(clean_df.iloc[i, j]) else ''
            repaired_val = str(repaired_df.iloc[i, j]) if pd.notna(repaired_df.iloc[i, j]) else ''

            is_error = dirty_val != clean_val
            was_changed = dirty_val != repaired_val

            if is_error:
                tpfn += 1
                # Only count corrections attempted in error cells
                if was_changed:
                    tpfp += 1
                # Count truly corrected errors
                if clean_val == repaired_val or (clean_val == '' and repaired_val == ''):
                    tp += 1

    return {'tp': tp, 'tpfp': tpfp, 'tpfn': tpfn, 'rows': min_rows, 'cols': min_cols}


def compute_metrics(tp, tpfp, tpfn):
    precision = tp / tpfp if tpfp > 0 else -1
    recall = tp / tpfn if tpfn > 0 else -1
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else -1
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Horizon per error type (FD, NO, Typo) for a data lake.'
    )
    parser.add_argument(
        'lake_dir',
        nargs='?',
        default='/home/fatemeh/data/horizon-code/Final_Datasets/flattened_partitioned_base',
        help='Path to the lake directory containing table folders'
    )
    parser.add_argument('--java_cp', default='src:commons-collections4-4.3.jar',
                        help='Java classpath for Horizon')
    parser.add_argument('--algo', type=int, default=2, help='Horizon algorithm (default: 2)')
    parser.add_argument('--skip_horizon', action='store_true', default=True,
                        help='Skip running Horizon, only evaluate existing results (default: True)')
    parser.add_argument('--run_horizon', action='store_true',
                        help='Run Horizon before evaluating')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: <lake_dir>/horizon_by_error_type.csv)')
    args = parser.parse_args()

    if args.run_horizon:
        args.skip_horizon = False

    lake_dir = args.lake_dir
    if not os.path.isdir(lake_dir):
        print(f"Error: '{lake_dir}' is not a directory.")
        sys.exit(1)

    output_csv = args.output or os.path.join(lake_dir, 'horizon_by_error_type.csv')
    manifest_lookup = load_manifest(lake_dir)

    type_totals = defaultdict(lambda: {'tp': 0, 'tpfp': 0, 'tpfn': 0, 'tables': 0, 'skipped': 0, 'runtime_ms': 0})
    per_table_rows = []
    total_runtime_ms = 0

    tables = list(find_tables(lake_dir))
    print(f"Found {len(tables)} table directories in {lake_dir}\n")

    for table_dir, dirname in tables:
        error_type = get_error_type(dirname, manifest_lookup)

        table_runtime_ms = 0
        if not args.skip_horizon:
            try:
                repaired_path, table_runtime_ms = run_horizon(table_dir, args.java_cp, args.algo)
                if repaired_path is None:
                    type_totals[error_type]['skipped'] += 1
                    continue
            except Exception as e:
                print(f"  Horizon failed for {dirname}: {e}")
                type_totals[error_type]['skipped'] += 1
                continue

        result = evaluate_table(table_dir)
        if result is None:
            type_totals[error_type]['skipped'] += 1
            continue

        type_totals[error_type]['tp'] += result['tp']
        type_totals[error_type]['tpfp'] += result['tpfp']
        type_totals[error_type]['tpfn'] += result['tpfn']
        type_totals[error_type]['tables'] += 1
        type_totals[error_type]['runtime_ms'] += table_runtime_ms
        total_runtime_ms += table_runtime_ms

        p, r, f = compute_metrics(result['tp'], result['tpfp'], result['tpfn'])
        per_table_rows.append({
            'table': dirname,
            'error_type': error_type,
            'tp': result['tp'],
            'tpfp': result['tpfp'],
            'tpfn': result['tpfn'],
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1': round(f, 4),
            'rows': result['rows'],
            'cols': result['cols'],
            'runtime_ms': table_runtime_ms,
        })

    # Print per-error-type results
    print("=" * 90)
    print(f"{'Error Type':<12} {'Tables':>7} {'TP':>8} {'TP+FP':>8} {'TP+FN':>8}   {'Prec':>8} {'Recall':>8} {'F1':>8} {'Runtime(ms)':>12}")
    print("-" * 90)

    grand_tp, grand_tpfp, grand_tpfn, grand_tables = 0, 0, 0, 0
    summary_rows = []

    for error_type in ['FD', 'NO', 'TYPO', 'UNKNOWN']:
        if error_type not in type_totals:
            continue
        t = type_totals[error_type]
        p, r, f = compute_metrics(t['tp'], t['tpfp'], t['tpfn'])
        skipped_str = f" ({t['skipped']} skipped)" if t['skipped'] else ""
        print(f"{error_type:<12} {t['tables']:>7}{skipped_str}  {t['tp']:>7} {t['tpfp']:>8} {t['tpfn']:>8}   {p:>8.4f} {r:>8.4f} {f:>8.4f} {t['runtime_ms']:>12}")

        grand_tp += t['tp']
        grand_tpfp += t['tpfp']
        grand_tpfn += t['tpfn']
        grand_tables += t['tables']

        summary_rows.append({
            'error_type': error_type,
            'tables': t['tables'],
            'tp': t['tp'],
            'tpfp': t['tpfp'],
            'tpfn': t['tpfn'],
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1': round(f, 4),
            'runtime_ms': t['runtime_ms'],
        })

    print("-" * 90)
    overall_p, overall_r, overall_f = compute_metrics(grand_tp, grand_tpfp, grand_tpfn)
    print(f"{'OVERALL':<12} {grand_tables:>7}  {grand_tp:>7} {grand_tpfp:>8} {grand_tpfn:>8}   {overall_p:>8.4f} {overall_r:>8.4f} {overall_f:>8.4f} {total_runtime_ms:>12}")
    print(f"Total lake runtime: {total_runtime_ms} ms ({total_runtime_ms/1000:.2f} s)")
    print("=" * 90)

    summary_rows.append({
        'error_type': 'OVERALL',
        'tables': grand_tables,
        'tp': grand_tp,
        'tpfp': grand_tpfp,
        'tpfn': grand_tpfn,
        'precision': round(overall_p, 4),
        'recall': round(overall_r, 4),
        'f1': round(overall_f, 4),
        'runtime_ms': total_runtime_ms,
    })

    # Save summary CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'error_type', 'tables', 'tp', 'tpfp', 'tpfn', 'precision', 'recall', 'f1', 'runtime_ms'
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to: {output_csv}")

    # Save per-table CSV
    per_table_csv = output_csv.replace('.csv', '_per_table.csv')
    with open(per_table_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'table', 'error_type', 'tp', 'tpfp', 'tpfn', 'precision', 'recall', 'f1', 'rows', 'cols', 'runtime_ms'
        ])
        writer.writeheader()
        writer.writerows(per_table_rows)
    print(f"Per-table results saved to: {per_table_csv}")


if __name__ == '__main__':
    main()
