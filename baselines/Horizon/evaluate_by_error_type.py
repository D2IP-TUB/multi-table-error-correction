#!/usr/bin/env python3
"""
Evaluate Horizon results grouped by error type (FD, NO, Typo).
- For isolated/disjoint/maximal datasets: error type from folder name
- For flattened_partial datasets: read lineage.csv and evaluate per-row

Outputs separate per-type results for each subdirectory in Final_Datasets.
"""
import os
import sys
import csv
import json
import argparse
from collections import defaultdict
import pandas as pd
from utils import read_csv


def is_row_level_dataset(subdir_name):
    """Check if this dataset type requires row-level evaluation from lineage."""
    # ALL datasets should use row-level evaluation from lineage.csv
    return True


def get_error_type_from_folder(table_dir):
    """Get error type from folder name suffix (e.g., __FD, __NO, __Typo)."""
    folder_name = os.path.basename(table_dir)
    
    if folder_name.endswith('__FD'):
        return 'FD'
    elif folder_name.endswith('__NO'):
        return 'NO'
    elif folder_name.endswith('__Typo'):
        return 'TYPO'
    
    # Check for these patterns in the name
    if '__FD' in folder_name or '_FD_' in folder_name:
        return 'FD'
    if '__NO' in folder_name or '_NO_' in folder_name:
        return 'NO'
    if '__Typo' in folder_name or '_Typo_' in folder_name:
        return 'TYPO'
    
    return 'UNKNOWN'


def load_lineage(table_dir):
    """Load lineage.csv and return a dict mapping row_idx -> source_variant."""
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


def evaluate_table_by_type(table_dir):
    """
    Evaluate a table and return per-error-type metrics.
    
    Returns dict like:
        {
            'FD': {'tp': ..., 'tpfp': ..., 'tpfn': ...},
            'NO': {...},
            'TYPO': {...}
        }
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
    
    # Check for row count mismatch
    if len(dirty_df) != len(repaired_df):
        print(f"  Warning: Row count mismatch in {os.path.basename(table_dir)}: "
              f"dirty={len(dirty_df)}, repaired={len(repaired_df)}")
        # Only evaluate overlapping rows
        min_rows = min(len(dirty_df), len(clean_df), len(repaired_df))
        dirty_df = dirty_df.iloc[:min_rows].reset_index(drop=True)
        clean_df = clean_df.iloc[:min_rows].reset_index(drop=True)
        repaired_df = repaired_df.iloc[:min_rows].reset_index(drop=True)
    
    # Get row->type mapping from lineage
    row_to_type = load_lineage(table_dir)
    
    # If no lineage or not a row-level dataset, use folder name for all rows
    parent_dir = os.path.basename(os.path.dirname(table_dir))
    use_row_level = is_row_level_dataset(parent_dir) and row_to_type is not None
    
    if not use_row_level:
        # Use folder name as error type for all rows
        folder_type = get_error_type_from_folder(table_dir)
        if row_to_type is None:
            row_to_type = {i: folder_type for i in range(len(dirty_df))}
        else:
            # If lineage exists but not row-level dataset, still use folder type
            row_to_type = {i: folder_type for i in range(len(dirty_df))}
    
    # Find errors (dirty vs clean) - cell by cell comparison
    errors = {}  # {(row, col): clean_value}
    for i in range(len(dirty_df)):
        for j in range(len(dirty_df.columns)):
            dirty_val = str(dirty_df.iloc[i, j]) if pd.notna(dirty_df.iloc[i, j]) else ''
            clean_val = str(clean_df.iloc[i, j]) if pd.notna(clean_df.iloc[i, j]) else ''
            if dirty_val != clean_val:
                errors[(i, j)] = clean_val
    
    # Initialize per-type counters
    type_metrics = defaultdict(lambda: {'tp': 0, 'tpfp': 0, 'tpfn': 0})
    
    # Count TP+FN (all errors) per type
    for (row, col), clean_val in errors.items():
        error_type = row_to_type.get(row, 'UNKNOWN')
        type_metrics[error_type]['tpfn'] += 1
    
    # Count TP+FP and TP *only on ground-truth error cells* (dirty != clean),
    # matching utils.evaluate/evaluate_repair.py conventions.
    for (row, col), clean_val in errors.items():
        error_type = row_to_type.get(row, 'UNKNOWN')

        repaired_val = str(repaired_df.iloc[row, col]) if pd.notna(repaired_df.iloc[row, col]) else ''
        dirty_val = str(dirty_df.iloc[row, col]) if pd.notna(dirty_df.iloc[row, col]) else ''

        # TP+FP: correction attempted on an error cell
        if dirty_val != repaired_val:
            type_metrics[error_type]['tpfp'] += 1

            # TP: correct repair on an error cell
            if repaired_val == clean_val:
                type_metrics[error_type]['tp'] += 1
    
    return dict(type_metrics)


def compute_metrics(tp, tpfp, tpfn):
    """Compute precision, recall, F1 from counts."""
    precision = tp / tpfp if tpfp > 0 else -1
    recall = tp / tpfn if tpfn > 0 else -1
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else -1
    return precision, recall, f1


def find_tables(root):
    """Yield table directories containing dirty.csv, clean.csv, and repaired file."""
    for dirpath, dirnames, filenames in os.walk(root):
        files = set(filenames)
        if 'dirty.csv' in files and 'clean.csv' in files:
            if 'clean.csv.a2.clean' in files:
                yield dirpath


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Horizon results grouped by error type (FD, NO, Typo).'
    )
    parser.add_argument(
        '--dataset_root',
        default='/home/fatemeh/data/horizon-code/Final_Datasets',
        help='Root directory containing datasets'
    )
    parser.add_argument(
        '--output_dir',
        default='/home/fatemeh/data/horizon-code/err_type_bug_fixed',
        help='Directory to save output files'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        print(f"Error: Dataset root '{args.dataset_root}' not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Get all subdirectories in Final_Datasets
    subdirs = sorted([d for d in os.listdir(args.dataset_root) 
                      if os.path.isdir(os.path.join(args.dataset_root, d))])

    all_overall_results = []
    lake_type_totals = defaultdict(lambda: {'tp': 0, 'tpfp': 0, 'tpfn': 0})

    for subdir in subdirs:
        subdir_path = os.path.join(args.dataset_root, subdir)
        print(f"\n{'='*60}")
        print(f"Processing: {subdir}")
        print(f"{'='*60}")
        
        # Aggregate per-type results for this subdirectory
        subdir_type_totals = defaultdict(lambda: {'tp': 0, 'tpfp': 0, 'tpfn': 0})
        table_count = 0
        
        for table_dir in find_tables(subdir_path):
            try:
                type_metrics = evaluate_table_by_type(table_dir)
                if type_metrics is None:
                    continue
                
                table_count += 1
                for error_type, counts in type_metrics.items():
                    subdir_type_totals[error_type]['tp'] += counts['tp']
                    subdir_type_totals[error_type]['tpfp'] += counts['tpfp']
                    subdir_type_totals[error_type]['tpfn'] += counts['tpfn']
                
            except Exception as e:
                print(f"  Error processing {table_dir}: {e}")
        
        print(f"  Processed {table_count} tables")
        
        # Compute and display per-type results
        aggregate_rows = []
        total_tp = 0
        total_tpfp = 0
        total_tpfn = 0
        
        for error_type in sorted(subdir_type_totals.keys()):
            counts = subdir_type_totals[error_type]
            tp, tpfp, tpfn = counts['tp'], counts['tpfp'], counts['tpfn']
            precision, recall, f1 = compute_metrics(tp, tpfp, tpfn)
            
            total_tp += tp
            total_tpfp += tpfp
            total_tpfn += tpfn

            lake_type_totals[error_type]['tp']   += tp
            lake_type_totals[error_type]['tpfp'] += tpfp
            lake_type_totals[error_type]['tpfn'] += tpfn
            
            print(f"  {error_type}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f} (TP={tp}, TP+FP={tpfp}, TP+FN={tpfn})")
            
            aggregate_rows.append({
                'error_type': error_type,
                'tp': tp,
                'tpfp': tpfp,
                'tpfn': tpfn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        # Overall for this subdirectory
        overall_precision, overall_recall, overall_f1 = compute_metrics(total_tp, total_tpfp, total_tpfn)
        print(f"  OVERALL: P={overall_precision:.4f}, R={overall_recall:.4f}, F1={overall_f1:.4f}")
        
        aggregate_rows.append({
            'error_type': 'OVERALL',
            'tp': total_tp,
            'tpfp': total_tpfp,
            'tpfn': total_tpfn,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        })
        
        # Save per-subdirectory CSV
        output_csv = os.path.join(args.output_dir, f'horizon_by_type_{subdir}.csv')
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'error_type', 'tp', 'tpfp', 'tpfn', 'precision', 'recall', 'f1'
            ])
            writer.writeheader()
            writer.writerows(aggregate_rows)
        print(f"  Saved: {output_csv}")
        
        # Track for overall summary
        all_overall_results.append({
            'dataset': subdir,
            'tables': table_count,
            'tp': total_tp,
            'tpfp': total_tpfp,
            'tpfn': total_tpfn,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        })
    
    # Save overall summary across all subdirectories (+ one accumulative row)
    summary_csv = os.path.join(args.output_dir, 'horizon_by_type_summary.csv')
    acc_tp   = sum(r['tp']   for r in all_overall_results)
    acc_tpfp = sum(r['tpfp'] for r in all_overall_results)
    acc_tpfn = sum(r['tpfn'] for r in all_overall_results)
    acc_p, acc_r, acc_f1 = compute_metrics(acc_tp, acc_tpfp, acc_tpfn)
    all_overall_results.append({
        'dataset': 'ACCUMULATIVE',
        'tables': sum(r['tables'] for r in all_overall_results),
        'tp': acc_tp, 'tpfp': acc_tpfp, 'tpfn': acc_tpfn,
        'precision': acc_p, 'recall': acc_r, 'f1': acc_f1
    })
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset', 'tables', 'tp', 'tpfp', 'tpfn', 'precision', 'recall', 'f1'
        ])
        writer.writeheader()
        writer.writerows(all_overall_results)
    print(f"\n\nOverall summary saved to: {summary_csv}")

    # Save accumulative scores by error type across ALL datasets
    print("\n=== ACCUMULATIVE SCORES BY ERROR TYPE (all datasets) ===")
    lake_type_rows = []
    lake_total_tp = lake_total_tpfp = lake_total_tpfn = 0
    for error_type in sorted(lake_type_totals.keys()):
        counts = lake_type_totals[error_type]
        tp, tpfp, tpfn = counts['tp'], counts['tpfp'], counts['tpfn']
        precision, recall, f1 = compute_metrics(tp, tpfp, tpfn)
        print(f"  {error_type}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f} "
              f"(TP={tp}, TP+FP={tpfp}, TP+FN={tpfn})")
        lake_type_rows.append({
            'error_type': error_type,
            'tp': tp, 'tpfp': tpfp, 'tpfn': tpfn,
            'precision': precision, 'recall': recall, 'f1': f1
        })
        lake_total_tp   += tp
        lake_total_tpfp += tpfp
        lake_total_tpfn += tpfn

    overall_p, overall_r, overall_f1 = compute_metrics(lake_total_tp, lake_total_tpfp, lake_total_tpfn)
    print(f"  OVERALL: P={overall_p:.4f}, R={overall_r:.4f}, F1={overall_f1:.4f} "
          f"(TP={lake_total_tp}, TP+FP={lake_total_tpfp}, TP+FN={lake_total_tpfn})")
    lake_type_rows.append({
        'error_type': 'OVERALL',
        'tp': lake_total_tp, 'tpfp': lake_total_tpfp, 'tpfn': lake_total_tpfn,
        'precision': overall_p, 'recall': overall_r, 'f1': overall_f1
    })

    lake_type_csv = os.path.join(args.output_dir, 'horizon_by_type_lake_accumulative.csv')
    with open(lake_type_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'error_type', 'tp', 'tpfp', 'tpfn', 'precision', 'recall', 'f1'
        ])
        writer.writeheader()
        writer.writerows(lake_type_rows)
    print(f"Accumulative by-type scores saved to: {lake_type_csv}")


if __name__ == '__main__':
    main()
