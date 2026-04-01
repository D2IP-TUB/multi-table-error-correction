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
import sys
from utils import read_csv, get_dataframes_difference, evaluate


def evaluate_repair(dirty_path: str, clean_path: str, repaired_path: str):
    """
    Evaluate repair quality using utils.py conventions.
    
    Args:
        dirty_path: Path to dirty data CSV
        clean_path: Path to clean (ground truth) data CSV
        repaired_path: Path to repaired data CSV
    
    Returns:
        Dictionary with evaluation metrics
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
    
    # Find errors (cells that differ between dirty and clean) using get_dataframes_difference
    print("\nIdentifying ground truth errors (dirty vs clean)...")
    detections = get_dataframes_difference(dirty_df, clean_df)
    print(f"  Total errors found: {len(detections)}")
    
    # Use the evaluate function from utils.py
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
    
    # Show some examples of errors and repairs
    print("\n" + "=" * 60)
    print("SAMPLE ERRORS AND REPAIRS")
    print("=" * 60)
    
    # Collect sample corrections - iterate through ALL errors to find examples
    sample_correct = []
    sample_incorrect = []
    sample_missed = []
    n_rep_rows, n_rep_cols = len(repaired_df), len(repaired_df.columns)

    for error in detections.keys():
        row_idx, col_idx = error
        if row_idx >= n_rep_rows or col_idx >= n_rep_cols:
            continue
        dirty_val = dirty_df.iloc[row_idx, col_idx]
        clean_val = clean_df.iloc[row_idx, col_idx]
        repaired_val = repaired_df.iloc[row_idx, col_idx]
        col_name = dirty_df.columns[col_idx]
        
        # Check if truly corrected
        is_correct = clean_val == repaired_val or (len(str(clean_val)) == 0 and len(str(repaired_val)) == 0)
        # Check if repair was attempted
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
        
        # Stop early if we have enough samples
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
    
    # Per-column analysis
    print("\n" + "=" * 60)
    print("PER-COLUMN ANALYSIS")
    print("=" * 60)
    
    columns = dirty_df.columns.tolist()
    print(f"\n{'Column':<20} {'Errors':<10} {'Corrected':<12} {'Correct':<10} {'Precision':<12} {'Recall':<12}")
    print("-" * 76)
    
    n_rep_rows, n_rep_cols = len(repaired_df), len(repaired_df.columns)
    for col_idx, col_name in enumerate(columns):
        # Errors in this column (only those in range of repaired df)
        col_errors = {e for e in detections.keys() if e[1] == col_idx and e[0] < n_rep_rows and col_idx < n_rep_cols}

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
        
        if n_col_errors > 0:  # Only show columns with errors
            print(f"{col_name:<20} {n_col_errors:<10} {n_col_all_corrected:<12} {n_col_truly_corrected:<10} {col_precision:<12.4f} {col_recall:<12.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate data repair quality')
    parser.add_argument('--dirty', '-d', required=True, help='Path to dirty data CSV', default='/Users/fatemehahmadi/Downloads/horizon-code/flights_splitted/joined/flights/dirty.csv')
    parser.add_argument('--clean', '-c', required=True, help='Path to clean (ground truth) data CSV', default='/Users/fatemehahmadi/Downloads/horizon-code/flights_splitted/joined/flights/clean.csv')
    parser.add_argument('--repaired', '-r', required=True, help='Path to repaired data CSV', default='/Users/fatemehahmadi/Downloads/horizon-code/flights_splitted/joined/flights/clean.csv.a2.clean')
    
    args = parser.parse_args()
    
    results = evaluate_repair(args.dirty, args.clean, args.repaired)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nPrecision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")


if __name__ == '__main__':
    main()
