#!/usr/bin/env python3
"""
Script to calculate per-column evaluation metrics (Precision, Recall, F1)
from cell analysis CSV files.

Recall is calculated based on the total number of error cells per column
from the original dataset (comparing dirty vs clean).
"""

import os
import sys
import html
import re
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path


def value_normalizer(value: str) -> str:
    """
    This method takes a value and minimally normalizes it. (Raha's value normalizer)
    """
    if value is not np.nan:
        value = html.unescape(value)
        value = re.sub(r"[\t\n ]+", " ", value, flags=re.UNICODE)
        value = value.strip("\t\n ")
    return value


def get_table_id(table_name: str) -> str:
    """Generate table_id (MD5 hash) from table name."""
    return hashlib.md5(table_name.encode()).hexdigest()


def count_errors_per_column(dataset_dir: Path) -> tuple:
    """
    Count the number of error cells per (table_id, column_idx) from the dataset.
    
    Compares dirty.csv vs clean.csv for each table to count actual errors.
    
    Returns:
        tuple: (error_counts, column_names, table_names)
            - error_counts: {(table_id, column_idx): error_count}
            - column_names: {(table_id, column_idx): column_name}
            - table_names: {table_id: table_name}
    """
    error_counts = {}
    column_names = {}
    table_names = {}
    
    # Find all table directories (those containing clean.csv and dirty.csv)
    for table_dir in dataset_dir.rglob("*"):
        if table_dir.is_dir():
            clean_file = table_dir / "clean.csv"
            dirty_file = table_dir / "dirty.csv"
            
            if clean_file.exists() and dirty_file.exists():
                table_name = table_dir.name
                table_id = get_table_id(table_name)
                table_names[table_id] = table_name
                
                try:
                    clean_df = pd.read_csv(clean_file, dtype=str, keep_default_na=False, encoding='latin-1')
                    dirty_df = pd.read_csv(dirty_file, dtype=str, keep_default_na=False, encoding='latin-1')
                    
                    # Apply value normalizer to both dataframes
                    clean_df = clean_df.map(lambda x: value_normalizer(x) if isinstance(x, str) else x)
                    dirty_df = dirty_df.map(lambda x: value_normalizer(x) if isinstance(x, str) else x)
                    
                    # Count errors per column
                    for col_idx, col_name in enumerate(clean_df.columns):
                        if col_name in dirty_df.columns:
                            errors = (clean_df[col_name] != dirty_df[col_name]).sum()
                            error_counts[(table_id, col_idx)] = errors
                            column_names[(table_id, col_idx)] = col_name
                            
                except Exception as e:
                    print(f"Warning: Could not process {table_dir}: {e}")
    
    return error_counts, column_names, table_names


def parse_sampled_cells(sampled_cells_file: Path) -> dict:
    """
    Parse the sampled_cells.txt file to count manual samples per (table_id, column_idx).
    
    Manual samples are considered as TPs since they are correctly labeled by oracle.
    
    Args:
        sampled_cells_file: Path to sampled_cells.txt file
        
    Returns:
        dict: {(table_id, column_idx): count}
    """
    manual_samples = {}
    
    if not sampled_cells_file.exists():
        return manual_samples
    
    with open(sampled_cells_file, 'r') as f:
        content = f.read()
    
    # Parse Cell entries using regex
    # Format: Cell(table_id='xxx', column_idx=N, ...)
    pattern = r"Cell\(table_id='([^']+)',\s*column_idx=(\d+)"
    matches = re.findall(pattern, content)
    
    for table_id, col_idx in matches:
        key = (table_id, int(col_idx))
        manual_samples[key] = manual_samples.get(key, 0) + 1
    
    return manual_samples


def calculate_metrics(df: pd.DataFrame, total_errors: int = None, manual_sample_count: int = 0) -> dict:
    """
    Calculate precision, recall, and F1 from correction status counts.
    
    Based on correction_status values:
    - CORRECT_CORRECTION: True Positive (TP) - correctly fixed an error
    - INCORRECT_CORRECTION: False Positive (FP) - incorrectly "fixed" something  
    
    Manual samples are also counted as TPs since they are correctly labeled by oracle.
    
    Recall is based on total_errors from the dataset (if provided), not from
    MISSED_ERROR count in the analysis files.
    """
    # Count each correction status
    status_counts = df['correction_status'].value_counts()
    
    tp = status_counts.get('CORRECT_CORRECTION', 0)
    fp = status_counts.get('INCORRECT_CORRECTION', 0)
    fn_from_file = status_counts.get('MISSED_ERROR', 0)
    
    # Add manual samples as TPs
    tp_with_manual = tp + manual_sample_count
    
    # Use total_errors from dataset if provided, otherwise fall back to file-based FN
    if total_errors is not None:
        fn = total_errors - tp_with_manual  # FN = total errors - correctly corrected (including manual)
    else:
        fn = fn_from_file
    
    # Calculate metrics
    precision = tp_with_manual / (tp_with_manual + fp) if (tp_with_manual + fp) > 0 else 0.0
    recall = tp_with_manual / (tp_with_manual + fn) if (tp_with_manual + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'TP': tp,
        'manual_samples': manual_sample_count,
        'TP_total': tp_with_manual,
        'FP': fp,
        'FN': fn,
        'total_errors': total_errors if total_errors is not None else (tp_with_manual + fn_from_file),
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


def evaluate_cell_analysis(input_dir: str, dataset_dir: str = None, sampled_cells_file: str = None) -> pd.DataFrame:
    """
    Read all CSV files in the input directory and calculate per-column metrics.
    
    Args:
        input_dir: Path to directory containing cell analysis CSV files
        dataset_dir: Path to dataset directory for counting true errors per column
        sampled_cells_file: Path to sampled_cells.txt for counting manual samples as TPs
        
    Returns:
        DataFrame with per-column evaluation metrics
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    # Find all CSV files (exclude previously generated metrics files)
    csv_files = [f for f in input_path.glob("*.csv") 
                 if not f.name.startswith("per_column_evaluation")]
    
    if not csv_files:
        print(f"Error: No CSV files found in '{input_dir}'")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f.name}")
    print()
    
    # Count errors per column from dataset if provided
    error_counts = {}
    column_names = {}
    table_names = {}
    if dataset_dir:
        dataset_path = Path(dataset_dir)
        if dataset_path.exists():
            print(f"Counting errors from dataset: {dataset_dir}")
            error_counts, column_names, table_names = count_errors_per_column(dataset_path)
            print(f"Found error counts for {len(error_counts)} (table, column) pairs")
            print()
        else:
            print(f"Warning: Dataset directory '{dataset_dir}' not found. Using file-based FN.")
            print()
    
    # Parse manual samples from sampled_cells.txt
    manual_samples = {}
    if sampled_cells_file:
        sampled_path = Path(sampled_cells_file)
        if sampled_path.exists():
            print(f"Parsing manual samples from: {sampled_cells_file}")
            manual_samples = parse_sampled_cells(sampled_path)
            total_manual = sum(manual_samples.values())
            print(f"Found {total_manual} manual samples across {len(manual_samples)} (table, column) pairs")
            print()
        else:
            print(f"Warning: Sampled cells file '{sampled_cells_file}' not found.")
            print()
    
    # Read and combine all CSV files
    all_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, low_memory=False)
        df['source_file'] = csv_file.name
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"Total rows in analysis files: {len(combined_df)}")
    print(f"Correction status distribution:")
    print(combined_df['correction_status'].value_counts().to_string())
    print()
    
    # Calculate per-column metrics
    results = []
    
    # Group by table_id and column_idx
    for (table_id, col_idx), group in combined_df.groupby(['table_id', 'column_idx']):
        total_errors = error_counts.get((table_id, col_idx), None)
        manual_sample_count = manual_samples.get((table_id, col_idx), 0)
        metrics = calculate_metrics(group, total_errors, manual_sample_count)
        results.append({
            'table_name': table_names.get(table_id, table_id),
            'column_name': column_names.get((table_id, col_idx), f"col_{col_idx}"),
            'table_id': table_id,
            'column_idx': col_idx,
            'num_cells_in_analysis': len(group),
            **metrics
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by table_name and column_idx
    results_df = results_df.sort_values(['table_name', 'column_idx']).reset_index(drop=True)
    
    return results_df, combined_df


def main():
    # Default directories
    default_input_dir = "/home/fatemeh/data/EC-at-Scale/results_clustering_based_mem_optimized_address_isolated/output_isolated_2_32_-1/cell_analysis"
    default_dataset_dir = "/home/fatemeh/data/EC-at-Scale/datasets/address_pr_dataset_processed_without_city/isolated"
    default_sampled_cells = "/home/fatemeh/data/EC-at-Scale/results_clustering_based_mem_optimized_address_isolated/output_isolated_2_32_-1/sampled_cells.txt"
    
    # Parse command line arguments
    input_dir = sys.argv[1] if len(sys.argv) > 1 else default_input_dir
    dataset_dir = sys.argv[2] if len(sys.argv) > 2 else default_dataset_dir
    sampled_cells_file = sys.argv[3] if len(sys.argv) > 3 else default_sampled_cells
    
    print("=" * 80)
    print("Cell Analysis Evaluation - Per Column Metrics")
    print("=" * 80)
    print(f"\nInput directory: {input_dir}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Sampled cells file: {sampled_cells_file}\n")
    
    # Calculate metrics
    results_df, combined_df = evaluate_cell_analysis(input_dir, dataset_dir, sampled_cells_file)
    
    # Display per-column results
    print("\n" + "=" * 80)
    print("PER-COLUMN EVALUATION METRICS")
    print("=" * 80)
    
    # Format for display
    display_df = results_df.copy()
    display_df['Precision'] = display_df['Precision'].apply(lambda x: f"{x:.4f}")
    display_df['Recall'] = display_df['Recall'].apply(lambda x: f"{x:.4f}")
    display_df['F1'] = display_df['F1'].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))
    
    # Calculate and display overall metrics
    print("\n" + "=" * 80)
    print("OVERALL METRICS (AGGREGATED)")
    print("=" * 80)
    
    total_tp = results_df['TP'].sum()
    total_manual = results_df['manual_samples'].sum()
    total_tp_total = results_df['TP_total'].sum()
    total_fp = results_df['FP'].sum()
    total_fn = results_df['FN'].sum()
    total_errors = results_df['total_errors'].sum()
    
    overall_precision = total_tp_total / (total_tp_total + total_fp) if (total_tp_total + total_fp) > 0 else 0.0
    overall_recall = total_tp_total / total_errors if total_errors > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    print(f"\nTotal Errors (from dataset):      {total_errors}")
    print(f"Total TP (Correct Corrections):   {total_tp}")
    print(f"Total Manual Samples (as TP):     {total_manual}")
    print(f"Total TP (including manual):      {total_tp_total}")
    print(f"Total FP (Incorrect Corrections): {total_fp}")
    print(f"Total FN (Missed Errors):         {total_fn}")
    print(f"\nOverall Precision: {overall_precision:.4f}")
    print(f"Overall Recall:    {overall_recall:.4f}")
    print(f"Overall F1:        {overall_f1:.4f}")
    
    # Save results to CSV
    output_file = Path(input_dir) / "per_column_evaluation_metrics.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")
    
    return results_df


if __name__ == "__main__":
    main()
