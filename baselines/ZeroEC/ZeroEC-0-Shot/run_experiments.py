#!/usr/bin/env python3
"""
Run ZeroEC zero-shot experiments on all datasets in a given folder
and aggregate the results. Reads metrics from output.txt produced by correction_zero_shot.py.
"""

import os
import subprocess
import sys

from experiment_utils import (
    aggregate_results,
    collect_existing_results,
    extract_metrics_from_output,
)


def run_zeroshot_experiment(dataset_name, dataset_path, results_base_path, base_zeroshot_path):
    """
    Run zero-shot correction on a single dataset.
    Output is written to results_base_path / dataset_name / human_repair_0 (for compatibility with experiment_utils).
    """
    clean_path = os.path.join(dataset_path, 'clean.csv')
    dirty_path = os.path.join(dataset_path, 'dirty.csv')
    detection_path = os.path.join(dataset_path, 'perfect_error_detection.csv')
    output_path = os.path.join(results_base_path, dataset_name, 'human_repair_0')
    os.makedirs(output_path, exist_ok=True)

    correction_script = os.path.join(base_zeroshot_path, 'ZeroEC', 'correction_zero_shot.py')
    if not os.path.isfile(correction_script):
        print(f"Error: script not found: {correction_script}")
        return output_path, False

    print(f"\n{'='*80}")
    print(f"Running ZeroEC zero-shot on {dataset_name}")
    print(f"{'='*80}")

    cmd = [
        sys.executable,
        correction_script,
        '--clean_data_path', clean_path,
        '--dirty_data_path', dirty_path,
        '--detection_path', detection_path,
        '--output_path', output_path,
        '--dataset_name', dataset_name,
    ]

    try:
        print(f"Dataset: {dataset_name}")
        print(f"Output path: {output_path}")
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            print(f"Error running zero-shot: {result.stderr}")
            return output_path, False
        return output_path, True
    except Exception as e:
        print(f"Exception running zero-shot: {e}")
        return output_path, False


def main():
    BASE_ZEROEC_ZEROSHOT_PATH = '/home/fatemeh/LakeCorrectionBench/ZeroEC-0-Shot'
    DATASETS_PATH = '/home/fatemeh/LakeCorrectionBench/datasets/Quintet_3_missing'
    RESULTS_BASE_PATH = '/home/fatemeh/LakeCorrectionBench/ZeroEC-0-Shot/results/Quintet_3_missing_zeroshot_gemini'

    datasets = [d for d in os.listdir(DATASETS_PATH)
                if os.path.isdir(os.path.join(DATASETS_PATH, d))]
    datasets.sort()

    print(f"Found {len(datasets)} datasets: {datasets}")
    all_results = []

    for dataset_name in datasets:
        dataset_path = os.path.join(DATASETS_PATH, dataset_name)
        result_dir = os.path.join(RESULTS_BASE_PATH, dataset_name, 'human_repair_0')
        if os.path.exists(result_dir) and os.path.exists(os.path.join(result_dir, 'corrections.csv')):
            print(f"Skipping {dataset_name}: results already exist at {result_dir}")
            metrics = extract_metrics_from_output(result_dir)
            if metrics:
                all_results.append({
                    'dataset': dataset_name,
                    'human_repair_num': metrics['human_repair_num'],
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'fn': metrics['fn'],
                    'total_errors': metrics['total_errors'],
                    'total_corrections': metrics['total_corrections'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'execution_time': metrics['execution_time']
                })
            continue

        output_path, executed = run_zeroshot_experiment(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            results_base_path=RESULTS_BASE_PATH,
            base_zeroshot_path=BASE_ZEROEC_ZEROSHOT_PATH,
        )
        if executed:
            metrics = extract_metrics_from_output(output_path)
            if metrics:
                all_results.append({
                    'dataset': dataset_name,
                    'human_repair_num': metrics['human_repair_num'],
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'fn': metrics['fn'],
                    'total_errors': metrics['total_errors'],
                    'total_corrections': metrics['total_corrections'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'execution_time': metrics['execution_time']
                })
                print(f"\nResults for {dataset_name}:")
                print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
                print(f"  Execution time: {metrics['execution_time']:.2f}s")
            else:
                print(f"  WARNING: Could not extract metrics from output.txt")

    if all_results:
        output_csv = os.path.join(RESULTS_BASE_PATH, 'zeroshot_results.csv')
        os.makedirs(RESULTS_BASE_PATH, exist_ok=True)
        aggregate_results(all_results, output_csv)
    else:
        print("\n" + "="*80)
        print("No experiments were executed and no existing results were aggregated.")
        print("="*80)


def evaluate_existing_results():
    """Evaluate existing results without running experiments."""
    DATASETS_PATH = '/home/fatemeh/LakeCorrectionBench/datasets/Quintet_3_missing'
    RESULTS_BASE_PATH = '/home/fatemeh/LakeCorrectionBench/ZeroEC-0-Shot/results/Quintet_3_missing_zeroshot_gemini'

    all_results = collect_existing_results(DATASETS_PATH, RESULTS_BASE_PATH)
    if all_results:
        output_csv = os.path.join(RESULTS_BASE_PATH, 'zeroshot_results.csv')
        aggregate_results(all_results, output_csv)
        return True
    print("No existing results found to evaluate.")
    return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run ZeroEC zero-shot experiments')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only aggregate existing results without running experiments')
    args = parser.parse_args()

    if args.evaluate_only:
        print("Evaluating existing results...")
        evaluate_existing_results()
    else:
        print("Running zero-shot experiments...")
        main()
