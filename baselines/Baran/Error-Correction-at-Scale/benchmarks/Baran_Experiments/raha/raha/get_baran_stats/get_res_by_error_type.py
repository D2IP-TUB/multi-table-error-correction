import html
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


ERROR_TYPE_PREFIXES = {
    "DGov_FD": "FD",
    "DGov_NO": "Numeric Outlier",
    "DGov_Typo": "Typo",
}


def value_normalizer(value):
    if isinstance(value, str):
        value = html.unescape(value)
        value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
        value = value.strip("\t\n ")
    return value


def classify_error_type(dataset_name):
    for prefix, label in ERROR_TYPE_PREFIXES.items():
        if dataset_name.startswith(prefix):
            return label
    return "Unknown"


def get_results_df(sandbox_path, results_path, algorithm, repition, labeling_budgets):
    datasets = []
    for dir in os.listdir(sandbox_path):
        if dir.startswith('union_summary') or dir.endswith('.json'):
            continue
        datasets.append(dir)
    print(f"  Found {len(datasets)} datasets")

    results_dict = {
        "algorithm": [], "dataset": [], "execution_number": [],
        "precision": [], "recall": [], "f1_score": [],
        "tp": [], "ec_tpfp": [], "ec_tpfn": [], "execution_time": [],
        "number_of_labeled_tuples": [], "number_of_labeled_cells": [],
    }

    for i in repition:
        for dataset in datasets:
            if dataset.startswith('union_summary') or dataset.startswith('dataset_info.json'):
                continue
            for label_budget in labeling_budgets:
                file_path = results_path + '/{}_{}_col_0_number#{}_${}$labels.json' \
                    .format(algorithm, dataset, str(i), str(label_budget))
                file_path = str(Path(file_path).resolve())
                if os.path.exists(file_path):
                    with open(file_path) as file:
                        json_content = json.load(file)
                        results_dict['algorithm'].append(algorithm)
                        results_dict['dataset'].append(dataset)
                        results_dict['execution_number'].append(i)
                        results_dict['precision'].append(json_content['precision'])
                        results_dict['recall'].append(json_content['recall'])
                        results_dict['f1_score'].append(json_content['f_score'])
                        results_dict['tp'].append(json_content['tp'])
                        results_dict['ec_tpfp'].append(json_content['ec_tpfp'])
                        results_dict['ec_tpfn'].append(json_content['ec_tpfn'])
                        results_dict['execution_time'].append(json_content['execution-time'])
                        results_dict['number_of_labeled_tuples'].append(json_content['number_of_labeled_tuples'])
                        results_dict['number_of_labeled_cells'].append(json_content['number_of_labeled_cells'])
                else:
                    print(f"  Missing: {file_path}")

    return pd.DataFrame.from_dict(results_dict)


def get_eds_n_errors_by_type(base_path, dirty_file_name, clean_file_name):
    """Count total errors per error type across all tables."""
    errors_by_type = {}
    tables = os.listdir(base_path)
    for table in sorted(tables):
        if table.startswith('union_summary') or table.endswith('.json'):
            continue
        if not os.path.isdir(os.path.join(base_path, table)):
            continue
        dirty_df = pd.read_csv(
            os.path.join(base_path, table, dirty_file_name),
            keep_default_na=False, dtype=str,
        ).map(lambda x: value_normalizer(x) if isinstance(x, str) else x)
        clean_df = pd.read_csv(
            os.path.join(base_path, table, clean_file_name),
            keep_default_na=False, dtype=str,
        ).map(lambda x: value_normalizer(x) if isinstance(x, str) else x)
        if dirty_df.shape != clean_df.shape:
            print(f"Shape mismatch for {table}")
        dirty_df.columns = clean_df.columns
        diff = dirty_df.compare(clean_df, keep_shape=True)
        self_diff = diff.xs('self', axis=1, level=1)
        other_diff = diff.xs('other', axis=1, level=1)
        label_df = ((self_diff != other_diff) & ~(self_diff.isna() & other_diff.isna())).astype(int)
        n_errors = label_df.sum().sum()

        etype = classify_error_type(table)
        errors_by_type[etype] = errors_by_type.get(etype, 0) + n_errors
    return errors_by_type


def get_total_results_by_error_type(labeling_budgets, repition, result_df, errors_by_type):
    """Aggregate Baran results per error type, computing macro metrics the same
    way the original get_total_results does (average across repetitions)."""
    result_df = result_df.copy()
    result_df['error_type'] = result_df['dataset'].apply(classify_error_type)

    all_rows = []
    for etype in sorted(result_df['error_type'].unique()):
        etype_df = result_df[result_df['error_type'] == etype]
        ed_tpfn = errors_by_type.get(etype, 0)

        for label_budget in labeling_budgets:
            avg_precision = 0
            avg_recall = 0
            avg_f_score = 0
            tp_sum = 0
            ed_tpfp_sum = 0
            f_scores = []
            execution_time = 0
            n_labeled_cells = 0
            n_labeled_tuples = 0

            for rep in repition:
                res_rep = etype_df[etype_df['execution_number'] == rep]
                res_rep_lab = res_rep[res_rep['number_of_labeled_tuples'] == label_budget]
                tp = res_rep_lab['tp'].sum()
                tpfp = res_rep_lab['ec_tpfp'].sum()
                tp_sum += tp
                ed_tpfp_sum += tpfp
                n_labeled_cells += res_rep_lab['number_of_labeled_cells'].sum()
                n_labeled_tuples += res_rep_lab['number_of_labeled_tuples'].sum()

                if tpfp == 0:
                    precision = recall = f_score = 0
                else:
                    precision = tp / tpfp
                    recall = tp / ed_tpfn if ed_tpfn > 0 else 0
                    if precision + recall > 0:
                        f_score = 2 * precision * recall / (precision + recall)
                    else:
                        f_score = 0
                avg_precision += precision
                avg_recall += recall
                avg_f_score += f_score
                f_scores.append(f_score)
                execution_time += res_rep_lab['execution_time'].sum()

            n_reps = len(repition)
            all_rows.append({
                'error_type': etype,
                'labeling_budget': label_budget,
                'precision': avg_precision / n_reps,
                'recall': avg_recall / n_reps,
                'f1_score': avg_f_score / n_reps,
                'f1_score_std': np.std(f_scores),
                'tp': tp_sum / n_reps,
                'ec_tpfp': ed_tpfp_sum / n_reps,
                'ec_tpfn': ed_tpfn,
                'execution_time': execution_time / n_reps,
                'n_labeled_cells': n_labeled_cells / n_reps,
                'n_labeled_tuples': n_labeled_tuples / n_reps,
            })

    return pd.DataFrame(all_rows)


def main():
    repition = range(1, 4)
    labeling_budgets = [1, 10]
    sandbox_path = "/home/fatemeh/data/LakeCorrectionBench/datasets/Final_Datasets/isolated_partitioned_base"
    dirty_file_name = "dirty.csv"
    clean_file_name = "clean.csv"
    results_path = "/home/fatemeh/data/LakeCorrectionBench/results_union/isolated/isolated_partitioned_base/exp_baran-enough-labels"
    algorithm = "raha"

    print("Collecting per-table results...")
    result_df = get_results_df(sandbox_path, results_path, algorithm, repition, labeling_budgets)

    print("Counting ground-truth errors per error type...")
    errors_by_type = get_eds_n_errors_by_type(sandbox_path, dirty_file_name, clean_file_name)
    for etype, count in sorted(errors_by_type.items()):
        print(f"  {etype}: {count} errors")

    print("Aggregating results by error type...")
    total_by_type = get_total_results_by_error_type(labeling_budgets, repition, result_df, errors_by_type)

    out_path = os.path.join(results_path, "baran_3iter_by_error_type.csv")
    total_by_type.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(total_by_type.to_string(index=False))


if __name__ == '__main__':
    main()
