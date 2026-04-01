import html
import json
import os
import pickle
import re
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def value_normalizer(value):
    """
    This method takes a value and minimally normalizes it.
    """
    if isinstance(value, str):
        value = html.unescape(value)
        value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
        value = value.strip("\t\n ")
    return value

def get_eds_n_errors(base_path, dirty_file_name, clean_file_name):
    total_n_errors = 0
    tables = os.listdir(base_path)
    for table in tables:
        if table.startswith('union_summary') or table.endswith('.json'):
            continue
        dirty_df = pd.read_csv(os.path.join(base_path, table, dirty_file_name), keep_default_na=False, dtype=str).map(lambda x: value_normalizer(x) if isinstance(x, str) else x)
        clean_df = pd.read_csv(os.path.join(base_path, table, clean_file_name), keep_default_na=False, dtype=str).map(lambda x: value_normalizer(x) if isinstance(x, str) else x)
        if dirty_df.shape != clean_df.shape:
            print("Shape mismatch")
        dirty_df.columns = clean_df.columns
        diff = dirty_df.compare(clean_df, keep_shape=True)
        self_diff = diff.xs('self', axis=1, level=1)
        other_diff = diff.xs('other', axis=1, level=1)
        # Custom comparison. True (or 1) only when values are different and not both NaN.
        label_df = ((self_diff != other_diff) & ~(self_diff.isna() & other_diff.isna())).astype(int)
        total_n_errors += label_df.sum().sum()
    return total_n_errors

def get_results_df_cells_limit(sandbox_path, results_path, algorithm, repition, cells_limits):
    """
    Get results from cell limit experiments.
    Results are organized in subdirectories: cell_limit_800/, cell_limit_900/, etc.
    """
    datasets = []
    for dir in os.listdir(sandbox_path):
        if dir.startswith('union_summary') or dir.endswith('.json'):
            continue
        datasets.append(dir)
    print(f"Found {len(datasets)} datasets")

    results_dict = {"algorithm": [], "dataset": [], "execution_number": [],
                        "precision": [], "recall": [], "f1_score": [],
                        "tp": [], "ec_tpfp": [], "ec_tpfn": [], "execution_time": [],
                        "number_of_labeled_tuples": [], "number_of_labeled_cells": [], 
                        "corrected_errors_keys": [], "cell_limit": []}

    for i in repition:
        for cell_limit in cells_limits:
            # Navigate to cell_limit subdirectory
            cell_limit_path = os.path.join(results_path, f"cell_limit_{cell_limit}")
            
            if not os.path.exists(cell_limit_path):
                print(f"Cell limit path does not exist: {cell_limit_path}")
                continue
            
            # Only iterate through files in cell_limit_path, NOT in skipped subfolder
            # (skipped files are excluded from results since they violated the limit)
            for file in os.listdir(cell_limit_path):
                # Skip the skipped folder and non-result files
                if file == 'skipped' or not file.startswith(algorithm) or not file.endswith('.json'):
                    continue
                
                # Check if this file belongs to the current execution number
                # Format: raha_dataset_name_col_0_number#1_$budget$labels.json
                if f"number#{i}" not in file:
                    continue
                
                # Extract dataset name from filename
                dataset = file.split('_col_')[0].replace(f'{algorithm}_', '')
                
                file_path = os.path.join(cell_limit_path, file)
                file_path = str(Path(file_path).resolve())
                
                if os.path.exists(file_path):
                    try:
                        with open(file_path) as f:
                            json_content = json.load(f)
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
                            results_dict['corrected_errors_keys'].append(json_content['corrected_errors_keys'])
                            results_dict['cell_limit'].append(cell_limit)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                else:
                    print(f"The file does not exist: {file_path}")

    result_df = pd.DataFrame.from_dict(results_dict)
    result_df.to_csv(os.path.join(results_path, f"{algorithm}_results_per_table_cells_limit.csv"), index=False)
    return result_df


def get_total_results_cells_limit(cells_limits, repition, result_df, ed_tpfn):
    """
    Get aggregated results for cell limit experiments.
    Metrics are calculated per repetition, then averaged.
    """
    total_results = {"cell_limit": [], "precision": [], "recall": [], "f1_score": [], 
                     "ec_tpfp": [], "ec_tpfn": [], "tp": [], "execution_time": [], 
                     "n_labeled_cells": [], "n_labeled_tuples": [], "n_tables": []}
    
    for cell_limit in cells_limits:
        precisions = []
        recalls = []
        f_scores = []
        tps = []
        ec_tpfps = []
        ec_tpfns = []
        execution_times = []
        n_labeled_cells_list = []
        n_labeled_tuples_list = []
        n_tables_list = []
        
        for rep in repition:
            res_rep = result_df[result_df['execution_number'] == rep]
            res_rep_lab = res_rep[res_rep['cell_limit'] == cell_limit]
            
            if len(res_rep_lab) == 0:
                continue
            
            # Calculate per-repetition metrics
            tp_rep = res_rep_lab['tp'].sum()
            ec_tpfp_rep = res_rep_lab['ec_tpfp'].sum()
            ec_tpfn_rep = res_rep_lab['ec_tpfn'].sum()
            
            if ec_tpfp_rep == 0:
                precision_rep = 0
            else:
                precision_rep = tp_rep / ec_tpfp_rep
            
            if ec_tpfn_rep > 0:
                recall_rep = tp_rep / ec_tpfn_rep
            else:
                recall_rep = 0
            
            if precision_rep + recall_rep > 0:
                f_score_rep = 2 * precision_rep * recall_rep / (precision_rep + recall_rep)
            else:
                f_score_rep = 0
            
            precisions.append(precision_rep)
            recalls.append(recall_rep)
            f_scores.append(f_score_rep)
            tps.append(tp_rep)
            ec_tpfps.append(ec_tpfp_rep)
            ec_tpfns.append(ec_tpfn_rep)
            execution_times.append(res_rep_lab['execution_time'].sum())
            n_labeled_cells_list.append(res_rep_lab['number_of_labeled_cells'].sum())
            n_labeled_tuples_list.append(res_rep_lab['number_of_labeled_tuples'].sum())
            n_tables_list.append(len(res_rep_lab))
        
        # Average across repetitions
        n_reps_with_data = len(precisions)
        if n_reps_with_data > 0:
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f_score = np.mean(f_scores)
            avg_tp = np.mean(tps)
            avg_ec_tpfp = np.mean(ec_tpfps)
            avg_ec_tpfn = np.mean(ec_tpfns)
            avg_execution_time = np.mean(execution_times)
            avg_n_labeled_cells = np.mean(n_labeled_cells_list)
            avg_n_labeled_tuples = np.mean(n_labeled_tuples_list)
            avg_n_tables = np.mean(n_tables_list)
        else:
            avg_precision = avg_recall = avg_f_score = avg_tp = avg_ec_tpfp = avg_ec_tpfn = 0
            avg_execution_time = avg_n_labeled_cells = avg_n_labeled_tuples = avg_n_tables = 0
        
        total_results['cell_limit'].append(cell_limit)
        total_results['precision'].append(avg_precision)
        total_results['recall'].append(avg_recall)
        total_results['f1_score'].append(avg_f_score)
        total_results['tp'].append(avg_tp)
        total_results['ec_tpfp'].append(avg_ec_tpfp)
        total_results['ec_tpfn'].append(avg_ec_tpfn)
        total_results['execution_time'].append(avg_execution_time)
        total_results['n_labeled_cells'].append(avg_n_labeled_cells)
        total_results['n_labeled_tuples'].append(avg_n_labeled_tuples)
        total_results['n_tables'].append(avg_n_tables)
    
    total_results_df = pd.DataFrame.from_dict(total_results)
    return total_results_df


def get_raha_res_cells_limit(repitions, cells_limits, sandbox_path, results_path, df_path, tp_fn=None):
    """
    Main function to process cell limit experiments.
    """
    algorithm = 'raha'
    result_df = get_results_df_cells_limit(sandbox_path, results_path, algorithm, repitions, cells_limits)
    total_results = get_total_results_cells_limit(cells_limits, repitions, result_df, tp_fn)
    total_results.to_csv(df_path, index=False)
    return total_results

@hydra.main(version_base=None, config_path="../ecs_run_experiments/hydra_configs", config_name="results")
def main(cfg):
    # Adjust these parameters
    repition = range(1, 4)
    # cells_limits = [685, 1259, 1767, 2677, 3951, 4734]   
    cells_limits = []
    
    sandbox_path = "/home/fatemeh/data/cells-limit/LakeCorrectionBench/datasets/Real_Lake_Default_Datasets/merged_strings_default_set_union/mit_dwh/merged"
    dirty_file_name = "dirty.csv"
    clean_file_name = "clean.csv"
    results_path = "/home/fatemeh/data/LakeCorrectionBench/results_mit_pm_runtime/merged/exp_baran-enough-labels"
    df_path = os.path.join(results_path, "baran_cells_limit_3iter.csv")
    
    tp_fn = get_eds_n_errors(sandbox_path, dirty_file_name, clean_file_name)
    print(f"Total errors in lake: {tp_fn}")
    
    total_results = get_raha_res_cells_limit(repition, cells_limits, sandbox_path, results_path, df_path, tp_fn)
    print("\nAggregated Results:")
    print(total_results)

if __name__ == '__main__':
    main()
