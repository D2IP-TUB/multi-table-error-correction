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


def load_error_type_map_for_table(sandbox_path, table_name, dirty_file_name="dirty.csv"):
    """
    Load an error-type map for one table from error_map.csv.
    Expects columns: 'row_number', 'column_name', 'error_type'.
    Uses the dirty CSV header to resolve column name -> column index.
    Returns dict {(row, col): error_type} or None if error_map.csv is absent.
    """
    emap_path = os.path.join(sandbox_path, str(table_name), "error_map.csv")
    if not os.path.exists(emap_path):
        return None
    df = pd.read_csv(emap_path, keep_default_na=False, dtype=str, encoding="latin1")
    if df.empty:
        return None
    required = {"row_number", "column_name", "error_type"}
    if not required.issubset(df.columns):
        return None

    dirty_path = os.path.join(sandbox_path, str(table_name), dirty_file_name)
    if not os.path.exists(dirty_path):
        return None
    dirty_cols = list(pd.read_csv(dirty_path, nrows=0, encoding="latin1").columns)
    col_name_to_idx = {name: idx for idx, name in enumerate(dirty_cols)}

    error_map = {}
    for _, row in df.iterrows():
        try:
            r = int(row["row_number"])
        except Exception:
            continue
        col_name = str(row["column_name"]).strip()
        c = col_name_to_idx.get(col_name)
        if c is None:
            continue
        et = str(row.get("error_type", "")).strip()
        if et:
            error_map[(r, c)] = et
    return error_map


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


def get_total_errors_by_error_type(base_path):
    """
    Count total errors per error type over the lake using error_map.csv.
    Each row in error_map.csv represents one erroneous cell.
    Returns (ec_tpfn_by_type, total_errors_all_types).
    """
    ec_tpfn_by_type = {}
    total = 0
    for table in os.listdir(base_path):
        if table.startswith("union_summary") or table.endswith(".json"):
            continue
        emap_path = os.path.join(base_path, table, "error_map.csv")
        if not os.path.exists(emap_path):
            continue
        df = pd.read_csv(emap_path, keep_default_na=False, dtype=str, encoding="latin1")
        if df.empty or "error_type" not in df.columns:
            continue
        for _, row in df.iterrows():
            et = str(row.get("error_type", "")).strip()
            if not et:
                continue
            ec_tpfn_by_type[et] = ec_tpfn_by_type.get(et, 0) + 1
            total += 1
    return ec_tpfn_by_type, total

def get_results_df(sandbox_path, results_path, algorithm, repition, labeling_budgets, dirty_file_name="dirty.csv"):
    datasets = []
    for dir in os.listdir(sandbox_path):
        if dir.startswith('union_summary') or dir.endswith('.json'):
            continue
        datasets.append(dir)
    print(len(datasets))

    results_dict = {
        "algorithm": [],
        "dataset": [],
        "execution_number": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "tp": [],
        "ec_tpfp": [],
        "ec_tpfn": [],
        "execution_time": [],
        "number_of_labeled_tuples": [],
        "number_of_labeled_cells": [],
        "corrected_errors_keys": [],
        # Per-error-type counts for this (dataset, rep, budget)
        # Stored as JSON: {error_type: {"tp": int, "fp": int, "total": int}, ...}
        "per_error_type_counts": [],
    }

    # Pre-load error-type maps per dataset from error_map.csv.
    error_type_cache = {}
    for ds in datasets:
        error_type_cache[ds] = load_error_type_map_for_table(sandbox_path, ds, dirty_file_name)

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

                    # Per-error-type counts for this result file.
                    error_map = error_type_cache.get(dataset)
                    tp_cells = json_content.get("true_postives_cells", [])
                    tp_set = set(tuple(c) for c in tp_cells)
                    corrected_cells = json_content.get("corrected_errors_keys", [])
                    per_type_counts = {}
                    if error_map is not None:
                        for cell in corrected_cells:
                            if not isinstance(cell, (list, tuple)) or len(cell) < 2:
                                continue
                            row, col = cell[0], cell[1]
                            if not isinstance(row, int) or not isinstance(col, int):
                                try:
                                    row = int(row)
                                    col = int(col)
                                except Exception:
                                    continue
                            et = error_map.get((row, col))
                            if not et:
                                continue
                            is_tp = (row, col) in tp_set
                            stats = per_type_counts.setdefault(et, {"tp": 0, "fp": 0, "total": 0})
                            stats["total"] += 1
                            if is_tp:
                                stats["tp"] += 1
                            else:
                                stats["fp"] += 1

                    results_dict["algorithm"].append(algorithm)
                    results_dict["dataset"].append(dataset)
                    results_dict["execution_number"].append(i)
                    results_dict["precision"].append(json_content["precision"])
                    results_dict["recall"].append(json_content["recall"])
                    results_dict["f1_score"].append(json_content["f_score"])
                    results_dict["tp"].append(json_content["tp"])
                    results_dict["ec_tpfp"].append(json_content["ec_tpfp"])
                    results_dict["ec_tpfn"].append(json_content["ec_tpfn"])
                    results_dict["execution_time"].append(json_content["execution-time"])
                    results_dict["number_of_labeled_tuples"].append(json_content["number_of_labeled_tuples"])
                    results_dict["number_of_labeled_cells"].append(json_content["number_of_labeled_cells"])
                    results_dict["corrected_errors_keys"].append(json_content["corrected_errors_keys"])
                    results_dict["per_error_type_counts"].append(json.dumps(per_type_counts))
                else:
                    print("The file does not exist: {}".format(file_path))

    result_df = pd.DataFrame.from_dict(results_dict)
    result_df.to_csv(os.path.join(results_path, f"{algorithm}_results_per_table.csv"), index=False)
    return result_df


def get_total_results(labeling_budgets, repition, result_df, ed_tpfn, ec_tpfn_by_type=None):
    total_results = {
        "labeling_budget": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "f1_score_std": [],
        "ec_tpfp": [],
        "ec_tpfn": [],
        "tp": [],
        "execution_time": [],
        "n_labeled_cells": [],
        "n_labeled_tuples": [],
        # Lake-wide per-error-type metrics (JSON per labeling_budget)
        "per_error_type_metrics": [],
    }
    for label_budget in labeling_budgets:
        avg_precision = 0
        avg_recall = 0
        avg_f_score = 0
        tp = 0
        ed_tpfp = 0
        f_scores = []
        execution_time = 0
        n_labeled_cells = 0
        n_labeled_tuples = 0
        # Per-error-type metrics aggregated over reps for this budget.
        # per_type_agg[error_type] -> dict with lists of per-rep metrics.
        per_type_agg = {}
        for rep in repition:
            res_rep = result_df[result_df['execution_number'] == rep]
            res_rep_lab = res_rep[res_rep['number_of_labeled_tuples'] == label_budget]
            tp += res_rep_lab['tp'].sum()
            ed_tpfp += res_rep_lab['ec_tpfp'].sum()
            n_labeled_cells += res_rep_lab['number_of_labeled_cells'].sum()
            n_labeled_tuples += res_rep_lab['number_of_labeled_tuples'].sum()
            if res_rep_lab['ec_tpfp'].sum() == 0:
                precision = 0
                recall = 0
                f_score = 0
            else:
                precision = res_rep_lab['tp'].sum() / res_rep_lab['ec_tpfp'].sum()
                recall = res_rep_lab['tp'].sum() / res_rep_lab['ec_tpfn'].sum()
                f_score = 2 * precision * recall / (precision + recall)
            avg_precision += precision
            avg_recall += recall
            avg_f_score += f_score
            f_scores.append(f_score)
            execution_time += res_rep_lab['execution_time'].sum()

            # Aggregate per-error-type counts for this (budget, rep) over all tables
            per_type_counts_rep = {}
            for _, row in res_rep_lab.iterrows():
                per_type_json = row.get("per_error_type_counts")
                if not isinstance(per_type_json, str) or not per_type_json:
                    continue
                try:
                    per_type = json.loads(per_type_json)
                except Exception:
                    continue
                for et, stats in per_type.items():
                    if not isinstance(stats, dict):
                        continue
                    tp_et = int(stats.get("tp", 0))
                    total_et = int(stats.get("total", 0))
                    if total_et <= 0 and tp_et <= 0:
                        continue
                    agg_counts = per_type_counts_rep.setdefault(et, {"tp": 0, "total": 0})
                    agg_counts["tp"] += tp_et
                    agg_counts["total"] += total_et

            # Convert per-type counts for this rep into precision/recall/f1, then
            # accumulate per-type metrics across reps.
            if ec_tpfn_by_type is not None:
                for et, stats in per_type_counts_rep.items():
                    tp_rep_et = stats["tp"]
                    ec_tpfp_rep_et = stats["total"]
                    p_et = tp_rep_et / ec_tpfp_rep_et if ec_tpfp_rep_et > 0 else 0.0
                    denom_et = ec_tpfn_by_type.get(et, 0)
                    r_et = tp_rep_et / denom_et if denom_et > 0 else 0.0
                    f_et = 2 * p_et * r_et / (p_et + r_et) if (p_et + r_et) > 0 else 0.0

                    agg = per_type_agg.setdefault(
                        et,
                        {
                            "precision": [],
                            "recall": [],
                            "f1": [],
                            "tp": [],
                            "ec_tpfp": [],
                        },
                    )
                    agg["precision"].append(p_et)
                    agg["recall"].append(r_et)
                    agg["f1"].append(f_et)
                    agg["tp"].append(tp_rep_et)
                    agg["ec_tpfp"].append(ec_tpfp_rep_et)

        precision = avg_precision / len(repition)
        recall = avg_recall / len(repition)
        f_score = avg_f_score / len(repition)
        total_results['labeling_budget'].append(label_budget)
        total_results['precision'].append(precision)
        total_results['recall'].append(recall)
        total_results['f1_score'].append(f_score)
        total_results['f1_score_std'].append(np.std(f_scores)) 
        total_results['tp'].append(tp / len(repition))
        total_results['ec_tpfp'].append(ed_tpfp / len(repition))
        total_results['ec_tpfn'].append(ed_tpfn)
        total_results['execution_time'].append(execution_time / len(repition))
        total_results['n_labeled_cells'].append(n_labeled_cells / len(repition))
        total_results['n_labeled_tuples'].append(n_labeled_tuples / len(repition))
        # Lake-wide per-error-type metrics (averaged over repetitions)
        per_error_type_metrics = {}
        if ec_tpfn_by_type is not None:
            for et, agg in per_type_agg.items():
                if not agg["precision"]:
                    continue
                per_error_type_metrics[et] = {
                    "precision": float(np.mean(agg["precision"])),
                    "recall": float(np.mean(agg["recall"])),
                    "f1": float(np.mean(agg["f1"])),
                    "tp": float(np.mean(agg["tp"])),
                    "ec_tpfp": float(np.mean(agg["ec_tpfp"])),
                    "ec_tpfn": float(ec_tpfn_by_type.get(et, 0)),
                }
        total_results["per_error_type_metrics"].append(json.dumps(per_error_type_metrics))
    total_results_df = pd.DataFrame.from_dict(total_results)
    return total_results_df


def get_results_df_non_standard(sandbox_path, results_path, algorithm, repition, labeling_budgets, dirty_file_name="dirty.csv"):
    results_dict = {
        "algorithm": [],
        "dataset": [],
        "execution_number": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "tp": [],
        "ec_tpfp": [],
        "ec_tpfn": [],
        "execution_time": [],
        "number_of_labeled_tuples": [],
        "number_of_labeled_cells": [],
        "corrected_errors_keys": [],
        "total_labeling_budget_exp": [],
        # Per-error-type counts for this (dataset, rep, budget)
        "per_error_type_counts": [],
    }

    # Pre-load error-type maps per dataset from error_map.csv.
    datasets = []
    for dir in os.listdir(sandbox_path):
        if dir.startswith("union_summary") or dir.endswith(".json"):
            continue
        datasets.append(dir)
    error_type_cache = {}
    for ds in datasets:
        error_type_cache[ds] = load_error_type_map_for_table(sandbox_path, ds, dirty_file_name)

    for i in repition:
        for label_budget in labeling_budgets:
            files_res_path = results_path + f"/results_{label_budget}_{i}"
            for file in os.listdir(files_res_path):
                if not file.startswith(algorithm) or not file.endswith('.json'):
                    continue
                dataset = file.split('_col_')[0]
                file_path = files_res_path + '/' + file
                file_path = str(Path(file_path).resolve())
                if os.path.exists(file_path):
                    with open(file_path) as file:
                        json_content = json.load(file)

                    # Per-error-type counts for this result file.
                    error_map = error_type_cache.get(dataset)
                    tp_cells = json_content.get("true_postives_cells", [])
                    tp_set = set(tuple(c) for c in tp_cells)
                    corrected_cells = json_content.get("corrected_errors_keys", [])
                    per_type_counts = {}
                    if error_map is not None:
                        for cell in corrected_cells:
                            if not isinstance(cell, (list, tuple)) or len(cell) < 2:
                                continue
                            row, col = cell[0], cell[1]
                            if not isinstance(row, int) or not isinstance(col, int):
                                try:
                                    row = int(row)
                                    col = int(col)
                                except Exception:
                                    continue
                            et = error_map.get((row, col))
                            if not et:
                                continue
                            is_tp = (row, col) in tp_set
                            stats = per_type_counts.setdefault(et, {"tp": 0, "fp": 0, "total": 0})
                            stats["total"] += 1
                            if is_tp:
                                stats["tp"] += 1
                            else:
                                stats["fp"] += 1

                    results_dict["algorithm"].append(algorithm)
                    results_dict["dataset"].append(dataset)
                    results_dict["execution_number"].append(i)
                    results_dict["precision"].append(json_content["precision"])
                    results_dict["recall"].append(json_content["recall"])
                    results_dict["f1_score"].append(json_content["f_score"])
                    results_dict["tp"].append(json_content["tp"])
                    results_dict["ec_tpfp"].append(json_content["ec_tpfp"])
                    results_dict["ec_tpfn"].append(json_content["ec_tpfn"])
                    results_dict["execution_time"].append(json_content["execution-time"])
                    results_dict["number_of_labeled_tuples"].append(json_content["number_of_labeled_tuples"])
                    results_dict["number_of_labeled_cells"].append(json_content["number_of_labeled_cells"])
                    results_dict["corrected_errors_keys"].append(json_content["corrected_errors_keys"])
                    results_dict["total_labeling_budget_exp"].append(label_budget)
                    results_dict["per_error_type_counts"].append(json.dumps(per_type_counts))
                else:
                    print("The file does not exist: {}".format(file_path))

    result_df = pd.DataFrame.from_dict(results_dict)
    result_df.to_csv(os.path.join(results_path, f"{algorithm}_results_per_table.csv"), index=False)
    return result_df

def get_total_results_non_standard(labeling_budgets, repition, result_df, ed_tpfn, ec_tpfn_by_type=None):
    total_results = {
        "labeling_budget": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "f1_score_std": [],
        "ec_tpfp": [],
        "ec_tpfn": [],
        "tp": [],
        "execution_time": [],
        "n_labeled_cells": [],
        "n_labeled_tuples": [],
        # Lake-wide per-error-type metrics (JSON per labeling_budget)
        "per_error_type_metrics": [],
    }
    for label_budget in labeling_budgets:
        avg_precision = 0
        avg_recall = 0
        avg_f_score = 0
        tp = 0
        ed_tpfp = 0
        f_scores = []
        execution_time = 0
        n_labeled_cells = 0
        n_labeled_tuples = 0
        # Per-error-type metrics aggregated over reps for this budget.
        per_type_agg = {}
        for rep in repition:
            res_rep = result_df[result_df['execution_number'] == rep]
            res_rep_lab = res_rep[res_rep['total_labeling_budget_exp'] == label_budget]
            tp += res_rep_lab['tp'].sum()
            ed_tpfp += res_rep_lab['ec_tpfp'].sum()
            n_labeled_cells += res_rep_lab['number_of_labeled_cells'].sum()
            n_labeled_tuples += res_rep_lab['number_of_labeled_tuples'].sum()
            if res_rep_lab['ec_tpfp'].sum() == 0:
                precision = 0
                recall = 0
                f_score = 0
            else:
                precision = res_rep_lab['tp'].sum() / res_rep_lab['ec_tpfp'].sum()
                recall = res_rep_lab['tp'].sum() / ed_tpfn
                f_score = 2 * precision * recall / (precision + recall)
            avg_precision += precision
            avg_recall += recall
            avg_f_score += f_score
            f_scores.append(f_score)
            execution_time += res_rep_lab['execution_time'].sum()

            # Aggregate per-error-type counts for this (budget, rep) over all tables
            per_type_counts_rep = {}
            for _, row in res_rep_lab.iterrows():
                per_type_json = row.get("per_error_type_counts")
                if not isinstance(per_type_json, str) or not per_type_json:
                    continue
                try:
                    per_type = json.loads(per_type_json)
                except Exception:
                    continue
                for et, stats in per_type.items():
                    if not isinstance(stats, dict):
                        continue
                    tp_et = int(stats.get("tp", 0))
                    total_et = int(stats.get("total", 0))
                    if total_et <= 0 and tp_et <= 0:
                        continue
                    agg_counts = per_type_counts_rep.setdefault(et, {"tp": 0, "total": 0})
                    agg_counts["tp"] += tp_et
                    agg_counts["total"] += total_et

            if ec_tpfn_by_type is not None:
                for et, stats in per_type_counts_rep.items():
                    tp_rep_et = stats["tp"]
                    ec_tpfp_rep_et = stats["total"]
                    p_et = tp_rep_et / ec_tpfp_rep_et if ec_tpfp_rep_et > 0 else 0.0
                    denom_et = ec_tpfn_by_type.get(et, 0)
                    r_et = tp_rep_et / denom_et if denom_et > 0 else 0.0
                    f_et = 2 * p_et * r_et / (p_et + r_et) if (p_et + r_et) > 0 else 0.0

                    agg = per_type_agg.setdefault(
                        et,
                        {
                            "precision": [],
                            "recall": [],
                            "f1": [],
                            "tp": [],
                            "ec_tpfp": [],
                        },
                    )
                    agg["precision"].append(p_et)
                    agg["recall"].append(r_et)
                    agg["f1"].append(f_et)
                    agg["tp"].append(tp_rep_et)
                    agg["ec_tpfp"].append(ec_tpfp_rep_et)

        precision = avg_precision / len(repition)
        recall = avg_recall / len(repition)
        f_score = avg_f_score / len(repition)
        total_results['labeling_budget'].append(label_budget)
        total_results['precision'].append(precision)
        total_results['recall'].append(recall)
        total_results['f1_score'].append(f_score)
        total_results['f1_score_std'].append(np.std(f_scores)) 
        total_results['tp'].append(tp / len(repition))
        total_results['ec_tpfp'].append(ed_tpfp / len(repition))
        total_results['ec_tpfn'].append(ed_tpfn)
        total_results['execution_time'].append(execution_time / len(repition))
        total_results['n_labeled_cells'].append(n_labeled_cells / len(repition))
        total_results['n_labeled_tuples'].append(n_labeled_tuples / len(repition))
        # Lake-wide per-error-type metrics (averaged over repetitions)
        per_error_type_metrics = {}
        if ec_tpfn_by_type is not None:
            for et, agg in per_type_agg.items():
                if not agg["precision"]:
                    continue
                per_error_type_metrics[et] = {
                    "precision": float(np.mean(agg["precision"])),
                    "recall": float(np.mean(agg["recall"])),
                    "f1": float(np.mean(agg["f1"])),
                    "tp": float(np.mean(agg["tp"])),
                    "ec_tpfp": float(np.mean(agg["ec_tpfp"])),
                    "ec_tpfn": float(ec_tpfn_by_type.get(et, 0)),
                }
        total_results["per_error_type_metrics"].append(json.dumps(per_error_type_metrics))
    total_results_df = pd.DataFrame.from_dict(total_results)
    return total_results_df


def get_raha_res(repitions, labeling_budgets, sandbox_path, results_path, df_path, tp_fn=None, dirty_file_name="dirty.csv"):
    algorithm = "raha"
    result_df = get_results_df(sandbox_path, results_path, algorithm, repitions, labeling_budgets, dirty_file_name)
    ec_tpfn_by_type, _ = get_total_errors_by_error_type(sandbox_path)
    total_results = get_total_results(labeling_budgets, repitions, result_df, tp_fn, ec_tpfn_by_type)
    total_results.to_csv(df_path, index=False)
    return total_results

def get_baran_res_non_standard(repitions, labeling_budgets, sandbox_path, results_path, df_path, tp_fn=None, dirty_file_name="dirty.csv"):
    algorithm = "raha"
    result_df = get_results_df_non_standard(sandbox_path, results_path, algorithm, repitions, labeling_budgets, dirty_file_name)
    ec_tpfn_by_type, _ = get_total_errors_by_error_type(sandbox_path)
    total_results = get_total_results_non_standard(labeling_budgets, repitions, result_df, tp_fn, ec_tpfn_by_type)
    total_results.to_csv(df_path, index=False)
    return total_results

@hydra.main(version_base=None, config_path="../ecs_run_experiments/hydra_configs", config_name="results")
def main(cfg):
    # repition = range(1, cfg["shared"]["repetitions"] + 1)
    repition = range(1, 4)
    # variant = cfg["results"]["variant"]
    variant = "standard"
    # labeling_budgets = cfg["results"]["labeling_budget"]
    labeling_budgets =  [1, 10]                          
    # sandbox_path = str(Path(cfg["shared"]["sandbox_path"]).resolve())
    sandbox_path = "/home/fatemeh/data/cells-limit/LakeCorrectionBench/datasets/Real_Lake_Default_Datasets/merged_strings_default_set_union/mit_dwh/merged"
    # dirty_file_name = cfg["shared"]["dirty_file_name"]
    dirty_file_name = "dirty.csv"
    # clean_file_name = cfg["shared"]["clean_file_name"]
    clean_file_name = "clean.csv"
    # results_path = str(Path(cfg["results"]["path_to_experiment_results_folder"]).resolve()) # execution not experiment folder
    results_path = "/home/fatemeh/data/LakeCorrectionBench/results_mit_pm_runtime/merged/exp_baran-enough-labels"
    # df_path = str(Path(cfg["results"]["path_to_benchmark_dataframe"]).resolve())
    df_path = "/home/fatemeh/data/LakeCorrectionBench/results_mit_pm_runtime/merged/exp_baran-enough-labels/baran_3iter_with_types.csv"
    if variant == "standard":
        tp_fn = get_eds_n_errors(sandbox_path, dirty_file_name, clean_file_name)
        total_results = get_raha_res(repition, labeling_budgets, sandbox_path, results_path, df_path, tp_fn, dirty_file_name)
    else:
        print("variant is not standard")
        tp_fn = get_eds_n_errors(sandbox_path, dirty_file_name, clean_file_name)
        total_results = get_baran_res_non_standard(repition, labeling_budgets, sandbox_path, results_path, df_path, tp_fn, dirty_file_name)

if __name__ == '__main__':
    main()
