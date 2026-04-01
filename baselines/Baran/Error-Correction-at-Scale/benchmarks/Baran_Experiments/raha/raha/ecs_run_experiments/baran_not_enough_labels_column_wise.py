import copy
import html
import logging
import os
import pickle
import random
import re
from pathlib import Path
from random import shuffle

import hydra
import pandas as pd
from raha.correction_column_wise import main


def value_normalizer(value):
    """
    This method takes a value and minimally normalizes it.
    """
    value = html.unescape(value)
    value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
    value = value.strip("\t\n ")
    return value

def run_raha(dataset_path, results_path, labeling_budget, exec_number, columns_budget_dict):
    main(results_path, dataset_path, os.path.basename(dataset_path),
         labeling_budget, exec_number, columns_budget_dict)


def distribute_labels(labeling_budget_cells, label_per_col, sandbox_path):

    columns_labeled_dict = {}
    for dir in os.listdir(sandbox_path):
        try:
            dataset_path = os.path.join(sandbox_path, dir)
            dirty_df = pd.read_csv(os.path.join(dataset_path, "dirty.csv"), sep=",", header="infer", encoding="utf-8",
                             dtype=str,
                             low_memory=False).map(lambda x: value_normalizer(x) if isinstance(x, str) else x)
            clean_df = pd.read_csv(os.path.join(dataset_path, "clean.csv"), sep=",", header="infer", encoding="utf-8",
                                dtype=str,
                                low_memory=False).map(lambda x: value_normalizer(x) if isinstance(x, str) else x)
            dirty_df.columns = clean_df.columns
            error_mask = ((dirty_df != clean_df) & ~(dirty_df.isna() & clean_df.isna())).values
            erroneous_columns = dirty_df.columns[error_mask.any(axis=0)]
            rows_with_any_error = error_mask.any(axis=1)
            for col in erroneous_columns:
                col_idx = dirty_df.columns.get_loc(col)
                # Rows with any error AND error in this column
                rows_with_col_error = rows_with_any_error & error_mask[:, col_idx]
                num_rows = rows_with_col_error.sum()
                columns_labeled_dict[dir, col_idx, num_rows] = False
        except Exception as e:
            print(dir, e)
            
    columns_budget_dict = {col:0 for col in columns_labeled_dict.keys()}
    assigned_labels = 0
    while assigned_labels + label_per_col <= labeling_budget_cells:
        if sum(columns_labeled_dict.values()) == 1:
            columns_labeled_dict.values = [False for _ in columns_labeled_dict.values()]
        while True:
            selected_col = random.choice(list(columns_labeled_dict.keys()))
            if selected_col[2] > label_per_col: # enough rows with errors
                break
        columns_budget_dict[selected_col] += label_per_col
        assigned_labels += label_per_col
    return columns_budget_dict


def run_experiments(sandbox_path, results_path, labeling_budget_cells, exec_number, labels_per_col):
    for label_budget in labeling_budget_cells:
        logging.info(f"label_budget: {label_budget}")
        columns_budget_dict = distribute_labels(label_budget, labels_per_col, sandbox_path)
        datasets = set(column[0] for column in columns_budget_dict.keys())
        for dataset in datasets:
            logging.info(f"dataset - col: {dataset}")
            try:
                dataset_columns_budget = {col[1]: budget for col, budget in columns_budget_dict.items() if col[0] == dataset}
                if sum(dataset_columns_budget.values()) > 0:
                    results_path_raha = os.path.join(results_path,
                                                     "results_" + str(label_budget) + "_" + str(exec_number))
                    if not os.path.exists(results_path_raha):
                        os.makedirs(results_path_raha)
                    dataset_path = os.path.join(sandbox_path, dataset)
                    run_raha(dataset_path, results_path_raha, label_budget, exec_number,
                             columns_budget_dict=dataset_columns_budget)
                else:
                    logging.info(f"dataset {dataset} has no labeling budget")
            except Exception as e:
                print(dataset, e)

def get_num_erroneous_columns(sandbox_path):
    n_all_erroneous_columns = 0
    for dir in os.listdir(sandbox_path):
        try:
            dataset_path = os.path.join(sandbox_path, dir)
            dirty_df = pd.read_csv(os.path.join(dataset_path, "dirty.csv"), sep=",", header="infer", encoding="utf-8",
                             dtype=str,
                             low_memory=False)
            clean_df = pd.read_csv(os.path.join(dataset_path, "clean.csv"), sep=",", header="infer", encoding="utf-8",
                                dtype=str,
                                low_memory=False)
            dirty_df.columns = clean_df.columns
            error_mask = ((dirty_df != clean_df) & ~(dirty_df.isna() & clean_df.isna())).values
            erroneous_columns = dirty_df.columns[error_mask.any(axis=0)]
            n_all_erroneous_columns += len(erroneous_columns)
        except Exception as e:
            print(dir, e)
            continue
    return n_all_erroneous_columns

@hydra.main(version_base=None, config_path="hydra_configs", config_name="column_wise")
def start(cfg):
    logging.basicConfig(filename=str(Path(cfg["logs"]["path_to_log_file"]).resolve()),
                        level=logging.DEBUG)

    repetitions = range(1, cfg["shared"]["repetitions"] + 1)
    labeling_budget_fractions = cfg["experiment"]["labeling_budgets"]
    sandbox_path = Path(cfg["shared"]["sandbox_path"]).resolve()
    n_err_cols = get_num_erroneous_columns(sandbox_path)
    labeling_budget_cells = [round(n_err_cols * x) for x in labeling_budget_fractions]  
    labels_per_col_list = cfg["experiment"]["labels_per_column"]

    
    dataset_name = sandbox_path.stem
    sandbox_path = str(sandbox_path)

    results_folder = str(Path(cfg["shared"]["results_path"]).resolve().joinpath(dataset_name).resolve())

    for exec_number in repetitions:
        for labels_per_col in labels_per_col_list:
            exp_name = f"raha-non-enough-labels-{labels_per_col}-per-col"
            results_path = os.path.join(results_folder, f"exp_{exp_name}")
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            logging.info(f"exec_number: {exec_number}")
            run_experiments(sandbox_path, results_path, labeling_budget_cells, exec_number, labels_per_col)


if __name__ == "__main__":
    start()
