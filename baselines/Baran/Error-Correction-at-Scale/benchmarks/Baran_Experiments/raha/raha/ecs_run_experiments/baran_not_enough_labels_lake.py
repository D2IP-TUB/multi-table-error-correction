import logging
import os
import pickle
from pathlib import Path
from random import shuffle

import hydra
import pandas as pd
from raha.correction import main
# from raha.send_log import send_log


def run_baran(dataset_path, results_path, labeling_budget, exec_number):
    main(results_path, dataset_path, os.path.basename(dataset_path),
         labeling_budget, exec_number, column_wise_evaluation=False, column_idx=0)


def distribute_labels(labeling_budget_cells, sandbox_path, results_path):
    datasets_error_shape = dict()
    datasets_budget = dict()
    datasets_num_erroneous_cells = dict()

    num_erroneous_cols = 0
    num_erroneous_rows = 0
    num_erroneous_cells = 0

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
            erroneous_rows = dirty_df.index[error_mask.any(axis=1)]
            datasets_error_shape[dataset_path] = (len(erroneous_rows), len(erroneous_columns))
            datasets_num_erroneous_cells[dataset_path] = error_mask.sum()
            datasets_budget[dataset_path] = 0
            num_erroneous_cols += len(erroneous_columns)
            num_erroneous_rows += len(erroneous_rows)
            num_erroneous_cells += error_mask.sum()
        except Exception as e:
            print(dir, e)

    if labeling_budget_cells % num_erroneous_cols == 0:
        labeling_budget_tuples_per_table = labeling_budget_cells / num_erroneous_cols
        for dataset in datasets_error_shape:
            datasets_budget[dataset] = round(labeling_budget_tuples_per_table)
    else:
        asssigned_labels = 0
        non_eligible_datasets = 0
        while labeling_budget_cells > asssigned_labels and non_eligible_datasets < len(datasets_error_shape):
            dataset_names = list(datasets_num_erroneous_cells.keys())
            shuffle(dataset_names)
            for dataset in dataset_names:
                dataset_num_err_cols = datasets_error_shape[dataset][1]
                remained_labels = labeling_budget_cells - asssigned_labels
                if remained_labels >= dataset_num_err_cols:
                    datasets_budget[dataset] += 1
                    asssigned_labels += dataset_num_err_cols
                    non_eligible_datasets = 0
                else:
                    non_eligible_datasets += 1

    with open(os.path.join(results_path, f"labeling_budget_{labeling_budget_cells}.pickle"), "wb") as f:
        pickle.dump(datasets_budget, f)
    return datasets_budget


def run_experiments(sandbox_path, results_path, labeling_budget_cells, exec_number):
    for label_budget in labeling_budget_cells:
        # send_log(f"Running experiment for labeling budget: {label_budget} for execution number: {exec_number}")
        logging.info(f"label_budget: {label_budget}")
        datasets_budget = distribute_labels(label_budget, sandbox_path, results_path)
        for dataset in datasets_budget:
            logging.info(f"dataset: {dataset}")
            try:
                if datasets_budget[dataset] > 0:

                    results_path_raha = os.path.join(results_path,
                                                     "results_" + str(label_budget) + "_" + str(exec_number))
                    if not os.path.exists(results_path_raha):
                        os.makedirs(results_path_raha)
                    # send_log(f"Running Baran for dataset: {dataset} with labeling budget: {datasets_budget[dataset]} ")
                    run_baran(dataset, results_path_raha, datasets_budget[dataset], exec_number)
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
    
@hydra.main(version_base=None, config_path="hydra_configs", config_name="table_wise")
def start(cfg):
    logging.basicConfig(filename=str(Path(cfg["logs"]["path_to_log_file"]).resolve()),
                        level=logging.DEBUG)

    repetition = range(1, cfg["shared"]["repetitions"] + 1)
    labeling_budget_fractions = cfg["experiment"]["labeling_budgets"]
    sandbox_path = Path(cfg["shared"]["sandbox_path"]).resolve()
    n_err_cols = get_num_erroneous_columns(sandbox_path)
    labeling_budget_cells = [round(n_err_cols * x) for x in labeling_budget_fractions]  
    dataset_name = sandbox_path.stem
    sandbox_path = str(sandbox_path)
    results_folder = str(Path(cfg["shared"]["results_path"]).resolve().joinpath(dataset_name).resolve())
    results_path = os.path.join(results_folder, f"exp_baran-non-enough-labels-RT")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for exec_number in repetition:
        # send_log(f"Running experiment for execution number: {exec_number}")
        logging.info(f"exec_number: {exec_number}")
        run_experiments(sandbox_path, results_path, labeling_budget_cells, exec_number)


if __name__ == "__main__":
    start()
