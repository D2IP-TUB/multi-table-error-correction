import html
import json
import logging
import math
import os
import pickle
import random
import re
import time
from pathlib import Path

import hydra
import pandas as pd
from raha.correction import main
from raha.correction_predefined_samples import main as main_predefined_samples
# from raha.send_log import send_log


def value_normalizer(value):
    """
    This method takes a value and minimally normalizes it.
    """
    value = html.unescape(value)
    value = re.sub(r"[\t\n ]+", " ", value, re.UNICODE)
    value = value.strip("\t\n ")
    return value


def get_num_erroneous_columns(sandbox_path, custom_logger):
    """
    Count the total number of erroneous columns across all datasets in the lake.
    """
    n_all_erroneous_columns = 0
    for dir in os.listdir(sandbox_path):
        try:
            dataset_path = os.path.join(sandbox_path, dir)
            if not os.path.isdir(dataset_path):
                continue
            dirty_df = pd.read_csv(
                os.path.join(dataset_path, "dirty.csv"),
                sep=",",
                header="infer",
                encoding="utf-8",
                dtype=str,
                low_memory=False
            )
            clean_df = pd.read_csv(
                os.path.join(dataset_path, "clean.csv"),
                sep=",",
                header="infer",
                encoding="utf-8",
                dtype=str,
                low_memory=False
            )
            # Normalize values
            dirty_df = dirty_df.apply(lambda col: col.apply(lambda x: value_normalizer(x) if isinstance(x, str) else x))
            clean_df = clean_df.apply(lambda col: col.apply(lambda x: value_normalizer(x) if isinstance(x, str) else x))
            
            dirty_df.columns = clean_df.columns
            error_mask = ((dirty_df != clean_df) & ~(dirty_df.isna() & clean_df.isna())).values
            erroneous_columns = dirty_df.columns[error_mask.any(axis=0)]
            n_erroneous_cols_in_table = len(erroneous_columns)
            n_all_erroneous_columns += n_erroneous_cols_in_table
            custom_logger.info(f"  {dir}: {n_erroneous_cols_in_table} erroneous columns")
        except Exception as e:
            custom_logger.error(f"Error processing {dir}: {e}")
            continue
    return n_all_erroneous_columns


@hydra.main(version_base=None, config_path="hydra_configs", config_name="standard")
def start(cfg):
    exp_name = "baran-enough-labels"
    log_file = str(Path(cfg["logs"]["path_to_log_file"]).resolve())
    par_log_directory = os.path.dirname(log_file)
    if not os.path.exists(par_log_directory):
        os.makedirs(par_log_directory)
    log_file = os.path.join(par_log_directory, f"time_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{exp_name}.log")
    print(f"Log file should be: {log_file}")
    
    with open(log_file, 'w') as f:
        f.write("TEST: Direct file write works\n")
    
    print("Direct file write completed - check if file exists")
    
    custom_logger = logging.getLogger('baran_experiment')
    custom_logger.setLevel(logging.DEBUG)
    
    custom_logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    custom_logger.addHandler(fh)
    
    custom_logger.info("TEST: Custom logger message")
    print("Custom logger test completed")
    
    custom_logger.info(f"Starting experiment: {exp_name}")
    

    repetition = range(1, cfg["shared"]["repetitions"] + 1)
    with_sampling = cfg["experiment"]["sampling"]
    predefined_samples_path = cfg["experiment"]["predefined_samples_path"]
    sandbox_path = Path(cfg["shared"]["sandbox_path"]).resolve()
    dataset_name = sandbox_path.stem
    sandbox_path_str = str(sandbox_path)
    results_folder = str(Path(cfg["shared"]["results_path"]).resolve().joinpath(dataset_name).resolve())
    
    # Check if we have labeling_budgets_cells_limit config
    use_cells_limit = "labeling_budgets_cells_limit" in cfg["experiment"] and cfg["experiment"]["labeling_budgets_cells_limit"] is not None
    
    if use_cells_limit:
        custom_logger.info("Using labeling_budgets_cells_limit configuration")
        # Count erroneous columns in the lake
        n_erroneous_columns = get_num_erroneous_columns(sandbox_path_str, custom_logger)
        custom_logger.info(f"Total erroneous columns in lake: {n_erroneous_columns}")
        
        if n_erroneous_columns == 0:
            custom_logger.error("No erroneous columns found in the lake!")
            return
        
        # Calculate labeling budgets per table (in tuples) using ceil division
        cells_limits = cfg["experiment"]["labeling_budgets_cells_limit"]
        labeling_budgets_per_table = [math.ceil(cell_limit / n_erroneous_columns) for cell_limit in cells_limits]
        
        # Store as tuples (cell_limit, budget_per_table) for tracking
        labeling_budgets = list(zip(cells_limits, labeling_budgets_per_table))
        custom_logger.info(f"Cell limits: {cells_limits}")
        custom_logger.info(f"Calculated labeling budgets per table (tuples): {labeling_budgets_per_table}")
        exp_name = "baran-enough-labels-cells-limit"
    else:
        custom_logger.info("Using standard labeling_budgets configuration")
        # For standard mode, use None as cell_limit
        labeling_budgets = [(None, budget) for budget in cfg["experiment"]["labeling_budgets"]]

    results_path = os.path.join(results_folder, f"exp_{exp_name}")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    datasets = []

    for dir in os.listdir(sandbox_path_str):
        datasets.append(os.path.join(sandbox_path_str, dir))

    # Randomly shuffle datasets before any iteration
    random.shuffle(datasets)
    
    custom_logger.info(f"datasets: {datasets}")

    if not with_sampling:
        with open(predefined_samples_path, 'rb') as f:
            predefined_samples = pickle.load(f)
    else:
        predefined_samples = None
    s_time = time.time()
    n_datasets = len(datasets)
    for execution_number in repetition:
        # send_log(f"Experiment Name: {exp_name}, Execution number: {execution_number} for dataset: {dataset_name} - {n_datasets} tables")
        custom_logger.info(f"Experiment Name: {exp_name}, Execution number: {execution_number} for dataset: {dataset_name} - {n_datasets} tables")
        for cell_limit, labeling_budget in labeling_budgets:
            # Create separate directory for each cell limit
            if cell_limit is not None:
                results_path_for_budget = os.path.join(results_path, f"cell_limit_{cell_limit}")
            else:
                results_path_for_budget = results_path
            
            if not os.path.exists(results_path_for_budget):
                os.makedirs(results_path_for_budget)
            
            i_l = 0
            total_cells_labeled = 0  # Track actual cells labeled
            skipped_tables = []  # Track tables that were skipped
            
            for dataset_path in datasets:
                try:
                    dataset_name = os.path.basename(dataset_path)
                    
                    # If using cell limit, check if we've reached the limit
                    if cell_limit is not None and total_cells_labeled >= cell_limit:
                        custom_logger.info(
                            f"Cell limit reached ({total_cells_labeled}/{cell_limit}). "
                            f"Skipping remaining tables."
                        )
                        break
                    
                    # Execute the table and get the number of labeled cells
                    if with_sampling:
                        cells_labeled = main(results_path_for_budget, dataset_path, dataset_name, labeling_budget, execution_number,
                            column_wise_evaluation=False, column_idx=0)
                    else:                   
                        predefined_sampled_tuples = predefined_samples[execution_number][labeling_budget]
                        cells_labeled = main_predefined_samples(results_path_for_budget, dataset_path, dataset_name, labeling_budget, execution_number,
                            column_wise_evaluation=False, column_idx=0, sampled_tuples=list(predefined_sampled_tuples))
                    
                    # Track actual cells labeled after execution
                    if cell_limit is not None:
                        total_cells_labeled += cells_labeled
                        custom_logger.info(
                            f"Table {dataset_name}: labeled {cells_labeled} cells. "
                            f"Total so far: {total_cells_labeled}/{cell_limit}"
                        )
                        
                        # If cell limit is violated, move result files to skipped folder and try next table
                        if total_cells_labeled > cell_limit:
                            skipped_folder = os.path.join(results_path_for_budget, "skipped")
                            if not os.path.exists(skipped_folder):
                                os.makedirs(skipped_folder)
                            
                            # Move result files to skipped folder
                            col_idx = 0  # from the main call
                            result_file_base = f"raha_{dataset_name}_col_{col_idx}_number#{execution_number}_${labeling_budget}"
                            result_files = [
                                result_file_base + "$labels.json",
                                result_file_base + "labels_samples.pickle"
                            ]
                            
                            for result_file in result_files:
                                src_path = os.path.join(results_path_for_budget, result_file)
                                if os.path.exists(src_path):
                                    dst_path = os.path.join(skipped_folder, result_file)
                                    os.rename(src_path, dst_path)
                                    custom_logger.info(f"Moved {result_file} to skipped folder")
                            
                            # Reduce the total cells labeled and track as skipped
                            total_cells_labeled -= cells_labeled
                            skipped_tables.append(dataset_path)
                            custom_logger.info(
                                f"Cell limit would be violated ({total_cells_labeled + cells_labeled}/{cell_limit}). "
                                f"Skipped {dataset_name}, reduced cell count back to {total_cells_labeled}. "
                                f"Continuing with next table."
                            )
                            i_l += 1
                            continue
                        
                except Exception as e:
                    custom_logger.error(f"Error processing {dataset_name}: {e}")
                i_l += 1
                custom_logger.info(f"Finished execution {execution_number} for labeling budget {labeling_budget} on dataset {dataset_name}")
                if i_l % 10 == 0:
                    # send_log(f"Completed {i_l} datasets for execution {execution_number} for experiment {exp_name}")
                    custom_logger.info(f"Completed {i_l} datasets for execution {execution_number} for experiment {exp_name}")
            
            # Try skipped tables with reduced budgets if cell limit not reached
            if cell_limit is not None and skipped_tables and total_cells_labeled < cell_limit:
                # Randomly shuffle skipped tables before retrying
                random.shuffle(skipped_tables)
                
                current_budget = labeling_budget - 1
                # Track which tables to stop retrying (they're too large even with budget-1)
                permanently_skipped = set()
                
                while current_budget > 0 and skipped_tables and total_cells_labeled < cell_limit:
                    custom_logger.info(
                        f"Cell limit not reached ({total_cells_labeled}/{cell_limit}). "
                        f"Retrying {len(skipped_tables)} skipped tables with budget {current_budget}."
                    )
                    remaining_skipped = []
                    
                    for dataset_path in skipped_tables:
                        # Skip tables we already know are too large
                        if dataset_path in permanently_skipped:
                            remaining_skipped.append(dataset_path)
                            continue
                            
                        try:
                            dataset_name = os.path.basename(dataset_path)
                            
                            if total_cells_labeled >= cell_limit:
                                remaining_skipped.append(dataset_path)
                                continue
                            
                            # Execute with reduced budget
                            if with_sampling:
                                cells_labeled = main(results_path_for_budget, dataset_path, dataset_name, current_budget, execution_number,
                                    column_wise_evaluation=False, column_idx=0)
                            else:
                                predefined_sampled_tuples = predefined_samples[execution_number][current_budget]
                                cells_labeled = main_predefined_samples(results_path_for_budget, dataset_path, dataset_name, current_budget, execution_number,
                                    column_wise_evaluation=False, column_idx=0, sampled_tuples=list(predefined_sampled_tuples))
                            
                            total_cells_labeled += cells_labeled
                            custom_logger.info(
                                f"Retry {dataset_name} with budget {current_budget}: labeled {cells_labeled} cells. "
                                f"Total: {total_cells_labeled}/{cell_limit}"
                            )
                            
                            # Check if this violated the limit
                            if total_cells_labeled > cell_limit:
                                skipped_folder = os.path.join(results_path_for_budget, "skipped")
                                if not os.path.exists(skipped_folder):
                                    os.makedirs(skipped_folder)
                                
                                col_idx = 0
                                result_file_base = f"raha_{dataset_name}_col_{col_idx}_number#{execution_number}_${current_budget}"
                                result_files = [
                                    result_file_base + "$labels.json",
                                    result_file_base + "labels_samples.pickle"
                                ]
                                
                                for result_file in result_files:
                                    src_path = os.path.join(results_path_for_budget, result_file)
                                    if os.path.exists(src_path):
                                        dst_path = os.path.join(skipped_folder, result_file)
                                        os.rename(src_path, dst_path)
                                
                                total_cells_labeled -= cells_labeled
                                
                                # If we're already at budget=1 and still violates, stop retrying this table
                                if current_budget == 1:
                                    permanently_skipped.add(dataset_path)
                                    custom_logger.info(f"{dataset_name} too large even with budget=1, permanently skipped")
                                
                                remaining_skipped.append(dataset_path)
                                custom_logger.info(f"Still violates limit with budget={current_budget}, will retry with lower budget")
                            else:
                                # Success - don't add to remaining_skipped
                                custom_logger.info(f"Successfully processed {dataset_name} with reduced budget {current_budget}")
                            
                        except Exception as e:
                            custom_logger.error(f"Error retrying {dataset_name} with budget {current_budget}: {e}")
                            remaining_skipped.append(dataset_path)
                    
                    # Stop early if no progress was made (all remaining are permanently skipped)
                    if len(remaining_skipped) == len(skipped_tables) and len(permanently_skipped) == len(skipped_tables):
                        custom_logger.info("All remaining tables are too large even with budget=1, stopping retry loop")
                        break
                    
                    skipped_tables = remaining_skipped
                    current_budget -= 1
            
            # If cell limit not reached, try upgrading existing tables with higher budgets
            if cell_limit is not None and total_cells_labeled < cell_limit:
                custom_logger.info(
                    f"Cell limit not reached ({total_cells_labeled}/{cell_limit}). "
                    f"Attempting to upgrade tables with higher budgets to maximize usage."
                )
                
                # Create list of all processed tables (not skipped)
                all_datasets_shuffled = datasets.copy()
                random.shuffle(all_datasets_shuffled)
                
                upgrade_budget = labeling_budget + 1
                current_working_budget = labeling_budget  # Track the budget of current files
                upgraded_this_round = set()  # Track tables upgraded in current budget level
                
                while total_cells_labeled < cell_limit:
                    custom_logger.info(
                        f"Trying to upgrade tables with budget {upgrade_budget}. "
                        f"Current: {total_cells_labeled}/{cell_limit}"
                    )
                    
                    upgraded_any = False
                    upgraded_this_round.clear()  # Clear for new budget level
                    
                    for dataset_path in all_datasets_shuffled:
                        try:
                            dataset_name = os.path.basename(dataset_path)
                            col_idx = 0
                            
                            # Skip if already upgraded in this budget level
                            if dataset_name in upgraded_this_round:
                                continue
                            
                            # Check if result file exists for current working budget
                            old_result_file = os.path.join(results_path_for_budget, f"raha_{dataset_name}_col_{col_idx}_number#{execution_number}_${current_working_budget}$labels.json")
                            if not os.path.exists(old_result_file):
                                continue
                            
                            # Read old result to get old cell count
                            with open(old_result_file) as f:
                                old_result = json.load(f)
                                old_cells = old_result.get('number_of_labeled_cells', 0)
                            
                            # Re-run with higher budget
                            if with_sampling:
                                new_cells = main(results_path_for_budget, dataset_path, dataset_name, upgrade_budget, execution_number,
                                    column_wise_evaluation=False, column_idx=0)
                            else:
                                predefined_sampled_tuples = predefined_samples[execution_number][upgrade_budget]
                                new_cells = main_predefined_samples(results_path_for_budget, dataset_path, dataset_name, upgrade_budget, execution_number,
                                    column_wise_evaluation=False, column_idx=0, sampled_tuples=list(predefined_sampled_tuples))
                            
                            # Check if upgrade fits within cell limit
                            cell_delta = new_cells - old_cells
                            if total_cells_labeled + cell_delta <= cell_limit:
                                # Accept the upgrade - delete old result files and keep new ones
                                old_result_files = [
                                    os.path.join(results_path_for_budget, f"raha_{dataset_name}_col_{col_idx}_number#{execution_number}_${current_working_budget}$labels.json"),
                                    os.path.join(results_path_for_budget, f"raha_{dataset_name}_col_{col_idx}_number#{execution_number}_${current_working_budget}labels_samples.pickle")
                                ]
                                
                                for old_file in old_result_files:
                                    if os.path.exists(old_file):
                                        os.remove(old_file)
                                
                                total_cells_labeled += cell_delta
                                upgraded_any = True
                                upgraded_this_round.add(dataset_name)  # Mark as upgraded this round
                                custom_logger.info(
                                    f"Upgraded {dataset_name} from budget {current_working_budget} to {upgrade_budget}: "
                                    f"+{cell_delta} cells. Total: {total_cells_labeled}/{cell_limit}"
                                )
                            else:
                                # Upgrade doesn't fit - delete the new result files and keep old ones
                                new_result_files = [
                                    os.path.join(results_path_for_budget, f"raha_{dataset_name}_col_{col_idx}_number#{execution_number}_${upgrade_budget}$labels.json"),
                                    os.path.join(results_path_for_budget, f"raha_{dataset_name}_col_{col_idx}_number#{execution_number}_${upgrade_budget}labels_samples.pickle")
                                ]
                                for new_file in new_result_files:
                                    if os.path.exists(new_file):
                                        os.remove(new_file)
                                
                                custom_logger.info(
                                    f"Upgrade of {dataset_name} to budget {upgrade_budget} would exceed limit "
                                    f"({total_cells_labeled} + {cell_delta} > {cell_limit}), skipped"
                                )
                        
                        except Exception as e:
                            custom_logger.error(f"Error upgrading {dataset_name} to budget {upgrade_budget}: {e}")
                    
                    if not upgraded_any:
                        custom_logger.info(f"No more tables could be upgraded at budget {upgrade_budget}, stopping")
                        break
                    
                    # After upgrading at least one table, move to next budget and continue
                    upgrade_budget += 1
                    current_working_budget = upgrade_budget - 1  # Update working budget for next iteration
            
            # Log final summary for this budget
            if cell_limit is not None:
                custom_logger.info(
                    f"Completed budget {labeling_budget} with cell limit {cell_limit}. "
                    f"Final total cells labeled: {total_cells_labeled}/{cell_limit}. "
                    f"Tables remaining skipped: {len(skipped_tables)}"
                )
                
    e_time = time.time()
    custom_logger.info("Total Run Time:")
    custom_logger.info(e_time - s_time)
    # send_log(f"Total Run Time: {e_time - s_time} seconds for experiment {exp_name}")

if __name__ == '__main__':
    start()
