import os
# Set GPU device before importing torch/HoloClean
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import logging
import multiprocessing
import os
import pickle
import random
import time
from dataclasses import dataclass

import hydra
import pandas as pd
import psutil
# from HoloFDExtractor.DataLakeFDExtractor.src.extract_fds import extract
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from run_holoclean import dcHoloCleaner
from utils import (aggregate_lake_results, get_dataframes_difference, read_csv,
                   sanitize_column_names)


@dataclass
class AppConfig:
    logs_dir: str = "logs"
    algorithm: str = "HoloClean"
    input_data_lake_path: str = "/home/fatemeh/LakeCorrectionBench/datasets/mit_blend_default"
    num_iterations: int = 1
    max_memory_gb: float = 900  # Maximum memory in GB before killing the process
    memory_check_interval: int = 10  # Check memory every N seconds    
    threads: int = 1  # Number of threads for HoloClean to use
# Register the configuration schema with Hydra
cs = ConfigStore.instance()
cs.store(name="base", node=AppConfig)

def setup_logging(logs_dir: str, to_console: bool = False):
    """
    Setup logging configurations

    Args:
        logs_dir (str): Path to the directory where the logs will be stored
        to_console (bool): If True, also log to the console
    """
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, "app.log")
    print(f"Logs will be stored in: {log_file_path}")

    root_logger = logging.getLogger()
    root_logger.handlers = []

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.DEBUG)

def process_table_with_memory_limit(result_queue, dir_name, dirty_df, clean_df, table_path, 
                                    actual_errors_expected, seed, iteration, threads):
    """
    Process a single table in a separate process with memory monitoring.
    Puts results in the queue: (success, dir_name, results, error_msg)
    """
    try:
        repaired_df, results = dcHoloCleaner(dir_name, dirty_df, clean_df, table_path, 
                                            actual_errors_expected, "with_init", seed, iteration, threads)
        result_queue.put((True, dir_name, results, None))
    except Exception as e:
        result_queue.put((False, dir_name, None, str(e)))

def monitor_process_memory(process, max_memory_gb, check_interval, logger):
    """
    Monitor a process's memory usage and kill it if it exceeds the threshold.
    Returns: (completed_normally, exceeded_memory)
    """
    try:
        ps_process = psutil.Process(process.pid)
        while process.is_alive():
            try:
                memory_info = ps_process.memory_info()
                memory_gb = memory_info.rss / (1024 ** 3)  # Convert to GB
                
                if memory_gb > max_memory_gb:
                    logger.warning(f"Process {process.pid} exceeded memory limit: {memory_gb:.2f}GB > {max_memory_gb}GB")
                    process.terminate()
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                    return (False, True)
                
                time.sleep(check_interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
        
        return (True, False)
    except Exception as e:
        logger.error(f"Error monitoring process: {e}")
        return (True, False)

@hydra.main(config_name="base")
def main(cfg: DictConfig):
    logs_dir = cfg.logs_dir
    algorithm = cfg.algorithm
    input_data_lake_path = cfg.input_data_lake_path
    dataset_name = os.path.basename(input_data_lake_path)
    # Include the algorithm name and a fragment of the path in the log directory name
    logs_subdir = os.path.join(logs_dir, f"{algorithm}_{os.path.basename(input_data_lake_path)}")
    setup_logging(logs_subdir, to_console=True)
    logger = logging.getLogger()
    logger.info("Starting application...")
    logger.info(f"Algorithm: {algorithm}")
    logger.info(f"Input Data Lake Path: {input_data_lake_path}")
    logger.info("----" * 15)
    if algorithm == "HoloClean":
        logger.info("Running HoloClean")
        num_iterations = cfg.num_iterations
        max_memory_gb = cfg.max_memory_gb
        memory_check_interval = cfg.memory_check_interval
        threads = cfg.threads
        logger.info(f"Running {num_iterations} iterations with different seeds")
        logger.info(f"Memory limit: {max_memory_gb}GB, check interval: {memory_check_interval}s")
        logger.info(f"HoloClean threads: {threads}")
        # logger.info("Extract FDs from the data lake...")
        # extract(input_data_lake_path)
        
        # Track skipped tables
        skipped_tables = []
        skipped_file_path = os.path.join(logs_subdir, "skipped_tables.json")
        
        counter = 0
        for dir in os.listdir(input_data_lake_path):
            logging.info(f"Running HoloClean on {dir}, {counter}. table")
            
            # # Check if all iterations for this table are already processed
            output_dir = ""
            # all_iterations_done = True
            # if os.path.exists(output_dir):
            #     for iteration in range(num_iterations):
            #         existing_files = []
            #         for file in os.listdir(output_dir):
            #             if file.startswith(f"repaired_holoclean_{dir}_") and file.endswith(f"_iter{iteration}.csv"):
            #                 existing_files.append(file)
            #         if not existing_files:
            #             all_iterations_done = False
            #             break
            # else:
            #     all_iterations_done = False
            
            # if all_iterations_done:
            #     logger.info(f"### Table {dir} - all iterations already processed. Skipping entire table...")
            #     counter += 1
            #     continue
            
            dirty_df = read_csv(os.path.join(input_data_lake_path, dir, "dirty.csv"),  data_type = 'str')
            clean_df = read_csv(os.path.join(input_data_lake_path, dir, "clean.csv"),  data_type = 'str')
            clean_df.columns = ["index_col" if col == "index" else col for col in clean_df.columns]
            dirty_df.columns = clean_df.columns
            
            # Sanitize column names to remove problematic characters like '::'
            dirty_df = sanitize_column_names(dirty_df)
            clean_df = sanitize_column_names(clean_df)
            
            # Create a SQL-safe table name by prefixing with 't_' if it starts with a number
            safe_table_name = f"t_{dir}" if dir[0].isdigit() else dir
            
            actual_errors_expected = get_dataframes_difference(dirty_df, clean_df)
            print(f"Number of actual errors in dirty data: {len(actual_errors_expected)}")
            # holoclean_gt_df = pd.DataFrame(columns=["_tid_", "attribute", "correct_val"])
            # for col_i in range(dirty_df.shape[1]):
            #     for row_i in range(dirty_df.shape[0]):
            #         holoclean_gt_df = holoclean_gt_df.append(
            #             {"_tid_": row_i, "attribute": dirty_df.columns[col_i], "correct_val": clean_df.iloc[row_i, col_i]}, ignore_index=True
            #         )
            # Vectorized approach: melt the clean_df to get ground truth in long format
            holoclean_gt_df = clean_df.reset_index(drop=True).reset_index().rename(columns={"index": "_tid_"})
            holoclean_gt_df = holoclean_gt_df.melt(id_vars=["_tid_"], var_name="attribute", value_name="correct_val")
            holoclean_gt_df.to_csv(os.path.join(os.path.dirname(input_data_lake_path), dataset_name, dir, "all_cells.csv"), index=False)
            
            # Run multiple iterations with different random seeds (avoiding default 45)
            for iteration in range(num_iterations):
                # Check if this specific iteration already exists
                existing_files = []
                if os.path.exists(output_dir):
                    for file in os.listdir(output_dir):
                        if file.startswith(f"repaired_holoclean_{dir}_") and file.endswith(f"_iter{iteration}.csv"):
                            existing_files.append(file)
                
                if existing_files:
                    logger.info(f"### Table {dir} iteration {iteration+1} already processed (found {existing_files[0]}). Skipping...")
                    continue
                
                seed = random.randint(1, 10000)
                while seed == 45:  # Avoid the default seed
                    seed = random.randint(1, 10000)
                logger.info(f"Iteration {iteration+1}/{num_iterations} with seed {seed}")
                
                # Create a queue to get results from the subprocess
                result_queue = multiprocessing.Queue()
                
                # Create a process to run the table processing
                process = multiprocessing.Process(
                    target=process_table_with_memory_limit,
                    args=(result_queue, safe_table_name, dirty_df, clean_df, os.path.join(input_data_lake_path, dir), 
                          actual_errors_expected, seed, iteration, threads)
                )
                
                table_skipped = False
                try:
                    process.start()
                    
                    # Monitor memory in the main process
                    completed_normally, exceeded_memory = monitor_process_memory(
                        process, max_memory_gb, memory_check_interval, logger
                    )
                    
                    process.join()  # Wait for process to finish (no timeout, wait until complete)
                    
                    if exceeded_memory:
                        logger.warning(f"Table {dir} iteration {iteration+1} was killed due to excessive memory usage")
                        skipped_info = {
                            "table": dir,
                            "iteration": iteration + 1,
                            "seed": seed,
                            "reason": "exceeded_memory",
                            "max_memory_gb": max_memory_gb
                        }
                        skipped_tables.append(skipped_info)
                        table_skipped = True
                        
                        # Save skipped tables after each skip
                        with open(skipped_file_path, 'w') as f:
                            json.dump(skipped_tables, f, indent=2)
                        
                    elif not completed_normally:
                        logger.error(f"Process for table {dir} iteration {iteration+1} did not complete normally")
                    else:
                        # Try to get results from the queue
                        try:
                            if not result_queue.empty():
                                success, table_name, results, error = result_queue.get(timeout=5)
                                if success:
                                    logger.info(f"Iteration {iteration+1} results: {results}")
                                else:
                                    logger.error(f"Error in iteration {iteration+1}: {error}")
                            else:
                                logger.info(f"Iteration {iteration+1} completed (no results in queue)")
                        except Exception as e:
                            logger.warning(f"Could not retrieve results from queue: {e}")
                        
                except Exception as e:
                    logger.error(f"Error occurred while processing {dir} iteration {iteration+1}: {e}")
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                finally:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                
                # If table was skipped due to memory, stop processing iterations for this table
                if table_skipped:
                    logger.info(f"Skipping remaining iterations for table {dir} due to memory issues")
                    break
                        
            counter += 1
        
        # Log summary of skipped tables
        if skipped_tables:
            logger.warning(f"Total skipped tables: {len(skipped_tables)}")
            logger.warning(f"Skipped tables list saved to: {skipped_file_path}")
            for skip_info in skipped_tables:
                logger.warning(f"  - {skip_info['table']} (iteration {skip_info['iteration']}): {skip_info['reason']}")
        else:
            logger.info("No tables were skipped")
        
        # Aggregate results across all tables and iterations
        logger.info("Aggregating results across the data lake...")
        results_directory = "dcHoloCleaner-with_init/HoloClean"
        if os.path.exists(results_directory):
            output_file = os.path.join(results_directory, "lake_aggregated_results.json")
            results_df, final_results = aggregate_lake_results(results_directory, output_file)
            with open(os.path.join(logs_subdir, "skipped_tables.json"), 'w') as f:
                json.dump(skipped_tables, f, indent=2)
            if final_results:
                logger.info(f"Lake-wide evaluation completed!")
                logger.info(f"Total tables: {final_results['total_tables']}")
                logger.info(f"Total iterations: {final_results['total_iterations']}")
                logger.info(f"Average Precision: {final_results['avg_precision']:.4f}")
                logger.info(f"Average Recall: {final_results['avg_recall']:.4f}")
                logger.info(f"Average F1 Score: {final_results['avg_f1_score']:.4f}")
                logger.info(f"F1 Score Std Dev: {final_results['f1_score_std']:.4f}")
                logger.info(f"Results saved to: {output_file}")
        else:
            logger.warning(f"Results directory not found: {results_directory}")
            
if __name__ == "__main__":
    print("Starting HoloClean run_baselines.py")
    main()
