import html
import json
import logging
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd


def convert_to_json_serializable(obj):
    """
    Recursively convert NumPy types to Python native types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


def get_dataframes_difference(dataframe_1, dataframe_2):
    """
    This method compares two dataframes and returns the different cells.
    """
    if dataframe_1.shape != dataframe_2.shape:
        sys.stderr.write("Two compared datasets do not have equal sizes!\n")
    difference_dictionary = {}
    difference_dataframe = dataframe_1.where(dataframe_1.values != dataframe_2.values).notna()
    for j in range(dataframe_1.shape[1]):
        for i in difference_dataframe.index[difference_dataframe.iloc[:, j]].tolist():
            difference_dictionary[(i, j)] = dataframe_2.iloc[i, j]
    return difference_dictionary

def value_normalizer(value: str) -> str:
    """
    This method takes a value and minimally normalizes it. (Raha's value normalizer)
    """
    if value is not np.nan:
        value = html.unescape(value)
        value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
        value = value.strip("\t\n ")
    return value


def read_csv(path: str, low_memory: bool = False, data_type: str = 'str') -> pd.DataFrame:
    """
    This method reads a table from a csv file path,
    with pandas default null values and str data type
    Args:
        low_memory: whether to use low memory mode (bool), default False
        path: table path (str)

    Returns:
        pandas dataframe of the table
    """
    logging.info("Reading table, name: %s", path)

    _base = dict(
        sep=",",
        header="infer",
        encoding="latin-1",
        engine="python",
    )
    _read_kw = {**_base, "on_bad_lines": "warn"}
    _normalize = lambda df: df.map(lambda x: value_normalizer(x) if isinstance(x, str) else x)

    def _read(**kwargs):
        try:
            return pd.read_csv(path, **kwargs)
        except TypeError:
            kwargs.pop("on_bad_lines", None)
            return pd.read_csv(path, **kwargs)

    if data_type == 'default':
        return _normalize(_read(**_read_kw))
    elif data_type == 'str':
        return _normalize(_read(dtype=str, keep_default_na=False, **_read_kw))


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    This method sanitizes column names to make them SQL-safe by replacing problematic characters.
    Specifically, it removes '::' and other special characters that can cause SQL syntax errors.
    
    Args:
        df: pandas dataframe with potentially problematic column names
        
    Returns:
        dataframe with sanitized column names
    """
    # Replace '::' with single underscore '_'
    df.columns = [col.replace('::', '_') if isinstance(col, str) else col for col in df.columns]
    return df


def get_cleaner_directory(output_path, cleaner_name):
        """
        This method creates a cleaner directory, if not exist, and return a path to this directory

        :Arguments:
        cleaner_name --String denoting the name of a cleaner

        Returns:
        cleaner_directory -- String denoting the path to the cleaner directory
        """

        cleaner_directory = os.path.join(output_path, cleaner_name)
        if not os.path.exists(cleaner_directory):
            # creating a new directory if it does not exit
            os.makedirs(cleaner_directory)

        return cleaner_directory

def store_cleaned_data(cleanedDF, cleaner_path):
        """
        stores given dataframe as .csv in cleaner_path

        Arguments:
        cleanedDF (dataframe) -- dataframe that was cleaned
        cleaner_path (String) -- path to the folder in which cleaned dataframe should be stored
        """
        cleanedDF.to_csv(cleaner_path, index=False, encoding="utf-8")


def evaluate(detections, dirty_df, clean_df, repaired_df):

    """
    This method evaluates data cleaning process.
    Skips errors whose row/column index is out of bounds for any dataframe (e.g. if repaired
    had fewer rows due to skipped bad CSV lines).
    """
    n_rows_rep, n_cols_rep = len(repaired_df), len(repaired_df.columns)
    n_rows_dirty, n_cols_dirty = len(dirty_df), len(dirty_df.columns)
    n_rows_clean, n_cols_clean = len(clean_df), len(clean_df.columns)
    correction_dict = {}
    n_truely_corrected_errors = 0
    n_all_corrected_errors = 0
    n_all_errors = len(detections)
    for error in detections:
        r, c = error[0], error[1]
        if r >= n_rows_rep or c >= n_cols_rep or r >= n_rows_dirty or c >= n_cols_dirty or r >= n_rows_clean or c >= n_cols_clean:
            continue
        correction_dict[error] = {"dirty": dirty_df.iloc[r, c], "ground_truth": clean_df.iloc[r, c], "repaired": repaired_df.iloc[r, c]}
    for error in correction_dict:
        err_info = correction_dict[error]
        if err_info["ground_truth"] == err_info["repaired"] or (len(err_info["ground_truth"])==0 and len(err_info["repaired"])==0):
             n_truely_corrected_errors += 1
        if err_info["dirty"] != err_info["repaired"]:
             n_all_corrected_errors += 1
    precision = n_truely_corrected_errors / n_all_corrected_errors if n_all_corrected_errors != 0 else -1
    recall = n_truely_corrected_errors / n_all_errors if n_all_errors != 0 else -1 
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else -1

    results = {"n_truely_corrected_errors": n_truely_corrected_errors, 
               "n_all_corrected_errors": n_all_corrected_errors,
               "n_all_errors": n_all_errors,
               "precision": precision,
               "recall": recall,
               "f1_score": f1_score}
    return results


def evaluate_by_error_type(dirty_df, clean_df, repaired_df, provenance_file_path):
    """
    This method evaluates data cleaning process grouped by error type.
    
    Args:
        dirty_df: dirty dataframe
        clean_df: clean dataframe
        repaired_df: repaired dataframe
        provenance_file_path: path to clean_changes_provenance.csv or source_mapping.json file
        
    Returns:
        Dictionary with overall results and results per error type
    """
    # Check file type and read accordingly
    if provenance_file_path.endswith('.json'):
        import json
        with open(provenance_file_path, 'r') as f:
            mapping_data = json.load(f)
        
        # Convert source_mapping.json to provenance format
        # Extract error type from source name (e.g., "DGov_FD" -> "FD")
        rows = []
        for row_idx, mapping_info in mapping_data.get('dirty_mapping', {}).items():
            source = mapping_info.get('source', '')
            # Extract error type from source name (e.g., DGov_FD, DGov_Typo, etc.)
            if '_' in source:
                error_type = source.split('_', 1)[1]  # Get part after first underscore
            else:
                error_type = 'UNKNOWN'
            
            # For each column, check if there's an error
            for col_idx, col_name in enumerate(dirty_df.columns):
                if int(row_idx) < len(dirty_df):
                    dirty_val = dirty_df.iloc[int(row_idx), col_idx]
                    clean_val = clean_df.iloc[int(row_idx), col_idx]
                    if dirty_val != clean_val:
                        rows.append({
                            'row_number': int(row_idx),
                            'column_name': col_name,
                            'error_type': error_type
                        })
        
        provenance_df = pd.DataFrame(rows)
    else:
        # Read CSV provenance file
        provenance_df = pd.read_csv(provenance_file_path)
    
    # Initialize results dictionary
    results_by_type = {}
    overall_results = {
        "n_truely_corrected_errors": 0,
        "n_all_corrected_errors": 0,
        "n_all_errors": 0
    }
    
    # Group errors by error type
    for error_type in provenance_df['error_type'].unique():
        type_errors = provenance_df[provenance_df['error_type'] == error_type]
        
        n_truely_corrected = 0
        n_all_corrected = 0
        n_errors = len(type_errors)
        
        for _, row in type_errors.iterrows():
            row_num = row['row_number']
            col_name = row['column_name']
            
            # Get column index
            if col_name not in clean_df.columns:
                continue
            col_idx = clean_df.columns.get_loc(col_name)
            
            ground_truth = clean_df.iloc[row_num, col_idx]
            repaired_val = repaired_df.iloc[row_num, col_idx]
            dirty_val = dirty_df.iloc[row_num, col_idx]
            
            # Check if truly corrected
            if ground_truth == repaired_val or (len(str(ground_truth))==0 and len(str(repaired_val))==0):
                n_truely_corrected += 1
            
            # Check if attempted correction
            if dirty_val != repaired_val:
                n_all_corrected += 1
        
        # Calculate metrics for this error type
        precision = n_truely_corrected / n_all_corrected if n_all_corrected != 0 else -1
        recall = n_truely_corrected / n_errors if n_errors != 0 else -1
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else -1
        
        results_by_type[error_type] = {
            "n_truely_corrected_errors": n_truely_corrected,
            "n_all_corrected_errors": n_all_corrected,
            "n_all_errors": n_errors,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
        
        # Update overall counts
        overall_results["n_truely_corrected_errors"] += n_truely_corrected
        overall_results["n_all_corrected_errors"] += n_all_corrected
        overall_results["n_all_errors"] += n_errors
    
    # Calculate overall metrics
    overall_results["precision"] = (overall_results["n_truely_corrected_errors"] / 
                                   overall_results["n_all_corrected_errors"] 
                                   if overall_results["n_all_corrected_errors"] != 0 else -1)
    overall_results["recall"] = (overall_results["n_truely_corrected_errors"] / 
                                overall_results["n_all_errors"] 
                                if overall_results["n_all_errors"] != 0 else -1)
    overall_results["f1_score"] = ((2 * overall_results["precision"] * overall_results["recall"]) / 
                                  (overall_results["precision"] + overall_results["recall"]) 
                                  if (overall_results["precision"] + overall_results["recall"]) != 0 else -1)
    
    return {
        "overall": overall_results,
        "by_error_type": results_by_type
    }


def aggregate_lake_results(results_directory, output_file=None):
    """
    Aggregates evaluation results across the data lake.
        
    Args:
        results_directory: Directory containing pickle files with results
        output_file: Optional path to save aggregated results
        
    Returns:
        DataFrame with per-table results and dictionary with aggregated lake metrics
    """
    import glob
    import json
    from collections import defaultdict

    # Find all result pickle files
    result_files = glob.glob(os.path.join(results_directory, "results_holoclean_*.pkl"))
    
    if not result_files:
        logging.warning(f"No result files found in {results_directory}")
        return None, None
    
    # Collect per-table, per-iteration results
    per_table_results = []
    
    for result_file in result_files:
        try:
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            
            # Extract metadata from filename
            filename = os.path.basename(result_file)
            if '_seed' in filename:
                parts = filename.replace('results_holoclean_', '').replace('.pkl', '').split('_seed')
                table_name = parts[0]
                seed_iter = parts[1].split('_iter')
                seed = int(seed_iter[0])
                iteration = int(seed_iter[1])
            else:
                table_name = filename.replace('results_holoclean_', '').replace('.pkl', '')
                seed = 45
                iteration = 0
            
            # Extract metrics
            custom_eval = result.get("custom_evaluation", {})
            tp = custom_eval.get("n_truely_corrected_errors", 0)
            ec_tpfp = custom_eval.get("n_all_corrected_errors", 0)
            ec_tpfn = custom_eval.get("n_all_errors", 0)
            exec_time = result.get("cleaning_runtime", 0)
            
            # Calculate per-table metrics
            precision = tp / ec_tpfp if ec_tpfp > 0 else 0
            recall = tp / ec_tpfn if ec_tpfn > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            table_result = {
                "algorithm": "HoloClean",
                "dataset": table_name,
                "execution_number": iteration,
                "seed": seed,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "tp": tp,
                "ec_tpfp": ec_tpfp,
                "ec_tpfn": ec_tpfn,
                "execution_time": exec_time
            }
            
            # Add error type metrics if available
            error_type_eval = result.get("error_type_evaluation", {})
            if error_type_eval and "by_error_type" in error_type_eval:
                for error_type, metrics in error_type_eval["by_error_type"].items():
                    table_result[f"{error_type}_tp"] = metrics.get("n_truely_corrected_errors", 0)
                    table_result[f"{error_type}_ec_tpfp"] = metrics.get("n_all_corrected_errors", 0)
                    table_result[f"{error_type}_ec_tpfn"] = metrics.get("n_all_errors", 0)
            
            per_table_results.append(table_result)
            
        except Exception as e:
            logging.error(f"Error processing {result_file}: {e}")
            continue
    
    # Create DataFrame
    results_df = pd.DataFrame(per_table_results)
    
    # Save per-table results
    if output_file:
        csv_file = output_file.replace('.json', '_per_table.csv')
        results_df.to_csv(csv_file, index=False)
        logging.info(f"Per-table results saved to {csv_file}")
    
    # Calculate total errors in the lake (ec_tpfn) - should be constant across iterations
    # Take from first iteration
    first_iter = results_df[results_df['execution_number'] == results_df['execution_number'].min()]
    total_errors_lake = first_iter['ec_tpfn'].sum()
    
    # Aggregate across iterations (seeds)
    iterations = sorted(results_df['execution_number'].unique())
    seeds_list = sorted(results_df['seed'].unique())
    
    aggregated_results = {
        "execution_number": [],
        "seed": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "tp": [],
        "ec_tpfp": [],
        "execution_time": []
    }
    
    # For each iteration, sum across all tables then calculate metrics
    for iter_num in iterations:
        iter_data = results_df[results_df['execution_number'] == iter_num]
        seed = iter_data['seed'].iloc[0] if len(iter_data) > 0 else 0
        
        # Sum across all tables for this iteration
        tp_sum = iter_data['tp'].sum()
        ec_tpfp_sum = iter_data['ec_tpfp'].sum()
        exec_time_sum = iter_data['execution_time'].sum()
        
        # Calculate lake-level metrics using the fixed total_errors_lake
        precision = tp_sum / ec_tpfp_sum if ec_tpfp_sum > 0 else 0
        recall = tp_sum / total_errors_lake if total_errors_lake > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        aggregated_results["execution_number"].append(iter_num)
        aggregated_results["seed"].append(seed)
        aggregated_results["precision"].append(precision)
        aggregated_results["recall"].append(recall)
        aggregated_results["f1_score"].append(f1)
        aggregated_results["tp"].append(tp_sum)
        aggregated_results["ec_tpfp"].append(ec_tpfp_sum)
        aggregated_results["execution_time"].append(exec_time_sum)
    
    # Calculate average across iterations
    avg_precision = np.mean(aggregated_results["precision"]) if aggregated_results["precision"] else 0
    avg_recall = np.mean(aggregated_results["recall"]) if aggregated_results["recall"] else 0
    avg_f1 = np.mean(aggregated_results["f1_score"]) if aggregated_results["f1_score"] else 0
    std_f1 = np.std(aggregated_results["f1_score"]) if len(aggregated_results["f1_score"]) > 1 else 0
    
    # Aggregate by error type (if available)
    error_type_results = {}
    error_type_columns = [col for col in results_df.columns if col.endswith('_tp')]
    error_types = [col.replace('_tp', '') for col in error_type_columns]
    
    for error_type in error_types:
        tp_col = f"{error_type}_tp"
        ec_tpfp_col = f"{error_type}_ec_tpfp"
        ec_tpfn_col = f"{error_type}_ec_tpfn"
        
        if tp_col not in results_df.columns:
            continue
        
        # Get total errors for this error type (from first iteration)
        total_errors_type = first_iter[ec_tpfn_col].sum() if ec_tpfn_col in first_iter.columns else 0
        
        type_aggregated = {
            "execution_number": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "tp": [],
            "ec_tpfp": []
        }
        
        # Aggregate per iteration
        for iter_num in iterations:
            iter_data = results_df[results_df['execution_number'] == iter_num]
            
            tp_sum = iter_data[tp_col].sum() if tp_col in iter_data.columns else 0
            ec_tpfp_sum = iter_data[ec_tpfp_col].sum() if ec_tpfp_col in iter_data.columns else 0
            
            precision = tp_sum / ec_tpfp_sum if ec_tpfp_sum > 0 else 0
            recall = tp_sum / total_errors_type if total_errors_type > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            type_aggregated["execution_number"].append(iter_num)
            type_aggregated["precision"].append(precision)
            type_aggregated["recall"].append(recall)
            type_aggregated["f1_score"].append(f1)
            type_aggregated["tp"].append(tp_sum)
            type_aggregated["ec_tpfp"].append(ec_tpfp_sum)
        
        # Calculate averages for this error type
        error_type_results[error_type] = {
            "total_errors": total_errors_type,
            "avg_precision": np.mean(type_aggregated["precision"]) if type_aggregated["precision"] else 0,
            "avg_recall": np.mean(type_aggregated["recall"]) if type_aggregated["recall"] else 0,
            "avg_f1_score": np.mean(type_aggregated["f1_score"]) if type_aggregated["f1_score"] else 0,
            "f1_score_std": np.std(type_aggregated["f1_score"]) if len(type_aggregated["f1_score"]) > 1 else 0,
            "avg_tp": np.mean(type_aggregated["tp"]) if type_aggregated["tp"] else 0,
            "avg_ec_tpfp": np.mean(type_aggregated["ec_tpfp"]) if type_aggregated["ec_tpfp"] else 0,
            "per_iteration_results": type_aggregated
        }
    
    final_results = {
        "total_tables": len(results_df['dataset'].unique()),
        "total_iterations": len(iterations),
        "seeds_used": seeds_list,
        "total_errors_in_lake": total_errors_lake,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1_score": avg_f1,
        "f1_score_std": std_f1,
        "avg_tp": np.mean(aggregated_results["tp"]) if aggregated_results["tp"] else 0,
        "avg_ec_tpfp": np.mean(aggregated_results["ec_tpfp"]) if aggregated_results["ec_tpfp"] else 0,
        "avg_execution_time": np.mean(aggregated_results["execution_time"]) if aggregated_results["execution_time"] else 0,
        "per_iteration_results": aggregated_results,
        "by_error_type": error_type_results
    }
    
    # Save final aggregated results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(convert_to_json_serializable(final_results), f, indent=2)
        logging.info(f"Aggregated lake results saved to {output_file}")
    
    return results_df, final_results
            