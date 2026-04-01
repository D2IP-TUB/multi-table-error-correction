import csv
import logging
import os
import pickle
import sys
import time
from pathlib import Path

# Parent of DataLakeFDExtractor/ (i.e. HoloFDExtractor/) must be on sys.path for `DataLakeFDExtractor.*` imports.
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
from DataLakeFDExtractor.src.configs.metanome_file_input import \
    run_metanome_with_cli

log = logging.getLogger(__name__)



def find_det_dep(fd):
    determinant = fd['result']['determinant']['columnIdentifiers']
    dependant = fd['result']['dependant']['columnIdentifier']
    print()
    return determinant, dependant


def find_det_dep_cli(fd):
    determinant = fd['determinant']['columnIdentifiers']
    dependant = fd['dependant']['columnIdentifier']
    return determinant, dependant


def get_fd_list(fd_results):
    fd_list = []
    for fd in fd_results:
        determinant, dependant = find_det_dep_cli(fd)
        if len(determinant) == 1 and determinant != dependant \
                and determinant != None and dependant != None \
                and (dependant, determinant) not in fd_list:
            fd_list.append((determinant[0]['columnIdentifier'], dependant))
    return fd_list

def get_all_files(directory: Path):
    if not directory.is_dir():
        raise ValueError("Input is no Directory")

    files = []
    for child in directory.iterdir():
        if child.is_file():
            files.append(child)
        elif child.is_dir():
            files.extend(get_all_files(child))

    return files

def normalize_index_col_name(col_name):
    if col_name == "index":
        return "index_col"
    else:
        return col_name

def fix_csv_for_metanome(file_path):
    """
    Fix CSV file formatting for Metanome by ensuring proper quoting.
    Reads data as strings and writes with proper CSV quoting.
    """
    temp_path = file_path + ".temp"
    try:
        # Read CSV with all columns as strings
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        
        # Write with proper quoting for all non-numeric fields
        df.to_csv(temp_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        
        return temp_path
    except Exception as e:
        log.error(f"Failed to fix CSV {file_path}: {e}")
        return file_path

def process_clean_csv(file_path):
    """
    Run HyFD on one clean.csv (same pipeline as extract()) and write
    fd_list.pkl and holo_constraints.txt in the same directory as clean.csv.
    """
    if not os.path.isfile(file_path):
        log.error("Missing clean.csv: %s", file_path)
        return
    temp_file = None
    try:
        temp_file = fix_csv_for_metanome(file_path)
        fd_results = run_metanome_with_cli(temp_file)
        if fd_results is None:
            log.error("HyFD returned no results for %s", file_path)
            return
        fd_list = get_fd_list(fd_results)
        holo_consts = ""
        for const in fd_list:
            const_str = (
                f"t1&t2&EQ(t1.{normalize_index_col_name(const[0])},"
                f"t2.{normalize_index_col_name(const[0])})&IQ("
                f"t1.{normalize_index_col_name(const[1])},"
                f"t2.{normalize_index_col_name(const[1])})\n"
            )
            holo_consts += const_str

        out_dir = os.path.dirname(file_path)
        fd_pkl = os.path.join(out_dir, "fd_list.pkl")
        holo_txt = os.path.join(out_dir, "holo_constraints.txt")
        if os.path.exists(fd_pkl):
            os.remove(fd_pkl)
        if os.path.exists(holo_txt):
            os.remove(holo_txt)

        with open(fd_pkl, "wb") as f:
            pickle.dump(fd_list, f)
        with open(holo_txt, "w") as f:
            f.write(holo_consts)
    except Exception as e:
        log.error(e)
    finally:
        if temp_file and temp_file != file_path and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass


def extract(input_data_lake_path):
    dirs = os.listdir(input_data_lake_path)
    time_0 = time.time()
    for dir in dirs:
        file_path = os.path.join(input_data_lake_path, dir, "clean.csv")
        process_clean_csv(file_path)

    time_1 = time.time()
    log.info("********time*******:{} seconds".format(time_1 - time_0))


if __name__ == '__main__':
    extract("/home/fatemeh/LakeCorrectionBench/datasets/ablations/mit_dwh_with_validation/merged_union_075")
