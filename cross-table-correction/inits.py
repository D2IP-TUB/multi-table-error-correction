import hashlib
import json
import logging
import os
import pickle
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from core.cell import Cell
from core.column import Column
from core.lake import Lake
from core.table import Table
from core.zone import Zone
from modules.zones.zone_detection import detect_zones
from utils.read_data import read_csv


def _process_single_table(args):
    folder, tables_dir, dirty_name, clean_name = args

    path = os.path.join(tables_dir, folder)
    dirty_path = os.path.join(path, dirty_name)
    clean_path = os.path.join(path, clean_name)

    dirty_df = read_csv(dirty_path, data_type="str")
    clean_df = read_csv(clean_path, data_type="str")

    dirty_df.columns = clean_df.columns
    table_id = hashlib.md5(folder.encode()).hexdigest()

    error_mask = ((dirty_df != clean_df) & ~(dirty_df.isna() & clean_df.isna())).values

    table = Table(folder, table_id, dirty_df, clean_df)
    table.columns = {}

    n_rows, n_cols = dirty_df.shape

    dirty_values = dirty_df.values
    clean_values = clean_df.values

    all_cells = {}

    for col_idx in range(n_cols):
        col_name = dirty_df.columns[col_idx]
        column = Column(table_id, col_idx, col_name)
        column.cells = {}

        col_errors = error_mask[:, col_idx]
        dirty_col = dirty_values[:, col_idx]
        clean_col = clean_values[:, col_idx]

        error_rows = np.where(col_errors)[0]

        for row_idx in error_rows:
            cell = Cell(table_id, col_idx, row_idx, is_error=True)
            cell.value = dirty_col[row_idx]
            cell.ground_truth = clean_col[row_idx]
            cell.is_error = True
            column.cells[row_idx] = cell
            all_cells[(row_idx, col_idx)] = cell

        for row_idx in range(n_rows):
            if row_idx not in column.cells:
                cell = Cell(table_id, col_idx, row_idx, is_error=False)
                cell.value = dirty_col[row_idx]
                cell.ground_truth = clean_col[row_idx]
                cell.is_error = False
                column.cells[row_idx] = cell
                all_cells[(row_idx, col_idx)] = cell

        table.columns[col_idx] = column

    return table_id, table


def initialize_lake(
    tables_dir: str,
    dirty_name: str = "dirty.csv",
    clean_name: str = "clean.csv",
    max_workers: int = None,
    cardinality_threshold: float = 1.0,
    output_dir: str = None,
) -> Lake:
    """
    Lake initialization
    """
    lake = Lake()
    table_folders = os.listdir(tables_dir)
    table_id_to_names = {}

    if max_workers is None:
        max_workers = min(len(table_folders), os.cpu_count())

    args_list = [
        (folder, tables_dir, dirty_name, clean_name) for folder in table_folders
    ]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {
            executor.submit(_process_single_table, args): args[0] for args in args_list
        }

        for future in as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                table_id, table = future.result()
                table_id_to_names[table_id] = {
                    "table_name": table.table_name,
                    "columns": [],
                }
                for col_idx, column in table.columns.items():
                    table_id_to_names[table_id]["columns"].append(
                        {"column_name": column.col_name, "column_index": col_idx}
                    )
                lake.add_table(table_id, table)
                print(f"✅ Processed table: {folder}")
            except Exception as e:
                print(f"❌ Error processing {folder}: {e}")

    _calculate_lake_stats(lake)
    set_columns_cardinality_type(lake, cardinality_threshold)
    with open(os.path.join(output_dir, "table_id_to_names.pickle"), "wb") as f:
        pickle.dump(table_id_to_names, f)
    return lake


def _calculate_lake_stats(lake):
    """Calculate lake statistics"""
    tables = list(lake.tables.values())

    lake.n_tables = len(tables)
    lake.n_columns = sum(len(t.columns) for t in tables)

    total_cells = 0
    total_errors = 0
    for table in tables:
        for column in table.columns.values():
            total_cells += len(column.cells)
            col_errors = sum(1 for cell in column.cells.values() if cell.is_error)
            total_errors += col_errors

    lake.n_cells = total_cells
    lake.n_errors = total_errors


def set_columns_cardinality_type(lake, threshold):
    """
    Set the cardinality type of each column in the lake.
    """
    for table in lake.tables.values():
        for column in table.columns.values():
            column.check_cardinality(threshold)


def initialize_zones_cell_wise(lake, config):
    """
    Initialize zones based on the cells in the lake using configured strategy.
    
    Args:
        lake: Lake object containing all tables
        config: PipelineConfig with zoning settings
        
    Returns:
        Dictionary of zones populated based on configured strategy
    """
    zone_names = [
        # Unique/High cardinality zones
        "unique_valid_pattern",      # Valid syntactic pattern
        "unique_invalid_pattern",    # Invalid pattern (needs pattern enforcement)
        # Non-unique/Low cardinality zones
        "non_unique_valid_pattern",  # Value in clean domain OR valid pattern
        "non_unique_invalid_pattern", # Invalid pattern (needs pattern enforcement)
    ]

    zones_dict = {name: Zone(name) for name in zone_names}
    detect_zones(lake, config, zones_dict)

    for zone in zones_dict.values():
        logging.info(f"Zone {zone.name} has {len(zone.cells)} cells.")

    return zones_dict
