import json
import logging
import unicodedata

from core.cell import Cell
from core.column import Column


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class Zone:
    def __init__(self, name: str, labeling_budget: int = 0):
        self.name = name
        self.labeling_budget = labeling_budget
        self.cells = {}  # (table_id, col_idx, row_idx) -> Cell
        self.columns = {}  # (table_id, col_idx) -> Column
        self.samples = {}  # (table_id, col_idx, row_idx) -> Cell
        self.zone_cols_to_err_cells = {}  # (table_id, col_idx) -> List[Cell]
        self.zone_cols_to_clean_cells = {}  # (table_id, col_idx) -> List[Cell]
        self._pattern_cache = {}  # Cache for patterns
        self.moved_cells = []
        self.cells_to_remove = []
        self.tp = 0
        self.fp = 0
        self.n_invalid_char_cells = 0

    def add_column(self, column):
        if len(column.get_error_cells()) != 0:
            self.columns[(column.table_id, column.col_idx)] = column
            for cell in column.get_error_cells():
                self.cells[(column.table_id, column.col_idx, cell.row_idx)] = cell

    def add_cell(self, cell):
        """Add a cell to the zone."""
        if cell.is_error:
            self.cells[(cell.table_id, cell.column_idx, cell.row_idx)] = cell
        else:
            # Non-error cells are not stored in the zone
            pass

    def remove_cell(self, cell):
        """Remove a cell from the zone."""
        if (cell.table_id, cell.column_idx, cell.row_idx) in self.cells:
            del self.cells[(cell.table_id, cell.column_idx, cell.row_idx)]
        else:
            logging.warning(f"Cell {cell} not found in zone {self.name} for removal.")

    def is_syntactic(self):
        """Check if the zone is syntactic."""
        if "invalid" in self.name:
            return True

    def to_dict(self):
        """Convert to JSON-serializable dict."""
        for attr_name, attr_value in vars(self).items():
            if attr_value.__class__.__name__ == "Table":
                raise RuntimeError(
                    f"❌ Cell contains a reference to a Table object: {attr_name}"
                )

        return {
            "name": self.name,
            "labeling_budget": self.labeling_budget,
            "cells": {self._key_to_str(k): v.to_dict() for k, v in self.cells.items()},
            "columns": {
                self._key_to_str(k): v.to_dict() for k, v in self.columns.items()
            },
            "samples": {
                self._key_to_str(k): v.to_dict() for k, v in self.samples.items()
            },
        }

    def get_zone_cols_to_err_cells(self):
        if self.zone_cols_to_err_cells and len(self.zone_cols_to_err_cells) > 0:
            return self.zone_cols_to_err_cells
        else:
            for cell in self.cells.values():
                if cell.is_error:
                    self.zone_cols_to_err_cells.setdefault(
                        (cell.table_id, cell.column_idx), []
                    ).append(cell)
        return self.zone_cols_to_err_cells

    def get_zone_cols_to_clean_cells(self, lake):
        if self.zone_cols_to_clean_cells and len(self.zone_cols_to_clean_cells) > 0:
            return self.zone_cols_to_clean_cells
        else:
            zone_cols_to_err_cells = self.get_zone_cols_to_err_cells()
            for col in zone_cols_to_err_cells:
                col_obj = lake.tables[col[0]].columns[col[1]]
                self.zone_cols_to_clean_cells[col] = [
                    cell for cell in col_obj.cells.values() if not cell.is_error
                ]
            return self.zone_cols_to_clean_cells

    @classmethod
    def from_dict(cls, data: dict):
        zone = cls(data["name"], data["labeling_budget"])
        zone.cells = {
            cls._str_to_key(k): Cell.from_dict(v) for k, v in data["cells"].items()
        }
        zone.columns = {
            cls._str_to_key(k): Column.from_dict(v) for k, v in data["columns"].items()
        }
        zone.samples = {
            cls._str_to_key(k): Cell.from_dict(v) for k, v in data["samples"].items()
        }
        return zone

    def save_to_disk(self, path: str):
        # with open(path, "w", encoding="utf-8") as f:
        #     json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder)
        # print(f"Saved to {path}")

        with open(path, "wb") as f:
            import pickle

            pickle.dump(self.to_dict(), f)

    @classmethod
    def load_from_disk(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @staticmethod
    def _key_to_str(key: tuple) -> str:
        return ",".join(map(str, key))  # e.g., "table1,2,5"

    @staticmethod
    def _str_to_key(key: str) -> tuple:
        parts = key.split(",")
        return (
            (parts[0], int(parts[1]), int(parts[2]))
            if len(parts) == 3
            else (parts[0], int(parts[1]))
        )
