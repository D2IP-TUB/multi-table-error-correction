class Column:
    def __init__(self, table_id, col_idx, col_name=None):
        self.table_id = table_id
        self.col_idx = col_idx
        self.col_name = col_name
        self.cells = {}  # row_idx -> Cell
        self.cardinality_type = None  # "high" or "low"
        self.zones = set()
        self.patterns = []  # List of patterns for this column

    def check_cardinality(self, threshold):
        """
        Check the cardinality of the column and set the cardinality type.
        """
        # Get all clean values (non-error cells)
        clean_values = [cell.value for cell in self.cells.values() if not cell.is_error]

        if not clean_values:  # Handle empty case
            self.cardinality_type = "low"
            return

        unique_values = len(set(clean_values))
        total_values = len(clean_values)

        # Use ratio for cardinality check
        cardinality_ratio = unique_values / total_values if total_values > 0 else 0

        if cardinality_ratio >= threshold:
            self.cardinality_type = "high"
        else:
            self.cardinality_type = "low"

    def to_dict(self):
        """
        Serialize column metadata for saving.
        """
        for attr_name, attr_value in vars(self).items():
            if attr_value.__class__.__name__ == "Table":
                raise RuntimeError(
                    f"❌ Cell contains a reference to a Table object: {attr_name}"
                )
        return {
            "table_id": self.table_id,
            "col_idx": int(self.col_idx),
            "col_name": self.col_name,
            "cells": {str(k): v.to_dict() for k, v in self.cells.items()},
            "cardinality_type": self.cardinality_type,
            "cardinality_ratio": self.cardinality_ratio,
            "n_cells": len(self.cells),
            "n_errors": len(self.get_error_cells()),
        }

    def get_clean_values(self):
        """Get all clean (non-error) values from the column"""
        return [cell.value for cell in self.cells.values() if not cell.is_error]

    def get_error_cells(self):
        """Get all error cells in the column"""
        return [cell for cell in self.cells.values() if cell.is_error]

    def get_unique_clean_values(self):
        """Get unique clean values"""
        return set(self.get_clean_values())

    def is_syntactic(self) -> bool:
        """
        Check if this column has primarily syntactic errors.
        A column is syntactic if most error values don't appear in clean data.
        """

        error_cells = self.get_error_cells()
        clean_values = [value for value in self.get_clean_values()]

        syntactic_errors = 0
        for cell in error_cells:
            dirty_value = cell.value
            if dirty_value not in clean_values:
                syntactic_errors += 1

        freq_ratio = syntactic_errors / len(error_cells) if error_cells else 0
        # Column is syntactic if >50% of errors are syntactic
        return freq_ratio > 0.5

    def is_semantic(self) -> bool:
        """
        Check if this column has primarily semantic errors.
        A column is semantic if most error values appear in clean data.
        """

        error_cells = self.get_error_cells()
        clean_values = [cell.value for cell in self.get_clean_values()]

        semantic_errors = 0
        for cell in error_cells:
            dirty_value = cell.value
            if dirty_value in clean_values:
                semantic_errors += 1

        # Column is semantic if >50% of errors are semantic
        return semantic_errors / len(error_cells) > 0.5

    @property
    def cardinality_ratio(self):
        """Calculate the cardinality ratio (unique/total for clean values)"""
        clean_values = self.get_clean_values()
        if not clean_values:
            return 0
        return len(set(clean_values)) / len(clean_values)
