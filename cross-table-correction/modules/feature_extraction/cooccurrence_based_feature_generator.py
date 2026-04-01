from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from core.candidate import Candidate
from core.cell import Cell
from core.lake import Lake
from core.table import Table


class BaseCooccurrenceFeatureGenerator:
    """Base class for computing cooccurrence features"""

    def __init__(self, config):
        self.config = config
        self.min_probability = getattr(config, "min_probability", 0.0)
        self.cell_values = Counter()
        self.cooccurrences = defaultdict(Counter)

    def _get_clean_row_values(
        self, table: Table, row_idx: int, error_cells: Set[Tuple[int, int]]
    ) -> List[str]:
        """Extract clean values from a row, skipping errors and NaN"""
        row = table.dataframe.iloc[row_idx]
        return [
            value
            for col_idx, value in enumerate(row)
            if (row_idx, col_idx) not in error_cells and pd.notna(value)
        ]

    def _update_cooccurrence_models(self, row_values: List[str]):
        """Update cooccurrence models with values from a clean row"""
        if len(row_values) <= 1:
            return

        # Update value frequencies
        self.cell_values.update(row_values)

        # Update cooccurrences
        for i, value1 in enumerate(row_values):
            other_values = row_values[:i] + row_values[i + 1 :]
            self.cooccurrences[value1].update(other_values)

    def _calculate_avg_probability(self, correction: str) -> float:
        """Calculate average cooccurrence probability for a correction"""
        correction_freq = self.cell_values[correction]
        co_occ_dict = self.cooccurrences[correction]

        if not co_occ_dict or correction_freq == 0:
            return 0.0

        probs = [freq / correction_freq for freq in co_occ_dict.values()]
        return np.mean(probs)

    def add_features_to_candidates(
        self, candidates: Dict[str, Candidate]
    ) -> Dict[str, Candidate]:
        """Add cooccurrence features to existing candidates"""
        feature_name = f"cooccurrence_based_{self.get_scope()}_avg_prob"

        for correction, candidate in candidates.items():
            if correction in self.cooccurrences:
                avg_prob = self._calculate_avg_probability(correction)
                candidate.features[feature_name] = avg_prob
            else:
                candidate.features[feature_name] = 0.0

        return candidates

    def get_scope(self) -> str:
        """Override in subclasses to specify scope (table/lake)"""
        raise NotImplementedError


class TableCooccurrenceFeatureGenerator(BaseCooccurrenceFeatureGenerator):
    """Table-specific cooccurrence feature generator"""

    def __init__(self, config):
        super().__init__(config)
        self.table_models = {}  # table_id -> BaseCooccurrenceFeatureGenerator

    def _ensure_table_model(self, table_id: str):
        """Ensure model exists for this table"""
        if table_id not in self.table_models:
            self.table_models[table_id] = BaseCooccurrenceFeatureGenerator(self.config)

    def update_from_table(self, table: Table):
        """Update cooccurrence patterns from clean rows in this table"""
        self._ensure_table_model(table.table_id)
        model = self.table_models[table.table_id]

        error_cells = {
            (cell.row_idx, cell.column_idx)
            for cell in table.get_all_cells()
            if cell.is_error
        }

        for row_idx in range(table.dataframe.shape[0]):
            row_values = self._get_clean_row_values(table, row_idx, error_cells)
            model._update_cooccurrence_models(row_values)

    def add_features_to_candidates(
        self, cell: Cell, candidates: Dict[str, Candidate]
    ) -> Dict[str, Candidate]:
        """Add table-level cooccurrence features to candidates"""
        if cell.table_id not in self.table_models:
            # No model for this table, add zero features
            feature_name = f"cooccurrence_based_{self.get_scope()}_avg_prob"
            for candidate in candidates.values():
                candidate.features[feature_name] = 0.0
            return candidates

        model = self.table_models[cell.table_id]
        return model.add_features_to_candidates(candidates)

    def get_scope(self) -> str:
        return "table"

    def is_syntactic_unique_supported(self) -> bool:
        return False

    def is_low_cardinality_supported(self) -> bool:
        return True


class LakeCooccurrenceFeatureGenerator(BaseCooccurrenceFeatureGenerator):
    """Lake-level cooccurrence feature generator"""

    def update_from_lake(self, lake: Lake):
        """Update cooccurrence patterns from all clean rows in the lake"""
        for table in lake.tables.values():
            error_cells = {
                (cell.row_idx, cell.column_idx)
                for cell in table.get_all_cells()
                if cell.is_error
            }

            for row_idx in range(table.dataframe.shape[0]):
                row_values = self._get_clean_row_values(table, row_idx, error_cells)
                self._update_cooccurrence_models(row_values)

    def get_scope(self) -> str:
        return "lake"

    def is_syntactic_unique_supported(self) -> bool:
        return False

    def is_low_cardinality_supported(self) -> bool:
        return True


class CooccurrenceTableBasedFeatureGenerator(TableCooccurrenceFeatureGenerator):
    """Backward compatibility wrapper"""

    def generate_features(
        self, cell: Cell, table: Table, candidates: Dict[str, Candidate] = None
    ) -> Dict[str, Candidate]:
        if candidates is None:
            candidates = {}
        return self.add_features_to_candidates(candidates)

    def get_strategy_name(self) -> str:
        return "cooccurrence_based_table"


class CooccurrenceLakeBasedFeatureGenerator(LakeCooccurrenceFeatureGenerator):
    """Backward compatibility wrapper"""

    def generate_features(
        self, cell: Cell, table: Table, candidates: Dict[str, Candidate] = None
    ) -> Dict[str, Candidate]:
        if candidates is None:
            candidates = {}
        return self.add_features_to_candidates(candidates)

    def get_strategy_name(self) -> str:
        return "cooccurrence_based_lake"
