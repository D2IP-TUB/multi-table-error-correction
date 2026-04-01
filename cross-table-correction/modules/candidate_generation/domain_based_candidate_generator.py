import logging
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from core.candidate import Candidate
from core.cell import Cell
from core.table import Table
from modules.candidate_generation.candidate_generator import CandidateGenerator


class DomainBasedCorrector(CandidateGenerator):
    """
    Domain corrector with pre-computed column domains.
    """

    def __init__(self, config):
        super().__init__(config)
        self._domain_cache = {}  # Cache domain models per (table_id, col_idx)
        self._clean_domain_counts = {}  # Cache clean cell counts per (table_id, col_idx)
        self._labeled_domain_counts = defaultdict(
            lambda: defaultdict(int)
        )  # Store counts from labeled samples per (table_id, col_idx)

    def _get_domain_model_cached(self, col_idx: int, table: Table) -> Dict[str, float]:
        """Get domain model with caching"""
        cache_key = (table.table_id, col_idx)

        if cache_key in self._domain_cache:
            return self._domain_cache[cache_key]

        # Build domain model efficiently
        domain_counts = defaultdict(int)

        # Get or compute clean values
        if cache_key not in self._clean_domain_counts:
            # Get error rows to exclude (vectorized)
            error_rows = {
                cell.row_idx
                for cell in table.get_all_cells()
                if cell.is_error and cell.column_idx == col_idx
            }

            # Collect clean values efficiently
            col_data = table.dataframe.iloc[:, col_idx]
            clean_counts = defaultdict(int)
            for row_idx, value in col_data.items():
                if row_idx not in error_rows and value:
                    clean_counts[str(value)] += 1
            self._clean_domain_counts[cache_key] = dict(clean_counts)

        # Combine clean counts with labeled sample counts
        for value, count in self._clean_domain_counts[cache_key].items():
            domain_counts[value] += count

        # Add counts from labeled samples if available
        if cache_key in self._labeled_domain_counts:
            for value, count in self._labeled_domain_counts[cache_key].items():
                domain_counts[value] += count

        # Convert to probabilities
        total = sum(domain_counts.values())
        domain_model = (
            {k: v / total for k, v in domain_counts.items()} if total > 0 else {}
        )

        self._domain_cache[cache_key] = domain_model
        return domain_model

    def update_from_labeled_samples(self, samples: List[Cell], tables: Dict[str, Table] = None):
        """Update domain model with labeled samples"""
        logging.info(f"Updating domain model from {len(samples)} labeled samples...")

        # Track which columns were updated
        updated_columns = set()

        for sample_cell in samples:
            if hasattr(sample_cell, "ground_truth") and sample_cell.ground_truth:
                column_key = (sample_cell.table_id, sample_cell.column_idx)
                new_value = str(sample_cell.ground_truth)
                self._labeled_domain_counts[column_key][new_value] += 1
                updated_columns.add(column_key)

        # Invalidate only the model cache (not clean counts) for affected columns
        for column_key in updated_columns:
            self._domain_cache.pop(column_key, None)


    def generate_candidates(self, cell: Cell, table: Table):
        """Generate domain candidates using cached domain models.
        
        Returns:
            Dict mapping value -> candidate pool key (table_id, column_idx, value)
        """
        col_idx = cell.column_idx

        # Get cached domain model
        domain_model = self._get_domain_model_cached(col_idx, table)

        if not domain_model:
            return {}

        candidates = {}

        # Generate candidates efficiently
        for value, probability in domain_model.items():
            if probability >= self.min_probability and value != cell.value:
                features = {
                    "domain_based": probability,
                }

                candidate = Candidate(
                    value, features, candidates_source_model={"domain_based": 1}
                )
                
                # Add to pool and store reference
                pool_key = self._add_candidate_to_pool(
                    table.table_id, col_idx, value, candidate
                )
                candidates[value] = pool_key

        return candidates

    def get_strategy_name(self) -> str:
        return "domain_based"
