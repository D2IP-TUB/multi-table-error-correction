from collections import defaultdict
from typing import Dict, List

from core.lake import Lake
from modules.sampling.sampling_strategy import SamplingStrategy


class LowCardinalitySamplingStrategy(SamplingStrategy):
    """Sampling strategy for low cardinality columns"""

    def sample(
        self,
        lake: Lake,
        cells: List,
        budget: int,
        strategy_type: str = "max_column_coverage",
        **kwargs,
    ) -> Dict:
        """
        Sample cells from low cardinality columns.

        Args:
            cells: List of cells to sample from
            budget: Number of cells to sample
            strategy_type: "max_column_coverage" or "random"
        """
        if not cells or budget <= 0:
            return []

        if strategy_type == "max_column_coverage":
            return self._max_column_coverage_sample(lake, cells, budget)
        elif strategy_type == "random":
            return self._random_sample(lake, cells, budget)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _max_column_coverage_sample(self, lake: Lake, cells: List, budget: int) -> List:
        """
        Sample to maximize column coverage.
        Based on low_card.py implementation.
        """
        selected = {}
        covered_columns = set()

        # Group cells by column
        cells_by_column = defaultdict(list)
        column_cell_counts = defaultdict(int)

        for cell in cells:
            # Assuming cell is (table_id, col_idx, row_idx)
            column = (cell[0], cell[1])
            cells_by_column[column].append(cell)
            column_cell_counts[column] += 1

        # Sort columns by cell count (descending)
        sorted_columns = sorted(
            column_cell_counts.keys(),
            key=lambda col: column_cell_counts[col],
            reverse=True,
        )

        # Sample one cell from each column first, then cycle
        while len(selected) < budget and any(cells_by_column.values()):
            added_in_round = 0
            for column in sorted_columns:
                if len(selected) >= budget:
                    break

                if column not in covered_columns and cells_by_column[column]:
                    cell = self.rng.choice(cells_by_column[column])
                    cells_by_column[column].remove(cell)
                    selected[cell] = cells[cell]
                    lake.tables[cells[cell].table_id].add_sample(cells[cell])
                    covered_columns.add(column)
                    added_in_round += 1

            # Reset covered columns for next iteration
            if len(covered_columns) == len(sorted_columns) or added_in_round == 0:
                covered_columns = set()
        return selected

    def _random_sample(self, lake: Lake, cells: List, budget: int) -> List:
        """Random sampling"""
        selected = self.rng.sample(cells, min(budget, len(cells)))
        for cell in selected:
            lake.tables[cells[cell].table_id].add_sample(cells[cell])
        return selected

    def get_strategy_name(self) -> str:
        return "low_cardinality_max_column_coverage"
