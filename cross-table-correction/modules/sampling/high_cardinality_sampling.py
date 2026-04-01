import heapq
import unicodedata
from collections import defaultdict
from typing import Dict, List

from core.lake import Lake
from modules.sampling.sampling_strategy import SamplingStrategy


class HighCardinalitySamplingStrategy(SamplingStrategy):
    """Sampling strategy for high cardinality columns using greedy coverage"""

    def sample(
        self,
        lake: Lake,
        cells: dict,
        zone_labeling_budget: int,
    ) -> Dict:
        """
        Greedy sampling to maximize coverage of similar cells.
        """
        if len(cells) == 0 or zone_labeling_budget <= 0:
            return []

        # Build similarity neighborhoods
        neighborhoods = self._build_neighborhoods(cells)

        # Greedy algorithm to maximize coverage
        covered = set()
        samples_dict = {}

        # Priority queue: (-coverage_size, counter, cell)
        pq = []
        counter = 0

        for cell in cells:
            coverage = len(neighborhoods.get(cell, set()))
            heapq.heappush(pq, (-coverage, counter, cell))
            counter += 1

        while len(samples_dict) < zone_labeling_budget and pq:
            neg_coverage, _, cell = heapq.heappop(pq)

            # Calculate actual uncovered neighbors
            neighbors = neighborhoods.get(cell, set())
            uncovered = neighbors - covered

            if not uncovered:
                continue

            # Add cell to result and update covered set
            samples_dict[cell] = cells[cell]
            lake.tables[cells[cell].table_id].add_sample(cells[cell])
            covered.update(uncovered)

            # Early termination if we've covered everything
            if len(covered) >= len(cells):
                break

        return samples_dict

    def _build_neighborhoods(self, cells: dict) -> Dict:
        """Build similarity neighborhoods for cells"""
        identity_map = defaultdict(set)
        unicode_map = defaultdict(set)

        # Group cells by identity and unicode patterns
        for cell_idx, cell in cells.items():
            value = cell.value

            # Identity grouping
            identity = tuple(value)
            identity_map[identity].add(cell_idx)

            # Unicode category grouping
            unicode_cats = tuple(unicodedata.category(c) for c in value)
            unicode_map[unicode_cats].add(cell_idx)

        # Build neighborhoods
        neighborhoods = {}
        for cell_idx, cell in cells.items():
            value = cell.value
            identity = tuple(value)
            unicode_cats = tuple(unicodedata.category(c) for c in value)

            # Union similar cells
            similar_cells = identity_map[identity] | unicode_map[unicode_cats]
            neighborhoods[cell_idx] = similar_cells

        return neighborhoods

    def _random_sample(self, cells: List, budget: int) -> List:
        """Fallback random sampling"""
        return self.rng.sample(cells, min(budget, len(cells)))

    def get_strategy_name(self) -> str:
        return "syntactic_unique_greedy_coverage"
