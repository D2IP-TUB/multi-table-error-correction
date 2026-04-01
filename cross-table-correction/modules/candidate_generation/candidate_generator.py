from abc import ABC, abstractmethod
from typing import Dict, Tuple

from core.candidate import Candidate
from core.cell import Cell
from core.table import Table
from core.candidate_pool import CandidatePool


class CandidateGenerator(ABC):
    """Abstract base class for candidate generation strategies"""

    def __init__(self, config):
        self.config = config
        self.min_probability = getattr(
            config.correction, "min_candidate_probability", 0.0
        )
        self.pool = CandidatePool.get_instance()

    @abstractmethod
    def generate_candidates(self, cell: Cell, table: Table) -> Dict[str, Tuple[str, int, str]]:
        """
        Generate correction candidates for a given cell.
        
        Returns:
            Dict mapping value -> candidate pool key (table_id, column_idx, value)
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this candidate generation strategy"""
        pass

    def _add_candidate_to_pool(
        self, 
        table_id: str, 
        column_idx: int, 
        value: str, 
        candidate: Candidate
    ) -> Tuple[str, int, str]:
        """Add a candidate to the pool and return the pool key"""
        return self.pool.add_candidate(table_id, column_idx, value, candidate)

    def _add_to_model(self, model: dict, key: str, value: str):
        """Helper method to incrementally add key-value pairs to a model"""
        if key != value:
            if key not in model:
                model[key] = {}
            if value not in model[key]:
                model[key][value] = 0.0
            model[key][value] += 1.0
