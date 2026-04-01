from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ZoneTestData:
    """Test data for a single zone"""

    zone_name: str
    X_test: List[List[float]]  # Feature vectors
    y_test: List[int]  # True labels (for evaluation)
    test_samples: List[Tuple]  # (table, col, row, candidate) tuples
    cells_with_correct_candidates: int
    total_test_cells: int
