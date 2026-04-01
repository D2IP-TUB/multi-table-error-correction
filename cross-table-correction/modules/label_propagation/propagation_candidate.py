from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PropagationCandidate:
    table: str
    col: int
    row: int
    candidate_id: str
    features: np.ndarray
    is_positive: bool = False

    @property
    def cell_key(self) -> Tuple[str, int, int]:
        return (self.table, self.col, self.row)

    @property
    def full_key(self) -> Tuple[str, int, int, str]:
        return (self.table, self.col, self.row, self.candidate_id)
