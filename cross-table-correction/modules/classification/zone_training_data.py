from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ZoneTrainingData:
    """Training data for a single zone"""

    zone_name: str
    X_train: List[List[float]]  # Feature vectors
    y_train: List[int]  # Labels (0 or 1)
    feature_names: List[str]  # Feature names in order
    train_samples: List[Tuple]  # (table, col, row, candidate) tuples
    n_positive: int
    n_negative: int
    n_from_samples: int
