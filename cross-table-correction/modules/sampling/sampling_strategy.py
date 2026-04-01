import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies"""

    def __init__(self, config):
        self.config = config
        self.random_state = getattr(config.experiment, "random_state", 42)
        self.rng = random.Random(self.random_state)
        self.np_rng = np.random.RandomState(self.random_state)

    @abstractmethod
    def sample(self, cells: List, budget: int, **kwargs) -> List:
        """
        Sample cells from the given list according to the strategy.

        Args:
            cells: List of cells to sample from
            budget: Number of cells to sample
            **kwargs: Additional strategy-specific parameters

        Returns:
            List of sampled cells
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this sampling strategy"""
        pass
