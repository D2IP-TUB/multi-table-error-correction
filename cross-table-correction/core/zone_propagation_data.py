from dataclasses import dataclass
from typing import List

from modules.label_propagation.propagation_candidate import PropagationCandidate


@dataclass
class ZonePropagationData:
    """Efficient zone data structure"""

    zone_name: str
    train_candidates: List[PropagationCandidate]
    test_candidates: List[PropagationCandidate]
    positive_count: int
    negative_count: int

    @property
    def total_candidates(self) -> int:
        return len(self.train_candidates) + len(self.test_candidates)

    @property
    def required_pseudo_labels(self) -> int:
        return max(0, self.negative_count - self.positive_count)
