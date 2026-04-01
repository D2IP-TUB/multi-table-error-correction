from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Candidate:
    correction_value: str
    features: dict  # feature_name -> value
    _feature_cache: dict = field(default_factory=dict, repr=False)
    candidates_source_model: dict = None  # Additional metadata for source model

    def get_features_array(
        self,
        include_pattern_features: bool = True,
        disabled_feature_groups: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Get features as numpy array, using cache.

        Args:
            include_pattern_features: False for rule-based zoning, True for clustering-based.
            disabled_feature_groups: Feature groups to exclude entirely (for ablation).
        """
        cache_key = (include_pattern_features, tuple(sorted(disabled_feature_groups or [])))

        if cache_key not in self._feature_cache:
            from modules.feature_extraction.extract_features import (
                extract_features_from_candidate,
            )

            self._feature_cache[cache_key] = extract_features_from_candidate(
                self, include_pattern_features, disabled_feature_groups
            )

        return self._feature_cache[cache_key]

    def to_dict(self) -> dict:
        for attr_name, attr_value in vars(self).items():
            if attr_value.__class__.__name__ == "Table":
                raise RuntimeError(
                    f"❌ Cell contains a reference to a Table object: {attr_name}"
                )
        return {
            "correction_value": self.correction_value,
            "features": {k: float(v) for k, v in self.features.items()},
            "candidates_source_model": self.candidates_source_model,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            correction_value=data["correction_value"],
            features=data["features"],
            candidates_source_model=data["candidates_source_model"],
        )
