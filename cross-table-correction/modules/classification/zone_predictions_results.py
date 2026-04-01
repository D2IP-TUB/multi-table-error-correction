import json
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ZonePredictionResults:
    """Prediction results for a zone"""

    zone_name: str
    y_pred: List[int]  # Predicted labels
    y_decision_score: List[float]  # Prediction confidence scores
    y_test: List[int]  # True labels (for evaluation)
    test_samples: List[Tuple]  # Corresponding (table, col, row, candidate)
    model_path: str  # Path where model was saved
    training_prediction_time: float
    prediction_time: float

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert numpy types to native Python types
        data["y_pred"] = [int(x) for x in data["y_pred"]]
        data["y_decision_score"] = [float(x) for x in data["y_decision_score"]]
        data["y_test"] = [int(x) for x in data["y_test"]]
        return data

    def to_json(self, indent=2):
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=self._json_serializer)

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def save_to_file(self, filepath):
        """Save to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=self._json_serializer)

    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str):
        """Create instance from JSON string"""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load_from_file(cls, filepath):
        """Load from JSON file"""
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))
