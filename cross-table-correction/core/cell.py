from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.candidate import Candidate


@dataclass
class Cell:
    table_id: str
    column_idx: int
    row_idx: int
    invalid_character: List[str] = field(default_factory=list)
    candidates: Dict[str, "Candidate"] = field(default_factory=dict)
    zone: str = None
    predicted_corrections = None
    is_error: bool = False
    ground_truth: Optional[str] = None
    value: str = None
    
    @property
    def coordinates(self):
        return (self.table_id, self.column_idx, self.row_idx)

    def add_candidate(self, correction_value: str, features: dict):
        """Add a correction candidate to this cell"""
        candidate = Candidate(correction_value, features)
        self.candidates[correction_value] = candidate

    def add_candidates(self, candidates: Dict[str, Candidate]):
        """Add multiple candidates to this cell"""
        for correction, candidate in candidates.items():
            if correction in self.candidates:
                # Merge features from multiple generators
                existing = self.candidates[correction]
                merged_features = {**existing.features, **candidate.features}
                self.candidates[correction] = Candidate(correction, merged_features)
            else:
                self.candidates[correction] = candidate

    def get_best_candidate(self) -> Optional[Candidate]:
        """Get the candidate with highest confidence"""
        if not self.candidates:
            return None
        return max(self.candidates.values(), key=lambda c: c.confidence)

    def clear_candidates(self):
        """Clear all candidates"""
        self.candidates.clear()

    def to_dict(self) -> dict:
        """
        Safely convert Cell to a JSON-serializable dictionary.
        Filters out non-serializable fields like unexpected object references.
        """
        for attr_name, attr_value in vars(self).items():
            if attr_value.__class__.__name__ == "Table":
                raise RuntimeError(
                    f"❌ Cell contains a reference to a Table object: {attr_name}"
                )

        return {
            "table_id": str(self.table_id),
            "column_idx": int(self.column_idx),
            "row_idx": int(self.row_idx),
            "value": self.value,
            "is_error": bool(self.is_error),
            "ground_truth": self.ground_truth,
            "candidates": {k: v.to_dict() for k, v in self.candidates.items()},
            "predicted_corrections": self.predicted_corrections,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Rebuild Cell object from a JSON-compatible dict."""
        cell = cls(
            table_id=data["table_id"],
            column_idx=data["column_idx"],
            row_idx=data["row_idx"],
            value=data["value"],
            is_error=data.get("is_error", False),
            ground_truth=data.get("ground_truth"),
        )
        from core.candidate import Candidate  # prevent circular import

        cell.candidates = {
            k: Candidate.from_dict(v) for k, v in data.get("candidates", {}).items()
        }
        cell.predicted_corrections = data.get("predicted_corrections")
        return cell
