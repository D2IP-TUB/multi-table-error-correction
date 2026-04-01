import logging
from dataclasses import dataclass
from typing import Dict

from core.zone import Zone


@dataclass
class CandidateGenerationMetrics:
    """Metrics for candidate generation evaluation"""

    zone_name: str
    total_error_cells: int
    cells_with_candidates: int
    cells_with_correct_candidates: int
    total_candidates_generated: int
    avg_candidates_per_cell: float
    coverage: float  # % of error cells that have candidates
    recall: float  # % of error cells that have the correct candidate

    @property
    def candidate_coverage(self) -> float:
        """Percentage of error cells that have at least one candidate"""
        return self.coverage

    @property
    def ground_truth_recall(self) -> float:
        """Percentage of error cells where correct answer is in candidates"""
        return self.recall


def evaluate_candidate_generation_zone(
    zone: Zone, candidate_results: Dict
) -> CandidateGenerationMetrics:
    """Evaluate candidate generation performance for a single zone"""

    logging.info(f"Evaluating candidate generation for zone: {zone.name}")

    # Get zone results
    zone_results = candidate_results.get(zone.name, {})
    if not zone_results:
        logging.warning(f"No results found for zone: {zone.name}")
        return CandidateGenerationMetrics(
            zone_name=zone.name,
            total_error_cells=0,
            cells_with_candidates=0,
            cells_with_correct_candidates=0,
            total_candidates_generated=0,
            avg_candidates_per_cell=0.0,
            coverage=0.0,
            recall=0.0,
        )

    # Count error cells and candidates
    total_error_cells = 0
    cells_with_candidates = 0
    cells_with_correct_candidates = 0
    total_candidates = 0

    for cell in zone.cells.values():
        if not cell.is_error:
            continue

        total_error_cells += 1

        # Check if cell has candidates
        if hasattr(cell, "candidates") and cell.candidates:
            cells_with_candidates += 1
            total_candidates += len(cell.candidates)

            # Check if correct candidate exists
            if hasattr(cell, "ground_truth") and cell.ground_truth:
                if cell.ground_truth in cell.candidates:
                    cells_with_correct_candidates += 1

    # Calculate metrics
    coverage = (
        cells_with_candidates / total_error_cells if total_error_cells > 0 else 0.0
    )
    recall = (
        cells_with_correct_candidates / total_error_cells
        if total_error_cells > 0
        else 0.0
    )
    avg_candidates = (
        total_candidates / cells_with_candidates if cells_with_candidates > 0 else 0.0
    )

    metrics = CandidateGenerationMetrics(
        zone_name=zone.name,
        total_error_cells=total_error_cells,
        cells_with_candidates=cells_with_candidates,
        cells_with_correct_candidates=cells_with_correct_candidates,
        total_candidates_generated=total_candidates,
        avg_candidates_per_cell=avg_candidates,
        coverage=coverage,
        recall=recall,
    )

    return metrics
