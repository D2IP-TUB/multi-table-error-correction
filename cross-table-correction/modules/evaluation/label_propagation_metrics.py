import logging
from dataclasses import dataclass
from typing import Dict, List

from core.zone import Zone


@dataclass
class LabelPropagationMetrics:
    """Metrics for label propagation evaluation"""

    zone_name: str
    pseudo_labels_generated: int
    pseudo_labels_correct: int
    pseudo_labels_incorrect: int
    propagation_precision: float
    training_samples_before: int
    training_samples_after: int


def evaluate_label_propagation_zone(
    zone: Zone, pseudo_labels: List[Dict[str, str]]
) -> LabelPropagationMetrics:
    """Evaluate label propagation for a single zone"""

    if not pseudo_labels:
        logging.warning(f"No pseudo labels found for zone '{zone.name}'")
        return LabelPropagationMetrics(
            zone_name=zone.name,
            pseudo_labels_generated=0,
            pseudo_labels_correct=0,
            pseudo_labels_incorrect=0,
            propagation_precision=0.0,
            training_samples_before=0,
            training_samples_after=0,
        )

    # Count training samples before propagation
    training_samples_before = (
        len(zone.samples) if hasattr(zone, "samples") and zone.samples else 0
    )

    # Evaluate pseudo labels
    pseudo_labels_correct = 0
    pseudo_labels_incorrect = 0

    for pseudo_label in pseudo_labels:
        if len(pseudo_label) >= 4:
            table_id = pseudo_label["table_id"]
            col = pseudo_label["col_idx"]
            row = pseudo_label["row_idx"]
            candidate_value = pseudo_label["candidate_value"]
            cell_coordinates = (table_id, col, row)

            # Find the cell
            cell = None
            for c in zone.cells.values():
                if c.coordinates == cell_coordinates:
                    cell = c
                    break

            if cell and hasattr(cell, "ground_truth") and cell.ground_truth:
                if candidate_value == cell.ground_truth:
                    pseudo_labels_correct += 1
                else:
                    pseudo_labels_incorrect += 1

    # Calculate metrics
    total_pseudo = len(pseudo_labels)
    training_samples_after = training_samples_before + total_pseudo

    propagation_precision = (
        pseudo_labels_correct / total_pseudo if total_pseudo > 0 else 0.0
    )

    return LabelPropagationMetrics(
        zone_name=zone.name,
        pseudo_labels_generated=total_pseudo,
        pseudo_labels_correct=pseudo_labels_correct,
        pseudo_labels_incorrect=pseudo_labels_incorrect,
        propagation_precision=propagation_precision,
        training_samples_before=training_samples_before,
        training_samples_after=training_samples_after,
    )
