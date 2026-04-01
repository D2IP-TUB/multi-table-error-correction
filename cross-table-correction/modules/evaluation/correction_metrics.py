import logging
from dataclasses import dataclass
from typing import Dict

from core.zone import Zone


@dataclass
class CorrectionMetrics:
    """End-to-end correction metrics"""

    zone_name: str
    total_error_cells: int
    cells_corrected: int
    cells_correctly_corrected: int
    cells_incorrectly_corrected: int
    cells_not_corrected: int
    correction_precision: float
    correction_recall: float
    correction_f1: float
    error_reduction_rate: float
    zone_correction_dict: Dict[str, Dict] = None
    zone_missed_cells: Dict[str, Dict] = None


def evaluate_end_to_end_corrections(
    zones_dict: Dict[str, Zone],
) -> Dict[str, CorrectionMetrics]:
    """Evaluate end-to-end correction performance"""

    logging.info("Evaluating end-to-end correction performance...")

    zone_metrics = {}

    for zone_name, zone in zones_dict.items():
        logging.info(f"Evaluating corrections for zone: {zone_name}")

        total_error_cells = 0
        cells_corrected = 0
        cells_correctly_corrected = 0
        cells_incorrectly_corrected = 0

        for cell in zone.cells.values():
            if not cell.is_error:
                continue

            total_error_cells += 1

            # Check if cell has predictions applied
            if hasattr(cell, "predicted_corrections") and cell.predicted_corrections:
                cells_corrected += 1

                # Check if any prediction is correct
                is_correctly_corrected = False
                if hasattr(cell, "ground_truth") and cell.ground_truth:
                    for prediction in cell.predicted_corrections:
                        if prediction["candidate"] == cell.ground_truth:
                            is_correctly_corrected = True
                            break

                if is_correctly_corrected:
                    cells_correctly_corrected += 1
                else:
                    cells_incorrectly_corrected += 1

        cells_not_corrected = total_error_cells - cells_corrected

        # Calculate metrics
        correction_precision = (
            cells_correctly_corrected / cells_corrected if cells_corrected > 0 else 0.0
        )
        correction_recall = (
            cells_correctly_corrected / total_error_cells
            if total_error_cells > 0
            else 0.0
        )
        correction_f1 = (
            2
            * (correction_precision * correction_recall)
            / (correction_precision + correction_recall)
            if (correction_precision + correction_recall) > 0
            else 0.0
        )
        error_reduction_rate = (
            cells_correctly_corrected / total_error_cells
            if total_error_cells > 0
            else 0.0
        )

        zone_metrics[zone_name] = CorrectionMetrics(
            zone_name=zone_name,
            total_error_cells=total_error_cells,
            cells_corrected=cells_corrected,
            cells_correctly_corrected=cells_correctly_corrected,
            cells_incorrectly_corrected=cells_incorrectly_corrected,
            cells_not_corrected=cells_not_corrected,
            correction_precision=correction_precision,
            correction_recall=correction_recall,
            correction_f1=correction_f1,
            error_reduction_rate=error_reduction_rate,
        )

        logging.info(
            f"Zone {zone_name}: Correction F1={correction_f1:.3f}, "
            f"Error reduction={error_reduction_rate:.3f}"
        )

    return zone_metrics


def evaluate_end_to_end_corrections_zone(
    zone: Zone,
) -> CorrectionMetrics:
    """Evaluate end-to-end corrections for a single zone"""

    logging.info(f"Evaluating end-to-end corrections for zone: {zone.name}")

    total_error_cells = 0
    cells_corrected = 0
    cells_correctly_corrected = 0
    cells_incorrectly_corrected = 0
    zone_correction_dict = {}
    zone_missed_cells = {}
    for cell_id, cell in zone.cells.items():
        if not cell.is_error:
            continue
        zone_correction_dict[cell_id] = {
            "cell_value": cell.value,
            "ground_truth": cell.ground_truth,
            "predicted_corrections": getattr(cell, "predicted_corrections", []),
        }
        total_error_cells += 1

        # Check if cell has predictions applied
        if hasattr(cell, "predicted_corrections") and cell.predicted_corrections:
            cells_corrected += 1

            # Check if any prediction is correct
            is_correctly_corrected = False
            if hasattr(cell, "ground_truth") and cell.ground_truth:
                for prediction in cell.predicted_corrections:
                    if prediction["candidate"] == cell.ground_truth:
                        is_correctly_corrected = True
                        break

            if is_correctly_corrected:
                cells_correctly_corrected += 1
            else:
                cells_incorrectly_corrected += 1
        else:
            # No predictions applied, count as not corrected
            zone_missed_cells[cell_id] = cell.to_dict()

    cells_correctly_corrected += len(zone.samples)
    cells_corrected += len(zone.samples)
    cells_not_corrected = total_error_cells - cells_corrected

    # Calculate metrics
    correction_precision = (
        cells_correctly_corrected / cells_corrected if cells_corrected > 0 else 0.0
    )
    correction_recall = (
        cells_correctly_corrected / total_error_cells if total_error_cells > 0 else 0.0
    )
    correction_f1 = (
        2
        * (correction_precision * correction_recall)
        / (correction_precision + correction_recall)
        if (correction_precision + correction_recall) > 0
        else 0.0
    )
    error_reduction_rate = (
        cells_correctly_corrected / total_error_cells if total_error_cells > 0 else 0.0
    )

    return CorrectionMetrics(
        zone_name=zone.name,
        total_error_cells=total_error_cells,
        cells_corrected=cells_corrected,
        cells_correctly_corrected=cells_correctly_corrected,
        cells_incorrectly_corrected=cells_incorrectly_corrected,
        cells_not_corrected=cells_not_corrected,
        correction_precision=correction_precision,
        correction_recall=correction_recall,
        correction_f1=correction_f1,
        error_reduction_rate=error_reduction_rate,
        zone_correction_dict=zone_correction_dict,
        zone_missed_cells=zone_missed_cells,
    )
