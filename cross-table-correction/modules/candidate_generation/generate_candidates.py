"""
Enhanced table-based candidate generation with caching and model sharing
"""

import logging
import time

from core.lake import Lake
from core.zone import Zone
from modules.candidate_generation.correction_pipeline import CorrectionPipeline


def update_with_labeled_samples_zone(
    zone: Zone, correction_pipeline: CorrectionPipeline
):
    """
    Update the correction pipeline with labeled samples from a specific zone.
    """
    if hasattr(zone, "samples") and zone.samples:
        sample_cells = list(zone.samples.values())
        logging.debug(
            f"Updating pipeline with {len(sample_cells)} labeled samples from zone '{zone.name}'"
        )
        correction_pipeline.update_with_labeled_samples(sample_cells)
    # if hasattr(zone, "flash_fill_samples") and zone.flash_fill_samples:
    #     synthetic_samples = zone.flash_fill_samples.values()
    #     logging.debug(
    #         f"Updating pipeline with {len(sample_cells)} labeled samples from zone '{zone.name}'"
    #     )
    #     correction_pipeline.update_with_synthetic_samples(
    #         synthetic_samples, "flash_filled_value"
    #     )
    else:
        logging.debug(
            f"No samples found in zone '{zone.name}' for candidate generation."
        )


def process_zone(zone: Zone, lake: Lake, correction_pipeline: CorrectionPipeline):
    """
    Process a single zone to generate candidates.
    """
    start_time = time.time()
    logging.info(f"Processing zone: {zone.name}")
    # Apply correction
    zone_results = correction_pipeline.correct_zone(zone, lake)
    zone_time = time.time() - start_time
    # Count results
    n_candidates = zone_results["n_candidates_generated"]
    n_cells = zone_results["n_cells_processed"]

    # Get table information
    tables_involved = list(
        set(cell.table_id for cell in zone.cells.values() if cell.is_error)
    )
    zone_results["tables_involved"] = tables_involved
    zone_results["zone_name"] = zone.name
    zone_results["n_candidates_generated"] = n_candidates
    zone_results["n_cells_processed"] = n_cells

    logging.info(f"Zone '{zone.name}': {n_candidates} candidates for {n_cells} cells")
    logging.info(
        f"  Time: {zone_time:.2f}s ({zone_time / max(n_cells, 1):.4f}s per cell)"
    )
    logging.info(f"  Tables: {len(tables_involved)}")

    return zone_results
