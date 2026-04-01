import logging

import numpy as np

from core.candidate import Candidate
from core.candidate_pool import CandidatePool
from core.zone import Zone
from modules.feature_extraction.edit_distance_feature_generator import (
    get_edit_distance_features_batch,
)


def extract_zone_candidates_features(zone: Zone, config):
    """
    Extract features from candidates in a zone, including co-occurrence and edit distance features.
    """

    error_cells = [
        cell
        for cell in zone.cells.values()
        if (cell.is_error and hasattr(cell, "candidates") and cell.candidates)
    ]

    no_candidate_cells = [cell for cell in zone.cells.values() if not cell.candidates]
    logging.info(f"Zone {zone.name}: Processing {len(error_cells)} error cells")
    logging.info(f"Zone {zone.name}: {len(no_candidate_cells)} cells with no candidates")

    # Determine if we should include pattern features based on zoning strategy
    # Pattern features are only meaningful for clustering-based zoning
    include_pattern_features = config.zoning.strategy == "clustering_based"
    
    # Get candidate pool
    pool = CandidatePool.get_instance()

    for cell in error_cells:
        cell_coords = (cell.table_id, cell.column_idx, cell.row_idx)

        # Extract candidate objects from pool for edit distance calculation
        candidates_dict = {}
        for candidate_value, pool_key in cell.candidates.items():
            candidate_obj = pool.get_candidate(pool_key)
            if candidate_obj:
                candidates_dict[candidate_value] = candidate_obj
        
        get_edit_distance_features_batch(cell, candidates_dict)

        for candidate_value, pool_key in cell.candidates.items():
            candidate_obj = pool.get_candidate(pool_key)
            if candidate_obj is None:
                continue

            try:
                if not hasattr(candidate_obj, "features") or not candidate_obj.features:
                    continue

                features_array = candidate_obj.get_features_array(include_pattern_features)

                if (
                    len(features_array) == 0
                    or np.any(np.isnan(features_array))
                    or np.any(np.isinf(features_array))
                ):
                    continue

            except Exception as e:
                logging.warning(
                    f"Error feature generation for candidate {candidate_value} for cell {cell_coords}: {e}"
                )
                continue
