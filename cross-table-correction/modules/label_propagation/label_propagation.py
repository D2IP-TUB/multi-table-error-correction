import logging
import math
import time
from typing import Any, Dict, List, Tuple

import hnswlib
import numpy as np

from core.candidate import Candidate  # Use existing class
from core.lake import Lake
from core.zone import Zone
from core.zone_propagation_data import ZonePropagationData
from modules.label_propagation.propagation_candidate import PropagationCandidate


def create_hnsw_index(
    zone_data: ZonePropagationData, config: Any
) -> Tuple[hnswlib.Index, np.ndarray]:
    """
    HNSW index creation
    """
    zone_name = zone_data.zone_name
    all_candidates = zone_data.train_candidates + zone_data.test_candidates
    n_candidates = len(all_candidates)

    if n_candidates == 0:
        raise ValueError(f"No candidates for zone {zone_name}")

    # Get feature dimensionality
    feature_dim = len(all_candidates[0].features)
    logging.info(
        f"Zone {zone_name}: Building index for {n_candidates:,} candidates, {feature_dim} dims"
    )

    # Override with config if available
    ef_construction = getattr(config.propagation, "ef_construction")
    M = getattr(config.propagation, "M")
    space = getattr(config.propagation, "space", "cosine")

    logging.info(
        f"Zone {zone_name}: Using params - ef_construction: {ef_construction}, M: {M}, space: {space}"
    )

    # Initialize HNSW index
    hnsw_index = hnswlib.Index(space=space, dim=feature_dim)
    hnsw_index.init_index(
        max_elements=n_candidates,
        ef_construction=ef_construction,
        M=M,
        random_seed=getattr(config.experiment, "random_state", 42),
    )

    # Optimize threading

    max_threads = getattr(config.runtime, "n_cores")
    hnsw_index.set_num_threads(max_threads)

    # Handle insertion order for deterministic results
    insertion_order = getattr(config.propagation, "insertion_order")
    if insertion_order == "column-based":
        # Sort by (col, row, table) for column-based ordering
        all_candidates.sort(key=lambda c: (c.col, c.row, c.table, c.candidate_id))
        logging.info(f"Zone {zone_name}: Using column-based insertion order")

    #  batch insertion
    features_matrix = np.vstack([candidate.features for candidate in all_candidates])
    ids_array = np.arange(n_candidates, dtype=np.int32)

    start_time = time.time()
    hnsw_index.add_items(features_matrix, ids_array)
    build_time = time.time() - start_time

    logging.info(
        f"Zone {zone_name}: Index built in {build_time:.2f}s ({n_candidates / build_time:.0f} candidates/sec)"
    )

    # Create efficient mapping array
    candidate_mapping = np.array(
        [candidate.full_key for candidate in all_candidates], dtype=object
    )

    return hnsw_index, candidate_mapping


def batch_query(
    hnsw_index: hnswlib.Index,
    candidate_mapping: np.ndarray,
    query_candidates: List[PropagationCandidate],
    k: int,
    config: Any,
) -> np.ndarray:
    """
    Batch querying with vectorized operations
    """

    try:
        if not query_candidates or k <= 0:
            return np.array([])

        n_test = len(query_candidates)
        logging.info(f"Batch querying {n_test:,} test candidates for top-{k}")

        # Configure HNSW for optimal querying
        ef_search = getattr(config.propagation, "ef_search")
        n_cores = getattr(config.runtime, "n_cores")

        hnsw_index.set_ef(ef_search)
        hnsw_index.set_num_threads(n_cores)

        logging.info(f"Using ef_search: {ef_search}, n_cores: {n_cores} for querying")
        # Vectorized query matrix creation
        query_matrix = np.vstack([candidate.features for candidate in query_candidates])
        logging.info(
            f"Query matrix created with shape {query_matrix.shape} for {n_test} candidates"
        )
        k_per_candidate = math.ceil(k / n_test) if n_test > 0 else k

        # Single batch query
        start_time = time.time()
        logging.info("Starting batch query...")
        ids_batch, distances_batch = hnsw_index.knn_query(
            query_matrix, k=k_per_candidate, num_threads=n_cores
        )
        query_time = time.time() - start_time

        logging.info(
            f"Batch query completed in {query_time:.2f}s ({n_test / query_time:.0f} queries/sec)"
        )

        # Top-k selection with vectorized operations
        all_neighbor_ids = ids_batch.flatten()
        all_distances = distances_batch.flatten()

        logging.info(
            f"Total neighbors found: {len(all_neighbor_ids)}, "
            f"Total distances computed: {len(all_distances)}"
        )

        # Use efficient numpy operations for top-k
        if len(all_distances) <= k:
            top_k_indices = np.argsort(all_distances)
        else:
            # Use argpartition for O(n) selection, then sort top-k
            top_k_indices = np.argpartition(all_distances, k)[:k]
            top_k_indices = top_k_indices[np.argsort(all_distances[top_k_indices])]

        top_k_neighbor_ids = all_neighbor_ids[top_k_indices]
    except Exception as e:
        logging.error(f"Error during batch query: {e}")
        return np.array([])
    return candidate_mapping[top_k_neighbor_ids]


def initialize_label_propagation(
    zones_dict: Dict[str, Zone], zones_data: dict, lake: Lake, config: Any
) -> Dict[str, List]:
    """
    Label propagation using Candidate class
    with same algorithm logic as attached file but better performance
    """
    start_time = time.time()
    logging.info("Starting label propagation with Zone integration")

    # Validate inputs
    if not zones_dict:
        logging.warning("No zones provided for label propagation")
        return {}

    # Pre-filter zones with candidates and samples (optimization)
    valid_zones = {}
    for zone_name, zone in zones_dict.items():
        # Check if zone has samples
        has_samples = hasattr(zone, "samples") and zone.samples

        # Check if zone has candidates
        has_candidates = any(
            hasattr(cell, "candidates")
            and cell.candidates
            and isinstance(next(iter(cell.candidates.values())), Candidate)
            for cell in zone.cells.values()
            if cell.is_error
        )

        if has_samples and has_candidates:
            valid_zones[zone_name] = zone
        else:
            logging.info(
                f"Zone {zone_name}: Skipped (samples: {has_samples}, candidates: {has_candidates})"
            )

    if not valid_zones:
        logging.warning(
            "No zones have both samples and valid candidates for label propagation"
        )
        return {}

    logging.info(f"Processing {len(valid_zones)} zones with samples and candidates")

    # Phase 2: Process each zone with operations
    pseudo_labels_zones = {}

    for zone_idx, (zone_name, zone_data) in enumerate(zones_data.items(), 1):
        zone_start = time.time()
        logging.info(f"[{zone_idx}/{len(zones_data)}] Processing zone: {zone_name}")

        try:
            # Check if pseudo labels are needed (same logic as attached)
            required_labels = zone_data.required_pseudo_labels
            if required_labels <= 0:
                logging.info(f"Zone {zone_name}: No pseudo labels needed (balanced)")
                pseudo_labels_zones[zone_name] = []
                continue

            if not zone_data.test_candidates:
                logging.warning(
                    f"Zone {zone_name}: No test candidates for pseudo labeling"
                )
                pseudo_labels_zones[zone_name] = []
                continue

            # Create  HNSW index
            logging.info(
                f"Zone {zone_name}: Creating HNSW index for {len(zone_data.train_candidates)} train candidates"
            )
            t0 = time.time()
            hnsw_index, candidate_mapping = create_hnsw_index(zone_data, config)
            index_time = time.time() - t0
            logging.info(
                f"Zone {zone_name}: HNSW index created with {len(candidate_mapping)} candidates, Index time: {index_time:.2f}s"
            )
            logging.info(
                f"Zone {zone_name}: Starting batch query for top-{required_labels} similar candidates"
            )

            t0 = time.time()
            # Batch query for top-k similar candidates
            top_k_candidates = batch_query(
                hnsw_index,
                candidate_mapping,
                zone_data.train_candidates,
                required_labels,
                config,
            )

            query_time = time.time() - t0
            logging.info(
                f"Zone {zone_name}: Batch query completed, found {len(top_k_candidates)} top-k candidates, Query time: {query_time:.2f}s"
            )

            # Convert to output format
            pseudo_labels = []
            for candidate_key in top_k_candidates:
                table_id, col_idx, row_idx, candidate_value = candidate_key
                pseudo_labels.append(
                    {
                        "table_id": table_id,
                        "col_idx": col_idx,
                        "row_idx": row_idx,
                        "candidate_value": candidate_value,
                    }
                )

            pseudo_labels_zones[zone_name] = pseudo_labels

            zone_time = time.time() - zone_start
            logging.info(
                f"Zone {zone_name}: Completed in {zone_time:.2f}s, "
                f"generated {len(pseudo_labels)} pseudo labels"
            )

            # ETA calculation
            if zone_idx < len(zones_data):
                avg_time = (time.time() - start_time) / zone_idx
                remaining = len(zones_data) - zone_idx
                eta = avg_time * remaining
                logging.info(f"ETA: {eta / 60:.1f} minutes remaining")

        except Exception as e:
            logging.error(f"Error processing zone {zone_name}: {e}")
            pseudo_labels_zones[zone_name] = []

    total_time = time.time() - start_time
    total_pseudo_labels = sum(len(labels) for labels in pseudo_labels_zones.values())

    logging.info(
        f"Label propagation completed in {total_time:.2f}s, "
        f"generated {total_pseudo_labels} total pseudo labels"
    )

    return pseudo_labels_zones
