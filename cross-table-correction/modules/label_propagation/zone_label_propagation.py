import logging
import time
from typing import Dict, List, Tuple

import hnswlib
import numpy as np

from core.lake import Lake
from core.zone import Zone
from core.zone_propagation_data import ZonePropagationData
from modules.feature_extraction.extract_features import extract_features_from_candidate
from modules.label_propagation.propagation_candidate import PropagationCandidate


def extract_zone_propagation_data(zone: Zone, zone_name: str) -> ZonePropagationData:
    """Extract propagation data from Zone object - performance optimized"""

    train_candidates = []
    test_candidates = []
    positive_count = 0
    negative_count = 0

    # O(1) lookup for sample cells instead of repeated iterations
    sample_cells = set()
    if hasattr(zone, "samples") and zone.samples:
        sample_cells = {
            sample_cell.coordinates for _, sample_cell in zone.samples.items()
        }

    # Process all error cells in the zone
    for cell in zone.cells.values():
        if not cell.is_error or not hasattr(cell, "candidates"):
            continue

        is_sample_cell = cell.coordinates in sample_cells

        # Process each candidate for this cell
        for correction_value, candidate in cell.candidates.items():
            try:
                # Extract features
                features_array = extract_features_from_candidate(candidate)

                # Create propagation candidate
                prop_candidate = PropagationCandidate(
                    table=cell.table_id,
                    col=cell.column_idx,
                    row=cell.row_idx,
                    candidate_id=correction_value,
                    features=features_array,
                    is_positive=False,  # Will set below for training data
                )

                if is_sample_cell:
                    # This is training data - check if positive
                    if (
                        hasattr(cell, "ground_truth")
                        and correction_value == cell.ground_truth
                    ):
                        prop_candidate.is_positive = True
                        positive_count += 1
                    else:
                        negative_count += 1

                    train_candidates.append(prop_candidate)
                else:
                    # This is test data (unlabeled)
                    test_candidates.append(prop_candidate)

            except Exception as e:
                logging.warning(
                    f"Error processing candidate {correction_value} for cell {cell.coordinates}: {e}"
                )
                continue

    return ZonePropagationData(
        zone_name=zone_name,
        train_candidates=train_candidates,
        test_candidates=test_candidates,
        positive_count=positive_count,
        negative_count=negative_count,
    )


def create_hnsw_index_for_zone(
    zone_data: ZonePropagationData, config
) -> Tuple[hnswlib.Index, np.ndarray]:
    """Create HNSW index for a zone - performance optimized"""

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

    # Adaptive parameters (keep exact logic)
    if n_candidates < 1000:
        ef_construction = 100
        M = 16
    elif n_candidates < 10000:
        ef_construction = 150
        M = 16
    else:
        ef_construction = 200
        M = 24

    # Initialize HNSW index
    space = getattr(config.propagation, "space", "cosine")
    hnsw_index = hnswlib.Index(space=space, dim=feature_dim)
    hnsw_index.init_index(
        max_elements=n_candidates,
        ef_construction=ef_construction,
        M=M,
        random_seed=getattr(config.experiment, "random_state", 42),
    )

    # Set threads
    max_threads = getattr(config.runtime, "n_cores", 4)
    hnsw_index.set_num_threads(max_threads)

    # Vectorized matrix creation for better performance
    features_matrix = np.vstack([candidate.features for candidate in all_candidates])
    ids_array = np.arange(n_candidates, dtype=np.int32)

    start_time = time.time()
    hnsw_index.add_items(features_matrix, ids_array)
    build_time = time.time() - start_time

    logging.info(f"Zone {zone_name}: Index built in {build_time:.2f}s")

    # Create mapping array
    candidate_mapping = np.array(
        [candidate.full_key for candidate in all_candidates], dtype=object
    )

    return hnsw_index, candidate_mapping


def query_similar_candidates(
    hnsw_index: hnswlib.Index,
    candidate_mapping: np.ndarray,
    test_candidates: List[PropagationCandidate],
    k: int,
    config,
) -> np.ndarray:
    """Query HNSW index for similar candidates - performance optimized"""

    if not test_candidates or k <= 0:
        return np.array([])

    n_test = len(test_candidates)
    logging.info(f"Querying {n_test:,} test candidates for top-{k}")

    # Configure HNSW
    ef_search = getattr(config.propagation, "ef_search", 200)
    n_cores = getattr(config.runtime, "n_cores", 4)

    hnsw_index.set_ef(ef_search)
    hnsw_index.set_num_threads(n_cores)

    # Vectorized query matrix creation
    query_matrix = np.vstack([candidate.features for candidate in test_candidates])

    # Batch query
    start_time = time.time()
    ids_batch, distances_batch = hnsw_index.knn_query(query_matrix, k=k)
    query_time = time.time() - start_time

    logging.info(f"Batch query completed in {query_time:.2f}s")

    # Get top-k globally
    all_neighbor_ids = ids_batch.flatten()
    all_distances = distances_batch.flatten()

    if len(all_distances) <= k:
        top_k_indices = np.argsort(all_distances)
    else:
        top_k_indices = np.argpartition(all_distances, k)[:k]
        top_k_indices = top_k_indices[np.argsort(all_distances[top_k_indices])]

    top_k_neighbor_ids = all_neighbor_ids[top_k_indices]
    return candidate_mapping[top_k_neighbor_ids]


def propagate_labels_for_zone(zone: Zone, zone_name: str, config) -> List[Tuple]:
    """Run label propagation for a single zone - performance optimized"""

    # Extract data from the Zone
    zone_data = extract_zone_propagation_data(zone, zone_name)

    # Check if propagation is needed
    required_labels = zone_data.required_pseudo_labels
    logging.info(
        f"Zone {zone_name}: Required pseudo labels = {required_labels}, "
        f"Positive count = {zone_data.positive_count}, Negative count = {zone_data.negative_count}"
    )
    if required_labels <= 0:
        logging.info(f"Zone {zone_name}: No pseudo labels needed")
        return []

    if not zone_data.test_candidates:
        logging.warning(f"Zone {zone_name}: No test candidates")
        return []

    logging.info(
        f"Zone {zone_name}: Need {required_labels} pseudo labels from {len(zone_data.test_candidates)} test candidates"
    )

    try:
        # Create HNSW index
        hnsw_index, candidate_mapping = create_hnsw_index_for_zone(zone_data, config)

        # Query for similar candidates
        pseudo_labels = query_similar_candidates(
            hnsw_index,
            candidate_mapping,
            zone_data.test_candidates,
            required_labels,
            config,
        )

        logging.info(f"Zone {zone_name}: Generated {len(pseudo_labels)} pseudo labels")
        return pseudo_labels.tolist()

    except Exception as e:
        logging.error(f"Error in label propagation for zone {zone_name}: {e}")
        return []


def apply_pseudo_labels_to_zone(zone: Zone, pseudo_labels: List[Tuple], lake: Lake):
    """Apply pseudo labels back to cells - performance optimized"""

    labels_applied = 0

    # Create O(1) coordinate lookup for better performance
    coord_to_cell = {cell.coordinates: cell for cell in zone.cells.values()}

    for label_info in pseudo_labels:
        if len(label_info) >= 4:
            table_id, col, row, candidate_id = label_info[:4]

            # Find the cell using O(1) lookup
            cell_coordinates = (table_id, col, row)
            cell = coord_to_cell.get(cell_coordinates)

            if cell:
                # Apply pseudo label
                if not hasattr(cell, "pseudo_labels"):
                    cell.pseudo_labels = []

                if candidate_id not in cell.pseudo_labels:
                    cell.pseudo_labels.append(candidate_id)
                    labels_applied += 1

    return labels_applied


def label_propagation_with_zones(
    zones_dict: Dict[str, Zone], lake: Lake, config
) -> Dict[str, List]:
    """
    Run label propagation using existing Zone and Lake structures - performance optimized

    Args:
        zones_dict: zones dictionary
        lake: Lake object
        config: configuration

    Returns:
        Dict mapping zone names to pseudo labels
    """

    start_time = time.time()
    logging.info("Starting label propagation with zones...")

    results = {}

    for zone_name, zone in zones_dict.items():
        zone_start = time.time()
        logging.info(f"Processing zone: {zone_name}")

        try:
            # Run propagation for this zone
            pseudo_labels = propagate_labels_for_zone(zone, zone_name, config)
            results[zone_name] = pseudo_labels

            # Apply labels back to cells
            labels_applied = apply_pseudo_labels_to_zone(zone, pseudo_labels, lake)

            zone_time = time.time() - zone_start
            logging.info(
                f"Zone {zone_name}: {len(pseudo_labels)} pseudo labels generated, "
                f"{labels_applied} applied in {zone_time:.2f}s"
            )

        except Exception as e:
            logging.error(f"Error processing zone {zone_name}: {e}")
            results[zone_name] = []

    total_time = time.time() - start_time
    total_labels = sum(len(labels) for labels in results.values())

    logging.info(
        f"Label propagation completed: {total_labels} total pseudo labels in {total_time:.2f}s"
    )

    return results
