"""
Utilities for creating and managing cluster-based zones.
"""

import logging
import math
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from core.cell import Cell
from core.zone import Zone
from modules.profiling.clustering_sampling import ErrorClusteringSampler


def cluster_and_sample_with_labels(
    dirty_cells: List[Cell],
    features_matrix: np.ndarray,
    labeling_budget: int,
    random_state: int,
    feature_names: List[str] = None,
    lake=None,
    cluster_sampling_strategy: str = "column_coverage",
) -> Tuple[List[Cell], Dict, np.ndarray]:
    """
    Single-level MiniBatchKMeans clustering + sampling (column coverage or centroid).

    Args:
        dirty_cells: List of dirty cells to cluster and sample
        features_matrix: Feature matrix for clustering
        labeling_budget: Total number of cells to sample
        random_state: Seed from config (same as EXPERIMENT.random_state)
        feature_names: Names of features (optional, for stats)
        lake: Lake object (unused, kept for API compatibility)
        cluster_sampling_strategy: "column_coverage" (default), "centroid"
            (k nearest to KMeans center per cluster), or "kmeans_pp" (k-means++
            seeding within each cluster in scaled space).

    Returns:
        Tuple of (sampled_cells, statistics_dict, cluster_labels)
    """
    logging.info("Starting simple KMeans clustering and sampling...")
    logging.info(f"Using random_state: {random_state}")
    cluster_sampling_strategy = (cluster_sampling_strategy or "column_coverage").strip().lower()
    logging.info(f"Cluster sampling strategy: {cluster_sampling_strategy}")

    logging.info("Starting clustering and sampling pipeline...")

    n_clusters = math.ceil(math.sqrt(labeling_budget))
    logging.info(f"Number of clusters: ceil(sqrt({labeling_budget})) = {n_clusters}")

    clusterer = ErrorClusteringSampler(
        n_clusters=n_clusters,
        random_state=random_state,
    )

    cluster_labels = clusterer.fit_clusters(features_matrix)

    sampled_cells = clusterer.sample_from_clusters(
        dirty_cells,
        cluster_labels,
        labeling_budget,
        lake,
        strategy=cluster_sampling_strategy,
    )

    stats = {
        "n_clusters": len(set(cluster_labels)),
        "total_cells": len(dirty_cells),
        "sampled_cells": len(sampled_cells),
        "cluster_sampling_strategy": cluster_sampling_strategy,
    }

    clusterer.print_sampling_summary(sampled_cells, dirty_cells, cluster_labels)

    if feature_names:
        detailed_stats = clusterer.get_cluster_statistics(
            dirty_cells,
            cluster_labels,
            include_feature_stats=True,
            features_matrix=features_matrix,
            feature_names=feature_names,
        )
        stats["detailed_cluster_stats"] = detailed_stats

    logging.info(
        f"Clustering and sampling completed: {len(sampled_cells)} cells sampled "
        f"from {stats['n_clusters']} clusters"
    )

    return sampled_cells, stats, cluster_labels


def create_zones_from_clusters(
    dirty_cells: List[Cell], cluster_labels: np.ndarray, sampled_cells: List[Cell]
) -> Dict[int, Zone]:
    """
    Create separate zones based on clusters.

    Note: This function handles the two-level clustering case where:
    - dirty_cells were assigned to zone clusters (e.g., 2 * n_erroneous_columns clusters)
    - sampled_cells were selected from finer sub-clusters for better diversity
    - We need to map the samples back to their corresponding zone clusters

    Args:
        dirty_cells: All dirty cells
        cluster_labels: Cluster assignment for each cell (zone-level clusters)
        sampled_cells: Cells selected for labeling (from finer clusters)

    Returns:
        Dictionary mapping cluster_id to Zone object
    """
    logging.info("Creating zones from clusters...")

    # Create mapping of sampled cells for O(1) lookup
    sampled_cell_coords = {
        (cell.table_id, cell.column_idx, cell.row_idx) for cell in sampled_cells
    }

    # Group cells by cluster
    clusters = {}
    for cell, cluster_id in zip(dirty_cells, cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(cell)

    # Create zones
    zones = {}
    total_sampled_distributed = 0

    for cluster_id, cluster_cells in clusters.items():
        zone_name = f"cluster_{cluster_id}"
        zone = Zone(
            zone_name, labeling_budget=0
        )  # Will set proper budget after counting samples

        # Add all cells from this cluster to the zone
        for cell in cluster_cells:
            zone.add_cell(cell)

            # Check if this cell is a sample
            cell_coords = (cell.table_id, cell.column_idx, cell.row_idx)
            if cell_coords in sampled_cell_coords:
                zone.samples[cell_coords] = cell
                total_sampled_distributed += 1

        # Set the actual labeling budget for this zone
        zone.labeling_budget = len(zone.samples)
        zones[cluster_id] = zone

        logging.debug(
            f"Zone {zone_name}: {len(zone.cells)} cells, {len(zone.samples)} samples"
        )

    # Log summary statistics
    zone_sample_counts = [len(zone.samples) for zone in zones.values()]
    zones_with_samples = sum(1 for count in zone_sample_counts if count > 0)

    logging.info(
        f"Created {len(zones)} zones from clusters. "
        f"Zones with samples: {zones_with_samples}/{len(zones)} "
        f"({zones_with_samples / len(zones) * 100:.1f}%)"
    )
    logging.info(
        f"Total samples distributed: {total_sampled_distributed}/{len(sampled_cells)} "
        f"({total_sampled_distributed / len(sampled_cells) * 100:.1f}%)"
    )

    if zone_sample_counts:
        logging.info(
            f"Samples per zone: avg={np.mean(zone_sample_counts):.1f}, "
            f"min={min(zone_sample_counts)}, max={max(zone_sample_counts)}"
        )

    return zones


def validate_zones(
    zones: Dict[int, Zone], total_dirty_cells: int, total_sampled_cells: int
) -> bool:
    """
    Validate that zones were created correctly.

    Args:
        zones: Dictionary of created zones
        total_dirty_cells: Expected total number of dirty cells
        total_sampled_cells: Expected total number of sampled cells

    Returns:
        True if validation passes, False otherwise
    """
    total_zone_cells = sum(len(zone.cells) for zone in zones.values())
    total_zone_samples = sum(len(zone.samples) for zone in zones.values())

    # Check that all cells are accounted for
    if total_zone_cells != total_dirty_cells:
        logging.error(
            f"Zone validation failed: Expected {total_dirty_cells} cells, "
            f"but zones contain {total_zone_cells} cells"
        )
        return False

    # Check that all samples are accounted for
    if total_zone_samples != total_sampled_cells:
        logging.error(
            f"Zone validation failed: Expected {total_sampled_cells} samples, "
            f"but zones contain {total_zone_samples} samples"
        )
        return False

    # Check that every zone has at least one sample
    zones_without_samples = [
        zone.name for zone in zones.values() if len(zone.samples) == 0
    ]

    if zones_without_samples:
        logging.warning(
            f"Zone validation: {len(zones_without_samples)} zones without samples: {zones_without_samples[:5]}..."
            f"{'...' if len(zones_without_samples) > 5 else ''}"
        )
        # Only fail if more than 50% of zones are empty (too many empty zones indicates a problem)
        empty_zone_ratio = len(zones_without_samples) / len(zones)
        if empty_zone_ratio > 0.5:
            logging.error(
                f"Zone validation failed: Too many empty zones ({empty_zone_ratio:.1%}). "
                f"This suggests a problem with the clustering/sampling strategy."
            )
            return False
        else:
            logging.info(
                f"Zone validation: {empty_zone_ratio:.1%} empty zones is acceptable for random sampling"
            )

    logging.info(
        f"Zone validation passed: {len(zones)} zones, "
        f"{total_zone_cells} cells, {total_zone_samples} samples"
    )
    return True


def get_zone_summary_stats(zones: Dict[int, Zone]) -> Dict:
    """
    Get summary statistics for all zones.

    Args:
        zones: Dictionary of zones

    Returns:
        Dictionary with summary statistics
    """
    if not zones:
        return {}

    zone_sizes = [len(zone.cells) for zone in zones.values()]
    zone_sample_counts = [len(zone.samples) for zone in zones.values()]

    stats = {
        "num_zones": len(zones),
        "total_cells": sum(zone_sizes),
        "total_samples": sum(zone_sample_counts),
        "avg_zone_size": np.mean(zone_sizes),
        "min_zone_size": min(zone_sizes),
        "max_zone_size": max(zone_sizes),
        "std_zone_size": np.std(zone_sizes),
        "avg_samples_per_zone": np.mean(zone_sample_counts),
        "min_samples_per_zone": min(zone_sample_counts),
        "max_samples_per_zone": max(zone_sample_counts),
        "std_samples_per_zone": np.std(zone_sample_counts),
    }

    return stats


def print_zone_summary(zones: Dict[int, Zone]) -> None:
    """
    Print a summary of all zones.

    Args:
        zones: Dictionary of zones
    """
    stats = get_zone_summary_stats(zones)

    if not stats:
        logging.info("No zones to summarize")
        return

    logging.info("=== ZONE SUMMARY ===")
    logging.info(f"Number of zones: {stats['num_zones']}")
    logging.info(f"Total cells across zones: {stats['total_cells']}")
    logging.info(f"Total samples across zones: {stats['total_samples']}")
    logging.info(f"Average zone size: {stats['avg_zone_size']:.1f} cells")
    logging.info(
        f"Zone size range: {stats['min_zone_size']} - {stats['max_zone_size']} cells"
    )
    logging.info(f"Average samples per zone: {stats['avg_samples_per_zone']:.1f}")
    logging.info(
        f"Samples per zone range: {stats['min_samples_per_zone']} - {stats['max_samples_per_zone']}"
    )

    # Show individual zone details
    logging.info("\nIndividual zone details:")
    sorted_zones = sorted(zones.items(), key=lambda x: len(x[1].cells), reverse=True)

    for cluster_id, zone in sorted_zones:
        logging.info(
            f"  {zone.name}: {len(zone.cells)} cells, {len(zone.samples)} samples"
        )


def save_zone_metadata(zones: Dict[int, Zone], output_path: str) -> None:
    """
    Save zone metadata to a file for analysis.

    Args:
        zones: Dictionary of zones
        output_path: Path to save the metadata
    """
    import json
    from pathlib import Path

    try:
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Collect zone metadata
        zone_metadata = {}

        for cluster_id, zone in zones.items():
            # Get table and column coverage for this zone
            tables_in_zone = set()
            columns_in_zone = set()
            table_column_pairs = set()

            for cell in zone.cells.values():
                tables_in_zone.add(cell.table_id)
                columns_in_zone.add(cell.column_idx)
                table_column_pairs.add((cell.table_id, cell.column_idx))

            zone_metadata[str(cluster_id)] = {
                "zone_name": zone.name,
                "cluster_id": cluster_id,
                "total_cells": len(zone.cells),
                "sample_cells": len(zone.samples),
                "labeling_budget": zone.labeling_budget,
                "num_tables": len(tables_in_zone),
                "num_columns": len(columns_in_zone),
                "num_table_column_pairs": len(table_column_pairs),
                "tables": list(tables_in_zone),
                "columns": list(columns_in_zone),
                "table_column_pairs": [list(pair) for pair in table_column_pairs],
                "diversity_metrics": {
                    "table_diversity": len(tables_in_zone) / len(zone.cells),
                    "column_diversity": len(columns_in_zone) / len(zone.cells),
                    "table_column_diversity": len(table_column_pairs) / len(zone.cells),
                },
            }

        # Add summary statistics
        summary_stats = get_zone_summary_stats(zones)

        metadata = {
            "zones": zone_metadata,
            "summary": summary_stats,
            "creation_timestamp": str(pd.Timestamp.now())
            if "pd" in globals()
            else "unknown",
        }

        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Zone metadata saved to: {output_path}")

    except Exception as e:
        logging.error(f"Failed to save zone metadata: {e}")


def aggregate_zone_results(zone_results: Dict, evaluation_results: Dict) -> Dict:
    """
    Aggregate results across all zones for final reporting.

    Args:
        zone_results: Results from processing all zones
        evaluation_results: Evaluation results from all zones

    Returns:
        Aggregated results dictionary
    """
    # Get basic aggregation from evaluation_results
    aggregate_stats = evaluation_results.get("aggregate_stats", {})
    zone_evaluations = evaluation_results.get("zone_evaluations", {})

    # Add additional aggregations from zone processing
    total_candidates_generated = 0
    total_predictions_applied = 0
    zones_with_classifiers = 0

    for cluster_id, results in zone_results.items():
        candidate_stats = results.get("candidate_stats", {})
        total_candidates_generated += candidate_stats.get("n_candidates_generated", 0)
        total_predictions_applied += results.get("predictions_applied", 0)

        if results.get("classifier") is not None:
            zones_with_classifiers += 1

    # Calculate per-zone averages
    num_zones = len(zone_results)

    aggregate_stats.update(
        {
            "total_candidates_generated": total_candidates_generated,
            "total_predictions_applied": total_predictions_applied,
            "zones_with_classifiers": zones_with_classifiers,
            "avg_candidates_per_zone": total_candidates_generated / num_zones
            if num_zones > 0
            else 0,
            "avg_predictions_per_zone": total_predictions_applied / num_zones
            if num_zones > 0
            else 0,
        }
    )

    # Get per-zone performance metrics
    zone_performances = []
    for cluster_id, evaluation in zone_evaluations.items():
        zone_metrics = evaluation.get("zone_metrics", {})
        total_error_cells_in_zone = zone_metrics.get("total_error_cells", 0)
        
        # Skip zones with no errors (they don't contribute to evaluation metrics)
        if total_error_cells_in_zone == 0:
            continue
        
        zone_performances.append(
            {
                "cluster_id": cluster_id,
                "zone_name": evaluation.get("zone_name", f"cluster_{cluster_id}"),
                "f1_score": zone_metrics.get("f1_score", 0.0),
                "precision": zone_metrics.get("precision", 0.0),
                "recall": zone_metrics.get("recall", 0.0),
                "total_error_cells": zone_metrics.get("total_error_cells", 0),
                "cells_corrected": zone_metrics.get("cells_corrected", 0),
                "correct_corrections": zone_metrics.get("correct_corrections", 0),
                "incorrect_corrections": zone_metrics.get("incorrect_corrections", 0),
                "pattern_enforcement_correct": zone_metrics.get("pattern_enforcement_correct", 0),
                "manual_samples_correct": zone_metrics.get("manual_samples_correct", 0),
                "manual_samples_incorrect": zone_metrics.get("manual_samples_incorrect", 0),
            }
        )

    # Sort by F1 score
    zone_performances.sort(key=lambda x: x["f1_score"], reverse=True)

    # Calculate performance statistics across zones
    if zone_performances:
        f1_scores = [z["f1_score"] for z in zone_performances]
        precisions = [z["precision"] for z in zone_performances]
        recalls = [z["recall"] for z in zone_performances]

        aggregate_stats.update(
            {
                "avg_zone_f1": np.mean(f1_scores),
                "std_zone_f1": np.std(f1_scores),
                "min_zone_f1": min(f1_scores),
                "max_zone_f1": max(f1_scores),
                "avg_zone_precision": np.mean(precisions),
                "avg_zone_recall": np.mean(recalls),
            }
        )

    return {
        "aggregate_stats": aggregate_stats,
        "zone_performances": zone_performances,
        "zone_evaluations": zone_evaluations,
        "zone_results": zone_results,
    }
