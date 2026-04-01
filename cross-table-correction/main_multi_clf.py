# -*- coding: utf-8 -*-
"""
Multi-Table Error Correction Pipeline with Cluster-based Zones and Multiple Classifiers.

This module bootstraps the complete pipeline including:
- Lake initialization and column profiling
- Feature extraction and clustering/sampling
- Creating zones based on clusters
- Candidate generation and classification per zone
- Evaluation and reporting
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from cluster_zone_utils import (aggregate_zone_results,
                                cluster_and_sample_with_labels,
                                create_zones_from_clusters, print_zone_summary,
                                validate_zones)
from config.parse_config import read_ecs_config
from core.candidate_pool import CandidatePool
from core.cell import Cell
from core.zone import Zone
from inits import initialize_lake, initialize_zones_cell_wise
from modules.candidate_generation.correction_pipeline import CorrectionPipeline
from modules.classification.train_test import (
    extract_test_data_from_zone,
    extract_training_data_from_zone,
    train_zone_classifier,
)
from modules.evaluation.eval import (apply_predictions_to_zone, evaluate_zone,
                                     save_detailed_results)
from modules.feature_extraction.extract_zone_data import \
    extract_zone_candidates_features
from modules.profiling.initialize import initialize_profiles
from modules.profiling.tane_wrapper import TANEWrapper
from modules.profiling.unusualness_feature_extractor import \
    extract_unusualness_features_for_lake
from utils.app_logger import setup_logging
from utils.memory_monitor import MemoryMonitor


def save_sampling_results(
    config, sampled_cells: List[Cell], stats: Dict, zones: Dict[int, Zone]
) -> None:
    """
    Save sampling results and zone information to files for analysis.

    Args:
        config: Pipeline configuration
        sampled_cells: List of sampled cells
        stats: Sampling statistics
        zones: Dictionary of zones created from clusters
    """
    output_dir = config.directories.output_dir

    try:
        # Save sampled cells
        with open(Path(output_dir) / "sampled_cells.txt", "w") as f:
            for cell in sampled_cells:
                f.write(f"{cell}\n")
                f.write(
                    f"Table: {cell.table_id}, Col: {cell.column_idx}, Row: {cell.row_idx}\n"
                )
                f.write("\n")

        # Save statistics
        with open(Path(output_dir) / "sampling_stats.txt", "w") as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
                f.write("\n")

        # Save zone information
        with open(Path(output_dir) / "zones_info.txt", "w") as f:
            f.write("Zone Information:\n")
            f.write("================\n\n")
            for cluster_id, zone in zones.items():
                # Update labeling budget to reflect actual samples
                zone.labeling_budget = len(zone.samples)
                
                f.write(f"Zone: {zone.name} (Cluster {cluster_id})\n")
                f.write(f"  Total cells: {len(zone.cells)}\n")
                f.write(f"  Sample cells: {len(zone.samples)}\n")
                f.write(f"  Labeling budget: {zone.labeling_budget}\n")
                f.write("\n")

        logging.info("Sampling results and zone info saved successfully")

    except Exception as e:
        logging.error(f"Failed to save sampling results: {e}")


def create_single_zone_cell_analysis(zone_name: str, zone, evaluation: Dict, lake, config, analysis_output_dir: str) -> List[Dict]:
    """
    Create cell-level analysis for a single zone and save results.
    
    Args:
        zone_name: Name of the zone
        zone: Zone object with cells
        evaluation: Pre-computed evaluation dict for this zone
        lake: Lake object
        config: Configuration
        analysis_output_dir: Output directory for analysis files
    
    Returns:
        List of analysis records for this zone
    """
    logging.info(f"Creating cell-level analysis for zone {zone_name}...")
    
    # Load clean data cache
    clean_data_cache = {}
    for table_name in lake.tables.keys():
        table = lake.tables[table_name]
        if hasattr(table, "clean_data") and table.clean_data is not None:
            clean_data_cache[table_name] = table.clean_data
    
    analysis_records = []
    
    # Process all cells in the zone
    for cell_coords, cell in zone.cells.items():
        table_id, column_idx, row_idx = cell_coords

        # Fast ground truth lookup
        ground_truth = "UNKNOWN"
        if table_id in clean_data_cache:
            try:
                clean_array = clean_data_cache[table_id]
                if (
                    row_idx < clean_array.shape[0]
                    and column_idx < clean_array.shape[1]
                ):
                    ground_truth = str(clean_array[row_idx, column_idx])
            except (IndexError, ValueError):
                pass
        elif hasattr(cell, "ground_truth") and cell.ground_truth:
            ground_truth = cell.ground_truth

        current_value = getattr(cell, "value", "UNKNOWN")

        # Extract prediction details
        if hasattr(cell, "predicted_corrections") and cell.predicted_corrections:
            selected_value = cell.predicted_corrections.get("candidate")
            prediction_source = cell.predicted_corrections.get("source", "classifier")
        else:
            selected_value = None
            prediction_source = None
        
        # Extract candidate count and score if available
        num_candidates = len(cell.candidates) if hasattr(cell, "candidates") and cell.candidates else 0
        selected_candidate_score = None
        if selected_value and hasattr(cell, "candidates") and selected_value in cell.candidates:
            # Get candidate from pool
            pool = CandidatePool.get_instance()
            pool_key = cell.candidates[selected_value]
            candidate_obj = pool.get_candidate(pool_key) if isinstance(pool_key, tuple) else pool_key
            if candidate_obj:
                selected_candidate_score = getattr(candidate_obj, "score", None)

        # Determine correction status
        if selected_value:
            correction_status = (
                "CORRECT_CORRECTION"
                if selected_value == ground_truth
                else "INCORRECT_CORRECTION"
            )
        elif current_value == ground_truth:
            correction_status = "NO_CORRECTION_NEEDED"
        else:
            correction_status = "MISSED_ERROR"

        # Determine if cell is an error
        is_error = (
            current_value != ground_truth if ground_truth != "UNKNOWN" else None
        )

        # Create record
        record = {
            "table_id": table_id,
            "column_idx": column_idx,
            "row_idx": row_idx,
            "zone_name": zone_name,
            "current_value": current_value,
            "ground_truth": ground_truth,
            "selected_value": selected_value,
            "correction_status": correction_status,
            "num_candidates": num_candidates,
            "selected_candidate_score": selected_candidate_score,
            "prediction_source": prediction_source,
            "is_error": is_error,
        }
        
        analysis_records.append(record)
    
    logging.info(f"Zone {zone_name}: Completed cell-level analysis for {len(analysis_records)} cells")
    
    return analysis_records


def save_zone_cell_analysis(zone_name: str, records: List[Dict], analysis_output_dir: str) -> None:
    """Save cell-level analysis records for a single zone to file."""
    if not records:
        return
    
    try:
        import pandas as pd
        
        df = pd.DataFrame(records)
        output_file = os.path.join(analysis_output_dir, f"cell_analysis_{zone_name}.csv")
        df.to_csv(output_file, index=False)
        logging.info(f"Saved cell analysis for {zone_name} to {output_file}")
    except Exception as e:
        logging.warning(f"Failed to save cell analysis for {zone_name}: {e}")


def process_all_zones(
    zones: Dict[int, Zone], lake, all_sampled_cells: List[Cell], config, memory_monitor=None
) -> Dict:
    """
    Process all zones: candidate generation, training, and prediction.

    Args:
        zones: Dictionary of zones to process
        lake: Lake object
        config: Configuration
        memory_monitor: Optional MemoryMonitor instance for memory enforcement

    Returns:
        Dictionary with results for all zones
    """
    logging.info(f"Processing {len(zones)} zones...")

    all_zone_results = {}
    execution_times = {
        "candidate_generation": 0,
        "candidate_feature_extraction": 0,
        "training": 0,
        "prediction": 0,
        "apply_predictions": 0,
        "evaluation": 0,
        "cell_analysis": 0,
    }
    
    # Create analysis output directory
    analysis_output_dir = os.path.join(config.directories.output_dir, "cell_analysis")
    os.makedirs(analysis_output_dir, exist_ok=True)

    # Sort zones to prioritize invalid zones first, then valid zones
    # This ensures cells get corrected via pattern enforcement before classification
    sorted_zones = sorted(
        zones.items(),
        key=lambda item: (item[1].name if "_valid_pattern" in item[1].name else "invalid_pattern"),
        reverse=False
    )

    for cluster_id, zone in sorted_zones:
        zone_name = zone.name
        logging.info(f"Processing zone {zone_name}...")

        # For invalid zones: ALWAYS run pattern enforcement before classifier stage
        # For valid zones: only process if they have samples (needs samples for classifier training)
        is_invalid_zone = "invalid_pattern" in zone_name
        
        if not is_invalid_zone and not zone.samples:
            logging.info(f"Zone {zone_name} has no samples, skipping...")
        # === CANDIDATE GENERATION ===
        logging.info(f"Zone {zone_name}: Generating candidates...")
        t0 = time.time()

        # Check memory before candidate generation
        if memory_monitor:
            memory_monitor.check_and_enforce(f"Before correction for zone {zone_name}")

        correction_pipeline = CorrectionPipeline(
            config,
            enabled_strategies=config.correction.strategies,
        )
        
        # Generate candidates for all cells in this zone
        zone_candidate_stats = correction_pipeline.correct_zone(zone, lake)
        
        # Check memory after candidate generation
        if memory_monitor:
            memory_monitor.check_and_enforce(f"After correction for zone {zone_name}")
        
        execution_times["candidate_generation"] += time.time() - t0
        if is_invalid_zone:
            logging.info(
                f"Zone {zone_name}: Pattern enforcement generated "
                f"{zone_candidate_stats.get('n_candidates_generated', 0)} candidates"
            )
            # Store results for this zone
            all_zone_results[cluster_id] = {
                "zone_name": zone_name,
                "zone": zone,
                "candidate_stats": zone_candidate_stats,
                "training_data": None,
                "classifier": None,
                "prediction_results": None,
                "predictions_applied": 0,
                "pattern_enforced": zone_candidate_stats.get("n_pattern_enforced", 0),
                "pattern_correct": zone_candidate_stats.get("n_pattern_correct", 0),
                "pattern_incorrect": zone_candidate_stats.get("n_pattern_incorrect", 0),
            }

        classifier = None
        training_data = None
        prediction_results = None
        predictions_applied = 0
        if zone_candidate_stats.get("n_candidates_generated", 0) == 0:
            logging.warning(f"Zone {zone_name}: No candidates generated, skipping...")
        else:
            # === CANDIDATE FEATURE EXTRACTION ===
            logging.info(f"Zone {zone_name}: Extracting candidate features...")
            t0 = time.time()

            extract_zone_candidates_features(zone, config)
            execution_times["candidate_feature_extraction"] += time.time() - t0

            # === TRAINING ===
            logging.info(f"Zone {zone_name}: Training classifier...")
            t0 = time.time()

            # Determine if we should include pattern features based on zoning strategy
            # Pattern features are only meaningful for clustering-based zoning
            include_pattern_features = config.zoning.strategy == "clustering_based"

            # Respect config.training.negative_pruning_enabled (do not use *_with_pruning,
            # which forces pruning on regardless of config).
            training_data = extract_training_data_from_zone(
                zone, config, include_pattern_features
            )
            classifier = train_zone_classifier(training_data, config)
            execution_times["training"] += time.time() - t0

        if not classifier:
            logging.warning(f"Zone {zone_name}: Training failed, skipping...")
            
        else:
            # === PREDICTION ===
            logging.info(f"Zone {zone_name}: Making predictions...")
            t0 = time.time()

            prediction_results = extract_test_data_from_zone(
                classifier, zone, zone_name, config, include_pattern_features
            )
            execution_times["prediction"] += time.time() - t0

            # === APPLY PREDICTIONS ===
            logging.info(f"Zone {zone_name}: Applying predictions...")
            t0 = time.time()

            predictions_applied = apply_predictions_to_zone(zone, prediction_results)
            execution_times["apply_predictions"] += time.time() - t0

        # === IMMEDIATE EVALUATION (while zone is hot in memory) ===
        logging.info(f"Zone {zone_name}: Evaluating...")
        t0 = time.time()
        
        evaluation = evaluate_zone(zone, lake)
        execution_times["evaluation"] = execution_times.get("evaluation", 0) + (time.time() - t0)
        
        zone_metrics = evaluation.get("zone_metrics", {})
        
        # === IMMEDIATE CELL-LEVEL ANALYSIS (while zone is hot in memory) ===
        t0 = time.time()
        
        # Check if cell analysis is enabled
        analysis_mode = getattr(config.experiment, "cell_analysis_mode", "full")
        if analysis_mode == "full":
            cell_records = create_single_zone_cell_analysis(
                zone_name, zone, evaluation, lake, config, analysis_output_dir
            )
            save_zone_cell_analysis(zone_name, cell_records, analysis_output_dir)
        
        execution_times["cell_analysis"] = execution_times.get("cell_analysis", 0) + (time.time() - t0)
        
        # Store results WITHOUT zone object (it will be cleared after analysis)
        all_zone_results[cluster_id] = {
            "zone_name": zone_name,
            "evaluation": evaluation,  # Full evaluation with all nested data
            "candidate_stats": zone_candidate_stats,
            "pattern_enforced": zone_candidate_stats.get("n_pattern_enforced", 0),
            "pattern_correct": zone_candidate_stats.get("n_pattern_correct", 0),
            "pattern_incorrect": zone_candidate_stats.get("n_pattern_incorrect", 0),
        }

        logging.info(
            f"Zone {zone_name} completed: "
            f"F1={zone_metrics.get('f1_score', 0):.3f}, "
            f"Precision={zone_metrics.get('precision', 0):.3f}, "
            f"Recall={zone_metrics.get('recall', 0):.3f}"
        )

    logging.info(f"All zones processed. Total execution times: {execution_times}")
    return all_zone_results, execution_times


def evaluate_all_zones(zone_results: Dict, lake) -> Dict:
    """
    Aggregate pre-computed zone evaluations (zones already evaluated during processing).

    Args:
        zone_results: Results from processing all zones (includes pre-computed evaluations)
        lake: Lake object

    Returns:
        Aggregated evaluation results
    """
    logging.info("Aggregating pre-computed zone evaluations...")

    all_evaluations = {}
    aggregate_stats = {
        "total_error_cells": 0,
        "total_corrected": 0,
        "total_correct_corrections": 0,
        "total_incorrect_corrections": 0,
        "total_pattern_enforcement_correct": 0,
        "total_manual_samples_correct": 0,
        "total_manual_samples_incorrect": 0,
        "zones_processed": 0,
        "total_pattern_enforced": 0,
        "total_pattern_correct": 0,
        "total_pattern_incorrect": 0,
    }

    aggregate_stats["total_error_cells"] = lake.n_errors

    for cluster_id, results in zone_results.items():
        zone_name = results.get("zone_name", f"Zone {cluster_id}")

        # Get pre-computed evaluation (already evaluated during processing)
        evaluation = results.get("evaluation", {})
        all_evaluations[cluster_id] = evaluation

        # Aggregate statistics from pre-computed evaluation
        zone_metrics = evaluation.get("zone_metrics", {})
        aggregate_stats["total_corrected"] += zone_metrics.get("cells_corrected", 0)
        aggregate_stats["total_correct_corrections"] += zone_metrics.get(
            "correct_corrections", 0
        )
        aggregate_stats["total_incorrect_corrections"] += zone_metrics.get(
            "incorrect_corrections", 0
        )
        aggregate_stats["total_pattern_enforcement_correct"] += zone_metrics.get(
            "pattern_enforcement_correct", 0
        )
        aggregate_stats["total_manual_samples_correct"] += zone_metrics.get(
            "manual_samples_correct", 0
        )
        aggregate_stats["total_manual_samples_incorrect"] += zone_metrics.get(
            "manual_samples_incorrect", 0
        )
        
        # Collect pattern enforcement stats (from invalid zones only, not double-counted in eval)
        aggregate_stats["total_pattern_enforced"] += results.get("pattern_enforced", 0)
        aggregate_stats["total_pattern_correct"] += results.get("pattern_correct", 0)
        aggregate_stats["total_pattern_incorrect"] += results.get("pattern_incorrect", 0)
        
        aggregate_stats["zones_processed"] += 1

        logging.info(
            f"Zone {zone_name}: F1={zone_metrics.get('f1_score', 0):.3f}, "
            f"Precision={zone_metrics.get('precision', 0):.3f}, "
            f"Recall={zone_metrics.get('recall', 0):.3f}"
        )

    # Validation: ensure all zones are properly processed
    expected_zones = set(zone_results.keys())
    if len(expected_zones) < 4:
        logging.warning(f"Only {len(expected_zones)} zones in results, expected 4: {expected_zones}")

    # Calculate overall metrics
    overall_precision = (
        aggregate_stats["total_correct_corrections"]
        / aggregate_stats["total_corrected"]
        if aggregate_stats["total_corrected"] > 0
        else 0.0
    )
    overall_recall = (
        aggregate_stats["total_correct_corrections"]
        / aggregate_stats["total_error_cells"]
        if aggregate_stats["total_error_cells"] > 0
        else 0.0
    )
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    aggregate_stats.update(
        {
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
        }
    )

    logging.info("Overall Results:")
    logging.info(f"  Total zones: {aggregate_stats['zones_processed']}")
    logging.info(f"  Overall F1: {overall_f1:.3f}")
    logging.info(f"  Overall Precision: {overall_precision:.3f}")
    logging.info(f"  Overall Recall: {overall_recall:.3f}")
    logging.info(f"  Pattern enforcement: {aggregate_stats['total_pattern_enforced']} attempted, "
                 f"{aggregate_stats['total_pattern_correct']} correct, "
                 f"{aggregate_stats['total_pattern_incorrect']} incorrect")
    logging.info(f"  Breakdown of correct corrections:")
    classifier_correct = (aggregate_stats['total_correct_corrections'] - 
                         aggregate_stats['total_pattern_enforcement_correct'] - 
                         aggregate_stats['total_manual_samples_correct'])
    logging.info(f"    - Pattern enforcement: {aggregate_stats['total_pattern_enforcement_correct']}")
    logging.info(f"    - Direct manual samples: {aggregate_stats['total_manual_samples_correct']}")
    logging.info(f"    - Classifier predictions: {classifier_correct}")

    return {"zone_evaluations": all_evaluations, "aggregate_stats": aggregate_stats}


def create_cell_analysis_summary_only(zone_results: Dict, lake, config) -> None:
    """
    Create lightweight summary of cell-level corrections without detailed CSV (for very large datasets).

    Args:
        zone_results: Dictionary containing zone analysis results
        lake: Lake object containing tables and cell information
        config: Configuration object
    """
    logging.info("Creating cell-level analysis summary (lightweight mode)...")
    start_time = time.time()

    try:
        summary_file = os.path.join(
            config.experiment.out_path, "cell_level_summary.json"
        )

        # Collect statistics by iterating zones and their cells
        stats = {
            "total_corrections": 0,
            "tables_with_corrections": set(),
            "zones_with_corrections": set(),
            "confidence_sum": 0.0,
            "classifier_corrections": 0,
            "pattern_enforcement_correct": 0,
            "corrections_per_zone": {},
            "corrections_per_table": {},
        }

        total_zones = len(zone_results)
        processed_zones = 0

        logging.info(f"Analyzing {total_zones} zones for summary statistics...")

        for zone_id, zone_result in zone_results.items():
            zone = zone_result.get("zone")
            if not zone:
                logging.warning(f"Zone {zone_id} has no zone object, skipping...")
                continue

            zone_correction_count = 0

            # Iterate cells in this zone to count corrections
            for _, cell in zone.cells.items():
                # Check if cell has a prediction
                if hasattr(cell, "predicted_corrections") and cell.predicted_corrections:
                    table_id = cell.table_id
                    
                    stats["total_corrections"] += 1
                    zone_correction_count += 1
                    stats["tables_with_corrections"].add(table_id)
                    stats["zones_with_corrections"].add(zone_id)

                    # Update per-table stats
                    if table_id not in stats["corrections_per_table"]:
                        stats["corrections_per_table"][table_id] = 0
                    stats["corrections_per_table"][table_id] += 1

                    # Extract correction details
                    pred_dict = cell.predicted_corrections
                    confidence = float(pred_dict.get("confidence", 0.0))
                    stats["confidence_sum"] += confidence
                    
                    # Track source of correction (classifier vs pattern enforcement correct)
                    source = pred_dict.get("source", "classifier")
                    if source == "pattern_enforcement_correct":
                        stats["pattern_enforcement_correct"] += 1
                    else:
                        stats["classifier_corrections"] += 1

            if zone_correction_count > 0:
                stats["corrections_per_zone"][zone_id] = zone_correction_count

            processed_zones += 1
            if processed_zones % 10 == 0:
                logging.info(f"Analyzed {processed_zones}/{total_zones} zones ({stats['total_corrections']} corrections so far)...")

        # Finalize statistics
        final_stats = {
            "total_corrections": stats["total_corrections"],
            "classifier_corrections": stats["classifier_corrections"],
            "pattern_enforcement_correct": stats["pattern_enforcement_correct"],
            "unique_tables_with_corrections": len(stats["tables_with_corrections"]),
            "unique_zones_with_corrections": len(stats["zones_with_corrections"]),
            "avg_confidence": stats["confidence_sum"]
            / max(stats["total_corrections"], 1),
            "top_tables_by_corrections": dict(
                sorted(
                    stats["corrections_per_table"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),  # Top 10 tables
            "top_zones_by_corrections": dict(
                sorted(
                    stats["corrections_per_zone"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),  # Top 10 zones
        }

        # Write summary
        with open(summary_file, "w") as f:
            json.dump(final_stats, f, indent=2)

        elapsed_time = time.time() - start_time
        logging.info(f"Cell-level summary completed in {elapsed_time:.2f} seconds")
        logging.info(f"Created {summary_file}")
        logging.info(
            f"Summary: {final_stats['total_corrections']} corrections across {final_stats['unique_tables_with_corrections']} tables and {final_stats['unique_zones_with_corrections']} zones"
        )

    except Exception as e:
        logging.error(f"Error creating cell-level summary: {e}")
        import traceback

        traceback.print_exc()


def create_cell_level_analysis(zone_results: Dict, lake, config) -> None:
    """
    Create detailed cell-level analysis CSV with predictions for all sampled cells.

    Args:
        zone_results: Results from processing all zones
        lake: Lake object with ground truth
        config: Configuration object
    """
    import time
    from pathlib import Path

    import pandas as pd

    start_time = time.time()
    logging.info("Creating detailed cell-level analysis...")

    # Pre-cache clean dataframes for faster lookup
    clean_data_cache = {}
    for table_id, table in lake.tables.items():
        if table.clean_dataframe is not None:
            clean_data_cache[table_id] = table.clean_dataframe.values

    batch_size = 10000
    cell_count = 0
    analysis_records = []

    for cluster_id, results in zone_results.items():
        zone = results["zone"]
        zone_name = results["zone_name"]

        # Process all cells in the zone
        processed_cells = 0
        for cell_coords, cell in zone.cells.items():
            # Include all cells (both with and without predictions)
            # to get complete picture of zone coverage

            table_id, column_idx, row_idx = cell_coords
            processed_cells += 1

            # Fast ground truth lookup
            ground_truth = "UNKNOWN"
            if table_id in clean_data_cache:
                try:
                    clean_array = clean_data_cache[table_id]
                    if (
                        row_idx < clean_array.shape[0]
                        and column_idx < clean_array.shape[1]
                    ):
                        ground_truth = str(clean_array[row_idx, column_idx])
                except (IndexError, ValueError):
                    pass
            elif hasattr(cell, "ground_truth") and cell.ground_truth:
                ground_truth = cell.ground_truth

            current_value = getattr(cell, "value", "UNKNOWN")

            # Extract prediction details (may not exist for all cells)
            if hasattr(cell, "predicted_corrections") and cell.predicted_corrections:
                selected_value = cell.predicted_corrections.get("candidate")
                prediction_source = cell.predicted_corrections.get("source", "classifier")
            else:
                selected_value = None
                prediction_source = None
            
            # Extract candidate count and score if available
            num_candidates = len(cell.candidates) if hasattr(cell, "candidates") and cell.candidates else 0
            selected_candidate_score = None
            if selected_value and hasattr(cell, "candidates") and selected_value in cell.candidates:
                # Get candidate from pool
                pool = CandidatePool.get_instance()
                pool_key = cell.candidates[selected_value]
                candidate_obj = pool.get_candidate(pool_key) if isinstance(pool_key, tuple) else pool_key
                if candidate_obj:
                    selected_candidate_score = getattr(candidate_obj, "score", None)

            # Determine correction status
            if selected_value:
                correction_status = (
                    "CORRECT_CORRECTION"
                    if selected_value == ground_truth
                    else "INCORRECT_CORRECTION"
                )
            elif current_value == ground_truth:
                correction_status = "NO_CORRECTION_NEEDED"
            else:
                correction_status = "MISSED_ERROR"

            # Determine if cell is an error
            is_error = (
                current_value != ground_truth if ground_truth != "UNKNOWN" else None
            )

            # Create record
            record = (
                table_id,
                column_idx,
                row_idx,
                zone_name,
                cluster_id,
                current_value,
                ground_truth,
                selected_value,
                correction_status,
                num_candidates,
                selected_candidate_score,
                prediction_source,
                is_error,
            )

            analysis_records.append(record)
            cell_count += 1

            if cell_count % batch_size == 0:
                elapsed = time.time() - start_time
                logging.info(
                    f"Processed {cell_count} cells with predictions in {elapsed:.1f}s ({cell_count / elapsed:.0f} cells/sec)"
                )

    # Create DataFrame from records
    column_names = [
        "table_id",
        "column_idx",
        "row_idx",
        "zone_name",
        "cluster_id",
        "current_value",
        "ground_truth",
        "predicted_value",
        "correction_status",
        "num_candidates",
        "predicted_score",
        "prediction_source",
        "is_error",
    ]

    logging.info(f"Creating DataFrame with {len(analysis_records)} sampled cells...")
    df = pd.DataFrame(analysis_records, columns=column_names)

    # Save CSV
    analysis_path = Path(config.directories.output_dir) / "cell_level_analysis.csv"
    logging.info(f"Saving analysis to {analysis_path}...")
    df.to_csv(analysis_path, index=False, chunksize=50000)

    # Compute summary statistics from the detailed analysis
    summary_stats = {
        "total_cells_analyzed": len(df),  # All cells in zones (not just with predictions)
        "total_error_cells": int(df["is_error"].sum()) if df["is_error"].notna().any() else 0,
        "corrections_applied": int(df["predicted_value"].notna().sum()),
        "correct_corrections": int(
            (df["correction_status"] == "CORRECT_CORRECTION").sum()
        ),
        "incorrect_corrections": int(
            (df["correction_status"] == "INCORRECT_CORRECTION").sum()
        ),
        "missed_errors": int((df["correction_status"] == "MISSED_ERROR").sum()),
        "no_correction_needed": int(
            (df["correction_status"] == "NO_CORRECTION_NEEDED").sum()
        ),
        "classifier_corrections": int(
            (df["prediction_source"] == "classifier").sum()
        ),
        "pattern_enforcement_corrections": int(
            df["prediction_source"]
            .isin(
                [
                    "pattern_enforcement",
                    "pattern_enforcement_correct",
                    "pattern_based_correct",
                ]
            )
            .sum()
        ),
        "manual_sample_corrections": int(
            (df["prediction_source"] == "manual").sum()
        ),
        "zones_analyzed": len(zone_results),  # Use actual zones processed, not unique in dataframe
    }

    # Save summary
    summary_path = Path(config.directories.output_dir) / "cell_analysis_summary.json"
    import json

    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    total_time = time.time() - start_time
    logging.info(
        f"Detailed cell-level analysis completed in {total_time:.1f}s ({len(df) / total_time:.0f} cells/sec)"
    )
    logging.info(f"Analysis saved to: {analysis_path}")
    logging.info(f"Summary saved to: {summary_path}")
    logging.info(
        f"Analyzed {len(df)} sampled cells across {summary_stats['zones_analyzed']} zones"
    )


def print_final_results(evaluation_results: Dict, execution_times: Dict) -> None:
    """
    Print final results to console for user visibility.

    Args:
        evaluation_results: Evaluation results from all zones
        execution_times: Execution time tracking
    """
    aggregate_stats = evaluation_results["aggregate_stats"]
    zone_evaluations = evaluation_results["zone_evaluations"]

    print("\n" + "=" * 60)
    print("MULTI-ZONE PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Zones processed: {aggregate_stats['zones_processed']}")
    print(f"Overall F1 Score: {aggregate_stats['overall_f1']:.3f}")
    print(f"Overall Precision: {aggregate_stats['overall_precision']:.3f}")
    print(f"Overall Recall: {aggregate_stats['overall_recall']:.3f}")
    total_runtime = execution_times.get("total_runtime", sum(execution_times.values()))
    print(f"Total execution time: {total_runtime:.2f}s")

    # Show execution time breakdown
    print("\nExecution time breakdown:")
    time_breakdown = [
        ("Lake initialization", execution_times.get("lake_initialization", 0)),
        ("Profile building", execution_times.get("profiles_initialization", 0)),
        ("Feature extraction", execution_times.get("feature_extraction", 0)),
        ("Clustering & sampling", execution_times.get("clustering_sampling", 0)),
        ("Zone creation", execution_times.get("zone_creation", 0)),
        ("Zone processing", execution_times.get("total_zone_processing", 0)),
        ("Evaluation", execution_times.get("evaluation", 0)),
    ]

    for phase_name, phase_time in time_breakdown:
        if phase_time > 0:
            percentage = (phase_time / total_runtime) * 100 if total_runtime > 0 else 0
            print(f"  {phase_name}: {phase_time:.2f}s ({percentage:.1f}%)")

    if zone_evaluations:
        print(f"\nProcessed {len(zone_evaluations)} zones")

        # Show top 3 performing zones
        zone_f1_scores = []
        for cluster_id, evaluation in zone_evaluations.items():
            zone_name = evaluation["zone_name"]
            f1_score = evaluation["zone_metrics"]["f1_score"]
            zone_f1_scores.append((zone_name, f1_score))

        zone_f1_scores.sort(key=lambda x: x[1], reverse=True)

        print("\nTop 3 performing zones:")
        for i, (zone_name, f1_score) in enumerate(zone_f1_scores[:3], 1):
            print(f"  {i}. {zone_name}: F1={f1_score:.3f}")

        if len(zone_f1_scores) > 3:
            print("\nWorst 3 performing zones:")
            for i, (zone_name, f1_score) in enumerate(zone_f1_scores[-3:], 1):
                print(f"  {i}. {zone_name}: F1={f1_score:.3f}")

    print("=" * 60)


def main() -> None:
    """Main pipeline execution function."""
    try:
        # Track total runtime
        total_start_time = time.time()

        logging.info("Starting Multi-Zone Error Correction Pipeline...")

        # Load configuration
        config = read_ecs_config()
        logging.info("Pipeline configuration loaded successfully")

        # Setup logging
        setup_logging(
            logs_dir=config.directories.logs_dir,
            log_level=config.experiment.log_level,
            log_file=f"{config.experiment.exp_name}.log",
        )
        logging.info("Logging setup completed")

        # Initialize memory monitor with 80% system memory limit
        memory_monitor = MemoryMonitor(memory_limit_percent=80.0)
        memory_monitor.check_memory("Startup")

        # Initialize execution time tracking
        execution_times = {}

        # === PHASE 1: LAKE INITIALIZATION ===
        logging.info("Phase 1: Initializing data lake...")
        t0 = time.time()

        lake = initialize_lake(
            config.directories.tables_dir,
            config.directories.dirty_files_name,
            config.directories.clean_files_name,
            max_workers=config.runtime.n_cores,
            cardinality_threshold=config.pruning.cardinality_threshold,
            output_dir=config.directories.output_dir,
        )

        execution_times["lake_initialization"] = time.time() - t0
        logging.info(
            f"Lake initialized in {execution_times['lake_initialization']:.2f}s"
        )
        
        # Check memory after lake initialization
        memory_monitor.check_and_enforce("After lake initialization")

        # === PHASE 2-5: ZONE CREATION (strategy-dependent) ===
        logging.info(f"Phases 2-5: Creating zones using {config.zoning.strategy} strategy...")

        if config.zoning.strategy == "rule_based":
            # Rule-based zoning: use predefined cardinality + pattern validity zones
            logging.info("Using RULE-BASED zone detection (skipping profiling and clustering)...")
            t0 = time.time()
            zones = initialize_zones_cell_wise(lake, config)
            execution_times["zone_creation"] = time.time() - t0
            logging.info(
                f"Rule-based zones created in {execution_times['zone_creation']:.2f}s"
            )
            
            # For rule-based, we need to sample cells from zones
            # Collect all error cells from zones
            dirty_cells = []
            sampled_cells = []
            labeling_budget = config.labeling.labeling_budget
            
            for zone_name, zone in (zones.items() if isinstance(zones, dict) else [(i, z) for i, z in enumerate(zones)]):
                zone_cells = list(zone.cells.values())
                dirty_cells.extend(zone_cells)
            
            # === SAMPLING ===
            # pattern_enforcement_mode affects budget allocation in rule-based zoning:
            # - check / always_accept: prioritize invalid zones with no-clean columns,
            #   then allocate remaining budget to valid-pattern zones.
            # - disabled: skip that prioritization and allocate full budget across all zones.
            # In all modes, within-zone sampling tries to maximize column coverage first.
            
            sampled_cells = []
            labeling_budget = config.labeling.labeling_budget
            remaining_budget = labeling_budget
            rng = np.random.RandomState(config.experiment.random_state)
            pe_mode = (
                getattr(getattr(config, "correction", None), "pattern_enforcement_mode", "check")
                .strip()
                .lower()
            )
            prioritize_invalid_no_clean = pe_mode in {"check", "always_accept"}
            
            from collections import defaultdict

            # ========== STEP 1-2: optional prioritization for invalid/no-clean columns ==========
            if prioritize_invalid_no_clean:
                invalid_zone_cols_no_clean = defaultdict(list)  # Maps (table_id, col_idx) -> list of cells
                
                for zone_name, zone in (zones.items() if isinstance(zones, dict) else [(i, z) for i, z in enumerate(zones)]):
                    if "invalid_pattern" not in zone_name:
                        continue
                    
                    zone_cells = list(zone.cells.values())
                    if not zone_cells:
                        continue
                    
                    # Check each cell's column
                    for cell in zone_cells:
                        column = lake.tables[cell.table_id].columns[cell.column_idx]
                        clean_values = column.get_unique_clean_values()
                        col_key = (cell.table_id, cell.column_idx)
                        
                        if len(clean_values) == 0:
                            invalid_zone_cols_no_clean[col_key].append(cell)
                
                labels_for_no_clean_cols = {}  # Maps col_key -> list of sampled cells
                
                if invalid_zone_cols_no_clean:
                    num_cols_no_clean = len(invalid_zone_cols_no_clean)
                    target_labels_per_col = 2
                    total_labels_needed = num_cols_no_clean * target_labels_per_col
                    
                    if total_labels_needed <= remaining_budget:
                        # We have enough budget: allocate 2 per column
                        for col_key, cells in invalid_zone_cols_no_clean.items():
                            samples_for_col = min(2, len(cells))
                            if samples_for_col > 0:
                                sampled_indices = rng.choice(len(cells), size=samples_for_col, replace=False)
                                sampled = [cells[i] for i in sampled_indices]
                                labels_for_no_clean_cols[col_key] = sampled
                                sampled_cells.extend(sampled)
                                remaining_budget -= samples_for_col
                        
                        logging.info(
                            f"Allocated 2 labels per column for {num_cols_no_clean} columns with no clean values "
                            f"(used {total_labels_needed} labels, remaining budget: {remaining_budget})"
                        )
                    else:
                        # Budget is tight: distribute proportionally
                        labels_per_col = max(1, remaining_budget // num_cols_no_clean)
                        extra_labels = remaining_budget % num_cols_no_clean
                        
                        for col_idx, (col_key, cells) in enumerate(invalid_zone_cols_no_clean.items()):
                            samples_for_col = labels_per_col
                            if col_idx < extra_labels:
                                samples_for_col += 1
                            
                            samples_for_col = min(samples_for_col, len(cells))
                            if samples_for_col > 0:
                                sampled_indices = rng.choice(len(cells), size=samples_for_col, replace=False)
                                sampled = [cells[i] for i in sampled_indices]
                                labels_for_no_clean_cols[col_key] = sampled
                                sampled_cells.extend(sampled)
                                remaining_budget -= samples_for_col
                        
                        logging.info(
                            f"Budget-constrained allocation for {num_cols_no_clean} columns with no clean values "
                            f"(allocated {len(sampled_cells)} labels, remaining budget: {remaining_budget})"
                        )
                    
                    # Add sampled cells to invalid zones
                    for zone_name, zone in (zones.items() if isinstance(zones, dict) else [(i, z) for i, z in enumerate(zones)]):
                        if "invalid_pattern" not in zone_name:
                            continue
                        
                        zone_sample = []
                        for cell in zone.cells.values():
                            col_key = (cell.table_id, cell.column_idx)
                            if col_key in labels_for_no_clean_cols:
                                if cell in labels_for_no_clean_cols[col_key]:
                                    zone_sample.append(cell)
                        
                        if zone_sample:
                            zone.samples = {(c.table_id, c.column_idx, c.row_idx): c for c in zone_sample}
            else:
                logging.info(
                    "Pattern enforcement mode is 'disabled': skipping invalid/no-clean "
                    "pre-allocation; distributing labeling budget across all zones."
                )
            
            # ========== STEP 3: Allocate remaining budget proportionally ==========
            candidate_zones = []
            for zone_name, zone in (zones.items() if isinstance(zones, dict) else [(i, z) for i, z in enumerate(zones)]):
                if prioritize_invalid_no_clean:
                    # Original behavior: remaining budget goes to valid-pattern zones.
                    if "_valid_pattern" not in zone_name:
                        continue
                # Disabled mode: include all zones (invalid + valid).
                zone_cells = list(zone.cells.values())
                if zone_cells:
                    candidate_zones.append((zone_name, zone, len(zone_cells)))
            
            total_candidate_cells = sum(cell_count for _, _, cell_count in candidate_zones)
            
            # Allocate budget proportionally to selected zones by size
            zones_budget_allocation = []
            for zone_name, zone, zone_cell_count in candidate_zones:
                if total_candidate_cells > 0 and remaining_budget > 0:
                    zone_allocated_budget = max(1, int(remaining_budget * zone_cell_count / total_candidate_cells))
                    zones_budget_allocation.append((zone_name, zone, zone_allocated_budget))
            
            # Verify total allocation
            total_allocated = sum(budget for _, _, budget in zones_budget_allocation)
            if total_allocated > remaining_budget:
                scale_factor = remaining_budget / total_allocated
                zones_budget_allocation = [
                    (name, zone, max(1, int(budget * scale_factor)))
                    for name, zone, budget in zones_budget_allocation
                ]
            
            # ========== STEP 4: Within each zone, ensure many columns get coverage ==========
            for zone_name, zone, zone_sample_size in zones_budget_allocation:
                if zone_sample_size == 0 or remaining_budget <= 0:
                    continue
                
                zone_cells = list(zone.cells.values())
                if not zone_cells:
                    continue
                
                # Group cells by column
                cells_by_col = defaultdict(list)
                for cell in zone_cells:
                    col_key = (cell.table_id, cell.column_idx)
                    cells_by_col[col_key].append(cell)
                
                num_columns = len(cells_by_col)
                zone_sample_size = min(zone_sample_size, remaining_budget)
                
                # Strategy: Ensure many columns get at least one sample, then randomly assign rest
                zone_sample = []
                
                if num_columns > 0:
                    # PHASE 1: Give at least 1 sample to as many columns as possible
                    cols_list = list(cells_by_col.keys())
                    # Sort by error count (most errors first) to prioritize columns with more errors
                    cols_list = sorted(cols_list, key=lambda col_key: len(cells_by_col[col_key]), reverse=True)
                    
                    samples_assigned = 0
                    col_sample_count = {}
                    
                    # First pass: assign 1 sample per column until we run out of budget or columns
                    for col_key in cols_list:
                        if samples_assigned >= zone_sample_size:
                            break
                        
                        col_cells = cells_by_col[col_key]
                        if col_cells:
                            cell = col_cells[rng.randint(len(col_cells))]
                            zone_sample.append(cell)
                            col_sample_count[col_key] = 1
                            samples_assigned += 1
                    
                    # PHASE 2: Randomly assign remaining budget to columns
                    if samples_assigned < zone_sample_size:
                        remaining_samples = zone_sample_size - samples_assigned
                        # Randomly pick cells from all zone cells for remaining samples
                        available_cells = [c for c in zone_cells if c not in zone_sample]
                        if available_cells and remaining_samples > 0:
                            remaining_samples = min(remaining_samples, len(available_cells))
                            indices = rng.choice(len(available_cells), size=remaining_samples, replace=False)
                            additional_cells = [available_cells[i] for i in indices]
                            zone_sample.extend(additional_cells)
                            samples_assigned += remaining_samples
                else:
                    indices = rng.choice(len(zone_cells), size=min(zone_sample_size, len(zone_cells)), replace=False)
                    zone_sample = [zone_cells[i] for i in indices]
                
                zone.samples = {(c.table_id, c.column_idx, c.row_idx): c for c in zone_sample}
                sampled_cells.extend(zone_sample)
                remaining_budget -= len(zone_sample)
                
                logging.info(
                    f"Zone {zone_name}: sampled {len(zone_sample)} cells from {num_columns} columns "
                    f"(strategy: ensure coverage then randomly assign remainder, remaining budget: {remaining_budget})"
                )
            
            stats = {"total_errors": len(dirty_cells), "sampled": len(sampled_cells)}

        elif config.zoning.strategy == "clustering_based":
            # Clustering-based zoning: profile, extract features, cluster, then create zones
            logging.info("Using CLUSTERING-BASED zone detection...")
            
            # === PHASE 2: PROFILE BUILDING ===
            logging.info("Phase 2: Building column profiles...")
            t0 = time.time()

            # Initialize TANE wrapper with TANE path
            tane_path = str(Path(__file__).parent / "tane")
            tane_wrapper = TANEWrapper(tane_repo_path=tane_path)

            # Determine profiles output directory
            profiles_outdir = getattr(config.directories, "profiles_dir", None)
            if not profiles_outdir:
                profiles_outdir = str(Path(config.directories.output_dir) / "profiles")

            profiles, profiles_path = initialize_profiles(
                lake=lake, output_dir=profiles_outdir
            )

            tane_wrapper.add_functional_dependencies_to_profiles(profiles, lake)
            lake.profiles = profiles  # Attach for convenience

            execution_times["profiles_initialization"] = time.time() - t0
            logging.info(
                f"Profiles initialized in {execution_times['profiles_initialization']:.2f}s "
                f"(saved to {profiles_path})"
            )

            # === PHASE 3: FEATURE EXTRACTION ===
            logging.info("Phase 3: Extracting unusualness features...")
            t0 = time.time()

            dirty_cells, features_matrix, feature_names = (
                extract_unusualness_features_for_lake(lake, profiles)
            )

            for i, cell in enumerate(dirty_cells):
                cell.unusualness_features = []
                for j, feature in enumerate(features_matrix[i]):
                    cell.unusualness_features.append(
                        {
                            "name": feature_names[j],
                            "value": features_matrix[i][j],
                        }
                    )

            execution_times["feature_extraction"] = time.time() - t0
            logging.info(
                f"Features extracted in {execution_times['feature_extraction']:.2f}s"
            )

            # === PHASE 4: CLUSTERING AND SAMPLING ===
            logging.info("Phase 4: Clustering and sampling...")
            t0 = time.time()

            sampled_cells, stats, cluster_labels = cluster_and_sample_with_labels(
                dirty_cells=dirty_cells,
                features_matrix=features_matrix,
                labeling_budget=config.labeling.labeling_budget,
                random_state=config.experiment.random_state,
                feature_names=feature_names,
                lake=lake,
                cluster_sampling_strategy=config.sampling.cluster_sampling_strategy,
            )

            execution_times["clustering_sampling"] = time.time() - t0
            logging.info(
                f"Clustering and sampling completed in {execution_times['clustering_sampling']:.2f}s"
            )

            # === PHASE 5: ZONE CREATION (CLUSTERING-BASED) ===
            logging.info("Phase 5: Creating zones from clusters...")
            t0 = time.time()

            zones_dict = create_zones_from_clusters(dirty_cells, cluster_labels, sampled_cells)
            zones = zones_dict

            execution_times["zone_creation"] = time.time() - t0
            logging.info(
                f"Zone creation completed in {execution_times['zone_creation']:.2f}s"
            )

        else:
            raise ValueError(f"Unknown zoning strategy: {config.zoning.strategy}")

        # Validate zones were created correctly
        # For rule-based, we don't pre-identify dirty cells, so we validate differently
        if config.zoning.strategy == "clustering_based":
            is_valid = validate_zones(zones, len(dirty_cells), len(sampled_cells))
            if not is_valid:
                raise RuntimeError("Zone validation failed - see logs for details")
        else:
            # For rule-based, just check that zones have some content
            total_zone_cells = sum(len(zone.cells) for zone in zones.values())
            total_zone_samples = sum(len(zone.samples) for zone in zones.values())
            logging.info(
                f"Rule-based zone validation: {len(zones)} zones with {total_zone_cells} total cells "
                f"and {total_zone_samples} samples"
            )

        # Print zone summary
        print_zone_summary(zones)

        # Save sampling results and zone info
        save_sampling_results(config, sampled_cells, stats, zones)

        # Save detailed zone metadata (commented out for now)
        # zone_metadata_path = os.path.join(
        #     config.directories.output_dir, "zone_metadata.json"
        # )
        # save_zone_metadata(zones, zone_metadata_path)

        # === PHASE 6-10: PROCESS ALL ZONES ===
        logging.info("Phase 6-10: Processing all zones...")
        t0 = time.time()

        zone_results, zone_execution_times = process_all_zones(
            zones, lake, sampled_cells, config, memory_monitor
        )

        # Add zone processing times to overall execution times
        for key, value in zone_execution_times.items():
            execution_times[key] = execution_times.get(key, 0) + value

        execution_times["total_zone_processing"] = time.time() - t0
        logging.info(
            f"All zones processed in {execution_times['total_zone_processing']:.2f}s"
        )

        # === PHASE 11: EVALUATION ===
        logging.info("Phase 11: Evaluating all zones...")
        t0 = time.time()

        evaluation_results = evaluate_all_zones(zone_results, lake)

        execution_times["evaluation"] = time.time() - t0
        logging.info(f"Evaluation completed in {execution_times['evaluation']:.2f}s")

        # === PHASE 12: RESULTS REPORTING ===
        logging.info("Phase 12: Generating comprehensive results...")

        # Aggregate all results
        final_results = aggregate_zone_results(zone_results, evaluation_results)

        # Save detailed results
        detailed_results_path = os.path.join(
            config.directories.output_dir, "evaluation_detailed"
        )

        # Save each zone's evaluation results
        for cluster_id, evaluation in evaluation_results["zone_evaluations"].items():
            zone_results_path = f"{detailed_results_path}_zone_{cluster_id}"
            save_detailed_results(evaluation, zone_results_path)

        # Save aggregated results
        aggregated_results_path = os.path.join(
            config.directories.output_dir, "aggregated_results.json"
        )

        try:
            import json

            def convert_numpy_types(obj):
                """Recursively convert numpy types to native Python types for JSON serialization"""
                if isinstance(obj, dict):
                    return {
                        key: convert_numpy_types(value) for key, value in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            with open(aggregated_results_path, "w") as f:
                # Convert all numpy types to native Python for JSON serialization
                safe_results = convert_numpy_types(
                    {
                        "aggregate_stats": final_results["aggregate_stats"],
                        "zone_performances": final_results["zone_performances"],
                        "execution_times": execution_times,
                    }
                )

                json.dump(safe_results, f, indent=2)

            logging.info(f"Aggregated results saved to: {aggregated_results_path}")
        except Exception as e:
            logging.warning(f"Failed to save aggregated results: {e}")

        # Calculate total runtime
        execution_times["total_runtime"] = time.time() - total_start_time

        # Cell-level analysis is now done per-zone during processing
        logging.info("Cell-level analysis completed during zone processing")

        # Print final results
        print_final_results(evaluation_results, execution_times)

        # Memory cleanup
        logging.info("Performing memory cleanup...")
        # del lake, zones, dirty_cells
        # print_memory_usage()

        logging.info(
            f"Multi-zone pipeline completed successfully in {execution_times['total_runtime']:.2f}s"
        )

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Setup early console logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
