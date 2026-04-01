"""
Enhanced evaluation functions for error correction pipeline.
Provides zone, column, and table-level metrics.
"""

import logging
from collections import defaultdict
from typing import Dict, List


def apply_predictions_to_zone(zone, prediction_results):
    """
    Apply predictions to zone cells

    Args:
        zone: Zone object containing cells
        prediction_results: Object with y_pred, y_decision_score, test_samples

    Returns:
        int: Number of predictions applied
    """
    # Pre-build O(1) cell lookup dictionary
    logging.debug(f"Building cell lookup for {len(zone.cells)} cells...")
    cell_lookup = {}
    for cell in zone.cells.values():
        if (
            hasattr(cell, "table_id")
            and hasattr(cell, "column_idx")
            and hasattr(cell, "row_idx")
        ):
            cell_coords = (cell.table_id, cell.column_idx, cell.row_idx)
            cell_lookup[cell_coords] = cell

    logging.debug(f"Cell lookup built with {len(cell_lookup)} entries")

    predictions_applied = 0
    
    # === PHASE 2: Classifier predictions ===
    if (
        not hasattr(prediction_results, "test_samples")
        or not prediction_results.test_samples
    ):
        logging.warning(f"No test samples found for zone {zone.name}")
        return predictions_applied

    if not hasattr(prediction_results, "y_pred") or not prediction_results.y_pred:
        logging.warning(f"No predictions found for zone {zone.name}")
        return predictions_applied

    # Validate data consistency - test_samples and y_pred must have same length
    if len(prediction_results.test_samples) != len(prediction_results.y_pred):
        logging.error(
            f"Data consistency error in zone {zone.name}: "
            f"{len(prediction_results.test_samples)} test_samples but {len(prediction_results.y_pred)} predictions"
        )
        return predictions_applied

    # Validate y_decision_score length if present
    if (
        hasattr(prediction_results, "y_decision_score")
        and prediction_results.y_decision_score
        and len(prediction_results.y_decision_score) != len(prediction_results.y_pred)
    ):
        logging.error(
            f"Data consistency error in zone {zone.name}: "
            f"{len(prediction_results.y_pred)} predictions but {len(prediction_results.y_decision_score)} decision scores"
        )
        return predictions_applied

    # Group predictions by cell coordinates
    cell_predictions = {}

    for i, (table_id, col_idx, row_idx, candidate_value) in enumerate(
        prediction_results.test_samples
    ):
        # Only consider positive predictions
        if prediction_results.y_pred[i] != 1:
            continue

        cell_coords = (table_id, col_idx, row_idx)
        
        if cell_coords not in cell_predictions:
            cell_predictions[cell_coords] = []

        # Get confidence score
        confidence = None  # default
        if hasattr(prediction_results, "y_decision_score") and prediction_results.y_decision_score:
            confidence = prediction_results.y_decision_score[i]

        cell_predictions[cell_coords].append(
            {"candidate": candidate_value, "confidence": confidence}
        )

    # Apply best prediction per cell using O(1) lookup
    lookup_misses = 0
    for cell_coords, predictions in cell_predictions.items():
        # O(1) cell lookup instead of O(n) linear search
        cell = cell_lookup.get(cell_coords)

        if not cell:
            lookup_misses += 1
            logging.debug(f"Could not find cell for coordinates {cell_coords}")
            continue

        # Skip cells that already have pattern enforcement corrections
        if (hasattr(cell, "predicted_corrections") and cell.predicted_corrections 
            and cell.predicted_corrections.get("source") == "pattern_enforcement_correct"):
            logging.debug(f"Skipping cell {cell_coords} - already has pattern enforcement correction")
            continue

        # Select best prediction (highest confidence)
        try:
            best_prediction = max(predictions, key=lambda x: x["confidence"])
        except (ValueError, KeyError) as e:
            logging.debug(f"Error selecting best prediction for {cell_coords}: {e}")
            continue

        # Apply the prediction with robust error handling
        try:
            cell.predicted_corrections = {
                "candidate": best_prediction["candidate"],
                "confidence": best_prediction["confidence"],
                "source": "classifier"
            }

            predictions_applied += 1

        except Exception as e:
            logging.error(f"Failed to apply prediction to cell {cell_coords}: {e}")
            continue

    if lookup_misses > 0:
        logging.warning(f"{lookup_misses} cells could not be found in lookup")

    logging.info(
        f"Applied {predictions_applied} predictions to zone {zone.name} via classifier"
    )
    return predictions_applied


def evaluate_zone(zone, lake=None):
    """
    Evaluation that provides zone, column, and table-level metrics.

    Args:
        zone: Zone object containing cells with predictions
        lake: Optional lake object for additional context

    Returns:
        dict: Comprehensive evaluation results with nested metrics
    """
    logging.info(f"Evaluation for zone: {zone.name}")

    # Initialize result structure
    results = {
        "zone_name": zone.name,
        "zone_metrics": {},
        "table_metrics": {},
        "column_metrics": {},
        "overall_summary": {},
    }

    # Group cells by table and column for hierarchical evaluation
    cells_by_table = defaultdict(list)
    cells_by_column = defaultdict(list)

    total_errors = 0
    total_corrected = 0
    total_correct_corrections = 0
    total_incorrect_corrections = 0
    total_pattern_enforcement_correct = 0
    total_manual_samples_correct = 0
    total_manual_samples_incorrect = 0

    # Pre-compute zone.samples coordinates for O(1) lookup
    samples_coords = set(zone.samples.keys()) if zone.samples else set()

    # === PASS 1: Count pattern-enforced cells from zone.samples ===
    # Pattern-enforced cells are moved to zone.samples during candidate generation
    samples_detail = {"pattern_enforced": 0, "manual_correct": 0, "manual_incorrect": 0}
    
    for cell_coords, cell in zone.samples.items():
        if not cell.is_error:
            continue

        total_errors += 1
        ground_truth = getattr(cell, "ground_truth", None)

        # Group by table and column
        table_key = cell.table_id
        column_key = (cell.table_id, cell.column_idx)
        cells_by_table[table_key].append(cell)
        cells_by_column[column_key].append(cell)

        # Check if this is pattern-enforced (has explicit source marker)
        has_predicted_corrections = hasattr(cell, "predicted_corrections") and cell.predicted_corrections
        source_value = cell.predicted_corrections.get("source") if has_predicted_corrections else None
        # Treat both pattern_enforcement_correct and pattern_based_correct as pattern enforcement
        is_pattern_enforced = source_value in ("pattern_enforcement_correct", "pattern_based_correct")

        # Process pattern-enforced cells
        if is_pattern_enforced:
            samples_detail["pattern_enforced"] += 1
            total_corrected += 1
            predicted_value = cell.predicted_corrections.get("candidate")
            if predicted_value == ground_truth:
                total_pattern_enforcement_correct += 1
                total_correct_corrections += 1
            else:
                total_incorrect_corrections += 1
        # Direct manual samples (in zone.samples but NOT pattern-enforced)
        else:
            total_corrected += 1
            # Direct manual samples are assumed correct (oracle-labeled)
            total_manual_samples_correct += 1
            total_correct_corrections += 1
            samples_detail["manual_correct"] += 1
    
    logging.debug(f"Zone {zone.name} PASS 1 samples: {samples_detail}")

    # === PASS 2: Count classifier predictions from zone.cells (excluding zone.samples) ===
    # Classifier predictions are in zone.cells but not in zone.samples
    for cell in zone.cells.values():
        if not cell.is_error:
            continue

        cell_coords = (cell.table_id, cell.column_idx, cell.row_idx)

        # Skip cells already processed from zone.samples (pattern-enforced and direct manual)
        if cell_coords in samples_coords:
            continue

        total_errors += 1

        # Group by table and column
        table_key = cell.table_id
        column_key = (cell.table_id, cell.column_idx)
        cells_by_table[table_key].append(cell)
        cells_by_column[column_key].append(cell)

        ground_truth = getattr(cell, "ground_truth", None)

        # Process classifier predictions
        if hasattr(cell, "predicted_corrections") and cell.predicted_corrections:
            total_corrected += 1
            is_correct = cell.predicted_corrections.get("candidate") == ground_truth
            if is_correct:
                total_correct_corrections += 1
            else:
                total_incorrect_corrections += 1

    # Compute zone-level metrics
    zone_precision = (
        total_correct_corrections / total_corrected if total_corrected > 0 else 0.0
    )
    zone_recall = total_correct_corrections / total_errors if total_errors > 0 else 0.0
    zone_f1 = (
        2 * zone_precision * zone_recall / (zone_precision + zone_recall)
        if (zone_precision + zone_recall) > 0
        else 0.0
    )

    results["zone_metrics"] = {
        "total_error_cells": total_errors,
        "cells_corrected": total_corrected,
        "correct_corrections": total_correct_corrections,
        "incorrect_corrections": total_incorrect_corrections,
        "pattern_enforcement_correct": total_pattern_enforcement_correct,
        "manual_samples_correct": total_manual_samples_correct,
        "manual_samples_incorrect": total_manual_samples_incorrect,
        "precision": zone_precision,
        "recall": zone_recall,
        "f1_score": zone_f1,
    }

    # Compute table-level metrics
    for table_id, table_cells in cells_by_table.items():
        table_metrics = _compute_metrics_for_cells(table_cells, zone.samples)
        table_metrics["table_id"] = table_id

        # Add table name if available
        if lake and table_id in lake.tables:
            table_metrics["table_name"] = lake.tables[table_id].table_name

        results["table_metrics"][table_id] = table_metrics

    # Compute column-level metrics
    for column_key, column_cells in cells_by_column.items():
        table_id, column_idx = column_key
        column_metrics = _compute_metrics_for_cells(column_cells, zone.samples)
        column_metrics["table_id"] = table_id
        column_metrics["column_idx"] = column_idx

        # Add column name if available
        if (
            lake
            and table_id in lake.tables
            and column_idx in lake.tables[table_id].columns
        ):
            column = lake.tables[table_id].columns[column_idx]
            column_metrics["column_name"] = getattr(
                column, "col_name", f"col_{column_idx}"
            )

        results["column_metrics"][column_key] = column_metrics

    # Compute overall summary statistics
    results["overall_summary"] = _compute_summary_statistics(results)

    # Log summary
    logging.info(f"Zone {zone.name} evaluation completed:")
    logging.info(f"  Tables processed: {len(results['table_metrics'])}")
    logging.info(f"  Columns processed: {len(results['column_metrics'])}")
    logging.info(f"  Overall F1: {zone_f1:.3f}")

    return results


def _compute_metrics_for_cells(cells: List, zone_samples: Dict) -> Dict:
    """
    Compute precision, recall, F1 for a list of cells.

    Args:
        cells: List of cells to evaluate

    Returns:
        Dict with metrics
    """
    total_errors = len([cell for cell in cells if cell.is_error])
    total_corrected = 0
    correct_corrections = 0
    incorrect_corrections = 0
    pattern_enforcement_correct = 0
    manual_samples_correct = 0
    manual_samples_incorrect = 0

    for cell in cells:
        if not cell.is_error:
            continue

        cell_coords = (cell.table_id, cell.column_idx, cell.row_idx)
        ground_truth = getattr(cell, "ground_truth", None)
        
        # Check if this is a pattern-enforced cell
        is_pattern_enforced = (
            hasattr(cell, "predicted_corrections") 
            and cell.predicted_corrections
            and cell.predicted_corrections.get("source") == "pattern_enforcement_correct"
        )
        
        # Check if this is a manually labeled sample
        is_manual_sample = cell_coords in zone_samples and not is_pattern_enforced
        
        # Process pattern-enforced cells
        if is_pattern_enforced:
            predicted_value = cell.predicted_corrections.get("candidate")
            if predicted_value == ground_truth:
                pattern_enforcement_correct += 1
                correct_corrections += 1
                total_corrected += 1
        # Process manually labeled samples (that aren't pattern-enforced)
        elif is_manual_sample:
            total_corrected += 1
                # Samples are assumed correct (they were labeled by oracle)
            manual_samples_correct += 1
            correct_corrections += 1
        # Process classifier predictions
        elif hasattr(cell, "predicted_corrections") and cell.predicted_corrections:
            total_corrected += 1
            is_correct = cell.predicted_corrections.get("candidate") == ground_truth
            if is_correct:
                correct_corrections += 1
            else:
                incorrect_corrections += 1

    # Calculate metrics
    precision = correct_corrections / total_corrected if total_corrected > 0 else 0.0
    recall = correct_corrections / total_errors if total_errors > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "total_error_cells": total_errors,
        "cells_corrected": total_corrected,
        "correct_corrections": correct_corrections,
        "incorrect_corrections": incorrect_corrections,
        "pattern_enforcement_correct": pattern_enforcement_correct,
        "manual_samples_correct": manual_samples_correct,
        "manual_samples_incorrect": manual_samples_incorrect,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def _compute_summary_statistics(results: Dict) -> Dict:
    """
    Compute summary statistics across tables and columns.

    Args:
        results: Results dictionary with table and column metrics

    Returns:
        Dict with summary statistics
    """
    summary = {}

    # Table-level summaries
    if results["table_metrics"]:
        table_f1_scores = [
            metrics["f1_score"] for metrics in results["table_metrics"].values()
        ]
        table_precisions = [
            metrics["precision"] for metrics in results["table_metrics"].values()
        ]
        table_recalls = [
            metrics["recall"] for metrics in results["table_metrics"].values()
        ]

        summary.update(
            {
                "num_tables": len(results["table_metrics"]),
                "avg_table_f1": sum(table_f1_scores) / len(table_f1_scores),
                "avg_table_precision": sum(table_precisions) / len(table_precisions),
                "avg_table_recall": sum(table_recalls) / len(table_recalls),
                "min_table_f1": min(table_f1_scores),
                "max_table_f1": max(table_f1_scores),
            }
        )

    # Column-level summaries
    if results["column_metrics"]:
        column_f1_scores = [
            metrics["f1_score"] for metrics in results["column_metrics"].values()
        ]
        column_precisions = [
            metrics["precision"] for metrics in results["column_metrics"].values()
        ]
        column_recalls = [
            metrics["recall"] for metrics in results["column_metrics"].values()
        ]

        summary.update(
            {
                "num_columns": len(results["column_metrics"]),
                "avg_column_f1": sum(column_f1_scores) / len(column_f1_scores),
                "avg_column_precision": sum(column_precisions) / len(column_precisions),
                "avg_column_recall": sum(column_recalls) / len(column_recalls),
                "min_column_f1": min(column_f1_scores),
                "max_column_f1": max(column_f1_scores),
            }
        )

    return summary


def print_detailed_results(results: Dict, top_k: int = 5):
    """
    Print detailed evaluation results in a readable format.

    Args:
        results: Results dictionary from evaluate_zone_enhanced
        top_k: Number of top/bottom performers to show
    """
    print("\n" + "=" * 80)
    print("DETAILED EVALUATION RESULTS")
    print("=" * 80)

    # Zone-level summary
    zone_metrics = results["zone_metrics"]
    print(f"\nZONE: {results['zone_name']}")
    print(f"  Total Errors: {zone_metrics['total_error_cells']}")
    print(f"  Corrected: {zone_metrics['cells_corrected']}")
    print(f"  Precision: {zone_metrics['precision']:.3f}")
    print(f"  Recall: {zone_metrics['recall']:.3f}")
    print(f"  F1 Score: {zone_metrics['f1_score']:.3f}")

    # Overall summary
    if results["overall_summary"]:
        summary = results["overall_summary"]
        print("\nOVERALL SUMMARY:")
        if "num_tables" in summary:
            print(f"  Tables: {summary['num_tables']}")
            print(f"  Avg Table F1: {summary['avg_table_f1']:.3f}")
            print(
                f"  Table F1 Range: {summary['min_table_f1']:.3f} - {summary['max_table_f1']:.3f}"
            )

        if "num_columns" in summary:
            print(f"  Columns: {summary['num_columns']}")
            print(f"  Avg Column F1: {summary['avg_column_f1']:.3f}")
            print(
                f"  Column F1 Range: {summary['min_column_f1']:.3f} - {summary['max_column_f1']:.3f}"
            )

    # Top performing tables
    if results["table_metrics"]:
        table_items = list(results["table_metrics"].items())
        table_items.sort(key=lambda x: x[1]["f1_score"], reverse=True)

        print(f"\nTOP {top_k} PERFORMING TABLES:")
        for i, (table_id, metrics) in enumerate(table_items[:top_k]):
            table_name = metrics.get("table_name", table_id)
            print(f"  {i + 1}. {table_name}")
            print(
                f"     F1: {metrics['f1_score']:.3f}, "
                f"Precision: {metrics['precision']:.3f}, "
                f"Recall: {metrics['recall']:.3f}"
            )
            print(
                f"     Errors: {metrics['total_error_cells']}, "
                f"Corrected: {metrics['cells_corrected']}"
            )

        if len(table_items) > top_k:
            print(f"\nBOTTOM {top_k} PERFORMING TABLES:")
            for i, (table_id, metrics) in enumerate(table_items[-top_k:]):
                table_name = metrics.get("table_name", table_id)
                print(f"  {i + 1}. {table_name}")
                print(
                    f"     F1: {metrics['f1_score']:.3f}, "
                    f"Precision: {metrics['precision']:.3f}, "
                    f"Recall: {metrics['recall']:.3f}"
                )
                print(
                    f"     Errors: {metrics['total_error_cells']}, "
                    f"Corrected: {metrics['cells_corrected']}"
                )

    # Top performing columns
    if results["column_metrics"]:
        column_items = list(results["column_metrics"].items())
        column_items.sort(key=lambda x: x[1]["f1_score"], reverse=True)

        print(f"\nTOP {top_k} PERFORMING COLUMNS:")
        for i, (column_key, metrics) in enumerate(column_items[:top_k]):
            table_id, column_idx = column_key
            column_name = metrics.get("column_name", f"col_{column_idx}")
            table_name = metrics.get("table_name", table_id)
            print(f"  {i + 1}. {table_name}.{column_name}")
            print(
                f"     F1: {metrics['f1_score']:.3f}, "
                f"Precision: {metrics['precision']:.3f}, "
                f"Recall: {metrics['recall']:.3f}"
            )
            print(
                f"     Errors: {metrics['total_error_cells']}, "
                f"Corrected: {metrics['cells_corrected']}"
            )

    print("=" * 80)


def save_detailed_results(results: Dict, output_path: str):
    """
    Save detailed results to files for further analysis.

    Args:
        results: Results dictionary from evaluate_zone_enhanced
        output_path: Base path for saving results
    """
    import json
    from pathlib import Path

    import pandas as pd

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to JSON-serializable format
    json_serializable_results = _convert_results_for_json(results)

    # Save complete results as JSON
    json_path = output_dir / f"{Path(output_path).stem}_detailed_results.json"
    with open(json_path, "w") as f:
        json.dump(json_serializable_results, f, indent=2, default=str)

    # Save table metrics as CSV
    if results["table_metrics"]:
        table_df = pd.DataFrame.from_dict(results["table_metrics"], orient="index")
        table_df.index.name = "table_id"
        table_csv_path = output_dir / f"{Path(output_path).stem}_table_metrics.csv"
        table_df.to_csv(table_csv_path, index=False)
        logging.info(f"Table metrics saved to: {table_csv_path}")

    # Save column metrics as CSV
    if results["column_metrics"]:
        column_data = []
        for (table_id, column_idx), metrics in results["column_metrics"].items():
            row = {"table_id": table_id, "column_idx": column_idx}
            row.update(metrics)
            column_data.append(row)

        column_df = pd.DataFrame(column_data)
        column_csv_path = output_dir / f"{Path(output_path).stem}_column_metrics.csv"
        column_df.to_csv(column_csv_path, index=False)
        logging.info(f"Column metrics saved to: {column_csv_path}")

    logging.info(f"Detailed results saved to: {json_path}")


def _convert_results_for_json(results: Dict) -> Dict:
    """
    Convert results dictionary to JSON-serializable format by handling tuple keys and numpy types.

    Args:
        results: Original results dictionary

    Returns:
        JSON-serializable results dictionary
    """
    import numpy as np

    def convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
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

    json_results = {
        "zone_name": results["zone_name"],
        "zone_metrics": convert_numpy_types(results["zone_metrics"]),
        "overall_summary": convert_numpy_types(results["overall_summary"]),
        "table_metrics": {},
        "column_metrics": {},
    }

    # Convert table metrics (keys are already strings)
    json_results["table_metrics"] = convert_numpy_types(results["table_metrics"])

    # Convert column metrics (keys are tuples, need to convert to strings)
    for (table_id, column_idx), metrics in results["column_metrics"].items():
        column_key = f"{table_id}_col_{column_idx}"
        # Add table_id and column_idx to the metrics for reference
        column_metrics = convert_numpy_types(metrics.copy())
        column_metrics["table_id"] = table_id
        column_metrics["column_idx"] = column_idx
        json_results["column_metrics"][column_key] = column_metrics

    return json_results


class SimplePipelineMetrics:
    """Simple pipeline metrics container for backward compatibility"""

    def __init__(self, eval_results, exec_times):
        self.overall_f1 = eval_results["f1_score"]
        self.overall_precision = eval_results["precision"]
        self.overall_recall = eval_results["recall"]
        self.total_error_cells = eval_results["total_error_cells"]
        self.total_corrections = eval_results["cells_corrected"]
        self.total_correct_corrections = eval_results["correct_corrections"]
        self.total_time = sum(exec_times.values())


class SimpleCorrectionMetrics:
    """Simple correction metrics container for backward compatibility"""

    def __init__(self, eval_results):
        self.zone_name = eval_results["zone_name"]
        self.total_error_cells = eval_results["total_error_cells"]
        self.cells_corrected = eval_results["cells_corrected"]
        self.cells_correctly_corrected = eval_results["correct_corrections"]
        self.cells_incorrectly_corrected = eval_results["incorrect_corrections"]
        self.correction_precision = eval_results["precision"]
        self.correction_recall = eval_results["recall"]
        self.correction_f1 = eval_results["f1_score"]


def create_simple_metrics(evaluation_results, execution_times):
    """
    Create simple metrics objects from evaluation results for backward compatibility.
    """
    # Handle both old and new result formats
    if "zone_metrics" in evaluation_results:
        zone_results = evaluation_results["zone_metrics"]
        zone_results["zone_name"] = evaluation_results["zone_name"]
    else:
        zone_results = evaluation_results

    pipeline_metrics = SimplePipelineMetrics(zone_results, execution_times)
    correction_metrics = SimpleCorrectionMetrics(zone_results)

    return pipeline_metrics, correction_metrics
