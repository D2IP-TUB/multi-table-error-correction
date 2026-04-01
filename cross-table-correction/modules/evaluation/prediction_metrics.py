import logging
from dataclasses import dataclass
from typing import Dict

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from core.zone import Zone


@dataclass
class PredictionMetrics:
    """Metrics for correction prediction evaluation"""

    zone_name: str
    total_predictions: int
    correct_predictions: int
    incorrect_predictions: int
    precision: float
    recall: float
    f1_score: float


def evaluate_predictions(
    zones_dict: Dict[str, Zone], prediction_results: Dict
) -> Dict[str, PredictionMetrics]:
    """Evaluate model prediction performance"""

    logging.info("Evaluating prediction performance...")

    zone_metrics = {}

    for zone_name, zone in zones_dict.items():
        if zone_name not in prediction_results:
            continue

        logging.info(f"Evaluating predictions for zone: {zone_name}")

        results = prediction_results[zone_name]
        y_pred = results.y_pred
        test_samples = results.test_samples
        y_true = results.y_test if hasattr(results, "y_test") else []

        if not y_pred or not test_samples:
            continue

        # Calculate metrics
        if len(y_true) > 0 and len(y_pred) == len(y_true):
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            correct_predictions = sum(
                1
                for i in range(len(y_pred))
                if y_pred[i] == y_true[i] and y_true[i] == 1
            )
            incorrect_predictions = sum(
                1 for i in range(len(y_pred)) if y_pred[i] != y_true[i]
            )

        else:
            precision = recall = f1 = 0.0
            correct_predictions = incorrect_predictions = 0

        zone_metrics[zone_name] = PredictionMetrics(
            zone_name=zone_name,
            total_predictions=len(y_pred),
            correct_predictions=correct_predictions,
            incorrect_predictions=incorrect_predictions,
            precision=precision,
            recall=recall,
            f1_score=f1,
        )

        logging.info(f"Zone {zone_name}: Prediction F1={f1:.3f}")

    return zone_metrics


def evaluate_predictions_zone(
    zone: Zone, prediction_results: Dict
) -> PredictionMetrics:
    """Evaluate predictions for a single zone"""

    logging.info(f"Evaluating predictions for zone: {zone.name}")

    if zone.name not in prediction_results:
        logging.warning(f"No predictions found for zone: {zone.name}")
        return PredictionMetrics(
            zone_name=zone.name,
            total_predictions=0,
            correct_predictions=0,
            incorrect_predictions=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
        )

    results = prediction_results[zone.name]
    y_pred = results.y_pred
    y_true = results.y_test if hasattr(results, "y_test") else []

    if not y_pred or not y_true:
        logging.warning(f"No predictions or true labels for zone: {zone.name}")
        return PredictionMetrics(
            zone_name=zone.name,
            total_predictions=0,
            correct_predictions=0,
            incorrect_predictions=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
        )

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    correct_predictions = sum(
        1 for i in range(len(y_pred)) if y_pred[i] == y_true[i] and y_true[i] == 1
    )
    incorrect_predictions = sum(1 for i in range(len(y_pred)) if y_pred[i] != y_true[i])

    return PredictionMetrics(
        zone_name=zone.name,
        total_predictions=len(y_pred),
        correct_predictions=correct_predictions,
        incorrect_predictions=incorrect_predictions,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )
