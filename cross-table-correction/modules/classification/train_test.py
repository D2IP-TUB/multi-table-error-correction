"""
Training and testing module for error correction with configurable negative pruning logic.
STREAMING PREDICTION VERSION - Extract and predict in one pass without storing test data
"""

import logging
import os
import pickle
import time
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import AdaBoostClassifier

from core.zone import Zone
from core.candidate_pool import CandidatePool
from modules.classification.zone_predictions_results import ZonePredictionResults
from modules.classification.zone_training_data import ZoneTrainingData
from modules.feature_extraction.extract_features import get_feature_names


def extract_training_data_from_zone(zone: Zone, config: Any = None, include_pattern_features: bool = True) -> ZoneTrainingData:
    """
    Extract training data from zone samples with configurable negative pruning.

    Args:
        zone: Zone object containing cells and samples
        config: Configuration object with negative_pruning_enabled flag
        include_pattern_features: Whether to include pattern-based reliability feature (True for clustering-based zoning, False for rule-based)

    Pruning strategies:
    - If config.training.negative_pruning_enabled = True: Use smart negative pruning
    - If config.training.negative_pruning_enabled = False: Include all negative samples
    """
    logging.info(f"Extracting training data for zone: {zone.name}")

    # Check if negative pruning is enabled
    use_negative_pruning = False
    if (
        config
        and hasattr(config, "training")
        and hasattr(config.training, "negative_pruning_enabled")
    ):
        use_negative_pruning = config.training.negative_pruning_enabled

    logging.info(
        f"Zone {zone.name}: Negative pruning {'ENABLED' if use_negative_pruning else 'DISABLED'}"
    )

    # Get sample cell coordinates for O(1) lookup
    sample_coords = set()
    if hasattr(zone, "samples") and zone.samples:
        sample_coords = {cell.coordinates for cell in zone.samples.values()}

    feature_vectors = []
    labels = []
    samples = []
    n_positive = n_negative = n_from_samples = 0

    # Get candidate pool
    pool = CandidatePool.get_instance()

    # Feature ablation config
    disabled_groups = (
        getattr(config.training, "disabled_feature_groups", [])
        if config and hasattr(config, "training")
        else []
    )

    # Track which vicinity models are used for positive samples per column (only if pruning enabled)
    selected_models_in_column = {} if use_negative_pruning else None

    # PASS 1: Extract positive samples and track vicinity models (if pruning enabled)
    # Pattern-enforced cells are now added to zone.samples in correction_pipeline, so they flow through normal processing
    for sample in zone.samples.values():
        ground_truth = sample.ground_truth
        for candidate_value, pool_key in sample.candidates.items():
            try:
                candidate = pool.get_candidate(pool_key)
                if candidate is None:
                    logging.warning(f"Candidate not found in pool: {pool_key}")
                    continue
                    
                if candidate_value == ground_truth:
                    # This is a positive sample
                    features = candidate.get_features_array(include_pattern_features, disabled_groups)
                    feature_vectors.append(features)
                    labels.append(1)
                    samples.append(
                        (sample.table_id, sample.column_idx, sample.row_idx, candidate_value)
                    )

                    # Track vicinity models used for this positive sample (only if pruning enabled)
                    if (
                        use_negative_pruning
                        and hasattr(candidate, "candidates_source_model")
                        and "vicinity_context_columns"
                        in candidate.candidates_source_model
                    ):
                        col_key = (sample.table_id, sample.column_idx)
                        if col_key not in selected_models_in_column:
                            selected_models_in_column[col_key] = []
                        selected_models_in_column[col_key].extend(
                            candidate.candidates_source_model[
                                "vicinity_context_columns"
                            ]
                        )

                    n_positive += 1
                    n_from_samples += 1

            except Exception as e:
                logging.debug(
                    f"Failed to extract positive features for {candidate_value}: {e}"
                )
                continue

    # PASS 2: Extract negative samples with configurable pruning
    n_pruned = 0
    for cell in zone.cells.values():
        if not _is_valid_sample_cell(cell, sample_coords):
            continue

        ground_truth = cell.ground_truth

        for candidate_value, pool_key in cell.candidates.items():
            try:
                candidate = pool.get_candidate(pool_key)
                if candidate is None:
                    logging.warning(f"Candidate not found in pool: {pool_key}")
                    continue
                    
                if candidate_value != ground_truth:
                    # This is a negative sample - apply pruning logic based on config
                    should_include = True

                    if use_negative_pruning:
                        should_include = _should_include_negative_sample(
                            cell, candidate, selected_models_in_column
                        )
                        if not should_include:
                            n_pruned += 1

                    if should_include:
                        features = candidate.get_features_array(include_pattern_features, disabled_groups)
                        feature_vectors.append(features)
                        labels.append(0)
                        samples.append(
                            (
                                cell.table_id,
                                cell.column_idx,
                                cell.row_idx,
                                candidate_value,
                            )
                        )
                        n_negative += 1
                        n_from_samples += 1

            except Exception as e:
                logging.debug(
                    f"Failed to extract negative features for {candidate_value}: {e}"
                )
                continue

    # Convert to training format
    X_train = [f.tolist() for f in feature_vectors]
    y_train = labels
    train_samples = samples

    # Get feature names (respects ablation config)
    feature_names = get_feature_names(include_pattern_features, disabled_groups)
    logging.info(
        f"Zone {zone.name}: classifier feature ablation — dim={len(feature_names)}, "
        f"disabled_groups={list(disabled_groups) if disabled_groups else []}, "
        f"active_features={feature_names}"
    )

    if use_negative_pruning and n_pruned > 0:
        logging.info(
            f"Zone {zone.name}: Pruned {n_pruned} negative samples based on vicinity models"
        )

    logging.info(
        f"Zone {zone.name}: Extracted {len(X_train)} training samples "
        f"({n_positive} positive, {n_negative} negative)"
    )

    return ZoneTrainingData(
        zone_name=zone.name,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        train_samples=train_samples,
        n_positive=n_positive,
        n_negative=n_negative,
        n_from_samples=n_from_samples,
    )


def _is_valid_sample_cell(cell, sample_coords) -> bool:
    """Check if cell is valid for training data extraction."""
    if cell.coordinates in sample_coords:
        print(f"Cell {cell.coordinates} is a sample cell.")
    return (
        cell.is_error
        and cell.coordinates in sample_coords
        and hasattr(cell, "candidates")
        and cell.candidates
        and hasattr(cell, "ground_truth")
        and cell.ground_truth
    )


def _should_include_negative_sample(cell, candidate, selected_models_in_column) -> bool:
    """
    Determine if a negative sample should be included based on vicinity model pruning logic.

    Logic:
    1. If no vicinity models were used for this column, only include value_based candidates
    2. If vicinity models were used, only include candidates that use the same vicinity models
    3. If selected_models_in_column is None, this function shouldn't be called
    """
    if selected_models_in_column is None:
        return True  # Safety fallback - include all if pruning data not available

    col_key = (cell.table_id, cell.column_idx)
    selected_vicinity_models = selected_models_in_column.get(col_key)

    # If no vicinity models were selected for this column
    if selected_vicinity_models is None:
        # Only include value_based candidates
        return (
            hasattr(candidate, "candidates_source_model")
            and "value_based" in candidate.candidates_source_model
        )

    # If vicinity models were used for positive samples
    if (
        hasattr(candidate, "candidates_source_model")
        and "vicinity_context_columns" in candidate.candidates_source_model
    ):
        # Check if this candidate uses any of the selected vicinity models
        candidate_context_cols = candidate.candidates_source_model[
            "vicinity_context_columns"
        ]
        return any(
            context_col in selected_vicinity_models
            for context_col in candidate_context_cols
        )

    # If candidate doesn't have vicinity context, exclude it when pruning is enabled
    return False


def extract_training_data_from_zone_no_pruning(
    zone: Zone, include_pattern_features: bool = True, config: Any = None,
) -> ZoneTrainingData:
    """
    Extract training data without any negative pruning.

    Args:
        zone: Zone object containing cells and samples
        include_pattern_features: Whether to include pattern-based reliability feature
        config: Pipeline config (forwarded so feature ablation settings are respected)
    """
    disabled = (
        getattr(config.training, "disabled_feature_groups", [])
        if config and hasattr(config, "training")
        else []
    )

    class NoPruningConfig:
        class Training:
            negative_pruning_enabled = False
            disabled_feature_groups = disabled

        training = Training()

    return extract_training_data_from_zone(zone, NoPruningConfig(), include_pattern_features)


def extract_training_data_from_zone_with_pruning(
    zone: Zone, include_pattern_features: bool = True, config: Any = None,
) -> ZoneTrainingData:
    """
    Extract training data with negative pruning enabled.

    Args:
        zone: Zone object containing cells and samples
        include_pattern_features: Whether to include pattern-based reliability feature
        config: Pipeline config (forwarded so feature ablation settings are respected)
    """
    disabled = (
        getattr(config.training, "disabled_feature_groups", [])
        if config and hasattr(config, "training")
        else []
    )

    class PruningConfig:
        class Training:
            negative_pruning_enabled = True
            disabled_feature_groups = disabled

        training = Training()

    return extract_training_data_from_zone(zone, PruningConfig(), include_pattern_features)


def _is_valid_test_cell(cell, sample_coords) -> bool:
    """Check if cell is valid for test data extraction."""
    return (
        cell.is_error
        and cell.coordinates not in sample_coords
        and hasattr(cell, "candidates")
        and cell.candidates
    )


def _process_prediction_batch(
    classifier: AdaBoostClassifier,
    batch_features: list,
    batch_labels: list,
    batch_samples: list,
    all_y_pred: list,
    all_y_decision_score: list,
    all_y_test: list,
    all_test_samples: list,
) -> None:
    """
    Process a batch of predictions and append results to accumulator lists.
    This helps avoid memory overload by processing predictions in batches.
    
    Args:
        classifier: The trained AdaBoost classifier
        batch_features: List of feature arrays for this batch
        batch_labels: List of labels for this batch
        batch_samples: List of (table_id, col_idx, row_idx, value) tuples
        all_y_pred: Accumulator list for predictions
        all_y_decision_score: Accumulator list for decision scores
        all_y_test: Accumulator list for test labels
        all_test_samples: Accumulator list for test samples
    """
    if not batch_features:
        return
    
    try:
        # Convert to numpy array
        X_test = np.vstack(batch_features)
        
        # Predict using decision function
        decision_scores = classifier.decision_function(X_test)
        predictions = (decision_scores > 0).astype(int)
        
        # Append to accumulators
        all_y_pred.extend(predictions.tolist())
        all_y_decision_score.extend(decision_scores.tolist())
        all_y_test.extend(batch_labels)
        all_test_samples.extend(batch_samples)
        
        # Clear references to help garbage collection
        del X_test
        del decision_scores
        del predictions
        
    except Exception as e:
        logging.error(f"Failed to process prediction batch: {e}")
        raise


def extract_test_data_from_zone(
    classifier: AdaBoostClassifier,
    zone: Zone,
    zone_name: str,
    config: Any,
    include_pattern_features: bool = True,
) -> ZonePredictionResults:
    """
    Extract test data and predict with batch processing to avoid memory overload.
    
    Args:
        classifier: The trained AdaBoost classifier
        zone: Zone object containing cells and candidates
        zone_name: Name of the zone being processed
        config: Configuration object
        include_pattern_features: Whether to include pattern-based features (must match training data)
    """
    logging.info(f"Extracting test data and predicting for zone: {zone_name}")

    if not classifier:
        logging.warning(f"No classifier provided for zone {zone_name}")
        return _empty_prediction_results(zone_name)

    # Get sample coordinates to exclude from test set
    sample_coords = set()
    if hasattr(zone, "samples") and zone.samples:
        sample_coords = {cell.coordinates for cell in zone.samples.values()}

    # Pre-filter valid test cells
    valid_test_cells = [
        cell for cell in zone.cells.values() if _is_valid_test_cell(cell, sample_coords)
    ]

    if not valid_test_cells:
        logging.info(f"Zone {zone_name}: No valid test cells found")
        return _empty_prediction_results(zone_name)

    disabled_groups = (
        getattr(config.training, "disabled_feature_groups", [])
        if config and hasattr(config, "training")
        else []
    )

    all_y_pred = []
    all_y_decision_score = []
    all_y_test = []
    all_test_samples = []

    cells_with_correct = 0
    total_test_cells = len(valid_test_cells)
    processed_cells = 0
    total_candidates = 0

    start_time = time.time()
    batch_size = getattr(config.prediction, "batch_size", 10000) if hasattr(config, "prediction") else 10000

    try:
        # Process cells and collect candidates in batches
        batch_features = []
        batch_labels = []
        batch_samples = []

        for cell in valid_test_cells:
            processed_cells += 1

            if processed_cells % 5000 == 0:
                logging.debug(
                    f"Processing cell {processed_cells}/{total_test_cells} for zone {zone_name}"
                )

            ground_truth = getattr(cell, "ground_truth", None)
            has_correct = False

            # Process all candidates for this cell
            pool = CandidatePool.get_instance()
            for candidate_value, pool_key in cell.candidates.items():
                try:
                    # Retrieve candidate from pool
                    candidate = pool.get_candidate(pool_key) if isinstance(pool_key, tuple) else pool_key
                    if candidate is None:
                        continue
                    
                    features = candidate.get_features_array(include_pattern_features, disabled_groups)
                    label = (
                        1 if (ground_truth and candidate_value == ground_truth) else 0
                    )

                    # Add to batch
                    batch_features.append(features)
                    batch_labels.append(label)
                    batch_samples.append(
                        (cell.table_id, cell.column_idx, cell.row_idx, candidate_value)
                    )
                    total_candidates += 1

                    if label == 1:
                        has_correct = True

                    # Process batch when it reaches batch_size
                    if len(batch_features) >= batch_size:
                        _process_prediction_batch(
                            classifier, batch_features, batch_labels, batch_samples,
                            all_y_pred, all_y_decision_score, all_y_test, all_test_samples,
                        )
                        batch_features = []
                        batch_labels = []
                        batch_samples = []

                except Exception as e:
                    logging.debug(f"Failed to process candidate {candidate_value}: {e}")
                    continue

            if has_correct:
                cells_with_correct += 1

        # Process remaining batch
        if batch_features:
            logging.info(f"Making predictions for {len(batch_features)} candidates (final batch)...")
            _process_prediction_batch(
                classifier, batch_features, batch_labels, batch_samples,
                all_y_pred, all_y_decision_score, all_y_test, all_test_samples,
            )

        prediction_time = time.time() - start_time

        logging.info(
            f"Zone {zone_name}: Completed processing of {total_candidates} candidates "
            f"from {total_test_cells} cells in {prediction_time:.2f}s"
        )

        return ZonePredictionResults(
            zone_name=zone_name,
            y_pred=all_y_pred,
            y_decision_score=all_y_decision_score,
            y_test=all_y_test,
            test_samples=all_test_samples,
            model_path="",
            training_prediction_time=0.0,
            prediction_time=prediction_time,
        )

    except Exception as e:
        logging.error(f"Prediction failed for zone {zone_name}: {e}")
        return _empty_prediction_results(zone_name)


def train_zone_classifier(
    training_data: ZoneTrainingData, config: Any
) -> Optional[AdaBoostClassifier]:
    """Train AdaBoost classifier and track feature importance."""

    if not training_data.X_train or training_data.n_positive == 0:
        logging.warning(
            f"Insufficient training data for zone {training_data.zone_name}"
        )
        return None

    logging.info(f"Training AdaBoost classifier for zone {training_data.zone_name}")

    try:
        X_train = np.array(training_data.X_train, dtype=np.float32)
        y_train = np.array(training_data.y_train, dtype=np.int8)

        start_time = time.time()

        # Create and train AdaBoost classifier
        classifier = AdaBoostClassifier(
            n_estimators=getattr(config.training, "n_estimators", 100),
            learning_rate=getattr(config.training, "learning_rate", 1.0),
            random_state=getattr(config.experiment, "random_state", 42),
        )

        classifier.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Log feature importance
        _log_feature_importance(
            classifier, training_data.feature_names, training_data.zone_name
        )

        logging.info(f"Zone {training_data.zone_name}: Trained in {training_time:.2f}s")

        return classifier

    except Exception as e:
        logging.error(f"Training failed for zone {training_data.zone_name}: {e}")
        return None


def _log_feature_importance(
    classifier: AdaBoostClassifier, feature_names: list, zone_name: str
):
    """Log feature importance from trained classifier."""
    try:
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_

            # Create list of (feature_name, importance) pairs
            feature_importance_pairs = list(zip(feature_names, importances))

            # Sort by importance (descending)
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

            # Log all features with non-zero importance
            non_zero_features = [
                (name, imp) for name, imp in feature_importance_pairs if imp > 0.0001
            ]

            logging.info(
                f"Feature importance for zone {zone_name} ({len(non_zero_features)} features with importance > 0.0001):"
            )

            for i, (feature_name, importance) in enumerate(non_zero_features):
                logging.info(f"  {i + 1:2d}. {feature_name}: {importance:.6f}")

            # Also log zero-importance features count
            zero_features = len(feature_importance_pairs) - len(non_zero_features)
            if zero_features > 0:
                logging.info(f"  ({zero_features} features have importance <= 0.0001)")

            # Store feature importance in classifier for later access
            classifier.feature_importance_ranking = feature_importance_pairs

    except Exception as e:
        logging.warning(
            f"Failed to compute feature importance for zone {zone_name}: {e}"
        )


def get_feature_importance(classifier: AdaBoostClassifier) -> list:
    """
    Get feature importance ranking from trained classifier.

    Returns:
        List of (feature_name, importance_score) tuples, sorted by importance (descending)
    """
    if hasattr(classifier, "feature_importance_ranking"):
        return classifier.feature_importance_ranking
    elif hasattr(classifier, "feature_importances_"):
        # Fallback: create ranking on the fly
        feature_names = [
            f"feature_{i}" for i in range(len(classifier.feature_importances_))
        ]
        importance_pairs = list(zip(feature_names, classifier.feature_importances_))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        return importance_pairs
    else:
        return []


def _empty_prediction_results(zone_name: str) -> ZonePredictionResults:
    """Create empty prediction results for error cases."""
    return ZonePredictionResults(
        zone_name=zone_name,
        y_pred=[],
        y_decision_score=[],
        y_test=[],
        test_samples=[],
        model_path="",
        training_prediction_time=0.0,
        prediction_time=0.0,
    )


def save_feature_importance(
    classifier: AdaBoostClassifier, output_dir: str, zone_name: str
) -> str:
    """
    Save complete feature importance ranking to a file.

    Args:
        classifier: Trained classifier with feature importance
        output_dir: Directory to save the file
        zone_name: Name of the zone for filename

    Returns:
        Path to saved file, empty string if failed
    """
    try:
        feature_importance = get_feature_importance(classifier)
        if not feature_importance:
            logging.warning(f"No feature importance available for zone {zone_name}")
            return ""

        os.makedirs(output_dir, exist_ok=True)

        # Save to text file
        importance_file = os.path.join(
            output_dir, f"{zone_name}_feature_importance.txt"
        )

        with open(importance_file, "w") as f:
            f.write(f"Feature Importance Ranking for Zone: {zone_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"{'Rank':<6} {'Feature Name':<40} {'Importance':<12}\n")
            f.write("-" * 60 + "\n")

            for i, (feature_name, importance) in enumerate(feature_importance):
                f.write(f"{i + 1:<6} {feature_name:<40} {importance:<12.6f}\n")

            f.write(f"\nTotal features: {len(feature_importance)}\n")
            non_zero = len([f for f in feature_importance if f[1] > 0.0001])
            f.write(f"Features with importance > 0.0001: {non_zero}\n")

        logging.info(f"Feature importance saved to: {importance_file}")
        return importance_file

    except Exception as e:
        logging.error(f"Failed to save feature importance for zone {zone_name}: {e}")
        return ""


def save_zone_model(
    classifier: AdaBoostClassifier, zone_name: str, output_dir: str
) -> str:
    """Save model with compression"""

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"zone_{zone_name}_model.pkl")

    try:
        with open(model_path, "wb") as f:
            pickle.dump(classifier, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Zone {zone_name}: Model saved to {model_path}")
        return model_path
    except Exception as e:
        logging.error(f"Failed to save model for zone {zone_name}: {e}")
        return ""
