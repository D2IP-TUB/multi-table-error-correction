from typing import Any, Dict, List

from core.lake import Lake
from core.zone import Zone
from modules.classification.train_test import train_and_predict_all_zones
from modules.classification.zone_predictions_results import ZonePredictionResults


def initialize_training_and_prediction(
    zones_dict: Dict[str, Zone],
    pseudo_labels_dict: Dict[str, List],
    lake: Lake,
    config: Any,
) -> Dict[str, ZonePredictionResults]:
    """
    Main integration function - call this after label propagation

    Args:
        zones_dict: zones with samples and candidates
        pseudo_labels_dict: Output from label propagation
        lake:  Lake object
        config:  configuration

    Returns:
        Dict mapping zone names to prediction results
    """

    return train_and_predict_all_zones(zones_dict, pseudo_labels_dict, lake, config)
