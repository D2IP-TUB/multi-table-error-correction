from dataclasses import dataclass


@dataclass
class PipelineMetrics:
    """Overall pipeline performance metrics"""

    total_error_cells: int
    total_candidates_generated: int
    total_pseudo_labels: int
    total_predictions: int
    total_corrections: int
    total_correct_corrections: int

    overall_precision: float
    overall_recall: float
    overall_f1: float

    candidate_generation_time: float
    feature_extraction_time: float
    label_propagation_time: float
    training_and_prediction: float
    sampling_time: float
    # prediction_time: float
    total_time: float

    zones_processed: int
    tables_processed: int
