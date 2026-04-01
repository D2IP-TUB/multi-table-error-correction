from dataclasses import dataclass, field
from typing import List


@dataclass
class DirectoryConfig:
    sandbox_dir: str
    tables_dir: str
    dirty_files_name: str
    clean_files_name: str
    output_dir: str
    logs_dir: str


@dataclass
class ExperimentConfig:
    exp_name: str
    random_state: int
    log_level: str
    cell_analysis_mode: str = "full"  # Options: 'full', 'summary', 'disabled'


@dataclass
class LabelingConfig:
    labeling_budget: int


@dataclass
class SamplingConfig:
    samples_path: str
    sampling_strategies_per_zone: list[str] = None
    # Used only when zoning.strategy == clustering_based: how to pick labeled cells
    # after KMeans: column_coverage | centroid | kmeans_pp (D² seeding within cluster).
    cluster_sampling_strategy: str = "column_coverage"


@dataclass
class RuntimeConfig:
    n_cores: int
    save_mediate_res_on_disk: bool


@dataclass
class PruningConfig:
    vicinity_confidence_threshold: float
    feature_pruning_enabled: bool
    candidate_pruning_enabled: bool
    cardinality_threshold: float
    feature_value_threshold: float


@dataclass
class SharingConfig:
    sharing_candidates_enabled: bool


@dataclass
class CorrectionConfig:
    """Correction configuration that extends existing config"""

    # Candidate generation strategies
    strategies: List[str] = field(
        default_factory=lambda: ["value_based", "vicinity_based", "domain_based"]
    )

    # Correction parameters
    min_candidate_probability: float = 0.0
    min_occurrence: int = 2
    max_value_length: int = 50
    value_encodings: List[str] = field(default_factory=lambda: ["identity", "unicode"])
    chunk_size: int = 1000
    # If False, skip "perfect detector" pattern enforcement pass in invalid pattern zones.
    # Candidate generation still runs as usual.
    enable_pattern_enforcement: bool = True
    # Pattern enforcement mode for invalid-pattern zones:
    # - "check": accept only when enforced value matches ground truth
    # - "always_accept": always accept top enforced value
    # - "disabled": skip pattern enforcement stage
    pattern_enforcement_mode: str = "check"


@dataclass
class ZoningConfig:
    """Zoning strategy configuration"""

    # Zoning strategy: "rule_based" or "clustering_based"
    strategy: str = "rule_based"


@dataclass
class TrainingConfig:
    # Classifier strategy
    classification_strategy: str = "multi"  # Options: "single" or "multi"
    # - "single": One global classifier for all errors
    # - "multi": Separate classifier per zone (recommended for heterogeneous errors)

    # Training parameters
    training_mode: str = "per_zone"  # Options: "per_column", "per_zone",
    # AdaBoost classifier parameters
    n_estimators: int = 50
    learning_rate: float = 1.0

    # Feature engineering
    use_similarity_feature: bool = False  # Set to true to add edit distance similarity

    classification_model: str = "adaboost"

    # Negative sample pruning for training
    # Filters out uninformative negative samples that don't use same vicinity context
    # as positive samples, reducing training noise and improving classifier focus
    negative_pruning_enabled: bool = False

    # Feature ablation: list of feature groups to disable (zero out) during training/testing.
    # Valid groups: "value_based", "vicinity_based", "domain_based", "levenshtein", "pattern_based"
    disabled_feature_groups: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    directories: DirectoryConfig
    experiment: ExperimentConfig
    labeling: LabelingConfig
    sampling: SamplingConfig
    runtime: RuntimeConfig
    pruning: PruningConfig
    sharing: SharingConfig
    correction: CorrectionConfig
    zoning: ZoningConfig
    training: TrainingConfig
