import logging
import os
from configparser import ConfigParser

from config.pipeline_config import (
    CorrectionConfig,
    DirectoryConfig,
    ExperimentConfig,
    LabelingConfig,
    PipelineConfig,
    PruningConfig,
    RuntimeConfig,
    SamplingConfig,
    SharingConfig,
    TrainingConfig,
    ZoningConfig,
)


def str_to_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def read_ecs_config(
    config_path: str = "config/config.ini",
) -> PipelineConfig:
    logging.info("Reading the configuration file.")
    config = ConfigParser()
    config.read(config_path)

    # DIRECTORIES
    sandbox_dir = config["DIRECTORIES"]["sandbox_dir"]
    tables_dir = os.path.join(sandbox_dir, config["DIRECTORIES"]["tables_dir"])
    dirty_files_name = config["DIRECTORIES"]["dirty_files_name"]
    clean_files_name = config["DIRECTORIES"]["clean_files_name"]
    output_dir = config["DIRECTORIES"]["output_dir"]
    logs_dir = os.path.join(sandbox_dir, config["DIRECTORIES"]["logs_dir"])
    exp_name = config["EXPERIMENT"]["exp_name"]
    logs_dir = os.path.join(logs_dir, exp_name)
    experiment_output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(experiment_output_dir, exist_ok=True)
    # EXPERIMENT
    experiment_config = ExperimentConfig(
        exp_name=exp_name,
        random_state=int(config["EXPERIMENT"]["random_state"]),
        log_level=config["EXPERIMENT"]["log_level"],
        cell_analysis_mode=config["EXPERIMENT"].get("cell_analysis_mode", "full"),
    )

    # LABELING
    labeling_config = LabelingConfig(
        labeling_budget=int(config["LABELING"]["labeling_budget"]),
    )

    # SAMPLING
    sampling_config = SamplingConfig(
        samples_path=config["SAMPLING"]["samples_path"],
        sampling_strategies_per_zone=[
            s.strip()
            for s in config["SAMPLING"]["sampling_strategies_per_zone"].split(",")
        ],
        cluster_sampling_strategy=config["SAMPLING"]
        .get("cluster_sampling_strategy", "column_coverage")
        .strip()
        .lower(),
    )

    # RUNTIME
    runtime_config = RuntimeConfig(
        n_cores=int(config["RUNTIME"]["n_cores"]),
        save_mediate_res_on_disk=str_to_bool(
            config["RUNTIME"]["save_mediate_res_on_disk"]
        ),
    )

    # PRUNING
    pruning_config = PruningConfig(
        vicinity_confidence_threshold=float(
            config["PRUNING"]["vicinity_confidence_threshold"]
        ),
        feature_pruning_enabled=str_to_bool(
            config["PRUNING"]["feature_pruning_enabled"]
        ),
        candidate_pruning_enabled=str_to_bool(
            config["PRUNING"]["candidate_pruning_enabled"]
        ),
        cardinality_threshold=float(config["PRUNING"]["cardinality_threshold"]),
        feature_value_threshold=float(config["PRUNING"]["feature_value_threshold"]),
    )

    sharing_config = SharingConfig(
        sharing_candidates_enabled=str_to_bool(
            config["SHARING"]["sharing_candidates_enabled"]
        ),
    )

    # Correction
    _legacy_enable = str_to_bool(
        config["CORRECTION"].get("enable_pattern_enforcement", "true")
    )
    _mode = config["CORRECTION"].get("pattern_enforcement_mode", "").strip().lower()
    if not _mode:
        _mode = "check" if _legacy_enable else "disabled"

    correction_config = CorrectionConfig(
        strategies=[
            strategy.strip()
            for strategy in config["CORRECTION"]["strategies"].split(",")
        ],
        min_candidate_probability=float(
            config["CORRECTION"]["min_candidate_probability"]
        ),
        min_occurrence=int(config["CORRECTION"]["min_occurrence"]),
        max_value_length=int(config["CORRECTION"]["max_value_length"]),
        value_encodings=[
            encoding.strip()
            for encoding in config["CORRECTION"]["value_encodings"].split(",")
        ],
        chunk_size=int(config["CORRECTION"]["chunk_size"]),
        enable_pattern_enforcement=_legacy_enable,
        pattern_enforcement_mode=_mode,
    )

    # Training parameters
    training_config = TrainingConfig(
        classification_strategy=config["TRAINING"].get(
            "classification_strategy", "multi"
        ),
        training_mode=config["TRAINING"].get("training_mode", "per_zone"),
        n_estimators=int(config["TRAINING"]["n_estimators"]),
        learning_rate=float(config["TRAINING"]["learning_rate"]),
        use_similarity_feature=str_to_bool(
            config["TRAINING"].get("use_similarity_feature", "false")
        ),
        classification_model=config["TRAINING"].get(
            "classification_model", "adaboost"
        ),
        negative_pruning_enabled=str_to_bool(
            config["TRAINING"].get("negative_pruning_enabled", "false")
        ),
        disabled_feature_groups=[
            g.strip()
            for g in config["TRAINING"].get("disabled_feature_groups", "").split(",")
            if g.strip()
        ],
    )

    # ZONING
    zoning_config = ZoningConfig(
        strategy=config["ZONING"].get("strategy", "rule_based"),
    )

    # FINAL CONFIG WRAP
    return PipelineConfig(
        directories=DirectoryConfig(
            sandbox_dir=sandbox_dir,
            tables_dir=tables_dir,
            dirty_files_name=dirty_files_name,
            clean_files_name=clean_files_name,
            output_dir=output_dir,
            logs_dir=logs_dir,
        ),
        experiment=experiment_config,
        labeling=labeling_config,
        sampling=sampling_config,
        runtime=runtime_config,
        pruning=pruning_config,
        sharing=sharing_config,
        correction=correction_config,
        zoning=zoning_config,
        training=training_config,
    )
