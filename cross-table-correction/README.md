# cross-table-correction

## Project Structure

```
cross-table-correction/
├── README.md                           # This file
├── LICENSE                             # Project license
├── requirements.txt                    # Python dependencies
│
├── main.py                             # Main entry point (dispatches to strategies)
├── main_multi_clf.py                   # Multi-classifier strategy (zone-specific classifiers)
├── main_per_col.py                     # Per-column classifier strategy
│
├── config/                             # Configuration Management
│   ├── config.ini                      # Configuration parameters (paths, thresholds)
│   ├── pipeline_config.py              # Dataclass definitions for config
│   ├── parse_config.py                 # Config file parser
│   └── __init__.py
│
├── core/                               # Core Data Models
│   ├── cell.py                         # Cell representation (value, position, zone)
│   ├── column.py                       # Column metadata and properties
│   ├── table.py                        # Table representation with cells/columns
│   ├── zone.py                         # Zone definition and management
│   ├── zone_propagation_data.py        # Zone propagation algorithms
│   ├── lake.py                         # Data lake (collection of tables)
│   ├── candidate.py                    # Candidate correction representation
│   ├── candidate_pool.py               # Pool of candidates per cell
│   └── __init__.py
│
├── modules/                            # Processing Pipeline Modules
│   ├── candidate_generation/           # Generate correction candidates
│   │   ├── candidate_generator.py      # Main candidate generation orchestrator
│   │   ├── value_based_candidate_generator.py
│   │   ├── vicinity_based_candidate_generator.py
│   │   ├── domain_based_candidate_generator.py
│   │   ├── pattern_based_candidate_generator.py
│   │   ├── flash_fill_candidate_generator.py
│   │   ├── correction_pipeline.py      # Correction application pipeline
│   │   └── generate_candidates.py
│   │
│   ├── classification/                 # classification
│   │   ├── train_test.py               # Training and prediction logic
│   │   ├── init_training_prediction.py # Initialize training/prediction
│   │   ├── zone_test_data.py           # Test data per zone
│   │   ├── zone_predictions_results.py # Prediction results aggregation
│   │   └── ... (additional analysis modules)
│   │
│   ├── evaluation/                     # Evaluation and Metrics
│   │   └── ... (evaluation modules)
│   │
│   ├── feature_extraction/             # Feature generation
│   │   └── ... (feature extraction modules)
│   │
│   ├── label_propagation/              # Label propagation strategies
│   │   └── ... (label propagation modules)
│   │
│   ├── profiling/                      # Data profiling utilities
│   │   └── ... (profiling modules)
│   │
│   ├── sampling/                       # Sampling strategies for labeling
│   │   └── ... (sampling modules)
│   │
│   └── zones/                          # Zone detection and management
│       └── ... (zone modules)
│
├── utils/                              # Utility Functions
│   ├── read_data.py                    # Data loading utilities
│   ├── app_logger.py                   # Logging configuration
│   ├── memory_monitor.py               # Memory usage monitoring
│   ├── memory_utils.py                 # Memory utilities
│   ├── aggregate_results.py            # Result aggregation
│   ├── analyse_zones.py                # Zone analysis utilities
│   ├── get_results_per_table.py        # Per-table result extraction
│   ├── translate_ids.py                # ID translation utilities
│   └── ForceGC.cs                      # (Optional) Mono GC control
│
├── tane/                               # TANE Algorithm (Dependency Discovery)
│   ├── tane.py                         # TANE algorithm implementation
│   ├── requirements.txt
│   └── LICENSE
│
├── Experiment Scripts
│   ├── run_multiple_exp.py             # Run multiple experiments
│   ├── run_feature_ablation.py         # Ablation study: feature importance
│   ├── run_centroid_sampling_ablation.py # Ablation: centroid-based sampling
│   ├── run_negative_pruning_ablation.py # Ablation: negative pruning
│   ├── run_pattern_enforcement_ablation.py # Ablation: pattern enforcement
│   ├── test_memory_monitor.py          # Test memory monitoring
│   │
│   ├── Evaluation & Analysis
│   ├── evaluate_cell_analysis.py       # Cell-level error analysis
│   ├── extract_results.py              # Extract and summarize results
│   ├── export_ablation_plots.py        # Generate ablation study plots
│   ├── cluster_zone_utils.py           # Clustering-based zone utilities
│   └── plot.py                         # Plotting utilities
│
└── .gitignore
```

---

## Quick Start

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the pipeline** (optional):
   Edit `config/config.ini` to customize paths, thresholds, and strategies.

### Running the Pipeline

**Basic usage**:
```bash
python main.py
```

**With custom config**:
```bash
python main.py --config /path/to/config.ini
```

**Override strategy**:
```bash
python main.py --strategy multi
```

### Main Entry Points

| Script | Purpose |
|--------|---------|
| `main.py` | Unified entry point (dispatches to selected strategy) |
| `main_multi_clf.py` | Multi-classifier strategy |

---

## Configuration

Configuration is managed through `config/config.ini` with the following sections:

### Directories
- `sandbox_dir`: Working directory for intermediate results
- `tables_dir`: Input tables directory
- `dirty_files_name`: Name pattern for dirty tables (e.g., `dirty.csv`)
- `clean_files_name`: Name pattern for clean tables (e.g., `clean.csv`)
- `output_dir`: Output directory for results
- `logs_dir`: Logging directory

### Experiment
- `exp_name`: Experiment name
- `random_state`: Random seed
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `cell_analysis_mode`: Cell analysis detail level (full, summary, disabled)

### Labeling
- `labeling_budget`: Maximum number of cells to label

### Sampling
- `samples_path`: Path to pre-defined labeled samples
- `sampling_strategies_per_zone`: Sampling strategy per zone
- `cluster_sampling_strategy`: Sampling within clusters (column_coverage, centroid, kmeans_pp)

### Correction
- `strategies`: Candidate generation strategies (value_based, vicinity_based, domain_based, pattern_based)
- `min_candidate_probability`: Minimum probability threshold for candidates
- `enable_pattern_enforcement`: Enable pattern-based correction enforcement
- `pattern_enforcement_mode`: check | always_accept | disabled

### Training
- `classification_strategy`: single | multi
- `training_mode`: per_zone | per_column
- `n_estimators`: Number of AdaBoost estimators

### Pruning
- `vicinity_confidence_threshold`: Confidence threshold for vicinity pruning
- `feature_pruning_enabled`: Enable feature-level pruning
- `candidate_pruning_enabled`: Enable candidate pruning
- `cardinality_threshold`, `feature_value_threshold`: Pruning thresholds

