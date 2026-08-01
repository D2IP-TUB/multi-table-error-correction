# cross-table-correction

Cross-table error correction with rule-driven (**RCC**) and data-driven (**DCC**) zoning.

## Layout

| Path | Role |
|------|------|
| `main.py` | Entry point (dispatches by strategy) |
| `main_multi_clf.py` | Multi-classifier (zone-specific) strategy |
| `config/config.ini` | Paths, labeling budget, zoning, training |
| `core/` | Cell / column / table / zone / lake models |
| `modules/` | Candidates, features, sampling, zones, classification, evaluation |
| `utils/` | I/O, logging, aggregation helpers |
| `tane/` | TANE FD discovery |
| `run_*_ablation.py` | Ablation runners used in the paper |

## Setup

```bash
pip install -r requirements.txt
```

Edit `config/config.ini` (table paths, `labeling_budget`, `classification_strategy`, etc.).

## Run

```bash
python main.py
python main.py --config /path/to/config.ini
python main.py --strategy multi
```

| Script | Purpose |
|--------|---------|
| `main.py` | Unified entry |
| `main_multi_clf.py` | Multi-classifier strategy |
| `run_multiple_exp.py` | Batch experiments |
| `run_feature_ablation.py` | Feature ablation |
| `run_centroid_sampling_ablation.py` | Sampling ablation |
| `run_negative_pruning_ablation.py` | Pruning ablation |
| `run_pattern_enforcement_ablation.py` | Pattern enforcement ablation |
| `extract_results.py` / `export_ablation_plots.py` | Result summarization |

Relative paths in `config.ini` resolve against the **repository root** (parent of `cross-table-correction/`).
