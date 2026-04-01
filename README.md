# Multi-Table Error Correction: Opportunities and Limitations

Reproducibility package for the paper:

> **Multi-Table Error Correction: Opportunities and Limitations [Experiment, Analysis & Benchmark]**
> Fatemeh Ahmadi, Luca Zecchini, Ziawasch Abedjan — BIFOLD & TU Berlin.

This repository contains code, datasets, and baselines for a systematic study of multi-table error correction. We evaluate three correction strategies — **isolated**, **post-merge**, and **cross-table** — across different table relationships (joinable, unionable, unrelated) and error types (typos, FD violations, formatting issues, missing values, outliers).

## Repository Structure

```
├── cross-table-correction/   # Cross-table correction system (RCC & DCC)
├── blend_merge_tables/        # BLEND-based table discovery and merging pipeline
├── baselines/                 # Baseline systems (Baran, HoloClean, Horizon, UniClean, ZeroEC)
└── datasets/                  # Benchmark datasets
```

### cross-table-correction/
The core cross-table correction system. It assigns correction techniques based on the latent error type of each cell, using two zoning strategies:
- **Rule-driven zoning (RCC)**: partitions cells by column uniqueness and pattern consistency.
- **Data-driven zoning (DCC)**: clusters cells based on deviation features.

Entry point: `main.py` / `main_multi_clf.py`. Configuration: `config/config.ini`.

### blend_merge_tables/
Discovers and merges joinable/unionable tables using the [BLEND](https://github.com/LUH-DBS/Blend) data discovery system as a preprocessing step for post-merge correction.

Three phases:
1. **Index** (`index_tables.py`) — build a DuckDB index over the table corpus.
2. **Merge** (`merge_tables.py`) — discover join/union candidates and materialize merged tables.
3. **Annotate** (`recreate_as_strings.py`) — produce merged tables with cell-level error provenance.

### baselines

Baran, HoloClean, Horizon, UniClean, ZeroEC

### Error Generator
[Data Lake Error Generator](https://github.com/LUH-DBS/Data-Lake-Error-Generator.git)
## Getting Started

### Requirements
- Python 3.10+
- Install dependencies per component:
  ```bash
  pip install -r cross-table-correction/requirements.txt
  pip install -r blend_merge_tables/requirements.txt
  ```
- Baselines have their own environments (see each subdirectory's README).

### Running Cross-Table Correction
```bash
cd cross-table-correction
python main.py          # or main_multi_clf.py
```
Configuration is in `config/config.ini` (dataset paths, labeling budget, zoning strategy, etc.).

### Running Post-Merge Correction
```bash
cd blend_merge_tables
python index_tables.py      # Step 1: index tables
python merge_tables.py      # Step 2: discover & merge
python recreate_as_strings.py  # Step 3: annotate merged tables
```
Then run any baseline on the merged output.
