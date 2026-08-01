# Multi-Table Error Correction: Opportunities and Limitations

Reproducibility package for:

> **Multi-Table Error Correction: Opportunities and Limitations [Experiment, Analysis & Benchmark]**  
> Fatemeh Ahmadi, Luca Zecchini, Ziawasch Abedjan — BIFOLD & TU Berlin.

Code, datasets, and baselines for isolated, post-merge, and cross-table error correction across joinable, unionable, and unrelated tables.

## Repository structure

```
├── cross-table-correction/   # RCC & DCC cross-table correction
├── blend_merge_tables/       # BLEND greedy discovery + merge (post-merge)
├── alite_merge_tables/       # ALITE Full Disjunction merge baseline
├── baselines/                # Baran, HoloClean, Horizon, UniClean, ZeroEC
└── datasets/                 # Benchmark lakes (OpenData-UK, Quintet, joins, unions)
```

Error injection tooling: [Data Lake Error Generator](https://github.com/LUH-DBS/Data-Lake-Error-Generator).

## Getting started

- Python 3.10+ recommended
- Install per component:

```bash
pip install -r cross-table-correction/requirements.txt
pip install -r blend_merge_tables/requirements.txt
pip install -r alite_merge_tables/requirements.txt
```

Baselines need their own environments; see each folder under `baselines/`.

### Cross-table correction

```bash
cd cross-table-correction
python main.py          # or main_multi_clf.py
```

Configure paths and zoning in `config/config.ini`.

### Post-merge (BLEND greedy)

```bash
cd blend_merge_tables
# set CORPUS / paths in config.py
python index_tables.py
python merge_tables.py
python recreate_as_strings.py
```

Then run any baseline on the merged tables.

### ALITE merge baseline

See `alite_merge_tables/readme/corpus_pipeline.md` and `alite_merge_tables/scripts/run_alite_corpus.sh`.

### Datasets

See `datasets/README.md`.
