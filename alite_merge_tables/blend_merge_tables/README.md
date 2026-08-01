# blend_merge_tables (ALITE prep)

BLEND **index + discovery** only — builds the DuckDB index and writes join/union clusters for ALITE Full Disjunction. There is **no greedy merge**, FD merge validation, or merged-table recreate here.

For the paper’s greedy Blend merge pipeline, use the top-level `blend_merge_tables/` directory in this repository.

---

## Pipeline

1. **Index** — `index_tables.py` builds a DuckDB BLEND index over isolated tables.
2. **Discover** — `discover_clusters.py` finds joinable/unionable pairs, groups connected components, and writes cluster folders for ALITE.

Downstream ALITE steps (`alite_fd.py`, etc.) live under `../codes/` — see `../readme/corpus_pipeline.md`.

---

## Files

| File | Description |
|------|-------------|
| `config.py` | Corpus paths, discovery thresholds, `ERROR_MODE` (indexing) |
| `error_cells.py` | GT vs detected error-cell logic (indexing) |
| `index_tables.py` | Build the DuckDB BLEND index |
| `merge_tables.py` | Join/union discovery helpers (`find_joinable_tables`, `find_unionable_tables`) |
| `discover_clusters.py` | Cluster connected components → `CLUSTERS_PATH` |
| `visualize_clusters.py` | Optional cluster graph plots |
| `generate_isolated_error_provenance.py` | Optional `error_map.csv` for isolated tables |
| `count_errors.py` | Optional dirty/clean error counts |
| `utils.py` | Tokenization helper (if `TOKENIZE=True`) |

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ERROR_MODE` | `'GT'` | `'GT'` (dirty vs clean) or `'DETECTED'` (per-table CSV). Used in **indexing** only. |
| `DETECTED_ERRORS_FILENAME` | `'detected_errors.csv'` | Filename when `ERROR_MODE='DETECTED'` |
| `DETECTED_ERRORS_DIR` | `None` | Optional separate root for detection files |
| `CORPUS` | `'open_data_uk'` | `'open_data_uk'` or `'mit_dwh'` |
| `DIR_PATH` | `.../tables/<corpus>/isolated` | Isolated corpus tables |
| `DB_PATH` | `.../<corpus>_blend_index.duckdb` | DuckDB index path |
| `CLUSTERS_PATH` | `results/alite/<corpus>/clusters_for_alite` | Cluster output |
| `JOIN` / `UNION` | `True` | Enable join / union discovery |
| `JOIN_THRESHOLD` / `UNION_THRESHOLD` | `0.5` | Overlap score cutoffs |
| `TOP_JOIN` / `TOP_UNION` | `10` | Top-k candidates |

### Error modes

**GT** (default): a cell is clean when normalized dirty equals clean (HTML-unescape + whitespace collapse).

**DETECTED**: `(row, column)` coordinates in `detected_errors.csv` are erroneous; other cells are clean.

Affects **indexing** (`cell_idx.is_clean`). Discovery always compares dirty vs clean directly.

---

## Usage

```bash
pip install -r requirements.txt

# Set CORPUS in config.py, then:
python index_tables.py
python discover_clusters.py

# Optional:
python visualize_clusters.py
python generate_isolated_error_provenance.py --input_dir <isolated_tables_dir>
python count_errors.py <directory>
```

Or use `../scripts/run_alite_corpus.sh` from the ALITE repo root.

---

## BLEND Index Schema

| Table | Columns |
|-------|---------|
| `cell_idx` | `tab_id`, `col_id`, `row_id`, `value`, `tokenized`, `is_clean` |
| `col_idx` | `tab_id`, `col_id`, `header`, `is_numeric` |
| `tab_idx` | `tab_id`, `name` |

---

## Cluster output

Under `CLUSTERS_PATH`:

```
cluster_<id>/
├── <table_a>.csv   # sanitized dirty (erroneous cells blanked)
├── <table_b>.csv
└── ...
```
