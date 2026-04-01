# blend_merge_tables

This module implements the **BLEND** pipeline for discovering and merging tables in a data lake corpus (join or union), and producing the annotated datasets used by multi-table error correction experiments.

---

## Overview

The pipeline has three phases:

1. **Index** — Build a DuckDB index (BLEND index) over the corpus tables.
2. **Merge** — Use the index to discover joinable / unionable table pairs and produce merged tables with cell-level provenance.
3. **Annotate** — Recreate merged tables as flat CSVs and generate per-cell error maps.

---

## Files

| File | Description |
|------|-------------|
| `config.py` | Central configuration (paths, thresholds, flags) |
| `index_tables.py` | Phase 1 — build the DuckDB BLEND index |
| `merge_tables.py` | Phase 2 — discover joinable/unionable pairs and merge them |
| `recreate_as_strings.py` | Phase 3 — write merged tables as flat CSVs with provenance |
| `recreate_as_strings_union.py` | Phase 3 variant for union-style merges (with deduplication) |
| `run_join_threshold_lakes.py` | Run Phase 2+3 for JOIN_THRESHOLD ∈ {0.25, 0.5, 0.75} |
| `run_union_threshold_lakes.py` | Run Phase 2+3 for UNION_THRESHOLD ∈ {0.25, 0.5, 0.75} |
| `generate_isolated_error_provenance.py` | Generate `error_map.csv` for every isolated table |
| `count_errors.py` | Count cell-level errors between `clean.csv` and `dirty.csv` |
| `sum_error_types.py` | Aggregate error-type counts from `isolated_error_map.csv` files |
| `utils.py` | Text tokenization helper (adapted from COCOA) |

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CORPUS` | `'mit_dwh'` | Target corpus (`'uk_open_data'` or `'mit_dwh'`) |
| `DIR_PATH` | `.../tables/<corpus>/isolated` | Directory containing isolated corpus tables |
| `MERGED_PATH` | `.../tables/<corpus>/merged` | Output directory for merged tables |
| `DB_PATH` | `.../indices/<corpus>_blend_index.duckdb` | DuckDB BLEND index path |
| `BATCH_SIZE` | `10_000` | Tuples inserted into the index at a time |
| `TAB_LIMIT` | `-1` | Max tables to index (`-1` = all) |
| `JOIN` / `UNION` | `True` | Enable join / union discovery |
| `JOIN_THRESHOLD` | `0.5` | Min ratio of joined tuples over table length |
| `JOIN_ROWS` | `0.1` | Min ratio of tuples joined per table |
| `TOP_JOIN` | `10` | Top-k joinable columns per primary key |
| `UNION_THRESHOLD` | `0.5` | Min ratio of matching tuples for union |
| `UNION_COLS` | `0.5` | Min ratio of matching columns for union |
| `TOP_UNION` | `10` | Top-k unionable tables per table |

Edit `DIR_PATH`, `MERGED_PATH`, and `DB_PATH` to match your local setup before running.

---

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the BLEND index

```bash
python index_tables.py
```

Reads all `dirty.csv` files from `DIR_PATH` and writes the DuckDB index to `DB_PATH`.

### 3. Merge tables

```bash
python merge_tables.py
```

Discovers joinable and unionable table pairs using the index and writes merged tables (with `dirty.csv`, `clean.csv`, and `provenance_map.csv`) to `MERGED_PATH`.

To sweep over multiple thresholds and produce separate lakes:

```bash
python run_join_threshold_lakes.py   # JOIN_THRESHOLD ∈ {0.25, 0.5, 0.75}
python run_union_threshold_lakes.py  # UNION_THRESHOLD ∈ {0.25, 0.5, 0.75}
```

### 4. Recreate merged tables as flat strings

```bash
python recreate_as_strings.py        # for join-merged tables
python recreate_as_strings_union.py  # for union-merged tables (with deduplication)
```

Produces flat CSVs with disambiguated headers and cell-level provenance in the format `table_id.col_id.row_id`.

### 5. Generate error provenance (isolated tables)

```bash
python generate_isolated_error_provenance.py --input_dir <path>
```

For each isolated table directory (containing `dirty.csv`, `clean.csv`, and optionally `clean_changes_provenance.csv`), writes:
- `error_map.csv` — per-cell errors with type annotation (`FD_VIOLATION`, `RANDOM_TYPO`, `UNKNOWN`)
- `error_map_summary.json` — summary statistics

### 6. Count and aggregate errors

```bash
python count_errors.py <directory>           # count dirty/clean diffs
python sum_error_types.py [<dir1> <dir2> ...]  # aggregate error types
```

---

## BLEND Index Schema

The DuckDB database contains three tables:

| Table | Columns |
|-------|---------|
| `cell_idx` | `tab_id`, `col_id`, `row_id`, `value`, `tokenized`, `is_clean` |
| `col_idx` | `tab_id`, `col_id`, `header`, `is_numeric` |
| `tab_idx` | `tab_id`, `name` |

---

## Output Structure

Each merged table directory under `MERGED_PATH` contains:

```
<merged_table_name>/
├── dirty.csv               # merged dirty data
├── clean.csv               # merged clean data
└── provenance_map.csv      # cell provenance: "source_table § col_id § row_id"
```

Merged column headers encode provenance: `table::col_id::col_name` (join) or `left::col_id::name | right::col_id::name` (union).

