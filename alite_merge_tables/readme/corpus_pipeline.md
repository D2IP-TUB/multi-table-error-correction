# Running ALITE on real-lake corpora (start to finish)

This guide walks through the full pipeline for **open_data_uk** and **mit_dwh**: environment setup, BLEND table discovery, cluster preparation, Full Disjunction (FD), error annotation, and optional visualization.

All paths below are relative to the **`alite_merge_tables/`** directory (unless noted).

---

## What you get at the end

For each corpus, the pipeline produces:

| Artifact | Location |
|----------|----------|
| BLEND index | `datasets/<corpus>_blend_index.duckdb` |
| ALITE input clusters | `results/alite/<corpus>/clusters_for_alite/` |
| FD-integrated tables | `results/alite/<corpus>/fd_output/<cluster>.csv` |
| Cell provenance | `results/alite/<corpus>/fd_output/<cluster>_merged_cell_source_map.csv` |
| Tuple subsumption map | `results/alite/<corpus>/fd_output/<cluster>_subsumption_map.csv` |
| Run statistics | `results/alite/<corpus>/fd_statistics.csv` |
| Plots (optional) | `results/alite/<corpus>/fd_output/plots/` |

**Inputs:** parent repo `../datasets/real_lakes/<corpus>/`. **Outputs:** `results/alite/<corpus>/` under `alite_merge_tables/`.

---

## 1. Environment setup

```bash
cd /path/to/multi-table-error-correction/alite_merge_tables

python3 -m venv env
source env/bin/activate

pip install -r requirements.txt
pip install -r blend_merge_tables/requirements.txt
```

Python 3.9+ is recommended. The BLEND stage needs **duckdb** and **tqdm**; visualization needs **networkx** and **matplotlib**.

Smoke-test ALITE on the tiny built-in example (no corpus data required):

```bash
cd codes
python test_provenance.py
cd ..
```

---

## 2. Prepare the corpus data

Each corpus is a directory of **one folder per table**. Every table folder must contain:

| File | Purpose |
|------|---------|
| `dirty.csv` | Table with injected errors (what BLEND indexes and ALITE integrates) |
| `clean.csv` | Ground-truth clean version (used to find erroneous cells) |
| `error_map.csv` | Per-cell error metadata (used in step 5) |

**Corpus locations:**

| Corpus | Isolated tables directory |
|--------|---------------------------|
| `open_data_uk` | `datasets/tables/uk_open_data/isolated/` |
| `mit_dwh` | `datasets/tables/mit_dwh/isolated/` |

Quick sanity check (replace corpus path as needed):

```bash
# Should list one subfolder per table
ls datasets/tables/mit_dwh/isolated/ | head

# Each folder should have the three CSVs
ls datasets/tables/mit_dwh/isolated/$(ls datasets/tables/mit_dwh/isolated/ | head -1)/
```

---

## 3. Select the active corpus

Edit `blend_merge_tables/config.py` and set:

```python
CORPUS = 'mit_dwh'      # or 'open_data_uk'
```

This controls which isolated tables are indexed (`../datasets/real_lakes/<corpus>/`). Steps 4–5 use `--corpus` on the command line for ALITE outputs under `results/alite/<corpus>/`.

Run the pipeline **once per corpus** (set `CORPUS`, then run steps 4–8).

---

## 4. Build the BLEND index

Indexes cell values from every table’s `dirty.csv` and marks cells that differ from `clean.csv` as erroneous.

```bash
cd blend_merge_tables
python index_tables.py
```

**Input:** `datasets/tables/<corpus>/isolated/*/dirty.csv`  
**Output:** `datasets/<corpus>_blend_index.duckdb`

Indexing can take a while on large corpora (tens of minutes to a few hours depending on size and disk). Progress is shown via `tqdm`.

To re-index from scratch, delete the old DuckDB file first:

```bash
rm -f ../datasets/${CORPUS}_blend_index.duckdb   # set CORPUS in your shell or edit path
```

---

## 5. Discover clusters for ALITE

Finds joinable/unionable table pairs, groups them into connected components (Union-Find), nullifies erroneous cells in the copies sent to ALITE, and writes one folder per cluster.

```bash
# still in blend_merge_tables/
python discover_clusters.py
```

**Output:** `results/alite/<corpus>/clusters_for_alite/`

- Multi-table clusters → `cluster_0001/`, `cluster_0002/`, …
- Singleton tables → `<table_name>/`

Each folder contains `<table_name>.csv` files (sanitized dirty tables, ready for FD).

Optional: visualize the BLEND relationship graph before running ALITE:

```bash
python visualize_clusters.py
# → results/alite/<corpus>/clusters_for_alite/cluster_graph.png
# → results/alite/<corpus>/clusters_for_alite/cluster_sizes.png
```

---

## 6. Run ALITE Full Disjunction

Runs exact FD on every cluster folder.

```bash
cd ../codes
python alite_fd.py --corpus mit_dwh
# or
python alite_fd.py --corpus open_data_uk
```

**Output per cluster** in `results/alite/<corpus>/fd_output/`:

| File | Description |
|------|-------------|
| `<cluster>.csv` | FD-integrated result table |
| `<cluster>_merged_cell_source_map.csv` | Which source row supplied each output cell value |
| `<cluster>_subsumption_map.csv` | Which source rows were absorbed into each output row |

Statistics accumulate in `results/alite/<corpus>/fd_statistics.csv`.

### Runtime notes

- Most clusters finish in minutes; a few large outer-union clusters can take **hours to days** (notably **mit_dwh `cluster_0027`**: ~37 tables, ~94k rows × 437 columns).
- The script processes clusters sequentially and **skips clusters that already have output CSVs** only if you delete partial outputs before re-running; on failure it prints `ERROR in cluster …` and continues with the next cluster.
- To re-run a single cluster, delete its three output files in `fd_output/` and run `alite_fd.py` again.

Monitor progress:

```bash
ls results/alite/mit_dwh/fd_output/*.csv 2>/dev/null | wc -l
tail -5 results/alite/mit_dwh/fd_statistics.csv
```

---

## 7. Backfill error types

Annotates each `*_subsumption_map.csv` with error information from the isolated `error_map.csv` files.

```bash
# still in codes/
python backfill_error_types.py --corpus mit_dwh
# or
python backfill_error_types.py --corpus open_data_uk
```

**Adds columns:**

| Column | Meaning |
|--------|---------|
| `error_column` | Column where the error occurred (empty if none) |
| `error_type` | e.g. `RANDOM_TYPO`, `FD_VIOLATION` |
| `corrected_value` | Clean value from `error_map.csv` |

**Why the subsumption map?** Erroneous cells are nullified before ALITE runs, so they never appear as value providers in the cell source map. The subsumption map still records every absorbed source row, including rows whose bad cells were blanked out — which is what you need for majority voting after correction.

---

## 8. Visualize results (optional)

```bash
python visualize_fd_clusters.py --corpus open_data_uk
python visualize_fd_clusters.py --corpus mit_dwh
```

**Plots** under `results/alite/<corpus>/fd_output/plots/`:

| Plot | Description |
|------|-------------|
| `connected_components_grid.png` | One panel per multi-table cluster (join/union edges) |
| `connected_components.png` | All multi-table nodes in one graph |
| `component_table_counts.png` | Tables per connected component |
| `fd_cluster_dimensions.png` | Rows and columns per FD output cluster |
| `fd_input_vs_output_rows.png` | Input table count vs FD output rows |
| `fd_cluster_sizes.csv` | Summary table of cluster sizes |

---

## One-shot script

From the repo root (after setup and data are in place):

```bash
./scripts/run_alite_corpus.sh mit_dwh
./scripts/run_alite_corpus.sh open_data_uk
```

Useful flags:

```bash
./scripts/run_alite_corpus.sh mit_dwh --from-step discover   # skip re-indexing
./scripts/run_alite_corpus.sh mit_dwh --skip-viz              # skip plots
./scripts/run_alite_corpus.sh mit_dwh --dry-run              # print commands only
```

The script sets `CORPUS` in `blend_merge_tables/config.py`, then runs: index → discover → FD → backfill → visualize.

## Manual command block

If you prefer to run steps yourself:

```bash
export CORPUS=mit_dwh   # or open_data_uk

# 1) Set CORPUS in blend_merge_tables/config.py to match $CORPUS

cd blend_merge_tables
python index_tables.py
python discover_clusters.py
cd ../codes
python alite_fd.py --corpus "$CORPUS"
python backfill_error_types.py --corpus "$CORPUS"
python visualize_fd_clusters.py --corpus "$CORPUS"
```

Repeat with `export CORPUS=open_data_uk` (and update `config.py`) for the second corpus.

### Export per-cluster dirty/clean tables

After FD (and ideally error backfill), export BLEND-style per-cluster directories:

```bash
./scripts/export_alite_merged_tables.sh open_data_uk
# or
cd codes && python export_alite_merged_tables.py --corpus open_data_uk
```

**Output:** `results/alite/<corpus>/merged_tables/<cluster>/`

| File | Description |
|------|-------------|
| `dirty.csv` | ALITE FD-integrated table |
| `clean.csv` | Cell-by-cell clean values from provenance + subsumption corrections |
| `merged_cell_source_map.csv` | ALITE cell provenance |
| `subsumption_map.csv` | Tuple absorption map (with error annotations if backfill ran) |
| `clean_changes_provenance.csv` | Source errors present in the merge |
| `isolated_error_map.csv` | Source error cells → merged row indices |
| `provenance.csv` | BLEND-style `table § col § row` grid |

---

## Quick reference

| Step | Where to run | Command | Key output |
|------|--------------|---------|------------|
| 0. Setup | repo root | `pip install -r requirements.txt -r blend_merge_tables/requirements.txt` | venv |
| 1. Index | `blend_merge_tables/` | `python index_tables.py` | `datasets/<corpus>_blend_index.duckdb` |
| 2. Clusters | `blend_merge_tables/` | `python discover_clusters.py` | `results/alite/<corpus>/clusters_for_alite/` |
| 3. FD | `codes/` | `python alite_fd.py --corpus <corpus>` | `results/alite/<corpus>/fd_output/` |
| 4. Errors | `codes/` | `python backfill_error_types.py --corpus <corpus>` | annotated `*_subsumption_map.csv` |
| 5. Plots | `codes/` | `python visualize_fd_clusters.py --corpus <corpus>` | `fd_output/plots/` |
| **All steps** | repo root | `./scripts/run_alite_corpus.sh <corpus>` | `results/alite/<corpus>/` |

---

## Directory map

```
alite_merge_tables/
├── results/
│   └── alite/
│       ├── open_data_uk/
│       │   ├── clusters_for_alite/
│       │   ├── fd_output/
│       │   └── fd_statistics.csv
│       └── mit_dwh/
│           ├── clusters_for_alite/
│           ├── fd_output/
│           └── fd_statistics.csv
├── blend_merge_tables/
│   ├── config.py          # set CORPUS here (uses ../datasets/ by default)
│   ├── index_tables.py
│   └── discover_clusters.py
└── codes/
    ├── alite_fd.py
    ├── backfill_error_types.py
    └── visualize_fd_clusters.py
```

Isolated lake tables live in the parent repo at `../datasets/real_lakes/<corpus>/`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `index_tables.py` finds no tables | Wrong `CORPUS` in `config.py` or missing isolated data | Check `DIR_PATH` in config and folder layout |
| `discover_clusters.py` fails on missing DB | Index not built | Run step 4 |
| `alite_fd.py` prints `Input folder: …` with 0 clusters | Clusters not discovered or wrong `--corpus` | Run step 5; check `results/alite/<corpus>/clusters_for_alite/` |
| One cluster hangs for hours | Large wide outer-union cluster (exact FD) | Expected for hard clusters; let it run or exclude that cluster manually |
| `backfill_error_types.py` finds 0 files | FD step not finished | Complete step 6 first |
| Import errors for `duckdb` / `networkx` | BLEND deps not installed | `pip install -r blend_merge_tables/requirements.txt` |

---

## Minimal test (no corpus data)

```bash
cd codes
python test_provenance.py
```

Runs FD on `codes/minimum_example/integration_set/` and prints FD output, subsumption map, and cell source map side by side.
