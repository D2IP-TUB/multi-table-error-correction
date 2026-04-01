# Unionable Tables , Benchmark Datasets
Starting from 17 real-world government datasets (each with 3 error-type variants), we generate controlled union scenarios at varying entity-overlap levels, with per-row provenance tracking.

Overlapping entities always come from *different* error-type variants (FD vs Typo vs NO), so shared rows have different surface values, just like real-world unionable tables. There are no artificial exact-duplicate overlaps.

Each generated dataset is a **data lake** , a collection of tables that should be unioned. Every table in the lake has a `dirty.csv` (the noisy input a system receives) and a `clean.csv` (cell-level ground truth, same rows, same columns). `dirty.csv` and `clean.csv` always have identical row counts everywhere, including in expected results.

---

## Source Data

### Location

```
Unified_Union_Exp_isolated/
  DGov_{FD,Typo,NO}_{TableName}/
    clean.csv          Ground-truth (cleaned) version
    dirty.csv          Version with errors
    clean_changes.csv  Audit log of transformations applied
```

### Variants (prefixes)

| Prefix | Meaning |
|--------|---------|
| **FD** | Functional Dependency violation errors , errors that violate functional dependencies in the data |
| **Typo** | Typographical errors , intentional typos injected into cell values |
| **NO** | Numeric Outlier errors , numeric values replaced with outlier values |

`clean.csv` is the ground-truth and is **identical** across all three variants for most tables. `dirty.csv` is where the variants actually differ , it contains the errors specific to each prefix.

### Tables and Schema Groups

17 unique tables fall into 8 schema groups (tables in the same group share the same column names):

| # | Columns | Tables | Rows |
|---|---------|--------|------|
| 1 | 3 (`years`, `historical_data`, `target`) | Emergency\_Operating\_Center\_Tools, Regional\_Tier\_Graduation\_Rate, Shared\_IT\_Support\_Services | 7, 8, 7 |
| 2 | 8 (agency/fund info) | State\_of\_Oklahoma\_Revolving\_Funds\_2011, \_2012 | 1 075, 1 094 |
| 3 | 9 (COVID inspections) | COVID\_Complaint, COVID\_Random\_Survey | 4 724, 3 002 |
| 4 | 12 (estuary assessment) | 305b\_Assessed\_2008\_Estuary, Impaired\_Estuary\_2008 | 210, 179 |
| 5 | 13 (facility permits) | Permitted\_Hotels\_and\_Motels, Permitted\_Swimming\_Areas | 131, 538 |
| 6 | 15 (council districts) | Louisville\_Council\_Districts, \_Former\_from\_2011 | 26, 26 |
| 7 | 16 (health statistics) | Health\_conditions\_children, Obesity\_children\_adolescents | 2 744, 840 |
| 8 | 23 (lake assessment) | 305b\_Assessed\_Lake\_2020, Impaired\_Lake\_2020 | 182, 45 |

---

## Generated Datasets (Data Lakes)

### Overview

```
generated_union_datasets/
  metadata.json               Top-level metadata for all lakes
  disjoint/                   0% entity overlap   , 10 lakes
  maximal_overlap/            100% entity overlap  , 17 lakes
  partial_overlap_25/         25% entity overlap   , 17 lakes
  partial_overlap_50/         50% entity overlap   , 17 lakes
  partial_overlap_75/         75% entity overlap   , 17 lakes
```

**78 data lakes total.**

### What is in each data lake

Each lake is a directory containing the input tables plus expected union results. Every CSV pair (`dirty.csv` + `clean.csv`) has identical row counts , the clean version provides cell-level ground truth for the exact same rows.

```
{lake}/
  {table_name}/                     Input tables (one directory per table)
    dirty.csv                       Error-injected version
    clean.csv                       Ground-truth (same rows, same columns)
  expected_union_all/               Concat of all dirty/clean tables
    dirty.csv                       All rows from all dirty tables
    clean.csv                       All rows from all clean tables (aligned)
  expected_union/                   Deduped on dirty, clean aligned
    dirty.csv                       Dirty rows with exact duplicates removed
    clean.csv                       Clean rows at same indices (aligned)
  lineage.csv                       Per-row provenance (see below)
  info.json                         Row counts and metadata
```

**Table naming conventions:**

| Category | Table directory names | Example |
|----------|---------------------|---------|
| Disjoint | Source table names | `Permitted_Hotels_and_Motels/`, `Permitted_Swimming_Areas/` |
| Maximal overlap | Variant prefixes | `FD/`, `Typo/`, `NO/` |
| Partial overlap | Variant prefixes | `NO/`, `Typo/` |

**Deduplication rule:** Exact string duplicates are removed from `dirty.csv` first. The `clean.csv` in `expected_union/` keeps the clean rows at the same positions , it does **not** independently deduplicate the clean side. This ensures dirty and clean always have identical row counts, providing a reliable ground-truth mapping for every cell.

---

## Three Categories

### 1. Disjoint Union (0 % entity overlap)

Pairs two **different** tables from the same schema group. They share column names but contain completely different entities. Uses the FD variant.

**Example:** Hotels (131 rows) + Swimming Areas (538 rows) , same 13-column schema, zero row overlap.

```
{lake}/
  Louisville_Metro_KY_-_Permitted_Hotels_and_Motels/
    dirty.csv    (131 rows)
    clean.csv    (131 rows)
  Louisville_Metro_KY_-_Permitted_Swimming_Areas/
    dirty.csv    (538 rows)
    clean.csv    (538 rows)
  expected_union_all/
    dirty.csv    (669 rows)
    clean.csv    (669 rows)
  expected_union/
    dirty.csv    (669 rows , no duplicates to remove)
    clean.csv    (669 rows)
```

**10 pairs** generated (one per combinatorial pair within each schema group).

### 2. Maximal Overlap (100 % entity overlap)

Unions **the same table** from all three variants (FD, Typo, NO dirty data). Every entity appears three times, but with different surface values due to different error types.

Because the surface values differ across variants, a literal SQL UNION barely removes anything , even though the entities are identical.

**Example , Hotels (131 entities):**

```
{lake}/
  FD/       dirty.csv (131 rows), clean.csv (131 rows)
  Typo/     dirty.csv (131 rows), clean.csv (131 rows)
  NO/       dirty.csv (131 rows), clean.csv (131 rows)
  expected_union_all/
    dirty.csv  (393 rows)
    clean.csv  (393 rows)
  expected_union/
    dirty.csv  (379 rows , only 14 exact string duplicates removed)
    clean.csv  (379 rows , aligned to dirty)
```

On the dirty side, only 14 rows are string-identical across variants , UNION barely helps, even though all 131 entities are shared. This gap between 393 (UNION ALL) and 379 (UNION) shows how little exact dedup helps when the same entities have different surface errors.

### 3. Partial Overlap (25 %, 50 %, 75 %)

Uses **horizontal partitioning** + **cross-variant slicing** to create controlled overlap between two tables derived from the same source.

#### How it works

Given a source table with N rows and desired overlap ratio r:

```
Output table size:   S = floor(N / (2 - r))

Split N rows into 3 disjoint buckets:
  SHARED     rows 0 … round(r*S)-1       → same entities in both tables
  UNIQUE_A   rows round(r*S) … S-1        → only in Table A
  UNIQUE_B   rows S … S+S-round(r*S)-1    → only in Table B

Table A = SHARED (variant X dirty) + UNIQUE_A (variant X dirty)   → S rows
Table B = SHARED (variant Y dirty) + UNIQUE_B (variant Y dirty)   → S rows
```

For each source table, two distinct variants are **randomly assigned once** from {FD, Typo, NO} (seeded with `random.seed(42)` for reproducibility) and **reused across all overlap levels**. This ensures experiments on the same table at 25%, 50%, and 75% overlap are directly comparable , only the overlap ratio changes, not the variant pair or error characteristics. The shared rows reference the **same source row indices** but from **different variants**, so the overlapping entities have different surface values , just like real-world duplicates across data sources.

**Example , Hotels (131 source rows, variant pair: NO vs Typo):**

| Overlap | Table size (S) | Shared | Unique/table | UNION ALL | UNION (dirty) | Dirty dupes removed |
|---------|---------------|--------|-------------|-----------|--------------|---------------------|
| 25 % | 74 | 18 | 56 | 148 | 146 | 2 |
| 50 % | 87 | 44 | 43 | 174 | 170 | 4 |
| 75 % | 104 | 78 | 26 | 208 | 199 | 9 |

All three levels use the **same variant pair** (NO vs Typo for Hotels). The "dirty dupes removed" column is small , most shared entities have different surface values across variants, so a naive SQL UNION barely deduplicates them. Each lake's `expected_union/` directory contains both dirty.csv and clean.csv with the same row count, where the clean side provides the ground truth for every row in the deduped dirty result.

---

## Lineage Tracking

Every lake folder contains `lineage.csv` that traces every row in every output file back to its original source.

### Columns

| Column | Description | Example values |
|--------|-------------|---------------|
| `file` | Which part of the lake | `NO`, `Typo`, `FD`, `expected_union_all`, `expected_union` |
| `row_idx` | 0-based row index in that part | `0`, `1`, `2`, … |
| `source_table` | Original table name | `Louisville_Metro_KY_-_Permitted_Hotels_and_Motels` |
| `source_variant` | Error-type prefix | `FD`, `Typo`, `NO` |
| `source_data_ver` | Which CSV was used | `dirty` |
| `source_row_idx` | Row index in the **original** source CSV | `0`, `1`, `2`, … |
| `partition` | Role of this row in the union | `shared`, `unique_a`, `unique_b`, `all` |

The `file` column uses the table directory name , variant prefixes (`FD`, `Typo`, `NO`) for overlap datasets, or source table names for disjoint datasets.

### Identifying overlapping entities

Two rows across the two input tables represent the **same entity** if they share the same `source_row_idx` and `partition == "shared"`:

```python
import csv

path = "generated_union_datasets/partial_overlap_50/"
path += "Louisville_Metro_KY_-_Permitted_Hotels_and_Motels/lineage.csv"
with open(path, newline="") as f:
    reader = csv.DictReader(f)
    lineage = list(reader)

# Shared rows in each table (file column = variant name)
shared_a = [r for r in lineage if r["file"] == "NO" and r["partition"] == "shared"]
shared_b = [r for r in lineage if r["file"] == "Typo" and r["partition"] == "shared"]

# These share the same source_row_idx values , same entities, different variants
print(len(shared_a))  # 44 shared entities
print(set(r["source_variant"] for r in shared_a))  # {'NO'}
print(set(r["source_variant"] for r in shared_b))  # {'Typo'}
```

---

## Generation Script

```bash
python generate_union_datasets.py
```

Re-running regenerates everything from the source data in `Unified_Union_Exp_isolated/`. The script uses Python's `csv` module (no pandas) so every cell value is preserved as its original string , no type coercion, no silent NaN insertion, no float promotion. Duplicate detection for `expected_union/dirty.csv` is **exact string comparison** across all columns; `expected_union/clean.csv` is then aligned to the same row selection. Configuration at the top of the script:

```python
OVERLAP_RATIOS = [0.25, 0.50, 0.75]   # partial overlap levels to generate
```

---

## Summary Table

| Category | Lakes | Overlap | Variants used | Table names | Challenge |
|----------|-------|---------|--------------|-------------|-----------|
| Disjoint | 10 | 0 % | FD dirty + clean | Source table names | Different entities, same schema , should union cleanly |
| Maximal | 17 | 100 % | FD + Typo + NO dirty | `FD/`, `Typo/`, `NO/` | All entities shared but surface values differ |
| Partial 25 % | 17 | 25 % | Fixed pair from {FD, Typo, NO} | Variant names | Low overlap, cross-variant shared rows |
| Partial 50 % | 17 | 50 % | Same fixed pair | Variant names | Medium overlap, cross-variant shared rows |
| Partial 75 % | 17 | 75 % | Same fixed pair | Variant names | High overlap, cross-variant shared rows |
| **Total** | **78** | | | | |
