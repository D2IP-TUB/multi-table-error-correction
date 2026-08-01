# Unionable tables (DGov)

Starting from 17 real-world government datasets (each with FD / Typo / NO error variants), we generate controlled union lakes at varying entity-overlap levels, with per-row provenance.

Overlapping entities always come from *different* error-type variants, so shared rows have different surface values. There are no artificial exact-duplicate overlaps.

Every table has `dirty.csv` and `clean.csv` with identical row counts (cell-level ground truth).

## Source data

```
support_material/initial_dgov_tables/
  DGov_{FD,Typo,NO}_{TableName}/
    clean.csv
    dirty.csv
    clean_changes.csv
```

Generation scripts live in `support_material/scripts/` (entry: `generate_union_datasets.py`). Partitioned intermediates are under `support_material/partitioned_init_tables/`.

| Prefix | Meaning |
|--------|---------|
| **FD** | Functional-dependency violation errors |
| **Typo** | Typographical errors |
| **NO** | Numeric outlier errors |

## Lakes used in experiments

```
union_datasets_used_in_exp/
  isolated/                              # per-table baselines
  disjoint_with_duplicates/              # 0% entity overlap (UNION ALL style)
  disjoint_without_duplicates/
  maximal_overlap_with_duplicates/       # 100% entity overlap
  maximal_overlap_without_duplicates/
  partial_overlap_{25,50,75}_with_duplicates/
  partial_overlap_{25,50,75}_without_duplicates/
```

Each lake directory contains input tables plus expected union results (`expected_union_all/`, `expected_union/` when present), `lineage.csv`, and `info.json`.

### Categories

1. **Disjoint (0%)** — two different tables from the same schema group (FD variant).
2. **Maximal (100%)** — same table from FD, Typo, and NO variants (surface values differ).
3. **Partial (25/50/75%)** — horizontal partition + cross-variant shared rows (`random.seed(42)` for variant assignment).

Regenerate from sources:

```bash
cd support_material/scripts
python generate_union_datasets.py
```

See script headers for overlap ratios and output layout.
