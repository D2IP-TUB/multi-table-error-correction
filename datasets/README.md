# Datasets

Benchmark data for multi-table error correction.

## Contents

| Folder | Role |
|--------|------|
| `unrelated_tables/Quintet/` | Flights, Hospital, Beers, Movies, Rayyan (isolated) |
| `joinable_tables/` | Flights (clean / noisy keys), Soccer (non-unique keys); each has `isolated/` and `joined/` |
| `unionable_tables/` | DGov-derived union lakes at controlled overlap; see `unionable_tables/README.md` |
| `real_lakes/open_data_uk/` | 93 UK Open Data tables (dirty/clean + FDs) |
| `real_lakes/open_data_uk_merged_*` | Pre-merged OpenData-UK variants (set / multiset union) |

## Layout sketch

```
datasets/
├── unrelated_tables/Quintet/{flights,hospital,beers,movies,rayyan}/
│   ├── clean.csv
│   └── dirty.csv
├── joinable_tables/
│   ├── flights_without_key_errors/{isolated,joined}/
│   ├── flights_with_join_key_error/{isolated,joined}/
│   └── soccer/{isolated,joined}/   # joined CSVs may be .zip
├── unionable_tables/
│   ├── support_material/           # sources + generation scripts
│   └── union_datasets_used_in_exp/ # lakes used in experiments
└── real_lakes/open_data_uk/<table>/
    ├── clean.csv
    ├── dirty.csv
    ├── holo_constraints.txt
    └── fds.txt
```
