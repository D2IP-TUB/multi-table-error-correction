# Quick Reference: Datasets for Multi-Table Error Correction

## Dataset Descriptions

### Unrelated Tables (Quintet)
- **Characteristics:** 5 independent datasets, different domains
- **Best For:** Testing systems on non-related tables

### Joinable Tables
- **Use:** Study impact of joins on correction
- **Characteristics:** Tables decomposed by functional dependencies
- **Variants:** 
  - Address (clean keys)- Proprietary
  - Flights (clean + noisy keys)
  - Soccer (non-unique keys)

### Unionable Tables (DGov)
- **Use:** Study impact of unions
- **Characteristics:** 363 tables
- **Variants:** Disjoint, 25%, 50%, 75%, 100% overlap (w/ & w/o duplicates)

### Real-World Data Lakes
- **Use:** Evaluate on real-world data lakes
- **Included:** OpenData-UK (93 tables, 5.6M cells)
- **Proprietary:** MIT-DW (86 tables, 2.0M cells) 
- **Characteristics:** Unknown relationships, mixed error types

## File Organization

```
datasets/
├── README.md                           # Main documentation (this file)
├── unrelated_tables/
│   ├── README.md                       # Unrelated details
│   └── Quintet/
│       ├── flights/
│       │   ├── clean.csv               # Ground truth
│       │   ├── dirty.csv               # With errors
│       │   └── holo_constraints.txt    # Constraints
│       ├── hospital/
│       ├── beers/
│       ├── movies/
│       └── rayyan/
├── joinable_tables/
│   ├── README.md                       # Joinable details
│   ├── flights_without_key_errors/    # Clean keys
│   │   ├── isolated/
│   │   └── joined/
│   ├── flights_with_join_key_error/   # 10% key noise
│   │   ├── isolated/
│   │   └── joined/
│   └── soccer/                         # Non-unique keys
│       ├── isolated/
│       └── joined/
├── unionable_tables/
│   ├── README.md                       # Unionable details
│   ├── support_material/
│   │   └── scripts/
│   │       ├── generate_union_datasets.py
│   │       ├── create_partitioned_base.py
│   │       └── ...
│   └── union_datasets_used_in_exp/
│       ├── isolated/                   # Baseline
│       ├── disjoint_with_duplicates/   # 0% overlap, UNION ALL
│       ├── partial_overlap_25_with_duplicates/
│       ├── partial_overlap_50_with_duplicates/
│       ├── partial_overlap_75_with_duplicates/
│       └── maximal_overlap_with_duplicates/  # 100%
|       |__ ...
└── real_lakes/
    ├── README.md                       # Real lakes details
    └── open_data_uk/                   # 93 UK government tables
        ├── UK_CSV0000000000000127/
        │   ├── clean.csv
        │   ├── dirty.csv
        │   ├── clean_changes.csv       # Changes log
        │   ├── holo_constraints.txt
        │   └── fds.txt
        └── ...
```