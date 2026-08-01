# UniClean experiment configuration

Working directory for runners: `uniclean_cleaners/`.

## Modes

| Mode | Flag | Cleaners |
|------|------|----------|
| **UniClean-ALL** | `--mode all` | Full type-specific cleaners (Pattern, Date, Number, Outlier, …) plus `AttrRelation` FDs where configured |
| **UniClean-FD** | `--mode fd` | Only `AttrRelation` (functional dependency) cleaners |

On **real lakes** and controlled join/union lakes we use FD-only rules derived from `holo_constraints.txt` (same FD sets as HoloClean / Horizon). Pattern/date cleaners are not auto-derived for those lakes.

## Quintet

```bash
cd uniclean_cleaners
python run_quintet3.py --mode all --lake_dir ../../../datasets/unrelated_tables/Quintet
python run_quintet3.py --mode fd  --lake_dir ../../../datasets/unrelated_tables/Quintet
```

Per-table cleaner lists are in `main_quintet3.py` (`OFFICIAL_CLEANERS`, `MOVIES_CLEANERS`).

## Data lakes

```bash
cd uniclean_cleaners
python run_lake.py --lake_dir /path/to/lake
```

Each table directory must contain `dirty.csv`, `clean.csv`, and `holo_constraints.txt`.

## Environment

```bash
# from baselines/UniClean/
./setup_env.sh
source ./activate_uniclean.sh
```

Evaluation helpers: `evaluate_result.py`, `evaluate_uniclean_by_error_type.py`, majority-voting scripts in this folder.
