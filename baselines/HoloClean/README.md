# HoloClean

HoloClean plus lake runners and HyFD-based FD extraction (`HoloFDExtractor/`).

## Setup

```bash
conda env create -f env.yml
conda activate holoclean   # name as in env.yml
```

## Run

```bash
# paths in config/base.yaml are relative to the repository root
python run_holoclean.py
python run_all_lakes.py
python run_baselines.py
```

Evaluation: `evaluate_repair.py`, `evaluate_lake.py`, `evaluate_holoclean_lake.py`, majority-voting helpers.

Default configs use the same FD constraints (`holo_constraints.txt`) as Horizon / UniClean-FD.
