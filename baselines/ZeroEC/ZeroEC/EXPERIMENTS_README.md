# ZeroEC batch experiments

Scripts for running and evaluating ZeroEC across multiple tables.

## Scripts

| Script | Role |
|--------|------|
| `run_experiments.py` | Evaluate existing results or plan runs (`--evaluate-only`) |
| `run_batch_experiments.py` | Batch runner over a lake (`--dry-run` to preview) |
| `correction.py` | Core ZeroEC correction (set model / API keys here) |

## Configuration

Edit paths in `run_batch_experiments.py` / `run_experiments.py` to point at this repository, for example:

- Datasets: `../../../datasets/unrelated_tables/Quintet/`
- Results: a local `results/zeroec/` directory under the repo root

Human-repair budgets typically: `[1, 2, 3, 5, 8, 10]`.

## Outputs

Per run (under your results root): `corrections.csv`, `output.txt`, `time_cost.txt`, and related logs. Aggregated CSVs are written by the batch/eval scripts.
