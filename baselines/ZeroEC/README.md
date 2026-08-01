# ZeroEC

LLM-based error correction baseline (paper code under `ZeroEC/` and zero-shot variant under `ZeroEC-0-Shot/`).

## Run

```bash
cd ZeroEC
# set API keys / model names in correction.py (or your env)
python run_batch_experiments.py --dry-run
python run_batch_experiments.py
python run_experiments.py --evaluate-only
```

Point dataset and output directories at this repo’s `datasets/` and a local `results/` folder (do not hard-code machine-specific absolute paths).

See also `ZeroEC/README.md` (upstream) and `ZeroEC/EXPERIMENTS_README.md` for batch options.
