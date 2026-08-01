# Baran

Baran (Error-Correction-at-Scale / Raha) experiment runners used in this paper.

## Entry points

```
Error-Correction-at-Scale/benchmarks/Baran_Experiments/raha/raha/ecs_run_experiments/
```

- `run_experiments.py` — main Hydra-driven runner  
- `baran_enough_labels_lake.py` / `baran_not_enough_labels_lake.py` — lake labeling budgets  
- Hydra configs: `hydra_configs/shared.yaml`, `hydra_configs/standard.yaml`

Stats helpers: `../get_baran_stats/`.

Configure dataset paths in the Hydra YAMLs (prefer repo-relative paths under `datasets/`).
