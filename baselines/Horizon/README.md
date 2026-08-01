# Horizon

Java Horizon build (author-provided) with wrappers for this paper’s lakes.

Upstream: https://github.com/D2IP-TUB/Horizon

## Run

```bash
# convert holo_constraints → fds.txt where needed, then:
python run_horizon_all.py
# or:
./run_horizon_collections.sh
```

Per-error-type / lineage runners: `run_horizon_by_error_type.py`, `run_horizon_by_error_type_lineage.py`.  
Evaluation: `evaluate_repair.py`, `evaluate_by_error_type.py`, `evaluate_per_original_table.py`.

FDs match the shared `holo_constraints.txt` / `fds.txt` used by HoloClean and UniClean-FD.
