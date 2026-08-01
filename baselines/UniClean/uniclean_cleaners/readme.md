# UniClean cleaners

Experiment configs (UniClean-ALL / UniClean-FD, lakes, sampling): see [../baseline_code.md](../baseline_code.md).

```bash
# Quintet — UniClean-ALL / UniClean-FD
python run_quintet3.py --mode all --lake_dir /path/to/Quintet
python run_quintet3.py --mode fd  --lake_dir /path/to/Quintet

# Data lake (FD-only from holo_constraints.txt)
python run_lake.py --lake_dir /path/to/lake
```

Main entry points: `main_quintet3.py`, `main.py`, `run_quintet3.py`, `run_lake.py`.  
Library code: `SampleScrubber/`, `Clean.py`.
