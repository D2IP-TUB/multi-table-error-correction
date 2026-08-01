# UniClean

Official UniClean cleaners adapted for this paper’s lakes and Quintet runs.

- Experiment configs (UniClean-ALL / UniClean-FD, sampling): **[baseline_code.md](baseline_code.md)**
- Cleaner library and runners: `uniclean_cleaners/`
- Setup: `setup_env.sh`, `activate_uniclean.sh`

```bash
./setup_env.sh
source ./activate_uniclean.sh
cd uniclean_cleaners
python run_quintet3.py --mode fd --lake_dir ../../../datasets/unrelated_tables/Quintet
python run_lake.py --lake_dir ../../../datasets/real_lakes/open_data_uk
```
