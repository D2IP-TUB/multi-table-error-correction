# ZeroEC Batch Experiments

This directory contains scripts for running batch experiments with ZeroEC on multiple datasets with varying configurations.

## Scripts

### 1. `run_experiments.py`
A script that can evaluate existing ZeroEC results or set up new experiments.

**Features:**
- Calculate TP, FP, FN, Precision, Recall, F1 scores
- Extract execution times from results
- Aggregate results across datasets and configurations
- Evaluate existing results without re-running experiments

**Usage:**
```bash
# Evaluate existing results
python run_experiments.py --evaluate-only

# Plan new experiments (without running)
python run_experiments.py
```

### 2. `run_batch_experiments.py`
Comprehensive batch experiment runner that creates temporary configured scripts and executes them.

**Features:**
- Automatically runs ZeroEC on all Quintet_3 datasets
- Tests multiple `human_repair_num` values: [1, 2, 3, 5, 8, 10]
- Creates temporary configured correction.py scripts for each experiment
- Calculates and aggregates metrics automatically
- Generates summary statistics and per-dataset results

**Usage:**
```bash
# Run all experiments
python run_batch_experiments.py

# Dry run (show plan without executing)
python run_batch_experiments.py --dry-run
```

### 3. `correction_cli.py`
Template for command-line wrapper (requires refactoring of correction.py).

## Configuration

The scripts are pre-configured for:
- **Datasets**: `/home/fatemeh/LakeCorrectionBench/datasets/Quintet_3/`
- **Results**: `/home/fatemeh/LakeCorrectionBench/ZeroEC/results/Quintet_3/`
- **Human repair nums**: [1, 2, 3, 5, 8, 10]

You can modify these in the script files if needed.

## Output

### Per-Experiment Output
Each experiment creates a directory: `results/Quintet_3/{dataset}/human_repair_{num}/`

Contains:
- `corrections.csv` - Corrected data
- `output.txt` - Detailed execution log
- `time_cost.txt` - Timing breakdown
- `specific_examples.txt` - Auto-generated examples
- `codes.txt` - Generated correction codes
- `fds.txt` - Discovered functional dependencies

### Aggregated Results
- `zeroec_quintet3_batch_results.csv` - All experiment results
- `zeroec_quintet3_batch_results_summary.csv` - Summary statistics

## Metrics Calculated

- **TP (True Positives)**: Errors correctly fixed
- **FP (False Positives)**: Incorrect changes made
- **FN (False Negatives)**: Errors not fixed
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: Harmonic mean of Precision and Recall
- **Execution Time**: Total time in seconds

## Example Workflow

### 1. Run all experiments
```bash
cd /home/fatemeh/LakeCorrectionBench/ZeroEC
python run_batch_experiments.py
```

This will:
1. Create 30 experiments (5 datasets × 6 human_repair_num values)
2. Run each experiment sequentially
3. Calculate metrics for each
4. Generate aggregated results

### 2. Evaluate existing results
If you've already run some experiments manually:

```bash
python run_experiments.py --evaluate-only
```

This will scan the results directory and aggregate any existing results.
