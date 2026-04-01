"""
Ablation: cluster sampling strategy (column_coverage vs kmeans_pp).

Only affects clustering_based zoning: after KMeans on unusualness features,
either the default column-coverage + random fill, or k-means++-style selection
within each cluster (D² to nearest chosen point in scaled space).

Rule-based runs are unchanged (this script runs clustering_based only).
"""

import configparser
import os
import random
import subprocess
from typing import Optional

EXECUTION_TIMES = 3

DATASET_BUDGETS = {
    "Quintet_3": [10, 22, 96, 173],
}

# Only clustering_based uses cluster_sampling_strategy
ZONING_STRATEGY = "clustering_based"

# Optional overrides:
#   CS_ABLATION_SEEDS=101,202,303

def _build_random_states(execution_times: int) -> list:
    """Build a seed pool that is random per invocation but fixed within a run.

    All ablation configs in one invocation share the same seeds, so comparisons
    are paired and fair.
    """
    manual = os.environ.get("CS_ABLATION_SEEDS", "").strip()
    if manual:
        seeds = [int(x.strip()) for x in manual.split(",") if x.strip()]
        if len(seeds) != execution_times:
            raise ValueError(
                "CS_ABLATION_SEEDS count must match EXECUTION_TIMES "
                f"({execution_times}); got {len(seeds)}"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("CS_ABLATION_SEEDS must be unique")
        return seeds

    # SystemRandom uses OS entropy — truly random, not seeded by Python state.
    rng = random.SystemRandom()
    return rng.sample(range(1, 2_147_483_647), execution_times)

SAMPLING_STRATEGIES = [
    "column_coverage",  # column coverage + random within cluster
    "kmeans_pp",  # k-means++ seeding within each cluster (same inter-cluster budget)
]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(
    _SCRIPT_DIR,
    "results_cluster_sampling_ablation_clustering_based",
)
CONFIG_FILE = os.path.join(_SCRIPT_DIR, "config", "config.ini")
PROJECT_DIR = _SCRIPT_DIR


def read_config(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def update(cfg, section, key, value):
    cfg.set(section, key, str(value))


def save_config(cfg, path):
    with open(path, "w") as f:
        cfg.write(f)


def run_ablation():
    cfg = read_config(CONFIG_FILE)
    random_states = _build_random_states(EXECUTION_TIMES)
    print(f"Seed pool for this run: {random_states}")

    total_runs = (
        EXECUTION_TIMES
        * sum(len(b) for b in DATASET_BUDGETS.values())
        * len(SAMPLING_STRATEGIES)
    )

    current_run = 0
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print(f"\n{'#'*70}")
    print("# Cluster sampling ablation (clustering_based only)")
    print(f"# Output: {OUTPUT_ROOT}")
    print(f"# Strategies: {SAMPLING_STRATEGIES}")
    print(f"{'#'*70}")

    for exec_idx in range(1, EXECUTION_TIMES + 1):
        seed = random_states[exec_idx - 1]

        for dataset, budgets in DATASET_BUDGETS.items():
            update(cfg, "DIRECTORIES", "tables_dir", dataset)
            update(cfg, "ZONING", "strategy", ZONING_STRATEGY)

            for budget in budgets:
                for sampling_strategy in SAMPLING_STRATEGIES:
                    current_run += 1
                    tag = f"{dataset}_{exec_idx}_{budget}_sampling_{sampling_strategy}"

                    print(
                        f"\n{'='*70}\n"
                        f"[{current_run}/{total_runs}]  {tag}\n"
                        f"  cluster_sampling_strategy={sampling_strategy}\n"
                        f"  random_state (from config iter): {seed}\n"
                        f"{'='*70}"
                    )

                    update(cfg, "EXPERIMENT", "random_state", seed)
                    update(cfg, "LABELING", "labeling_budget", budget)
                    update(
                        cfg,
                        "SAMPLING",
                        "cluster_sampling_strategy",
                        sampling_strategy,
                    )
                    update(cfg, "TRAINING", "disabled_feature_groups", "")

                    update(
                        cfg,
                        "DIRECTORIES",
                        "output_dir",
                        os.path.join(OUTPUT_ROOT, f"output_{tag}"),
                    )
                    update(
                        cfg,
                        "DIRECTORIES",
                        "logs_dir",
                        os.path.join(OUTPUT_ROOT, f"logs_{tag}"),
                    )

                    save_config(cfg, CONFIG_FILE)

                    try:
                        subprocess.run(
                            ["python", "-m", "main"],
                            cwd=PROJECT_DIR,
                            check=True,
                        )
                    except subprocess.CalledProcessError as exc:
                        print(f"  FAILED (exit {exc.returncode}) — continuing…")

    print(f"\nAll {total_runs} cluster-sampling ablation runs finished.")


if __name__ == "__main__":
    run_ablation()
