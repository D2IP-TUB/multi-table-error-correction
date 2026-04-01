"""
Negative-pruning ablation runner.

Runs the full pipeline with training negative-pruning ON vs OFF, for every
labeling budget and both zoning strategies (rule_based, clustering_based).

Each run uses all classifier features (no disabled_feature_groups).
"""

import configparser
import os
import random
import subprocess
from typing import Optional

# ─── Experiment parameters (match run_feature_ablation.py) ───────────────────

EXECUTION_TIMES = 3

DATASET_BUDGETS = {
    "Quintet_3": [10, 22, 96, 173],
}

ZONING_STRATEGIES = ["rule_based", "clustering_based"]

# Optional overrides:
#   NP_ABLATION_ZONING=clustering_based,rule_based
#   NP_ABLATION_SEEDS=101,202,303
_ZONING_ENV = os.environ.get("NP_ABLATION_ZONING", "").strip()
if _ZONING_ENV:
    ZONING_STRATEGIES = [z.strip() for z in _ZONING_ENV.split(",") if z.strip()]

# Toggle negative sample pruning during training
NEGATIVE_PRUNING_SETTINGS = [False, True]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT_TEMPLATE = os.path.join(
    _SCRIPT_DIR,
    "results_negative_pruning_ablation_{zoning}",
)
CONFIG_FILE = os.path.join(_SCRIPT_DIR, "config", "config.ini")
PROJECT_DIR = _SCRIPT_DIR


def _build_random_states(execution_times: int) -> list:
    """Build a seed pool that is random per invocation but fixed within a run.

    All ablation configs in one invocation share the same seeds, so comparisons
    are paired and fair.
    """
    manual = os.environ.get("NP_ABLATION_SEEDS", "").strip()
    if manual:
        seeds = [int(x.strip()) for x in manual.split(",") if x.strip()]
        if len(seeds) != execution_times:
            raise ValueError(
                "NP_ABLATION_SEEDS count must match EXECUTION_TIMES "
                f"({execution_times}); got {len(seeds)}"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("NP_ABLATION_SEEDS must be unique")
        return seeds

    # SystemRandom uses OS entropy — truly random, not seeded by Python state.
    rng = random.SystemRandom()
    return rng.sample(range(1, 2_147_483_647), execution_times)


def read_config(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def update(cfg, section, key, value):
    cfg.set(section, key, str(value))


def save_config(cfg, path):
    with open(path, "w") as f:
        cfg.write(f)


def bool_to_ini(b: bool) -> str:
    return "true" if b else "false"


def run_ablation():
    cfg = read_config(CONFIG_FILE)
    random_states = _build_random_states(EXECUTION_TIMES)
    print(f"Seed pool for this run: {random_states}")

    total_runs = (
        EXECUTION_TIMES
        * sum(len(b) for b in DATASET_BUDGETS.values())
        * len(ZONING_STRATEGIES)
        * len(NEGATIVE_PRUNING_SETTINGS)
    )

    current_run = 0

    for zoning_strategy in ZONING_STRATEGIES:
        output_root = OUTPUT_ROOT_TEMPLATE.format(zoning=zoning_strategy)
        os.makedirs(output_root, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"# Zoning strategy: {zoning_strategy}")
        print(f"# Output: {output_root}")
        print(f"{'#'*70}")

        for exec_idx in range(1, EXECUTION_TIMES + 1):
            seed = random_states[exec_idx - 1]

            for dataset, budgets in DATASET_BUDGETS.items():
                update(cfg, "DIRECTORIES", "tables_dir", dataset)
                update(cfg, "ZONING", "strategy", zoning_strategy)

                for budget in budgets:
                    for neg_prune in NEGATIVE_PRUNING_SETTINGS:
                        current_run += 1
                        tag_suffix = "negprune_on" if neg_prune else "negprune_off"
                        tag = f"{dataset}_{exec_idx}_{budget}_{tag_suffix}"

                        print(
                            f"\n{'='*70}\n"
                            f"[{current_run}/{total_runs}]  {zoning_strategy} | {tag}\n"
                            f"  negative_pruning_enabled: {neg_prune}\n"
                            f"  random_state (from config iter): {seed}\n"
                            f"{'='*70}"
                        )

                        update(cfg, "EXPERIMENT", "random_state", seed)
                        update(cfg, "LABELING", "labeling_budget", budget)
                        update(cfg, "TRAINING", "negative_pruning_enabled", bool_to_ini(neg_prune))
                        # Baseline: no feature-group ablation
                        update(cfg, "TRAINING", "disabled_feature_groups", "")
                        update(
                            cfg,
                            "DIRECTORIES",
                            "output_dir",
                            os.path.join(output_root, f"output_{tag}"),
                        )
                        update(
                            cfg,
                            "DIRECTORIES",
                            "logs_dir",
                            os.path.join(output_root, f"logs_{tag}"),
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

    print(f"\nAll {total_runs} negative-pruning ablation runs finished.")


if __name__ == "__main__":
    run_ablation()
