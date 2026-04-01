"""
Rule-based pattern-enforcement ablation runner.

Compares three invalid-zone pattern-enforcement modes:
  - check: validate against ground truth, accept only if correct
  - always_accept: always accept top enforced value
  - disabled: skip pattern enforcement stage
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

ZONING_STRATEGY = "rule_based"
PATTERN_ENFORCEMENT_MODES = ["check", "always_accept", "disabled"]

# Optional override:
#   PE_ABLATION_SEEDS=101,202,303

def _build_random_states(execution_times: int) -> list:
    """Build a seed pool that is random per invocation but fixed within a run.

    All ablation configs in one invocation share the same seeds, so comparisons
    are paired and fair.
    """
    manual = os.environ.get("PE_ABLATION_SEEDS", "").strip()
    if manual:
        seeds = [int(x.strip()) for x in manual.split(",") if x.strip()]
        if len(seeds) != execution_times:
            raise ValueError(
                "PE_ABLATION_SEEDS count must match EXECUTION_TIMES "
                f"({execution_times}); got {len(seeds)}"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("PE_ABLATION_SEEDS must be unique")
        return seeds

    # SystemRandom uses OS entropy — truly random, not seeded by Python state.
    rng = random.SystemRandom()
    return rng.sample(range(1, 2_147_483_647), execution_times)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(
    _SCRIPT_DIR,
    "results_pattern_enforcement_ablation_rule_based",
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
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    total_runs = (
        EXECUTION_TIMES
        * sum(len(b) for b in DATASET_BUDGETS.values())
        * len(PATTERN_ENFORCEMENT_MODES)
    )
    current_run = 0

    print(f"\n{'#'*70}")
    print("# Pattern-enforcement ablation (rule_based)")
    print(f"# Output: {OUTPUT_ROOT}")
    print(f"{'#'*70}")

    for exec_idx in range(1, EXECUTION_TIMES + 1):
        seed = random_states[exec_idx - 1]

        for dataset, budgets in DATASET_BUDGETS.items():
            update(cfg, "DIRECTORIES", "tables_dir", dataset)
            update(cfg, "ZONING", "strategy", ZONING_STRATEGY)

            for budget in budgets:
                for mode in PATTERN_ENFORCEMENT_MODES:
                    current_run += 1
                    suffix = f"pattern_enforcement_{mode}"
                    tag = f"{dataset}_{exec_idx}_{budget}_{suffix}"

                    print(
                        f"\n{'='*70}\n"
                        f"[{current_run}/{total_runs}]  {ZONING_STRATEGY} | {tag}\n"
                        f"  correction.pattern_enforcement_mode: {mode}\n"
                        f"  random_state (from config iter): {seed}\n"
                        f"{'='*70}"
                    )

                    update(cfg, "EXPERIMENT", "random_state", seed)
                    update(cfg, "LABELING", "labeling_budget", budget)
                    update(cfg, "TRAINING", "disabled_feature_groups", "")
                    update(
                        cfg,
                        "CORRECTION",
                        "pattern_enforcement_mode",
                        mode,
                    )
                    # Keep legacy boolean in sync for older code paths.
                    update(
                        cfg,
                        "CORRECTION",
                        "enable_pattern_enforcement",
                        "false" if mode == "disabled" else "true",
                    )
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

    print(f"\nAll {total_runs} pattern-enforcement ablation runs finished.")


if __name__ == "__main__":
    run_ablation()
