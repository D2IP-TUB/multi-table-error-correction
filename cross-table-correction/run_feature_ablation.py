"""
Feature ablation micro-benchmark runner.

Runs the full pipeline for every ablation configuration, keeping candidate
generation identical and only excluding disabled classifier features.

Ablation structure
──────────────────
  • baseline        – all features enabled
  • leave-one-out   – disable one feature group at a time (drop_*)
  • include-only    – enable exactly one group; disable all others (only_*)

Mode is controlled by env ``FEATURE_ABLATION_MODE``:
  • drop  – default; ``all_features`` + ``drop_<group>`` (backward compatible)
  • only  – ``all_features`` + ``only_<group>`` (single-group classifier)
  • both  – drop and only suites in one grid (more runs)
"""

import configparser
import os
import random
import subprocess
from collections import OrderedDict
from typing import Optional

# ─── Experiment parameters ────────────────────────────────────────────────────

EXECUTION_TIMES = 3

DATASET_BUDGETS = {
    "Quintet_3": [10, 22, 96, 173],
}

ZONING_STRATEGIES = ["rule_based"]

# Optional overrides
#   FEATURE_ABLATION_ZONING=clustering_based,rule_based
#   FEATURE_ABLATION_SEEDS=101,202,303
_ZONING_ENV = os.environ.get("FEATURE_ABLATION_ZONING", "rule_based").strip()
if _ZONING_ENV:
    ZONING_STRATEGIES = [
        z.strip() for z in _ZONING_ENV.split(",") if z.strip()
    ]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT_TEMPLATE = os.path.join(
    _SCRIPT_DIR,
    "results_feature_ablation_{zoning}",
)
CONFIG_FILE = os.path.join(_SCRIPT_DIR, "config", "config.ini")
PROJECT_DIR = _SCRIPT_DIR

# ─── Ablation definitions ────────────────────────────────────────────────────

ALL_GROUPS = [
    "value_based",
    "vicinity_based",
    "domain_based",
    "levenshtein",
    "pattern_based",
]

# drop | only | both  (override with env FEATURE_ABLATION_MODE)
FEATURE_ABLATION_MODE = os.environ.get("FEATURE_ABLATION_MODE", "drop").strip().lower()


def _build_random_states(execution_times: int) -> list[int]:
    """Build a seed pool used across all ablations in this invocation.

    Seeds are random per invocation but fixed within the run so that all
    ablation configs are compared on the same sampled splits.
    """
    manual = os.environ.get("FEATURE_ABLATION_SEEDS", "").strip()
    if manual:
        seeds = [int(x.strip()) for x in manual.split(",") if x.strip()]
        if len(seeds) != execution_times:
            raise ValueError(
                "FEATURE_ABLATION_SEEDS count must match EXECUTION_TIMES "
                f"({execution_times}); got {len(seeds)}"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("FEATURE_ABLATION_SEEDS must be unique")
        return seeds

    # Use SystemRandom for non-deterministic OS entropy.
    rng = random.SystemRandom()
    # Wide range reduces accidental collisions across different invocations.
    return rng.sample(range(1, 2_147_483_647), execution_times)


def active_feature_groups(zoning_strategy: str) -> list:
    """Classifier feature groups that exist for this zoning (pattern only for clustering)."""
    groups = [
        "value_based",
        "vicinity_based",
        "domain_based",
        "levenshtein",
    ]
    if zoning_strategy == "clustering_based":
        groups.append("pattern_based")
    return groups


def build_ablations(zoning_strategy: str, mode: Optional[str] = None):
    """
    Build ablation name -> disabled_feature_groups list.

    - drop: leave-one-out on disabled groups
    - only: keep one group; disable every other active group
    - both: union of drop and only (all_features once at the start)
    """
    mode = (mode or FEATURE_ABLATION_MODE).strip().lower()
    if mode not in ("drop", "only", "both"):
        raise ValueError(
            f"FEATURE_ABLATION_MODE must be drop, only, or both; got {mode!r}"
        )

    ablations = OrderedDict()
    ablations["all_features"] = []
    active = active_feature_groups(zoning_strategy)

    if mode in ("drop", "both"):
        ablations["drop_value_based"] = ["value_based"]
        ablations["drop_vicinity_based"] = ["vicinity_based"]
        ablations["drop_domain_based"] = ["domain_based"]
        ablations["drop_levenshtein"] = ["levenshtein"]
        if zoning_strategy == "clustering_based":
            ablations["drop_pattern_based"] = ["pattern_based"]

    if mode in ("only", "both"):
        for g in active:
            disabled = [x for x in active if x != g]
            ablations[f"only_{g}"] = disabled

    return ablations


# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_config(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def update(cfg, section, key, value):
    cfg.set(section, key, str(value))


def save_config(cfg, path):
    with open(path, "w") as f:
        cfg.write(f)


# ─── Main loop ────────────────────────────────────────────────────────────────

def run_ablation():
    cfg = read_config(CONFIG_FILE)
    # Random per invocation, fixed within this run for paired comparisons.
    random_states = _build_random_states(EXECUTION_TIMES)
    print(f"Seed pool for this run: {random_states}")

    total_runs = 0
    for zoning in ZONING_STRATEGIES:
        total_runs += (
            EXECUTION_TIMES
            * sum(len(b) for b in DATASET_BUDGETS.values())
            * len(build_ablations(zoning))
        )

    current_run = 0

    for zoning_strategy in ZONING_STRATEGIES:
        output_root = OUTPUT_ROOT_TEMPLATE.format(zoning=zoning_strategy)
        os.makedirs(output_root, exist_ok=True)
        ablations = build_ablations(zoning_strategy)

        print(f"\n{'#'*70}")
        print(
            f"# Zoning strategy: {zoning_strategy}  ({len(ablations)} ablation configs, "
            f"mode={FEATURE_ABLATION_MODE})"
        )
        print(f"# Output: {output_root}")
        print(f"{'#'*70}")

        for exec_idx in range(1, EXECUTION_TIMES + 1):
            seed = random_states[exec_idx - 1]

            for dataset, budgets in DATASET_BUDGETS.items():
                update(cfg, "DIRECTORIES", "tables_dir", dataset)
                update(cfg, "ZONING", "strategy", zoning_strategy)

                for budget in budgets:
                    for ablation_name, disabled_groups in ablations.items():
                        current_run += 1
                        tag = f"{dataset}_{exec_idx}_{budget}_{ablation_name}"

                        _disabled_set = set(disabled_groups)
                        _active = active_feature_groups(zoning_strategy)
                        active_line = (
                            f"  active (classifier): {', '.join(g for g in _active if g not in _disabled_set)}\n"
                            if disabled_groups
                            else ""
                        )
                        print(
                            f"\n{'='*70}\n"
                            f"[{current_run}/{total_runs}]  {zoning_strategy} | {tag}\n"
                            f"  disabled: {disabled_groups or '(none – baseline)'}\n"
                            f"{active_line}"
                            f"  random_state (from config iter): {seed}\n"
                            f"{'='*70}"
                        )

                        update(cfg, "EXPERIMENT", "random_state", seed)
                        update(cfg, "LABELING", "labeling_budget", budget)
                        update(
                            cfg,
                            "TRAINING",
                            "disabled_feature_groups",
                            ", ".join(disabled_groups),
                        )
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

    print(f"\nAll {total_runs} ablation runs finished.")


if __name__ == "__main__":
    run_ablation()
