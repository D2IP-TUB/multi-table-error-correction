#!/usr/bin/env python3
"""
Run the BLEND merge pipeline on correction lake corpora (index + merge + annotate).

Corpora:
  corr-datasets/mit_dw_lake_exp/<table>/dirty.csv
  corr-datasets/open_data_uk_93/<table>/dirty.csv

GT mode also uses clean.csv in each table directory.
Detected mode reads per-table detection CSVs from corr-results-budget-2/<corpus>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from error_cells import clear_error_profile_cache
from index_tables import index_tables
from merge_tables import merge_tables
from recreate_as_strings import recreate_merged_tables_as_strings
from recreate_as_strings_union import recreate_merged_tables_as_strings_union

_BASE = Path(__file__).resolve().parent
CORPORA = ("mit_dw_lake_exp", "open_data_uk_93")
DEFAULT_DETECTED_RESULTS_SUBDIR = "corr-results-budget-2"
RECREATE_MODES = ("set_union", "bag")
RECREATE_OUTPUT_DIRS = {
    "set_union": "merged_strings_default_set_union",
    "bag": "merged_strings_default_union_all",
}
MERGE_PRIORITIES = ("union", "join")
VALIDATION_STRATEGIES = ("fd", "distribution", "fd_and_distribution")
VALIDATION_SCOPES = ("all", "join", "union")


def _results_root(merge_priority: str) -> Path:
    priority = merge_priority.strip().lower()
    if priority not in MERGE_PRIORITIES:
        raise ValueError(f"Unknown merge priority {merge_priority!r}; expected one of {MERGE_PRIORITIES}")
    return _BASE / "results" / f"{priority}_first"


def configure(
    corpus: str,
    *,
    error_mode: str = "GT",
    detected_run: int = 1,
    detected_results_subdir: str = DEFAULT_DETECTED_RESULTS_SUBDIR,
    merge_priority: str = "union",
    merge_validation: bool = False,
    validation_scope: str = "all",
    validation_strategy: str = "fd",
) -> None:
    if corpus not in CORPORA:
        raise ValueError(f"Unknown corpus {corpus!r}; expected one of {CORPORA}")

    config.CORPUS = corpus
    config.ERROR_MODE = error_mode.upper()
    config.MERGE_PRIORITY = merge_priority.strip().lower()
    config.MERGE_VALIDATION = bool(merge_validation)
    config.VALIDATION_SCOPE = validation_scope.strip().lower()
    config.VALIDATION_STRATEGY = validation_strategy.strip().lower()
    config.RESULTS_ROOT = _results_root(config.MERGE_PRIORITY)
    config.DIR_PATH = _BASE / "corr-datasets" / corpus

    if config.ERROR_MODE == "GT":
        config.DETECTED_ERRORS_DIR = None
        config.DETECTED_ERRORS_FILENAME = "detected_errors.csv"
        out_root = config.RESULTS_ROOT / "output_gt_mode" / corpus / "gt"
    elif config.ERROR_MODE == "DETECTED":
        config.DETECTED_ERRORS_DIR = _BASE / detected_results_subdir / corpus
        config.DETECTED_ERRORS_FILENAME = f"detected_errors_run_{detected_run}.csv"
        out_root = config.RESULTS_ROOT / "output" / corpus / f"detected_run_{detected_run}"
    else:
        raise ValueError(f"Unknown error mode {error_mode!r}; expected 'GT' or 'DETECTED'")

    config.MERGED_PATH = out_root / "merged"
    config.DB_PATH = out_root / "blend_index.duckdb"
    config.TRACKER_PATH = config.MERGED_PATH / "tracker.json"


def normalize_recreate_modes(recreate: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(recreate, str):
        if recreate == "both":
            return RECREATE_MODES
        if recreate not in RECREATE_MODES:
            raise ValueError(f"Unknown recreate mode {recreate!r}; expected one of {RECREATE_MODES + ('both',)}")
        return (recreate,)
    modes = tuple(recreate)
    unknown = [mode for mode in modes if mode not in RECREATE_MODES]
    if unknown:
        raise ValueError(f"Unknown recreate mode(s) {unknown}; expected {RECREATE_MODES}")
    return modes


def run_recreate_phases(recreate: str | tuple[str, ...] = "both") -> list[str]:
    output_dirs: list[str] = []
    output_root = config.RESULTS_ROOT if config.RESULTS_ROOT is not None else Path('.')
    for mode in normalize_recreate_modes(recreate):
        if mode == "set_union":
            print("Phase 3: recreate_as_strings_union (set union)")
            recreate_merged_tables_as_strings_union()
        else:
            print("Phase 3: recreate_as_strings (bag / union all)")
            recreate_merged_tables_as_strings()
        output_dirs.append(
            str(output_root / RECREATE_OUTPUT_DIRS[mode] / config.CORPUS / config.MERGED_PATH.name)
        )
        print()
    return output_dirs


def run_pipeline(*, script_name: str, recreate: str | tuple[str, ...] = "both") -> None:
    clear_error_profile_cache()

    mode_label = config.ERROR_MODE
    if config.ERROR_MODE == "DETECTED":
        mode_label = f"{config.ERROR_MODE} ({config.DETECTED_ERRORS_FILENAME})"

    print(f"BLEND merge — {config.CORPUS} ({mode_label})")
    print(f"  DIR_PATH    = {config.DIR_PATH}")
    print(f"  ERROR_MODE  = {config.ERROR_MODE}")
    if config.ERROR_MODE == "DETECTED":
        print(f"  DETECTED_ERRORS_DIR = {config.DETECTED_ERRORS_DIR}")
        print(f"  DETECTED_ERRORS_FILE = {config.DETECTED_ERRORS_FILENAME}")
    print(f"  DB_PATH     = {config.DB_PATH}")
    print(f"  MERGED_PATH = {config.MERGED_PATH}")
    print(f"  MERGE_VALIDATION = {config.MERGE_VALIDATION}")
    print(f"  VALIDATION_SCOPE = {config.VALIDATION_SCOPE}")
    print(f"  VALIDATION_STRATEGY = {config.VALIDATION_STRATEGY}")
    if config.MERGE_VALIDATION and config.VALIDATION_STRATEGY in {"distribution", "fd_and_distribution"}:
        print(f"  DIST_TVD_THRESHOLD = {config.DIST_TVD_THRESHOLD}")
        print(f"  DIST_KS_THRESHOLD = {config.DIST_KS_THRESHOLD}")
    print(f"  MERGE_PRIORITY = {config.MERGE_PRIORITY}")
    print(f"  RESULTS_ROOT = {config.RESULTS_ROOT}")
    recreate_modes = normalize_recreate_modes(recreate)
    print(f"  RECREATE    = {', '.join(recreate_modes)}")
    print()

    config.MERGED_PATH.mkdir(parents=True, exist_ok=True)
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    config_path = config.save_experiment_config(
        script=script_name,
        recreate=",".join(recreate_modes),
    )
    print(f"  CONFIG      = {config_path}")
    print()

    print("Phase 1: index_tables")
    index_tables(config.DIR_PATH, config.DB_PATH, config.BATCH_SIZE, config.TAB_LIMIT)
    print()

    print("Phase 2: merge_tables")
    merge_tables()
    print()

    string_output_dirs = run_recreate_phases(recreate)

    print("Done.")
    print(f"Merged lake: {config.MERGED_PATH}")
    for output_dir in string_output_dirs:
        print(f"String output: {output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        choices=[*CORPORA, "all"],
        required=True,
        help="Corpus to run, or 'all' for every corpus in sequence",
    )
    parser.add_argument(
        "--mode",
        choices=("gt", "detected"),
        default="gt",
        help="Error detection mode (default: gt)",
    )
    parser.add_argument(
        "--detected-run",
        type=int,
        default=1,
        help="Detection run number when --mode detected (default: 1)",
    )
    parser.add_argument(
        "--detected-results-subdir",
        default=DEFAULT_DETECTED_RESULTS_SUBDIR,
        help=f"Subdirectory under blend_merge_tables for detections (default: {DEFAULT_DETECTED_RESULTS_SUBDIR})",
    )
    parser.add_argument(
        "--recreate",
        choices=("set_union", "bag", "both"),
        default="both",
        help="Union semantics for string recreation: set_union (dedup), bag (union all), or both (default: both)",
    )
    parser.add_argument(
        "--merge-priority",
        choices=MERGE_PRIORITIES,
        default="union",
        help="Whether to seed merge comparisons with union or join candidates (default: union)",
    )
    parser.add_argument(
        "--merge-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable optional FD/distribution merge validation (default: off, paper §5.9.3)",
    )
    parser.add_argument(
        "--validation-scope",
        choices=VALIDATION_SCOPES,
        default="all",
        help="When validation is on: score all / joins only / unions only (default: all)",
    )
    parser.add_argument(
        "--validation-strategy",
        choices=VALIDATION_STRATEGIES,
        default="fd",
        help="Merge validation strategy when --merge-validation: fd, distribution, or fd_and_distribution",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    corpora = CORPORA if args.corpus == "all" else (args.corpus,)

    for corpus in corpora:
        if len(corpora) > 1:
            print(f"\n{'=' * 60}")
            print(f"Corpus: {corpus}")
            print(f"{'=' * 60}\n")

        configure(
            corpus,
            error_mode=args.mode.upper(),
            detected_run=args.detected_run,
            detected_results_subdir=args.detected_results_subdir,
            merge_priority=args.merge_priority,
            merge_validation=args.merge_validation,
            validation_scope=args.validation_scope,
            validation_strategy=args.validation_strategy,
        )
        run_pipeline(script_name=Path(__file__).name, recreate=args.recreate)


if __name__ == "__main__":
    main()
