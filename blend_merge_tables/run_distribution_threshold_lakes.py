#!/usr/bin/env python3
"""
Build separate merged lakes per TVD/KS distribution threshold.

FD-based merge validation is disabled; candidates are selected with
VALIDATION_STRATEGY='distribution' only. Each run sets both
DIST_TVD_THRESHOLD and DIST_KS_THRESHOLD to the same value.

Thresholds (default): 0, 0.25, 0.5, 0.75, 1

Phase 1 (index_tables) runs once per corpus; phase 2–3 repeat per threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
from error_cells import clear_error_profile_cache
from index_tables import index_tables
from merge_tables import merge_tables
from run_lake_exp import (
    CORPORA,
    DEFAULT_DETECTED_RESULTS_SUBDIR,
    MERGE_PRIORITIES,
    configure,
    normalize_recreate_modes,
    run_recreate_phases,
)

_BASE = Path(__file__).resolve().parent
DIST_THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _thresh_dir_name(threshold: float) -> str:
    if threshold == int(threshold):
        return f"dist_{int(threshold)}"
    return f"dist_{threshold}"


def _distribution_base_out(
    corpus: str,
    *,
    error_mode: str,
    detected_run: int,
) -> Path:
    if error_mode.upper() == "GT":
        return config.RESULTS_ROOT / "output_distribution_validation" / corpus / "gt"
    return (
        config.RESULTS_ROOT
        / "output_distribution_validation"
        / corpus
        / f"detected_run_{detected_run}"
    )


def _configure_distribution_threshold(
    corpus: str,
    threshold: float,
    *,
    error_mode: str,
    detected_run: int,
    detected_results_subdir: str,
    merge_priority: str,
) -> Path:
    """Apply lake-exp paths, then override validation + per-threshold output dirs."""
    configure(
        corpus,
        error_mode=error_mode,
        detected_run=detected_run,
        detected_results_subdir=detected_results_subdir,
        merge_priority=merge_priority,
        merge_validation=True,
        validation_strategy="distribution",
    )

    config.MERGE_VALIDATION = True
    config.VALIDATION_STRATEGY = "distribution"
    config.DIST_TVD_THRESHOLD = threshold
    config.DIST_KS_THRESHOLD = threshold

    base_out = _distribution_base_out(corpus, error_mode=error_mode, detected_run=detected_run)
    run_root = base_out / _thresh_dir_name(threshold)
    config.MERGED_PATH = run_root
    config.DB_PATH = base_out / "blend_index.duckdb"
    config.TRACKER_PATH = config.MERGED_PATH / "tracker.json"
    return run_root


def run_corpus(
    corpus: str,
    *,
    thresholds: tuple[float, ...],
    error_mode: str,
    detected_run: int,
    detected_results_subdir: str,
    merge_priority: str,
    recreate: str | tuple[str, ...],
) -> list[Path]:
    clear_error_profile_cache()
    _configure_distribution_threshold(
        corpus,
        thresholds[0],
        error_mode=error_mode,
        detected_run=detected_run,
        detected_results_subdir=detected_results_subdir,
        merge_priority=merge_priority,
    )

    print(f"BLEND distribution-threshold sweep — {corpus}")
    print(f"  DIR_PATH    = {config.DIR_PATH}")
    print(f"  ERROR_MODE  = {config.ERROR_MODE}")
    print(f"  MERGE_PRIORITY = {config.MERGE_PRIORITY}")
    print(f"  VALIDATION_STRATEGY = {config.VALIDATION_STRATEGY}")
    print(f"  MERGE_VALIDATION = {config.MERGE_VALIDATION}")
    print(f"  THRESHOLDS  = {list(thresholds)}")
    print(f"  RECREATE    = {', '.join(normalize_recreate_modes(recreate))}")
    print()

    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.save_experiment_config(
        script=Path(__file__).name,
        recreate=",".join(normalize_recreate_modes(recreate)),
        distribution_thresholds=list(thresholds),
    )

    print("Phase 1: index_tables (once per corpus)")
    index_tables(config.DIR_PATH, config.DB_PATH, config.BATCH_SIZE, config.TAB_LIMIT)
    print()

    run_roots: list[Path] = []
    for threshold in thresholds:
        print(f"\n{'=' * 60}")
        print(
            f"Corpus={corpus}  DIST_TVD_THRESHOLD={threshold}  "
            f"DIST_KS_THRESHOLD={threshold}"
        )
        print(f"{'=' * 60}\n")

        run_root = _configure_distribution_threshold(
            corpus,
            threshold,
            error_mode=error_mode,
            detected_run=detected_run,
            detected_results_subdir=detected_results_subdir,
            merge_priority=merge_priority,
        )
        run_roots.append(run_root)
        clear_error_profile_cache()

        config.MERGED_PATH.mkdir(parents=True, exist_ok=True)
        config.save_experiment_config(
            result_dir=run_root,
            script=Path(__file__).name,
            recreate=",".join(normalize_recreate_modes(recreate)),
            dist_tvd_threshold=threshold,
            dist_ks_threshold=threshold,
        )

        print("Phase 2: merge_tables")
        merge_tables()
        print()

        run_recreate_phases(recreate)

    return run_roots


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        choices=[*CORPORA, "all"],
        default="mit_dw_lake_exp",
        help="Corpus to run (default: mit_dw_lake_exp)",
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
        help=f"Detections subdirectory (default: {DEFAULT_DETECTED_RESULTS_SUBDIR})",
    )
    parser.add_argument(
        "--merge-priority",
        choices=MERGE_PRIORITIES,
        default="union",
        help="Union-first or join-first candidate ordering (default: union)",
    )
    parser.add_argument(
        "--recreate",
        choices=("set_union", "bag", "both"),
        default="both",
        help="String recreation mode(s) (default: both)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DIST_THRESHOLDS),
        help="TVD/KS thresholds to sweep (default: 0 0.25 0.5 0.75 1)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    thresholds = tuple(args.thresholds)
    corpora = CORPORA if args.corpus == "all" else (args.corpus,)
    error_mode = args.mode.upper()

    all_roots: list[Path] = []
    for corpus in corpora:
        if len(corpora) > 1:
            print(f"\n{'#' * 60}")
            print(f"Corpus: {corpus}")
            print(f"{'#' * 60}\n")
        try:
            roots = run_corpus(
                corpus,
                thresholds=thresholds,
                error_mode=error_mode,
                detected_run=args.detected_run,
                detected_results_subdir=args.detected_results_subdir,
                merge_priority=args.merge_priority,
                recreate=args.recreate,
            )
            all_roots.extend(roots)
        except Exception as exc:
            print(f"Run failed for corpus={corpus}: {exc}", file=sys.stderr)
            raise

    print("\nDone. Merged lakes:")
    for root in all_roots:
        print(f"  {root}")


if __name__ == "__main__":
    main()
