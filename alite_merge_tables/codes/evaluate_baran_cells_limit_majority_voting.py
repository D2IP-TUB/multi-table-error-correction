#!/usr/bin/env python3
"""
Evaluate Baran cell-limit results on ALITE FD-merged lakes with source-level majority voting.

ALITE differs from BLEND join/union merges (see get_res_cells_limit_majority_voting.py):
  - FD integration subsumes many source tuples into one output row.
  - Erroneous source cells are nullified before FD, so they are absent from the cell
    source map but still recorded in subsumption_map / clean_changes_provenance.
  - Multiple distinct source errors can map to the same merged (row, col).

This script therefore keys majority voting on isolated source errors
  (source_table, source_row, source_column)
using clean_changes_provenance.csv and isolated_error_map.csv exported by
export_alite_merged_tables.py — not on the BLEND-style provenance grid alone.

For each source error that Baran attempted to fix (via any merged cell it maps to),
all proposed correction values are collected and the most frequent value wins.
That value is compared to the source-specific clean value from clean_changes_provenance.

Recall (ec_tpfn) uses the full isolated corpus error count (error_map.csv across all
source tables), so results are comparable to single-table / isolated baselines.
Errors on source tuples dropped by FD merge are included in the denominator but
cannot be corrected in the merged lake (they count as false negatives).

Usage (from alite/codes/):
    python evaluate_baran_cells_limit_majority_voting.py
    python evaluate_baran_cells_limit_majority_voting.py \\
        --sandbox /path/to/alite_merged_tables \\
        --results /path/to/exp_baran-enough-labels-cells-limit \\
        --cell-limits 627 4026 \\
        --repetitions 1 2 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alite_paths import CORPORA, isolated_dir

logger = logging.getLogger(__name__)

ENCODING = "latin1"
SourceErrorKey = tuple[str, int, str]


def normalize_source_table(name: str) -> str:
    return str(name).strip().removesuffix(".csv")


def source_error_key(source_table: str, source_row: int | str, source_column: str) -> SourceErrorKey:
    return (
        normalize_source_table(source_table),
        int(source_row),
        str(source_column).strip().lower(),
    )


@dataclass
class AliteErrorIndex:
    """Per-table index of source errors and their merged-cell locations."""

    # (merged_row, merged_col) -> [(source_key, clean_value), ...]
    cell_sources: dict[tuple[int, int], list[tuple[SourceErrorKey, str]]] = field(
        default_factory=dict
    )
    # source_key -> clean ground-truth value
    clean_by_source: dict[SourceErrorKey, str] = field(default_factory=dict)
    # source_key -> error type label
    error_type_by_source: dict[SourceErrorKey, str] = field(default_factory=dict)


def list_sandbox_tables(sandbox_path: str) -> list[str]:
    return sorted(
        d
        for d in os.listdir(sandbox_path)
        if os.path.isdir(os.path.join(sandbox_path, d))
        and not d.startswith("union_summary")
        and not d.endswith(".json")
    )


def load_alite_error_index(sandbox_path: str, table_name: str) -> AliteErrorIndex | None:
    """
    Build error index from clean_changes_provenance.csv.
    Returns None when the table has no recorded source errors in the merge.
    """
    changes_path = os.path.join(sandbox_path, str(table_name), "clean_changes_provenance.csv")
    if not os.path.exists(changes_path):
        return None

    df = pd.read_csv(changes_path, keep_default_na=False, dtype=str, encoding=ENCODING)
    if df.empty:
        return None

    index = AliteErrorIndex()
    for _, row in df.iterrows():
        try:
            merged_row = int(row["row_number"])
            merged_col = int(row["column_id"])
            clean_value = str(row["new_value"])
            err_type = str(row.get("error_type", "")).strip()
            key = source_error_key(row["source_table"], row["source_row"], row["source_column"])
        except (KeyError, TypeError, ValueError):
            continue

        index.clean_by_source[key] = clean_value
        if err_type:
            index.error_type_by_source[key] = err_type
        index.cell_sources.setdefault((merged_row, merged_col), []).append((key, clean_value))

    return index if index.clean_by_source else None


def get_isolated_corpus_errors(
    isolated_path: str,
) -> tuple[int, set[SourceErrorKey], dict[str, int]]:
    """
    Count all unique source errors in the isolated corpus (error_map.csv per table).
    This is the recall denominator when comparing to isolated-table baselines.
    """
    all_keys: set[SourceErrorKey] = set()
    by_type: dict[str, int] = {}
    root = Path(isolated_path)

    for table_dir in sorted(root.iterdir()):
        if not table_dir.is_dir():
            continue
        error_map_path = table_dir / "error_map.csv"
        if not error_map_path.exists():
            continue
        df = pd.read_csv(error_map_path, keep_default_na=False, dtype=str, encoding=ENCODING)
        for _, row in df.iterrows():
            try:
                key = source_error_key(table_dir.name, row["row_number"], row["column_name"])
            except (KeyError, TypeError, ValueError):
                continue
            if key in all_keys:
                continue
            all_keys.add(key)
            et = str(row.get("error_type", "")).strip() or "UNKNOWN"
            by_type[et] = by_type.get(et, 0) + 1

    logger.info(
        "Isolated corpus unique source errors: %d (%d error types)",
        len(all_keys),
        len(by_type),
    )
    return len(all_keys), all_keys, by_type


def get_unique_errors_in_merged_lake(
    sandbox_path: str,
) -> tuple[int, set[SourceErrorKey], dict[str, int]]:
    """
    Count unique source errors present in the merged lake (isolated_error_map.csv).
    Strictly fewer than the isolated corpus when FD drops incompatible tuples.
    """
    all_keys: set[SourceErrorKey] = set()
    by_type: dict[str, int] = {}

    for table in list_sandbox_tables(sandbox_path):
        path = os.path.join(sandbox_path, table, "isolated_error_map.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, keep_default_na=False, dtype=str, encoding=ENCODING)
        for _, row in df.iterrows():
            try:
                key = source_error_key(
                    row["source_table"], row["source_row"], row["source_column"]
                )
            except (KeyError, TypeError, ValueError):
                continue
            if key in all_keys:
                continue
            all_keys.add(key)
            et = str(row.get("error_type", "")).strip()
            if et:
                by_type[et] = by_type.get(et, 0) + 1

    logger.info(
        "Merged-lake unique source errors: %d (%d error types)",
        len(all_keys),
        len(by_type),
    )
    return len(all_keys), all_keys, by_type


def load_result_json(file_path: str) -> dict[str, Any] | None:
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", file_path, exc)
        return None


def parse_fp_cells(fp_cells: dict[str, Any]) -> dict[tuple[int, int], str]:
    values: dict[tuple[int, int], str] = {}
    for key_str, corr_val in fp_cells.items():
        key_str = str(key_str).strip("() ")
        parts = key_str.split(",")
        if len(parts) != 2:
            continue
        try:
            values[(int(parts[0].strip()), int(parts[1].strip()))] = str(corr_val)
        except ValueError:
            continue
    return values


def correction_value_for_cell(
    row: int,
    col: int,
    tp_set: set[tuple[int, int]],
    fp_values: dict[tuple[int, int], str],
    clean_df: pd.DataFrame,
) -> str:
    if (row, col) in tp_set:
        return str(clean_df.iloc[row, col])
    return fp_values.get((row, col), "")


def majority_vote_per_source_error(
    source_corrections: dict[SourceErrorKey, list[tuple[str, str]]],
    error_index: AliteErrorIndex,
) -> tuple[int, int, dict[str, dict[str, int]]]:
    """
    For each source error with at least one proposed correction, pick the most
    frequent correction value (ties -> lexicographically smallest).
    Compare against the source-specific clean value.
    """
    tp_count = 0
    fp_count = 0
    per_type_counts: dict[str, dict[str, int]] = {}

    for key, corrections in source_corrections.items():
        if not corrections:
            continue

        value_counts: dict[str, int] = {}
        for corr_val, _ in corrections:
            value_counts[corr_val] = value_counts.get(corr_val, 0) + 1
        max_count = max(value_counts.values())
        winner = min(v for v, c in value_counts.items() if c == max_count)

        clean_value = error_index.clean_by_source.get(key, corrections[0][1])
        is_tp = winner == clean_value
        if is_tp:
            tp_count += 1
        else:
            fp_count += 1

        et = error_index.error_type_by_source.get(key, "")
        if et:
            stats = per_type_counts.setdefault(et, {"tp": 0, "fp": 0, "total": 0})
            stats["total"] += 1
            if is_tp:
                stats["tp"] += 1
            else:
                stats["fp"] += 1

    return tp_count, fp_count, per_type_counts


def evaluate_one_result_alite(
    sandbox_path: str,
    result_json: dict[str, Any],
    table_name: str,
    error_index: AliteErrorIndex,
) -> tuple[int, int, dict[str, dict[str, int]]]:
    """
    Map Baran corrected merged cells back to isolated source errors, then vote.
    """
    clean_path = os.path.join(sandbox_path, str(table_name), "clean.csv")
    clean_df = pd.read_csv(clean_path, keep_default_na=False, dtype=str, encoding=ENCODING)

    corrected_keys = result_json.get("corrected_errors_keys", [])
    tp_set = {tuple(c) for c in result_json.get("true_postives_cells", [])}
    fp_values = parse_fp_cells(result_json.get("false_positives_cells", {}))

    source_corrections: dict[SourceErrorKey, list[tuple[str, str]]] = {}
    for cell in corrected_keys:
        row, col = int(cell[0]), int(cell[1])
        corr_val = correction_value_for_cell(row, col, tp_set, fp_values, clean_df)
        for key, clean_val in error_index.cell_sources.get((row, col), []):
            source_corrections.setdefault(key, []).append((corr_val, clean_val))

    return majority_vote_per_source_error(source_corrections, error_index)


def get_results_df_cells_limit_alite(
    sandbox_path: str,
    results_path: str,
    algorithm: str,
    repetitions: list[int],
    cells_limits: list[int],
) -> pd.DataFrame:
    tables = list_sandbox_tables(sandbox_path)
    logger.info("Indexing ALITE source errors for %d tables", len(tables))

    error_index_cache: dict[str, AliteErrorIndex | None] = {}
    indexed = 0
    for table in tables:
        idx = load_alite_error_index(sandbox_path, table)
        error_index_cache[table] = idx
        if idx is not None:
            indexed += 1
    logger.info("Tables with source errors in merge: %d/%d", indexed, len(tables))

    rows: list[dict[str, Any]] = []
    for rep in repetitions:
        for cell_limit in cells_limits:
            cell_limit_path = os.path.join(results_path, f"cell_limit_{cell_limit}")
            if not os.path.exists(cell_limit_path):
                logger.warning("Missing cell-limit directory: %s", cell_limit_path)
                continue

            n_files = 0
            for fname in os.listdir(cell_limit_path):
                if fname == "skipped" or not fname.startswith(algorithm) or not fname.endswith(".json"):
                    continue
                if f"number#{rep}" not in fname:
                    continue

                dataset = fname.split("_col_")[0].replace(f"{algorithm}_", "")
                error_index = error_index_cache.get(dataset)
                if error_index is None:
                    continue

                file_path = os.path.join(cell_limit_path, fname)
                result = load_result_json(file_path)
                if result is None:
                    continue

                tp_dedup, ec_tpfp_dedup, per_type_counts = evaluate_one_result_alite(
                    sandbox_path, result, dataset, error_index
                )
                rows.append(
                    {
                        "algorithm": algorithm,
                        "dataset": dataset,
                        "execution_number": rep,
                        "cell_limit": cell_limit,
                        "tp": result.get("tp", 0),
                        "ec_tpfp": result.get("ec_tpfp", 0),
                        "ec_tpfn": result.get("ec_tpfn", 0),
                        "precision": result.get("precision", 0),
                        "recall": result.get("recall", 0),
                        "f_score": result.get("f_score", 0),
                        "tp_dedup": tp_dedup,
                        "ec_tpfp_dedup": ec_tpfp_dedup,
                        "execution_time": result.get("execution-time", 0),
                        "number_of_labeled_cells": result.get("number_of_labeled_cells", 0),
                        "number_of_labeled_tuples": result.get("number_of_labeled_tuples", 0),
                        "per_error_type_counts": json.dumps(per_type_counts),
                    }
                )
                n_files += 1

            if n_files:
                logger.info("  rep=%d cell_limit=%d: %d result files", rep, cell_limit, n_files)

    logger.info("Total per-table result rows: %d", len(rows))
    return pd.DataFrame(rows)


def get_total_results_cells_limit_alite(
    cells_limits: list[int],
    repetitions: list[int],
    result_df: pd.DataFrame,
    ec_tpfn_isolated: int,
    ec_tpfn_by_type: dict[str, int],
    ec_tpfn_merged: int,
) -> pd.DataFrame:
    """Aggregate deduplicated metrics per cell limit, then average over repetitions."""
    total_results: dict[str, list[Any]] = {
        "cell_limit": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "ec_tpfp": [],
        "ec_tpfn": [],
        "ec_tpfn_merged": [],
        "tp": [],
        "execution_time": [],
        "n_labeled_cells": [],
        "n_labeled_tuples": [],
        "n_tables": [],
        "per_error_type_metrics": [],
    }

    for cell_limit in cells_limits:
        precisions: list[float] = []
        recalls: list[float] = []
        f_scores: list[float] = []
        tps: list[float] = []
        ec_tpfps: list[float] = []
        execution_times: list[float] = []
        n_labeled_cells_list: list[float] = []
        n_labeled_tuples_list: list[float] = []
        n_tables_list: list[float] = []
        per_type_agg: dict[str, dict[str, list[float]]] = {}

        for rep in repetitions:
            sub = result_df[
                (result_df["execution_number"] == rep) & (result_df["cell_limit"] == cell_limit)
            ]
            if sub.empty:
                continue

            tp_rep = float(sub["tp_dedup"].sum())
            ec_tpfp_rep = float(sub["ec_tpfp_dedup"].sum())
            precision_rep = tp_rep / ec_tpfp_rep if ec_tpfp_rep > 0 else 0.0
            recall_rep = tp_rep / ec_tpfn_isolated if ec_tpfn_isolated > 0 else 0.0
            f1_rep = (
                2 * precision_rep * recall_rep / (precision_rep + recall_rep)
                if (precision_rep + recall_rep) > 0
                else 0.0
            )

            precisions.append(precision_rep)
            recalls.append(recall_rep)
            f_scores.append(f1_rep)
            tps.append(tp_rep)
            ec_tpfps.append(ec_tpfp_rep)
            execution_times.append(float(sub["execution_time"].sum()))
            n_labeled_cells_list.append(float(sub["number_of_labeled_cells"].sum()))
            n_labeled_tuples_list.append(float(sub["number_of_labeled_tuples"].sum()))
            n_tables_list.append(float(len(sub)))

            per_type_counts_rep: dict[str, dict[str, int]] = {}
            for _, row in sub.iterrows():
                raw = row.get("per_error_type_counts")
                if not isinstance(raw, str) or not raw:
                    continue
                try:
                    per_type = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for et, stats in per_type.items():
                    if not isinstance(stats, dict):
                        continue
                    tp_et = int(stats.get("tp", 0))
                    total_et = int(stats.get("total", 0))
                    if total_et <= 0 and tp_et <= 0:
                        continue
                    agg = per_type_counts_rep.setdefault(et, {"tp": 0, "total": 0})
                    agg["tp"] += tp_et
                    agg["total"] += total_et

            for et, stats in per_type_counts_rep.items():
                tp_rep_et = stats["tp"]
                total_rep_et = stats["total"]
                p_et = tp_rep_et / total_rep_et if total_rep_et > 0 else 0.0
                denom_et = ec_tpfn_by_type.get(et, 0)
                r_et = tp_rep_et / denom_et if denom_et > 0 else 0.0
                f_et = 2 * p_et * r_et / (p_et + r_et) if (p_et + r_et) > 0 else 0.0
                bucket = per_type_agg.setdefault(
                    et,
                    {"precision": [], "recall": [], "f1": [], "tp": [], "ec_tpfp": []},
                )
                bucket["precision"].append(p_et)
                bucket["recall"].append(r_et)
                bucket["f1"].append(f_et)
                bucket["tp"].append(float(tp_rep_et))
                bucket["ec_tpfp"].append(float(total_rep_et))

        if not precisions:
            continue

        total_results["cell_limit"].append(cell_limit)
        total_results["precision"].append(float(np.mean(precisions)))
        total_results["recall"].append(float(np.mean(recalls)))
        total_results["f1_score"].append(float(np.mean(f_scores)))
        total_results["tp"].append(float(np.mean(tps)))
        total_results["ec_tpfp"].append(float(np.mean(ec_tpfps)))
        total_results["ec_tpfn"].append(float(ec_tpfn_isolated))
        total_results["ec_tpfn_merged"].append(float(ec_tpfn_merged))
        total_results["execution_time"].append(float(np.mean(execution_times)))
        total_results["n_labeled_cells"].append(float(np.mean(n_labeled_cells_list)))
        total_results["n_labeled_tuples"].append(float(np.mean(n_labeled_tuples_list)))
        total_results["n_tables"].append(float(np.mean(n_tables_list)))

        per_error_type_metrics = {
            et: {
                "precision": float(np.mean(agg["precision"])),
                "recall": float(np.mean(agg["recall"])),
                "f1": float(np.mean(agg["f1"])),
                "tp": float(np.mean(agg["tp"])),
                "ec_tpfp": float(np.mean(agg["ec_tpfp"])),
                "ec_tpfn": float(ec_tpfn_by_type.get(et, 0)),
            }
            for et, agg in per_type_agg.items()
            if agg["precision"]
        }
        total_results["per_error_type_metrics"].append(json.dumps(per_error_type_metrics))

        logger.info(
            "  cell_limit=%s: precision=%.4f recall=%.4f f1=%.4f",
            cell_limit,
            total_results["precision"][-1],
            total_results["recall"][-1],
            total_results["f1_score"][-1],
        )

    return pd.DataFrame(total_results)


def run_alite_majority_voting_evaluation(
    sandbox_path: str,
    results_path: str,
    isolated_path: str,
    algorithm: str = "raha",
    repetitions: list[int] | None = None,
    cells_limits: list[int] | None = None,
    out_csv_path: str | None = None,
    out_per_table_path: str | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if repetitions is None:
        repetitions = [1, 2, 3]
    if cells_limits is None:
        cells_limits = [627, 4026]

    logger.info(
        "ALITE majority-vote eval: sandbox=%s results=%s reps=%s limits=%s",
        sandbox_path,
        results_path,
        repetitions,
        cells_limits,
    )

    ec_tpfn_isolated, _, ec_tpfn_by_type = get_isolated_corpus_errors(isolated_path)
    ec_tpfn_merged, _, _ = get_unique_errors_in_merged_lake(sandbox_path)
    logger.info("Recall denominator (isolated corpus): %d", ec_tpfn_isolated)
    logger.info(
        "Errors present in merged lake: %d (dropped by FD: %d)",
        ec_tpfn_merged,
        ec_tpfn_isolated - ec_tpfn_merged,
    )
    if ec_tpfn_by_type:
        logger.info("  isolated by type: %s", ec_tpfn_by_type)

    result_df = get_results_df_cells_limit_alite(
        sandbox_path, results_path, algorithm, repetitions, cells_limits
    )
    if result_df.empty:
        logger.warning("No result rows produced.")
        return None, None

    if out_per_table_path:
        result_df.to_csv(out_per_table_path, index=False)
        logger.info("Wrote per-table results to %s", out_per_table_path)

    total_df = get_total_results_cells_limit_alite(
        cells_limits,
        repetitions,
        result_df,
        ec_tpfn_isolated,
        ec_tpfn_by_type,
        ec_tpfn_merged,
    )
    if out_csv_path:
        total_df.to_csv(out_csv_path, index=False)
        logger.info("Wrote aggregated results to %s", out_csv_path)

    return total_df, result_df


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    default_sandbox = (
        repo.parent / "revision_datasets/merging_experiments/alite_merged_tables"
    )
    default_results = (
        repo.parent
        / "revision_results/baran_alite/alite_merged_tables/exp_baran-enough-labels-cells-limit"
    )

    parser = argparse.ArgumentParser(
        description="Evaluate Baran cell-limit results on ALITE merged tables "
        "with source-level majority voting."
    )
    parser.add_argument("--sandbox", type=Path, default=default_sandbox)
    parser.add_argument("--results", type=Path, default=default_results)
    parser.add_argument(
        "--corpus",
        choices=CORPORA,
        default="open_data_uk",
        help="Corpus for default isolated tables path",
    )
    parser.add_argument(
        "--isolated",
        type=Path,
        default=None,
        help="Isolated source tables root (default: alite/datasets/.../isolated)",
    )
    parser.add_argument("--algorithm", default="raha")
    parser.add_argument("--repetitions", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--cell-limits", type=int, nargs="+", default=[627, 4026])
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Aggregated CSV path (default: <results>/baran_cells_limit_alite_majority_voting.csv)",
    )
    parser.add_argument(
        "--out-per-table",
        type=Path,
        default=None,
        help="Per-table CSV path (default: <results>/raha_results_per_table_cells_limit_alite_majority_voting.csv)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    sandbox_path = str(args.sandbox.resolve())
    results_path = str(args.results.resolve())
    isolated_path = str(
        (args.isolated or isolated_dir(args.corpus)).resolve()
    )
    out_csv = args.out_csv or Path(results_path) / "baran_cells_limit_alite_majority_voting.csv"
    out_per_table = (
        args.out_per_table
        or Path(results_path) / "raha_results_per_table_cells_limit_alite_majority_voting.csv"
    )

    total_df, _ = run_alite_majority_voting_evaluation(
        sandbox_path=sandbox_path,
        results_path=results_path,
        isolated_path=isolated_path,
        algorithm=args.algorithm,
        repetitions=args.repetitions,
        cells_limits=args.cell_limits,
        out_csv_path=str(out_csv),
        out_per_table_path=str(out_per_table),
    )
    if total_df is not None:
        print("\nAggregated results (ALITE source-level majority voting):")
        print(total_df.to_string(index=False))


if __name__ == "__main__":
    main()
