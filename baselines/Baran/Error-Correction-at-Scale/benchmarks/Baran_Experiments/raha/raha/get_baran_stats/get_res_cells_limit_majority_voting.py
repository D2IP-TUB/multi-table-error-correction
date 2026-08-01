"""
Evaluate error correction on an integrated (merged) lake with provenance-based majority voting.

When the lake has joins, the same source error cell can appear in multiple rows and get
corrected multiple times. We track each (row, col) by its provenance (source table § col § row),
then for each unique provenance that was corrected 2+ times we collect all proposed correction
values and pick the one with the highest frequency. That winning value is compared to the
ground-truth clean value: if it matches the cell counts as TP, otherwise FP.
This avoids double-counting errors in evaluation.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd


# Provenance format in merged tables: "table_name § col_id § row_id" per cell
PROVENANCE_SEP = " § "

logger = logging.getLogger(__name__)


def load_provenance_for_table(sandbox_path, table_name):
    """
    Load provenance matrix for one merged table.
    Returns a list of lists: prov[r][c] = provenance string for cell (r,c), or '' if empty.
    """
    prov_path = os.path.join(sandbox_path, str(table_name), "provenance.csv")
    if not os.path.exists(prov_path):
        logger.debug("Provenance missing for table %s: %s", table_name, prov_path)
        return None
    df = pd.read_csv(prov_path, keep_default_na=False, dtype=str, encoding="latin1")
    # df: rows = merged rows, columns = merged cols; cell = "TableName § col_id § row_id"
    return [df.iloc[r].tolist() for r in range(len(df))]


def load_error_types_for_table(sandbox_path, table_name):
    """
    Load an error-type matrix for one merged table, aligned with dirty/clean/provenance.
    Returns a list of lists: err[r][c] = error_type string for cell (r,c), or '' if none.
    """
    ets_path = os.path.join(sandbox_path, str(table_name), "merged_cell_source_map.csv")
    if not os.path.exists(ets_path):
        logger.debug("Error-type map missing for table %s: %s", table_name, ets_path)
        return None
    df = pd.read_csv(ets_path, keep_default_na=False, dtype=str, encoding="latin1")
    if df.empty:
        return None
    # merged row/column indices in the unioned/merged table
    try:
        df["row_number"] = df["row_number"].astype(int)
        df["column_id"] = df["column_id"].astype(int)
    except Exception as e:
        logger.debug("Failed to parse row_number/column_id in %s: %s", ets_path, e)
        return None
    n_rows = int(df["row_number"].max()) + 1
    n_cols = int(df["column_id"].max()) + 1
    err = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for _, row in df.iterrows():
        r = int(row["row_number"])
        c = int(row["column_id"])
        if r < 0 or c < 0 or r >= n_rows or c >= n_cols:
            continue
        et = str(row.get("error_type", "")).strip()
        if et:
            err[r][c] = et
    return err


def get_unique_errors_by_provenance(
    sandbox_path, dirty_file_name="dirty.csv", clean_file_name="clean.csv"
):
    """
    Count unique error cells by provenance across all tables in the sandbox.
    Same source cell (same provenance) appearing in multiple rows (e.g. due to join) counts once.

    Also, when error-type maps are available, compute how many unique provenances
    belong to each error type (lake-wide).
    """
    tables = [
        d for d in os.listdir(sandbox_path)
        if os.path.isdir(os.path.join(sandbox_path, d))
        and not d.startswith("union_summary")
        and not d.endswith(".json")
    ]
    logger.info("Counting unique errors by provenance across %d tables in %s", len(tables), sandbox_path)
    all_error_provenances = set()
    error_type_by_provenance = {}
    skipped = 0
    for i, table in enumerate(sorted(tables)):
        dirty_path = os.path.join(sandbox_path, table, dirty_file_name)
        clean_path = os.path.join(sandbox_path, table, clean_file_name)
        prov_path = os.path.join(sandbox_path, table, "provenance.csv")
        if not all(os.path.exists(p) for p in [dirty_path, clean_path, prov_path]):
            logger.debug("Skipping table %s: missing dirty/clean/provenance", table)
            skipped += 1
            continue
        enc = "latin1"
        dirty_df = pd.read_csv(dirty_path, keep_default_na=False, dtype=str, encoding=enc)
        clean_df = pd.read_csv(clean_path, keep_default_na=False, dtype=str, encoding=enc)
        prov_df = pd.read_csv(prov_path, keep_default_na=False, dtype=str, encoding=enc)
        # Optional error-type matrix aligned with merged dirty/clean/provenance
        err_matrix = load_error_types_for_table(sandbox_path, table)
        if dirty_df.shape != clean_df.shape or dirty_df.shape != prov_df.shape:
            logger.debug("Skipping table %s: shape mismatch dirty %s clean %s prov %s",
                         table, dirty_df.shape, clean_df.shape, prov_df.shape)
            skipped += 1
            continue
        n_errors_this = 0
        for r in range(len(dirty_df)):
            for c in range(len(dirty_df.columns)):
                if str(dirty_df.iloc[r, c]) != str(clean_df.iloc[r, c]):
                    prov = str(prov_df.iloc[r, c]).strip()
                    if prov and PROVENANCE_SEP in prov:
                        # Normalize: one cell might have multiple provenances
                        # (e.g. "A § 0 § 0 | B § 0 § 0")
                        for p in prov.split("|"):
                            p = p.strip()
                            if not p:
                                continue
                            all_error_provenances.add(p)
                            n_errors_this += 1
                            # Attach an error type to this provenance when available
                            if err_matrix is not None:
                                et = row_col_to_error_type(err_matrix, r, c)
                                if et:
                                    # Do not overwrite if we already recorded a type for p
                                    error_type_by_provenance.setdefault(p, et)
        if (i + 1) % 10 == 0 or i == 0:
            logger.info("  Processed %d/%d tables, unique error provenances so far: %d",
                        i + 1, len(tables), len(all_error_provenances))
    if skipped:
        logger.info("Skipped %d tables (missing files or shape mismatch)", skipped)
    logger.info("Total unique errors by provenance (ec_tpfn): %d", len(all_error_provenances))

    # Lake-wide unique errors per error type (each provenance counts once)
    ec_tpfn_by_type = {}
    for prov in all_error_provenances:
        et = error_type_by_provenance.get(prov)
        if et:
            ec_tpfn_by_type[et] = ec_tpfn_by_type.get(et, 0) + 1

    if ec_tpfn_by_type:
        logger.info(
            "Unique errors by provenance per error type: %s",
            {k: int(v) for k, v in ec_tpfn_by_type.items()},
        )

    return len(all_error_provenances), all_error_provenances, ec_tpfn_by_type


def row_col_to_provenance(provenance_matrix, row, col):
    """Get provenance string for (row, col); return None if empty or out of bounds."""
    if provenance_matrix is None or row < 0 or col < 0:
        return None
    if row >= len(provenance_matrix) or col >= len(provenance_matrix[0]):
        return None
    p = str(provenance_matrix[row][col]).strip()
    if not p or PROVENANCE_SEP not in p:
        return None
    # If multiple provenances in one cell, take the first (primary source)
    return p.split("|")[0].strip()


def row_col_to_error_type(error_type_matrix, row, col):
    """Get error-type string for (row, col); return None if empty or out of bounds."""
    if error_type_matrix is None or row < 0 or col < 0:
        return None
    if row >= len(error_type_matrix) or col >= len(error_type_matrix[0]):
        return None
    et = str(error_type_matrix[row][col]).strip()
    return et or None


def load_result_json(file_path):
    """Load result JSON; return None if missing or invalid."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception:
        return None


def majority_vote_per_provenance(provenance_corrections, provenance_error_types=None):
    """
    provenance_corrections: dict provenance_key -> list of (correction_value, clean_value).
    provenance_error_types: optional dict provenance_key -> error_type string.

    For each unique provenance, pick the correction value with the highest frequency.
    If that value == clean_value, count as TP; else FP.

    Returns (tp_count, fp_count, per_type_counts):
      - per_type_counts: dict error_type -> {"tp": int, "fp": int, "total": int}
    """
    tp_count = 0
    fp_count = 0
    per_type_counts = {}
    for prov, corrections in provenance_corrections.items():
        value_counts = {}
        for corr_val, _ in corrections:
            value_counts[corr_val] = value_counts.get(corr_val, 0) + 1
        max_count = max(value_counts.values())
        most_common_value = min(v for v, c in value_counts.items() if c == max_count)
        clean_value = corrections[0][1]
        is_tp = most_common_value == clean_value
        if is_tp:
            tp_count += 1
        else:
            fp_count += 1

        if provenance_error_types:
            et = provenance_error_types.get(prov)
            if et:
                stats = per_type_counts.setdefault(et, {"tp": 0, "fp": 0, "total": 0})
                stats["total"] += 1
                if is_tp:
                    stats["tp"] += 1
                else:
                    stats["fp"] += 1
    return tp_count, fp_count, per_type_counts


def evaluate_one_result_with_provenance(
    sandbox_path, result_json, table_name, provenance_matrix, error_type_matrix=None
):
    """
    For one result JSON, map each corrected cell to provenance and collect all proposed
    correction values, then apply majority voting on the correction value per provenance.
    The most frequent correction value is compared to the clean value to determine TP/FP.
    Returns (tp_dedup, ec_tpfp_dedup, provenance_corrections, per_type_counts).

    per_type_counts is a dict: error_type -> {"tp": int, "fp": int, "total": int}
    """
    clean_path = os.path.join(sandbox_path, str(table_name), "clean.csv")
    clean_df = pd.read_csv(clean_path, keep_default_na=False, dtype=str, encoding="latin1")

    corrected_keys = result_json.get("corrected_errors_keys", [])
    tp_cells = result_json.get("true_postives_cells", [])
    fp_cells = result_json.get("false_positives_cells", {})
    tp_set = set(tuple(c) for c in tp_cells)

    # Build lookup for FP correction values: (row, col) -> correction_value
    fp_values = {}
    for key_str, corr_val in fp_cells.items():
        key_str = key_str.strip("() ")
        parts = key_str.split(",")
        if len(parts) == 2:
            fp_values[(int(parts[0].strip()), int(parts[1].strip()))] = str(corr_val)

    provenance_corrections = {}  # prov -> list of (correction_value, clean_value)
    provenance_error_types = {}
    for cell in corrected_keys:
        row, col = cell[0], cell[1]
        prov = row_col_to_provenance(provenance_matrix, row, col)
        if prov is None:
            continue
        clean_value = str(clean_df.iloc[row, col])
        if (row, col) in tp_set:
            correction_value = clean_value
        else:
            correction_value = fp_values.get((row, col), "")
        provenance_corrections.setdefault(prov, []).append((correction_value, clean_value))
        if error_type_matrix is not None and prov not in provenance_error_types:
            et = row_col_to_error_type(error_type_matrix, row, col)
            if et:
                provenance_error_types[prov] = et

    tp_dedup, fp_dedup, per_type_counts = majority_vote_per_provenance(
        provenance_corrections, provenance_error_types
    )
    ec_tpfp_dedup = tp_dedup + fp_dedup
    return tp_dedup, ec_tpfp_dedup, provenance_corrections, per_type_counts


def get_results_df_cells_limit_with_provenance(
    sandbox_path, results_path, algorithm, repetitions, cells_limits
):
    """
    Load all result JSONs and attach provenance-based (majority-vote) metrics per row.
    Also loads provenance per table and computes tp_dedup, ec_tpfp_dedup per result.
    """
    tables = [
        d for d in os.listdir(sandbox_path)
        if os.path.isdir(os.path.join(sandbox_path, d))
        and not d.startswith("union_summary")
        and not d.endswith(".json")
    ]
    logger.info("Loading provenance matrices for %d tables", len(tables))
    provenance_cache = {}
    error_type_cache = {}
    for t in tables:
        provenance_cache[t] = load_provenance_for_table(sandbox_path, t)
        error_type_cache[t] = load_error_types_for_table(sandbox_path, t)
    loaded_prov = sum(1 for v in provenance_cache.values() if v is not None)
    loaded_err = sum(1 for v in error_type_cache.values() if v is not None)
    logger.info(
        "Loaded provenance for %d/%d tables; error-type maps for %d/%d tables",
        loaded_prov,
        len(tables),
        loaded_err,
        len(tables),
    )

    rows = []
    for rep in repetitions:
        for cell_limit in cells_limits:
            cell_limit_path = os.path.join(results_path, f"cell_limit_{cell_limit}")
            if not os.path.exists(cell_limit_path):
                logger.debug("Cell limit path does not exist: %s", cell_limit_path)
                continue
            n_in_dir = 0
            for fname in os.listdir(cell_limit_path):
                if fname == "skipped" or not fname.startswith(algorithm) or not fname.endswith(".json"):
                    continue
                if f"number#{rep}" not in fname:
                    continue
                dataset = fname.split("_col_")[0].replace(f"{algorithm}_", "")
                file_path = os.path.join(cell_limit_path, fname)
                result = load_result_json(file_path)
                if result is None:
                    logger.debug("Failed to load or parse: %s", file_path)
                    continue
                prov_matrix = provenance_cache.get(dataset)
                if prov_matrix is None:
                    logger.debug("No provenance for dataset %s, skipping result file %s", dataset, fname)
                    continue
                err_matrix = error_type_cache.get(dataset)
                tp_dedup, ec_tpfp_dedup, _, per_type_counts = evaluate_one_result_with_provenance(
                    sandbox_path, result, dataset, prov_matrix, err_matrix
                )
                rows.append({
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
                    # Per-error-type majority-vote counts for this (dataset, rep, cell_limit)
                    # Stored as JSON: {error_type: {"tp": int, "fp": int, "total": int}, ...}
                    "per_error_type_counts": json.dumps(per_type_counts),
                })
                n_in_dir += 1
            if n_in_dir > 0:
                logger.info("  rep=%d cell_limit=%s: processed %d result files",
                            rep, cell_limit, n_in_dir)
    logger.info("Total result rows loaded: %d", len(rows))
    return pd.DataFrame(rows)


def get_total_results_cells_limit_majority_voting(
    cells_limits, repetitions, result_df, ec_tpfn_unique, ec_tpfn_by_type
):
    """
    Aggregate per (cell_limit, rep) using dedup metrics, then average over repetitions.
    ec_tpfn_unique: total unique errors by provenance (single number for the whole lake).
    ec_tpfn_by_type: dict error_type -> total unique errors by provenance (lake-wide).
    """
    logger.info("Aggregating results by cell_limit (majority-vote metrics, ec_tpfn=%d)", ec_tpfn_unique)
    total_results = {
        "cell_limit": [], "precision": [], "recall": [], "f1_score": [],
        "ec_tpfp": [], "ec_tpfn": [], "tp": [], "execution_time": [],
        "n_labeled_cells": [], "n_labeled_tuples": [], "n_tables": [],
        # Lake-wide per-error-type metrics (JSON per cell_limit)
        "per_error_type_metrics": [],
    }
    for cell_limit in cells_limits:
        precisions = []
        recalls = []
        f_scores = []
        tps = []
        ec_tpfps = []
        execution_times = []
        n_labeled_cells_list = []
        n_labeled_tuples_list = []
        n_tables_list = []
        # For each cell_limit, we aggregate per-error-type metrics across reps.
        # per_type_agg[error_type] -> dict with lists of per-rep metrics.
        per_type_agg = {}
        for rep in repetitions:
            sub = result_df[(result_df["execution_number"] == rep) & (result_df["cell_limit"] == cell_limit)]
            if len(sub) == 0:
                continue
            tp_rep = sub["tp_dedup"].sum()
            ec_tpfp_rep = sub["ec_tpfp_dedup"].sum()
            precisions.append(tp_rep / ec_tpfp_rep if ec_tpfp_rep > 0 else 0.0)
            recalls.append(tp_rep / ec_tpfn_unique if ec_tpfn_unique > 0 else 0.0)
            p, r = precisions[-1], recalls[-1]
            f_scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
            tps.append(tp_rep)
            ec_tpfps.append(ec_tpfp_rep)
            execution_times.append(sub["execution_time"].sum())
            n_labeled_cells_list.append(sub["number_of_labeled_cells"].sum())
            n_labeled_tuples_list.append(sub["number_of_labeled_tuples"].sum())
            n_tables_list.append(len(sub))

            # Aggregate per-error-type counts for this (cell_limit, rep) over all tables
            per_type_counts_rep = {}
            for _, row in sub.iterrows():
                per_type_json = row.get("per_error_type_counts")
                if not isinstance(per_type_json, str) or not per_type_json:
                    continue
                try:
                    per_type = json.loads(per_type_json)
                except Exception:
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

            # Convert per-type counts for this rep into precision/recall/f1, then
            # accumulate per-type metrics across reps.
            for et, stats in per_type_counts_rep.items():
                tp_rep_et = stats["tp"]
                ec_tpfp_rep_et = stats["total"]
                # Per-type precision for this rep
                p_et = tp_rep_et / ec_tpfp_rep_et if ec_tpfp_rep_et > 0 else 0.0
                # Per-type recall for this rep (lake-wide denominator)
                denom_et = ec_tpfn_by_type.get(et, 0)
                r_et = tp_rep_et / denom_et if denom_et > 0 else 0.0
                f_et = 2 * p_et * r_et / (p_et + r_et) if (p_et + r_et) > 0 else 0.0

                agg = per_type_agg.setdefault(
                    et,
                    {
                        "precision": [],
                        "recall": [],
                        "f1": [],
                        "tp": [],
                        "ec_tpfp": [],
                    },
                )
                agg["precision"].append(p_et)
                agg["recall"].append(r_et)
                agg["f1"].append(f_et)
                agg["tp"].append(tp_rep_et)
                agg["ec_tpfp"].append(ec_tpfp_rep_et)
        n_reps = len(precisions)
        if n_reps > 0:
            total_results["cell_limit"].append(cell_limit)
            total_results["precision"].append(float(np.mean(precisions)))
            total_results["recall"].append(float(np.mean(recalls)))
            total_results["f1_score"].append(float(np.mean(f_scores)))
            total_results["tp"].append(float(np.mean(tps)))
            total_results["ec_tpfp"].append(float(np.mean(ec_tpfps)))
            total_results["ec_tpfn"].append(float(ec_tpfn_unique))
            total_results["execution_time"].append(float(np.mean(execution_times)))
            total_results["n_labeled_cells"].append(float(np.mean(n_labeled_cells_list)))
            total_results["n_labeled_tuples"].append(float(np.mean(n_labeled_tuples_list)))
            total_results["n_tables"].append(float(np.mean(n_tables_list)))
            # Lake-wide per-error-type metrics (averaged over repetitions)
            per_error_type_metrics = {}
            for et, agg in per_type_agg.items():
                if not agg["precision"]:
                    continue
                per_error_type_metrics[et] = {
                    "precision": float(np.mean(agg["precision"])),
                    "recall": float(np.mean(agg["recall"])),
                    "f1": float(np.mean(agg["f1"])),
                    "tp": float(np.mean(agg["tp"])),
                    "ec_tpfp": float(np.mean(agg["ec_tpfp"])),
                    "ec_tpfn": float(ec_tpfn_by_type.get(et, 0)),
                }
            total_results["per_error_type_metrics"].append(json.dumps(per_error_type_metrics))
            logger.info("  cell_limit=%s: %d reps, avg precision=%.4f recall=%.4f f1=%.4f",
                        cell_limit, n_reps,
                        total_results["precision"][-1], total_results["recall"][-1], total_results["f1_score"][-1])
    return pd.DataFrame(total_results)


def run_majority_voting_evaluation(
    sandbox_path,
    results_path,
    algorithm="raha",
    repetitions=None,
    cells_limits=None,
    out_csv_path=None,
    out_per_table_path=None,
):
    """
    Full pipeline: compute unique errors by provenance, load results with provenance,
    apply majority voting per (rep, cell_limit), aggregate and save.
    """
    if repetitions is None:
        repetitions = range(1, 4)
    if cells_limits is None:
        cells_limits = [405, 1540, 2620]

    logger.info("Starting majority-voting evaluation: sandbox=%s results=%s algorithm=%s reps=%s cell_limits=%s",
                sandbox_path, results_path, algorithm, list(repetitions), list(cells_limits))

    ec_tpfn_unique, _, ec_tpfn_by_type = get_unique_errors_by_provenance(sandbox_path)
    logger.info("Unique errors by provenance (ec_tpfn for recall): %d", ec_tpfn_unique)

    logger.info("Loading result JSONs and applying provenance-based majority voting...")
    result_df = get_results_df_cells_limit_with_provenance(
        sandbox_path, results_path, algorithm, list(repetitions), cells_limits
    )
    if result_df.empty:
        logger.warning("No results found.")
        return None, None

    if out_per_table_path:
        result_df.to_csv(out_per_table_path, index=False)
        logger.info("Wrote per-table results to %s", out_per_table_path)

    total_df = get_total_results_cells_limit_majority_voting(
        cells_limits, list(repetitions), result_df, ec_tpfn_unique, ec_tpfn_by_type
    )
    if out_csv_path:
        total_df.to_csv(out_csv_path, index=False)
        logger.info("Wrote aggregated (majority-voting) results to %s", out_csv_path)

    logger.info("Done.")
    return total_df, result_df


if __name__ == "__main__":
    # For real-time logs when piping, run: python -u get_res_cells_limit_majority_voting.py
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sandbox_path = str(Path(__file__).resolve().parents[9] / 'datasets' / 'unionable_tables' / 'union_datasets_used_in_exp' / 'maximal_overlap_without_duplicates')
    results_path = str(Path(__file__).resolve().parents[9] / 'results' / 'baran' / 'cells_limit')
    out_csv = os.path.join(results_path, "baran_cells_limit_majority_voting_and_type.csv")
    out_per_table = os.path.join(results_path, "raha_results_per_table_cells_limit_majority_voting_and_type.csv")

    total_df, _ = run_majority_voting_evaluation(
        sandbox_path=sandbox_path,
        results_path=results_path,
        algorithm="raha",
        repetitions=range(1, 2),
        cells_limits=[5167],
        out_csv_path=out_csv,
        out_per_table_path=out_per_table,
    )
    if total_df is not None:
        logger.info("Aggregated results (provenance-based majority voting):")
        print("\n", total_df.to_string())
