#!/usr/bin/env python3
from __future__ import annotations

"""
Lake-level evaluation script for Uniclean with MAJORITY VOTING for joined tables.

This mirrors the logic of
`/home/fatemeh/LakeCorrectionBench/HoloClean/evaluate_lake_majority_voting.py`,
but is adapted to Uniclean's file layout:

- Input (`--lake-dir`): root directory of the (joined) data lake.
  It must contain one subdirectory per table, and each table directory must have:
    - dirty.csv
    - clean.csv
    - provenance.csv  (or merged_cell_source_map.csv)
    - result/<table_name>/<table_name>Cleaned.csv  (Uniclean output)

The evaluation:
  1. Counts total unique errors in the lake using provenance, to avoid
     double-counting the same source error across joins.
  2. For each table, groups error cells by provenance
     ("source_table § col § row") and applies majority voting over the
     Uniclean repairs for that provenance.
  3. Computes per-table and lake-wide precision / recall / F1 w.r.t.
     unique errors.
  4. Optionally saves per-table results to CSV.

Usage example:

    python evaluate_lake_majority_voting_uniclean.py \\
        --lake-dir  datasets_and_rules/Real_Lake_Default_Datasets/merged_strings_default_set_union/mit_dwh/merged \\
        --save-csv  datasets_and_rules/Real_Lake_Default_Datasets/merged_strings_default_set_union/mit_dwh/merged/uniclean_results/lake_majority_voting_uniclean.csv
"""

import argparse
import logging
import os
import sys
import traceback
from collections import Counter, defaultdict

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV reading helpers
# ---------------------------------------------------------------------------

def read_csv(path: str) -> pd.DataFrame:
    """
    Read CSV in a way compatible with LakeCorrectionBench / HoloClean:
    - latin-1 encoding
    - all columns as string
    - keep_default_na=False (so empty cells stay empty strings)
    """
    return pd.read_csv(
        path,
        sep=",",
        header="infer",
        encoding="latin-1",
        dtype=str,
        keep_default_na=False,
    )


def read_csv_robust(path: str, kind: str) -> pd.DataFrame:
    """
    Read CSV robustly for evaluation:
      1) Try the standard settings (matches LakeCorrectionBench/HoloClean).
      2) If that fails (e.g., tokenizer error on a bad row), retry with
         engine='python' and on_bad_lines='skip' so that we still evaluate
         all parsable rows instead of dropping the table.
    """
    try:
        return read_csv(path)
    except Exception as exc:
        logger.warning(
            "[%s] Failed to read %s with default engine: %s",
            os.path.basename(path),
            kind,
            exc,
        )
        try:
            return pd.read_csv(
                path,
                sep=",",
                header="infer",
                encoding="latin-1",
                dtype=str,
                keep_default_na=False,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as exc2:
            logger.warning(
                "[%s] Failed to read %s even with python engine/on_bad_lines=skip: %s",
                os.path.basename(path),
                kind,
                exc2,
            )
            # Re-raise so the caller can decide on a fallback (e.g., treat as no-output)
            raise


def _drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the synthetic 'index' column if present.

    The merged lake tables were preprocessed with an 'index' column added
    for Uniclean, but provenance.csv is defined over the original columns
    only. To keep (row_idx, col_idx) aligned with provenance, we must
    remove 'index' from the evaluation dataframes.
    """
    if "index" in df.columns:
        return df.drop(columns=["index"])
    return df


# ---------------------------------------------------------------------------
# Provenance & error-type loading (adapted from HoloClean script)
# ---------------------------------------------------------------------------

def load_provenance_map(table_dir: str) -> dict[tuple[int, int], str]:
    """
    Load provenance mapping from provenance.csv or merged_cell_source_map.csv.

    Returns:
        dict mapping (row_idx, col_idx) -> "source_table § col § row"
        where col_idx is zero-based over NON-index columns.
    """
    provenance_csv = os.path.join(table_dir, "provenance.csv")
    merged_map_csv = os.path.join(table_dir, "merged_cell_source_map.csv")

    provenance_map: dict[tuple[int, int], str] = {}

    # Prefer provenance.csv (with '§' separator)
    if os.path.exists(provenance_csv):
        try:
            with open(provenance_csv, "r", encoding="latin-1") as f:
                lines = f.readlines()

            if len(lines) < 2:
                logger.warning("[%s] provenance.csv has insufficient data", os.path.basename(table_dir))
                return provenance_map

            # Data rows; columns correspond to non-index columns in dirty/clean.
            for row_idx, line in enumerate(lines[1:]):  # Skip header
                cells = line.strip().split(",")
                for col_idx, cell_value in enumerate(cells):
                    cell_value = cell_value.strip()
                    if cell_value and "§" in cell_value:
                        provenance_map[(row_idx, col_idx)] = cell_value

            logger.info("[%s] Loaded %d provenance entries from provenance.csv",
                        os.path.basename(table_dir), len(provenance_map))
            return provenance_map

        except Exception as e:
            logger.warning("[%s] Failed to parse provenance.csv: %s",
                           os.path.basename(table_dir), e)

    # Fallback: merged_cell_source_map.csv
    if os.path.exists(merged_map_csv):
        try:
            df = pd.read_csv(merged_map_csv)
            for _, row in df.iterrows():
                row_idx = int(row["row_number"])
                col_idx = int(row["column_id"])
                source_table = str(row["source_table"])
                source_row = str(row["source_row"])
                source_col = str(row["source_column"])
                prov = f"{source_table} § {source_col} § {source_row}"
                provenance_map[(row_idx, col_idx)] = prov

            logger.info("[%s] Loaded %d provenance entries from merged_cell_source_map.csv",
                        os.path.basename(table_dir), len(provenance_map))
            return provenance_map

        except Exception as e:
            logger.warning("[%s] Failed to parse merged_cell_source_map.csv: %s",
                           os.path.basename(table_dir), e)

    logger.warning("[%s] No provenance information found", os.path.basename(table_dir))
    return provenance_map


def load_error_type_map(table_dir: str) -> dict[tuple[int, int], str]:
    """
    Load (row_idx, col_idx) -> error_type from merged_cell_source_map.csv.

    Returns empty dict if file or column is missing. Empty/missing error_type
    is stored as "" (caller should normalize to "unknown" when used).
    """
    merged_map_csv = os.path.join(table_dir, "merged_cell_source_map.csv")
    error_type_map: dict[tuple[int, int], str] = {}
    if not os.path.exists(merged_map_csv):
        return error_type_map
    try:
        df = pd.read_csv(merged_map_csv)
        if (
            "error_type" not in df.columns
            or "row_number" not in df.columns
            or "column_id" not in df.columns
        ):
            return error_type_map
        for _, row in df.iterrows():
            row_idx = int(row["row_number"])
            col_idx = int(row["column_id"])
            et = str(row["error_type"]).strip() if pd.notna(row.get("error_type")) else ""
            error_type_map[(row_idx, col_idx)] = et
        logger.info(
            "[%s] Loaded %d error-type entries from merged_cell_source_map.csv",
            os.path.basename(table_dir),
            len(error_type_map),
        )
    except Exception as e:
        logger.warning(
            "[%s] Failed to load error types from merged_cell_source_map.csv: %s",
            os.path.basename(table_dir),
            e,
        )
    return error_type_map


# ---------------------------------------------------------------------------
# Core majority-voting evaluation (adapted from HoloClean script)
# ---------------------------------------------------------------------------

def _sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '::' and other problematic characters in column names."""
    df.columns = [col.replace("::", "_") for col in df.columns]
    return df


def get_dataframes_difference(
    dirty_df: pd.DataFrame, clean_df: pd.DataFrame
) -> list[tuple[int, int]]:
    """Return list of (row, col) tuples where dirty != clean."""
    detections: list[tuple[int, int]] = []
    for i in range(len(dirty_df)):
        for j in range(len(dirty_df.columns)):
            dirty_val = str(dirty_df.iloc[i, j]).strip()
            clean_val = str(clean_df.iloc[i, j]).strip()
            if dirty_val != clean_val:
                detections.append((i, j))
    return detections


def evaluate_with_majority_voting(
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    repaired_df: pd.DataFrame,
    provenance_map: dict[tuple[int, int], str],
    error_type_map: dict[tuple[int, int], str] | None = None,
) -> dict:
    """
    Evaluate repairs using majority voting for cells with the same provenance.

    Algorithm (identical to HoloClean script):
      1. Identify all errors (dirty != clean)
      2. For each error, get its provenance
      3. Group errors by provenance
      4. For provenances that appear 2+ times:
         - Collect all proposed corrections (repaired values)
         - Apply majority voting to pick the winning value
         - Compare winning value to ground truth
      5. For provenances that appear only once:
         - Directly compare repaired value to ground truth

    If error_type_map is provided, metrics are also computed per error type.
    """

    def _error_type(row_idx: int, col_idx: int) -> str:
        et = (error_type_map or {}).get((row_idx, col_idx), "").strip()
        return et if et else "unknown"

    def _repaired_value(row_idx: int, col_idx: int) -> str:
        """
        Safely get repaired_df value for (row_idx, col_idx).
        If the repaired dataframe is shorter or has fewer columns (e.g. due to
        a skipped bad line during parsing), fall back to the dirty value,
        which semantically means "no attempted repair" for that cell.
        """
        try:
            return str(repaired_df.iloc[row_idx, col_idx]).strip()
        except Exception:
            return str(dirty_df.iloc[row_idx, col_idx]).strip()

    detections = get_dataframes_difference(dirty_df, clean_df)
    if len(detections) == 0:
        return {
            "n_unique_errors": 0,
            "n_truely_corrected_errors": 0,
            "n_all_corrected_errors": 0,
            "n_duplicate_errors": 0,
            "n_majority_voted": 0,
            "precision": -1,
            "recall": -1,
            "f1_score": -1,
            "by_error_type": {},
            "n_total_error_cells": 0,
        }

    # Per-error-type accumulators
    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_unique": 0, "n_tp": 0, "n_attempted": 0}
    )

    provenance_to_cells: dict[str, list[tuple[int, int]]] = defaultdict(list)
    errors_without_provenance: list[tuple[int, int]] = []

    for row_idx, col_idx in detections:
        prov = provenance_map.get((row_idx, col_idx))
        if prov:
            provenance_to_cells[prov].append((row_idx, col_idx))
        else:
            errors_without_provenance.append((row_idx, col_idx))

    n_truely_corrected = 0
    n_attempted_corrections = 0
    n_duplicate_errors = 0
    n_majority_voted = 0

    # Provenance-backed errors
    for prov, cells in provenance_to_cells.items():
        ground_truth = str(clean_df.iloc[cells[0][0], cells[0][1]]).strip()
        dirty_val = str(dirty_df.iloc[cells[0][0], cells[0][1]]).strip()
        err_type = _error_type(cells[0][0], cells[0][1])
        by_type[err_type]["n_unique"] += 1

        if len(cells) == 1:
            row_idx, col_idx = cells[0]
            repaired_val = _repaired_value(row_idx, col_idx)
            if dirty_val != repaired_val:
                n_attempted_corrections += 1
                by_type[err_type]["n_attempted"] += 1
                if repaired_val == ground_truth or (
                    len(repaired_val) == 0 and len(ground_truth) == 0
                ):
                    n_truely_corrected += 1
                    by_type[err_type]["n_tp"] += 1
        else:
            n_duplicate_errors += len(cells) - 1
            n_majority_voted += 1

            proposed_values: list[str] = []
            correction_attempted = False
            for row_idx, col_idx in cells:
                repaired_val = _repaired_value(row_idx, col_idx)
                proposed_values.append(repaired_val)
                if dirty_val != repaired_val:
                    correction_attempted = True

            if correction_attempted:
                n_attempted_corrections += 1
                by_type[err_type]["n_attempted"] += 1

                value_counts = Counter(proposed_values)
                max_count = value_counts.most_common(1)[0][1]
                tied_winners = [v for v, c in value_counts.items() if c == max_count]
                winning_value = min(tied_winners)

                if winning_value == ground_truth or (
                    len(winning_value) == 0 and len(ground_truth) == 0
                ):
                    n_truely_corrected += 1
                    by_type[err_type]["n_tp"] += 1

    # Errors without provenance -> standard cell-level evaluation
    for row_idx, col_idx in errors_without_provenance:
        err_type = _error_type(row_idx, col_idx)
        by_type[err_type]["n_unique"] += 1
        dirty_val = str(dirty_df.iloc[row_idx, col_idx]).strip()
        clean_val = str(clean_df.iloc[row_idx, col_idx]).strip()
        repaired_val = _repaired_value(row_idx, col_idx)

        if dirty_val != repaired_val:
            n_attempted_corrections += 1
            by_type[err_type]["n_attempted"] += 1
            if repaired_val == clean_val or (
                len(repaired_val) == 0 and len(clean_val) == 0
            ):
                n_truely_corrected += 1
                by_type[err_type]["n_tp"] += 1

    n_unique_errors = len(provenance_to_cells) + len(errors_without_provenance)

    precision = (
        n_truely_corrected / n_attempted_corrections
        if n_attempted_corrections > 0
        else -1
    )
    recall = n_truely_corrected / n_unique_errors if n_unique_errors > 0 else -1
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else -1
    )

    by_error_type: dict[str, dict[str, float | int]] = {}
    for et, acc in by_type.items():
        if acc["n_unique"] == 0:
            continue
        p = acc["n_tp"] / acc["n_attempted"] if acc["n_attempted"] > 0 else -1.0
        r = acc["n_tp"] / acc["n_unique"] if acc["n_unique"] > 0 else -1.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else -1.0
        by_error_type[et] = {
            "n_unique_errors": acc["n_unique"],
            "n_truely_corrected_errors": acc["n_tp"],
            "n_all_corrected_errors": acc["n_attempted"],
            "precision": p,
            "recall": r,
            "f1_score": f,
        }

    return {
        "n_unique_errors": n_unique_errors,
        "n_total_error_cells": len(detections),
        "n_duplicate_errors": n_duplicate_errors,
        "n_majority_voted": n_majority_voted,
        "n_truely_corrected_errors": n_truely_corrected,
        "n_all_corrected_errors": n_attempted_corrections,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "by_error_type": by_error_type,
    }


def evaluate_single_table(
    dirty_path: str,
    clean_path: str,
    repaired_path: str | None,
    provenance_map: dict[tuple[int, int], str],
    table_name: str,
    error_type_map: dict[tuple[int, int], str] | None = None,
) -> dict | None:
    """
    Evaluate one table with majority voting.

    If `repaired_path` is None / missing or cannot be parsed, we fall back to a
    "no output" scenario by using `dirty_df` as `repaired_df` so that:
      - all errors are still counted, but
      - there are no attempted corrections (TP = FP = 0).
    """

    def _load_and_eval(use_dirty_as_repaired: bool) -> dict | None:
        dirty_df_local = read_csv_robust(dirty_path, "dirty")
        clean_df_local = read_csv_robust(clean_path, "clean")

        if use_dirty_as_repaired:
            repaired_df_local = dirty_df_local.copy()
        else:
            repaired_df_local = read_csv_robust(repaired_path, "repaired")  # type: ignore[arg-type]

        # Drop synthetic index column and normalise column names
        dirty_df_local = _sanitize_column_names(_drop_index_column(dirty_df_local))
        clean_df_local = _sanitize_column_names(_drop_index_column(clean_df_local))
        repaired_df_local = _sanitize_column_names(_drop_index_column(repaired_df_local))

        if dirty_df_local.shape != clean_df_local.shape:
            logger.warning(
                "[%s] Shape mismatch between dirty and clean (%s vs %s) – skipping",
                table_name,
                dirty_df_local.shape,
                clean_df_local.shape,
            )
            return None
        results_local = evaluate_with_majority_voting(
            dirty_df_local, clean_df_local, repaired_df_local, provenance_map, error_type_map
        )

        if results_local["n_unique_errors"] == 0:
            logger.warning(
                "[%s] No errors detected (dirty == clean) – skipping", table_name
            )
            return None

        return results_local

    try:
        # First, try normal evaluation (with repaired output if available).
        if repaired_path is not None and os.path.exists(repaired_path):
            return _load_and_eval(use_dirty_as_repaired=False)
        else:
            # No repaired file: treat as no-output table.
            return _load_and_eval(use_dirty_as_repaired=True)

    except Exception as exc:
        # If anything fails (e.g. malformed repaired CSV), try a final fallback
        # where we treat this as a no-output table but still count all errors.
        logger.warning(
            "[%s] Evaluation failed with repaired output (%s). Falling back to no-output evaluation: %s",
            table_name,
            repaired_path,
            exc,
        )
        logger.debug(traceback.format_exc())

        try:
            fallback_results = _load_and_eval(use_dirty_as_repaired=True)
            if fallback_results is not None:
                logger.warning(
                    "[%s] Fallback no-output evaluation succeeded; "
                    "table will be included with TP=0, FP=0.",
                    table_name,
                )
            return fallback_results
        except Exception as exc2:
            logger.warning(
                "[%s] Fallback no-output evaluation also failed: %s",
                table_name,
                exc2,
            )
            logger.debug(traceback.format_exc())
            return None


# ---------------------------------------------------------------------------
# Lake-level aggregation
# ---------------------------------------------------------------------------

def discover_table_dirs(lake_dir: str) -> list[str]:
    """Return sorted list of table directory names under lake_dir."""
    table_names: list[str] = []
    for name in sorted(os.listdir(lake_dir)):
        full = os.path.join(lake_dir, name)
        if not os.path.isdir(full):
            continue
        dirty_path = os.path.join(full, "dirty.csv")
        clean_path = os.path.join(full, "clean.csv")
        if os.path.exists(dirty_path) and os.path.exists(clean_path):
            table_names.append(name)
    return table_names


def count_lake_errors(lake_dir: str) -> tuple[int, int]:
    """
    Count total unique errors across ALL tables in the input lake.
    Returns (total_unique_errors, total_error_cells).
    """
    total_unique = 0
    total_cells = 0

    for table_name in discover_table_dirs(lake_dir):
        table_dir = os.path.join(lake_dir, table_name)
        dirty_path = os.path.join(table_dir, "dirty.csv")
        clean_path = os.path.join(table_dir, "clean.csv")

        try:
            dirty_df = _sanitize_column_names(_drop_index_column(read_csv_robust(dirty_path, "dirty")))
            clean_df = _sanitize_column_names(_drop_index_column(read_csv_robust(clean_path, "clean")))

            if dirty_df.shape != clean_df.shape:
                logger.warning(
                    "[%s] Shape mismatch when counting lake errors (%s vs %s) – skipping",
                    table_name,
                    dirty_df.shape,
                    clean_df.shape,
                )
                continue

            provenance_map = load_provenance_map(table_dir)
            detections = get_dataframes_difference(dirty_df, clean_df)

            provenance_set = set()
            errors_without_prov = 0

            for row_idx, col_idx in detections:
                prov = provenance_map.get((row_idx, col_idx))
                if prov:
                    provenance_set.add(prov)
                else:
                    errors_without_prov += 1

            n_unique = len(provenance_set) + errors_without_prov
            total_unique += n_unique
            total_cells += len(detections)

        except Exception as exc:
            logger.warning(
                "[%s] Could not count errors for lake total: %s", table_name, exc
            )

    return total_unique, total_cells


def count_lake_errors_per_type(lake_dir: str) -> dict[str, int]:
    """
    Count total unique errors per error-type across ALL tables in the lake.

    Mirrors count_lake_errors but also tracks error_type using the same
    provenance de-duplication logic.  Used as the per-type recall denominator
    (analogous to get_lake_ec_tpfn in evaluate_holoclean_lake.py).

    Returns {error_type: n_unique_errors}.
    """
    ec_tpfn_by_type: dict[str, int] = defaultdict(int)

    for table_name in discover_table_dirs(lake_dir):
        table_dir = os.path.join(lake_dir, table_name)
        dirty_path = os.path.join(table_dir, "dirty.csv")
        clean_path = os.path.join(table_dir, "clean.csv")

        try:
            dirty_df = _sanitize_column_names(
                _drop_index_column(read_csv_robust(dirty_path, "dirty"))
            )
            clean_df = _sanitize_column_names(
                _drop_index_column(read_csv_robust(clean_path, "clean"))
            )
            if dirty_df.shape != clean_df.shape:
                continue

            provenance_map  = load_provenance_map(table_dir)
            error_type_map  = load_error_type_map(table_dir)
            detections      = get_dataframes_difference(dirty_df, clean_df)

            def _et(row_idx: int, col_idx: int) -> str:
                et = error_type_map.get((row_idx, col_idx), "").strip()
                return et if et else "unknown"

            # De-duplicate by provenance (same logic as count_lake_errors)
            prov_to_type: dict[str, str] = {}
            for row_idx, col_idx in detections:
                prov = provenance_map.get((row_idx, col_idx))
                et   = _et(row_idx, col_idx)
                if prov:
                    # First occurrence wins for the error type of this provenance key
                    prov_to_type.setdefault(prov, et)
                else:
                    ec_tpfn_by_type[et] += 1

            for et in prov_to_type.values():
                ec_tpfn_by_type[et] += 1

        except Exception as exc:
            logger.warning(
                "[%s] Could not count per-type errors for lake total: %s",
                table_name,
                exc,
            )

    return dict(ec_tpfn_by_type)


def _fmt(v: float) -> str:
    return f"{v:.4f}" if v >= 0 else "  N/A "


def run_lake_evaluation_uniclean(lake_dir: str, save_csv: str | None = None) -> None:
    table_names = discover_table_dirs(lake_dir)
    if not table_names:
        logger.error("No table directories found under %s", lake_dir)
        sys.exit(1)

    logger.info("Found %d table(s) in lake_dir=%s", len(table_names), lake_dir)

    logger.info("Counting total errors across all tables in the lake...")
    total_unique_errors, total_error_cells = count_lake_errors(lake_dir)
    logger.info(
        "Total unique errors in lake: %d (total error cells: %d)",
        total_unique_errors,
        total_error_cells,
    )

    logger.info("Counting per-type errors across all tables in the lake...")
    lake_ec_tpfn_by_type = count_lake_errors_per_type(lake_dir)
    logger.info(
        "Per-type error counts: %s",
        {k: v for k, v in sorted(lake_ec_tpfn_by_type.items())},
    )

    per_table_rows: list[dict] = []
    skipped_tables: list[str] = []

    lake_tp = 0
    lake_corrected = 0
    tables_evaluated = 0

    print("\n" + "=" * 110)
    print(
        f"{'TABLE':<25} {'UNIQ_ERR':>9} {'TOT_CELLS':>10} "
        f"{'DUP':>5} {'MV':>4} {'CORR':>6} {'TP':>6} "
        f"{'PREC':>8} {'REC':>8} {'F1':>8}"
    )
    print("=" * 110)

    for table_name in table_names:
        table_dir = os.path.join(lake_dir, table_name)
        dirty_path = os.path.join(table_dir, "dirty.csv")
        clean_path = os.path.join(table_dir, "clean.csv")
        repaired_path = os.path.join(
            table_dir, "result", table_name, f"{table_name}Cleaned.csv"
        )

        if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
            logger.warning(
                "[%s] dirty.csv or clean.csv not found – skipping", table_name
            )
            skipped_tables.append(table_name)
            continue

        provenance_map = load_provenance_map(table_dir)
        if not provenance_map:
            logger.warning("[%s] No provenance information – skipping", table_name)
            skipped_tables.append(table_name)
            continue

        error_type_map = load_error_type_map(table_dir)

        if os.path.exists(repaired_path):
            res = evaluate_single_table(
                dirty_path,
                clean_path,
                repaired_path,
                provenance_map,
                table_name,
                error_type_map,
            )
        else:
            # No Uniclean output for this table: evaluate as "no corrections"
            logger.warning(
                "[%s] No Uniclean result found at %s – treating as no-output table",
                table_name,
                repaired_path,
            )
            res = evaluate_single_table(
                dirty_path,
                clean_path,
                None,
                provenance_map,
                table_name,
                error_type_map,
            )

        if res is None:
            skipped_tables.append(table_name)
            print(f"  {table_name:<25}  SKIPPED")
            continue

        tables_evaluated += 1
        lake_tp += res["n_truely_corrected_errors"]
        lake_corrected += res["n_all_corrected_errors"]

        per_table_rows.append(
            {
                "table": table_name,
                **res,
            }
        )

        print(
            f"  {table_name:<25} "
            f"{res['n_unique_errors']:>9}  {res['n_total_error_cells']:>10}  "
            f"{res['n_duplicate_errors']:>5}  {res['n_majority_voted']:>4}  "
            f"{res['n_all_corrected_errors']:>6}  {res['n_truely_corrected_errors']:>6}  "
            f"{_fmt(res['precision']):>8}  "
            f"{_fmt(res['recall']):>8}  "
            f"{_fmt(res['f1_score']):>8}"
        )

    # Lake-wide metrics aggregated over unique errors
    lake_precision = (
        lake_tp / lake_corrected if lake_corrected > 0 else -1.0
    )
    lake_recall = (
        lake_tp / total_unique_errors if total_unique_errors > 0 else -1.0
    )
    lake_f1 = (
        2 * lake_precision * lake_recall / (lake_precision + lake_recall)
        if (lake_precision + lake_recall) > 0
        else -1.0
    )

    print("\n" + "=" * 110)
    print("LAKE-WIDE SUMMARY (Uniclean with Majority Voting)")
    print("=" * 110)
    print(f"  Tables evaluated:             {tables_evaluated}")
    print(f"  Skipped:                      {len(skipped_tables)}")
    print(f"  Total unique errors in lake:  {total_unique_errors}")
    print(f"  Total error cells in lake:    {total_error_cells}")
    print(f"  Duplicate error cells:        {total_error_cells - total_unique_errors}")
    print()
    print(f"  Precision: {_fmt(lake_precision)}")
    print(f"  Recall:    {_fmt(lake_recall)}")
    print(f"  F1:        {_fmt(lake_f1)}")
    if skipped_tables:
        print(f"\n  Skipped: {skipped_tables}")

    # Lake-wide effectiveness per error type
    lake_by_type: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "n_unique_errors": 0,
            "n_truely_corrected_errors": 0,
            "n_all_corrected_errors": 0,
        }
    )
    for row in per_table_rows:
        for et, metrics in row.get("by_error_type", {}).items():
            lake_by_type[et]["n_unique_errors"] += metrics["n_unique_errors"]
            lake_by_type[et]["n_truely_corrected_errors"] += metrics[
                "n_truely_corrected_errors"
            ]
            lake_by_type[et]["n_all_corrected_errors"] += metrics[
                "n_all_corrected_errors"
            ]

    effectiveness_rows: list[dict] = []
    all_error_types = set(lake_by_type.keys()) | set(lake_ec_tpfn_by_type.keys())
    if all_error_types:
        for et in sorted(all_error_types):
            acc           = lake_by_type.get(et, {})
            n_tp          = acc.get("n_truely_corrected_errors", 0)
            n_corr        = acc.get("n_all_corrected_errors", 0)
            # Use the lake-wide total as the recall denominator (mirrors HoloClean)
            lake_ec_tpfn  = lake_ec_tpfn_by_type.get(et, 0)
            prec = n_tp / n_corr       if n_corr       > 0 else -1.0
            rec  = n_tp / lake_ec_tpfn if lake_ec_tpfn > 0 else -1.0
            f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else -1.0
            effectiveness_rows.append(
                {
                    "error_type":                et,
                    "lake_ec_tpfn":              lake_ec_tpfn,
                    "n_truely_corrected_errors": n_tp,
                    "n_all_corrected_errors":    n_corr,
                    "precision":                 prec,
                    "recall":                    rec,
                    "f1_score":                  f1,
                }
            )

        print("\n" + "=" * 110)
        print("EFFECTIVENESS PER ERROR TYPE (lake-wide)")
        print("=" * 110)
        print(
            f"  {'ERROR_TYPE':<25} {'LAKE_EC_TPFN':>13} {'CORR':>8} {'TP':>8} "
            f"{'PREC':>10} {'REC':>10} {'F1':>10}"
        )
        print("-" * 110)
        for r in effectiveness_rows:
            print(
                f"  {r['error_type']:<25} {r['lake_ec_tpfn']:>13} "
                f"{r['n_all_corrected_errors']:>8} "
                f"{r['n_truely_corrected_errors']:>8} {_fmt(r['precision']):>10} "
                f"{_fmt(r['recall']):>10} {_fmt(r['f1_score']):>10}"
            )
        print("=" * 110)

    # Optional CSV output
    if save_csv and per_table_rows:
        detail_df = pd.DataFrame(per_table_rows)
        if "by_error_type" in detail_df.columns:
            detail_df = detail_df.drop(columns=["by_error_type"])
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        detail_df.to_csv(save_csv, index=False)
        logger.info("Per-table results saved to: %s", save_csv)

        base, ext = os.path.splitext(save_csv)
        summary_path = f"{base}_summary{ext}"
        summary_df = pd.DataFrame(per_table_rows)
        if "by_error_type" in summary_df.columns:
            summary_df = summary_df.drop(columns=["by_error_type"])
        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    [
                        {
                            "table": "*** LAKE TOTAL ***",
                            "n_unique_errors": total_unique_errors,
                            "n_total_error_cells": total_error_cells,
                            "precision": lake_precision,
                            "recall": lake_recall,
                            "f1_score": lake_f1,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        summary_df.to_csv(summary_path, index=False)
        logger.info("Per-table summary saved to:    %s", summary_path)

        if effectiveness_rows:
            by_type_path = f"{base}_by_error_type{ext}"
            pd.DataFrame(effectiveness_rows).to_csv(by_type_path, index=False)
            logger.info("Effectiveness per error type: %s", by_type_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Uniclean repairs with majority voting for joined lake tables "
        "(adapted from HoloClean evaluate_lake_majority_voting.py)."
    )
    parser.add_argument(
        "--lake-dir",
        "-l",
        required=True,
        help="Root of the input data lake (contains merged table sub-directories with provenance).",
    )
    parser.add_argument(
        "--save-csv",
        "-s",
        help="Optional path to save per-table results as CSV.",
    )
    args = parser.parse_args()

    run_lake_evaluation_uniclean(args.lake_dir, args.save_csv)


if __name__ == "__main__":
    main()

