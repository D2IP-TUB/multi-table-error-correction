#!/usr/bin/env python3
"""
Majority voting evaluation for Uniclean on the joined soccer table.

Algorithm (per column):
  1. Determine provenance: column_provenance.json → split_a or split_b,
     then provenance_map.csv → original isolated row ID for each joined row.
  2. Identify errors: cells where dirty ≠ clean (true errors in the joined table).
  3. Collect votes: for each unique original cell (split, split_row_id, col),
     gather all Uniclean-repaired values from every joined row that references it
     (only where repaired ≠ dirty, i.e. Uniclean changed something).
  4. Majority vote: most-frequent repaired value; ties broken lexicographically.
  5. Evaluate: voted value vs clean value → TP or FP.

Usage:
    python evaluate_joined_soccer_majority_voting_uniclean.py \\
        --soccer-dir datasets_and_rules/joined_soccer/soccer \\
        --table-name soccer_joined_fixed_prov \\
        --save-csv datasets_and_rules/joined_soccer/soccer/result/soccer_joined_fixed_prov/majority_voting_uniclean.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def read_col(path: str, col: str) -> np.ndarray:
    """Read a single column from a CSV as a stripped string numpy array."""
    s = pd.read_csv(
        path,
        usecols=[col],
        dtype=str,
        keep_default_na=False,
        encoding="latin-1",
        low_memory=False,
    )[col]
    return np.char.strip(s.to_numpy(dtype=str))


def read_prov_map(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read provenance_map.csv.
    Returns (split_a_row_ids, split_b_row_ids) as int64 numpy arrays
    indexed by joined_row_id (i.e. position = joined_row_id).
    """
    pmap = pd.read_csv(
        path,
        dtype={"joined_row_id": np.int64,
               "split_a_row_id": np.int64,
               "split_b_row_id": np.int64},
    )
    # Sort by joined_row_id to ensure correct positional alignment
    pmap = pmap.sort_values("joined_row_id").reset_index(drop=True)
    return pmap["split_a_row_id"].to_numpy(), pmap["split_b_row_id"].to_numpy()


# ---------------------------------------------------------------------------
# Per-column majority-vote evaluation
# ---------------------------------------------------------------------------

def evaluate_column(
    dirty_vals: np.ndarray,
    clean_vals: np.ndarray,
    repaired_vals: np.ndarray,
    prov_ids: np.ndarray,
) -> dict[str, int]:
    """
    Apply majority voting for one column.

    Returns dict with keys:
        n_unique_errors, n_attempted, n_tp, n_fp
    """
    # 1. Identify error positions
    is_error = dirty_vals != clean_vals
    n_error_cells = int(is_error.sum())
    if n_error_cells == 0:
        return {"n_unique_errors": 0, "n_attempted": 0, "n_tp": 0, "n_fp": 0}

    err_prov   = prov_ids[is_error]
    err_dirty  = dirty_vals[is_error]
    err_clean  = clean_vals[is_error]
    err_repair = repaired_vals[is_error]

    # 2. Count unique errors (unique provenance IDs among errors)
    n_unique_errors = int(pd.Series(err_prov).nunique())

    # 3. Build vote DataFrame: only rows where Uniclean changed something
    changed = err_repair != err_dirty
    if not changed.any():
        return {
            "n_unique_errors": n_unique_errors,
            "n_attempted": 0,
            "n_tp": 0,
            "n_fp": 0,
        }

    vote_df = pd.DataFrame({
        "prov":    err_prov[changed],
        "repair":  err_repair[changed],
    })

    # Also build ground-truth lookup: prov_id → clean_value
    # (clean is same for all rows with the same prov_id)
    gt_df = pd.DataFrame({
        "prov":  err_prov,
        "clean": err_clean,
    }).drop_duplicates("prov").set_index("prov")["clean"]

    # 4. Count votes per (prov, repair), then sort to pick winner
    #    Sort order: by prov asc, count desc, repair asc (lex tie-break)
    counts = (
        vote_df
        .groupby(["prov", "repair"], sort=False)
        .size()
        .reset_index(name="cnt")
    )
    counts_sorted = counts.sort_values(
        by=["prov", "cnt", "repair"],
        ascending=[True, False, True],
    )
    winners = counts_sorted.groupby("prov", sort=False).first().reset_index()

    # 5. Compare winners to ground truth
    winners["clean"] = winners["prov"].map(gt_df)
    is_tp = (winners["repair"] == winners["clean"]).to_numpy()

    n_attempted = len(winners)
    n_tp = int(is_tp.sum())
    n_fp = n_attempted - n_tp

    return {
        "n_unique_errors": n_unique_errors,
        "n_attempted": n_attempted,
        "n_tp": n_tp,
        "n_fp": n_fp,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run(
    soccer_dir: str,
    table_name: str,
    save_csv: str | None,
) -> None:
    # Load column provenance metadata
    with open(os.path.join(soccer_dir, "column_provenance.json")) as f:
        col_prov_meta = json.load(f)
    column_provenance: dict[str, str] = col_prov_meta["column_provenance"]
    data_columns: list[str] = col_prov_meta["final_columns"]

    logger.info("Final columns:      %s", data_columns)
    logger.info("Column provenance:  %s", column_provenance)

    # Load provenance map (once, reused for all columns)
    logger.info("Loading provenance_map.csv …")
    split_a_ids, split_b_ids = read_prov_map(
        os.path.join(soccer_dir, "provenance_map.csv")
    )
    logger.info("Provenance map loaded: %d joined rows", len(split_a_ids))

    dirty_path    = os.path.join(soccer_dir, "dirty.csv")
    clean_path    = os.path.join(soccer_dir, "clean.csv")
    repaired_path = os.path.join(
        soccer_dir, "result", table_name, f"{table_name}Cleaned.csv"
    )

    # Evaluate per column
    results_by_col: dict[str, dict] = {}
    total_unique = total_attempted = total_tp = total_fp = 0

    for col in data_columns:
        split    = column_provenance[col]
        prov_ids = split_a_ids if split == "split_a" else split_b_ids

        logger.info("Processing column '%s' (provenance: %s) …", col, split)

        d = read_col(dirty_path,    col)
        c = read_col(clean_path,    col)
        r = read_col(repaired_path, col)

        res = evaluate_column(d, c, r, prov_ids)
        results_by_col[col] = {"split": split, **res}

        total_unique    += res["n_unique_errors"]
        total_attempted += res["n_attempted"]
        total_tp        += res["n_tp"]
        total_fp        += res["n_fp"]

        logger.info(
            "  unique_errors=%d  attempted=%d  TP=%d  FP=%d",
            res["n_unique_errors"], res["n_attempted"], res["n_tp"], res["n_fp"],
        )

    # Aggregate metrics
    def _fmt(v: float) -> str:
        return f"{v:.4f}" if v >= 0 else "N/A"

    precision = total_tp / total_attempted if total_attempted > 0 else -1.0
    recall    = total_tp / total_unique    if total_unique    > 0 else -1.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else -1.0
    )

    # Print summary
    W = 80
    print("\n" + "=" * W)
    print("MAJORITY VOTING EVALUATION — Uniclean on joined soccer")
    print("=" * W)
    print(f"  Table:    {table_name}")
    print(f"  Dir:      {soccer_dir}")
    print()
    hdr = f"  {'COLUMN':<12} {'SPLIT':<10} {'UNIQ_ERR':>10} {'ATTEMPTED':>10} {'TP':>7} {'FP':>7} {'PREC':>8} {'REC':>8} {'F1':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for col in data_columns:
        r  = results_by_col[col]
        p_ = r["n_tp"] / r["n_attempted"] if r["n_attempted"] > 0 else -1.0
        rc = r["n_tp"] / r["n_unique_errors"] if r["n_unique_errors"] > 0 else -1.0
        f_ = 2 * p_ * rc / (p_ + rc) if (p_ + rc) > 0 else -1.0
        print(
            f"  {col:<12} {r['split']:<10}"
            f" {r['n_unique_errors']:>10}"
            f" {r['n_attempted']:>10}"
            f" {r['n_tp']:>7}"
            f" {r['n_fp']:>7}"
            f" {_fmt(p_):>8}"
            f" {_fmt(rc):>8}"
            f" {_fmt(f_):>8}"
        )

    print("  " + "-" * (len(hdr) - 2))
    print(
        f"  {'TOTAL':<12} {'':<10}"
        f" {total_unique:>10}"
        f" {total_attempted:>10}"
        f" {total_tp:>7}"
        f" {total_fp:>7}"
        f" {_fmt(precision):>8}"
        f" {_fmt(recall):>8}"
        f" {_fmt(f1):>8}"
    )
    print("=" * W)

    # Save CSV
    if save_csv:
        rows = []
        for col in data_columns:
            r  = results_by_col[col]
            p_ = r["n_tp"] / r["n_attempted"] if r["n_attempted"] > 0 else -1.0
            rc = r["n_tp"] / r["n_unique_errors"] if r["n_unique_errors"] > 0 else -1.0
            f_ = 2 * p_ * rc / (p_ + rc) if (p_ + rc) > 0 else -1.0
            rows.append({
                "column":           col,
                "split":            r["split"],
                "n_unique_errors":  r["n_unique_errors"],
                "n_attempted":      r["n_attempted"],
                "n_tp":             r["n_tp"],
                "n_fp":             r["n_fp"],
                "precision":        round(p_, 6) if p_ >= 0 else "N/A",
                "recall":           round(rc, 6) if rc >= 0 else "N/A",
                "f1_score":         round(f_, 6) if f_ >= 0 else "N/A",
            })
        rows.append({
            "column":           "*** TOTAL ***",
            "split":            "",
            "n_unique_errors":  total_unique,
            "n_attempted":      total_attempted,
            "n_tp":             total_tp,
            "n_fp":             total_fp,
            "precision":        round(precision, 6) if precision >= 0 else "N/A",
            "recall":           round(recall,    6) if recall    >= 0 else "N/A",
            "f1_score":         round(f1,        6) if f1        >= 0 else "N/A",
        })
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        pd.DataFrame(rows).to_csv(save_csv, index=False)
        logger.info("Results saved to: %s", save_csv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Majority voting evaluation for Uniclean on the joined soccer table."
    )
    parser.add_argument(
        "--soccer-dir",
        default=(
            "/home/fatemeh/Uniclean-bench-Result/"
            "datasets_and_rules/joined_soccer/soccer"
        ),
        help="Path to the soccer directory (contains dirty.csv, clean.csv, "
             "column_provenance.json, provenance_map.csv, result/).",
    )
    parser.add_argument(
        "--table-name",
        default="soccer_joined_fixed_prov",
        help="Name of the result table (determines the repaired CSV path).",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional path to save per-column results as CSV.",
    )
    args = parser.parse_args()
    run(args.soccer_dir, args.table_name, args.save_csv)


if __name__ == "__main__":
    main()
