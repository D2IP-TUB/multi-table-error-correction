#!/usr/bin/env python3
"""
Evaluate Horizon repair per original table: isolated vs blended (majority voting).

For each original table referenced in a merge_summary CSV:
  - Isolated:  standard precision/recall/F1 from the table's own dirty/clean/repaired
  - Blended:   provenance-based majority-voting P/R/F1: for each source cell (provenance),
               collect repaired and clean values across blended rows, take most-frequent
               repaired value as prediction and most-frequent clean as ground truth;
               one TP/FP per provenance. Same logic as evaluate_horizon_majority_voting_by_error_type.py.

This lets you compare per-table repair quality before and after joining/union.

Usage:
  python evaluate_per_original_table.py \\
    --merge-summary blend/merge_summary_opendata.csv \\
    --original-root OpenData/open_data_uk_filtered \\
    --blend-root blend/uk_open_data_blend \\
    [--repaired clean.csv.a2.clean] [--csv out.csv] [--json out.json] [--sort table|delta_f1]
"""

import argparse
import csv
import json
import logging
import os
from collections import Counter, defaultdict

from utils import read_csv

logger = logging.getLogger(__name__)

PROVENANCE_SEP = " \u00a7 "          # " § "  (table § col § row)
REPAIRED_SUFFIX = "clean.csv.a2.clean"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(v):
    return str(v).strip() if v is not None else ""


def _eq(a, b):
    """Treat empty/null as equal (same convention as utils.evaluate)."""
    a, b = _norm(a), _norm(b)
    return a == b or (len(a) == 0 and len(b) == 0)


def _metrics(tp, tpfp, tpfn):
    p = tp / tpfp if tpfp else -1
    r = tp / tpfn if tpfn else -1
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else -1
    return p, r, f1


def _fmt(v, w=8):
    return f"{v:{w}.4f}" if v >= 0 else f"{'N/A':>{w}}"


# ---------------------------------------------------------------------------
# Merge-summary parsing
# ---------------------------------------------------------------------------

def parse_merge_summary(path):
    """
    Return the set of all original table names referenced in the merge summary.
    """
    tables = set()
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            left = row.get("left_table", "").strip()
            right = row.get("right_table", "").strip()
            if left:
                tables.add(left)
            if right:
                tables.add(right)
    return tables


# ---------------------------------------------------------------------------
# Isolated (pre-join) evaluation
# ---------------------------------------------------------------------------

def evaluate_isolated(table_dir, repaired_name=REPAIRED_SUFFIX):
    """
    Standard cell-level evaluation on an original table.
    Returns dict  {tp, fp, tpfp, tpfn, precision, recall, f1_score}  or None.
    Follows the same logic as utils.evaluate.
    """
    dirty_path    = os.path.join(table_dir, "dirty.csv")
    clean_path    = os.path.join(table_dir, "clean.csv")
    repaired_path = os.path.join(table_dir, repaired_name)
    for p in [dirty_path, clean_path, repaired_path]:
        if not os.path.exists(p):
            logger.debug("Missing %s", p)
            return None

    dirty_df    = read_csv(dirty_path)
    clean_df    = read_csv(clean_path)
    repaired_df = read_csv(repaired_path)
    if "_tid_" in repaired_df.columns:
        repaired_df = repaired_df.drop(columns=["_tid_"])
    if dirty_df.shape != clean_df.shape or dirty_df.shape != repaired_df.shape:
        logger.warning("Shape mismatch in %s: dirty=%s clean=%s repaired=%s",
                        table_dir, dirty_df.shape, clean_df.shape, repaired_df.shape)
        return None

    tp = fp = tpfn = 0
    for r in range(len(dirty_df)):
        for c in range(len(dirty_df.columns)):
            d  = _norm(dirty_df.iloc[r, c])
            cl = _norm(clean_df.iloc[r, c])
            if d == cl:
                continue
            tpfn += 1
            rep = _norm(repaired_df.iloc[r, c])
            if d != rep:                         # correction attempted
                if _eq(rep, cl):
                    tp += 1
                else:
                    fp += 1

    tpfp = tp + fp
    p, rec, f1 = _metrics(tp, tpfp, tpfn)
    return {"tp": tp, "fp": fp, "tpfp": tpfp, "tpfn": tpfn,
            "precision": p, "recall": rec, "f1_score": f1}


# ---------------------------------------------------------------------------
# Blended (post-join) collection
# ---------------------------------------------------------------------------

def _most_frequent(values):
    """Return most frequent value; break ties deterministically by lexical order."""
    if not values:
        return None
    counts = Counter(values)
    max_count = max(counts.values())
    winners = [v for v, c in counts.items() if c == max_count]
    return sorted(winners)[0]


def collect_blended_votes(blend_root, repaired_name=REPAIRED_SUFFIX):
    """
    Scan ALL blended tables.  For every cell that is an error (dirty != clean),
    record the provenance key under the originating table.  For every corrected
    error (dirty != repaired), collect the repaired value and clean value.

    Returns
    -------
    provenance_values :  dict  original_table -> { prov_key -> {'repaired_values': [...], 'clean_values': [...]} }
    errors : dict  original_table -> set of error provenance keys  (for TP+FN)
    """
    provenance_values = defaultdict(lambda: defaultdict(lambda: {"repaired_values": [], "clean_values": []}))
    errors = defaultdict(set)

    blend_dirs = sorted(
        d for d in os.listdir(blend_root)
        if os.path.isdir(os.path.join(blend_root, d))
    )
    logger.info("Scanning %d blended tables in %s ...", len(blend_dirs), blend_root)

    for i, bid in enumerate(blend_dirs):
        table_dir     = os.path.join(blend_root, bid)
        dirty_path    = os.path.join(table_dir, "dirty.csv")
        clean_path    = os.path.join(table_dir, "clean.csv")
        repaired_path = os.path.join(table_dir, repaired_name)
        prov_path     = os.path.join(table_dir, "provenance.csv")

        if not all(os.path.exists(p) for p in [dirty_path, clean_path, prov_path]):
            continue

        dirty_df = read_csv(dirty_path)
        clean_df = read_csv(clean_path)
        prov_df  = read_csv(prov_path)
        has_repaired = os.path.exists(repaired_path)
        repaired_df = None
        if has_repaired:
            repaired_df = read_csv(repaired_path)
            if "_tid_" in repaired_df.columns:
                repaired_df = repaired_df.drop(columns=["_tid_"])

        if dirty_df.shape != clean_df.shape or dirty_df.shape != prov_df.shape:
            logger.debug("Shape mismatch in blend %s, skipping", bid)
            continue
        if repaired_df is not None and dirty_df.shape != repaired_df.shape:
            logger.debug("Repaired shape mismatch in blend %s, skipping repaired", bid)
            repaired_df = None

        for r in range(len(dirty_df)):
            for c in range(len(dirty_df.columns)):
                prov_str = _norm(prov_df.iloc[r, c])
                if not prov_str or PROVENANCE_SEP not in prov_str:
                    continue

                prov_key = prov_str.split("|")[0].strip()
                parts = prov_key.split(PROVENANCE_SEP)
                if len(parts) < 3:
                    continue
                orig_table = parts[0].strip()

                d  = _norm(dirty_df.iloc[r, c])
                cl = _norm(clean_df.iloc[r, c])

                if d != cl:                         # error cell
                    errors[orig_table].add(prov_key)

                    if repaired_df is not None:
                        rep = _norm(repaired_df.iloc[r, c])
                        if d != rep:                 # correction attempted
                            entry = provenance_values[orig_table][prov_key]
                            entry["repaired_values"].append(rep)
                            entry["clean_values"].append(cl)

        if (i + 1) % 10 == 0 or i == 0:
            logger.info("  Scanned blend %d/%d (%s)", i + 1, len(blend_dirs), bid)

    return provenance_values, errors


# ---------------------------------------------------------------------------
# Majority voting (value-based: most frequent repaired vs most frequent clean per provenance)
# ---------------------------------------------------------------------------

def majority_vote_per_provenance(prov_values):
    """
    For each provenance key, voted_repaired = most_frequent(repaired_values),
    voted_clean = most_frequent(clean_values). TP if _eq(voted_repaired, voted_clean), else FP.
    Returns (tp, fp). Consistent with evaluate_horizon_majority_voting_by_error_type.py.
    """
    tp = fp = 0
    for values in prov_values.values():
        predicted_value = _most_frequent(values.get("repaired_values", []))
        clean_value = _most_frequent(values.get("clean_values", []))
        if predicted_value is None or clean_value is None:
            continue
        if _eq(predicted_value, clean_value):
            tp += 1
        else:
            fp += 1
    return tp, fp


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(merge_summary_path, original_root, blend_root,
        repaired_name=REPAIRED_SUFFIX, out_csv=None, out_json=None,
        sort_by="table"):
    """
    For every original table in the merge summary, compute isolated and
    blended (majority-voting) metrics, then print & optionally save results.
    sort_by: 'table' (alphabetical) or 'delta_f1' (blend F1 − iso F1, largest first).
    """
    # 1. Original table names from merge summary
    logger.info("Parsing merge summary: %s", merge_summary_path)
    all_tables = parse_merge_summary(merge_summary_path)
    logger.info("Found %d original tables in merge summary", len(all_tables))

    # 2. Collect blended provenance value sets across all blended tables
    blend_provenance_values, blend_errors = collect_blended_votes(blend_root, repaired_name)
    logger.info("Collected blended provenance values for %d original tables", len(blend_provenance_values))

    # 3. Per-table evaluation
    results = []
    for table_name in sorted(all_tables):
        table_dir = os.path.join(original_root, table_name)

        # --- Isolated ---
        iso = evaluate_isolated(table_dir, repaired_name)

        # --- Blended (value-based majority voting: most frequent repaired vs clean per provenance) ---
        prov_values = blend_provenance_values.get(table_name, {})
        error_provs = blend_errors.get(table_name, set())
        tpfn_blend = len(error_provs)

        if prov_values:
            tp_b, fp_b = majority_vote_per_provenance(prov_values)
            tpfp_b = tp_b + fp_b
        else:
            tp_b = fp_b = tpfp_b = 0
        p_b, r_b, f1_b = _metrics(tp_b, tpfp_b, tpfn_blend)

        row = {
            "table": table_name,
            # Isolated
            "iso_tp":        iso["tp"]        if iso else -1,
            "iso_fp":        iso["fp"]        if iso else -1,
            "iso_tpfp":      iso["tpfp"]      if iso else -1,
            "iso_tpfn":      iso["tpfn"]      if iso else -1,
            "iso_precision": iso["precision"] if iso else -1,
            "iso_recall":    iso["recall"]    if iso else -1,
            "iso_f1":        iso["f1_score"]  if iso else -1,
            # Blended (majority voting)
            "blend_tp":        tp_b,
            "blend_fp":        fp_b,
            "blend_tpfp":      tpfp_b,
            "blend_tpfn":      tpfn_blend,
            "blend_precision": p_b,
            "blend_recall":    r_b,
            "blend_f1":        f1_b,
        }
        results.append(row)
        logger.info("  %s: iso=(P=%s R=%s F1=%s) blend=(P=%s R=%s F1=%s)",
                     table_name,
                     _fmt(row["iso_precision"], 6),
                     _fmt(row["iso_recall"], 6),
                     _fmt(row["iso_f1"], 6),
                     _fmt(row["blend_precision"], 6),
                     _fmt(row["blend_recall"], 6),
                     _fmt(row["blend_f1"], 6))

    results = sort_results(results, sort_by=sort_by)

    # 4. Micro-aggregated totals
    iso_tp_all   = sum(r["iso_tp"]   for r in results if r["iso_tp"]   >= 0)
    iso_tpfp_all = sum(r["iso_tpfp"] for r in results if r["iso_tpfp"] >= 0)
    iso_tpfn_all = sum(r["iso_tpfn"] for r in results if r["iso_tpfn"] >= 0)
    iso_p, iso_r, iso_f1 = _metrics(iso_tp_all, iso_tpfp_all, iso_tpfn_all)

    blend_tp_all   = sum(r["blend_tp"]   for r in results)
    blend_tpfp_all = sum(r["blend_tpfp"] for r in results)
    blend_tpfn_all = sum(r["blend_tpfn"] for r in results)
    blend_p, blend_r, blend_f1 = _metrics(blend_tp_all, blend_tpfp_all, blend_tpfn_all)

    aggregated = {
        "iso_precision": iso_p, "iso_recall": iso_r, "iso_f1": iso_f1,
        "iso_tp": iso_tp_all, "iso_tpfp": iso_tpfp_all, "iso_tpfn": iso_tpfn_all,
        "blend_precision": blend_p, "blend_recall": blend_r, "blend_f1": blend_f1,
        "blend_tp": blend_tp_all, "blend_tpfp": blend_tpfp_all, "blend_tpfn": blend_tpfn_all,
        "n_tables": len(results),
    }

    # 5. Output
    if out_csv:
        fields = list(results[0].keys()) if results else []
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow(r)
        logger.info("Wrote per-table CSV to %s", out_csv)

    if out_json:
        with open(out_json, "w") as f:
            json.dump({"aggregated": aggregated, "per_table": results}, f, indent=2)
        logger.info("Wrote JSON to %s", out_json)

    return aggregated, results


def sort_results(results, sort_by="table"):
    """Return a new list sorted by table name (default) or by blend F1 minus iso F1 (desc)."""
    if sort_by == "table":
        return sorted(results, key=lambda r: r["table"])
    if sort_by == "delta_f1":

        def key(r):
            iso, bld = r["iso_f1"], r["blend_f1"]
            if iso >= 0 and bld >= 0:
                return (0, -(bld - iso), r["table"])
            return (1, r["table"])

        return sorted(results, key=key)
    raise ValueError(f"unknown sort_by: {sort_by!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_results(aggregated, results):
    hdr = (f"{'Table':<40} "
           f"{'iso_P':>8} {'iso_R':>8} {'iso_F1':>8} "
           f"{'bld_P':>8} {'bld_R':>8} {'bld_F1':>8} "
           f"{'iso_TP':>7} {'iso_TPFP':>9} {'iso_TPFN':>9} "
           f"{'bld_TP':>7} {'bld_TPFP':>9} {'bld_TPFN':>9}")
    sep = "-" * len(hdr)

    print("\n=== PER-ORIGINAL-TABLE: ISOLATED vs BLENDED (majority voting) ===")
    print(hdr)
    print(sep)
    for r in results:
        print(f"{r['table']:<40} "
              f"{_fmt(r['iso_precision']):>8} "
              f"{_fmt(r['iso_recall']):>8} "
              f"{_fmt(r['iso_f1']):>8} "
              f"{_fmt(r['blend_precision']):>8} "
              f"{_fmt(r['blend_recall']):>8} "
              f"{_fmt(r['blend_f1']):>8} "
              f"{r['iso_tp']:>7} {r['iso_tpfp']:>9} {r['iso_tpfn']:>9} "
              f"{r['blend_tp']:>7} {r['blend_tpfp']:>9} {r['blend_tpfn']:>9}")
    print(sep)

    a = aggregated
    print("\n=== MICRO-AGGREGATED ===")
    print(f"  Isolated:  Precision={_fmt(a['iso_precision'])}  "
          f"Recall={_fmt(a['iso_recall'])}  F1={_fmt(a['iso_f1'])}  "
          f"TP={a['iso_tp']}  TP+FP={a['iso_tpfp']}  TP+FN={a['iso_tpfn']}")
    print(f"  Blended:   Precision={_fmt(a['blend_precision'])}  "
          f"Recall={_fmt(a['blend_recall'])}  F1={_fmt(a['blend_f1'])}  "
          f"TP={a['blend_tp']}  TP+FP={a['blend_tpfp']}  TP+FN={a['blend_tpfn']}")
    print(f"  Tables evaluated: {a['n_tables']}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Compare Horizon per-original-table: isolated vs blended (majority voting)"
    )
    parser.add_argument("--merge-summary", required=True,
                        help="Path to merge_summary CSV (e.g. blend/merge_summary_opendata.csv)")
    parser.add_argument("--original-root", required=True,
                        help="Root dir of original (isolated) tables "
                             "(e.g. OpenData/open_data_uk_filtered)")
    parser.add_argument("--blend-root", required=True,
                        help="Root dir of blended tables (e.g. blend/uk_open_data_blend)")
    parser.add_argument("--repaired", default=REPAIRED_SUFFIX,
                        help="Repaired filename (default: %(default)s)")
    parser.add_argument("--csv", default=None,
                        help="Output CSV path for per-table results")
    parser.add_argument("--json", default=None,
                        help="Output JSON path for aggregated + per-table")
    parser.add_argument(
        "--sort",
        choices=("table", "delta_f1"),
        default="table",
        help="Order per-table rows: by name, or by (blend F1 − iso F1) descending "
             "(rows with missing F1 last). Applies to console and --csv/--json.",
    )
    args = parser.parse_args()

    agg, per_table = run(
        merge_summary_path=args.merge_summary,
        original_root=args.original_root,
        blend_root=args.blend_root,
        repaired_name=args.repaired,
        out_csv=args.csv,
        out_json=args.json,
        sort_by=args.sort,
    )
    print_results(agg, per_table)
