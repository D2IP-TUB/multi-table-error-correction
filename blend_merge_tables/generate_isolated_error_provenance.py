#!/usr/bin/env python3
"""
Generate a 0-based error provenance file for every isolated table.

Which cells are errors: always the dirty/clean diff with normalization,
matching count_errors.py exactly (HTML-unescape + whitespace collapse).

Error type annotation: looked up from clean_changes_provenance.csv (preferred)
or clean_changes.csv. Cells not found in either map get error_type=UNKNOWN.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path

import pandas as pd

from recreate_as_strings import (
    _infer_error_type_from_clean_changes_codes,
    _resolve_error_type,
)

OUTPUT_CSV = "error_map.csv"
OUTPUT_SUMMARY = "error_map_summary.json"
OUTPUT_ALL_CSV = "error_map_all_tables.csv"

CSV_COLS = [
    "cell_id", "row_number", "column_name", "old_value", "new_value",
    "error_type", "fd_rule", "violated_dependencies",
]
CSV_COLS_ALL = ["table_id"] + CSV_COLS

_UNKNOWN_ANNOTATION = {"error_type": "UNKNOWN", "fd_rule": "", "violated_dependencies": ""}


def _normalize(value: str) -> str:
    """
    Matches count_errors.py::value_normalizer exactly.
    Two values are considered equal (no error) iff their normalized forms match.
    """
    value = html.unescape(value)
    value = re.sub(r"[\t\n ]+", " ", value)
    return value.strip("\t\n ")


def _parse_cell_id_dirty(cell_id: str, row_fallback: str, col_fallback: str) -> tuple[int, str] | None:
    """
    Parse a potentially dirty cell_id of the form "row_1based.col_name".
    cell_id may have data garbage appended after a comma, e.g.:
        "1261.full_name,\"Kimball, Richard W\""
    Falls back to row_fallback / col_fallback fields when cell_id is missing or malformed.
    Returns (row_0based, col_name) or None on unrecoverable parse failure.
    """
    if cell_id:
        parts = cell_id.split('.', 1)
        if len(parts) == 2:
            try:
                row_0based = int(parts[0]) - 1
            except ValueError:
                try:
                    row_0based = int(row_fallback) - 1
                except ValueError:
                    return None
            col_name = parts[1].split(',')[0].strip()
            return row_0based, col_name

    try:
        row_0based = int(row_fallback) - 1
    except (ValueError, TypeError):
        return None
    col_name = (col_fallback or '').split(',')[0].strip()
    return row_0based, col_name


# ---------------------------------------------------------------------------
# Type-annotation map builders — tell us WHY a cell is an error, not WHETHER
# ---------------------------------------------------------------------------

def _build_type_map_from_provenance(path: Path) -> dict[tuple[int, str], dict]:
    """
    Build (row_0based, col_name) -> annotation from clean_changes_provenance.csv.
    Handles dirty cell_id values with comma-appended garbage.
    """
    prov_df = pd.read_csv(path, dtype=str, keep_default_na=False)
    type_map: dict[tuple[int, str], dict] = {}
    for _, row in prov_df.iterrows():
        parsed = _parse_cell_id_dirty(
            row.get('cell_id', ''),
            row.get('row_number', ''),
            row.get('column_name', ''),
        )
        if parsed is None:
            continue
        row_0based, col_name = parsed
        error_type = _resolve_error_type(
            row.get('error_type', ''),
            row.get('violated_dependencies', ''),
        )
        type_map[(row_0based, col_name)] = {
            "error_type": error_type,
            "fd_rule": row.get('fd_rule', ''),
            "violated_dependencies": row.get('violated_dependencies', ''),
        }
    return type_map


def _build_type_map_from_clean_changes(path: Path) -> dict[tuple[int, str], dict]:
    """
    Build (row_0based, col_name) -> annotation from clean_changes.csv.
    Handles dirty cell_id values with comma-appended garbage.
    """
    type_map: dict[tuple[int, str], dict] = {}
    with open(path, encoding="latin1", newline="") as f:
        for row in csv.reader(f):
            if not row or len(row) < 4:
                continue
            cell_id = row[0]
            error_codes = ",".join(row[3:]).strip()
            parts = cell_id.split(".")
            if len(parts) < 2:
                continue
            try:
                row_0based = int(parts[0]) - 1
            except ValueError:
                continue
            col_name = ".".join(parts[1:]).split(',')[0].strip()
            error_type = _resolve_error_type(
                _infer_error_type_from_clean_changes_codes(error_codes), error_codes
            )
            type_map[(row_0based, col_name)] = {
                "error_type": error_type,
                "fd_rule": "",
                "violated_dependencies": error_codes,
            }
    return type_map


# ---------------------------------------------------------------------------
# Ground-truth error detection — matches count_errors.py exactly
# ---------------------------------------------------------------------------

def _build_records(
    dirty_path: Path,
    clean_path: Path,
    type_map: dict[tuple[int, str], dict],
) -> list[dict]:
    """
    Diff dirty vs clean using normalized comparison (same as count_errors.py).
    For each cell that differs, look up its error type from type_map.
    Cells absent from type_map get error_type=UNKNOWN.
    """
    dirty_df = pd.read_csv(dirty_path, dtype=str, keep_default_na=False, encoding="latin1")
    clean_df = pd.read_csv(clean_path, dtype=str, keep_default_na=False, encoding="latin1")
    if dirty_df.shape != clean_df.shape or list(dirty_df.columns) != list(clean_df.columns):
        return []

    records = []
    for row_idx in range(len(dirty_df)):
        for col_name in dirty_df.columns:
            old_val = dirty_df.iloc[row_idx][col_name]
            new_val = clean_df.iloc[row_idx][col_name]
            if _normalize(old_val) == _normalize(new_val):
                continue
            annotation = type_map.get((row_idx, col_name), _UNKNOWN_ANNOTATION)
            records.append({
                "cell_id": f"{row_idx}.{col_name}",
                "row_number": row_idx,
                "column_name": col_name,
                "old_value": old_val,
                "new_value": new_val,
                **annotation,
            })
    return records


# ---------------------------------------------------------------------------
# Per-table processing
# ---------------------------------------------------------------------------

def process_table(table_dir: Path, write_summary: bool) -> list[dict] | None:
    """
    Write error_map.csv (0-based) for one table.
    Returns the records list (with 'table_id' added) on success, None if skipped.
    """
    dirty_path = table_dir / "dirty.csv"
    clean_path = table_dir / "clean.csv"
    if not dirty_path.exists() or not clean_path.exists():
        return None

    prov_path = table_dir / "clean_changes_provenance.csv"
    cc_path = table_dir / "clean_changes.csv"
    if prov_path.exists():
        type_map = _build_type_map_from_provenance(prov_path)
    elif cc_path.exists():
        type_map = _build_type_map_from_clean_changes(cc_path)
    else:
        type_map = {}

    records = _build_records(dirty_path, clean_path, type_map)

    out_path = table_dir / OUTPUT_CSV
    with open(out_path, "w", encoding="latin1", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(records)

    if write_summary and records:
        from collections import Counter
        summary = {
            "total_changes": len(records),
            "error_type_counts": dict(Counter(r["error_type"] for r in records)),
            "columns_affected": sorted({r["column_name"] for r in records}),
        }
        with open(table_dir / OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    table_id = table_dir.name
    return [{"table_id": table_id, **r} for r in records]


def _print_report(all_records: list[dict], n_tables: int, n_total: int) -> None:
    from collections import Counter
    counts = Counter(r["error_type"] for r in all_records)
    total_errors = len(all_records)

    print(f"\n{'=' * 56}")
    print(f"  ERROR TYPE REPORT  —  {n_tables} tables processed / {n_total} found")
    print(f"{'=' * 56}")
    print(f"  {'Error type':<22}  {'Count':>8}  {'Share':>7}")
    print(f"  {'-' * 22}  {'-' * 8}  {'-' * 7}")
    for error_type, count in sorted(counts.items(), key=lambda x: -x[1]):
        share = count / total_errors * 100 if total_errors else 0
        print(f"  {error_type:<22}  {count:>8,}  {share:>6.1f}%")
    print(f"  {'─' * 22}  {'─' * 8}  {'─' * 7}")
    print(f"  {'TOTAL':<22}  {total_errors:>8,}  {'100.0%':>7}")
    print(f"{'=' * 56}\n")


def main():
    p = argparse.ArgumentParser(
        description=(
            f"Generate 0-based {OUTPUT_CSV} for isolated tables. "
            f"Errors are determined by dirty/clean diff (matching count_errors.py). "
            f"Error types are annotated from clean_changes_provenance.csv or clean_changes.csv. "
            f"Also writes {OUTPUT_ALL_CSV} in the root dir with all tables combined."
        )
    )
    p.add_argument(
        "isolated_dir", type=Path, nargs="?",
        default=Path("/home/ahmadi/Blend_X/tables/uk_open_data/isolated"),
    )
    p.add_argument("--summary", action="store_true", help=f"Write {OUTPUT_SUMMARY} per table")
    args = p.parse_args()

    root = args.isolated_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    subdirs = sorted(d for d in root.iterdir() if d.is_dir())

    all_records: list[dict] = []
    n_processed = 0
    for d in subdirs:
        result = process_table(d, args.summary)
        if result is not None:
            n_processed += 1
            all_records.extend(result)

    all_out = root / OUTPUT_ALL_CSV
    with open(all_out, "w", encoding="latin1", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS_ALL)
        w.writeheader()
        w.writerows(all_records)

    print(f"Wrote {OUTPUT_CSV} for {n_processed} tables (of {len(subdirs)} subdirs).")
    print(f"Wrote combined map: {all_out}  ({len(all_records):,} total error records)")
    _print_report(all_records, n_processed, len(subdirs))


if __name__ == "__main__":
    main()
