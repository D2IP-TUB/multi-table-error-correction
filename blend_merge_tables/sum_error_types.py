#!/usr/bin/env python3
"""
Aggregate error type counts from isolated_error_map.csv files.

By default, this script scans:
/home/ahmadi/Blend_X/merged_strings_default_set_union/mit_dwh/merged

You can also pass multiple directories (for example both merged and isolated
outputs), and the script will recursively discover isolated_error_map.csv
files under each provided path.

Expected CSV columns:
- error_type
- occurrence_count (optional; defaults to 1 when missing/invalid)
- source_table, source_row, source_column (used for unique-cell counting)
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_MERGED_DIR = Path(
    "/home/ahmadi/Blend_X/merged_strings_default_set_union/mit_dwh/merged"
)


def iter_error_map_files(input_paths: Iterable[Path]) -> Iterable[Path]:
    """Yield all isolated_error_map.csv files under the given paths."""
    seen: set[Path] = set()
    for input_path in input_paths:
        if input_path.is_file():
            if input_path.name == "isolated_error_map.csv":
                resolved = input_path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield input_path
            continue

        if not input_path.is_dir():
            continue

        for error_map in sorted(input_path.rglob("isolated_error_map.csv")):
            if not error_map.is_file():
                continue
            resolved = error_map.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield error_map


def parse_occurrence_count(raw_value: str | None) -> int:
    """Parse occurrence_count safely; fallback to 1 if missing/invalid."""
    if raw_value is None:
        return 1
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 1
    return value if value >= 0 else 1


def _parse_source_cell_key(row: dict[str, str | None]) -> tuple[str, str, str] | None:
    """Build a source-cell identity key for unique erroneous-cell counting."""
    source_table = (row.get("source_table") or "").strip()
    source_row = (row.get("source_row") or "").strip()
    source_column = (row.get("source_column") or "").strip()

    if not source_table or source_row == "" or not source_column:
        return None
    return source_table, source_row, source_column


def _value_normalizer(value: str) -> str:
    """Normalize values exactly as in count_errors.py for consistency."""
    normalized = html.unescape(value)
    normalized = re.sub(r"[\t\n ]+", " ", normalized, flags=re.UNICODE)
    return normalized.strip("\t\n ")


def count_unique_errors_from_clean_dirty(isolated_dir: Path) -> tuple[int, int]:
    """
    Count unique erroneous source cells from clean/dirty files in isolated tables.

    Returns:
        (unique_error_cells, datasets_processed)
    """
    unique_cells: set[tuple[str, int, str]] = set()
    datasets_processed = 0

    if not isolated_dir.exists() or not isolated_dir.is_dir():
        return 0, 0

    for table_dir in sorted(isolated_dir.iterdir()):
        if not table_dir.is_dir():
            continue
        clean_path = table_dir / "clean.csv"
        dirty_path = table_dir / "dirty.csv"
        if not (clean_path.exists() and dirty_path.exists()):
            continue

        clean_df = pd.read_csv(clean_path, dtype=str, keep_default_na=False)
        dirty_df = pd.read_csv(dirty_path, dtype=str, keep_default_na=False)

        # Mirror count_errors.py behavior exactly.
        clean_df = clean_df.map(_value_normalizer)
        dirty_df = dirty_df.map(_value_normalizer)
        dirty_df.columns = clean_df.columns

        if clean_df.shape != dirty_df.shape:
            continue

        diff = (clean_df != dirty_df)
        for col_name in clean_df.columns:
            for row_idx in diff.index[diff[col_name]].tolist():
                unique_cells.add((table_dir.name, int(row_idx), col_name))

        datasets_processed += 1

    return len(unique_cells), datasets_processed


def _auto_detect_isolated_dir(input_paths: list[Path]) -> Path | None:
    """
    Try to infer tables/<corpus>/isolated from merged_strings_*/*/<corpus>/... paths.
    """
    for input_path in input_paths:
        parts = input_path.parts
        if "merged_strings_default_set_union" in parts:
            idx = parts.index("merged_strings_default_set_union")
            if idx + 1 < len(parts):
                corpus = parts[idx + 1]
                candidate = Path("tables") / corpus / "isolated"
                if candidate.exists() and candidate.is_dir():
                    return candidate
    return None


def aggregate_error_types(input_paths: Iterable[Path]) -> tuple[Counter, Counter, int, int, int]:
    """
    Aggregate counts by error_type.

    Returns:
        (
            weighted_error_type_counter,
            unique_error_type_counter,
            files_processed,
            rows_processed,
            unique_cells_count,
        )
    """
    totals: Counter[str] = Counter()
    unique_totals: Counter[str] = Counter()
    files_processed = 0
    rows_processed = 0
    unique_cells: set[tuple[str, str, str]] = set()
    unique_cells_by_type: set[tuple[str, str, str, str]] = set()

    for error_map_file in iter_error_map_files(input_paths):
        files_processed += 1
        with error_map_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows_processed += 1
                error_type = (row.get("error_type") or "UNKNOWN").strip() or "UNKNOWN"
                count = parse_occurrence_count(row.get("occurrence_count"))
                totals[error_type] += count

                source_key = _parse_source_cell_key(row)
                if source_key is not None:
                    unique_cells.add(source_key)
                    typed_key = (source_key[0], source_key[1], source_key[2], error_type)
                    if typed_key not in unique_cells_by_type:
                        unique_cells_by_type.add(typed_key)
                        unique_totals[error_type] += 1

    return totals, unique_totals, files_processed, rows_processed, len(unique_cells)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sum all error types from isolated_error_map.csv files."
    )
    parser.add_argument(
        "input_paths",
        nargs="*",
        default=[str(DEFAULT_MERGED_DIR)],
        help=(
            "One or more files/directories to scan recursively for "
            "isolated_error_map.csv"
        ),
    )
    parser.add_argument(
        "--source-isolated-dir",
        default=None,
        help=(
            "Optional path to source isolated tables (with clean.csv and dirty.csv) "
            "to compute ground-truth unique erroneous cells."
        ),
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input_paths]
    invalid = [str(p) for p in input_paths if not p.exists()]
    if invalid:
        raise SystemExit(f"Error: input path(s) do not exist: {', '.join(invalid)}")

    totals, unique_totals, files_processed, rows_processed, unique_cells_count = aggregate_error_types(input_paths)

    source_isolated_dir: Path | None = None
    if args.source_isolated_dir:
        source_isolated_dir = Path(args.source_isolated_dir)
    else:
        source_isolated_dir = _auto_detect_isolated_dir(input_paths)

    if files_processed == 0:
        print("No isolated_error_map.csv files found in input path(s):")
        for path in input_paths:
            print(f"- {path}")
        return

    print("Scanned input path(s):")
    for path in input_paths:
        print(f"- {path}")
    print(f"Table folders processed: {files_processed}")
    print(f"Error rows processed: {rows_processed}")
    print("\nTotal errors by type (weighted by occurrence_count):")

    for error_type, total_count in sorted(totals.items(), key=lambda x: (-x[1], x[0])):
        print(f"{error_type}: {total_count}")

    print(f"\nGrand total errors (weighted): {sum(totals.values())}")
    print(f"Grand total unique erroneous source cells (from isolated_error_map): {unique_cells_count}")

    if source_isolated_dir is not None:
        gt_unique_count, gt_datasets = count_unique_errors_from_clean_dirty(source_isolated_dir)
        if gt_datasets > 0:
            print(
                "Grand total unique erroneous source cells "
                f"(from clean-vs-dirty in {source_isolated_dir}): {gt_unique_count}"
            )
        else:
            print(
                "Grand total unique erroneous source cells "
                f"(from clean-vs-dirty): not available (no valid datasets in {source_isolated_dir})"
            )

    if unique_totals:
        print("\nUnique erroneous source cells by type:")
        for error_type, total_count in sorted(unique_totals.items(), key=lambda x: (-x[1], x[0])):
            print(f"{error_type}: {total_count}")
    else:
        print("\nUnique erroneous source cells by type: not available (missing source_* columns).")


if __name__ == "__main__":
    main()
