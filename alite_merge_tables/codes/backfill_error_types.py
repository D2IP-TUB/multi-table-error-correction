"""
backfill_error_types.py

Annotates ALITE's *_subsumption_map.csv files with error information from
the per-table error_map.csv files in the isolated dataset.

Why the subsumption map (not the cell source map):
  Erroneous cells were nullified by discover_clusters.py before ALITE ran.
  Null cells never appear as value providers in the cell source map — so the
  cell source map only ever contains clean (non-erroneous) cells.
  The subsumption map, however, contains ALL source rows absorbed into each
  output row, including rows whose erroneous cells were nullified.  This is
  the right place to record which source rows had errors and in which columns,
  enabling downstream majority voting after correction.

Output schema added to subsumption map:
    error_column   — column name where the error occurred (empty if none)
    error_type     — RANDOM_TYPO / FD_VIOLATION (empty if none)
    corrected_value — the clean value from error_map new_value (empty if none)

error_map.csv format  (0-based row indexing — matches source_row directly):
    cell_id, row_number, column_name, old_value, new_value,
    error_type, fd_rule, violated_dependencies

Run from the codes/ directory:
    python backfill_error_types.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from alite_paths import CORPORA, get_paths, isolated_dir

_CORPUS_PATHS = {
    corpus: {
        "fd_output": get_paths(corpus)["fd_output"],
        "isolated": isolated_dir(corpus),
    }
    for corpus in CORPORA
}

_SUBSUMPTION_COLS = ["output_row_number", "source_table", "source_row"]
_ERROR_COLS = ["error_column", "error_type", "corrected_value"]
_OUT_COLS = _SUBSUMPTION_COLS + _ERROR_COLS
_EMPTY_ERRORS = pd.DataFrame(columns=["source_table", "source_row", *_ERROR_COLS])


def load_errors_df(table_name: str, iso_dir: Path) -> pd.DataFrame:
    """Return errors for one source table keyed by (source_table, source_row)."""
    folder = table_name.removesuffix(".csv")
    error_map_path = iso_dir / folder / "error_map.csv"
    if not error_map_path.exists():
        return _EMPTY_ERRORS.copy()

    df = pd.read_csv(
        error_map_path,
        dtype=str,
        keep_default_na=False,
        encoding="latin1",
        usecols=["row_number", "column_name", "new_value", "error_type"],
    )
    source_row = pd.to_numeric(df["row_number"], errors="coerce")
    mask = source_row.notna()
    if not mask.any():
        return _EMPTY_ERRORS.copy()

    return pd.DataFrame({
        "source_table": table_name,
        "source_row": source_row[mask].astype(int).to_numpy(),
        "error_column": df.loc[mask, "column_name"].str.strip().str.lower().to_numpy(),
        "error_type": df.loc[mask, "error_type"].str.strip().to_numpy(),
        "corrected_value": df.loc[mask, "new_value"].str.strip().to_numpy(),
    })


def _errors_for_tables(table_names: list[str], iso_dir: Path) -> pd.DataFrame:
    parts = [load_errors_df(name, iso_dir) for name in table_names]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return _EMPTY_ERRORS.copy()
    return pd.concat(parts, ignore_index=True)


def backfill_subsumption(sub_path: Path, iso_dir: Path) -> None:
    """
    Expand the subsumption map: for each (output_row, source_table, source_row)
    add one row per error found in that source row.  Rows with no errors keep a
    single record with empty error columns.
    """
    started = time.perf_counter()
    sub_df = pd.read_csv(sub_path, dtype=str, keep_default_na=False, encoding="latin1")
    # Re-run safe: ignore columns from a previous backfill pass.
    sub_df = sub_df[_SUBSUMPTION_COLS].copy()
    sub_df["source_row"] = pd.to_numeric(sub_df["source_row"], errors="coerce").astype(int)

    table_names = sub_df["source_table"].unique().tolist()
    errors_df = _errors_for_tables(table_names, iso_dir)

    if errors_df.empty:
        result = sub_df.copy()
        for col in _ERROR_COLS:
            result[col] = ""
    else:
        result = sub_df.merge(errors_df, on=["source_table", "source_row"], how="left")
        for col in _ERROR_COLS:
            result[col] = result[col].fillna("")

    result = result[_OUT_COLS]
    added = int((result["error_column"] != "").sum())
    result.to_csv(sub_path, index=False, encoding="latin1")
    elapsed = time.perf_counter() - started
    print(
        f"  {sub_path.name}: {len(sub_df)} rows in, {len(result)} rows out, "
        f"{added} error annotations ({elapsed:.1f}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=_CORPUS_PATHS, default="open_data_uk")
    parser.add_argument("--file", type=Path, default=None, help="Backfill only this subsumption map")
    args = parser.parse_args()
    fd_output_dir = _CORPUS_PATHS[args.corpus]["fd_output"]
    iso = _CORPUS_PATHS[args.corpus]["isolated"]

    if args.file:
        sub_files = [args.file.resolve()]
    else:
        sub_files = sorted(fd_output_dir.glob("*_subsumption_map.csv"))

    print(f"Corpus: {args.corpus}")
    print(f"Found {len(sub_files)} subsumption map file(s) in {fd_output_dir}\n")
    for f in sub_files:
        backfill_subsumption(f, iso)
    print("\nDone.")


if __name__ == "__main__":
    main()
