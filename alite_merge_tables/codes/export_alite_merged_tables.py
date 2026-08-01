#!/usr/bin/env python3
"""
export_alite_merged_tables.py

Export each ALITE FD cluster as its own directory with dirty/clean merged CSVs,
mirroring the BLEND multiset-union layout under datasets/real_lakes/..._merged_multiset_union/.

For every cluster in fd_output/:
  - dirty.csv              — FD-integrated table (ALITE output)
  - clean.csv              — cell-by-cell clean values from source provenance
  - merged_cell_source_map.csv
  - subsumption_map.csv
  - clean_changes_provenance.csv   — merged cells where dirty != clean
  - isolated_error_map.csv         — source errors appearing in the merge
  - provenance.csv                 — BLEND-style grid (table § col § row)

Usage (from codes/):
    python export_alite_merged_tables.py --corpus open_data_uk
    python export_alite_merged_tables.py --corpus mit_dwh --cluster cluster_0007
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_BLEND = _REPO / "blend_merge_tables"
if str(_BLEND) not in sys.path:
    sys.path.insert(0, str(_BLEND))

from alite_paths import CORPORA, get_paths, isolated_dir  # noqa: E402

ENCODING = "latin1"
NULL_STRINGS = {"", "nan", "none", "null"}
_CHUNK_SIZE = 2_000_000
_CELL_MAP_COLS = ["row_number", "column_name", "source_table", "source_row", "source_column"]
_SUB_ERROR_COLS = [
    "output_row_number", "source_table", "source_row",
    "error_column", "error_type", "corrected_value",
]


def _cell_str(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    if text.lower() in NULL_STRINGS:
        return ""
    return text


def _table_folder_name(source_table: str) -> str:
    return source_table.removesuffix(".csv")


def _load_source_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding=ENCODING, dtype=str, keep_default_na=False)


class SourceCache:
    def __init__(self, isolated_root: Path):
        self._root = isolated_root
        self._clean: dict[str, pd.DataFrame] = {}
        self._error_map: dict[str, dict[tuple[int, str], str]] = {}

    def clean(self, source_table: str) -> pd.DataFrame | None:
        key = _table_folder_name(source_table)
        if key not in self._clean:
            path = self._root / key / "clean.csv"
            if not path.exists():
                return None
            self._clean[key] = _load_source_table(path)
        return self._clean[key]

    def error_types(self, source_table: str) -> dict[tuple[int, str], str]:
        key = _table_folder_name(source_table)
        if key not in self._error_map:
            path = self._root / key / "error_map.csv"
            mapping: dict[tuple[int, str], str] = {}
            if path.exists():
                df = pd.read_csv(
                    path, encoding=ENCODING, dtype=str, keep_default_na=False,
                    usecols=["row_number", "column_name", "error_type"],
                )
                row_idx = pd.to_numeric(df["row_number"], errors="coerce")
                mask = row_idx.notna()
                if mask.any():
                    cols = df.loc[mask, "column_name"].astype(str).str.strip().str.lower()
                    types = df.loc[mask, "error_type"].astype(str).str.strip()
                    for r, c, t in zip(row_idx[mask].astype(int), cols, types):
                        mapping[(r, c)] = t
            self._error_map[key] = mapping
        return self._error_map[key]


def _cluster_names(fd_output: Path) -> list[str]:
    names = []
    for path in sorted(fd_output.glob("*.csv")):
        stem = path.stem
        if stem.endswith("_merged_cell_source_map") or stem.endswith("_subsumption_map"):
            continue
        names.append(stem)
    return names


def _duckdb_csv_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _providers_from_cell_map(path: Path) -> pd.DataFrame:
    """
    First provider per (output_row, column_name) without loading the full cell map.
    Uses DuckDB when available; falls back to chunked Python scan.
    """
    try:
        import duckdb

        p = _duckdb_csv_path(path)
        con = duckdb.connect()
        try:
            return con.execute(f"""
                SELECT CAST(row_number AS INTEGER) AS row_number,
                       column_name,
                       source_table,
                       CAST(source_row AS INTEGER) AS source_row,
                       source_column
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY row_number, column_name
                               ORDER BY source_table, CAST(source_row AS BIGINT)
                           ) AS rn
                    FROM read_csv('{p}', header=true, all_varchar=true)
                ) t
                WHERE rn = 1
            """).fetchdf()
        finally:
            con.close()
    except Exception:
        pass

    best: dict[tuple[int, str], tuple[str, int, str]] = {}
    for chunk in pd.read_csv(
        path,
        encoding=ENCODING,
        dtype=str,
        keep_default_na=False,
        usecols=_CELL_MAP_COLS,
        chunksize=_CHUNK_SIZE,
    ):
        chunk["row_number"] = pd.to_numeric(chunk["row_number"], errors="coerce")
        chunk["source_row"] = pd.to_numeric(chunk["source_row"], errors="coerce")
        chunk = chunk.dropna(subset=["row_number", "source_row"])
        if chunk.empty:
            continue
        chunk["row_number"] = chunk["row_number"].astype(int)
        chunk["source_row"] = chunk["source_row"].astype(int)
        for row in chunk.itertuples(index=False):
            key = (row.row_number, row.column_name)
            cand = (row.source_table, row.source_row, row.source_column)
            prev = best.get(key)
            if prev is None or (cand[0], cand[1]) < (prev[0], prev[1]):
                best[key] = cand

    if not best:
        return pd.DataFrame(columns=_CELL_MAP_COLS)
    return pd.DataFrame(
        [
            {
                "row_number": k[0],
                "column_name": k[1],
                "source_table": v[0],
                "source_row": v[1],
                "source_column": v[2],
            }
            for k, v in best.items()
        ]
    )


def _iter_error_row_chunks(path: Path):
    """Yield subsumption rows that carry error annotations (all source rows preserved)."""
    p = _duckdb_csv_path(path)
    try:
        import duckdb

        con = duckdb.connect()
        try:
            cols = {
                row[0]
                for row in con.execute(
                    f"DESCRIBE SELECT * FROM read_csv('{p}', header=true, all_varchar=true) LIMIT 0"
                ).fetchall()
            }
            if "error_column" not in cols:
                return
            result = con.execute(f"""
                SELECT output_row_number, source_table, source_row,
                       error_column, error_type, corrected_value
                FROM read_csv('{p}', header=true, all_varchar=true)
                WHERE trim(COALESCE(error_column, '')) != ''
                  AND trim(COALESCE(error_type, '')) != ''
            """)
            out_cols = [d[0] for d in result.description]
            while True:
                batch = result.fetchmany(_CHUNK_SIZE)
                if not batch:
                    break
                yield pd.DataFrame(batch, columns=out_cols)
            return
        finally:
            con.close()
    except Exception:
        pass

    for chunk in pd.read_csv(
        path, encoding=ENCODING, dtype=str, keep_default_na=False, chunksize=_CHUNK_SIZE,
    ):
        if "error_column" not in chunk.columns:
            return
        mask = (
            chunk["error_column"].astype(str).str.strip().ne("")
            & chunk["error_type"].astype(str).str.strip().ne("")
        )
        if mask.any():
            yield chunk.loc[mask, [c for c in _SUB_ERROR_COLS if c in chunk.columns]]


def _prepare_error_chunk(
    chunk: pd.DataFrame,
    col_map: dict[str, str],
    n_rows: int,
) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["col"] = chunk["error_column"].astype(str).str.strip().str.lower().map(col_map)
    chunk = chunk.dropna(subset=["col"])
    chunk["output_row_number"] = pd.to_numeric(chunk["output_row_number"], errors="coerce")
    chunk["source_row"] = pd.to_numeric(chunk["source_row"], errors="coerce")
    chunk = chunk.dropna(subset=["output_row_number", "source_row"])
    if chunk.empty:
        return chunk
    chunk["output_row_number"] = chunk["output_row_number"].astype(int)
    chunk["source_row"] = chunk["source_row"].astype(int)
    chunk["corrected_value"] = chunk["corrected_value"].astype(str).str.strip()
    chunk["error_type"] = chunk["error_type"].astype(str).str.strip()
    chunk["error_column"] = chunk["error_column"].astype(str).str.strip()
    valid = (chunk["output_row_number"] >= 0) & (chunk["output_row_number"] < n_rows)
    return chunk.loc[valid]


def _apply_subsumption_corrections_chunked(
    sub_path: Path,
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    cluster: str,
    changes_path: Path,
) -> int:
    """
    Fill clean cells and stream one clean_changes_provenance row per subsumption
    error row (every source row kept for correction evaluation).
    """
    col_map = {c.lower(): c for c in dirty_df.columns}
    col_to_idx = {name: idx for idx, name in enumerate(dirty_df.columns)}
    out_cols = [
        "cell_id", "table_id", "column_id", "row_number", "column_name",
        "old_value", "new_value", "error_type",
        "source_table", "source_row", "source_column",
    ]
    if changes_path.exists():
        changes_path.unlink()

    n_total = 0
    wrote_header = False
    for chunk in _iter_error_row_chunks(sub_path):
        chunk = _prepare_error_chunk(chunk, col_map, len(clean_df))
        if chunk.empty:
            continue

        batch = []
        for row in chunk.itertuples(index=False):
            out_row = row.output_row_number
            col = row.col
            corrected = row.corrected_value
            clean_df.at[out_row, col] = corrected
            col_id = col_to_idx[col]
            batch.append({
                "cell_id": f"{cluster}.{col_id}.{out_row}",
                "table_id": cluster,
                "column_id": col_id,
                "row_number": out_row,
                "column_name": col,
                "old_value": _cell_str(dirty_df.at[out_row, col]),
                "new_value": corrected,
                "error_type": row.error_type,
                "source_table": str(row.source_table),
                "source_row": row.source_row,
                "source_column": row.error_column,
            })

        pd.DataFrame(batch, columns=out_cols).to_csv(
            changes_path,
            mode="a",
            header=not wrote_header,
            index=False,
            encoding=ENCODING,
        )
        wrote_header = True
        n_total += len(batch)

    return n_total


def _write_isolated_error_map(sub_path: Path, out_path: Path, cache: SourceCache) -> bool:
    """Aggregate isolated_error_map from every annotated subsumption row."""
    p = _duckdb_csv_path(sub_path)
    try:
        import duckdb

        con = duckdb.connect()
        try:
            cols = {
                row[0]
                for row in con.execute(
                    f"DESCRIBE SELECT * FROM read_csv('{p}', header=true, all_varchar=true) LIMIT 0"
                ).fetchall()
            }
            if "error_column" not in cols:
                return False
            agg = con.execute(f"""
                SELECT source_table,
                       CAST(source_row AS INTEGER) AS source_row,
                       error_column AS source_column,
                       max(error_type) AS error_type,
                       COUNT(DISTINCT CAST(output_row_number AS INTEGER)) AS occurrence_count,
                       string_agg(
                           DISTINCT CAST(output_row_number AS VARCHAR), ','
                           ORDER BY CAST(output_row_number AS VARCHAR)
                       ) AS merged_row_indices
                FROM read_csv('{p}', header=true, all_varchar=true)
                WHERE trim(COALESCE(error_column, '')) != ''
                  AND trim(COALESCE(error_type, '')) != ''
                GROUP BY source_table, source_row, error_column
                ORDER BY source_table, source_row, source_column
            """).fetchdf()
        finally:
            con.close()
    except Exception:
        return False

    if agg.empty:
        return False

    records = []
    for row in agg.itertuples(index=False):
        source_table = str(row.source_table)
        source_row = int(row.source_row)
        source_column = str(row.source_column)
        error_type = str(row.error_type).strip()
        if not error_type:
            error_type = cache.error_types(source_table).get(
                (source_row, source_column.lower()), "",
            )
        records.append({
            "source_table": _table_folder_name(source_table),
            "source_row": source_row,
            "source_column": source_column,
            "occurrence_count": int(row.occurrence_count),
            "merged_row_indices": str(row.merged_row_indices),
            "error_type": error_type,
        })

    pd.DataFrame(records).to_csv(out_path, index=False, encoding=ENCODING)
    return True


def _isolated_error_map_fallback(
    changes_path: Path,
    out_path: Path,
    cache: SourceCache,
) -> bool:
    """Build isolated_error_map from clean_changes when DuckDB aggregation is unavailable."""
    if not changes_path.exists():
        return False

    source_occurrences: dict[tuple[str, int, str], list[int]] = {}
    error_type_by_source: dict[tuple[str, int, str], str] = {}
    usecols = ["row_number", "source_table", "source_row", "source_column", "error_type"]
    for chunk in pd.read_csv(
        changes_path, encoding=ENCODING, dtype=str, keep_default_na=False,
        usecols=usecols, chunksize=_CHUNK_SIZE,
    ):
        for row in chunk.itertuples(index=False):
            key = (str(row.source_table), int(row.source_row), str(row.source_column))
            source_occurrences.setdefault(key, []).append(int(row.row_number))
            if str(row.error_type).strip():
                error_type_by_source[key] = str(row.error_type).strip()

    if not source_occurrences:
        return False

    records = []
    for (source_table, source_row, source_column), merged_rows in sorted(source_occurrences.items()):
        error_type = error_type_by_source.get(
            (source_table, source_row, source_column), "",
        )
        if not error_type:
            error_type = cache.error_types(source_table).get(
                (source_row, source_column.lower()), "",
            )
        records.append({
            "source_table": _table_folder_name(source_table),
            "source_row": source_row,
            "source_column": source_column,
            "occurrence_count": len(merged_rows),
            "merged_row_indices": ",".join(str(i) for i in sorted(set(merged_rows))),
            "error_type": error_type,
        })

    pd.DataFrame(records).to_csv(out_path, index=False, encoding=ENCODING)
    return True


def _build_clean_and_provenance(
    providers_df: pd.DataFrame,
    n_rows: int,
    columns: list[str],
    cache: SourceCache,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_df = pd.DataFrame("", index=range(n_rows), columns=columns)
    prov_df = pd.DataFrame("", index=range(n_rows), columns=columns)
    if providers_df.empty:
        return clean_df, prov_df

    providers_df = providers_df.copy()
    providers_df["source_row"] = providers_df["source_row"].astype(int)
    providers_df["row_number"] = providers_df["row_number"].astype(int)
    providers_df["source_column_key"] = providers_df["source_column"].astype(str).str.strip().str.lower()

    for src_table, grp in providers_df.groupby("source_table", sort=False):
        clean_src = cache.clean(src_table)
        if clean_src is None:
            continue
        folder = _table_folder_name(src_table)
        col_lower_to_idx = {str(c).lower(): i for i, c in enumerate(clean_src.columns)}

        clean_reset = clean_src.reset_index(names="source_row")
        clean_long = clean_reset.melt(id_vars="source_row", var_name="_col", value_name="clean_val")
        clean_long["source_row"] = clean_long["source_row"].astype(int)
        clean_long["source_column_key"] = clean_long["_col"].astype(str).str.strip().str.lower()

        merged = grp.merge(
            clean_long[["source_row", "source_column_key", "clean_val"]],
            on=["source_row", "source_column_key"],
            how="left",
        )
        merged["clean_val"] = merged["clean_val"].map(_cell_str)
        col_idx = merged["source_column_key"].map(col_lower_to_idx)
        merged["prov"] = (
            f"{folder} § " + col_idx.astype("Int64").astype(str) + " § " + merged["source_row"].astype(str)
        )
        merged.loc[col_idx.isna(), "prov"] = ""

        for col_name, sub in merged.groupby("column_name", sort=False):
            if col_name not in clean_df.columns:
                continue
            idx = sub["row_number"].to_numpy()
            clean_df.loc[idx, col_name] = sub["clean_val"].to_numpy()
            prov_df.loc[idx, col_name] = sub["prov"].to_numpy()

    return clean_df, prov_df


def export_cluster(
    cluster: str,
    fd_output: Path,
    out_root: Path,
    cache: SourceCache,
) -> None:
    started = time.perf_counter()
    dirty_path = fd_output / f"{cluster}.csv"
    cell_map_path = fd_output / f"{cluster}_merged_cell_source_map.csv"
    sub_path = fd_output / f"{cluster}_subsumption_map.csv"

    if not dirty_path.exists() or not cell_map_path.exists():
        print(f"  skip {cluster}: missing FD output or cell source map")
        return

    dirty_df = _load_source_table(dirty_path)
    providers_df = _providers_from_cell_map(cell_map_path)
    clean_df, prov_df = _build_clean_and_provenance(
        providers_df, len(dirty_df), list(dirty_df.columns), cache,
    )

    out_dir = out_root / cluster
    out_dir.mkdir(parents=True, exist_ok=True)

    changes_path = out_dir / "clean_changes_provenance.csv"
    n_changes = 0
    if sub_path.exists():
        n_changes = _apply_subsumption_corrections_chunked(
            sub_path, clean_df, dirty_df, cluster, changes_path,
        )
    elif changes_path.exists():
        changes_path.unlink()

    dirty_df.to_csv(out_dir / "dirty.csv", index=False, encoding=ENCODING)
    clean_df.to_csv(out_dir / "clean.csv", index=False, encoding=ENCODING)
    shutil.copy2(cell_map_path, out_dir / "merged_cell_source_map.csv")
    if sub_path.exists():
        shutil.copy2(sub_path, out_dir / "subsumption_map.csv")
    prov_df.to_csv(out_dir / "provenance.csv", index=False, encoding=ENCODING)

    isolated_path = out_dir / "isolated_error_map.csv"
    has_isolated = False
    if sub_path.exists():
        has_isolated = _write_isolated_error_map(sub_path, isolated_path, cache)
    if not has_isolated and n_changes > 0:
        has_isolated = _isolated_error_map_fallback(changes_path, isolated_path, cache)
    if not has_isolated and isolated_path.exists():
        isolated_path.unlink()

    elapsed = time.perf_counter() - started
    print(
        f"  {cluster}: {len(dirty_df)} rows × {len(dirty_df.columns)} cols, "
        f"{len(providers_df)} providers → {out_dir} "
        f"({n_changes} source-error provenance rows, {elapsed:.1f}s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ALITE FD clusters as per-table dirty/clean dirs")
    parser.add_argument("--corpus", choices=CORPORA, default="open_data_uk")
    parser.add_argument("--fd-output", type=Path, default=None, help="Override fd_output directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override output root")
    parser.add_argument("--cluster", default=None, help="Export only this cluster name")
    args = parser.parse_args()

    paths = get_paths(args.corpus)
    fd_output = (args.fd_output or paths["fd_output"]).resolve()
    out_root = (args.out_dir or paths["fd_output"].parent / "merged_tables").resolve()
    iso = isolated_dir(args.corpus)

    if not fd_output.is_dir():
        raise SystemExit(f"fd_output not found: {fd_output}")

    clusters = [args.cluster] if args.cluster else _cluster_names(fd_output)
    if not clusters:
        raise SystemExit(f"No cluster CSVs found in {fd_output}")

    print(f"Corpus:    {args.corpus}")
    print(f"Input:     {fd_output}")
    print(f"Isolated:  {iso}")
    print(f"Output:    {out_root}")
    print(f"Clusters:  {len(clusters)}\n")

    cache = SourceCache(iso)
    for cluster in clusters:
        export_cluster(cluster, fd_output, out_root, cache)

    print(f"\nDone. Wrote {len(clusters)} cluster director{'y' if len(clusters) == 1 else 'ies'} under {out_root}")


if __name__ == "__main__":
    main()
