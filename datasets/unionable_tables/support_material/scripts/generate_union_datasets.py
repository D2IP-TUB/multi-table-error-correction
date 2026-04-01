#!/usr/bin/env python3
"""
Generate unionable table datasets with controlled overlap levels.

Recommended two-step flow:
  1. Run create_partitioned_base.py to build a horizontally partitioned
     base from Unified_Union_Exp_isolated (one partition set per table).
  2. Run this script with USE_PARTITIONED_BASE=True so disjoint, maximal,
     and partial-overlap datasets are all derived from the same partitions.
     Partial overlap then uses partition assignment (shared / unique_a / unique_b)
     so you get reproducible "multiple versions from the same thing".

Each subdirectory under the output root is a **data lake** — a flat
collection of tables.  Tables within a lake share the same schema.

Output structure:

  generated_union_datasets/
    isolated_all_partitions/             – all source tables, one per variant
      DGov_{variant}_{table}/
        dirty.csv
        clean.csv

    disjoint_with_duplicates/            – UNION ALL of disjoint table pairs
      {pair}/dirty.csv, clean.csv, lineage.csv

    disjoint_without_duplicates/         – UNION (deduped) of disjoint pairs
      {pair}/dirty.csv, clean.csv, lineage.csv

    maximal_overlap_with_duplicates/     – UNION ALL of FD/Typo/NO variants
      {table}/dirty.csv, clean.csv, lineage.csv

    maximal_overlap_without_duplicates/
      {table}/dirty.csv, clean.csv, lineage.csv

    partial_overlap_{pct}_with_duplicates/
      {table}/dirty.csv, clean.csv, lineage.csv

    partial_overlap_{pct}_without_duplicates/
      {table}/dirty.csv, clean.csv, lineage.csv

    metadata.json

Error-type variants:
  FD   – functional dependency violation errors
  Typo – typographical errors
  NO   – numeric outlier errors

Overlapping rows always come from *different* error-type variants (e.g.
FD dirty vs Typo dirty), so shared entities have different surface values —
just like real-world unionable tables.

All CSV data is read and written using Python's csv module so that every
value is preserved as its original string — no type conversion, no silent
NaN insertion, no float promotion.  Duplicate detection for UNION is
exact string comparison across all columns.

Deduplication is always performed on dirty data first; the clean side
keeps exactly the same rows (aligned by index), so dirty.csv and
clean.csv always have identical row counts everywhere.

lineage.csv columns (per result table):
  row_idx        – 0-based row index in the output file
  source_table   – original table name
  source_variant – FD / Typo / NO
  source_row_idx – 0-based row index in the source variant's CSV
  partition      – "all", "shared", "unique_a", "unique_b"
"""

import csv
import json
import os
import random
import shutil
from collections import defaultdict
from itertools import combinations

# Reproducible variant assignment
random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# When True, read from partitioned_union_base (run create_partitioned_base.py first).
# Partial overlap is then done by partition assignment (shared / unique_a / unique_b).
USE_PARTITIONED_BASE = True
PARTITIONED_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "partitioned_union_base")
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Unified_Union_Exp_isolated")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "generated_union_datasets")
PREFIXES = ["FD", "Typo", "NO"]
OVERLAP_RATIOS = [0.25, 0.50, 0.75]

LINEAGE_HEADERS = ["row_idx", "source_table", "source_variant",
                   "source_row_idx", "partition"]


# ---------------------------------------------------------------------------
# CSV I/O  (all values stay as raw strings — no type conversion)
# ---------------------------------------------------------------------------
def read_csv_file(path):
    """Read a CSV file, returning (headers, rows) with all values as strings."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
    return headers, rows


def write_csv_file(path, headers, rows):
    """Write headers + rows to a CSV file."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def discover_tables(base_dir):
    """Return {table_name: {prefix: {"clean": path, "dirty": path}}}."""
    tables = defaultdict(dict)
    for dirname in sorted(os.listdir(base_dir)):
        dirpath = os.path.join(base_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
        parts = dirname.split("_", 2)
        if len(parts) < 3 or parts[0] != "DGov":
            continue
        prefix, table_name = parts[1], parts[2]
        clean_csv = os.path.join(dirpath, "clean.csv")
        dirty_csv = os.path.join(dirpath, "dirty.csv")
        if os.path.exists(clean_csv) and os.path.exists(dirty_csv):
            tables[table_name][prefix] = {
                "clean": clean_csv,
                "dirty": dirty_csv,
            }
    return dict(tables)


def discover_tables_partitioned(base_dir):
    """Discover from partitioned_union_base: {table: {variant: {partitions: [(dirty, clean), ...], num_partitions}}}."""
    tables = defaultdict(dict)
    with open(os.path.join(base_dir, "manifest.json")) as f:
        global_manifest = json.load(f)
    for table_name, info in global_manifest.get("tables", {}).items():
        table_dir = os.path.join(base_dir, table_name.replace(os.sep, "_"))
        if not os.path.isdir(table_dir):
            continue
        table_manifest_path = os.path.join(table_dir, "manifest.json")
        if os.path.exists(table_manifest_path):
            with open(table_manifest_path) as f:
                table_manifest = json.load(f)
        else:
            table_manifest = {"variants": info["variants"], "num_partitions": info["num_partitions"]}
        num_partitions = table_manifest["num_partitions"]
        for variant in table_manifest["variants"]:
            variant_dir = os.path.join(table_dir, variant)
            if not os.path.isdir(variant_dir):
                continue
            partitions = []
            for p in range(num_partitions):
                part_dir = os.path.join(variant_dir, f"partition_{p:03d}")
                dirty_path = os.path.join(part_dir, "dirty.csv")
                clean_path = os.path.join(part_dir, "clean.csv")
                if os.path.exists(dirty_path) and os.path.exists(clean_path):
                    partitions.append((dirty_path, clean_path))
            if len(partitions) == num_partitions:
                tables[table_name][variant] = {
                    "partitions": partitions,
                    "num_partitions": num_partitions,
                }
    return dict(tables)


def read_clean(tables, table_name, prefix):
    """Read the clean side for a given table and variant (single file or concatenated partitions)."""
    entry = tables[table_name][prefix]
    if "partitions" in entry:
        headers, all_rows = None, []
        for _dirty_path, clean_path in entry["partitions"]:
            h, rows = read_csv_file(clean_path)
            if headers is None:
                headers = h
            all_rows.extend(rows)
        return (headers, all_rows)
    return read_csv_file(entry["clean"])


def read_dirty(tables, table_name, prefix):
    """Read the dirty side for a given table and variant (single file or concatenated partitions)."""
    entry = tables[table_name][prefix]
    if "partitions" in entry:
        headers, all_rows = None, []
        for dirty_path, _clean_path in entry["partitions"]:
            h, rows = read_csv_file(dirty_path)
            if headers is None:
                headers = h
            all_rows.extend(rows)
        return (headers, all_rows)
    return read_csv_file(entry["dirty"])


def is_partitioned_base(tables):
    """True if tables were loaded from partitioned base (entries have 'partitions' key)."""
    for variants in tables.values():
        for entry in variants.values():
            return "partitions" in entry
    return False


def get_partition_slice(tables, table_name, variant, partition_ids):
    """Read and concatenate only the given partition IDs for one table variant. Returns (dirty_headers, dirty_rows, clean_headers, clean_rows)."""
    entry = tables[table_name][variant]
    if "partitions" not in entry:
        raise ValueError("get_partition_slice requires partitioned base")
    dirty_headers, dirty_rows = None, []
    clean_headers, clean_rows = None, []
    for p in partition_ids:
        dirty_path, clean_path = entry["partitions"][p]
        dh, dr = read_csv_file(dirty_path)
        ch, cr = read_csv_file(clean_path)
        if dirty_headers is None:
            dirty_headers, clean_headers = dh, ch
        dirty_rows.extend(dr)
        clean_rows.extend(cr)
    return dirty_headers, dirty_rows, clean_headers, clean_rows


def group_by_schema(tables):
    """Group tables whose clean-csv columns are identical (ignoring order).

    Returns {frozenset(col_names): [table_name, …]}
    """
    schema_groups = defaultdict(list)
    for table_name, variants in tables.items():
        entry = next(iter(variants.values()))
        if "partitions" in entry:
            headers, _ = read_csv_file(entry["partitions"][0][1])
        else:
            with open(entry["clean"], newline='', encoding='utf-8-sig') as f:
                headers = next(csv.reader(f))
        cols = frozenset(headers)
        schema_groups[cols].append(table_name)
    return dict(schema_groups)


def reorder_columns(headers, rows, target_cols):
    """Reorder columns in rows to match target_cols order."""
    col_map = {col: i for i, col in enumerate(headers)}
    order = [col_map[col] for col in target_cols]
    return [[row[i] for i in order] for row in rows]


def make_lineage(source_table, source_variant, source_row_indices, partition):
    """Build lineage rows (without row_idx prefix) for one slice.

    Returns list of [source_table, source_variant, source_row_idx, partition].
    The row_idx column is added later when the final output is assembled.
    """
    return [[source_table, source_variant, str(idx), partition]
            for idx in source_row_indices]


def _dedup_rows(rows):
    """Drop exact-duplicate rows (keep first). Returns (kept_rows, keep_indices)."""
    seen = set()
    keep_indices = []
    for i, row in enumerate(rows):
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            keep_indices.append(i)
    return [rows[i] for i in keep_indices], keep_indices


def safe_name(s, maxlen=100):
    """Shorten a table name for use as a directory component."""
    return s[:maxlen]


def save_result_table(lake_dir, name, dirty_headers, dirty_rows,
                      clean_headers, clean_rows, lineage_rows):
    """Save one result table (dirty.csv + clean.csv + lineage.csv) into a lake."""
    table_dir = os.path.join(lake_dir, name)
    os.makedirs(table_dir, exist_ok=True)
    write_csv_file(os.path.join(table_dir, "dirty.csv"),
                   dirty_headers, dirty_rows)
    write_csv_file(os.path.join(table_dir, "clean.csv"),
                   clean_headers, clean_rows)
    write_csv_file(os.path.join(table_dir, "lineage.csv"),
                   LINEAGE_HEADERS, lineage_rows)


def save_isolated_partition(lake_dir, table_dir_name, subdir, dirty_headers,
                            dirty_rows, clean_headers, clean_rows, lineage_rows):
    """Save a single partition as isolated (no union) under table_dir/subdir.
    lineage_rows must include row_idx as first column (same as save_result_table).
    """
    table_dir = os.path.join(lake_dir, table_dir_name, subdir)
    os.makedirs(table_dir, exist_ok=True)
    write_csv_file(os.path.join(table_dir, "dirty.csv"), dirty_headers, dirty_rows)
    write_csv_file(os.path.join(table_dir, "clean.csv"), clean_headers, clean_rows)
    write_csv_file(os.path.join(table_dir, "lineage.csv"), LINEAGE_HEADERS, lineage_rows)


def compute_and_save_unions(with_dup_lake, without_dup_lake, name,
                            dirty_tables, clean_tables, lineage_parts):
    """Compute UNION ALL and UNION, saving each to its own lake directory.

    Parameters
    ----------
    with_dup_lake : str
        Lake directory for UNION ALL results (with duplicates).
    without_dup_lake : str
        Lake directory for UNION results (deduplicated).
    name : str
        Result table name within the lake.
    dirty_tables : list of (headers, rows)
    clean_tables : list of (headers, rows)
    lineage_parts : list of list-of-lists
        One lineage list per input table (rows WITHOUT row_idx prefix).
    """
    dirty_headers = dirty_tables[0][0]
    clean_headers = clean_tables[0][0]

    # --- UNION ALL: concatenate all input tables ---
    all_dirty, all_clean, all_lineage = [], [], []
    for (dh, dr), (ch, cr), lin in zip(dirty_tables, clean_tables,
                                        lineage_parts):
        all_dirty.extend(dr)
        all_clean.extend(cr)
        all_lineage.extend(lin)

    ua_lineage = [[str(i)] + row for i, row in enumerate(all_lineage)]
    save_result_table(with_dup_lake, name,
                      dirty_headers, all_dirty,
                      clean_headers, all_clean,
                      ua_lineage)

    # --- UNION: dedup on dirty, align clean + lineage ---
    dedup_dirty, keep_idx = _dedup_rows(all_dirty)
    dedup_clean = [all_clean[i] for i in keep_idx]
    u_lineage = [[str(i)] + all_lineage[orig]
                 for i, orig in enumerate(keep_idx)]
    save_result_table(without_dup_lake, name,
                      dirty_headers, dedup_dirty,
                      clean_headers, dedup_clean,
                      u_lineage)

    return {
        "union_all_rows": len(all_dirty),
        "union_rows": len(dedup_dirty),
        "duplicates_removed": len(all_dirty) - len(dedup_dirty),
    }


# ---------------------------------------------------------------------------
# 0. Isolated partitions
# ---------------------------------------------------------------------------
def generate_isolated(base_dir, out_base):
    """Copy all source tables into the isolated-all-partitions lake (raw layout)."""
    count = 0
    for dirname in sorted(os.listdir(base_dir)):
        src_dir = os.path.join(base_dir, dirname)
        if not os.path.isdir(src_dir):
            continue
        parts = dirname.split("_", 2)
        if len(parts) < 3 or parts[0] != "DGov":
            continue
        dst_dir = os.path.join(out_base, dirname)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in ["dirty.csv", "clean.csv"]:
            src = os.path.join(src_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_dir, fname))
        count += 1
    return count


def generate_isolated_from_tables(tables, out_base):
    """Write each (table, variant) as isolated_all_partitions/DGov_{variant}_{table}/ from in-memory tables (e.g. partitioned base)."""
    count = 0
    for table_name, variants in sorted(tables.items()):
        for variant in sorted(variants.keys()):
            dh, dr = read_dirty(tables, table_name, variant)
            ch, cr = read_clean(tables, table_name, variant)
            dirname = f"DGov_{variant}_{table_name}"
            dst_dir = os.path.join(out_base, safe_name(dirname))
            os.makedirs(dst_dir, exist_ok=True)
            write_csv_file(os.path.join(dst_dir, "dirty.csv"), dh, dr)
            write_csv_file(os.path.join(dst_dir, "clean.csv"), ch, cr)
            count += 1
    return count


# ---------------------------------------------------------------------------
# 1. Disjoint union (0 % entity overlap)
# ---------------------------------------------------------------------------
def generate_disjoint(tables, schema_groups, with_dup_dir, without_dup_dir):
    """Union all tables in the same schema group into one table per variant.

    Tables in a schema group share the same columns but have disjoint entities.
    We produce one union (table1 ∪ table2 ∪ ... ∪ tableN) per variant (FD, Typo, NO)
    when all tables in the group have that variant. Because entities are disjoint,
    UNION = UNION ALL (no dupes).
    """
    metadata = []
    for _cols, group_tables in sorted(schema_groups.items(),
                                       key=lambda kv: sorted(kv[1])):
        if len(group_tables) < 2:
            continue
        sorted_tables = sorted(group_tables)
        # Common variants across all tables in the group
        common_variants = set(tables[sorted_tables[0]].keys())
        for t in sorted_tables[1:]:
            common_variants &= set(tables[t].keys())
        if not common_variants:
            continue
        for variant in sorted(common_variants):
            dirty_tables_list = []
            clean_tables_list = []
            lineage_parts = []
            dirty_cols = clean_cols = None
            for t in sorted_tables:
                dh, dr = read_dirty(tables, t, variant)
                ch, cr = read_clean(tables, t, variant)
                if dirty_cols is None:
                    dirty_cols = list(dh)
                    clean_cols = list(ch)
                else:
                    dr = reorder_columns(dh, dr, dirty_cols)
                    cr = reorder_columns(ch, cr, clean_cols)
                dirty_tables_list.append((dirty_cols, dr))
                clean_tables_list.append((clean_cols, cr))
                lineage_parts.append(make_lineage(t, variant, list(range(len(dr))), "all"))
            # One name for the whole group, e.g. "TableA__TableB__TableC__FD"
            group_name = "__".join(safe_name(t) for t in sorted_tables)
            union_name = f"{group_name}__{variant}"
            info = compute_and_save_unions(
                with_dup_dir, without_dup_dir, union_name,
                dirty_tables=dirty_tables_list,
                clean_tables=clean_tables_list,
                lineage_parts=lineage_parts)
            info.update({
                "name": union_name,
                "category": "disjoint",
                "tables": sorted_tables,
                "variant": variant,
                "overlap_ratio": 0.0,
            })
            metadata.append(info)
    return metadata


# ---------------------------------------------------------------------------
# 2. Maximal overlap (~100 % entity overlap)
# ---------------------------------------------------------------------------
def generate_maximal_overlap(tables, with_dup_dir, without_dup_dir):
    """100 % entity overlap: same table from all three variants (dirty).

    Each input table is the dirty version from a different error-type
    variant (FD, Typo, NO).  Because surface values differ across
    variants, UNION barely removes anything while the entities are
    identical.
    """
    metadata = []
    for table_name, variants in sorted(tables.items()):
        if not all(p in variants for p in PREFIXES):
            continue

        tname = safe_name(table_name)

        # Read dirty + clean for each variant, reorder to FD column order
        fd_dh, fd_dr = read_dirty(tables, table_name, "FD")
        dirty_cols = list(fd_dh)
        fd_ch, fd_cr = read_clean(tables, table_name, "FD")
        clean_cols = list(fd_ch)

        typo_dh, typo_dr = read_dirty(tables, table_name, "Typo")
        typo_dr = reorder_columns(typo_dh, typo_dr, dirty_cols)
        typo_ch, typo_cr = read_clean(tables, table_name, "Typo")
        typo_cr = reorder_columns(typo_ch, typo_cr, clean_cols)

        no_dh, no_dr = read_dirty(tables, table_name, "NO")
        no_dr = reorder_columns(no_dh, no_dr, dirty_cols)
        no_ch, no_cr = read_clean(tables, table_name, "NO")
        no_cr = reorder_columns(no_ch, no_cr, clean_cols)

        n = len(fd_dr)
        src_rows = list(range(n))

        lins = [
            make_lineage(table_name, "FD", src_rows, "all"),
            make_lineage(table_name, "Typo", src_rows, "all"),
            make_lineage(table_name, "NO", src_rows, "all"),
        ]

        info = compute_and_save_unions(
            with_dup_dir, without_dup_dir, tname,
            dirty_tables=[(dirty_cols, fd_dr),
                          (dirty_cols, typo_dr),
                          (dirty_cols, no_dr)],
            clean_tables=[(clean_cols, fd_cr),
                          (clean_cols, typo_cr),
                          (clean_cols, no_cr)],
            lineage_parts=lins)

        info.update({
            "name": tname,
            "category": "maximal_overlap",
            "table": table_name,
            "variants": PREFIXES,
            "overlap_ratio": 1.0,
        })
        metadata.append(info)

    return metadata


# ---------------------------------------------------------------------------
# 3. Partial overlap  (25 %, 50 %, 75 %)
# ---------------------------------------------------------------------------
def _partition_indices(n_rows, overlap_ratio):
    """Compute disjoint index ranges for two output tables (row-based slicing).

    Given N rows and overlap ratio r:
        S = floor(N / (2 - r))       # rows per output table
        shared   = round(r * S)
        unique   = S - shared         # unique rows per table

    Returns (shared_idx, unique_a_idx, unique_b_idx, S, shared, unique).
    """
    r = overlap_ratio
    S = int(n_rows / (2 - r))
    shared = round(r * S)
    unique = S - shared

    shared_idx = list(range(0, shared))
    unique_a_idx = list(range(shared, shared + unique))
    unique_b_idx = list(range(shared + unique, shared + 2 * unique))
    return shared_idx, unique_a_idx, unique_b_idx, S, shared, unique


def _partition_ids_for_overlap(num_partitions, overlap_ratio, table_name=""):
    """Assign partition IDs to shared / unique_a / unique_b for partial overlap.

    Overlap is defined as row ratio: shared_rows / (shared_rows + unique_rows)
    per table. With equal-sized partitions, that equals n_shared/(n_shared + n_unique).
    So we need n_shared such that n_shared/(n_shared + n_unique) ≈ overlap_ratio
    with n_shared + n_unique_a + n_unique_b = k and n_unique_a ≈ n_unique_b.
    Solving: n_unique = n_shared*(1-r)/r per side, so n_shared + 2*n_shared*(1-r)/r = k
    => n_shared = k*r/(2-r).

    Partition IDs are shuffled with a seed derived from (table_name, overlap_ratio)
    so that (a) the shared set is not biased toward low-numbered partitions and
    (b) each overlap level gets an independent assignment.

    Returns (shared_ids, unique_a_ids, unique_b_ids).
    """
    k = num_partitions
    r = overlap_ratio
    # Row-overlap target: shared/(shared+unique) = r  =>  n_shared = k*r/(2-r)
    n_shared = max(1, min(k - 2, round(k * r / (2 - r))))
    remaining = k - n_shared
    n_unique_a = remaining // 2
    n_unique_b = remaining - n_unique_a
    # Shuffle partition IDs so assignment is not positionally biased and each
    # overlap level is independent.
    rng = random.Random(hash((table_name, overlap_ratio)))
    ids = list(range(k))
    rng.shuffle(ids)
    shared_ids = ids[:n_shared]
    unique_a_ids = ids[n_shared:n_shared + n_unique_a]
    unique_b_ids = ids[n_shared + n_unique_a:]
    return shared_ids, unique_a_ids, unique_b_ids


def generate_partial_overlap(tables, overlap_ratio, with_dup_dir,
                             without_dup_dir, variant_assignments,
                             partition_manifest=None):
    """Partial entity overlap: shared rows from both variants, unique rows each.

    When using partitioned base, assigns partition IDs to shared/unique_a/unique_b
    so the same base yields reproducible partial-overlap datasets. Otherwise
    uses row-index slicing (legacy).
    partition_manifest: optional {table_name: {"num_partitions", "partition_row_counts"}} from partitioned base.
    """
    pct = int(overlap_ratio * 100)
    metadata = []
    use_partitions = is_partitioned_base(tables) and partition_manifest is not None

    for table_name, variants in sorted(tables.items()):
        if table_name not in variant_assignments:
            continue

        var_a, var_b = variant_assignments[table_name]

        if use_partitions and table_name in partition_manifest:
            # Partition-based: assign partitions to shared / unique_a / unique_b
            info_man = partition_manifest[table_name]
            num_partitions = info_man["num_partitions"]
            partition_row_counts = info_man.get("partition_row_counts") or []
            if num_partitions < 3:
                continue
            shared_ids, unique_a_ids, unique_b_ids = _partition_ids_for_overlap(num_partitions, overlap_ratio, table_name)
            if not shared_ids or not unique_a_ids or not unique_b_ids:
                continue

            dh_a, dr_a, ch_a, cr_a = get_partition_slice(tables, table_name, var_a, shared_ids + unique_a_ids)
            dh_b, dr_b, ch_b, cr_b = get_partition_slice(tables, table_name, var_b, shared_ids + unique_b_ids)
            dirty_cols = list(dh_a)
            clean_cols = list(ch_a)
            dr_b = reorder_columns(dh_b, dr_b, dirty_cols)
            cr_b = reorder_columns(ch_b, cr_b, clean_cols)

            # Lineage: we don't have per-partition row counts in get_partition_slice; rebuild from partition_row_counts
            def lineage_for_partitions(variant, part_ids, label):
                out = []
                idx = 0
                for p in part_ids:
                    n = partition_row_counts[p] if p < len(partition_row_counts) else 0
                    for _ in range(n):
                        out.append([table_name, variant, str(idx), label])
                        idx += 1
                return out
            lin_a = (lineage_for_partitions(var_a, shared_ids, "shared") +
                     lineage_for_partitions(var_a, unique_a_ids, "unique_a"))
            lin_b = (lineage_for_partitions(var_b, shared_ids, "shared") +
                     lineage_for_partitions(var_b, unique_b_ids, "unique_b"))

            def lineage_global(tname, variant, part_ids_list, labels_list, partition_row_counts):
                """Build lineage with source_row_idx = global row index in the source variant."""
                rows = []
                for part_ids, label in zip(part_ids_list, labels_list):
                    if not part_ids:
                        continue
                    start = sum(partition_row_counts[i] for i in range(part_ids[0]) if i < len(partition_row_counts))
                    for p in part_ids:
                        n = partition_row_counts[p] if p < len(partition_row_counts) else 0
                        for i in range(n):
                            rows.append([tname, variant, str(start + i), label])
                        start += n
                return rows
            lin_a = lineage_global(table_name, var_a, [shared_ids, unique_a_ids], ["shared", "unique_a"], partition_row_counts)
            lin_b = lineage_global(table_name, var_b, [shared_ids, unique_b_ids], ["shared", "unique_b"], partition_row_counts)

            n_shared = sum(partition_row_counts[p] for p in shared_ids if p < len(partition_row_counts))
            n_ua = sum(partition_row_counts[p] for p in unique_a_ids if p < len(partition_row_counts))
            n_ub = sum(partition_row_counts[p] for p in unique_b_ids if p < len(partition_row_counts))
            S = n_shared + n_ua
        else:
            # Row-index slicing (raw or fallback)
            hdrs_da, rows_da = read_dirty(tables, table_name, var_a)
            hdrs_ca, rows_ca = read_clean(tables, table_name, var_a)
            hdrs_db, rows_db = read_dirty(tables, table_name, var_b)
            hdrs_cb, rows_cb = read_clean(tables, table_name, var_b)
            dirty_cols = list(hdrs_da)
            clean_cols = list(hdrs_ca)
            rows_db = reorder_columns(hdrs_db, rows_db, dirty_cols)
            rows_cb = reorder_columns(hdrs_cb, rows_cb, clean_cols)
            N = len(rows_da)
            if N < 6:
                continue
            shared_idx, ua_idx, ub_idx, S, n_shared, n_unique = _partition_indices(N, overlap_ratio)
            if n_shared == 0 or n_unique == 0:
                continue
            ta_dirty = [rows_da[i] for i in shared_idx] + [rows_da[i] for i in ua_idx]
            ta_clean = [rows_ca[i] for i in shared_idx] + [rows_ca[i] for i in ua_idx]
            tb_dirty = [rows_db[i] for i in shared_idx] + [rows_db[i] for i in ub_idx]
            tb_clean = [rows_cb[i] for i in shared_idx] + [rows_cb[i] for i in ub_idx]
            lin_a = make_lineage(table_name, var_a, shared_idx, "shared") + make_lineage(table_name, var_a, ua_idx, "unique_a")
            lin_b = make_lineage(table_name, var_b, shared_idx, "shared") + make_lineage(table_name, var_b, ub_idx, "unique_b")
            dr_a, dr_b, cr_a, cr_b = ta_dirty, tb_dirty, ta_clean, tb_clean
            n_shared = len(shared_idx)
            n_ua = len(ua_idx)
            n_ub = len(ub_idx)
            n_unique = n_ua

        tname = safe_name(table_name)
        info = compute_and_save_unions(
            with_dup_dir, without_dup_dir, tname,
            dirty_tables=[(dirty_cols, dr_a), (dirty_cols, dr_b)],
            clean_tables=[(clean_cols, cr_a), (clean_cols, cr_b)],
            lineage_parts=[lin_a, lin_b])

        # Partitions not in the union (unique_b from var_a, unique_a from var_b)
        # are saved as isolated — one table per partition — so no data is missed.
        if use_partitions and table_name in partition_manifest:
            for p in unique_b_ids:
                p_dh, p_dr, p_ch, p_cr = get_partition_slice(tables, table_name, var_a, [p])
                p_lin = lineage_global(table_name, var_a, [[p]], ["isolated_unique_b"], partition_row_counts)
                p_lineage = [[str(i)] + row for i, row in enumerate(p_lin)]
                for lake_dir in (with_dup_dir, without_dup_dir):
                    save_isolated_partition(lake_dir, tname, f"isolated_unique_b_from_var_a_p{p}",
                                            p_dh, p_dr, p_ch, p_cr, p_lineage)
            for p in unique_a_ids:
                p_dh, p_dr, p_ch, p_cr = get_partition_slice(tables, table_name, var_b, [p])
                p_lin = lineage_global(table_name, var_b, [[p]], ["isolated_unique_a"], partition_row_counts)
                p_lineage = [[str(i)] + row for i, row in enumerate(p_lin)]
                for lake_dir in (with_dup_dir, without_dup_dir):
                    save_isolated_partition(lake_dir, tname, f"isolated_unique_a_from_var_b_p{p}",
                                            p_dh, p_dr, p_ch, p_cr, p_lineage)

        info.update({
            "name": tname,
            "category": f"partial_overlap_{pct}",
            "table": table_name,
            "variant_a": var_a,
            "variant_b": var_b,
            "overlap_ratio": overlap_ratio,
        })
        if use_partitions and table_name in partition_manifest:
            info.update({
                "source_rows": partition_manifest[table_name].get("total_rows"),
                "output_table_size": S,
                "shared_rows": n_shared,
                "unique_a_rows": n_ua,
                "unique_b_rows": n_ub,
                "partition_based": True,
            })
        else:
            info.update({
                "source_rows": N,
                "output_table_size": S,
                "shared_rows": n_shared,
                "unique_rows_per_table": n_ua,
            })
        metadata.append(info)

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    source_dir = PARTITIONED_BASE_DIR if USE_PARTITIONED_BASE else BASE_DIR
    print(f"Source directory : {source_dir}")
    if USE_PARTITIONED_BASE:
        print("  (partitioned base — partial overlap uses partition assignment)")
    print(f"Output directory : {OUTPUT_DIR}\n")

    # Clean previous output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    if USE_PARTITIONED_BASE:
        tables = discover_tables_partitioned(PARTITIONED_BASE_DIR)
        partition_manifest = {}
        for table_name in tables:
            manifest_path = os.path.join(
                PARTITIONED_BASE_DIR,
                table_name.replace(os.sep, "_"),
                "manifest.json",
            )
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    partition_manifest[table_name] = json.load(f)
    else:
        tables = discover_tables(BASE_DIR)
        partition_manifest = None

    print(f"Discovered {len(tables)} unique tables "
          f"({sum(len(v) for v in tables.values())} variants total)\n")

    schema_groups = group_by_schema(tables)
    print(f"Schema groups ({len(schema_groups)}):")
    for cols, group in sorted(schema_groups.items(), key=lambda kv: sorted(kv[1])):
        print(f"  {len(cols):2d} cols → {sorted(group)}")
    print()

    all_metadata = {}

    # Pre-compute variant pair assignments for partial overlap
    variant_assignments = {}
    for table_name, variants in sorted(tables.items()):
        available = [p for p in PREFIXES if p in variants]
        if len(available) >= 2:
            variant_assignments[table_name] = tuple(random.sample(available, 2))
    print(f"Variant assignments (partial overlap):")
    for tn, (va, vb) in sorted(variant_assignments.items()):
        print(f"  {tn}: {va} vs {vb}")
    print()

    # 0. Isolated partitions
    print("═══ Isolated All Partitions ═══")
    iso_dir = os.path.join(OUTPUT_DIR, "isolated_all_partitions")
    if USE_PARTITIONED_BASE:
        count = generate_isolated_from_tables(tables, iso_dir)
    else:
        count = generate_isolated(BASE_DIR, iso_dir)
    print(f"  → {count} partition(s)\n")

    # 1. Disjoint
    print("═══ Disjoint Union (0 % overlap) ═══")
    all_metadata["disjoint"] = generate_disjoint(
        tables, schema_groups,
        os.path.join(OUTPUT_DIR, "disjoint_with_duplicates"),
        os.path.join(OUTPUT_DIR, "disjoint_without_duplicates"))
    print(f"  → {len(all_metadata['disjoint'])} dataset(s)\n")

    # 2. Maximal overlap
    print("═══ Maximal Overlap (~100 % overlap) ═══")
    all_metadata["maximal_overlap"] = generate_maximal_overlap(
        tables,
        os.path.join(OUTPUT_DIR, "maximal_overlap_with_duplicates"),
        os.path.join(OUTPUT_DIR, "maximal_overlap_without_duplicates"))
    print(f"  → {len(all_metadata['maximal_overlap'])} dataset(s)\n")

    # 3. Partial overlap at each level (fixed variant assignments)
    for r in OVERLAP_RATIOS:
        pct = int(r * 100)
        print(f"═══ Partial Overlap ({pct} %) ═══")
        all_metadata[f"partial_overlap_{pct}"] = generate_partial_overlap(
            tables, r,
            os.path.join(OUTPUT_DIR, f"partial_overlap_{pct}_with_duplicates"),
            os.path.join(OUTPUT_DIR,
                         f"partial_overlap_{pct}_without_duplicates"),
            variant_assignments,
            partition_manifest=partition_manifest)
        print(f"  → {len(all_metadata[f'partial_overlap_{pct}'])} dataset(s)\n")

    # Persist top-level metadata
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2, default=str)
    print(f"Metadata written to {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
