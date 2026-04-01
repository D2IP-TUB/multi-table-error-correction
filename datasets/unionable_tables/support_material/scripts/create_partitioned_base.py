#!/usr/bin/env python3
"""
Build a horizontally partitioned base dataset from Unified_Union_Exp_isolated.

Each (table, variant) is split into K contiguous row partitions. The same
partition index refers to the same entities across variants (FD, Typo, NO),
so we can later build:
  - Disjoint unions: different tables (same schema)
  - Maximal overlap: same table, union all variants (all partitions)
  - Partial overlap: same table, assign partitions to shared / unique_a / unique_b

Output layout:

  partitioned_union_base/
    {table_name}/
      manifest.json       # variants, num_partitions, partition_row_counts
      {variant}/          # FD, Typo, NO
        partition_000/
          dirty.csv
          clean.csv
        partition_001/
          ...
    manifest.json         # global: tables, default_num_partitions

All CSV I/O preserves values as strings (no type conversion).
"""

import csv
import json
import os
import shutil
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Unified_Union_Exp_isolated")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "partitioned_union_base")
PREFIXES = ["FD", "Typo", "NO"]
# Number of horizontal partitions per table (same for all variants of a table)
NUM_PARTITIONS = 10
# Only partition if each partition can have at least this many rows; otherwise keep table as one partition.
MIN_ROWS_PER_PARTITION = 10


def read_csv_file(path):
    """Read a CSV file; return (headers, rows) with all values as strings."""
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
            tables[table_name][prefix] = {"clean": clean_csv, "dirty": dirty_csv}
    return dict(tables)


def partition_indices(n_rows, k):
    """Return k contiguous index ranges (start, end) for n_rows. Last partition may be smaller."""
    if k <= 0 or n_rows <= 0:
        return []
    base_size = n_rows // k
    remainder = n_rows % k
    ranges = []
    start = 0
    for i in range(k):
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def build_partitioned_base():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tables = discover_tables(SOURCE_DIR)
    global_manifest = {
        "source_dir": SOURCE_DIR,
        "num_partitions": NUM_PARTITIONS,
        "tables": {},
    }

    for table_name, variants in sorted(tables.items()):
        if not variants:
            continue
        # Use first variant to get row count and headers; all variants must have same count for alignment
        first_prefix = next(iter(variants))
        _, rows0 = read_csv_file(variants[first_prefix]["dirty"])
        n_rows = len(rows0)
        if n_rows == 0:
            continue  # no data to partition
        # Partition only if each partition can have at least MIN_ROWS_PER_PARTITION rows; else keep as one partition
        max_partitions_by_size = n_rows // MIN_ROWS_PER_PARTITION
        k = max(1, min(NUM_PARTITIONS, max_partitions_by_size)) if max_partitions_by_size else 1
        ranges = partition_indices(n_rows, k)
        table_dir = os.path.join(OUTPUT_DIR, table_name.replace(os.sep, "_"))
        os.makedirs(table_dir, exist_ok=True)
        partition_row_counts = []

        partition_row_counts = [r[1] - r[0] for r in ranges]
        for variant, paths in sorted(variants.items()):
            variant_dir = os.path.join(table_dir, variant)
            headers_d, rows_d = read_csv_file(paths["dirty"])
            headers_c, rows_c = read_csv_file(paths["clean"])
            if len(rows_d) != n_rows or len(rows_c) != n_rows:
                continue
            for p, (start, end) in enumerate(ranges):
                part_dir = os.path.join(variant_dir, f"partition_{p:03d}")
                os.makedirs(part_dir, exist_ok=True)
                write_csv_file(
                    os.path.join(part_dir, "dirty.csv"),
                    headers_d,
                    rows_d[start:end],
                )
                write_csv_file(
                    os.path.join(part_dir, "clean.csv"),
                    headers_c,
                    rows_c[start:end],
                )

        if ranges:
            table_manifest = {
                "variants": sorted(variants.keys()),
                "num_partitions": len(ranges),
                "total_rows": n_rows,
                "partition_row_counts": partition_row_counts,
            }
            with open(os.path.join(table_dir, "manifest.json"), "w") as f:
                json.dump(table_manifest, f, indent=2)
            global_manifest["tables"][table_name] = {
                "variants": table_manifest["variants"],
                "num_partitions": table_manifest["num_partitions"],
                "total_rows": table_manifest["total_rows"],
            }

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(global_manifest, f, indent=2, default=str)
    return global_manifest


def main():
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Partitions per table: {NUM_PARTITIONS}\n")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    manifest = build_partitioned_base()
    n_tables = len(manifest["tables"])
    n_variants = sum(len(t["variants"]) for t in manifest["tables"].values())
    print(f"Partitioned {n_tables} tables ({n_variants} variant(s) total).")
    print(f"Manifest: {os.path.join(OUTPUT_DIR, 'manifest.json')}")
    print("Done.")


if __name__ == "__main__":
    main()
