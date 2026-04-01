#!/usr/bin/env python3
"""
Flatten partitioned_union_base into a single level of directories.

Input (nested):
  partitioned_union_base/
    {table_name}/
      {variant}/
        partition_000/ dirty.csv, clean.csv
        partition_001/ ...

Output (flat, like Unified_Union_Exp_isolated):
  flattened_partitioned_base/
    DGov_FD_{table_name}_partition_000/
      dirty.csv
      clean.csv
    DGov_FD_{table_name}_partition_001/
    DGov_Typo_{table_name}_partition_000/
    ...
    manifest.json   # partition_dir -> { table, variant, partition_id, row_count }

Each directory is one partition; names mirror the input style (DGov_{variant}_{table}).
"""

import json
import os
import shutil

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PARTITIONED_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "partitioned_union_base")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "flattened_partitioned_base")
# Max length for dir name (filesystem-friendly)
MAX_DIR_NAME_LEN = 200


def safe_dir_name(s, maxlen=MAX_DIR_NAME_LEN):
    """Use as directory name; truncate if needed."""
    return s[:maxlen] if len(s) > maxlen else s


def flatten():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest_path = os.path.join(PARTITIONED_BASE, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"Missing {manifest_path}; run create_partitioned_base.py first.")
    with open(manifest_path) as f:
        global_manifest = json.load(f)

    flat_manifest = {
        "source": PARTITIONED_BASE,
        "partitions": [],
    }
    count = 0

    for table_name, info in sorted(global_manifest.get("tables", {}).items()):
        table_dir = os.path.join(PARTITIONED_BASE, table_name.replace(os.sep, "_"))
        if not os.path.isdir(table_dir):
            continue
        table_manifest_path = os.path.join(table_dir, "manifest.json")
        table_manifest = None
        if os.path.exists(table_manifest_path):
            with open(table_manifest_path) as f:
                table_manifest = json.load(f)
        num_partitions = info.get("num_partitions", table_manifest.get("num_partitions", 0) if table_manifest else 0)
        partition_row_counts = (table_manifest or {}).get("partition_row_counts", [])

        for variant in info.get("variants", []):
            variant_dir = os.path.join(table_dir, variant)
            if not os.path.isdir(variant_dir):
                continue
            for p in range(num_partitions):
                part_dir = os.path.join(variant_dir, f"partition_{p:03d}")
                if not os.path.isdir(part_dir):
                    continue
                dirty_csv = os.path.join(part_dir, "dirty.csv")
                clean_csv = os.path.join(part_dir, "clean.csv")
                if not os.path.exists(dirty_csv) or not os.path.exists(clean_csv):
                    continue
                # Name like input: DGov_{variant}_{table_name}_partition_{p}
                dir_name = safe_dir_name(f"DGov_{variant}_{table_name}_partition_{p:03d}")
                out_dir = os.path.join(OUTPUT_DIR, dir_name)
                os.makedirs(out_dir, exist_ok=True)
                shutil.copy2(dirty_csv, os.path.join(out_dir, "dirty.csv"))
                shutil.copy2(clean_csv, os.path.join(out_dir, "clean.csv"))
                row_count = partition_row_counts[p] if p < len(partition_row_counts) else None
                flat_manifest["partitions"].append({
                    "dir": dir_name,
                    "table": table_name,
                    "variant": variant,
                    "partition_id": p,
                    "row_count": row_count,
                })
                count += 1

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(flat_manifest, f, indent=2, default=str)
    return count, flat_manifest


def main():
    print(f"Source: {PARTITIONED_BASE}")
    print(f"Output: {OUTPUT_DIR}\n")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    count, manifest = flatten()
    print(f"Flattened {count} partition(s) into {OUTPUT_DIR}")
    print(f"Manifest: {os.path.join(OUTPUT_DIR, 'manifest.json')}")
    print("Done.")


if __name__ == "__main__":
    main()
