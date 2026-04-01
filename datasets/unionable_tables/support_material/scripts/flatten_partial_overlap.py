#!/usr/bin/env python3
"""
Flatten partial-overlap nested dirs into a single level of directories.

Input (nested), e.g. partial_overlap_50_with_duplicates/:
  {table}/
    dirty.csv, clean.csv, lineage.csv          (main union)
    isolated_unique_a_from_var_b/
      dirty.csv, clean.csv, lineage.csv
    isolated_unique_b_from_var_a/
      dirty.csv, clean.csv, lineage.csv

Output (flat):
  flattened_partial_overlap_50_with_duplicates/
    {table}__partial50_union/
    {table}__partial50_isolated_unique_a_from_var_b/
    {table}__partial50_isolated_unique_b_from_var_a/
    DGov_FD_OtherTable/   (if --include-missing-from used)
  manifest.json

So that total rows/cells/errors match the input: partial only has 12 tables (2 variants
each); the rest of the input table-variants are copied in when using --include-missing-from.

Usage:
  python flatten_partial_overlap.py [input_dir [output_dir]]
  python flatten_partial_overlap.py --include-missing-from UNIFIED_INPUT [input_dir [output_dir]]
  If no args: flattens all partial_overlap_* dirs under generated_union_datasets.
  Missing table-variants are always included by default from Unified_Union_Exp_isolated so
  total rows/cells/errors match the input (63 datasets per partial dir).
  --no-include-missing: do not add missing table-variants (only partial union + isolated).
  --include-missing-from DIR: use DIR instead of Unified_Union_Exp_isolated for missing.
"""

import json
import os
import re
import shutil
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GENERATED_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "generated_union_datasets")
UNIFIED_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "Unified_Union_Exp_isolated")
METADATA_PATH = os.path.join(GENERATED_ROOT, "metadata.json")
MAX_DIR_NAME_LEN = 200

MAIN_FILES = ("dirty.csv", "clean.csv", "lineage.csv")


def safe_dir_name(s, maxlen=MAX_DIR_NAME_LEN):
    return s[:maxlen] if len(s) > maxlen else s


def extract_pct_from_dirname(dirname):
    """e.g. partial_overlap_50_with_duplicates -> 50."""
    m = re.search(r"partial_overlap_(\d+)_", dirname)
    return m.group(1) if m else ""


def get_partial_table_variants(metadata_path, partial_key="partial_overlap_50"):
    """Return set of (table_name, variant) that are used in partial overlap (var_a and var_b per table)."""
    used = set()
    if not os.path.exists(metadata_path):
        return used
    with open(metadata_path) as f:
        meta = json.load(f)
    for key in ("partial_overlap_25", "partial_overlap_50", "partial_overlap_75"):
        for entry in meta.get(key, []):
            t = entry.get("table")
            va = entry.get("variant_a")
            vb = entry.get("variant_b")
            if t and va:
                used.add((t, va))
            if t and vb:
                used.add((t, vb))
    return used


def add_missing_table_variants(output_dir, input_root, partial_table_variants):
    """
    Copy from input_root every DGov_{variant}_{table} whose (table, variant) is not
    in partial_table_variants, so that total rows/cells/errors match the full input.
    """
    if not os.path.isdir(input_root):
        return 0
    count = 0
    for name in sorted(os.listdir(input_root)):
        if not name.startswith("DGov_"):
            continue
        dirpath = os.path.join(input_root, name)
        if not os.path.isdir(dirpath):
            continue
        parts = name.split("_", 2)
        if len(parts) < 3:
            continue
        variant, table_name = parts[1], parts[2]
        if (table_name, variant) in partial_table_variants:
            continue
        dirty = os.path.join(dirpath, "dirty.csv")
        clean = os.path.join(dirpath, "clean.csv")
        if not os.path.exists(dirty) or not os.path.exists(clean):
            continue
        out = os.path.join(output_dir, name)
        os.makedirs(out, exist_ok=True)
        shutil.copy2(dirty, os.path.join(out, "dirty.csv"))
        shutil.copy2(clean, os.path.join(out, "clean.csv"))
        if os.path.exists(os.path.join(dirpath, "lineage.csv")):
            shutil.copy2(os.path.join(dirpath, "lineage.csv"), os.path.join(out, "lineage.csv"))
        count += 1
    return count


def flatten_one(source_dir, output_dir, pct_label=None, include_missing_from=None):
    """
    Flatten one partial_overlap_* directory into output_dir.
    pct_label: e.g. "50" from partial_overlap_50_*; used in flat dir names.
    """
    if pct_label is None:
        pct_label = extract_pct_from_dirname(os.path.basename(source_dir.rstrip(os.sep)))
    prefix = f"partial{pct_label}_" if pct_label else "partial_"
    os.makedirs(output_dir, exist_ok=True)
    manifest_entries = []
    count = 0

    for name in sorted(os.listdir(source_dir)):
        table_dir = os.path.join(source_dir, name)
        if not os.path.isdir(table_dir):
            continue
        # Main union: dirty.csv, clean.csv, lineage.csv in table_dir
        main_dirty = os.path.join(table_dir, "dirty.csv")
        main_clean = os.path.join(table_dir, "clean.csv")
        main_lineage = os.path.join(table_dir, "lineage.csv")
        if os.path.exists(main_dirty) and os.path.exists(main_clean):
            flat_name = safe_dir_name(f"{name}__{prefix}union")
            out = os.path.join(output_dir, flat_name)
            os.makedirs(out, exist_ok=True)
            for f in MAIN_FILES:
                src = os.path.join(table_dir, f)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(out, f))
            manifest_entries.append({"dir": flat_name, "table": name, "type": "union"})
            count += 1
        # Isolated subdirs (discovered dynamically, e.g. isolated_unique_b_from_var_a_p3)
        for sub in sorted(os.listdir(table_dir)):
            subpath = os.path.join(table_dir, sub)
            if not os.path.isdir(subpath) or not sub.startswith("isolated_"):
                continue
            if os.path.exists(os.path.join(subpath, "dirty.csv")) and os.path.exists(os.path.join(subpath, "clean.csv")):
                flat_name = safe_dir_name(f"{name}__{prefix}{sub}")
                out = os.path.join(output_dir, flat_name)
                os.makedirs(out, exist_ok=True)
                for f in MAIN_FILES:
                    src = os.path.join(subpath, f)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(out, f))
                manifest_entries.append({"dir": flat_name, "table": name, "type": sub})
                count += 1

    if include_missing_from:
        partial_pairs = get_partial_table_variants(
            os.path.join(os.path.dirname(source_dir), "metadata.json")
        )
        added = add_missing_table_variants(output_dir, include_missing_from, partial_pairs)
        if added:
            manifest_entries.append({"dir": f"(+{added} missing table-variants from input)", "type": "include_missing"})
        count += added

    manifest = {"source": source_dir, "overlap_pct": pct_label, "partitions": manifest_entries}
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return count, manifest


def main():
    args = list(sys.argv[1:])
    include_missing_from = UNIFIED_INPUT  # default: always include all table-variants
    while args:
        if args[0] == "--no-include-missing":
            include_missing_from = None
            args = args[1:]
        elif args[0] == "--include-missing-from":
            if len(args) < 2:
                print("Error: --include-missing-from requires DIR")
                sys.exit(1)
            include_missing_from = os.path.abspath(args[1])
            args = args[2:]
        else:
            break
    if len(args) >= 2:
        source_dir = os.path.abspath(args[0])
        output_dir = os.path.abspath(args[1])
        if not os.path.isdir(source_dir):
            print(f"Error: not a directory: {source_dir}")
            sys.exit(1)
        pct = extract_pct_from_dirname(os.path.basename(source_dir.rstrip(os.sep)))
        count, _ = flatten_one(source_dir, output_dir, pct, include_missing_from=include_missing_from)
        print(f"Flattened {count} partition(s) into {output_dir}")
        return
    if len(args) == 1:
        source_dir = os.path.abspath(args[0])
        if not os.path.isdir(source_dir):
            print(f"Error: not a directory: {source_dir}")
            sys.exit(1)
        base = os.path.basename(source_dir.rstrip(os.sep))
        output_dir = os.path.join(os.path.dirname(source_dir), f"flattened_{base}")
        pct = extract_pct_from_dirname(base)
        count, _ = flatten_one(source_dir, output_dir, pct, include_missing_from=include_missing_from)
        print(f"Flattened {count} partition(s) into {output_dir}")
        return
    # No args: find all partial_overlap_* under generated_union_datasets
    if not os.path.isdir(GENERATED_ROOT):
        print(f"Error: generated root not found: {GENERATED_ROOT}")
        sys.exit(1)
    total = 0
    for name in sorted(os.listdir(GENERATED_ROOT)):
        if not name.startswith("partial_overlap_") or ("_with_duplicates" not in name and "_without_duplicates" not in name):
            continue
        source_dir = os.path.join(GENERATED_ROOT, name)
        if not os.path.isdir(source_dir):
            continue
        output_dir = os.path.join(GENERATED_ROOT, f"flattened_{name}")
        pct = extract_pct_from_dirname(name)
        count, _ = flatten_one(source_dir, output_dir, pct, include_missing_from=include_missing_from)
        print(f"  {name} -> flattened_{name}: {count} partition(s)")
        total += count
    print(f"Total: {total} partition(s) across all partial overlap dirs.")


if __name__ == "__main__":
    main()
