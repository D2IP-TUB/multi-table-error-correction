#!/usr/bin/env python3
"""
Sort datasets in open_data_uk_filtered by size (rows, columns, cells, file size).

Reads dirty.csv from each dataset folder, collects size metrics,
and prints a sorted table from smallest to largest.
"""

import argparse
import os
import csv
import json
from pathlib import Path


def get_dataset_size_info(dataset_dir: Path) -> dict | None:
    dirty_path = dataset_dir / "dirty.csv"
    if not dirty_path.exists():
        return None

    file_size_bytes = dirty_path.stat().st_size

    with open(dirty_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return None
        num_cols = len(header)
        num_rows = sum(1 for _ in reader)

    return {
        "dataset": dataset_dir.name,
        "rows": num_rows,
        "columns": num_cols,
        "cells": num_rows * num_cols,
        "file_size_bytes": file_size_bytes,
    }


def human_readable_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Sort datasets by size.")
    parser.add_argument(
        "--data-dir",
        default="/home/fatemeh/data/horizon-code/OpenData/open_data_uk_93",
        help="Path to the folder containing dataset directories.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["rows", "columns", "cells", "file_size_bytes"],
        default="rows",
        help="Metric to sort by (default: rows).",
    )
    parser.add_argument(
        "--desc",
        action="store_true",
        help="Sort in descending order (default: ascending).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save results as JSON.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a valid directory.")
        return

    dataset_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir()], key=lambda d: d.name
    )

    results = []
    for d in dataset_dirs:
        info = get_dataset_size_info(d)
        if info:
            results.append(info)

    results.sort(key=lambda x: x[args.sort_by], reverse=args.desc)

    header = f"{'#':>4}  {'Dataset':<45} {'Rows':>8} {'Cols':>6} {'Cells':>10} {'File Size':>10}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for i, r in enumerate(results, 1):
        print(
            f"{i:>4}  {r['dataset']:<45} {r['rows']:>8,} {r['columns']:>6} "
            f"{r['cells']:>10,} {human_readable_size(r['file_size_bytes']):>10}"
        )
    print(separator)
    print(f"Total datasets: {len(results)}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
