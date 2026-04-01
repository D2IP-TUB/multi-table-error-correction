#!/usr/bin/env python3
"""
Run merge_tables for multiple JOIN_THRESHOLD values, each writing to a separate lake
(merged directory): 0.25, 0.5, 0.75.

This is the complement of run_union_threshold_lakes.py:
here UNION is kept fixed while JOIN_THRESHOLD varies.
"""

import sys

import config
from merge_tables import merge_tables
from recreate_as_strings_union import recreate_merged_tables_as_strings_union


JOIN_THRESHOLDS = (0.25, 0.5, 0.75)


def main():
    base_merged = config.MERGED_PATH.parent  # e.g. .../tables/mit_dwh

    for thresh in JOIN_THRESHOLDS:
        print(f"\n{'='*60}")
        print(f"Running with JOIN_THRESHOLD = {thresh}")
        print(f"{'='*60}\n")

        config.JOIN_THRESHOLD = thresh
        config.MERGED_PATH = base_merged / f"merged_join_{thresh}"
        config.TRACKER_PATH = config.MERGED_PATH / "tracker.json"

        try:
            merge_tables()
            recreate_merged_tables_as_strings_union()
        except Exception as e:
            print(f"Run failed for join_threshold={thresh}: {e}", file=sys.stderr)
            raise

    print(f"\nDone. Lakes written to:")
    for thresh in JOIN_THRESHOLDS:
        print(f"  {base_merged / f'merged_join_{thresh}'}")


if __name__ == "__main__":
    main()
