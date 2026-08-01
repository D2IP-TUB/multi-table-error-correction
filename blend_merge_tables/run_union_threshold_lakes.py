#!/usr/bin/env python3
"""
Run merge_tables for multiple UNION_THRESHOLD values, each writing to a separate lake
(merged directory): 0.25, 0.5, 0.75.
"""

import sys
from pathlib import Path

import config
from merge_tables import merge_tables
from run_lake_exp import run_recreate_phases


UNION_THRESHOLDS = (0.25, 0.5, 0.75)


def main():
    base_merged = config.MERGED_PATH.parent  # e.g. .../tables/mit_dwh

    for thresh in UNION_THRESHOLDS:
        print(f"\n{'='*60}")
        print(f"Running with UNION_THRESHOLD = {thresh}")
        print(f"{'='*60}\n")

        config.UNION_THRESHOLD = thresh
        config.MERGED_PATH = base_merged / f"merged_union_{thresh}"
        config.TRACKER_PATH = config.MERGED_PATH / "tracker.json"

        try:
            merge_tables()
            config.save_experiment_config(
                result_dir=config.MERGED_PATH,
                script=Path(__file__).name,
                recreate="set_union,bag",
            )
            run_recreate_phases("both")
        except Exception as e:
            print(f"Run failed for union_threshold={thresh}: {e}", file=sys.stderr)
            raise

    print(f"\nDone. Lakes written to:")
    for thresh in UNION_THRESHOLDS:
        print(f"  {base_merged / f'merged_union_{thresh}'}")


if __name__ == "__main__":
    main()
