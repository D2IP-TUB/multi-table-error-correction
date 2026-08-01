"""
Run run_baselines.py for multiple data lakes sequentially.
Each lake is run as a separate subprocess to ensure clean Hydra state.
"""

from pathlib import Path
import subprocess
import sys
import time

# List of data lake paths to process
_REPO = Path(__file__).resolve().parents[2]
LAKES = [
    str(_REPO / 'datasets' / 'real_lakes' / 'open_data_uk_merged_set_union'),
    str(_REPO / 'datasets' / 'real_lakes' / 'open_data_uk_merged_multiset_union'),
    # MIT-DW proprietary: add datasets/real_lakes/mit_dwh_merged_* when available
]

# Optional: override any other Hydra config parameters here
EXTRA_OVERRIDES = [
    # "num_iterations=1",
    # "threads=1",
]


def main():
    results = {}
    for i, lake_path in enumerate(LAKES, 1):
        lake_name = lake_path.rstrip("/").split("/")[-1]
        print(f"\n{'='*60}")
        print(f"[{i}/{len(LAKES)}] Running baselines for lake: {lake_name}")
        print(f"  Path: {lake_path}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, "run_baselines.py",
            f"input_data_lake_path={lake_path}",
            *EXTRA_OVERRIDES,
        ]

        start = time.time()
        result = subprocess.run(cmd, cwd=sys.path[0] or ".")
        elapsed = time.time() - start

        status = "SUCCESS" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
        results[lake_name] = status
        print(f"\n>> {lake_name}: {status}  ({elapsed:.1f}s)")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for lake_name, status in results.items():
        print(f"  {lake_name}: {status}")


if __name__ == "__main__":
    main()
