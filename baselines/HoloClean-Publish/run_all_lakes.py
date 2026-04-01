"""
Run run_baselines.py for multiple data lakes sequentially.
Each lake is run as a separate subprocess to ensure clean Hydra state.
"""

import subprocess
import sys
import time

# List of data lake paths to process
LAKES = [
    "/home/fatemeh/LakeCorrectionBench/datasets/Real_Lake_Default_Datasets/merged_strings_default_set_union/mit_dwh/merged",
    "/home/fatemeh/LakeCorrectionBench/datasets/Real_Lake_Default_Datasets/merged_strings_default_union_all/mit_dwh/merged",
    "/home/fatemeh/LakeCorrectionBench/datasets/Real_Lake_Default_Datasets/merged_strings_default_set_union/uk_open_data/merged",
    "/home/fatemeh/LakeCorrectionBench/datasets/Real_Lake_Default_Datasets/merged_strings_default_union_all/uk_open_data/merged",
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
