#!/usr/bin/env python3
"""Rerun HyFD (metanome-cli) for ablation table dirs with empty holo_constraints.txt."""
import logging
import os
import sys
from pathlib import Path

HFE = Path(__file__).resolve().parent
ABLATIONS = Path("/home/fatemeh/LakeCorrectionBench/datasets/ablations")

os.chdir(HFE)
sys.path.insert(0, str(HFE))

from DataLakeFDExtractor.src.extract_fds import process_clean_csv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("rerun_empty_hyfd")


def empty_table_clean_csvs():
    for a in sorted(ABLATIONS.iterdir()):
        if not a.is_dir():
            continue
        for b in sorted(a.iterdir()):
            if not b.is_dir():
                continue
            for c in sorted(b.iterdir()):
                if not c.is_dir():
                    continue
                hc = c / "holo_constraints.txt"
                if hc.is_file() and hc.stat().st_size == 0:
                    yield c / "clean.csv"


def main():
    paths = sorted(empty_table_clean_csvs(), key=lambda p: str(p))
    log.info("Rerunning HyFD for %d tables (empty holo_constraints.txt)", len(paths))
    still_empty = []
    for i, clean in enumerate(paths, 1):
        rel = clean.relative_to(ABLATIONS)
        log.info("[%d/%d] %s", i, len(paths), rel)
        process_clean_csv(str(clean))
        hc = clean.parent / "holo_constraints.txt"
        if hc.is_file() and hc.stat().st_size == 0:
            still_empty.append(str(rel.parent))

    log.info("Done. Still empty: %d", len(still_empty))
    for p in still_empty:
        log.warning("  %s", p)


if __name__ == "__main__":
    main()
