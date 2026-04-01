#!/usr/bin/env python3
"""
Sum wall-time spent on Uniclean lake runs from per-table logs in uniclean_results/.

Rules (per-table logs: ``table_*.log`` or ``<digits>.log`` in ``uniclean_results/``):
  - If the log ends with a subprocess timeout kill (or timeout appears after any
    "Total cleaning time"), count the configured timeout seconds (default 3600)
    for that attempt — you waited that long.
  - Else if "Total cleaning time: X" is present, use X.
  - Else 0 (no run recorded).

This is sequential-time accounting (sum over tables). Parallel runs lower wall clock.

Usage:
  python summarize_uniclean_lake_runtime.py /path/to/uniclean_results
  python summarize_uniclean_lake_runtime.py /path/to/uniclean_results --timeout 7200
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "uniclean_results_dir",
        help="Directory containing per-table logs (table_*.log or numeric names like 0.log, 42.log)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Seconds to attribute to each timed-out run (default: 3600)",
    )
    return p.parse_args()


def seconds_for_log(text: str, default_timeout: int) -> tuple[float, str]:
    time_pat = re.compile(r"Total cleaning time:\s*([\d.]+)", re.I)
    timeout_pat = re.compile(r"exceeded\s+(\d+)s\s+timeout", re.I)

    m_time = time_pat.search(text)
    has_kill = "KILLED" in text and "timeout" in text.lower()
    m_to = timeout_pat.search(text)
    timeout_sec = int(m_to.group(1)) if m_to else default_timeout

    last_kill = text.rfind("KILLED")
    last_time = text.rfind("Total cleaning time")

    if has_kill and last_kill > last_time:
        return float(timeout_sec), "timeout"
    if m_time:
        return float(m_time.group(1)), "completed"
    return 0.0, "no_run"


def discover_log_paths(d: str) -> list[str]:
    """Per-table logs: table_*.log (open_data_uk, mit_dw) or <digits>.log (merged lakes)."""
    num_pat = re.compile(r"^\d+\.log$")
    out: list[str] = []
    for path in glob.glob(os.path.join(d, "*.log")):
        base = os.path.basename(path)
        if base.startswith("table_") or num_pat.fullmatch(base):
            out.append(path)
    return sorted(out)


def main() -> None:
    args = parse_args()
    d = args.uniclean_results_dir
    if not os.path.isdir(d):
        print(f"Error: not a directory: {d}", file=sys.stderr)
        sys.exit(1)

    paths = discover_log_paths(d)
    total = 0.0
    by_kind: dict[str, list[tuple[str, float]]] = {"completed": [], "timeout": [], "no_run": []}

    for path in paths:
        name = os.path.basename(path)
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError as e:
            print(f"Warning: could not read {name}: {e}", file=sys.stderr)
            by_kind["no_run"].append((name, 0.0))
            continue
        sec, kind = seconds_for_log(text, args.timeout)
        total += sec
        by_kind[kind].append((name, sec))

    print(f"Directory: {d}")
    print(f"Logs: {len(paths)}")
    print(f"Default timeout for kill lines without explicit Ns: {args.timeout}s")
    print()
    for kind in ("completed", "timeout", "no_run"):
        items = by_kind[kind]
        s = sum(x[1] for x in items)
        print(f"  {kind:12}  {len(items):3} tables   {s:12.2f} s   ({s/3600:.4f} h)")
    print()
    print(f"  TOTAL         {len(paths):3} tables   {total:12.2f} s   ({total/3600:.4f} h)")
    print(f"                ({total/60:.2f} min)")


if __name__ == "__main__":
    main()
