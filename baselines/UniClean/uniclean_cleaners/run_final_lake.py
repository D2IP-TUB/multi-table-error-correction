"""
Run main.py on every table in a Final_Datasets lake directory, then
aggregate metrics with optional per-partition and per-source-variant
breakdowns from lineage.csv.

Runtime logging (when not --skip_cleaning):
  - Master log: logs/<lake_name>_runtime_<timestamp>.log
  - Per-table: === START/END === lines with wall_s, clean_time_s, charged_s
  - Summary:   <output_dir>/runtime_summary.json
               <output_dir>/per_table_runtime.csv

Charged seconds: time(s) for OK, timeout for TIMEOUT, wall for FAILED, 0 for
EMPTY. Idle time between manual restarts is not included.

Usage:
    python run_final_lake.py --lake_dir /path/to/flattened_partial_overlap_50_without_duplicates
    python run_final_lake.py --lake_dir /path/to/maximal_overlap_with_duplicates --skip_cleaning
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from AnalyticsCache.getScore import normalize_for_cmp
from quintet_eval import (
    INDEX_COL,
    aggregate_lake_metrics,
    count_ground_truth_errors,
    ensure_index_column,
    evaluate_quintet_table,
    format_lake_summary,
    lake_evaluation_json,
    metrics_result_row,
    skipped_table_metrics,
    unify_missing_tokens,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run UniClean on all tables in a Final_Datasets lake and aggregate results."
    )
    p.add_argument('--lake_dir', type=str, required=True,
                   help="Root directory of the lake (contains one sub-dir per table).")
    p.add_argument('--output_dir', type=str, default=None,
                   help="Where to write aggregated evaluation. "
                        "Defaults to <lake_dir>/uniclean_results/.")
    p.add_argument('--single_max', type=int, default=10000)
    p.add_argument('--timeout', type=int, default=3600,
                   help="Per-table timeout in seconds (default: 3600).")
    p.add_argument('--driver_memory', type=str, default='48g',
                   help="Spark driver memory (default: 48g).")
    p.add_argument('--spark_master', type=str, default=None,
                   help="Spark master URL, e.g. 'local[16]' to limit cores per table.")
    p.add_argument('--missing_token', type=str, default='empty',
                   help="Missing-value token for evaluation (default: empty).")
    p.add_argument('--skip_cleaning', action='store_true',
                   help="Skip cleaning; only aggregate existing results.")
    return p.parse_args()


def _table_size_mb(table_dir):
    try:
        return os.path.getsize(os.path.join(table_dir, 'dirty.csv')) / (1024 * 1024)
    except Exception:
        return 0.0


def discover_table_dirs(lake_dir):
    """Return valid table directories sorted by dirty.csv size."""
    dirs = []
    for name in sorted(os.listdir(lake_dir)):
        full = os.path.join(lake_dir, name)
        if not os.path.isdir(full):
            continue
        required = ['dirty.csv', 'clean.csv', 'holo_constraints.txt']
        if all(os.path.isfile(os.path.join(full, f)) for f in required):
            dirs.append(full)
    dirs.sort(key=_table_size_mb)
    return dirs


def align_dirty_columns_to_clean(table_dir):
    """Rewrite dirty.csv header to match clean.csv."""
    clean_path = os.path.join(table_dir, 'clean.csv')
    dirty_path = os.path.join(table_dir, 'dirty.csv')

    with open(clean_path, 'r', newline='') as f:
        clean_header = next(csv.reader(f))
    with open(dirty_path, 'r', newline='') as f:
        reader = csv.reader(f)
        dirty_header = next(reader)
        if dirty_header == clean_header:
            return False
        rows = list(reader)

    if len(dirty_header) != len(clean_header):
        print(f"  WARNING: column count mismatch in {os.path.basename(table_dir)} "
              f"(dirty={len(dirty_header)}, clean={len(clean_header)}), skipping rename")
        return False

    with open(dirty_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(clean_header)
        writer.writerows(rows)
    return True


_DGOV_VARIANT_RE = re.compile(r'^DGov_(FD|NO|Typo)_')
_TIME_S_RE = re.compile(r'time\(s\):\s*([0-9.]+)')
_EMPTY_CONSTRAINTS_RE = re.compile(r'No valid constraints found')


def _utc_now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _local_now_stamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _parse_clean_time_s(log_text):
    matches = _TIME_S_RE.findall(log_text)
    return float(matches[-1]) if matches else None


def _is_empty_constraints(log_text):
    return bool(_EMPTY_CONSTRAINTS_RE.search(log_text))


class RuntimeLogger:
    """Master runtime log + structured per-table records for lake charging.

    Charged seconds (idle gaps between manual restarts are not included — only
    recorded attempts):
      - OK: last ``time(s):`` from the table log (else wall seconds)
      - TIMEOUT: configured timeout seconds
      - EMPTY: 0 (no valid constraints)
      - FAILED: wall seconds of the attempt
    """

    def __init__(self, lake_dir, output_dir, timeout_s):
        self.lake_dir = lake_dir
        self.output_dir = output_dir
        self.timeout_s = timeout_s
        self.records = []
        self._lake_started = time.time()
        self._lake_started_iso = _utc_now_iso()

        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        lake_name = os.path.basename(os.path.abspath(lake_dir).rstrip(os.sep)) or 'lake'
        stamp = _local_now_stamp()
        self.master_path = os.path.join(logs_dir, f'{lake_name}_runtime_{stamp}.log')
        self.summary_json_path = os.path.join(output_dir, 'runtime_summary.json')
        self.per_table_csv_path = os.path.join(output_dir, 'per_table_runtime.csv')
        self._fh = open(self.master_path, 'w', buffering=1)

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.close()

    def log(self, msg, also_print=True):
        line = msg if msg.endswith('\n') else msg + '\n'
        self._fh.write(line)
        self._fh.flush()
        if also_print:
            print(msg)

    def write_header(self, n_tables, args):
        self.log('=' * 70)
        self.log(f'RUNTIME LOG  started={self._lake_started_iso}')
        self.log(f'lake_dir     = {self.lake_dir}')
        self.log(f'output_dir   = {self.output_dir}')
        self.log(f'tables       = {n_tables}')
        self.log(f'timeout_s    = {self.timeout_s}')
        self.log(f'single_max   = {args.single_max}')
        self.log(f'driver_memory= {args.driver_memory}')
        self.log(f'spark_master = {args.spark_master}')
        self.log(f'master_log   = {self.master_path}')
        self.log('=' * 70)

    def start_table(self, idx, n_tables, tname, size_mb):
        started = time.time()
        started_iso = _utc_now_iso()
        self.log(
            f'=== START table={tname} idx={idx}/{n_tables} '
            f'size_mb={size_mb:.2f} at={started_iso} ==='
        )
        return started, started_iso

    def end_table(self, tname, started, started_iso, outcome, exit_code, log_file):
        wall_s = time.time() - started
        ended_iso = _utc_now_iso()
        log_text = ''
        if log_file and os.path.isfile(log_file):
            try:
                with open(log_file, 'r', errors='replace') as f:
                    log_text = f.read()
            except OSError:
                log_text = ''

        clean_time_s = _parse_clean_time_s(log_text)
        empty = _is_empty_constraints(log_text)

        if outcome == 'timeout':
            category = 'timeout'
            charged_s = float(self.timeout_s)
        elif empty and outcome != 'ok':
            category = 'empty'
            charged_s = 0.0
            outcome = 'empty'
        elif outcome == 'ok':
            category = 'ok'
            charged_s = float(clean_time_s) if clean_time_s is not None else wall_s
        else:
            category = 'failed'
            charged_s = wall_s

        rec = {
            'table': tname,
            'outcome': outcome,
            'category': category,
            'exit_code': exit_code,
            'started_at': started_iso,
            'ended_at': ended_iso,
            'wall_s': round(wall_s, 3),
            'clean_time_s': None if clean_time_s is None else round(clean_time_s, 6),
            'charged_s': round(charged_s, 3),
            'log_file': log_file,
        }
        self.records.append(rec)
        self.log(
            f'=== END table={tname} outcome={outcome} category={category} '
            f'exit={exit_code} wall_s={wall_s:.3f} '
            f'clean_time_s={clean_time_s if clean_time_s is not None else "NA"} '
            f'charged_s={charged_s:.3f} at={ended_iso} ==='
        )
        return rec

    def write_summary(self):
        by_cat = defaultdict(lambda: {'n': 0, 'charged_s': 0.0, 'wall_s': 0.0})
        for rec in self.records:
            b = by_cat[rec['category']]
            b['n'] += 1
            b['charged_s'] += rec['charged_s']
            b['wall_s'] += rec['wall_s']

        total_charged = sum(r['charged_s'] for r in self.records)
        total_wall = sum(r['wall_s'] for r in self.records)
        lake_wall = time.time() - self._lake_started
        summary = {
            'lake_dir': self.lake_dir,
            'output_dir': self.output_dir,
            'master_log': self.master_path,
            'started_at': self._lake_started_iso,
            'ended_at': _utc_now_iso(),
            'timeout_s': self.timeout_s,
            'n_tables': len(self.records),
            'by_category': {
                k: {
                    'n': v['n'],
                    'charged_s': round(v['charged_s'], 3),
                    'wall_s': round(v['wall_s'], 3),
                }
                for k, v in sorted(by_cat.items())
            },
            'total_charged_s': round(total_charged, 3),
            'total_charged_rounded': round(total_charged),
            'total_attempt_wall_s': round(total_wall, 3),
            'lake_wall_s': round(lake_wall, 3),
            'note': (
                'charged_s = time(s) for ok, timeout_s for TIMEOUT, '
                'wall_s for FAILED, 0 for EMPTY. Idle gaps between '
                'manual restarts are not included.'
            ),
            'tables': self.records,
        }

        self.log('')
        self.log('=' * 70)
        self.log('RUNTIME SUMMARY')
        self.log('=' * 70)
        for cat, v in summary['by_category'].items():
            self.log(
                f"  {cat:10s}  n={v['n']:3d}  "
                f"charged_s={v['charged_s']:10.3f}  wall_s={v['wall_s']:10.3f}"
            )
        self.log(
            f"  TOTAL      n={summary['n_tables']:3d}  "
            f"charged_s={summary['total_charged_s']:10.3f}  "
            f"rounded={summary['total_charged_rounded']}  "
            f"({summary['total_charged_rounded']/3600:.2f} h)"
        )
        self.log(
            f"  attempt_wall_s={summary['total_attempt_wall_s']:.3f}  "
            f"lake_wall_s={summary['lake_wall_s']:.3f}  "
            f"(lake_wall includes only this process; not prior manual gaps)"
        )
        self.log('=' * 70)

        with open(self.summary_json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        if self.records:
            pd.DataFrame(self.records).to_csv(self.per_table_csv_path, index=False)

        self.log(f'Wrote {self.summary_json_path}')
        self.log(f'Wrote {self.per_table_csv_path}')
        return summary


def load_lineage(table_dir):
    """Load lineage.csv if present; synthesize for DGov_* tables when absent."""
    lineage_path = os.path.join(table_dir, 'lineage.csv')
    if os.path.isfile(lineage_path):
        return pd.read_csv(lineage_path, dtype={'row_idx': int, 'source_row_idx': int})

    tname = os.path.basename(table_dir)
    m = _DGOV_VARIANT_RE.match(tname)
    if m:
        variant = m.group(1)
        source_table = _DGOV_VARIANT_RE.sub('', tname)
        dirty_path = os.path.join(table_dir, 'dirty.csv')
        n_rows = sum(1 for _ in open(dirty_path)) - 1
        return pd.DataFrame({
            'row_idx': range(n_rows),
            'source_table': source_table,
            'source_variant': variant,
            'source_row_idx': range(n_rows),
            'partition': 'all',
        })

    return None


def _read_table_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def compute_lineage_counts(clean_df, dirty_df, cleaned_df, lineage_df, missing_token='empty'):
    """Correction TP/FP/errors broken down by partition and source_variant."""
    index_col = INDEX_COL
    for df in [clean_df, dirty_df, cleaned_df]:
        if index_col not in df.columns:
            df.insert(0, index_col, range(len(df)))
        unify_missing_tokens(df, missing_token)

    clean_r = clean_df.set_index(index_col)
    dirty_r = dirty_df.set_index(index_col)
    cleaned_r = cleaned_df.set_index(index_col)
    data_cols = [c for c in clean_r.columns if c in dirty_r.columns and c in cleaned_r.columns]

    def _zero():
        return {"tp": 0, "fp": 0, "errors": 0}

    by_partition = defaultdict(_zero)
    by_variant = defaultdict(_zero)

    for _, lin_row in lineage_df.iterrows():
        row_idx = int(lin_row['row_idx'])
        partition = str(lin_row['partition'])
        variant = str(lin_row['source_variant'])

        if row_idx not in clean_r.index:
            continue

        for col in data_cols:
            clean_v = normalize_for_cmp(clean_r.at[row_idx, col])
            dirty_v = normalize_for_cmp(dirty_r.at[row_idx, col])
            cleaned_v = normalize_for_cmp(cleaned_r.at[row_idx, col])

            actual_error = dirty_v != clean_v
            changed = dirty_v != cleaned_v

            if actual_error:
                by_partition[partition]["errors"] += 1
                by_variant[variant]["errors"] += 1
            if actual_error and changed and cleaned_v == clean_v:
                by_partition[partition]["tp"] += 1
                by_variant[variant]["tp"] += 1
            if actual_error and changed and cleaned_v != clean_v:
                by_partition[partition]["fp"] += 1
                by_variant[variant]["fp"] += 1

    return dict(by_partition), dict(by_variant)


def _accumulate(target, source):
    for key, counts in source.items():
        if key not in target:
            target[key] = {"tp": 0, "fp": 0, "errors": 0}
        target[key]["tp"] += counts["tp"]
        target[key]["fp"] += counts["fp"]
        target[key]["errors"] += counts["errors"]


def _prf(tp, fp, errors):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / errors if errors > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _print_breakdown(title, breakdown):
    print(f"\n{'=' * 85}")
    print(title)
    print('=' * 85)
    print(f"  {'Category':<30} {'errors':>9} {'cor_TP':>9} {'cor_FP':>9} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-' * 83}")
    for key in sorted(breakdown.keys()):
        m = breakdown[key]
        prec, rec, f1 = _prf(m["tp"], m["fp"], m["errors"])
        print(f"  {key:<30} {m['errors']:>9} {m['tp']:>9} {m['fp']:>9} "
              f"{prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")
    print('=' * 85)


def main():
    args = parse_args()
    lake_dir = args.lake_dir
    output_dir = args.output_dir or os.path.join(lake_dir, 'uniclean_results')
    os.makedirs(output_dir, exist_ok=True)

    table_dirs = discover_table_dirs(lake_dir)
    print(f"Discovered {len(table_dirs)} table(s) in {lake_dir}")

    print("Aligning dirty.csv columns to clean.csv ...")
    renamed = sum(1 for t in table_dirs if align_dirty_columns_to_clean(t))
    print(f"  Renamed headers in {renamed} file(s)." if renamed
          else "  All headers already aligned.")

    print("Checking index columns ...")
    indexed = 0
    for tdir in table_dirs:
        for fname in ('dirty.csv', 'clean.csv'):
            if ensure_index_column(os.path.join(tdir, fname)):
                indexed += 1
    print(f"  Added index column to {indexed} file(s)." if indexed
          else "  All files already have an index column.")

    main_py = os.path.join(os.path.dirname(__file__), 'main.py')
    runtime = None

    if not args.skip_cleaning:
        runtime = RuntimeLogger(lake_dir, output_dir, args.timeout)
        try:
            runtime.write_header(len(table_dirs), args)
            for i, tdir in enumerate(table_dirs):
                tname = os.path.basename(tdir)
                size_mb = _table_size_mb(tdir)
                log_file = os.path.join(output_dir, f'{tname}.log')
                runtime.log(
                    f"[{i+1}/{len(table_dirs)}] Cleaning: {tname}  "
                    f"({size_mb:.2f} MB, timeout={args.timeout}s)"
                )
                started, started_iso = runtime.start_table(
                    i + 1, len(table_dirs), tname, size_mb,
                )

                cmd = [
                    sys.executable, main_py,
                    '--dataset_dir', tdir,
                    '--table_name', tname,
                    '--single_max', str(args.single_max),
                    '--driver_memory', args.driver_memory,
                    '--missing_token', args.missing_token,
                ]
                if args.spark_master:
                    cmd += ['--spark_master', args.spark_master]
                try:
                    with open(log_file, 'w') as lf:
                        ret = subprocess.run(
                            cmd, stdout=lf, stderr=subprocess.STDOUT,
                            cwd=os.path.dirname(main_py), timeout=args.timeout,
                        )
                    if ret.returncode != 0:
                        runtime.log(
                            f"  -> FAILED (exit {ret.returncode}), see {log_file}"
                        )
                        runtime.end_table(
                            tname, started, started_iso,
                            outcome='failed', exit_code=ret.returncode,
                            log_file=log_file,
                        )
                    else:
                        runtime.log("  -> OK")
                        runtime.end_table(
                            tname, started, started_iso,
                            outcome='ok', exit_code=0, log_file=log_file,
                        )
                except subprocess.TimeoutExpired:
                    runtime.log(
                        f"  -> TIMEOUT after {args.timeout}s — killed, "
                        f"moving to next table"
                    )
                    with open(log_file, 'a') as lf:
                        lf.write(
                            f"\n\n=== KILLED: exceeded {args.timeout}s timeout ===\n"
                        )
                    runtime.end_table(
                        tname, started, started_iso,
                        outcome='timeout', exit_code=124, log_file=log_file,
                    )
        finally:
            if runtime.records:
                try:
                    runtime.write_summary()
                except Exception as e:
                    print(f"WARNING: failed to write runtime summary: {e}")
            runtime.close()
            print(f"Runtime master log: {runtime.master_path}")

    print("\n" + "=" * 70)
    print("AGGREGATED EVALUATION")
    print("=" * 70)

    lake_rows = 0
    tables_ok, tables_skipped, tables_failed = 0, 0, 0
    per_table_rows = []
    lake_by_partition = {}
    lake_by_variant = {}

    for tdir in table_dirs:
        tname = os.path.basename(tdir)
        cleaned_csv = os.path.join(tdir, 'result', tname, f'{tname}Cleaned.csv')
        clean_path = os.path.join(tdir, 'clean.csv')
        dirty_path = os.path.join(tdir, 'dirty.csv')

        try:
            lineage_df = load_lineage(tdir)
        except Exception as e:
            print(f"  WARNING: could not load lineage for {tname}: {e}")
            lineage_df = None

        if not os.path.isfile(cleaned_csv):
            try:
                errors = count_ground_truth_errors(clean_path, dirty_path, args.missing_token)
                lake_rows += len(_read_table_csv(clean_path))
                tables_skipped += 1

                if lineage_df is not None:
                    clean_df = _read_table_csv(clean_path)
                    dirty_df = _read_table_csv(dirty_path)
                    bp, bv = compute_lineage_counts(
                        clean_df, dirty_df, dirty_df, lineage_df, args.missing_token,
                    )
                    _accumulate(lake_by_partition, bp)
                    _accumulate(lake_by_variant, bv)

                per_table_rows.append(metrics_result_row(tname, 'no_result', skipped_table_metrics(errors)))
            except Exception as e:
                tables_failed += 1
                per_table_rows.append(metrics_result_row(tname, f'load_error: {e}', {}))
            continue

        try:
            metrics = evaluate_quintet_table(
                clean_path, dirty_path, cleaned_csv,
                missing_token=args.missing_token,
            )
            lake_rows += len(_read_table_csv(clean_path))
            tables_ok += 1

            if lineage_df is not None:
                clean_df = _read_table_csv(clean_path)
                dirty_df = _read_table_csv(dirty_path)
                cleaned_df = _read_table_csv(cleaned_csv)
                bp, bv = compute_lineage_counts(
                    clean_df, dirty_df, cleaned_df, lineage_df, args.missing_token,
                )
                _accumulate(lake_by_partition, bp)
                _accumulate(lake_by_variant, bv)

            per_table_rows.append(metrics_result_row(tname, 'ok', metrics))
        except Exception as e:
            try:
                errors = count_ground_truth_errors(clean_path, dirty_path, args.missing_token)
                lake_rows += len(_read_table_csv(clean_path))
                tables_failed += 1

                if lineage_df is not None:
                    clean_df = _read_table_csv(clean_path)
                    dirty_df = _read_table_csv(dirty_path)
                    bp, bv = compute_lineage_counts(
                        clean_df, dirty_df, dirty_df, lineage_df, args.missing_token,
                    )
                    _accumulate(lake_by_partition, bp)
                    _accumulate(lake_by_variant, bv)

                per_table_rows.append(
                    metrics_result_row(
                        tname, f'eval_error_but_counted: {str(e)[:50]}', skipped_table_metrics(errors)
                    )
                )
            except Exception:
                tables_failed += 1
                per_table_rows.append(metrics_result_row(tname, f'eval_error: {e}', {}))

    lake = aggregate_lake_metrics(per_table_rows)
    summary = format_lake_summary(
        lake_dir, table_dirs, tables_ok, tables_skipped, tables_failed, lake, lake_rows,
    )
    print(summary)

    if lake_by_partition:
        _print_breakdown("PER-PARTITION LAKE-WIDE CORRECTION RESULTS", lake_by_partition)
    else:
        print("\n(No lineage partition data available.)")

    if lake_by_variant:
        _print_breakdown("PER-SOURCE-VARIANT LAKE-WIDE CORRECTION RESULTS", lake_by_variant)
    else:
        print("\n(No lineage source-variant data available.)")

    pd.DataFrame(per_table_rows).to_csv(
        os.path.join(output_dir, 'per_table_results.csv'), index=False
    )

    with open(os.path.join(output_dir, 'lake_evaluation.txt'), 'w') as f:
        f.write(summary)

    def _breakdown_to_list(breakdown):
        rows = []
        for key in sorted(breakdown.keys()):
            m = breakdown[key]
            prec, rec, f1 = _prf(m["tp"], m["fp"], m["errors"])
            rows.append({
                "category": key, "errors": m["errors"],
                "cor_tp": m["tp"], "cor_fp": m["fp"],
                "precision": prec, "recall": rec, "f1": f1,
            })
        return rows

    partition_rows = _breakdown_to_list(lake_by_partition) if lake_by_partition else []
    variant_rows = _breakdown_to_list(lake_by_variant) if lake_by_variant else []

    lake_json = lake_evaluation_json(tables_ok, tables_skipped, tables_failed, lake_rows, lake)
    lake_json['per_partition'] = partition_rows
    lake_json['per_source_variant'] = variant_rows

    with open(os.path.join(output_dir, 'lake_evaluation.json'), 'w') as f:
        json.dump(lake_json, f, indent=2)

    if partition_rows:
        pd.DataFrame(partition_rows).to_csv(
            os.path.join(output_dir, 'per_partition_results.csv'), index=False
        )
    if variant_rows:
        pd.DataFrame(variant_rows).to_csv(
            os.path.join(output_dir, 'per_variant_results.csv'), index=False
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
