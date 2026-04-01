def holo_to_fd_line(holo_line, colname_to_idx):
    # Example: t1&t2&EQ(t1.A,t2.A)&IQ(t1.B,t2.B)  -->  <A_idx> -> <B_idx>
    if not holo_line.strip():
        return None
    parts = holo_line.strip().split('&')
    eqs = [p for p in parts if p.startswith('EQ(')]
    iqs = [p for p in parts if p.startswith('IQ(')]
    if not eqs or not iqs:
        return None
    def extract_attr(expr):
        inside = expr[expr.find('(')+1:expr.find(')')]
        left, _ = inside.split(',')
        return left.split('.',1)[1]
    lhs_names = [extract_attr(eq) for eq in eqs]
    rhs_names = [extract_attr(iq) for iq in iqs]
    try:
        lhs = ', '.join(str(colname_to_idx[name]) for name in lhs_names)
        rhs = ', '.join(str(colname_to_idx[name]) for name in rhs_names)
    except KeyError as e:
        print(f"Warning: attribute {e} not found in header for FD conversion.")
        return None
    return f"{lhs} -> {rhs}"

def convert_holo_to_fds(table_dir):
    holo_path = os.path.join(table_dir, 'holo_constraints.txt')
    fds_path = os.path.join(table_dir, 'fds.txt')
    clean_path = os.path.join(table_dir, 'clean.csv')
    if os.path.exists(holo_path) and os.path.exists(clean_path):
        with open(clean_path) as f:
            header = f.readline().strip().split(',')
        colname_to_idx = {name: idx for idx, name in enumerate(header)}
        with open(holo_path) as f:
            lines = f.readlines()
        fd_lines = [holo_to_fd_line(line, colname_to_idx) for line in lines]
        fd_lines = [l for l in fd_lines if l]
        with open(fds_path, 'w') as f:
            for l in fd_lines:
                f.write(l+'\n')
        print(f"Converted {holo_path} to {fds_path} using column indices from {clean_path}")
#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import json
import logging
from evaluate_repair import evaluate_repair

logger = logging.getLogger(__name__)

def find_tables(root, require_fds=True, repaired_name='clean.csv.a2.clean'):
    """Yield table directories needed for run/evaluation.

    If require_fds is True, require dirty.csv, clean.csv, fds.txt (for running Horizon).
    If require_fds is False, require dirty.csv, clean.csv, repaired file (for eval-only mode).
    """
    for dirpath, dirnames, filenames in os.walk(root):
        files = set(filenames)
        if require_fds:
            if {'dirty.csv', 'clean.csv', 'fds.txt'}.issubset(files):
                yield dirpath
        else:
            if {'dirty.csv', 'clean.csv', repaired_name}.issubset(files):
                yield dirpath

def run_horizon(table_dir, java_cp, algo=2):
    import shlex
    dirty = os.path.join(table_dir, 'dirty.csv')
    clean = os.path.join(table_dir, 'clean.csv')
    fds = os.path.join(table_dir, 'fds.txt')
    # Java from java_env conda environment (needs LD_LIBRARY_PATH for libjli.so)
    java_home = '/home/fatemeh/.conda/envs/java_env/lib/jvm'
    java_bin = os.path.join(java_home, 'bin', 'java')
    java_lib = os.path.join(java_home, 'lib')
    cmd_parts = [java_bin, '-cp', java_cp, 'Graph', dirty, clean, fds, str(algo)]
    shell_cmd = f"LD_LIBRARY_PATH={shlex.quote(java_lib)} {' '.join(shlex.quote(p) for p in cmd_parts)}"
    print(f"Running Horizon: {shell_cmd}")
    # Run from the project root so relative classpath (src:commons-collections4-4.3.jar) resolves correctly
    project_root = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(shell_cmd, check=True, cwd=project_root, capture_output=True, text=True, shell=True)
    # Parse runtime from Java stdout ("Elapsed time: <ms>")
    runtime_ms = 0
    for line in proc.stdout.splitlines():
        if line.startswith('Elapsed time:'):
            try:
                runtime_ms = int(line.split(':')[1].strip())
            except ValueError:
                pass
    # Assume output is clean.csv.a2.clean
    repaired = os.path.join(table_dir, 'clean.csv.a2.clean')
    return dirty, clean, repaired, runtime_ms

def main():
    parser = argparse.ArgumentParser(description='Run Horizon and aggregate evaluation results.')
    parser.add_argument('dataset_root', help='Root directory containing tables')
    parser.add_argument('--java_cp', default='src:commons-collections4-4.3.jar', help='Java classpath for Horizon')
    parser.add_argument('--algo', type=int, default=2, help='Horizon algorithm (default: 2)')
    parser.add_argument('--results_file', default='horizon_eval_results.json', help='File to save per-table results')
    parser.add_argument('--skip_horizon', action='store_true', help='Skip running Horizon, only evaluate existing results')
    args = parser.parse_args()

    # Convert holo_constraints.txt to fds.txt for all tables before running Horizon
    for dirpath, dirnames, filenames in os.walk(args.dataset_root):
        files = set(filenames)
        if 'holo_constraints.txt' in files:
            convert_holo_to_fds(dirpath)

    all_results = []
    table_dirs = list(find_tables(
        args.dataset_root,
        require_fds=not args.skip_horizon,
        repaired_name='clean.csv.a2.clean',
    ))
    if args.skip_horizon:
        logger.info("Found %d tables for eval-only mode (dirty/clean/repaired)", len(table_dirs))
    else:
        logger.info("Found %d tables for run+eval mode (dirty/clean/fds)", len(table_dirs))

    for table_dir in table_dirs:
        try:
            dirty = os.path.join(table_dir, 'dirty.csv')
            clean = os.path.join(table_dir, 'clean.csv')
            repaired = os.path.join(table_dir, 'clean.csv.a2.clean')
            
            runtime_ms = 0
            if not args.skip_horizon:
                dirty, clean, repaired, runtime_ms = run_horizon(table_dir, args.java_cp, args.algo)
            
            if not os.path.exists(repaired):
                print(f"Skipping {table_dir}: repaired file not found")
                continue
                
            results = evaluate_repair(dirty, clean, repaired)
            results['table'] = table_dir
            results['runtime_ms'] = runtime_ms
            all_results.append(results)
            print(f"Results for {table_dir}: {results}")
        except Exception as e:
            print(f"Error processing {table_dir}: {e}")

    # Save all results
    with open(args.results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Aggregate
    tp = sum(r['n_truely_corrected_errors'] for r in all_results)
    tpfp = sum(r['n_all_corrected_errors'] for r in all_results)
    tpfn = sum(r['n_all_errors'] for r in all_results)
    total_runtime_ms = sum(r.get('runtime_ms', 0) for r in all_results)
    precision = tp / tpfp if tpfp else -1
    recall = tp / tpfn if tpfn else -1
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else -1
    print("\n=== AGGREGATED RESULTS ===")
    print(f"TP (truly corrected): {tp}")
    print(f"TP+FP (all corrected): {tpfp}")
    print(f"TP+FN (all errors): {tpfn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Total runtime (lake): {total_runtime_ms} ms ({total_runtime_ms/1000:.2f} s)")

    # Save final result in CSV
    import csv
    csv_file = 'horizon_eval_aggregate.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['precision', 'recall', 'f1_score', 'tp', 'tpfp', 'tpfn', 'total_runtime_ms'])
        writer.writerow([precision, recall, f1, tp, tpfp, tpfn, total_runtime_ms])
    print(f"\nAggregated results saved to {csv_file}")

if __name__ == '__main__':
    main()
