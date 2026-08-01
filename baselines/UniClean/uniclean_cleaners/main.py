import argparse
import os
import re
import time

from pyspark.sql import SparkSession

from Clean import CleanonLocal, CleanonLocalWithnoSmple
from SampleScrubber.cleaner.multiple import AttrRelation
from quintet_eval import ensure_index_column
from util import evaluate_cleaning_performance, save_cleaned_data

NON_SAMPLING_THRESHOLD = 10000


def count_csv_rows(csv_path):
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        return max(0, sum(1 for _ in f) - 1)


def parse_holo_constraints(constraints_path):
    """Parse a holo_constraints.txt file and return a list of AttrRelation cleaners.

    Each line has the format:
        t1&t2&EQ(t1.col1,t2.col1)&...&EQ(t1.colN,t2.colN)&IQ(t1.target,t2.target)

    EQ columns become the source (LHS of the FD).
    IQ column becomes the target (RHS of the FD).
    """
    cleaners = []
    eq_pattern = re.compile(r'EQ\(t1\.(\w+),t2\.\w+\)')
    iq_pattern = re.compile(r'IQ\(t1\.(\w+),t2\.\w+\)')

    with open(constraints_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            source = eq_pattern.findall(line)
            target = iq_pattern.findall(line)

            if not target:
                print(f"Warning: skipping line {idx + 1}, no IQ (target) found: {line}")
                continue

            cleaners.append(AttrRelation(source, target, str(idx)))

    return cleaners


def parse_args():
    parser = argparse.ArgumentParser(description="Unified data cleaning using holo_constraints.")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help="Path to the dataset directory (must contain dirty.csv, clean.csv, and holo_constraints.txt).")
    parser.add_argument('--table_name', type=str, default=None,
                        help="Name for the result table. Defaults to the directory name.")
    parser.add_argument('--single_max', type=int, default=10000,
                        help="Maximum records to process in a single run.")
    parser.add_argument('--driver_memory', type=str, default='200g',
                        help="Spark driver memory (default: 200g). Set higher for large tables.")
    parser.add_argument('--spark_master', type=str, default=None,
                        help="Spark master URL, e.g. 'local[16]'.")
    parser.add_argument('--missing_token', type=str, default='empty',
                        help="Missing-value token (default: empty).")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = args.dataset_dir
    dirty_path = os.path.join(dataset_dir, 'dirty.csv')
    clean_path = os.path.join(dataset_dir, 'clean.csv')
    constraints_path = os.path.join(dataset_dir, 'holo_constraints.txt')
    save_path = os.path.join(dataset_dir, 'result')
    table_name = args.table_name or os.path.basename(os.path.normpath(dataset_dir))
    single_max = args.single_max
    missing_token = args.missing_token

    for path, label in [(dirty_path, 'dirty.csv'), (clean_path, 'clean.csv'), (constraints_path, 'holo_constraints.txt')]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    for path in (dirty_path, clean_path):
        ensure_index_column(path)

    cleaners = parse_holo_constraints(constraints_path)
    if not cleaners:
        raise ValueError(f"No valid constraints found in {constraints_path}")

    row_count = count_csv_rows(dirty_path)
    use_sampling = row_count >= NON_SAMPLING_THRESHOLD
    clean_fn = CleanonLocal if use_sampling else CleanonLocalWithnoSmple

    print(f"Dataset dir : {dataset_dir}")
    print(f"Table name  : {table_name}")
    print(f"Loaded {len(cleaners)} AttrRelation cleaner(s) from {constraints_path}")
    for c in cleaners:
        print(f"  [{c.name}] {list(c.source)} -> {list(c.target)}")
    print(f"Rows        : {row_count}")
    print(f"Cleaning fn : {clean_fn.__name__} ({'sampling' if use_sampling else 'no sampling'})")

    driver_memory = args.driver_memory
    # Must set before JVM starts — spark.driver.memory alone doesn't work in local mode
    os.environ['PYSPARK_SUBMIT_ARGS'] = f'--driver-memory {driver_memory} pyspark-shell'

    builder = (
        SparkSession.builder
        .appName(f"DataCleaning_{table_name}")
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", driver_memory)
        .config("spark.executor.memoryOverhead", "8g")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.maxResultSize", "0")
    )
    if args.spark_master:
        builder = builder.master(args.spark_master)
    spark = builder.getOrCreate()

    data = spark.read.csv(dirty_path, header=True, inferSchema=True)
    if 'index' not in data.columns:
        raise RuntimeError(f"'index' column missing from {dirty_path}")
    data.persist()

    start_time = time.perf_counter()
    table_path = os.path.join(save_path, table_name)
    os.makedirs(table_path, exist_ok=True)
    data = clean_fn(spark, cleaners, data, table_path, single_max=single_max)
    elapsed_time = time.perf_counter() - start_time
    print(f"Total cleaning time: {elapsed_time:.4f} seconds")

    save_cleaned_data(data, table_path, table_name)

    evaluate_cleaning_performance(
        clean_path, dirty_path,
        os.path.join(table_path, f'{table_name}Cleaned.csv'),
        elapsed_time, table_path, table_name,
        missing_token=missing_token,
    )

    spark.stop()


if __name__ == '__main__':
    main()
