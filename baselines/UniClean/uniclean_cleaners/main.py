import argparse
import os
import re
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

from Clean import CleanonLocal, CleanonLocalWithnoSmple
from SampleScrubber.cleaner.multiple import AttrRelation
from util import evaluate_cleaning_performance, save_cleaned_data


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
                        help="Spark driver memory (default: 48g). Set higher for large tables.")
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

    for path, label in [(dirty_path, 'dirty.csv'), (clean_path, 'clean.csv'), (constraints_path, 'holo_constraints.txt')]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    cleaners = parse_holo_constraints(constraints_path)
    if not cleaners:
        raise ValueError(f"No valid constraints found in {constraints_path}")

    print(f"Dataset dir : {dataset_dir}")
    print(f"Table name  : {table_name}")
    print(f"Loaded {len(cleaners)} AttrRelation cleaner(s) from {constraints_path}")
    for c in cleaners:
        print(f"  [{c.name}] {list(c.source)} -> {list(c.target)}")

    driver_memory = args.driver_memory
    # Must set before JVM starts — spark.driver.memory alone doesn't work in local mode
    os.environ['PYSPARK_SUBMIT_ARGS'] = f'--driver-memory {driver_memory} pyspark-shell'

    spark = SparkSession.builder \
        .appName(f"DataCleaning_{table_name}") \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.executor.memory", driver_memory) \
        .config("spark.executor.memoryOverhead", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.maxResultSize", "0") \
        .getOrCreate()

    data = spark.read.csv(dirty_path, header=True, inferSchema=True)
    if 'index' not in data.columns:
        data = data.withColumn("index", monotonically_increasing_id())
    data.persist()

    start_time = time.perf_counter()
    table_path = os.path.join(save_path, table_name)
    os.makedirs(table_path, exist_ok=True)
    data = CleanonLocal(spark, cleaners, data, table_path, single_max=single_max)
    elapsed_time = time.perf_counter() - start_time
    print(f"Total cleaning time: {elapsed_time:.4f} seconds")

    save_cleaned_data(data, table_path, table_name)

    evaluate_cleaning_performance(
        clean_path, dirty_path,
        os.path.join(table_path, f'{table_name}Cleaned.csv'),
        elapsed_time, table_path, table_name
    )

    spark.stop()


if __name__ == '__main__':
    main()
