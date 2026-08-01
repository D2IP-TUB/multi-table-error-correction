import argparse
from pathlib import Path
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

from Clean import CleanonLocal, CleanonLocalWithnoSmple
from SampleScrubber.cleaner.multiple import AttrRelation
from util import evaluate_cleaning_performance, save_cleaned_data

NON_SAMPLING_THRESHOLD = 10000

# Translated from holoclean constraints:
# t1&t2&EQ(...) → left side attributes, IQ(...) → right side attribute
cleaners = [
    # t1&t2&EQ(t1.name,t2.name)&IQ(t1.surname,t2.surname)
    AttrRelation(["name"], ["surname"], '0'),
    # t1&t2&EQ(t1.team,t2.team)&EQ(t1.manager,t2.manager)&IQ(t1.season,t2.season)
    AttrRelation(["team", "manager"], ["season"], '1'),
    # t1&t2&EQ(t1.season,t2.season)&EQ(t1.manager,t2.manager)&IQ(t1.team,t2.team)
    AttrRelation(["season", "manager"], ["team"], '2'),
    # t1&t2&EQ(t1.surname,t2.surname)&EQ(t1.manager,t2.manager)&IQ(t1.name,t2.name)
    AttrRelation(["surname", "manager"], ["name"], '3'),
    # t1&t2&EQ(t1.surname,t2.surname)&EQ(t1.team,t2.team)&IQ(t1.name,t2.name)
    AttrRelation(["surname", "team"], ["name"], '4'),
    # t1&t2&EQ(t1.name,t2.name)&EQ(t1.team,t2.team)&EQ(t1.season,t2.season)&IQ(t1.manager,t2.manager)
    AttrRelation(["name", "team", "season"], ["manager"], '5'),
    # t1&t2&EQ(t1.surname,t2.surname)&EQ(t1.team,t2.team)&EQ(t1.season,t2.season)&IQ(t1.manager,t2.manager)
    AttrRelation(["surname", "team", "season"], ["manager"], '6'),
]

# Defaults (repo-relative; joined soccer ships as *.csv.zip — unzip or override)
_REPO = Path(__file__).resolve().parents[3]
_SOCCER = _REPO / 'datasets' / 'joinable_tables' / 'soccer' / 'joined' / 'soccer'
file_load = str(_SOCCER / 'dirty.csv')
clean_path = str(_SOCCER / 'clean.csv')
save_path = str(_REPO / 'results' / 'uniclean' / 'soccer' / 'joined') + '/'
table_name = 'soccer_joined_fixed_prov'
single_max = 10000


def count_csv_rows(csv_path):
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        return max(0, sum(1 for _ in f) - 1)


def resolve_clean_fn(mode, row_count):
    """Pick CleanonLocal vs CleanonLocalWithnoSmple.

    mode:
      auto       — no sampling if rows < 10k (matches main.py / main_quintet3)
      sampling   — always CleanonLocal
      nosampling — always CleanonLocalWithnoSmple
    """
    if mode == 'sampling':
        use_sampling = True
    elif mode == 'nosampling':
        use_sampling = False
    else:
        use_sampling = row_count >= NON_SAMPLING_THRESHOLD
    clean_fn = CleanonLocal if use_sampling else CleanonLocalWithnoSmple
    return clean_fn, use_sampling


def parse_args():
    parser = argparse.ArgumentParser(description="Data cleaning for soccer dataset.")
    parser.add_argument('--file_load', type=str, default=file_load, help="Path to the dirty dataset.")
    parser.add_argument('--clean_path', type=str, default=clean_path, help="Path to the clean dataset.")
    parser.add_argument('--save_path', type=str, default=save_path, help="Directory to save cleaned data.")
    parser.add_argument('--table_name', type=str, default=table_name, help="Name of the result table.")
    parser.add_argument('--single_max', type=int, default=single_max, help="Maximum records to process in a single run.")
    parser.add_argument(
        '--mode',
        type=str,
        default='auto',
        choices=['auto', 'sampling', 'nosampling'],
        help=(
            "Cleaning mode: 'auto' uses no-sampling for tables under 10k rows "
            "(default, matches UniClean official scripts); "
            "'sampling' / 'nosampling' force CleanonLocal / CleanonLocalWithnoSmple."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    file_load = args.file_load
    clean_path = args.clean_path
    save_path = args.save_path
    table_name = args.table_name
    single_max = args.single_max

    row_count = count_csv_rows(file_load)
    clean_fn, use_sampling = resolve_clean_fn(args.mode, row_count)

    print(f"Dirty path  : {file_load}")
    print(f"Table name  : {table_name}")
    print(f"Rows        : {row_count}")
    print(f"Mode        : {args.mode}")
    print(f"Cleaning fn : {clean_fn.__name__} ({'sampling' if use_sampling else 'no sampling'})")

    spark = SparkSession.builder \
        .appName("SoccerDataCleaning") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memoryOverhead", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    data = spark.read.csv(file_load, header=True, inferSchema=True)
    if 'index' not in data.columns:
        data = data.withColumn("index", monotonically_increasing_id())
    data.persist()

    start_time = time.perf_counter()
    table_path = os.path.join(save_path, table_name)
    os.makedirs(table_path, exist_ok=True)
    data = clean_fn(spark, cleaners, data, table_path, single_max=single_max)
    elapsed_time = time.perf_counter() - start_time
    print(f"Total cleaning time: {elapsed_time:.4f} seconds")

    save_cleaned_data(data, table_path, table_name)

    evaluate_cleaning_performance(
        clean_path, file_load,
        os.path.join(table_path, f'{table_name}Cleaned.csv'),
        elapsed_time, table_path, table_name,
    )

    spark.stop()


if __name__ == '__main__':
    main()
