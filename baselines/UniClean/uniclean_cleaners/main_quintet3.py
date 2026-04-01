"""
main_quintet3.py — Like main.py but uses hard-coded FD rules from Table 2
of the paper instead of reading holo_constraints.txt.

Supported tables: hospital, flights, beers, rayyan, movies_1

Format constraints from the paper (digits, letters, not-null, date) are
included where columns exist using Pattern, Number, DisguisedMissHandler,
and Date cleaners. Constraints referencing absent columns are skipped.

Usage:
    python main_quintet3.py --dataset_dir /path/to/Quintet_3/hospital \\
                            --table_name hospital
"""

import argparse
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

from Clean import CleanonLocal
from SampleScrubber.cleaner.multiple import AttrRelation
from SampleScrubber.cleaner.single import Date, DisguisedMissHandler, Number, Pattern
from util import evaluate_cleaning_performance, save_cleaned_data


# ---------------------------------------------------------------------------
# Hard-coded rules from Table 2 of the paper.
# ---------------------------------------------------------------------------
# Hospital  — city→zip, city→county, zip→city, zip→state, zip→county,
#            county→state; index (digits), provider number (digits),
#            zip (5 digits), state (2 letters), phone (digits)
# Flights   — flight→act_dep_time, flight→act_arr_time,
#             flight→sched_dep_time, flight→sched_arr_time
# Beers     — brewery_id→brewery-name, brewery_id→city, brewery_id→state;
#            brewery_id (digits), state (2 letters)
# Rayyan    — jounral_abbreviation→journal_title (jounral_abbreviation is a typo
#             in the actual data); journal_abbreviation→journal_issn and
#             journal_issn→journal_title skipped (journal_issn column absent);
#             not-null on: author_list, jounral_abbreviation, article_title,
#             article_language, journal_title, article_jissue, article_jvolumn;
#             article_pagination/journal_issn/journal_created_at absent in data
# movies_1  — FDs from holo_constraints.txt (Id→*) +
#             format constraints from Table 2:
#               id ('tt' + digits), year (4 digits), rating value (float),
#               rating count (digits), duration (digits + 'min')
# ---------------------------------------------------------------------------

QUINTET3_CLEANERS = {
    'hospital': [
        # FD rules from Table 2
        AttrRelation(["City"],       ["ZipCode"],    '1'),
        AttrRelation(["City"],       ["CountyName"], '2'),
        AttrRelation(["ZipCode"],    ["City"],       '3'),
        AttrRelation(["ZipCode"],    ["State"],      '4'),
        AttrRelation(["ZipCode"],    ["CountyName"], '5'),
        AttrRelation(["CountyName"], ["State"],      '6'),
        # Format constraints from Table 2
        Pattern("index",          r'^\d+$',       'fmt_index'),           # digits
        Pattern("ProviderNumber", r'^\d+$',       'fmt_providernumber'),  # digits
        Pattern("ZipCode",        r'^\d{5}$',     'fmt_zip'),             # 5 digits
        Pattern("State",          r'^[A-Za-z]{2}$', 'fmt_state'),         # 2 letters
        Pattern("PhoneNumber",    r'^\d+$',       'fmt_phone'),           # digits
    ],
    'flights': [
        AttrRelation(["flight"], ["act_dep_time"],   '1'),
        AttrRelation(["flight"], ["act_arr_time"],   '2'),
        AttrRelation(["flight"], ["sched_dep_time"], '3'),
        AttrRelation(["flight"], ["sched_arr_time"], '4'),
    ],
    'beers': [
        # FD rules from Table 2
        AttrRelation(["brewery_id"], ["brewery-name"], '1'),
        AttrRelation(["brewery_id"], ["city"],         '2'),
        AttrRelation(["brewery_id"], ["state"],        '3'),
        # Format constraints from Table 2
        Pattern("brewery_id", r'^\d+$',         'fmt_brewery_id'),  # digits
        Pattern("state",      r'^[A-Za-z]{2}$', 'fmt_state'),       # 2 letters
    ],
    'rayyan': [
        # FD rules from Table 2 (only those with columns present in the data)
        AttrRelation(["jounral_abbreviation"], ["journal_title"], '1'),
        # journal_abbreviation→journal_issn: skipped (journal_issn column absent)
        # journal_issn→journal_title: skipped (journal_issn column absent)
        # Not-null constraints from Table 2
        DisguisedMissHandler("author_list",         name='nn_author_list'),
        DisguisedMissHandler("jounral_abbreviation", name='nn_journal_abbr'),
        DisguisedMissHandler("article_title",       name='nn_article_title'),
        DisguisedMissHandler("article_language",    name='nn_article_language'),
        DisguisedMissHandler("journal_title",       name='nn_journal_title'),
        DisguisedMissHandler("article_jissue",      name='nn_article_jissue'),
        DisguisedMissHandler("article_jvolumn",     name='nn_article_jvolumn'),
        # article_pagination not-null: skipped (column absent in data)
        # journal_issn not-null: skipped (column absent in data)
        # journal created at (date): skipped (column absent in data)
    ],
    # movies_1: FDs (Id→*) + paper format constraints
    'movies_1': [
        # FD rules (Id is a unique key)
        AttrRelation(["Id"], ["Cast"],             '1'),
        AttrRelation(["Id"], ["Actors"],           '2'),
        AttrRelation(["Id"], ["Name"],             '3'),
        AttrRelation(["Id"], ["RatingCount"],      '4'),
        AttrRelation(["Id"], ["ReviewCount"],      '5'),
        AttrRelation(["Id"], ["FilmingLocations"], '6'),
        AttrRelation(["Id"], ["Language"],         '7'),
        AttrRelation(["Id"], ["Country"],          '8'),
        AttrRelation(["Id"], ["Duration"],         '9'),
        AttrRelation(["Id"], ["Year"],             '10'),
        AttrRelation(["Id"], ["RatingValue"],      '11'),
        # Format constraints from Table 2 of the paper
        Pattern("Id",          r'^tt\d+$',   'fmt_id'),        # 'tt' + digits
        Pattern("Year",        r'^\d{4}$',   'fmt_year'),      # 4 digits
        Number("RatingValue",                name='fmt_ratingvalue'),  # float
        Pattern("RatingCount", r'^\d+$',     'fmt_ratingcount'),  # digits
        Pattern("Duration",    r'^\d+\s*min$', 'fmt_duration'),  # digits + 'min'
    ],
}


def get_cleaners(table_name):
    """Return hard-coded AttrRelation cleaners for *table_name* (case-insensitive)."""
    key = table_name.strip().lower()
    if key not in QUINTET3_CLEANERS:
        raise ValueError(
            f"No hard-coded rules for table '{table_name}'. "
            f"Known tables: {sorted(QUINTET3_CLEANERS)}"
        )
    return QUINTET3_CLEANERS[key]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean a Quintet-3 table using paper-defined FD rules."
    )
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help="Path to the table directory (must contain dirty.csv, clean.csv).")
    parser.add_argument('--table_name', type=str, default=None,
                        help="Table name used for rule lookup. Defaults to directory name.")
    parser.add_argument('--single_max', type=int, default=10000,
                        help="Maximum records to process in a single run.")
    parser.add_argument('--driver_memory', type=str, default='48g',
                        help="Spark driver memory (default: 48g).")
    parser.add_argument('--spark_master', type=str, default=None,
                        help="Spark master URL, e.g. 'local[16]'.")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = args.dataset_dir
    dirty_path  = os.path.join(dataset_dir, 'dirty.csv')
    clean_path  = os.path.join(dataset_dir, 'clean.csv')
    save_path   = os.path.join(dataset_dir, 'result')
    table_name  = args.table_name or os.path.basename(os.path.normpath(dataset_dir))
    single_max  = args.single_max

    for path, label in [(dirty_path, 'dirty.csv'), (clean_path, 'clean.csv')]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    cleaners = get_cleaners(table_name)

    print(f"Dataset dir : {dataset_dir}")
    print(f"Table name  : {table_name}")
    print(f"Using {len(cleaners)} hard-coded FD rule(s) from paper (Table 2)")
    for c in cleaners:
        print(f"  [{c.name}] {list(c.source)} -> {list(c.target)}")

    driver_memory = args.driver_memory
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
