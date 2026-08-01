"""
Run UniClean on Quintet-3 tables using per-dataset cleaner configs.

Usage:
    python main_quintet3.py --dataset_dir /path/to/Quintet_3/hospital \\
                            --table_name hospital --mode all
    python main_quintet3.py --dataset_dir /path/to/Quintet_3/flights \\
                            --table_name flights --mode fd
"""

import argparse
import os
import time

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, trim, when

from Clean import CleanonLocal, CleanonLocalWithnoSmple
from SampleScrubber.cleaner.multiple import AttrRelation
from SampleScrubber.cleaner.single import Date, DisguisedMissHandler, Number, Outlier, Pattern
from quintet_eval import ensure_index_column
from util import evaluate_cleaning_performance, save_cleaned_data

NON_SAMPLING_THRESHOLD = 10000
FLIGHT_TIME_PATTERN = r"^(0[1-9]|1[0-2]):[0-5][0-9]\s?(AM|PM)$"
VALID_MODES = ('all', 'fd')

# Cleaner configs from main_hospitals / main_flights / main_beers / main_rayyan.
# --mode all  keeps the full list (UniClean-ALL).
# --mode fd   keeps only AttrRelation cleaners (UniClean-FD).
OFFICIAL_CLEANERS = {
    'hospital': [
        AttrRelation(["HospitalName"], ["ProviderNumber"], '1'),
        AttrRelation(["Condition", "MeasureName"], ["HospitalType"], '2'),
        AttrRelation(["HospitalName", "PhoneNumber", "HospitalOwner"], ["State"], '3'),
        AttrRelation(["HospitalName"], ["ZipCode"], '4'),
        AttrRelation(["HospitalName"], ["PhoneNumber"], '5'),
        AttrRelation(["MeasureCode"], ["MeasureName"], '6'),
        AttrRelation(["MeasureCode"], ["Stateavg"], '7'),
        AttrRelation(["ProviderNumber"], ["HospitalName"], '8'),
        AttrRelation(["MeasureCode"], ["Condition"], '9'),
        AttrRelation(["HospitalName"], ["Address1"], '10'),
        AttrRelation(["HospitalName"], ["HospitalOwner"], '11'),
        AttrRelation(["City"], ["CountyName"], '12'),
        AttrRelation(["ZipCode"], ["EmergencyService"], '13'),
        AttrRelation(["HospitalName"], ["City"], '14'),
    ],
    'flights': [
        # Pattern/Date included for UniClean-ALL; dropped under --mode fd.
        # On Quintet they normalize to UniClean time format and hurt F1 vs Quintet GT.
        Pattern("sched_dep_time", FLIGHT_TIME_PATTERN, '0'),
        Pattern("act_dep_time", FLIGHT_TIME_PATTERN, '1'),
        Pattern("sched_arr_time", FLIGHT_TIME_PATTERN, '2'),
        Date("sched_dep_time", "%I:%M %p", '5'),
        Date("act_dep_time", "%I:%M %p", '6'),
        Date("sched_arr_time", "%I:%M %p", '7'),
        Date("act_arr_time", "%I:%M %p", '8'),
        AttrRelation(["flight"], ["sched_dep_time"], '3'),
        AttrRelation(["flight"], ["act_dep_time"], '4'),
        AttrRelation(["flight"], ["sched_arr_time"], '9'),
        AttrRelation(["flight"], ["act_arr_time"], '10'),
    ],
    'beers': [
        Number("ounces", name="Number_ounces"),
        Number("abv", name="Outlier_abv"),
        # Quintet column name: brewery-name
        AttrRelation(["brewery_id"], ["brewery-name"], '0'),
        AttrRelation(["brewery_id"], ["city"], '1'),
        AttrRelation(["brewery_id"], ["state"], '2'),
    ],
    'rayyan': [
        Outlier("article_title", [], '3'),
        Outlier("journal_title", [], '4'),
        Outlier("author_list", [], '8'),
        # Columns removed from Quintet
        # Date("journal_issn", "%y-%b", '5',
            #  valid_date_pattern=r'^[A-Za-z]{3}-\d{2}$|^\d{2}-[A-Za-z]{3}$'),
        # Date("article_pagination", "%b-%y", '9',
            #  valid_date_pattern=r'^(?:\d{1}-\d{2}|\d{2}-\d{1})$'),
        # Date("article_jcreated_at", "%-m/%-d/%y", currenFormat="%y/%m/%d", name='10'),
        DisguisedMissHandler("article_jvolumn", "-1", "6"),
        DisguisedMissHandler("article_jissue", "-1", "7"),
        # Quintet column name: jounral_abbreviation
        AttrRelation(["jounral_abbreviation"], ["journal_title"], '0'),
        AttrRelation(["jounral_abbreviation"], ["journal_issn"], '1'),
        AttrRelation(["journal_title"], ["journal_issn"], '2'),
    ],
}

# movies_1: Baran Table-2 rules (no dedicated UniClean script).
MOVIES_CLEANERS = [
    AttrRelation(["Id"], ["Cast"], '1'),
    AttrRelation(["Id"], ["Actors"], '2'),
    AttrRelation(["Id"], ["Name"], '3'),
    AttrRelation(["Id"], ["RatingCount"], '4'),
    AttrRelation(["Id"], ["ReviewCount"], '5'),
    AttrRelation(["Id"], ["FilmingLocations"], '6'),
    AttrRelation(["Id"], ["Language"], '7'),
    AttrRelation(["Id"], ["Country"], '8'),
    AttrRelation(["Id"], ["Duration"], '9'),
    AttrRelation(["Id"], ["Year"], '10'),
    AttrRelation(["Id"], ["RatingValue"], '11'),
    Pattern("Id", r'^tt\d+$', 'fmt_id'),
    Pattern("Year", r'^\d{4}$', 'fmt_year'),
    Number("RatingValue", name='fmt_ratingvalue'),
    Pattern("RatingCount", r'^\d+$', 'fmt_ratingcount'),
    Pattern("Duration", r'^\d+\s*min$', 'fmt_duration'),
]

DATASET_CONFIG = {
    'hospital': {
        'mse_attributes': ['Score'],
        'infer_schema': False,
        'missing_token': 'empty',
    },
    'flights': {
        'missing_token': 'empty',
    },
    'beers': {
        'mse_attributes': ['ibu', 'abv'],
        'missing_token': 'empty',
    },
    'rayyan': {
        'missing_token': 'empty',
        'infer_schema': False,
        'load_with_pandas': True,
    },
    'movies_1': {
        'missing_token': 'empty',
    },
    'movies': {
        'missing_token': 'empty',
    },
}


def _cleaner_columns(cleaner):
    if hasattr(cleaner, 'source') and hasattr(cleaner, 'target'):
        src, tgt = cleaner.source, cleaner.target
        if isinstance(src, str):
            return [src] if src == tgt else [src, tgt]
        return list(src) + list(tgt)
    if hasattr(cleaner, 'domain'):
        dom = cleaner.domain
        if isinstance(dom, str):
            return [dom]
        return list(dom)
    return []


def _describe_cleaner(cleaner):
    if hasattr(cleaner, 'source') and hasattr(cleaner, 'target'):
        return f"AttrRelation {list(cleaner.source)} -> {list(cleaner.target)}"
    if hasattr(cleaner, 'domain'):
        return f"{type(cleaner).__name__}({cleaner.domain})"
    return repr(cleaner)


def filter_cleaners_for_columns(cleaners, columns):
    """Drop cleaners whose columns are missing from the dataset."""
    available = set(columns)
    kept, skipped = [], []
    for cleaner in cleaners:
        cols = _cleaner_columns(cleaner)
        if cols and all(col in available for col in cols):
            kept.append(cleaner)
        else:
            skipped.append((cleaner, cols))
    return kept, skipped


def filter_cleaners_for_mode(cleaners, mode):
    """UniClean-ALL keeps all cleaners; UniClean-FD keeps AttrRelation only."""
    if mode == 'all':
        return list(cleaners), []
    if mode != 'fd':
        raise ValueError(f"Unknown mode '{mode}'. Expected one of {VALID_MODES}.")
    kept, dropped = [], []
    for cleaner in cleaners:
        if isinstance(cleaner, AttrRelation):
            kept.append(cleaner)
        else:
            dropped.append(cleaner)
    return kept, dropped


def get_cleaners(table_name, mode='all'):
    key = table_name.strip().lower()
    if key in ('movies_1', 'movies'):
        cleaners = list(MOVIES_CLEANERS)
    elif key not in OFFICIAL_CLEANERS:
        raise ValueError(
            f"No cleaner configuration for table '{table_name}'. "
            f"Known tables: {sorted(OFFICIAL_CLEANERS)} + movies_1"
        )
    else:
        cleaners = list(OFFICIAL_CLEANERS[key])
    return filter_cleaners_for_mode(cleaners, mode)


def count_csv_rows(csv_path):
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        return max(0, sum(1 for _ in f) - 1)


def _apply_missing_token_spark(data, missing_token):
    """Fill null/blank cells with missing_token."""
    fill_cols = [c for c in data.columns if c != 'index']
    if not fill_cols:
        return data
    data = data.fillna(missing_token, subset=fill_cols)
    for c in fill_cols:
        data = data.withColumn(
            c,
            when(col(c).isNull() | (trim(col(c).cast('string')) == ''), lit(missing_token))
            .otherwise(col(c)),
        )
    return data


def load_dirty_table(spark, dirty_path, *, infer_schema=True, missing_token=None, use_pandas=False):
    """Load dirty.csv into Spark."""
    if use_pandas:
        pdf = pd.read_csv(dirty_path, dtype=str, keep_default_na=False)
        if missing_token:
            pdf = pdf.replace('', missing_token)
            pdf = pdf.fillna(missing_token)
        return spark.createDataFrame(pdf)

    data = spark.read.csv(dirty_path, header=True, inferSchema=infer_schema)
    if missing_token:
        data = _apply_missing_token_spark(data, missing_token)
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean a Quintet-3 table."
    )
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help="Path to the table directory (must contain dirty.csv, clean.csv).")
    parser.add_argument('--table_name', type=str, default=None,
                        help="Table name used for rule lookup. Defaults to directory name.")
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=list(VALID_MODES),
        help=(
            "Cleaner mode: 'all' = UniClean-ALL (Pattern/Date/Number/Outlier + AttrRelation); "
            "'fd' = UniClean-FD (AttrRelation only). Results are written under result/<mode>/."
        ),
    )
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
    dirty_path = os.path.join(dataset_dir, 'dirty.csv')
    clean_path = os.path.join(dataset_dir, 'clean.csv')
    mode = args.mode
    # Keep ALL and FD outputs separate so both variants can coexist.
    save_path = os.path.join(dataset_dir, 'result', mode)
    table_name = args.table_name or os.path.basename(os.path.normpath(dataset_dir))
    single_max = args.single_max
    table_key = table_name.strip().lower()
    cfg = DATASET_CONFIG.get(table_key, {})

    for path in (dirty_path, clean_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    for path in (dirty_path, clean_path):
        ensure_index_column(path)

    row_count = count_csv_rows(dirty_path)
    use_sampling = row_count >= NON_SAMPLING_THRESHOLD
    clean_fn = CleanonLocal if use_sampling else CleanonLocalWithnoSmple

    driver_memory = args.driver_memory
    os.environ['PYSPARK_SUBMIT_ARGS'] = f'--driver-memory {driver_memory} pyspark-shell'

    builder = (
        SparkSession.builder
        .appName(f"DataCleaning_{table_name}_{mode}")
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", driver_memory)
        .config("spark.executor.memoryOverhead", "8g")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.maxResultSize", "0")
    )
    if args.spark_master:
        builder = builder.master(args.spark_master)
    spark = builder.getOrCreate()

    infer_schema = cfg.get('infer_schema', True)
    missing_token = cfg.get('missing_token')
    data = load_dirty_table(
        spark,
        dirty_path,
        infer_schema=infer_schema,
        missing_token=missing_token,
        use_pandas=cfg.get('load_with_pandas', False),
    )
    if 'index' not in data.columns:
        raise RuntimeError(f"'index' column missing from {dirty_path}")

    mode_cleaners, mode_dropped = get_cleaners(table_name, mode=mode)
    cleaners, skipped = filter_cleaners_for_columns(mode_cleaners, data.columns)

    print(f"Dataset dir : {dataset_dir}")
    print(f"Table name  : {table_name}")
    print(f"Mode        : {mode} ({'UniClean-ALL' if mode == 'all' else 'UniClean-FD'})")
    print(f"Rows        : {row_count}")
    print(f"Cleaning fn : {clean_fn.__name__} ({'sampling' if use_sampling else 'no sampling'})")
    print(f"Active cleaners ({len(cleaners)}):")
    for cleaner in cleaners:
        print(f"  [{cleaner.name}] {_describe_cleaner(cleaner)}")
    if mode_dropped:
        print(f"Dropped by --mode {mode} ({len(mode_dropped)}):")
        for cleaner in mode_dropped:
            print(f"  [{cleaner.name}] {_describe_cleaner(cleaner)}")
    if skipped:
        print(f"Skipped cleaners ({len(skipped)}):")
        for cleaner, cols in skipped:
            print(f"  [{cleaner.name}] needs {cols}")

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
        mse_attributes=cfg.get('mse_attributes', []),
        missing_token=cfg.get('missing_token', 'empty'),
        col_alias=cfg.get('col_alias'),
        index_col=cfg.get('index_col', 'index'),
    )

    spark.stop()


if __name__ == '__main__':
    main()
