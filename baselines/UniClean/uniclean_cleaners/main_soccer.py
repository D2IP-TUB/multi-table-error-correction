import argparse
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

from AnalyticsCache.getScore import calculate_accuracy_and_recall, calculate_all_metrics
from AnalyticsCache.insert_null import inject_missing_values
from Clean import CleanonLocal
# from Clean_test import CleanRamdonCL
from SampleScrubber.cleaner.multiple import AttrRelation
from util import evaluate_cleaning_performance, save_cleaned_data

# 清洗规则定义
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

# 默认参数
file_load = '/home/fatemeh/Uniclean-bench-Result/datasets_and_rules/joined_soccer/soccer/dirty.csv'
clean_path = '/home/fatemeh/Uniclean-bench-Result/datasets_and_rules/joined_soccer/soccer/clean.csv'
save_path = '/home/fatemeh/Uniclean-bench-Result/datasets_and_rules/joined_soccer/soccer/result/'
table_name = 'soccer_joined_fixed_prov'
single_max = 10000

# 添加动态参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Data cleaning for soccer dataset.")
    parser.add_argument('--file_load', type=str, default=file_load, help="Path to the dirty dataset.")
    parser.add_argument('--clean_path', type=str, default=clean_path, help="Path to the clean dataset.")
    parser.add_argument('--save_path', type=str, default=save_path, help="Directory to save cleaned data.")
    parser.add_argument('--table_name', type=str, default=table_name, help="Name of the result table.")
    parser.add_argument('--single_max', type=int, default=single_max, help="Maximum records to process in a single run.")
    return parser.parse_args()

def main():
    args = parse_args()

    file_load = args.file_load
    clean_path = args.clean_path
    save_path = args.save_path
    table_name = args.table_name
    single_max = args.single_max

    # 初始化 SparkSession
    spark = SparkSession.builder \
        .appName("SoccerDataCleaning") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memoryOverhead", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    # 读取数据并添加索引列
    data = spark.read.csv(file_load, header=True, inferSchema=True)
    if 'index' not in data.columns:
        data = data.withColumn("index", monotonically_increasing_id())
    data.persist()

    # 数据清洗及时间记录
    start_time = time.perf_counter()
    table_path = os.path.join(save_path, table_name)
    os.makedirs(table_path, exist_ok=True)
    data = CleanonLocal(spark, cleaners, data, table_path, single_max=single_max)
    elapsed_time = time.perf_counter() - start_time
    print(f"清洗总执行时间: {elapsed_time:.4f} 秒")

    # 保存清洗后的数据
    save_cleaned_data(data, table_path, table_name)

    # 性能评估
    evaluate_cleaning_performance(clean_path, file_load, os.path.join(table_path, f'{table_name}Cleaned.csv'),
                                  elapsed_time, table_path, table_name)

    spark.stop()


if __name__ == '__main__':
    main()