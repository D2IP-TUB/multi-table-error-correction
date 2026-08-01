import os
import sys

import numpy as np

from AnalyticsCache.getScore import calculate_all_metrics
from quintet_eval import load_eval_frames


def save_cleaned_data(data, table_path, table_name):
    target_file = os.path.join(table_path, f'{table_name}Cleaned.csv')
    # Pandas preserves quoted CSV fields better than Spark's CSV writer.
    # PySpark 3.1 still references np.bool, removed by NumPy 1.24.
    if "bool" not in np.__dict__:
        np.bool = np.bool_
    data.toPandas().to_csv(target_file, index=False)
    print(f"清洗结果已保存到: {target_file}")


def evaluate_cleaning_performance(
    clean_path,
    dirty_path,
    cleaned_path,
    elapsed_time,
    output_path,
    table_name,
    mse_attributes=None,
    index_col='index',
    missing_token='empty',
    col_alias=None,
):
    if mse_attributes is None:
        mse_attributes = []

    print("测评性能开始：")
    clean_data, dirty_data, cleaned_data, eval_attrs = load_eval_frames(
        clean_path,
        dirty_path,
        cleaned_path,
        missing_token=missing_token,
        index_col=index_col,
        col_alias=col_alias,
    )

    results = calculate_all_metrics(
        clean_data,
        dirty_data,
        cleaned_data,
        eval_attrs,
        output_path,
        table_name,
        index_attribute='index',
        mse_attributes=mse_attributes,
    )

    results_path = os.path.join(output_path, f"{table_name}_total_evaluation.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print_results(results, elapsed_time, clean_data)
        sys.stdout = sys.__stdout__

    print_results(results, elapsed_time, clean_data)
    print(f"测评结束，详细测评日志见：{output_path}")


def print_results(results, elapsed_time, clean_data):
    print("测试结果:")
    print("Detection:")
    print(f"Detection Precision: {results.get('det_precision')}")
    print(f"Detection Recall:    {results.get('det_recall')}")
    print(f"Detection F1:        {results.get('det_f1')}")
    print(f"  det_TP={results.get('det_tp')}  det_FP={results.get('det_fp')}  det_FN={results.get('det_fn')}")
    print("Correction:")
    print(f"Correction Precision: {results.get('cor_precision')}")
    print(f"Correction Recall:    {results.get('cor_recall')}")
    print(f"Correction F1:        {results.get('cor_f1')}")
    print(f"  cor_TP={results.get('cor_tp')}  cor_FP={results.get('cor_fp')}  errors={results.get('errors')}")
    print("Legacy:")
    print(f"Accuracy: {results.get('accuracy')}")
    print(f"Recall: {results.get('recall')}")
    print(f"F1 Score: {results.get('f1_score')}")
    print(f"EDR: {results.get('edr')}")
    print(f"Hybrid Distance: {results.get('hybrid_distance')}")
    print(f"R-EDR: {results.get('r_edr')}")
    print(f"time(s): {elapsed_time}")
    print(f"speed: {100 * float(elapsed_time) / clean_data.shape[0]} seconds/100num")
