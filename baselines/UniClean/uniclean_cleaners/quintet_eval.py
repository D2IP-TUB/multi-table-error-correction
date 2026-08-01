"""Quintet evaluation helpers."""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from AnalyticsCache.getScore import calF1, normalize_for_cmp
from AnalyticsCache.insert_null import inject_missing_values

INDEX_COL = "index"
DEFAULT_MISSING_TOKEN = "empty"

_MISSING_TOKENS_BASE = {"nan", "NaN", "NULL", "null", "None", "NONE", ""}


def ensure_index_column(csv_path: str) -> bool:
    """Add a 0-based index column if missing. Returns True if added."""
    with open(csv_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        if INDEX_COL in header:
            return False
        rows = list(reader)

    header.insert(0, INDEX_COL)
    for i, row in enumerate(rows):
        row.insert(0, str(i))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return True


def _is_intish(s: str) -> bool:
    return bool(s) and s.lstrip("-").isdigit()


def _strip_dot_zero(v: str) -> str:
    if isinstance(v, str) and v.endswith(".0") and _is_intish(v[:-2]):
        return v[:-2]
    return v


def normalize_cleaned_against_clean(
    cleaned_path: str,
    clean_path: str,
    missing_token: str = DEFAULT_MISSING_TOKEN,
) -> None:
    """Align cleaned.csv columns to clean.csv."""
    clean = pd.read_csv(clean_path, dtype=str, keep_default_na=False)
    cleaned = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False)
    for col in cleaned.columns:
        if col not in clean.columns:
            continue
        clean_vals = set(clean[col].unique())
        if not any(v.endswith(".0") and _is_intish(v[:-2]) for v in clean_vals):
            cleaned[col] = cleaned[col].apply(_strip_dot_zero)
        if missing_token and missing_token not in clean_vals:
            cleaned[col] = cleaned[col].replace(missing_token, "")
    cleaned.to_csv(cleaned_path, index=False)


def prepare_cleaned_csv(
    cleaned_path: str,
    clean_path: str,
    missing_token: str = DEFAULT_MISSING_TOKEN,
) -> None:
    """Normalize cleaned output before evaluation."""
    inject_missing_values(
        csv_file=cleaned_path,
        output_file=cleaned_path,
        attributes_error_ratio=None,
        missing_value_in_ori_data="NULL",
        missing_value_representation=missing_token,
    )
    normalize_cleaned_against_clean(cleaned_path, clean_path, missing_token=missing_token)


def unify_missing_tokens(df: pd.DataFrame, missing_token: str) -> None:
    """Map missing-value tokens to missing_token in place."""
    tokens = _MISSING_TOKENS_BASE | {missing_token}
    df.fillna(missing_token, inplace=True)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).where(
                ~df[col].astype(str).isin(tokens), missing_token
            )


def _ensure_index(dfs: Iterable[pd.DataFrame]) -> None:
    for df in dfs:
        if INDEX_COL not in df.columns:
            df.insert(0, INDEX_COL, range(len(df)))


def load_eval_frames(
    clean_path: str,
    dirty_path: str,
    cleaned_path: str,
    *,
    missing_token: str = DEFAULT_MISSING_TOKEN,
    index_col: str = INDEX_COL,
    col_alias: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """Load clean/dirty/cleaned frames for evaluation."""
    prepare_cleaned_csv(cleaned_path, clean_path, missing_token=missing_token)

    clean_df = pd.read_csv(clean_path, dtype=str, keep_default_na=False)
    dirty_df = pd.read_csv(dirty_path, dtype=str, keep_default_na=False)
    cleaned_df = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False)

    if index_col != INDEX_COL:
        for df in (clean_df, dirty_df, cleaned_df):
            if index_col in df.columns and INDEX_COL not in df.columns:
                df.rename(columns={index_col: INDEX_COL}, inplace=True)

    if col_alias:
        for df in (clean_df, dirty_df, cleaned_df):
            rename_map = {
                old: new
                for old, new in col_alias.items()
                if old in df.columns and new not in df.columns
            }
            if rename_map:
                df.rename(columns=rename_map, inplace=True)

    _ensure_index((clean_df, dirty_df, cleaned_df))

    for df in (clean_df, dirty_df, cleaned_df):
        unify_missing_tokens(df, missing_token)

    common_cols = (
        set(clean_df.columns) & set(dirty_df.columns) & set(cleaned_df.columns)
    ) - {INDEX_COL}
    eval_attrs = [c for c in clean_df.columns if c in common_cols]
    if not eval_attrs:
        eval_attrs = [c for c in clean_df.columns if c != INDEX_COL]

    return clean_df, dirty_df, cleaned_df, eval_attrs


def compute_detection_correction_counts(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    attributes: list,
    index_col: str = INDEX_COL,
) -> dict:
    """Count detection and correction TP/FP/FN by cell."""
    clean = clean_df.set_index(index_col, drop=False)
    dirty = dirty_df.set_index(index_col, drop=False)
    cleaned = cleaned_df.set_index(index_col, drop=False)

    det_tp = det_fp = det_fn = 0
    cor_tp = cor_fp = 0
    errors = 0

    for attr in attributes:
        if attr not in clean.columns or attr not in dirty.columns or attr not in cleaned.columns:
            continue

        clean_v = clean[attr].apply(normalize_for_cmp)
        dirty_v = dirty[attr].apply(normalize_for_cmp)
        cleaned_v = cleaned[attr].apply(normalize_for_cmp)

        common = clean_v.index.intersection(dirty_v.index).intersection(cleaned_v.index)
        clean_v = clean_v.loc[common]
        dirty_v = dirty_v.loc[common]
        cleaned_v = cleaned_v.loc[common]

        actual_error = dirty_v != clean_v
        changed = dirty_v != cleaned_v

        n_errors = int(actual_error.sum())
        errors += n_errors

        det_tp += int((actual_error & changed).sum())
        det_fp += int((~actual_error & changed).sum())
        det_fn += int((actual_error & ~changed).sum())

        cor_attempted = actual_error & changed
        cor_tp += int((cor_attempted & (cleaned_v == clean_v)).sum())
        cor_fp += int((cor_attempted & (cleaned_v != clean_v)).sum())

    return {
        "errors": errors,
        "det_tp": det_tp,
        "det_fp": det_fp,
        "det_fn": det_fn,
        "cor_tp": cor_tp,
        "cor_fp": cor_fp,
    }


def metrics_from_counts(counts: dict) -> dict:
    """Compute P/R/F1 from raw counts."""
    det_tp, det_fp, det_fn = counts["det_tp"], counts["det_fp"], counts["det_fn"]
    cor_tp, cor_fp = counts["cor_tp"], counts["cor_fp"]
    errors = counts["errors"]

    det_precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0.0
    det_recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0.0
    det_f1 = calF1(det_precision, det_recall)

    cor_precision = cor_tp / (cor_tp + cor_fp) if (cor_tp + cor_fp) > 0 else 0.0
    cor_recall = cor_tp / errors if errors > 0 else 0.0
    cor_f1 = calF1(cor_precision, cor_recall)

    return {
        "errors": errors,
        "det_tp": det_tp,
        "det_fp": det_fp,
        "det_fn": det_fn,
        "det_precision": det_precision,
        "det_recall": det_recall,
        "det_f1": det_f1,
        "cor_tp": cor_tp,
        "cor_fp": cor_fp,
        "cor_precision": cor_precision,
        "cor_recall": cor_recall,
        "cor_f1": cor_f1,
        "accuracy": cor_precision,
        "recall": cor_recall,
        "f1_score": cor_f1,
    }


def count_ground_truth_errors(
    clean_path: str,
    dirty_path: str,
    missing_token: str = DEFAULT_MISSING_TOKEN,
) -> int:
    """Count ground-truth error cells."""
    clean_df = pd.read_csv(clean_path, dtype=str, keep_default_na=False)
    dirty_df = pd.read_csv(dirty_path, dtype=str, keep_default_na=False)
    _ensure_index((clean_df, dirty_df))
    unify_missing_tokens(clean_df, missing_token)
    unify_missing_tokens(dirty_df, missing_token)

    common = (set(clean_df.columns) & set(dirty_df.columns)) - {INDEX_COL}
    attrs = [c for c in clean_df.columns if c in common]

    clean = clean_df.set_index(INDEX_COL, drop=False)
    dirty = dirty_df.set_index(INDEX_COL, drop=False)
    errors = 0
    for attr in attrs:
        errors += int(
            (dirty[attr].apply(normalize_for_cmp) != clean[attr].apply(normalize_for_cmp)).sum()
        )
    return errors


def evaluate_quintet_table(
    clean_path: str,
    dirty_path: str,
    cleaned_path: str,
    *,
    missing_token: str = DEFAULT_MISSING_TOKEN,
    col_alias: Optional[Dict[str, str]] = None,
) -> dict:
    """Evaluate one Quintet table."""
    if not os.path.isfile(cleaned_path):
        raise FileNotFoundError(cleaned_path)

    clean_df, dirty_df, cleaned_df, eval_attrs = load_eval_frames(
        clean_path,
        dirty_path,
        cleaned_path,
        missing_token=missing_token,
        col_alias=col_alias,
    )
    counts = compute_detection_correction_counts(
        clean_df, dirty_df, cleaned_df, eval_attrs
    )
    return metrics_from_counts(counts)


def skipped_table_metrics(errors: int) -> dict:
    """Metrics for tables with no usable cleaned output.

    All ground-truth error cells count toward the lake recall denominators:
    detection FN (missed) and correction errors (uncorrected).
    """
    return {
        "errors": errors,
        "det_tp": 0,
        "det_fp": 0,
        "det_fn": errors,
        "cor_tp": 0,
        "cor_fp": 0,
    }


def metrics_result_row(tname: str, status: str, metrics: dict) -> dict:
    return {
        'table': tname,
        'status': status,
        'errors': metrics.get('errors', 0),
        'det_tp': metrics.get('det_tp', 0),
        'det_fp': metrics.get('det_fp', 0),
        'det_fn': metrics.get('det_fn', 0),
        'det_precision': metrics.get('det_precision'),
        'det_recall': metrics.get('det_recall'),
        'det_f1': metrics.get('det_f1'),
        'cor_tp': metrics.get('cor_tp', 0),
        'cor_fp': metrics.get('cor_fp', 0),
        'cor_precision': metrics.get('cor_precision'),
        'cor_recall': metrics.get('cor_recall'),
        'cor_f1': metrics.get('cor_f1'),
    }


def aggregate_lake_metrics(per_table_rows: list) -> dict:
    """Sum counts across tables, then compute lake-level P/R/F1.

    Successfully cleaned tables contribute full detection/correction counts.
    Skipped/failed tables still contribute their ground-truth error cells to
    the recall denominators (counted as missed detection / uncorrected).
    """
    totals = {
        'det_tp': 0, 'det_fp': 0, 'det_fn': 0,
        'cor_tp': 0, 'cor_fp': 0, 'errors': 0,
    }
    for row in per_table_rows:
        status = row.get('status')
        if status == 'ok':
            for k in totals:
                totals[k] += row.get(k, 0) or 0
            continue

        errors = row.get('errors', 0) or 0
        if errors <= 0:
            continue
        skipped = skipped_table_metrics(errors)
        for k in totals:
            totals[k] += skipped[k]

    lake = metrics_from_counts(totals)
    lake['tables_included'] = sum(1 for row in per_table_rows if row.get('status') == 'ok')
    return lake


def format_lake_summary(lake_dir, table_dirs, tables_ok, tables_skipped, tables_failed,
                        lake: dict, lake_rows: int) -> str:
    return (
        f"\nLake directory : {lake_dir}\n"
        f"Tables total   : {len(table_dirs)}\n"
        f"Tables cleaned : {tables_ok}\n"
        f"Tables skipped : {tables_skipped}\n"
        f"Tables failed  : {tables_failed}\n"
        f"\nDetection\n"
        f"det_TP         : {lake.get('det_tp', 0)}\n"
        f"det_FP         : {lake.get('det_fp', 0)}\n"
        f"det_FN         : {lake.get('det_fn', 0)}\n"
        f"Detection P/R/F1 : {lake.get('det_precision', 0):.6f} / "
        f"{lake.get('det_recall', 0):.6f} / {lake.get('det_f1', 0):.6f}\n"
        f"\nCorrection\n"
        f"cor_TP         : {lake.get('cor_tp', 0)}\n"
        f"cor_FP         : {lake.get('cor_fp', 0)}\n"
        f"Total errors   : {lake.get('errors', 0)}\n"
        f"Correction P/R/F1: {lake.get('cor_precision', 0):.6f} / "
        f"{lake.get('cor_recall', 0):.6f} / {lake.get('cor_f1', 0):.6f}\n"
        f"Total rows     : {lake_rows}\n"
    )


def lake_evaluation_json(tables_ok, tables_skipped, tables_failed, lake_rows, lake: dict) -> dict:
    return {
        'tables_cleaned': tables_ok,
        'tables_skipped': tables_skipped,
        'tables_failed': tables_failed,
        'total_rows': lake_rows,
        'detection': {
            'tp': lake.get('det_tp', 0),
            'fp': lake.get('det_fp', 0),
            'fn': lake.get('det_fn', 0),
            'precision': lake.get('det_precision', 0),
            'recall': lake.get('det_recall', 0),
            'f1': lake.get('det_f1', 0),
        },
        'correction': {
            'tp': lake.get('cor_tp', 0),
            'fp': lake.get('cor_fp', 0),
            'errors': lake.get('errors', 0),
            'precision': lake.get('cor_precision', 0),
            'recall': lake.get('cor_recall', 0),
            'f1': lake.get('cor_f1', 0),
        },
    }
