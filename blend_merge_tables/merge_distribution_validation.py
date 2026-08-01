"""
Distribution-based merge validation.

Union: compare clean marginals on each aligned column pair (left vs right).

Join: compare each source column's clean marginal before merge (isolated table)
vs after merge (values projected into the merged table via provenance). This
captures distribution shift from dangling tuples and join filtering.

Categorical columns use TVD; numeric columns use KS. With default thresholds of
1.0, only total mismatch or missing clean data rejects a merge; metrics are
still logged for ablation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from error_cells import get_error_profile
from merge_validation import is_empty_value, load_table_dirty, parse_provenance


@dataclass
class ColumnDistributionCheck:
    check_kind: str
    left_table: str
    right_table: str
    left_col: int
    right_col: int
    value_type: str
    metric: str
    metric_value: float
    threshold: float
    passed: bool
    left_clean_count: int
    right_clean_count: int

    def to_row(self) -> dict:
        return {
            "check_kind": self.check_kind,
            "left_table": self.left_table,
            "right_table": self.right_table,
            "left_col": self.left_col,
            "right_col": self.right_col,
            "value_type": self.value_type,
            "metric": self.metric,
            "metric_value": f"{self.metric_value:.6f}",
            "threshold": self.threshold,
            "passed": self.passed,
            "left_clean_count": self.left_clean_count,
            "right_clean_count": self.right_clean_count,
        }


@dataclass
class DistributionValidationResult:
    passed: bool
    checks: List[ColumnDistributionCheck] = field(default_factory=list)
    max_tvd: float = 0.0
    max_ks: float = 0.0

    def summary(self) -> str:
        n_failed = sum(1 for c in self.checks if not c.passed)
        return (
            f"distribution_passed={self.passed} "
            f"max_tvd={self.max_tvd:.4f} max_ks={self.max_ks:.4f} "
            f"checks={len(self.checks)} failed={n_failed}"
        )


def column_appears_numeric(values: List[str]) -> bool:
    domain = {value for value in values if not is_empty_value(value)}
    if not domain:
        return False
    try:
        [float(value) for value in domain]
    except ValueError:
        return False
    return True


def clean_values_for_column(tab_name: str, col_id: int) -> List[str]:
    tab_path = config.DIR_PATH / tab_name
    profile = get_error_profile(tab_path)
    _, data = load_table_dirty(tab_path)
    if col_id >= len(data) or not data[col_id]:
        return []

    values: List[str] = []
    for row_id, value in enumerate(data[col_id]):
        if profile.is_clean(row_id, col_id, dirty_value=value) and not is_empty_value(value):
            values.append(value)
    return values


def clean_merged_values_for_source_column(
    data: List[List[str]],
    tracker: List[List[str]],
    tab_name: str,
    source_col_id: int,
) -> List[str]:
    if not data or not data[0]:
        return []

    profile = get_error_profile(config.DIR_PATH / tab_name)
    values: List[str] = []
    num_rows = len(data[0])
    for merged_col in range(len(data)):
        for row_id in range(num_rows):
            prov = tracker[merged_col][row_id]
            parsed = parse_provenance(prov)
            if parsed is None:
                continue
            table_name, source_row, source_col = parsed
            if table_name != tab_name or source_col != source_col_id:
                continue
            value = data[merged_col][row_id]
            if profile.is_clean(source_row, source_col, dirty_value=value) and not is_empty_value(value):
                values.append(value)
    return values


def normalized_hist(values: List[str]) -> Dict[str, float]:
    counts = Counter(value for value in values if not is_empty_value(value))
    total = sum(counts.values())
    if total == 0:
        return {}
    return {value: count / total for value, count in counts.items()}


def total_variation_distance(left: List[str], right: List[str]) -> float:
    p = normalized_hist(left)
    q = normalized_hist(right)
    if not p and not q:
        return 0.0
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(key, 0.0) - q.get(key, 0.0)) for key in keys)


def _to_floats(values: List[str]) -> List[float]:
    floats: List[float] = []
    for value in values:
        if is_empty_value(value):
            continue
        try:
            floats.append(float(value))
        except ValueError:
            continue
    return floats


def kolmogorov_smirnov_statistic(left: List[str], right: List[str]) -> float:
    left_nums = sorted(_to_floats(left))
    right_nums = sorted(_to_floats(right))
    if not left_nums or not right_nums:
        return 0.0

    points = sorted(set(left_nums + right_nums))
    left_idx = 0
    right_idx = 0
    max_diff = 0.0
    for point in points:
        while left_idx < len(left_nums) and left_nums[left_idx] <= point:
            left_idx += 1
        while right_idx < len(right_nums) and right_nums[right_idx] <= point:
            right_idx += 1
        max_diff = max(max_diff, abs(left_idx / len(left_nums) - right_idx / len(right_nums)))
    return max_diff


def _compare_distributions(
    *,
    check_kind: str,
    left_table: str,
    right_table: str,
    left_col: int,
    right_col: int,
    left_values: List[str],
    right_values: List[str],
) -> ColumnDistributionCheck:
    tvd_threshold = config.DIST_TVD_THRESHOLD
    ks_threshold = config.DIST_KS_THRESHOLD

    left_count = len(left_values)
    right_count = len(right_values)
    if left_count == 0 or right_count == 0:
        return ColumnDistributionCheck(
            check_kind=check_kind,
            left_table=left_table,
            right_table=right_table,
            left_col=left_col,
            right_col=right_col,
            value_type="unknown",
            metric="insufficient_clean",
            metric_value=1.0,
            threshold=0.0,
            passed=False,
            left_clean_count=left_count,
            right_clean_count=right_count,
        )

    pooled = left_values + right_values
    if column_appears_numeric(pooled):
        metric_value = kolmogorov_smirnov_statistic(left_values, right_values)
        threshold = ks_threshold
        metric_name = "ks"
        value_type = "numeric"
    else:
        metric_value = total_variation_distance(left_values, right_values)
        threshold = tvd_threshold
        metric_name = "tvd"
        value_type = "categorical"

    return ColumnDistributionCheck(
        check_kind=check_kind,
        left_table=left_table,
        right_table=right_table,
        left_col=left_col,
        right_col=right_col,
        value_type=value_type,
        metric=metric_name,
        metric_value=metric_value,
        threshold=threshold,
        passed=metric_value <= threshold,
        left_clean_count=left_count,
        right_clean_count=right_count,
    )


def _record_metric_extremes(check: ColumnDistributionCheck, max_tvd: float, max_ks: float) -> tuple[float, float]:
    if check.metric == "tvd":
        max_tvd = max(max_tvd, check.metric_value)
    elif check.metric == "ks":
        max_ks = max(max_ks, check.metric_value)
    return max_tvd, max_ks


def _validate_union_distribution(
    left_table: str,
    right_table: str,
    mapping: Dict[int, int],
) -> DistributionValidationResult:
    checks: List[ColumnDistributionCheck] = []
    max_tvd = 0.0
    max_ks = 0.0

    for left_col, right_col in sorted(mapping.items()):
        left_values = clean_values_for_column(left_table, left_col)
        right_values = clean_values_for_column(right_table, right_col)
        check = _compare_distributions(
            check_kind="union_cross_table",
            left_table=left_table,
            right_table=right_table,
            left_col=left_col,
            right_col=right_col,
            left_values=left_values,
            right_values=right_values,
        )
        checks.append(check)
        max_tvd, max_ks = _record_metric_extremes(check, max_tvd, max_ks)

    return DistributionValidationResult(
        passed=all(check.passed for check in checks),
        checks=checks,
        max_tvd=max_tvd,
        max_ks=max_ks,
    )


def _source_column_count(tab_name: str) -> int:
    _, data = load_table_dirty(config.DIR_PATH / tab_name)
    return len(data)


def _validate_join_distribution(
    left_table: str,
    right_table: str,
    data: List[List[str]],
    tracker: List[List[str]],
) -> DistributionValidationResult:
    checks: List[ColumnDistributionCheck] = []
    max_tvd = 0.0
    max_ks = 0.0

    for tab_name in (left_table, right_table):
        for col_id in range(_source_column_count(tab_name)):
            before_values = clean_values_for_column(tab_name, col_id)
            after_values = clean_merged_values_for_source_column(
                data, tracker, tab_name, col_id
            )
            check = _compare_distributions(
                check_kind="join_before_after",
                left_table=tab_name,
                right_table=tab_name,
                left_col=col_id,
                right_col=col_id,
                left_values=before_values,
                right_values=after_values,
            )
            checks.append(check)
            max_tvd, max_ks = _record_metric_extremes(check, max_tvd, max_ks)

    return DistributionValidationResult(
        passed=all(check.passed for check in checks),
        checks=checks,
        max_tvd=max_tvd,
        max_ks=max_ks,
    )


def validate_merge_distribution(
    left_table: str,
    right_table: str,
    mapping: Dict[int, int],
    operation: str,
    *,
    data: Optional[List[List[str]]] = None,
    tracker: Optional[List[List[str]]] = None,
) -> DistributionValidationResult:
    if not mapping:
        return DistributionValidationResult(passed=False, checks=[])

    op = operation.strip().lower()
    if op == "union":
        return _validate_union_distribution(left_table, right_table, mapping)
    if op == "join":
        if data is None or tracker is None:
            return DistributionValidationResult(passed=False, checks=[])
        return _validate_join_distribution(left_table, right_table, data, tracker)

    return DistributionValidationResult(passed=False, checks=[])
