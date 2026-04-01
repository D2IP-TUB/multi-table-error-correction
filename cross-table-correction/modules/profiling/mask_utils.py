"""
mask utilities
"""

import re
import statistics
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ------------- MASK & CHAR HISTOGRAM -------------


def value_to_mask(value: str) -> str:
    """
    Convert a value to a Unicode category mask using 2-letter categories
    (e.g., 'Ll' for lowercase letter, 'Lu' for uppercase, 'Nd' for decimal digit).
    Example: 'John123' -> 'LuLlLlLlNdNdNd'
    """
    if not value:
        return ""
    return "".join(unicodedata.category(ch) for ch in value)


# ------------- CARDINALITY -------------


def compute_cardinality_stats(values: List[str], total_rows: int) -> Dict[str, Any]:
    clean_values = [v for v in values if v not in (None, "")]
    clean_count = len(clean_values)
    unique_values = len(set(clean_values)) if clean_values else 0
    return {
        "total_clean_cells": clean_count,
        "unique_clean_values": unique_values,
        "uniqueness": (unique_values / clean_count) if clean_count > 0 else 0.0,
    }


# ------------- VALUE DISTRIBUTION -------------


def compute_frequency_histogram(
    values: List[str], k: Optional[int] = 100
) -> Tuple[List[Tuple[str, int]], float]:
    if not values:
        return [], 0.0
    value_counts = Counter(v for v in values if v not in ("", None))

    if k is None:
        # Return all values
        top_k = value_counts.most_common()
    else:
        # Return top k
        top_k = value_counts.most_common(k)

    constancy = (top_k[0][1] / sum(value_counts.values())) if top_k else 0.0
    return top_k, constancy


# ------------- NUMERIC STATS -------------

_NUM_PAT = re.compile(
    r"""
    ^\s*
    (?P<sign>[+-])?
    (?:
        (?:
            (?:\d{1,3}(?:[_,\s]\d{3})+|\d+)
            (?:\.\d+)?     # decimals
        )
        |
        (?:\.\d+)         # .5 style
    )
    (?:[eE][+-]?\d+)?     # scientific
    \s*$
""",
    re.X,
)


def _parse_float_safe(s: str) -> Optional[float]:
    if not s or not s.strip():
        return None
    s = s.strip()
    if not _NUM_PAT.match(s):
        return None
    # normalize thousands separators (comma/underscore/space)
    s_norm = re.sub(r"[,\s_](?=\d{3}\b)", "", s)
    try:
        return float(s_norm)
    except ValueError:
        return None


def compute_numeric_statistics(values: List[str]) -> Dict[str, Optional[float]]:
    nums = []
    for v in values:
        f = (
            _parse_float_safe(v)
            if isinstance(v, str)
            else (float(v) if v is not None else None)
        )
        if f is not None:
            nums.append(f)
    if len(nums) < 2:
        return {
            "min_value": None,
            "max_value": None,
            "mean_value": None,
            "std_value": None,
            "q1": None,
            "q2": None,
            "q3": None,
        }
    nums_sorted = sorted(nums)
    min_val = nums_sorted[0]
    max_val = nums_sorted[-1]
    mean_val = statistics.mean(nums_sorted)
    std_val = statistics.stdev(nums_sorted) if len(nums_sorted) > 1 else 0.0
    try:
        q1, q3 = (
            statistics.quantiles(nums_sorted, n=4)[0],
            statistics.quantiles(nums_sorted, n=4)[2],
        )
    except statistics.StatisticsError:
        q1 = q3 = None
    q2 = statistics.median(nums_sorted)
    return {
        "min_value": min_val,
        "max_value": max_val,
        "mean_value": mean_val,
        "std_value": std_val,
        "q1": q1,
        "q2": q2,
        "q3": q3,
    }


# ------------- PATTERN HISTOS -------------


def compute_mask_histogram(values: List[str]) -> Dict[str, int]:
    counts = Counter()
    for v in values:
        if v:
            counts[value_to_mask(v)] += 1
    return dict(counts)


def compute_length_statistics(values: List[str]) -> Dict[str, float]:
    lens = [len(str(v)) for v in values if v not in (None, "")]
    if not lens:
        return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0, "median": 0.0}
    return {
        "mean": statistics.mean(lens),
        "std": statistics.stdev(lens) if len(lens) > 1 else 0.0,
        "min": min(lens),
        "max": max(lens),
        "median": statistics.median(lens),
    }


def compute_numeric_format_stats(values: List[str]) -> Dict[str, Any]:
    max_digits = 0
    max_decimals = 0
    has_negatives = False
    has_scientific = False
    for v in values:
        if not v:
            continue
        s = v.strip()
        if s.startswith("-"):
            has_negatives = True
        if "e" in s.lower():
            # don't try to split; just flag
            if _NUM_PAT.match(s):
                has_scientific = True
            continue
        # strip sign and thousands separators
        s_norm = re.sub(r"^[+-]", "", s)
        s_norm = re.sub(r"[,\s_](?=\d{3}\b)", "", s_norm)
        if "." in s_norm:
            a, b = s_norm.split(".", 1)
            if a.replace(" ", "").isdigit() and b.isdigit():
                max_digits = max(max_digits, len(a))
                max_decimals = max(max_decimals, len(b))
        elif s_norm.isdigit():
            max_digits = max(max_digits, len(s_norm))
    return {
        "max_digits": max_digits,
        "max_decimals": max_decimals,
        "has_negatives": has_negatives,
        "has_scientific": has_scientific,
    }


# ------------- TYPE INFERENCE (with dateutil) -------------

try:
    from dateutil import parser as _du_parser

    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False


def _is_date_like(value: str) -> bool:
    """
    Robust date/datetime detection using python-dateutil.
    """
    if not DATEUTIL_AVAILABLE or not value or not value.strip():
        return False
    s = value.strip()
    # quick pre-filter: must have at least 2 digits somewhere
    if len(re.findall(r"\d", s)) < 2:
        return False
    try:
        # Try strict-ish parse first
        _du_parser.parse(s, default=None, dayfirst=False, yearfirst=False, fuzzy=False)
        return True
    except Exception:
        # Try a cautious fuzzy parse; reject if we had to drop almost everything
        try:
            dt = _du_parser.parse(s, default=None, fuzzy=True)
            # sanity check year range
            if 1900 <= dt.year <= 2100:
                return True
            return False
        except Exception:
            return False


def _is_float(value: str) -> bool:
    return _parse_float_safe(value) is not None


def infer_basic_type(values: List[str]) -> str:
    if not values:
        return "unknown"
    all_vals = [v for v in values if v not in (None, "")]
    counts = {"integer": 0, "float": 0, "boolean": 0, "date": 0, "string": 0}
    for v in all_vals:
        s = str(v).strip()
        if not s:
            continue
        # boolean
        if s.lower() in {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}:
            counts["boolean"] += 1
            continue
        # integer
        if re.fullmatch(r"[+-]?\d+", s):
            counts["integer"] += 1
            continue
        # float
        if _is_float(s):
            counts["float"] += 1
            continue
        # date
        if _is_date_like(s):
            counts["date"] += 1
            continue
        counts["string"] += 1

    # numeric override
    if counts["integer"] > 0 and counts["float"] > 0:
        return "numeric"
    # dominant
    return max(counts.items(), key=lambda kv: kv[1])[0] if all_vals else "unknown"


def infer_data_type(values: List[str], basic_type: str) -> str:
    if not values:
        return "varchar"
    if basic_type == "integer":
        return "int"
    elif basic_type in ("float", "numeric"):
        return "decimal"
    elif basic_type == "boolean":
        return "boolean"
    elif basic_type == "date":
        return "timestamp"
    else:
        return "varchar"


# ------------- TOP-K (kept for backward compatibility) -------------


def get_top_k_values(values: List[str], k: int = 100) -> List[Tuple[str, int]]:
    if not values:
        return []
    counts = Counter(v for v in values if v not in ("", None))
    return counts.most_common(k)
