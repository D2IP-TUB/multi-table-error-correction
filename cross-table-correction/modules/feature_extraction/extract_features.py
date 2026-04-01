from typing import List, Optional

import numpy as np

from core.candidate import Candidate

FEATURE_GROUPS = [
    ("value_based", [
        "remover_identity", "remover_unicode",
        "adder_identity", "adder_unicode",
        "replacer_identity", "replacer_unicode",
        "swapper_identity", "swapper_unicode",
    ]),
    ("vicinity_based", [
        "vicinity_based_avg_prob", "vicinity_based_first_col",
        "vicinity_based_left_neighbor", "vicinity_based_right_neighbor",
    ]),
    ("domain_based", ["domain_based"]),
    ("levenshtein", ["levenshtein_similarity"]),
    ("pattern_based", ["pattern_based_reliability"]),
]


def extract_features_from_candidate(
    candidate: Candidate,
    include_pattern_features: bool = True,
    disabled_feature_groups: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extract numerical features from the Candidate object.

    Args:
        candidate: The candidate object to extract features from
        include_pattern_features: Whether to include pattern-based features
                                  (False for rule-based zoning)
        disabled_feature_groups: Feature groups to exclude entirely.
                                 Valid names: value_based, vicinity_based,
                                 domain_based, levenshtein, pattern_based
    """
    disabled = set(disabled_feature_groups or [])
    features = []

    for group_name, feature_keys in FEATURE_GROUPS:
        if group_name in disabled:
            continue
        if group_name == "pattern_based" and not include_pattern_features:
            continue
        for key in feature_keys:
            features.append(candidate.features.get(key, 0.0))

    features_array = np.array(features, dtype=np.float32)
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    return features_array


def get_feature_names(
    include_pattern_features: bool = True,
    disabled_feature_groups: Optional[List[str]] = None,
) -> list:
    """
    Return the ordered list of feature names matching the extraction output.

    Args:
        include_pattern_features: Whether to include pattern-based features
                                  (False for rule-based zoning)
        disabled_feature_groups: Feature groups to exclude.
    """
    disabled = set(disabled_feature_groups or [])
    names = []

    for group_name, feature_keys in FEATURE_GROUPS:
        if group_name in disabled:
            continue
        if group_name == "pattern_based" and not include_pattern_features:
            continue
        names.extend(feature_keys)

    return names
