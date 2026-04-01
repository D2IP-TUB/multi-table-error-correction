"""
Profiling module for EC-at-Scale.

This module provides functionality to build profiles of clean data
and compute deviation scores for error correction.
"""

from modules.profiling.build_profiles import (
    build_column_profiles,
)
from modules.profiling.column_profile import ColumnProfile
from modules.profiling.initialize import (
    initialize_profiles,
    load_or_build_profiles,
    update_profiles_after_sampling,
)
from modules.profiling.io import get_profile_summary, load_profiles, save_profiles
from modules.profiling.mask_utils import (
    compute_cardinality_stats,
    # Distribution functions
    compute_frequency_histogram,
    compute_length_statistics,
    # Pattern functions
    compute_mask_histogram,
    compute_numeric_format_stats,
    compute_numeric_statistics,
    # Legacy function
    get_top_k_values,
    # Type inference functions
    infer_basic_type,
    infer_data_type,
    # Basic pattern functions
    value_to_mask,
)
from modules.profiling.tane_wrapper import (
    TANEWrapper,
    setup_tane_integration,
)

__all__ = [
    # Core classes
    "ColumnProfile",
    # Profile building
    "build_column_profiles",
    # Initialization functions
    "initialize_profiles",
    "load_or_build_profiles",
    "update_profiles_after_sampling",
    # I/O functions
    "save_profiles",
    "load_profiles",
    "get_profile_summary",
    # Utility functions - Basic
    "value_to_mask",
    "get_top_k_values",
    # Utility functions - Cardinality
    "compute_cardinality_stats",
    # Utility functions - Distributions
    "compute_frequency_histogram",
    "compute_numeric_statistics",
    # Utility functions - Patterns
    "compute_mask_histogram",
    "compute_length_statistics",
    "compute_numeric_format_stats",
    # Utility functions - Type Inference
    "infer_basic_type",
    "infer_data_type",
    # TANE integration
    "TANEWrapper",
    "setup_tane_integration",
]
