"""
building functions
"""

import logging
from typing import Dict, Tuple

from core.lake import Lake
from modules.profiling.column_profile import ColumnProfile
from modules.profiling.mask_utils import (
    compute_cardinality_stats,
    compute_frequency_histogram,
    compute_length_statistics,
    compute_mask_histogram,
    compute_numeric_format_stats,
    compute_numeric_statistics,
    infer_data_type,
)


def build_column_profiles(lake: Lake) -> Dict[Tuple[str, int], ColumnProfile]:
    """
    Build profiles for all columns in the lake using only clean cells.

    Args:
        lake: Lake containing tables with clean/dirty data

    Returns:
        Dictionary mapping (table_id, column_idx) to ColumnProfile
    """
    logging.info(f"Building column profiles for {len(lake.tables)} tables...")

    profiles = {}
    total_columns = 0

    for table_id, table in lake.tables.items():
        logging.debug(f"Processing table {table_id} ({table.table_name})")

        for column_idx, column in table.columns.items():
            total_columns += 1
            profile_key = (table_id, column_idx)

            # Extract clean values from this column
            clean_values = []
            total_cells = 0

            error_values = []
            for cell in column.cells.values():
                total_cells += 1
                if not cell.is_error and cell.value is not None:
                    clean_values.append(str(cell.value))
                elif cell.is_error:
                    error_values.append(str(cell.value))

            if not clean_values or not error_values:
                logging.warning(
                    f"No clean/error values found for column {column_idx} in table {table_id}"
                )
                # Create empty profile with basic structure
                profiles[profile_key] = ColumnProfile(
                    table_id=table_id,
                    column_idx=column_idx,
                    column_name=getattr(column, "col_name", None),
                )
                continue

            logging.debug(
                f"Column {column_idx}: {len(clean_values)} clean values out of {total_cells} total"
            )

            # Build complete profile
            profile = _build_single_column_profile(
                table_id=table_id,
                column_idx=column_idx,
                column_name=getattr(column, "col_name", None),
                clean_values=clean_values,
                total_rows=total_cells,
            )

            profiles[profile_key] = profile

    logging.info(f"Built {len(profiles)} column profiles from {total_columns} columns")
    return profiles


def _build_single_column_profile(
    table_id: str,
    column_idx: int,
    column_name: str,
    clean_values: list,
    total_rows: int,
) -> ColumnProfile:
    """
    Build a profile for a single column.

    Args:
        table_id: Table identifier
        column_idx: Column index
        column_name: Column name (optional)
        clean_values: List of clean string values
        total_rows: Total number of rows in column (including nulls/errors)

    Returns:
        ColumnProfile instance
    """
    # === CARDINALITY STATISTICS ===
    cardinality_stats = compute_cardinality_stats(clean_values, total_rows)
    uniqueness = cardinality_stats["uniqueness"]

    # === VALUE DISTRIBUTION STATISTICS ===
    # Only compute value histogram if not 100% unique
    if uniqueness == 1.0:
        value_histogram = {}
    else:
        top_values, _ = compute_frequency_histogram(clean_values, k=None)  # No limit
        value_histogram = dict(top_values)

    # Numeric statistics (computed by casting string values)
    numeric_stats = compute_numeric_statistics(clean_values)

    # === PATTERN AND TYPE ANALYSIS ===
    mask_histogram = compute_mask_histogram(clean_values)
    length_stats = compute_length_statistics(clean_values)
    numeric_format_stats = compute_numeric_format_stats(clean_values)

    # === TYPE INFERENCE ===
    # Use basic type detection, then infer DBMS type
    from modules.profiling.mask_utils import infer_basic_type

    basic_type = infer_basic_type(clean_values)
    inferred_data_type = infer_data_type(clean_values, basic_type)

    # === SEMANTIC PROFILES ===
    # Note: Embeddings, n-grams, and FDs will be computed in separate steps
    clean_value_embeddings = {}
    clean_char_ngrams = set()
    functional_dependencies = []
    fd_confidence_scores = {}

    # Generate character n-grams
    clean_char_ngrams = set()
    for value in clean_values:
        for i in range(len(value) - 2 + 1):  # 2-grams
            clean_char_ngrams.add(value[i : i + 2])

    # Generate embeddings using robust fallback method
    clean_value_embeddings = {}
    try:
        from modules.profiling.embedder import get_embedder
        embedder = get_embedder("bert-base-uncased")
        if embedder is not None:
            logging.debug(f"Generating embeddings for {table_id} column {column_idx}")
            embeddings = embedder.encode(clean_values)
            clean_value_embeddings = {
                value: embedding for value, embedding in zip(clean_values, embeddings)
            }
            logging.debug(f"Generated embeddings for {table_id} column {column_idx} ({len(clean_value_embeddings)} values)")
    except Exception as e:
        logging.warning(
            f"Error generating embeddings for {table_id} column {column_idx}: {type(e).__name__}: {e}. "
            f"Continuing without embeddings."
        )

    # Create complete profile
    profile = ColumnProfile(
        table_id=table_id,
        column_idx=column_idx,
        column_name=column_name,
        value_histogram=value_histogram,
        min_value=numeric_stats["min_value"],
        max_value=numeric_stats["max_value"],
        q1=numeric_stats["q1"],
        q2=numeric_stats["q2"],
        q3=numeric_stats["q3"],
        mean_value=numeric_stats["mean_value"],
        std_value=numeric_stats["std_value"],
        # Contextual dependencies (placeholder)
        functional_dependencies=functional_dependencies,
        fd_confidence_scores=fd_confidence_scores,
        # Semantic profiles (placeholder)
        clean_value_embeddings=clean_value_embeddings,
        clean_char_ngrams=clean_char_ngrams,
        # Patterns and types
        basic_type=basic_type,
        inferred_data_type=inferred_data_type,
        mask_histogram=mask_histogram,
        # Length statistics
        length_min=int(length_stats["min"]),
        length_max=int(length_stats["max"]),
        length_median=length_stats["median"],
        length_mu=length_stats["mean"],
        length_sigma=length_stats["std"],
        # Numeric format
        max_digits=numeric_format_stats["max_digits"],
        max_decimals=numeric_format_stats["max_decimals"],
        has_negatives=numeric_format_stats["has_negatives"],
        has_scientific=numeric_format_stats["has_scientific"],
    )

    logging.debug(f"Built profile: {profile}")
    return profile
