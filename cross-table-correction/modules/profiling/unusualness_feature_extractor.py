"""
Unusualness Feature Extractor for Global Cell Representation.

Computes 10 unusualness features for dirty cells based on:
- Syntactic deviation from column profiles
- Semantic deviation from clean domain
- Contextual violations (functional dependencies)
- Column meta-features
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from rapidfuzz.distance import Levenshtein

from core.cell import Cell
from modules.profiling.column_profile import ColumnProfile
from modules.profiling.mask_utils import _parse_float_safe, value_to_mask


class UnusualnessFeaturesExtractor:
    """Extract unusualness features for dirty cells using column profiles"""

    def __init__(
        self,
        profiles: Dict[Tuple[str, int], ColumnProfile],
        lake=None,
        embedding_model=None,
    ):
        """
        Initialize feature extractor with column profiles.

        Args:
            profiles: Dictionary mapping (table_id, column_idx) to ColumnProfile
            lake: Optional Lake object for FD violation checking
            embedding_model: Optional SentenceTransformer model for computing dirty cell embeddings
        """
        self.profiles = profiles
        self.lake = lake
        self.embedding_model = embedding_model
        self._clean_data_cache = {}  # Cache for FD lookups

    def extract_features(
        self, cell: Cell, row_data: Optional[Dict[int, str]] = None
    ) -> np.ndarray:
        """
        Extract all 10 unusualness features for a dirty cell.

        Args:
            cell: Dirty cell to extract features for
            row_data: Optional dictionary mapping column_idx -> value for the full row

        Returns:
            numpy array of 10 features, each in [0,1] range
        """
        profile_key = (cell.table_id, cell.column_idx)
        profile = self.profiles.get(profile_key)

        if profile is None:
            logging.warning(
                f"No profile found for {profile_key}, returning zero features"
            )
            return np.zeros(10)

        features = []

        # === SYNTACTIC UNUSUALNESS FEATURES ===
        features.append(self._pattern_unusualness(cell.value, profile))  # Feature 1
        features.append(self._length_deviation(cell.value, profile))  # Feature 2
        features.append(self._data_type_violation(cell.value, profile))  # Feature 3
        features.append(self._numeric_range_violation(cell.value, profile))  # Feature 4
        features.append(
            self._numeric_format_violation(cell.value, profile)
        )  # Feature 5

        # === SEMANTIC UNUSUALNESS FEATURES ===
        features.append(self._value_unusualness(cell.value, profile))  # Feature 6
        features.append(self._min_embedding_distance(cell.value, profile))  # Feature 7
        features.append(self._char_ngram_overlap(cell.value, profile))  # Feature 8
        features.append(self._min_edit_distance(cell.value, profile))  # Feature 9

        # === CONTEXTUAL UNUSUALNESS FEATURES ===
        features.append(self._fd_violation_score(cell, profile, row_data))  # Feature 10

        return np.array(features, dtype=np.float32)

    def extract_features_batch(self, cells: List[Cell]) -> np.ndarray:
        """
        Extract features for multiple cells efficiently.

        Args:
            cells: List of dirty cells

        Returns:
            numpy array of shape (n_cells, 10)
        """
        features_matrix = []
        for cell in cells:
            features = self.extract_features(cell)
            features_matrix.append(features)

        return np.array(features_matrix, dtype=np.float32)

    # ========== SYNTACTIC FEATURES ==========

    def _pattern_unusualness(self, value: str, profile: ColumnProfile) -> float:
        """How unusual is this pattern compared to clean patterns? [0,1]"""
        if not value:
            return 1.0

        pattern = value_to_mask(value)
        pattern_prob = profile.mask_prob(pattern)

        if pattern_prob == 0.0:
            return 1.0  # Completely novel pattern
        else:
            return 1.0 - pattern_prob  # High probability = low unusualness

    def _length_deviation(self, value: str, profile: ColumnProfile) -> float:
        """How unusual is this length compared to clean lengths? [0,1]"""
        if not value:
            return 1.0

        value_length = len(value)

        if profile.length_sigma == 0.0:
            # All clean values have same length
            return 1.0 if value_length != profile.length_mu else 0.0

        # Z-score based unusualness, capped at 3 standard deviations
        z_score = abs(value_length - profile.length_mu) / profile.length_sigma
        return min(z_score / 3.0, 1.0)

    def _data_type_violation(self, value: str, profile: ColumnProfile) -> float:
        """Does this value violate the expected data type? [0,1]"""
        if not value:
            return 0.0

        expected_type = profile.inferred_data_type

        # Check type violations
        if expected_type in ["int", "tinyint", "smallint", "bigint"]:
            # Should be integer
            if not re.fullmatch(r"[+-]?\d+", value.strip()):
                return 1.0
        elif expected_type == "decimal":
            # Should be numeric
            if _parse_float_safe(value) is None:
                return 1.0
        elif expected_type == "boolean":
            # Should be boolean
            if value.lower().strip() not in {
                "true",
                "false",
                "yes",
                "no",
                "1",
                "0",
                "t",
                "f",
                "y",
                "n",
            }:
                return 1.0
        elif expected_type == "timestamp":
            # Should be date-like (simplified check)
            if not re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", value):
                return 1.0

        return 0.0  # No violation

    def _numeric_range_violation(self, value: str, profile: ColumnProfile) -> float:
        """Is this numeric value outside expected range? [0,1]"""
        if (
            not profile.is_numeric()
            or profile.min_value is None
            or profile.max_value is None
        ):
            return 0.0

        numeric_value = _parse_float_safe(value)
        if numeric_value is None:
            return 0.0  # Not numeric, handled by data_type_violation

        # Check if outside range
        if numeric_value < profile.min_value or numeric_value > profile.max_value:
            return 1.0
        else:
            return 0.0

    def _numeric_format_violation(self, value: str, profile: ColumnProfile) -> float:
        """Does this value violate numeric format constraints? [0,1]"""
        if not profile.is_numeric():
            return 0.0

        numeric_value = _parse_float_safe(value)
        if numeric_value is None:
            return 1.0  # Should be numeric but isn't

        # Check format constraints
        value_str = value.strip()

        # Check negative constraint
        if not profile.has_negatives and value_str.startswith("-"):
            return 1.0

        # Check scientific notation constraint
        if not profile.has_scientific and ("e" in value_str.lower()):
            return 1.0

        # Check digits/decimals constraints (simplified)
        if "." in value_str:
            parts = value_str.split(".")
            if (
                len(parts[0]) > profile.max_digits
                or len(parts[1]) > profile.max_decimals
            ):
                return 1.0
        elif len(value_str.lstrip("+-")) > profile.max_digits:
            return 1.0

        return 0.0

    # ========== SEMANTIC FEATURES ==========

    def _value_unusualness(self, value: str, profile: ColumnProfile) -> float:
        """How unusual is this value in categorical domain? [0,1]"""
        if profile.uniqueness == 1.0:
            return 0.0  # Unique columns - any value is equally unusual

        value_prob = profile.value_prob(value)
        if value_prob == 0.0:
            return 1.0  # Never seen this value
        else:
            return 1.0 - value_prob  # High probability = low unusualness

    def _min_embedding_distance(self, value: str, profile: ColumnProfile) -> float:
        """Minimum semantic distance to clean embeddings [0,1]"""
        if not profile.has_embeddings():
            return 0.0  # No embeddings available

        try:
            # Get embedding for dirty value (would need embedding model here)
            dirty_embedding = self._compute_embedding(value)

            if dirty_embedding is None:
                return 1.0

            # Find minimum cosine distance to clean embeddings
            min_distance = 1.0
            for clean_embedding in profile.clean_value_embeddings.values():
                # Cosine similarity -> cosine distance
                similarity = np.dot(dirty_embedding, clean_embedding) / (
                    np.linalg.norm(dirty_embedding) * np.linalg.norm(clean_embedding)
                )
                distance = 1.0 - similarity
                min_distance = min(min_distance, distance)

            return min_distance

        except Exception:
            return 1.0  # Error computing embeddings

    def _char_ngram_overlap(self, value: str, profile: ColumnProfile) -> float:
        """Character n-gram overlap with clean values [0,1]"""
        overlap_ratio = profile.get_char_ngram_overlap(value, n=2)
        return 1.0 - overlap_ratio  # High overlap = low unusualness

    def _min_edit_distance(self, value: str, profile: ColumnProfile) -> float:
        """Minimum edit distance to clean domain values [0,1]"""
        if not profile.value_histogram or profile.uniqueness == 1.0:
            return 0.0  # No meaningful comparison possible

        try:
            # Find minimum normalized edit distance to clean values
            min_distance = 1.0

            for clean_value in profile.value_histogram.keys():
                if clean_value:  # Skip empty values
                    distance = Levenshtein.normalized_distance(value, clean_value)
                    min_distance = min(min_distance, distance)

            return min_distance

        except Exception:
            return 1.0  # Error computing distances

    # ========== CONTEXTUAL FEATURES ==========

    def _fd_violation_score(
        self,
        cell: Cell,
        profile: ColumnProfile,
        row_data: Optional[Dict[int, str]] = None,
    ) -> float:
        """Functional dependency violation score [0,1]"""
        if not profile.has_functional_dependencies():
            return 0.0  # No FDs to violate

        if row_data is None:
            # Get row data from the table if not provided
            row_data = self._get_row_data(cell)

        if not row_data:
            return 0.0  # Can't check FDs without row context

        # Check each FD where this column is the RHS
        max_violation = 0.0

        for lhs_cols in profile.functional_dependencies:
            # Look up confidence for this FD
            fd_key = (lhs_cols, cell.column_idx)
            confidence = profile.fd_confidence_scores.get(fd_key, 1.0)

            # Get LHS values for this row
            lhs_values = []
            for lhs_col in lhs_cols:
                if lhs_col in row_data:
                    lhs_values.append(row_data[lhs_col])
                else:
                    # Missing LHS value, can't check this FD
                    continue

            if len(lhs_values) != len(lhs_cols):
                continue  # Incomplete LHS, skip this FD

            # Create LHS tuple for lookup
            lhs_tuple = tuple(lhs_values)

            # Check if this FD is violated
            # We need the expected RHS value given this LHS
            expected_rhs = self._get_expected_rhs_from_clean_data(
                cell.table_id, lhs_cols, lhs_tuple, cell.column_idx
            )

            if expected_rhs is not None and expected_rhs != cell.value:
                # FD violation detected
                # Weight by confidence: higher confidence FDs matter more
                violation_score = confidence
                max_violation = max(max_violation, violation_score)

        return min(max_violation, 1.0)  # Cap at 1.0

    def _get_row_data(self, cell: Cell) -> Dict[int, str]:
        """Get the full row data for a cell"""
        if self.lake is None:
            return {}

        table = self.lake.tables.get(cell.table_id)
        if table is None:
            return {}

        table_df = table.dataframe
        if cell.row_idx >= len(table_df):
            return {}

        # Extract all column values for this row
        row_data = {}
        for col_idx in range(len(table_df.columns)):
            try:
                row_data[col_idx] = str(table_df.iloc[cell.row_idx, col_idx])
            except (IndexError, KeyError):
                # Handle missing data gracefully
                row_data[col_idx] = ""

        return row_data

    def _get_expected_rhs_from_clean_data(
        self,
        table_id: str,
        lhs_cols: List[int],
        lhs_values: Tuple[str, ...],
        rhs_col: int,
    ) -> Optional[str]:
        """
        Look up expected RHS value from clean data given LHS values.

        This queries the clean data to find what RHS value
        typically appears with these LHS values.
        """
        cache_key = (table_id, tuple(lhs_cols), lhs_values, rhs_col)
        if cache_key in self._clean_data_cache:
            return self._clean_data_cache[cache_key]

        # Try to get table data from either lake or cached table data
        table_df = None
        clean_rows = None
        
        # First check if we have cached table data (from parallel processing)
        if hasattr(self, '_table_clean_data') and table_id in self._table_clean_data:
            table_df = self._table_clean_data[table_id]['dataframe']
            clean_rows = self._table_clean_data[table_id]['clean_rows']
        # Otherwise try lake
        elif self.lake is not None:
            table = self.lake.tables.get(table_id)
            if table is None:
                return None
            
            table_df = table.dataframe
            
            # Get clean rows (rows with no errors)
            error_positions = set()
            for cell in table.get_all_cells():
                if cell.is_error:
                    error_positions.add((cell.row_idx, cell.column_idx))
            
            clean_rows = set()
            for row_idx in range(len(table_df)):
                row_has_errors = any(
                    (row_idx, col_idx) in error_positions
                    for col_idx in range(len(table_df.columns))
                )
                if not row_has_errors:
                    clean_rows.add(row_idx)
        else:
            return None
        
        if table_df is None or clean_rows is None:
            return None

        # Look through clean rows for this LHS pattern
        expected_rhs = None

        for row_idx in clean_rows:
            if row_idx >= len(table_df):
                continue
            
            # Check if LHS matches
            row_lhs_values = []
            for lhs_col in lhs_cols:
                if lhs_col < len(table_df.columns):
                    row_lhs_values.append(str(table_df.iloc[row_idx, lhs_col]))

            if tuple(row_lhs_values) == lhs_values:
                # Found matching LHS, get RHS value
                if rhs_col < len(table_df.columns):
                    expected_rhs = str(table_df.iloc[row_idx, rhs_col])
                    break

        # Cache the result
        self._clean_data_cache[cache_key] = expected_rhs
        return expected_rhs

    # ========== HELPER METHODS ==========

    def _compute_embedding(self, value: str) -> Optional[np.ndarray]:
        """Compute embedding for a value using the same model used for profiles"""
        if self.embedding_model is None or not value.strip():
            return None

        try:
            embedding = self.embedding_model.encode([value], show_progress_bar=False)[0]
            return embedding
        except Exception as e:
            logging.debug(f"Failed to compute embedding for '{value}': {e}")
            return None

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        return [
            "pattern_unusualness",
            "length_deviation",
            "data_type_violation",
            "numeric_range_violation",
            "numeric_format_violation",
            "value_unusualness",
            "min_embedding_distance",
            "char_ngram_overlap",
            "min_edit_distance",
            "fd_violation_score",
        ]


def extract_unusualness_features_for_lake(
    lake,
    profiles: Dict[Tuple[str, int], ColumnProfile],
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_workers: int = None,
) -> Tuple[List[Cell], np.ndarray, List[str]]:
    """
    Extract unusualness features for all dirty cells in a lake.

    Args:
        lake: Lake object containing tables with dirty cells
        profiles: Dictionary of column profiles
        embedding_model_name: Name of embedding model to use for dirty cell embeddings
        n_workers: Number of parallel workers (None = auto-detect)

    Returns:
        Tuple of (dirty_cells_list, features_matrix, feature_names)
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    logging.info("Extracting unusualness features for all dirty cells in lake...")

    if n_workers is None:
        n_workers = min(len(lake.tables), os.cpu_count())

    # Prepare work units (one per table)
    work_units = []
    for table_id, table in lake.tables.items():
        # Group dirty cells by row for this table
        dirty_cells_by_row = {}
        for cell in table.get_all_cells():
            if cell.is_error:
                if cell.row_idx not in dirty_cells_by_row:
                    dirty_cells_by_row[cell.row_idx] = []
                dirty_cells_by_row[cell.row_idx].append(cell)

        if dirty_cells_by_row:  # Only process tables with dirty cells
            work_units.append(
                (
                    table_id,
                    table.dataframe.copy(),  # Pass DataFrame copy to avoid serialization issues
                    dirty_cells_by_row,
                    profiles,
                    embedding_model_name,
                )
            )

    if not work_units:
        logging.warning("No dirty cells found in lake")
        return [], np.array([]), _get_feature_names()

    # Process tables in parallel
    all_dirty_cells = []
    all_features = []

    if n_workers == 1 or len(work_units) == 1:
        # Single-threaded processing
        for work_unit in work_units:
            cells, features = _process_table_features(work_unit)
            all_dirty_cells.extend(cells)
            all_features.extend(features)
    else:
        # Multi-process execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all table processing jobs
            future_to_table = {
                executor.submit(_process_table_features, work_unit): work_unit[0]
                for work_unit in work_units
            }

            # Collect results as they complete
            for future in as_completed(future_to_table):
                table_id = future_to_table[future]
                try:
                    cells, features = future.result()
                    all_dirty_cells.extend(cells)
                    all_features.extend(features)
                    logging.debug(f"Processed {len(cells)} cells from table {table_id}")
                except Exception as e:
                    logging.error(f"Error processing table {table_id}: {e}")

    features_matrix = np.array(all_features, dtype=np.float32)
    feature_names = _get_feature_names()

    logging.info(
        f"Extracted {features_matrix.shape[1]} features for {features_matrix.shape[0]} cells using {n_workers} workers"
    )

    return all_dirty_cells, features_matrix, feature_names


def _process_table_features(work_unit) -> Tuple[List[Cell], List[np.ndarray]]:
    """
    Process features for all dirty cells in a single table.
    This function runs in a separate process.
    """
    table_id, table_df, dirty_cells_by_row, profiles, embedding_model_name = work_unit

    # Load embedding model in this process with fallback
    embedding_model = None
    try:
        from modules.profiling.embedder import get_embedder
        embedding_model = get_embedder("bert-base-uncased")
    except Exception as e:
        logging.warning(f"Could not load embedding model: {e}. Continuing without embeddings.")

    # Create extractor for this process (no lake reference needed here)
    extractor = UnusualnessFeaturesExtractor(
        profiles, lake=None, embedding_model=embedding_model
    )
    
    # Build clean data mapping for FD lookups
    # Get error positions from dirty_cells_by_row
    error_positions = set()
    for row_idx, cells in dirty_cells_by_row.items():
        for cell in cells:
            error_positions.add((cell.row_idx, cell.column_idx))
    
    # Pre-compute clean data for FD lookups (identify rows with no errors in any column)
    clean_rows = set()
    for row_idx in range(len(table_df)):
        row_has_errors = any(
            (row_idx, col_idx) in error_positions
            for col_idx in range(len(table_df.columns))
        )
        if not row_has_errors:
            clean_rows.add(row_idx)
    
    # Store clean data in extractor for FD lookups
    extractor._table_clean_data = {
        table_id: {
            'dataframe': table_df,
            'clean_rows': clean_rows
        }
    }

    processed_cells = []
    processed_features = []

    # Process each row with dirty cells
    for row_idx, cells_in_row in dirty_cells_by_row.items():
        if row_idx < len(table_df):
            # Build row_data once per row
            row_data = {}
            for col_idx in range(len(table_df.columns)):
                row_data[col_idx] = str(table_df.iloc[row_idx, col_idx])

            # Extract features for all cells in this row
            for cell in cells_in_row:
                features = extractor.extract_features(cell, row_data)
                processed_cells.append(cell)
                processed_features.append(features)

    return processed_cells, processed_features


def _get_feature_names() -> List[str]:
    """Get feature names without needing an extractor instance"""
    return [
        "pattern_unusualness",
        "length_deviation",
        "data_type_violation",
        "numeric_range_violation",
        "numeric_format_violation",
        "value_unusualness",
        "min_embedding_distance",
        "char_ngram_overlap",
        "min_edit_distance",
        "fd_violation_score",
        # "column_uniqueness",
    ]
