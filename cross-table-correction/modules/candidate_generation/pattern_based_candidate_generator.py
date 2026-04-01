import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from core.candidate import Candidate
from core.cell import Cell
from core.table import Table
from modules.candidate_generation.candidate_generator import CandidateGenerator

PYPROSE_AVAILABLE = None
_learn_patterns = None


def _ensure_pyprose():
    """Lazily load PyProse (and the Mono runtime) on first use."""
    global PYPROSE_AVAILABLE, _learn_patterns
    if PYPROSE_AVAILABLE is not None:
        return PYPROSE_AVAILABLE
    try:
        from pyprose.matching.text import learn_patterns as _lp
        _learn_patterns = _lp
        PYPROSE_AVAILABLE = True
    except ImportError:
        PYPROSE_AVAILABLE = False
        logging.warning("PyProse not available. Pattern-based correction disabled.")
    return PYPROSE_AVAILABLE


class PatternBasedCorrector(CandidateGenerator):
    """
    Pattern-based corrector using PyProse for pattern learning and enforcement.
    """

    def __init__(self, config):
        super().__init__(config)
        self._column_best_patterns = {}  # Cache: (table_id, col_idx) -> best_pattern
        self._pattern_reliability = {}  # Track: (table_id, col_idx) -> reliability_score
        self._labeled_samples_by_column = defaultdict(
            list
        )  # Store samples for pattern testing
        self._cells_to_move = defaultdict(list)  # Track cells that moved zones after pattern enforcement

        if not _ensure_pyprose():
            logging.warning(
                "PyProse not available. Pattern-based correction will not work."
            )

    def find_best_pattern(self, patterns):
        """Find the best pattern based on matching fraction"""
        if not patterns:
            return None

        patterns_matching_fractions = [
            pattern.matching_fraction for pattern in patterns
        ]
        max_matching_fraction_index = patterns_matching_fractions.index(
            max(patterns_matching_fractions)
        )
        best_pattern = patterns[max_matching_fraction_index]
        return best_pattern

    def _normalize_extract_result(self, result):
        """
        Normalize PyProse extract() return value.
        
        PyProse.extract() can return:
        - A string: use as-is
        - A list: join with space (multiple extracted tokens)
        - None: return None
        
        Args:
            result: Return value from pattern.extract()
            
        Returns:
            Normalized string or None
        """
        if result is None:
            return None
        
        # If it's a list, join the tokens with space
        if isinstance(result, list):
            return " ".join(str(item).strip() for item in result if item)
        
        # Otherwise treat as string
        return result

    def update_from_labeled_samples(self, samples: List[Cell]):
        """Store labeled samples for later pattern effectiveness testing"""
        if not PYPROSE_AVAILABLE:
            return

        logging.info(
            f"Storing {len(samples)} labeled samples for pattern effectiveness testing..."
        )

        # Just store the samples - we'll use them when we learn patterns from clean data
        self._labeled_samples_by_column = defaultdict(list)
        for sample in samples:
            if hasattr(sample, "ground_truth") and sample.ground_truth:
                column_key = (sample.table_id, sample.column_idx)
                self._labeled_samples_by_column[column_key].append(sample)
        
        # Invalidate pattern cache when new labeled samples arrive
        # Patterns need to be re-learned with updated reliability scores
        self._column_best_patterns = {}
        self._pattern_reliability = {}
        logging.debug("Pattern cache invalidated due to new labeled samples")

    def _learn_column_pattern(self, cell: Cell, table: Table):
        """Learn best pattern from clean data AND test it with labeled samples"""
        column_key = (cell.table_id, cell.column_idx)

        # Skip if already learned
        if column_key in self._column_best_patterns:
            return

        try:
            # Step 1: Get clean values from column
            clean_values = self._extract_clean_column_values(cell.column_idx, table)
            
            # Step 2: Get sample ground truths
            sample_ground_truths = self._get_sample_ground_truths(column_key)
            
            # Step 3: Learn patterns from combined clean values + sample ground truths
            all_values = clean_values + sample_ground_truths
            
            if len(clean_values) > 0 and len(sample_ground_truths) > 0:
                logging.debug(
                    f"Column {column_key}: Using {len(clean_values)} clean values + {len(sample_ground_truths)} sample ground truths (combined)"
                )
            elif len(clean_values) > 0:
                logging.debug(
                    f"Column {column_key}: Using {len(clean_values)} clean values from column"
                )
            elif len(sample_ground_truths) > 0:
                logging.debug(
                    f"Column {column_key}: Using {len(sample_ground_truths)} sample ground truths"
                )

            if len(all_values) >= 2:
                patterns = _learn_patterns(all_values, include_outlier_patterns=True)
                best_pattern = self.find_best_pattern(patterns)
                self._column_best_patterns[column_key] = best_pattern

                # Step 4: Determine reliability score
                if best_pattern:
                    # Case 1: Have both clean values and labeled samples
                    if len(clean_values) > 0 and column_key in self._labeled_samples_by_column:
                        # Get clean values match count (from matching_fraction)
                        clean_success_count = int(best_pattern.matching_fraction * len(clean_values))
                        clean_total_count = len(clean_values)
                        
                        # Get labeled samples fix count
                        labeled_success_count, labeled_total_count = self._get_pattern_test_counts(
                            column_key, best_pattern
                        )
                        
                        # Combine: total successes / total count
                        total_success = clean_success_count + labeled_success_count
                        total_count = clean_total_count + labeled_total_count
                        reliability = total_success / total_count if total_count > 0 else 0.0
                        self._pattern_reliability[column_key] = reliability
                        logging.debug(
                            f"Column {column_key}: Pattern reliability = {reliability:.3f} (combined: {clean_success_count}/{clean_total_count} clean + {labeled_success_count}/{labeled_total_count} labeled = {total_success}/{total_count})"
                        )
                    
                    # Case 2: Only have samples (no clean values) AND have labeled samples to test
                    elif len(clean_values) == 0 and column_key in self._labeled_samples_by_column:
                        reliability = self._test_pattern_effectiveness(
                            column_key, best_pattern
                        )
                        self._pattern_reliability[column_key] = reliability
                        logging.debug(
                            f"Column {column_key}: Pattern reliability = {reliability:.3f} (tested with labeled samples, learned from samples only)"
                        )
                    
                    # Case 3: No labeled samples available (regardless of learning source)
                    else:
                        reliability = best_pattern.matching_fraction
                        self._pattern_reliability[column_key] = reliability
                        logging.debug(
                            f"Column {column_key}: Pattern reliability = {reliability:.3f} (no labeled samples available, using matching_fraction)"
                        )

                if best_pattern:
                    logging.debug(
                        f"Learned best pattern for column {column_key} (matching_fraction: {best_pattern.matching_fraction:.3f})"
                    )
            else:
                self._column_best_patterns[column_key] = None
                self._pattern_reliability[column_key] = 0.0
                logging.debug(
                    f"Column {column_key}: Insufficient data to learn pattern (< 2 values)"
                )

        except Exception as e:
            logging.warning(f"Failed to learn patterns for column {column_key}: {e}")
            self._column_best_patterns[column_key] = None
            self._pattern_reliability[column_key] = 0.0

    def _extract_clean_column_values(self, col_idx: int, table: Table) -> List[str]:
        """Extract clean values from column (excluding error cells)"""
        clean_values = []

        # Get error positions
        error_positions = {
            cell.row_idx
            for cell in table.get_all_cells()
            if cell.is_error and cell.column_idx == col_idx
        }

        # Extract clean values
        column_data = table.dataframe.iloc[:, col_idx]
        for row_idx, value in column_data.items():
            if (
                row_idx not in error_positions
                and value is not None
                and str(value).strip()
            ):
                clean_values.append(str(value).strip())

        return clean_values

    def _get_sample_ground_truths(self, column_key) -> List[str]:
        """Get ground truth values from labeled samples for this column"""
        clean_values = []
        column_samples = self._labeled_samples_by_column.get(column_key, [])

        for sample in column_samples:
            if hasattr(sample, "ground_truth") and sample.ground_truth:
                clean_values.append(str(sample.ground_truth).strip())

        return clean_values

    def _test_pattern_effectiveness(self, column_key, best_pattern):
        """Test pattern effectiveness using labeled samples"""
        success_count, total_attempts = self._get_pattern_test_counts(column_key, best_pattern)
        return success_count / total_attempts if total_attempts > 0 else 0.0

    def _get_pattern_test_counts(self, column_key, best_pattern):
        """Get success and total counts from testing pattern on labeled samples"""
        samples = self._labeled_samples_by_column.get(column_key, [])
        if not samples:
            return 0, 0

        success_count = 0
        total_attempts = 0

        for sample in samples:
            if hasattr(sample, "value") and hasattr(sample, "ground_truth"):
                corrupted_value = str(sample.value).strip()
                expected_value = str(sample.ground_truth).strip()

                if (
                    corrupted_value
                    and expected_value
                    and corrupted_value != expected_value
                ):
                    total_attempts += 1

                    try:
                        extracted = best_pattern.extract(corrupted_value)
                        corrected_value = self._normalize_extract_result(extracted)

                        if (
                            corrected_value
                            and str(corrected_value).strip() == expected_value
                        ):
                            success_count += 1

                    except Exception:
                        pass

        return success_count, total_attempts

    def generate_candidates(self, cell: Cell, table: Table):
        """
        Generate pattern-based candidates using the best learned pattern.
        
        Returns:
            Dict mapping value -> candidate pool key (table_id, column_idx, value)
        """
        if not PYPROSE_AVAILABLE:
            return {}

        # Learn pattern for this column (uses both clean data and tests with samples)
        self._learn_column_pattern(cell, table)

        column_key = (cell.table_id, cell.column_idx)
        best_pattern = self._column_best_patterns.get(column_key)

        if not best_pattern:
            return {}

        candidates = {}
        cell_value = str(cell.value).strip() if cell.value else ""

        if not cell_value:
            return {}

        try:
            # Use the best PyProse pattern to extract/correct the value
            extracted = best_pattern.extract(cell_value)
            corrected_value = self._normalize_extract_result(extracted)

            if (
                corrected_value
                and str(corrected_value).strip()
                and str(corrected_value).strip() != cell_value
            ):
                corrected_value = str(corrected_value).strip()

                # Get the pre-calculated reliability
                pattern_reliability = self._pattern_reliability.get(column_key, 0.0)

                # Create candidate with only the reliability feature
                features = {
                    "pattern_based_reliability": pattern_reliability,
                }

                candidate = Candidate(
                    corrected_value,
                    features,
                    candidates_source_model={"pattern_based": 1},
                )
                
                # Add to pool and store reference
                pool_key = self._add_candidate_to_pool(
                    table.table_id, cell.column_idx, corrected_value, candidate
                )
                candidates[corrected_value] = pool_key

        except Exception as e:
            logging.debug(f"Best pattern application failed for '{cell_value}': {e}")

        return candidates

    def enforce_pattern_on_invalid_zone(
        self, cell: Cell, table: Table
    ) -> Tuple[Dict[str, Candidate], Optional[None]]:
        """
        Attempt to fix cells in invalid pattern zones using pattern enforcement.
        
        If a pattern match succeeds, returns the candidates and zone change information.
        The caller is responsible for applying zone transitions.
        
        Args:
            cell: Cell from invalid_pattern zone
            table: The table containing the cell
            
        Returns:
            Tuple of (candidates_dict, zone_change_info)
            - candidates_dict: Dict of candidates from pattern enforcement {value: Candidate}, empty dict if no patterns matched
            - zone_change_info: Tuple of (old_zone, new_zone, cell) if cell should move zones, None otherwise
        """
        if not PYPROSE_AVAILABLE:
            return {}, None
            
        # Learn pattern for this column
        self._learn_column_pattern(cell, table)
        
        column_key = (cell.table_id, cell.column_idx)
        best_pattern = self._column_best_patterns.get(column_key)
        
        if not best_pattern:
            return {}, None
            
        cell_value = str(cell.value).strip() if cell.value else ""
        if not cell_value:
            return {}, None
        
        pattern_candidates = {}
        zone_new_samples = []
        
        try:
            # Attempt to enforce/fix the pattern
            extracted = best_pattern.extract(cell_value)
            corrected_value = self._normalize_extract_result(extracted)
            
            if (
                corrected_value
                and str(corrected_value).strip()
                and str(corrected_value).strip() != cell_value
            ):
                # Pattern enforcement succeeded!
                corrected_value = str(corrected_value).strip()
                
                # Get the pre-calculated reliability
                pattern_reliability = self._pattern_reliability.get(column_key, 0.0)
                
                # Create candidate with pattern enforcement reliability feature
                features = {
                    "pattern_enforcement_reliability": pattern_reliability,
                    "pattern_enforced": 1.0,  # Flag to track this came from enforcement
                }
                
                pattern_candidates[corrected_value] = Candidate(
                    corrected_value,
                    features,
                    candidates_source_model={"pattern_enforcement": 1},
                )
                
        except Exception as e:
            logging.debug(f"Pattern enforcement failed for '{cell_value}': {e}")
            
        return pattern_candidates, None

    def enforce_pattern_batch(
        self, cells: List[Cell], table: Table
    ) -> Dict[int, Dict[str, Candidate]]:
        """
        Batch enforce patterns on multiple cells for the same table.
        This is much more efficient than calling enforce_pattern_on_invalid_zone() for each cell.
        
        Args:
            cells: List of cells from invalid_pattern zone in the same table
            table: The table containing the cells
            
        Returns:
            Dict mapping cell row index to candidates {row_idx: {value: Candidate}}
        """
        if not PYPROSE_AVAILABLE or not cells:
            return {}
        
        # Group cells by column
        cells_by_column = defaultdict(list)
        for cell in cells:
            cells_by_column[cell.column_idx].append(cell)
        
        results = {}
        
        # Process each column - learn pattern once per column, then apply to all cells in that column
        for col_idx, column_cells in cells_by_column.items():
            column_key = (table.table_id, col_idx)
            
            # Learn pattern once for this column (cached if already learned)
            if column_cells:
                self._learn_column_pattern(column_cells[0], table)
            
            best_pattern = self._column_best_patterns.get(column_key)
            if not best_pattern:
                continue
            
            pattern_reliability = self._pattern_reliability.get(column_key, 0.0)
            
            # Now enforce pattern on all cells in this column
            for cell in column_cells:
                cell_value = str(cell.value).strip() if cell.value else ""
                if not cell_value:
                    continue
                
                try:
                    extracted = best_pattern.extract(cell_value)
                    corrected_value = self._normalize_extract_result(extracted)
                    
                    if (
                        corrected_value
                        and str(corrected_value).strip()
                        and str(corrected_value).strip() != cell_value
                    ):
                        corrected_value = str(corrected_value).strip()
                        features = {
                            "pattern_enforcement_reliability": pattern_reliability,
                            "pattern_enforced": 1.0,
                        }
                        results[cell.row_idx] = {
                            corrected_value: Candidate(
                                corrected_value,
                                features,
                                candidates_source_model={"pattern_enforcement": 1},
                            )
                        }
                except Exception as e:
                    logging.debug(f"Pattern enforcement failed for '{cell_value}': {e}")
                    continue
        
        return results

    def get_strategy_name(self) -> str:
        return "pattern_based"

