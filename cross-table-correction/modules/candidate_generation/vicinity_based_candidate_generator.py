import logging
from collections import defaultdict
from typing import Dict, List

from core.candidate import Candidate
from core.cell import Cell
from core.table import Table
from modules.candidate_generation.candidate_generator import CandidateGenerator


class VicinityBasedCorrector(CandidateGenerator):
    """
    vicinity corrector
    """

    def __init__(self, config):
        super().__init__(config)
        self._table_clean_data = {}  # Pre-computed clean data per table
        self._fd_confidence_cache = {}  # Cache FD confidences
        self._vicinity_model_cache = {}  # Cache vicinity models
        self._labeled_relationships = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        )  # Store relationships from labeled samples: table_id -> context_col -> col_idx -> context_val -> target_val -> count
        self.vicinity_confidence_threshold = (
            config.pruning.vicinity_confidence_threshold
        )

    def _precompute_table_data(self, table: Table):
        if table.table_id in self._table_clean_data:
            return

        error_positions = {
            (cell.row_idx, cell.column_idx)
            for cell in table.get_all_cells()
            if cell.is_error
        }

        context_rows = []
        n_cols = table.dataframe.shape[1]

        for row_idx, row in table.dataframe.iterrows():
            row_data = {}
            for col_idx in range(n_cols):
                if (row_idx, col_idx) not in error_positions:
                    val = row.iloc[col_idx]
                    row_data[col_idx] = str(val)
            if row_data:
                context_rows.append(row_data)

        col_relationships = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        )

        for row_data in context_rows:
            for context_col, context_val in row_data.items():
                for target_col, target_val in row_data.items():
                    if context_col != target_col:
                        col_relationships[context_col][target_col][context_val][
                            target_val
                        ] += 1

        self._table_clean_data[table.table_id] = {
            "clean_rows": context_rows,
            "col_relationships": col_relationships,
            "n_cols": n_cols,
        }

    def _get_fd_confidence(
        self, context_col: int, target_col: int, table: Table
    ) -> float:
        cache_key = (table.table_id, context_col, target_col)
        if cache_key in self._fd_confidence_cache:
            return self._fd_confidence_cache[cache_key]

        self._precompute_table_data(table)
        col_relationships = self._table_clean_data[table.table_id]["col_relationships"]

        confidence = 0.0
        if (
            context_col in col_relationships
            and target_col in col_relationships[context_col]
        ):
            context_to_targets = col_relationships[context_col][target_col]
            violations = sum(
                1 for targets in context_to_targets.values() if len(targets) > 1
            )
            total = len(context_to_targets)
            confidence = (total - violations) / total if total > 0 else 0.0

        self._fd_confidence_cache[cache_key] = confidence
        # logging.debug(
        #     f"FD confidence for context_col {context_col}, target_col {target_col} in table {table.table_name}: {confidence}"
        # )
        return confidence

    def _get_vicinity_distribution(
        self, context_val: str, context_col: int, target_col: int, table: Table
    ) -> Dict[str, float]:
        cache_key = (table.table_id, context_col, target_col, context_val)
        if cache_key in self._vicinity_model_cache:
            return self._vicinity_model_cache[cache_key]

        self._precompute_table_data(table)
        col_relationships = self._table_clean_data[table.table_id]["col_relationships"]

        # Combine clean data relationships with labeled sample relationships
        combined_dist = defaultdict(int)
        
        if context_col in col_relationships:
            if target_col in col_relationships[context_col]:
                if context_val in col_relationships[context_col][target_col]:
                    dist = col_relationships[context_col][target_col][context_val]
                    for val, count in dist.items():
                        combined_dist[val] += count

        # Add counts from labeled samples (Baran's approach: just add counts)
        table_key = table.table_id
        if (
            table_key in self._labeled_relationships
            and context_col in self._labeled_relationships[table_key]
            and target_col in self._labeled_relationships[table_key][context_col]
            and context_val in self._labeled_relationships[table_key][context_col][target_col]
        ):
            labeled_dist = self._labeled_relationships[table_key][context_col][target_col][context_val]
            for val, count in labeled_dist.items():
                combined_dist[val] += count

        # Convert counts to probabilities
        corrections = {}
        total = sum(combined_dist.values())
        if total > 0:
            corrections = {val: count / total for val, count in combined_dist.items()}

        self._vicinity_model_cache[cache_key] = corrections
        return dict(corrections)

    def update_from_labeled_samples(self, samples: List[Cell], tables: Dict[str, Table] = None):
        """Update vicinity model with labeled samples using Baran's approach.
        
        For each labeled correction, we learn that when a context column has
        a certain value, the target column should have the corrected value.
        Also learns from clean cells in the same row to capture valid relationships.
        """
        logging.info(f"Updating vicinity model from {len(samples)} labeled samples...")

        if tables is None or not tables:
            logging.warning(
                "Vicinity model update skipped: tables required to extract row context for vicinity relationships"
            )
            return

        # Group samples by row to process entire labeled tuples
        samples_by_row = defaultdict(list)
        for sample_cell in samples:
            if hasattr(sample_cell, "ground_truth") and sample_cell.ground_truth:
                row_key = (sample_cell.table_id, sample_cell.row_idx)
                samples_by_row[row_key].append(sample_cell)

        # Track which cache keys need to be invalidated
        updated_fd_keys = set()
        updated_vicinity_keys = set()

        for (table_id, row_idx), error_samples in samples_by_row.items():
            # Get the correct table for this row
            row_table = tables.get(table_id)
            
            if row_table is None:
                logging.warning(
                    f"Could not find table {table_id} for row {row_idx}, skipping row"
                )
                continue
            
            # Get the complete row with all corrected values
            row = row_table.dataframe.iloc[row_idx]
            n_cols = row_table.dataframe.shape[1]
            
            # Build the corrected row: use ground truth for error cells, original for clean cells
            corrected_row = {}
            error_cols = set()
            for sample_cell in error_samples:
                target_col = sample_cell.column_idx
                corrected_row[target_col] = str(sample_cell.ground_truth)
                error_cols.add(target_col)
            
            # Add clean cells (non-error columns)
            for col_idx in range(n_cols):
                if col_idx not in corrected_row:
                    corrected_row[col_idx] = str(row.iloc[col_idx])
            
            # Update vicinity relationships following Baran's logic:
            # For error cells: use all context columns (full vicinity)
            # For clean cells: use only error columns as context (shows valid combinations)
            for col_idx in range(n_cols):
                target_val = corrected_row[col_idx]
                
                if col_idx in error_cols:
                    # Error cell: learn from all context columns
                    for context_col in range(n_cols):
                        if context_col == col_idx:
                            continue
                        context_val = corrected_row[context_col]
                        self._labeled_relationships[table_id][context_col][col_idx][context_val][
                            target_val
                        ] += 1
                        # Track affected cache keys
                        updated_fd_keys.add((table_id, context_col, col_idx))
                        updated_vicinity_keys.add((table_id, context_col, col_idx, context_val))
                else:
                    # Clean cell: learn only from error columns as context
                    for context_col in error_cols:
                        context_val = corrected_row[context_col]
                        self._labeled_relationships[table_id][context_col][col_idx][context_val][
                            target_val
                        ] += 1
                        # Track affected cache keys
                        updated_fd_keys.add((table_id, context_col, col_idx))
                        updated_vicinity_keys.add((table_id, context_col, col_idx, context_val))

        # Invalidate only affected cache entries
        for key in updated_fd_keys:
            self._fd_confidence_cache.pop(key, None)
        for key in updated_vicinity_keys:
            self._vicinity_model_cache.pop(key, None)


    def generate_candidates(self, cell: Cell, table: Table):
        """
        Generate vicinity-based candidates.
        
        Returns:
            Dict mapping value -> candidate pool key (table_id, column_idx, value)
        """
        try:
            target_col = cell.column_idx
            row = table.dataframe.iloc[cell.row_idx]
            self._precompute_table_data(table)
            n_cols = self._table_clean_data[table.table_id]["n_cols"]

            position_corrections = {}

            for context_col in range(n_cols):
                if context_col == target_col:
                    continue
                context_val = row.iloc[context_col]
                fd_confidence = self._get_fd_confidence(context_col, target_col, table)
                if fd_confidence < self.vicinity_confidence_threshold:
                    continue

                corrections = self._get_vicinity_distribution(
                    context_val, context_col, target_col, table
                )

                if corrections and fd_confidence >= self.vicinity_confidence_threshold:
                    position_corrections[context_col] = {
                        "corrections": corrections,
                        "fd_confidence": fd_confidence,
                    }

            all_corrections = set()
            for context_col, data in position_corrections.items():
                all_corrections.update(data["corrections"].keys())

            candidates = {}
            candidates_source_model = {}
            for val in all_corrections:
                candidates_source_model[val] = {}
                candidates_source_model[val]["vicinity_context_columns"] = []
                features = {
                    "vicinity_based_avg_prob": 0.0,
                    "vicinity_based_first_col": 0.0,
                    "vicinity_based_left_neighbor": 0.0,
                    "vicinity_based_right_neighbor": 0.0,
                }
                weighted_probs = []

                for context_col, data in position_corrections.items():
                    if val in data["corrections"]:
                        prob = data["corrections"][val]
                        fd = data["fd_confidence"]
                        score = prob * fd
                        if context_col == 0:
                            features["vicinity_based_first_col"] = score
                        elif context_col == target_col - 1:
                            features["vicinity_based_left_neighbor"] = score
                        elif context_col == target_col + 1:
                            features["vicinity_based_right_neighbor"] = score
                        weighted_probs.append(score)
                        candidates_source_model[val]["vicinity_context_columns"].append(
                            context_col
                        )

                if weighted_probs:
                    features["vicinity_based_avg_prob"] = sum(weighted_probs) / len(
                        weighted_probs
                    )
                if len(candidates_source_model[val]) == 0:
                    print("**********")
                
                candidate = Candidate(
                    val,
                    features,
                    candidates_source_model=candidates_source_model[val],
                )
                
                # Add to pool and store reference
                pool_key = self._add_candidate_to_pool(
                    table.table_id, target_col, val, candidate
                )
                candidates[val] = pool_key
        except Exception as e:
            logging.error(f"Error generating candidates for cell : {e}")
            return {}
        return candidates

    def get_strategy_name(self) -> str:
        return "vicinity_based"