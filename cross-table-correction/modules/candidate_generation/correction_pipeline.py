import logging
from collections import defaultdict
from typing import Dict, List

from core.candidate import Candidate
from core.candidate_pool import CandidatePool
from core.cell import Cell
from core.table import Table
from modules.candidate_generation.candidate_generator import CandidateGenerator
from modules.candidate_generation.domain_based_candidate_generator import (
    DomainBasedCorrector,
)
from modules.candidate_generation.pattern_based_candidate_generator import (
    PatternBasedCorrector,
)
from modules.candidate_generation.value_based_candidate_generator import (
    ValueBasedCorrector,
)
from modules.candidate_generation.vicinity_based_candidate_generator import (
    VicinityBasedCorrector,
)


class CorrectionPipeline:
    """
    Correction Pipeline
    """

    def __init__(self, config, enabled_strategies: List[str]):
        self.config = config
        self.enbled_strategies = enabled_strategies
        self.correctors = self._initialize_correctors(enabled_strategies)
        self._total_cells_processed = 0
        self.enable_pattern_enforcement = getattr(
            getattr(config, "correction", None),
            "enable_pattern_enforcement",
            True,
        )
        mode = getattr(
            getattr(config, "correction", None),
            "pattern_enforcement_mode",
            "",
        )
        mode = (mode or "").strip().lower()
        if mode in {"check", "always_accept", "disabled"}:
            self.pattern_enforcement_mode = mode
        else:
            # Backward-compatible fallback for legacy boolean flag.
            self.pattern_enforcement_mode = (
                "check" if self.enable_pattern_enforcement else "disabled"
            )

    def _initialize_correctors(self, enabled_strategies) -> List[CandidateGenerator]:
        """Initialize correctors"""
        correctors = []

        if "value_based" in enabled_strategies:
            correctors.append(ValueBasedCorrector(self.config))

        if "vicinity_based" in enabled_strategies:
            correctors.append(VicinityBasedCorrector(self.config))

        if "domain_based" in enabled_strategies:
            correctors.append(DomainBasedCorrector(self.config))

        if "pattern_based" in enabled_strategies:
            correctors.append(PatternBasedCorrector(self.config))

        logging.info(
            f"Initialized {len(correctors)} correctors: {[c.get_strategy_name() for c in correctors]}"
        )

        return correctors

    def update_with_labeled_samples(self, samples: List[Cell], tables: Dict[str, Table] = None):
        """Update correctors with labeled samples"""
        logging.info(f"Updating correctors with {len(samples)} labeled samples...")

        for corrector in self.correctors:
            if hasattr(corrector, "update_from_labeled_samples"):
                # For vicinity and domain based correctors that need table access
                import inspect
                sig = inspect.signature(corrector.update_from_labeled_samples)
                
                # Pass appropriate parameters based on what the corrector accepts
                if len(sig.parameters) > 1:
                    # Corrector accepts table/tables parameter
                    # Pass tables dict for correctors that work with multiple tables
                    corrector.update_from_labeled_samples(samples, tables)
                else:
                    # Legacy correctors that only take samples
                    corrector.update_from_labeled_samples(samples)

    def generate_candidates_for_cell(
        self, zone_name: str, cell: Cell, table: Table
    ) -> Dict[str, Candidate]:
        """Generate candidates for a single cell.
        
        Zone-aware strategy:
        - Unique zones: Only pattern-based and value-based (no vicinity/domain)
        - Non-unique zones: All strategies
        - Valid zones: Skip pattern-based (patterns already verified)
        - Invalid zones: Use all applicable strategies
        """
        all_candidates = {}
        n_candidates_per_corrector = defaultdict(int)
        
        # Determine which strategies to use based on zone
        for corrector in self.correctors:
            strategy_name = corrector.get_strategy_name()
            
            # For unique zones (high cardinality), skip vicinity and domain-based
            is_unique_zone = zone_name.startswith("unique_")
            if is_unique_zone:
                if strategy_name in ["vicinity_based", "domain_based"]:
                    logging.debug(f"Skipping {strategy_name} for unique zone: {zone_name}")
                    continue
            
            # For VALID zones (contain "valid_pattern"), skip pattern-based (patterns already verified)
            # For INVALID zones (contain "invalid_pattern"), keep pattern-based (needed for pattern enforcement)
            is_valid_zone = "invalid_pattern" not in zone_name
            if is_valid_zone:
                if strategy_name == "pattern_based":
                    logging.debug(f"Skipping pattern_based for valid zone: {zone_name}")
                    continue
            
            try:
                if cell.column_idx == 9:
                    logging.debug(f"Generating candidates for cell {cell.coordinates} using {strategy_name}")
                candidates = corrector.generate_candidates(cell, table)
                n_candidates_per_corrector[strategy_name] += len(candidates)
                pool = CandidatePool.get_instance()
                for correction_value, pool_key in candidates.items():
                    # Retrieve actual candidate from pool
                    candidate = pool.get_candidate(pool_key) if isinstance(pool_key, tuple) else pool_key
                    if correction_value in all_candidates:
                        existing_key = all_candidates[correction_value]
                        existing = pool.get_candidate(existing_key) if isinstance(existing_key, tuple) else existing_key
                        existing.features.update(candidate.features)
                    else:
                        all_candidates[correction_value] = pool_key

            except Exception as e:
                logging.warning(f"Error in {strategy_name}: {e}")
                continue

        self._total_cells_processed += 1
        return all_candidates, n_candidates_per_corrector

    def correct_zone(self, zone, lake):
        """Apply correction to all error cells in a zone.
        
        For invalid pattern zones:
        1. Attempt pattern enforcement on each cell
        2. Apply configured mode:
           - check: accept only when proposal matches ground truth
           - always_accept: accept top proposal unconditionally
           - disabled: skip pattern enforcement stage
        3. Move accepted cells to samples for classifier training
        
        For valid pattern zones:
        1. Skip pattern-based candidates (pattern already verified)
        2. Generate domain/vicinity/value-based candidates
        3. Train classifier (may include cells from incorrect pattern enforcement + correct pattern enforcement cells as training data)
        
        Args:
            zone: Zone object containing cells to process
            lake: Lake object with tables
            zones_dict: Dictionary of all zones for applying zone transitions (required for invalid pattern zones)
        """
        error_cells = [cell for cell in zone.cells.values() if cell.is_error]
        total_candidates = 0
        zone_sample_count = len(zone.samples)

        # Step 1: Pattern enforcement for invalid pattern zones
        n_pattern_enforced = 0
        n_pattern_correct = 0
        n_pattern_incorrect = 0
        samples_to_add = []
        if "invalid_pattern" in zone.name and self.pattern_enforcement_mode != "disabled":
            logging.info(
                f"Zone '{zone.name}': Attempting pattern enforcement on {len(error_cells)} cells "
                f"(mode={self.pattern_enforcement_mode})"
            )
            self.update_with_labeled_samples(
                list(zone.samples.values()), tables=lake.tables
            )
            # Get pattern-based corrector if available
            pattern_corrector = None
            for corrector in self.correctors:
                if corrector.get_strategy_name() == "pattern_based":
                    pattern_corrector = corrector
                    break
            
            if pattern_corrector and hasattr(pattern_corrector, "enforce_pattern_on_invalid_zone"):
                cells_by_table = defaultdict(list)
                for cell in error_cells:
                    cells_by_table[cell.table_id].append(cell)
                
                for table_id, table_cells in cells_by_table.items():
                    table = lake.tables[table_id]
                    
                    # Use batch enforcement if available for performance with large cell counts
                    if hasattr(pattern_corrector, "enforce_pattern_batch") and len(table_cells) > 100:
                        logging.info(f"Using batch pattern enforcement for {len(table_cells)} cells")
                        batch_results = pattern_corrector.enforce_pattern_batch(table_cells, table)
                        
                        for cell in table_cells:
                            if cell.coordinates in zone.samples:
                                continue
                            
                            if cell.row_idx in batch_results:
                                n_pattern_enforced += 1
                                pattern_candidates = batch_results[cell.row_idx]
                                best_pattern_value = list(pattern_candidates.keys())[0]
                                accept = False
                                if self.pattern_enforcement_mode == "always_accept":
                                    accept = True
                                else:  # "check"
                                    if not (hasattr(cell, "ground_truth") and cell.ground_truth is not None):
                                        logging.error(
                                            f"Cell {cell.coordinates} in invalid zone has no ground truth - cannot validate pattern enforcement in check mode"
                                        )
                                        raise ValueError(
                                            f"Ground truth required for pattern enforcement check mode in invalid zones. Cell: {cell.coordinates}"
                                        )
                                    accept = best_pattern_value == cell.ground_truth

                                if accept:
                                    n_pattern_correct += 1
                                    pool = CandidatePool.get_instance()
                                    candidate_pool_key = pattern_candidates[best_pattern_value]
                                    candidate_obj = pool.get_candidate(candidate_pool_key) if isinstance(candidate_pool_key, tuple) else candidate_pool_key
                                    cell.predicted_corrections = {
                                        "candidate": best_pattern_value,
                                        "confidence": candidate_obj.features.get("pattern_enforcement_reliability", 0),
                                        "source": "pattern_enforcement_correct"
                                    }
                                    samples_to_add.append(cell)
                                    logging.debug(
                                        f"Pattern enforcement accepted for cell {cell.coordinates}, "
                                        "moving to samples"
                                    )
                                else:
                                    n_pattern_incorrect += 1
                                    logging.debug(
                                        f"Pattern enforcement rejected for cell {cell.coordinates}, "
                                        "proceeding to candidate generation"
                                    )
                    else:
                        # Fall back to per-cell enforcement for small cell counts
                        for cell in table_cells:
                            if cell.coordinates in zone.samples:
                                # Already a sample, skip pattern enforcement
                                continue
                            pattern_candidates, _ = pattern_corrector.enforce_pattern_on_invalid_zone(cell, table)
                            if pattern_candidates:
                                n_pattern_enforced += 1
                                best_pattern_value = list(pattern_candidates.keys())[0]
                                accept = False
                                if self.pattern_enforcement_mode == "always_accept":
                                    accept = True
                                else:  # "check"
                                    if not (hasattr(cell, "ground_truth") and cell.ground_truth is not None):
                                        logging.error(
                                            f"Cell {cell.coordinates} in invalid zone has no ground truth - cannot validate pattern enforcement in check mode"
                                        )
                                        raise ValueError(
                                            f"Ground truth required for pattern enforcement check mode in invalid zones. Cell: {cell.coordinates}"
                                        )
                                    accept = best_pattern_value == cell.ground_truth

                                if accept:
                                    n_pattern_correct += 1
                                    pool = CandidatePool.get_instance()
                                    candidate_pool_key = pattern_candidates[best_pattern_value]
                                    candidate_obj = pool.get_candidate(candidate_pool_key) if isinstance(candidate_pool_key, tuple) else candidate_pool_key
                                    cell.predicted_corrections = {
                                        "candidate": best_pattern_value,
                                        "confidence": candidate_obj.features.get("pattern_enforcement_reliability", 0),
                                        "source": "pattern_enforcement_correct"
                                    }
                                    samples_to_add.append(cell)
                                    logging.debug(
                                        f"Pattern enforcement accepted for cell {cell.coordinates}, "
                                        "moving to samples"
                                    )
                                else:
                                    n_pattern_incorrect += 1
                                    logging.debug(
                                        f"Pattern enforcement rejected for cell {cell.coordinates}, "
                                        "proceeding to candidate generation"
                                    )
                            
                # Re-fetch error cells after potential moves
                error_cells = [cell for cell in zone.cells.values() if cell.is_error]
                logging.info(
                    f"Zone '{zone.name}': Pattern enforcement - "
                    f"{n_pattern_correct} accepted, {n_pattern_incorrect} rejected, "
                    f"{n_pattern_enforced} attempts"
                )
        elif "invalid_pattern" in zone.name:
            logging.info(
                f"Zone '{zone.name}': Pattern enforcement disabled by config "
                "(correction.pattern_enforcement_mode=disabled). "
                f"Will still use {zone_sample_count} labeled samples for candidate generation/classifier."
            )
        
        for cell in samples_to_add:
            zone.samples[cell.coordinates] = cell

        # Always update correctors with currently available labeled samples,
        # including in "disabled" mode (no pattern-enforcement stage).
        self.update_with_labeled_samples(
            list(zone.samples.values()), tables=lake.tables
        )
        cells_by_table = defaultdict(list)
        for cell in error_cells:
            cells_by_table[cell.table_id].append(cell)
        n_candidates_per_corrector_zone = defaultdict(int)
        
        # Process each table's cells together
        for table_id, table_cells in cells_by_table.items():
            logging.info(
                f"Processing {len(table_cells)} error cells in table {table_id} for zone '{zone.name}'"
            )
            table = lake.tables[table_id]

            for cell in table_cells:
                candidates, n_candidates_per_corrector = (
                    self.generate_candidates_for_cell(zone.name, cell, table)
                )
                
                cell.candidates = candidates
                total_candidates += len(candidates)
                for key, value in n_candidates_per_corrector.items():
                    n_candidates_per_corrector_zone[key] += (
                        value if key in n_candidates_per_corrector else 0
                    )

        logging.info(
            f"Zone '{zone.name}': Generated {total_candidates} candidates across {len(error_cells)} cells"
        )
        logging.info(
            f"  Candidates per corrector: {dict(n_candidates_per_corrector_zone)}"
        )

        return {
            "n_cells_processed": len(error_cells),
            "n_pattern_enforced": n_pattern_enforced,
            "n_pattern_correct": n_pattern_correct,
            "n_pattern_incorrect": n_pattern_incorrect,
            "n_candidates_generated": total_candidates,
            "strategies_used": [c.get_strategy_name() for c in self.correctors],
        }

    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {
            "total_cells_processed": self._total_cells_processed,
        }

        return stats
