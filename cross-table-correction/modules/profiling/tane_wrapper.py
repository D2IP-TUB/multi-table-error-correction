"""
TANE wrapper module for functional dependency discovery.

This module provides an integration layer for the TANE algorithm
(Huhtala et al. 1999) to discover functional dependencies in tabular data.
The actual TANE implementation lives in tane/tane.py.
"""

import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_tane_imported = False
_TANE_cls = None
_PPattern_cls = None


def _ensure_tane_imported(tane_repo_path: Optional[str] = None):
    """Import TANE classes, adding the tane package's parent to sys.path if needed."""
    global _tane_imported, _TANE_cls, _PPattern_cls
    if _tane_imported:
        return _TANE_cls, _PPattern_cls

    if tane_repo_path:
        parent = str(Path(tane_repo_path).parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

    from tane.tane import TANE, PPattern
    _TANE_cls = TANE
    _PPattern_cls = PPattern
    _tane_imported = True
    return TANE, PPattern


class TANEWrapper:
    """
    Wrapper class for TANE algorithm integration.

    TANE (TAble-based decomposition discovering functional dependencies) is used to
    discover functional dependencies in the clean data, which are then attached to
    column profiles as contextual dependencies.
    """

    def __init__(self, tane_repo_path: str = None):
        """
        Initialize TANEWrapper.

        Args:
            tane_repo_path: Path to the TANE repository directory (e.g. ``<project>/tane``)
        """
        self.tane_repo_path = tane_repo_path
        self.discovered_fds = {}
        self._TANE = None
        self._PPattern = None
        logger.info(f"TANEWrapper initialized with repo path: {tane_repo_path}")

    def _get_tane(self):
        """Lazily import and cache TANE classes."""
        if self._TANE is None:
            self._TANE, self._PPattern = _ensure_tane_imported(self.tane_repo_path)
        return self._TANE, self._PPattern

    def add_functional_dependencies_to_profiles(self, profiles: Dict, lake) -> None:
        """
        Discover functional dependencies from clean data and attach to column profiles.

        Args:
            profiles: Dictionary of ColumnProfile objects keyed by (table_id, col_idx)
            lake: Lake object containing table data
        """
        try:
            self._get_tane()
        except ImportError:
            logger.error(
                "TANE algorithm not available — skipping FD discovery. "
                "Install fca==3.2 to enable."
            )
            return

        try:
            logger.info("Starting functional dependency discovery via TANE...")

            for table_id, table in lake.tables.items():
                logger.debug(f"Processing table {table_id} for FD discovery")

                clean_rows, col_indices = self._extract_clean_rows(table)

                if len(clean_rows) < 2 or len(col_indices) < 2:
                    logger.debug(
                        f"Table {table_id}: insufficient clean data for FD discovery "
                        f"({len(clean_rows)} rows, {len(col_indices)} columns)"
                    )
                    continue

                table_fds = self._discover_functional_dependencies(
                    clean_rows, col_indices
                )

                self._attach_fds_to_profiles(table_id, table, table_fds, profiles)

                self.discovered_fds[table_id] = table_fds
                logger.debug(f"Discovered {len(table_fds)} FDs for table {table_id}")

            total = sum(len(fds) for fds in self.discovered_fds.values())
            logger.info(
                f"Functional dependency discovery completed: "
                f"{total} FDs across {len(self.discovered_fds)} tables"
            )

        except Exception as e:
            logger.error(f"Error discovering functional dependencies: {e}")

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_clean_rows(
        self, table
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Extract row-aligned clean data from a table.

        Only rows where **every** cell is clean are included, since TANE
        requires aligned tabular data.

        Returns:
            (clean_rows, col_indices) — clean_rows is a list of rows (each a
            list of string values), col_indices is the ordered column indices.
        """
        if not hasattr(table, "columns") or not table.columns:
            return [], []

        n_cols = len(table.columns)
        col_indices = list(range(n_cols))

        n_rows = (
            len(table.columns[0].cells)
            if hasattr(table.columns[0], "cells")
            else 0
        )

        clean_rows: List[List[str]] = []
        for row_idx in range(n_rows):
            row_values = []
            row_clean = True
            for col_idx in range(n_cols):
                col = table.columns[col_idx]
                if not hasattr(col, "cells") or row_idx >= len(col.cells):
                    row_clean = False
                    break
                cell = col.cells[row_idx]
                if cell.is_error:
                    row_clean = False
                    break
                row_values.append(str(cell.value).strip())

            if row_clean:
                clean_rows.append(row_values)

        return clean_rows, col_indices

    # ------------------------------------------------------------------
    # TANE integration
    # ------------------------------------------------------------------

    def _build_partitions(self, clean_rows: List[List[str]], n_cols: int):
        """
        Convert clean row data into stripped partitions (the format TANE expects).

        For each column the rows are grouped by value into equivalence classes;
        singleton classes are then stripped (``PPattern.fix_desc``).
        """
        _, PPattern = self._get_tane()

        hashes: Dict[int, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        for row_idx, row in enumerate(clean_rows):
            for col_idx, value in enumerate(row):
                hashes[col_idx][value].add(row_idx)

        partitions = []
        for col_idx in range(n_cols):
            equiv_classes = list(hashes[col_idx].values())
            partitions.append(PPattern.fix_desc(equiv_classes))

        return partitions

    def _discover_functional_dependencies(
        self, clean_rows: List[List[str]], col_indices: List[int]
    ) -> Set[Tuple]:
        """
        Run the TANE algorithm on clean row data and return discovered FDs.

        Returns:
            Set of ``(lhs_cols_tuple, rhs_col)`` where column indices refer to
            the original table columns.
        """
        TANE, _ = self._get_tane()

        n_cols = len(col_indices)
        partitions = self._build_partitions(clean_rows, n_cols)

        t0 = time.time()
        tane = TANE(partitions)
        tane.run()
        elapsed = time.time() - t0

        fds: Set[Tuple] = set()
        for lhs, rhs in tane.rules:
            lhs_cols = tuple(col_indices[i] for i in lhs) if lhs else ()
            rhs_col = col_indices[rhs]
            fds.add((lhs_cols, rhs_col))

        logger.debug(
            f"TANE found {len(fds)} FDs from {len(clean_rows)} rows x "
            f"{n_cols} cols in {elapsed:.3f}s"
        )
        return fds

    # ------------------------------------------------------------------
    # Attach results to profiles
    # ------------------------------------------------------------------

    def _attach_fds_to_profiles(
        self, table_id: str, table, table_fds: Set[Tuple], profiles: Dict
    ) -> None:
        """
        Attach discovered FDs to column profiles.

        Args:
            table_id: ID of the table
            table: Table object
            table_fds: Set of discovered functional dependencies
            profiles: Dictionary of ColumnProfile objects
        """
        for lhs_cols, rhs_col in table_fds:
            profile_key = (table_id, rhs_col)

            if profile_key in profiles:
                profile = profiles[profile_key]

                if profile.functional_dependencies is None:
                    profile.functional_dependencies = []
                if profile.fd_confidence_scores is None:
                    profile.fd_confidence_scores = {}

                profile.functional_dependencies.append(lhs_cols)
                profile.fd_confidence_scores[(lhs_cols, rhs_col)] = 1.0

                logger.debug(
                    f"Added FD {lhs_cols} -> {rhs_col} to profile "
                    f"({table_id}, {rhs_col})"
                )


def setup_tane_integration(config=None) -> TANEWrapper:
    """
    Set up TANE integration.

    Args:
        config: Configuration object (optional)

    Returns:
        Initialized TANEWrapper instance
    """
    tane_repo_path = None

    if config and hasattr(config, "directories"):
        tane_repo_path = getattr(config.directories, "sandbox_dir", None)

    return TANEWrapper(tane_repo_path=tane_repo_path)
