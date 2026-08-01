import csv
import html
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import config

CellCoord = Tuple[int, int]
_PROFILE_CACHE: Dict[str, "TableErrorProfile"] = {}


def normalize_value(value: str) -> str:
    """
    Minimally normalize a cell value before comparison.
    Matches count_errors.py / generate_isolated_error_provenance.py behavior.
    """
    value = html.unescape(value)
    value = re.sub(r"[\t\n ]+", " ", value, flags=re.UNICODE)
    return value.strip("\t\n ")


def values_equal(dirty_value: str, clean_value: str) -> bool:
    return normalize_value(dirty_value) == normalize_value(clean_value)


def is_gt_mode() -> bool:
    return config.ERROR_MODE.upper() == "GT"


def detected_errors_filename() -> str:
    name = config.DETECTED_ERRORS_FILENAME or 'detected_errors.csv'
    return str(name)


def detected_errors_path(tab_path: Path) -> Path:
    """Resolve the per-table detected-errors CSV for the configured layout."""
    filename = detected_errors_filename()
    if config.DETECTED_ERRORS_DIR:
        return Path(config.DETECTED_ERRORS_DIR) / tab_path.name / filename
    return tab_path / filename


def load_detected_errors(tab_path: Path) -> Set[CellCoord]:
    """
    Load detected error coordinates from a per-table CSV with header:
        row,column
        0,0
        0,1
        ...
    """
    errors_path = detected_errors_path(tab_path)
    if not errors_path.exists():
        raise FileNotFoundError(
            f"ERROR_MODE is DETECTED but {errors_path} was not found"
        )

    errors: Set[CellCoord] = set()
    with open(errors_path, encoding="latin1", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return errors
        for row in reader:
            row_idx = int(row["row"])
            col_idx = int(row["column"])
            errors.add((row_idx, col_idx))
    return errors


def load_clean_data(tab_path: Path) -> List[List[str]]:
    with open(tab_path / "clean.csv", encoding="latin1") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        return [list(col) for col in zip(*list(csv_reader))]


class TableErrorProfile:
    """Per-table view of which cells are clean vs erroneous."""

    def __init__(
        self,
        tab_path: Path,
        clean_data: Optional[List[List[str]]] = None,
        detected_errors: Optional[Set[CellCoord]] = None,
    ):
        self.tab_path = tab_path
        if is_gt_mode():
            if clean_data is None:
                clean_data = load_clean_data(tab_path)
            self._clean_data = clean_data
            self._detected_errors = None
        else:
            if detected_errors is None:
                detected_errors = load_detected_errors(tab_path)
            self._detected_errors = detected_errors
            self._clean_data = None

    def is_clean(
        self,
        row_id: int,
        col_id: int,
        dirty_value: Optional[str] = None,
        clean_value: Optional[str] = None,
    ) -> bool:
        if is_gt_mode():
            if dirty_value is None:
                raise ValueError("dirty_value is required in GT mode")
            if clean_value is None:
                if (
                    self._clean_data is None
                    or col_id < 0
                    or col_id >= len(self._clean_data)
                    or row_id < 0
                    or row_id >= len(self._clean_data[col_id])
                ):
                    return False
                clean_value = self._clean_data[col_id][row_id]
            return values_equal(dirty_value, clean_value)
        return (row_id, col_id) not in self._detected_errors

    def is_error(
        self,
        row_id: int,
        col_id: int,
        dirty_value: Optional[str] = None,
        clean_value: Optional[str] = None,
    ) -> bool:
        return not self.is_clean(row_id, col_id, dirty_value, clean_value)


def get_error_profile(
    tab_path: Path,
    clean_data: Optional[List[List[str]]] = None,
) -> TableErrorProfile:
    # Include mode + detection path so GT/DETECTED (and different detection files)
    # never share a stale cache entry across index / discovery / merge.
    mode = config.ERROR_MODE.upper()
    if mode == "GT":
        key = f"GT::{tab_path.resolve()}"
    else:
        key = f"DETECTED::{tab_path.resolve()}::{detected_errors_path(tab_path)}"
    if key not in _PROFILE_CACHE:
        _PROFILE_CACHE[key] = TableErrorProfile(tab_path, clean_data=clean_data)
    elif clean_data is not None and is_gt_mode():
        _PROFILE_CACHE[key]._clean_data = clean_data
    return _PROFILE_CACHE[key]


def clear_error_profile_cache() -> None:
    _PROFILE_CACHE.clear()
