"""
FD-based merge validation: score joins and unions by newly correctable source cells.

Source-cell identity is always (table_name, source_row, source_col) parsed from provenance
``table_name § col_id § row_id``.

Score (v1):
    max(0, |correctable_after - correctable_before| / |errors_in_merged|)

Before: all unary FDs discovered on each full source table in isolation.
After:  all unary FDs rediscovered on the full merged table.
Each candidate FD X → Y uses its own evidence rows: rows where X and Y are
both clean (other columns may be erroneous). FDs can disappear or appear
across the merge; only newly correctable cells count. Correction witnesses
may use any row where both attributes are present; scored errors are all
erroneous source cells appearing anywhere in the merged output.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import config
from error_cells import get_error_profile, normalize_value

SourceCell = Tuple[str, int, int]  # (table_name, source_row, source_col)
CanonicalJoinFD = Tuple[str, int, str, int]  # (det_table, det_col, dep_table, dep_col)
CanonicalUnionFD = Tuple[int, int]  # (merged_det_col, merged_dep_col)
FdKey = CanonicalJoinFD | CanonicalUnionFD


@dataclass
class FDDiff:
    before: Set[FdKey]
    after: Set[FdKey]
    introduced: Set[FdKey]
    broken: Set[FdKey]
    retained: Set[FdKey]
    before_per_table: Dict[str, Set[FdKey]]

    @staticmethod
    def empty() -> "FDDiff":
        return FDDiff(set(), set(), set(), set(), set(), {})

    def counts(self) -> Dict[str, int]:
        return {
            "before": len(self.before),
            "after": len(self.after),
            "introduced": len(self.introduced),
            "broken": len(self.broken),
            "retained": len(self.retained),
        }


@dataclass
class SourceTableFDState:
    """Per-source-table data used for pre-merge FD discovery and correction baseline."""

    tab_name: str
    data: List[List[str]]
    present: List[Set[int]]
    errors: List[Set[int]]
    source_at: List[List[Optional[SourceCell]]]
    fds: List[Tuple[int, int]]


@dataclass
class ValidationResult:
    score: float
    errors_in_merged: int
    correctable_before: int
    correctable_after: int
    newly_correctable: int
    correctable_lost: int = 0
    fd_diff: FDDiff = field(default_factory=FDDiff.empty)
    operation: str = ""
    logical_col_names: List[str] = field(default_factory=list)
    distribution_passed: bool = True
    distribution_max_tvd: float = 0.0
    distribution_max_ks: float = 0.0
    distribution_checks: List[dict] = field(default_factory=list)

    @property
    def errors_in_scope(self) -> int:
        """Deprecated alias for errors_in_merged."""
        return self.errors_in_merged

    def summary(self) -> str:
        fd = self.fd_diff.counts()
        base = (
            f"errors_in_merged={self.errors_in_merged} "
            f"correctable_before={self.correctable_before} "
            f"correctable_after={self.correctable_after} "
            f"newly_correctable={self.newly_correctable} "
            f"correctable_lost={self.correctable_lost} "
            f"validation_score={self.score:.4f} "
            f"fds_before={fd['before']} fds_after={fd['after']} "
            f"fds_introduced={fd['introduced']} fds_broken={fd['broken']} "
            f"fds_retained={fd['retained']}"
        )
        if self.distribution_checks or not self.distribution_passed:
            base += (
                f" distribution_passed={self.distribution_passed}"
                f" max_tvd={self.distribution_max_tvd:.4f}"
                f" max_ks={self.distribution_max_ks:.4f}"
            )
        return base


def format_join_fd(fd: CanonicalJoinFD) -> str:
    det_tab, det_col, dep_tab, dep_col = fd
    return f"{det_tab}::{det_col} -> {dep_tab}::{dep_col}"


def format_union_fd(fd: CanonicalUnionFD, logical_col_names: List[str]) -> str:
    x, y = fd
    nx = logical_col_names[x] if 0 <= x < len(logical_col_names) else f"col[{x}]"
    ny = logical_col_names[y] if 0 <= y < len(logical_col_names) else f"col[{y}]"
    return f"{nx} -> {ny}"


def format_fd_key(fd: FdKey, operation: str, logical_col_names: List[str]) -> str:
    if operation == "union":
        return format_union_fd(fd, logical_col_names)  # type: ignore[arg-type]
    return format_join_fd(fd)  # type: ignore[arg-type]


def canonicalize_join_fds(
    fds: List[Tuple[int, int]],
    col_sources: List[Tuple[str, int]],
) -> Set[CanonicalJoinFD]:
    return {
        (col_sources[x][0], col_sources[x][1], col_sources[y][0], col_sources[y][1])
        for x, y in fds
    }


def canonicalize_union_fds(fds: List[Tuple[int, int]]) -> Set[CanonicalUnionFD]:
    return {(x, y) for x, y in fds}


def parse_union_logical_column_names(header: List[str]) -> List[str]:
    """Logical attribute name for each merged union column (index = merged col id)."""
    names: List[str] = []
    for h in header:
        if not h:
            names.append("")
            continue
        first_segment = h.split(" | ", 1)[0]
        parts = first_segment.split("::", 2)
        names.append(parts[2] if len(parts) == 3 else first_segment)
    return names


def build_union_source_col_map(header: List[str]) -> Dict[Tuple[str, int], int]:
    """Map each source (table, col_id) to its merged union column index."""
    source_map: Dict[Tuple[str, int], int] = {}
    for merged_idx, h in enumerate(header):
        if not h:
            continue
        for segment in h.split(" | "):
            parts = segment.split("::", 2)
            if len(parts) != 3:
                continue
            source_map[(parts[0], int(parts[1]))] = merged_idx
    return source_map


def compare_fds(
    before: Set[FdKey],
    after: Set[FdKey],
    before_per_table: Dict[str, Set[FdKey]],
) -> FDDiff:
    return FDDiff(
        before=before,
        after=after,
        introduced=after - before,
        broken=before - after,
        retained=before & after,
        before_per_table=before_per_table,
    )


def is_empty_value(value: str) -> bool:
    return normalize_value(value) == ""


def parse_provenance(prov: str) -> Optional[SourceCell]:
    if not prov:
        return None
    parts = prov.rsplit(" § ", 2)
    if len(parts) != 3:
        return None
    return parts[0], int(parts[2]), int(parts[1])


def parse_header(header: List[str]) -> Tuple[List[Tuple[str, int]], Set[str]]:
    col_sources: List[Tuple[str, int]] = []
    for h in header:
        parts = h.split("::")
        tab_name, orig_col_id = parts[0], int(parts[1])
        col_sources.append((tab_name, orig_col_id))
    return col_sources, table_names_from_header(header)


def table_names_from_header(header: List[str]) -> Set[str]:
    """All source table names referenced in join or union headers."""
    names: Set[str] = set()
    for h in header:
        if not h:
            continue
        for segment in h.split(" | "):
            parts = segment.split("::", 2)
            if parts:
                names.add(parts[0])
    return names


def load_table_dirty(tab_path) -> Tuple[List[str], List[List[str]]]:
    with open(tab_path / "dirty.csv", encoding="latin1") as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [list(col) for col in zip(*list(reader))]
    return header, data


def fd_evidence_rows(
    scope_rows: Set[int],
    present: List[Set[int]],
    errors: List[Set[int]],
    x: int,
    y: int,
) -> Set[int]:
    """Rows where both determinant x and dependent y are clean (in scope and present).

    Empty values are filtered later in fd_holds_on_evidence, matching correction.
    """
    return (
        (scope_rows & present[x] & present[y])
        - errors[x]
        - errors[y]
    )


def fd_holds_on_evidence(
    data: List[List[str]],
    x: int,
    y: int,
    evidence_rows: Set[int],
) -> bool:
    if len(evidence_rows) < 2:
        return False
    val_map: Dict[str, str] = {}
    x_col = data[x]
    y_col = data[y]
    usable = 0
    for r in evidence_rows:
        xv = x_col[r]
        yv = y_col[r]
        if is_empty_value(xv) or is_empty_value(yv):
            continue
        usable += 1
        prev = val_map.get(xv)
        if prev is None:
            val_map[xv] = yv
        elif prev != yv:
            return False
    return usable >= 2


def discover_unary_fds(
    data: List[List[str]],
    scope_rows: Set[int],
    present: List[Set[int]],
    errors: List[Set[int]],
    col_filter: Optional[Set[int]] = None,
) -> List[Tuple[int, int]]:
    """
    Discover all 1→1 FDs X → Y.

    For each pair (X, Y), evidence is built independently: rows in scope where
    both X and Y are clean. Rows with empty determinant or dependent values are
    excluded from the consistency check (same policy as correction).
    """
    num_cols = len(data)
    fds: List[Tuple[int, int]] = []
    for x in range(num_cols):
        if col_filter is not None and x not in col_filter:
            continue
        for y in range(num_cols):
            if x == y:
                continue
            if col_filter is not None and y not in col_filter:
                continue
            evidence = fd_evidence_rows(scope_rows, present, errors, x, y)
            if fd_holds_on_evidence(data, x, y, evidence):
                fds.append((x, y))
    return fds


def correctable_from_fds(
    fds: List[Tuple[int, int]],
    data: List[List[str]],
    present: List[Set[int]],
    errors: List[Set[int]],
    source_at: List[List[Optional[SourceCell]]],
) -> Set[SourceCell]:
    """
    Mark erroneous source cells correctable via FD witness groups.

    Rules (correction-oriented, stricter than FD discovery):
      - determinant must be clean in the row
      - skip empty determinant values
      - determinant group size >= 2
      - at least one clean, non-empty witness on the dependent
      - not all rows in the group may be erroneous on Y

    Witness rows are any merged row where both X and Y are present (not
    restricted to FD evidence rows); FD discovery itself uses the full table.
    """
    correctable: Set[SourceCell] = set()
    for x, y in fds:
        candidates = (present[y] & present[x]) - errors[x]
        det_groups: Dict[str, List[int]] = {}
        x_col = data[x]
        for r in candidates:
            xv = x_col[r]
            if is_empty_value(xv):
                continue
            det_groups.setdefault(xv, []).append(r)

        y_errs = errors[y]
        y_col = data[y]
        for group_rows in det_groups.values():
            if len(group_rows) < 2:
                continue
            has_witness = any(
                r not in y_errs and not is_empty_value(y_col[r]) for r in group_rows
            )
            if not has_witness:
                continue
            errs_in_group = y_errs.intersection(group_rows)
            if not errs_in_group or len(errs_in_group) == len(group_rows):
                continue
            for r in errs_in_group:
                src = source_at[y][r]
                if src is not None:
                    correctable.add(src)
    return correctable


def source_table_fd_state(tab_name: str) -> Optional[SourceTableFDState]:
    """Load one source table and discover unary FDs on the full table."""
    tab_path = config.DIR_PATH / tab_name
    profile = get_error_profile(tab_path)
    _, data = load_table_dirty(tab_path)
    if not data:
        return None

    num_cols = len(data)
    num_rows = len(data[0])
    present = [set(range(num_rows)) for _ in range(num_cols)]
    errors = [set() for _ in range(num_cols)]
    source_at: List[List[Optional[SourceCell]]] = [
        [(tab_name, r, c) for r in range(num_rows)] for c in range(num_cols)
    ]
    for c in range(num_cols):
        for r in range(num_rows):
            if profile.is_error(r, c, dirty_value=data[c][r]):
                errors[c].add(r)

    all_rows = set(range(num_rows))
    fds = discover_unary_fds(data, all_rows, present, errors)
    return SourceTableFDState(tab_name, data, present, errors, source_at, fds)


def pre_merge_source_states(table_names: Set[str]) -> Dict[str, SourceTableFDState]:
    """One load + FD discovery pass per source table (shared by diff and baseline)."""
    states: Dict[str, SourceTableFDState] = {}
    for tab_name in sorted(table_names):
        state = source_table_fd_state(tab_name)
        if state is not None:
            states[tab_name] = state
    return states


def fds_before_join_from_states(
    table_names: Set[str],
    states: Dict[str, SourceTableFDState],
) -> Tuple[Set[CanonicalJoinFD], Dict[str, Set[CanonicalJoinFD]]]:
    """All unary FDs per source table in isolation (join canonical form)."""
    union: Set[CanonicalJoinFD] = set()
    per_table: Dict[str, Set[CanonicalJoinFD]] = {}
    for tab_name in sorted(table_names):
        state = states.get(tab_name)
        if state is None:
            per_table[tab_name] = set()
            continue
        can = {(tab_name, x, tab_name, y) for x, y in state.fds}
        per_table[tab_name] = can
        union |= can
    return union, per_table


def fds_before_union_from_states(
    table_names: Set[str],
    header: List[str],
    states: Dict[str, SourceTableFDState],
) -> Tuple[Set[CanonicalUnionFD], Dict[str, Set[CanonicalUnionFD]]]:
    """
    Unary FDs per source table, mapped to merged logical column indices.

    Aligned attributes from L and R collapse to the same (det_col, dep_col) pair.
    """
    source_to_merged = build_union_source_col_map(header)
    union: Set[CanonicalUnionFD] = set()
    per_table: Dict[str, Set[CanonicalUnionFD]] = {}

    for tab_name in sorted(table_names):
        state = states.get(tab_name)
        if state is None:
            per_table[tab_name] = set()
            continue

        logical: Set[CanonicalUnionFD] = set()
        for x, y in state.fds:
            merged_x = source_to_merged.get((tab_name, x))
            merged_y = source_to_merged.get((tab_name, y))
            if merged_x is not None and merged_y is not None:
                logical.add((merged_x, merged_y))
        per_table[tab_name] = logical
        union |= logical
    return union, per_table


def discover_fds_before_join(
    table_names: Set[str],
) -> Tuple[Set[CanonicalJoinFD], Dict[str, Set[CanonicalJoinFD]]]:
    return fds_before_join_from_states(table_names, pre_merge_source_states(table_names))


def discover_fds_before_union(
    table_names: Set[str],
    header: List[str],
) -> Tuple[Set[CanonicalUnionFD], Dict[str, Set[CanonicalUnionFD]]]:
    return fds_before_union_from_states(
        table_names, header, pre_merge_source_states(table_names)
    )


def correctable_before_merge(
    states: Dict[str, SourceTableFDState],
    merged_errors: Set[SourceCell],
) -> Set[SourceCell]:
    """
    Pre-merge baseline: unary FDs on each full source table in isolation.

    Uses the same per-table FD discovery as fds_before_*.
    """
    baseline: Set[SourceCell] = set()
    for state in states.values():
        tab_correctable = correctable_from_fds(
            state.fds, state.data, state.present, state.errors, state.source_at
        )
        baseline.update(tab_correctable & merged_errors)
    return baseline


def build_merged_state(
    data: List[List[str]],
    tracker: List[List[str]],
    header: List[str],
) -> Tuple[
    List[Tuple[str, int]],
    Set[str],
    Dict[str, object],
    List[Set[int]],
    List[Set[int]],
    List[List[Optional[SourceCell]]],
    Set[SourceCell],
]:
    col_sources, table_names = parse_header(header)
    num_cols = len(data)
    num_rows = len(data[0]) if num_cols else 0

    error_profiles: Dict[str, object] = {}

    def profile_for(tab_name: str):
        if tab_name not in error_profiles:
            error_profiles[tab_name] = get_error_profile(config.DIR_PATH / tab_name)
        return error_profiles[tab_name]

    present = [set() for _ in range(num_cols)]
    errors = [set() for _ in range(num_cols)]
    source_at: List[List[Optional[SourceCell]]] = [
        [None for _ in range(num_rows)] for _ in range(num_cols)
    ]

    for c in range(num_cols):
        for r in range(num_rows):
            prov = tracker[c][r]
            src = parse_provenance(prov)
            if src is None:
                continue
            present[c].add(r)
            source_at[c][r] = src
            if profile_for(src[0]).is_error(src[1], src[2], dirty_value=data[c][r]):
                errors[c].add(r)

    return (
        col_sources,
        table_names,
        error_profiles,
        present,
        errors,
        source_at,
    )


def all_merged_errors(
    errors: List[Set[int]],
    source_at: List[List[Optional[SourceCell]]],
) -> Set[SourceCell]:
    """All erroneous source cells that appear anywhere in the merged output."""
    err_cells: Set[SourceCell] = set()
    for c, col_errors in enumerate(errors):
        for r in col_errors:
            src = source_at[c][r]
            if src is not None:
                err_cells.add(src)
    return err_cells


def score_merge(
    data: List[List[str]],
    tracker: List[List[str]],
    header: List[str],
    operation: str,
    *,
    left_table: str = "",
    right_table: str = "",
    mapping: Optional[Dict[int, int]] = None,
) -> ValidationResult:
    if not config.MERGE_VALIDATION:
        return ValidationResult(1.0, 0, 0, 0, 0, 0, FDDiff.empty())

    scope = str(getattr(config, "VALIDATION_SCOPE", "all")).strip().lower()
    if scope not in {"all", "join", "union"}:
        raise ValueError(
            f"VALIDATION_SCOPE must be 'all', 'join', or 'union'; got {scope!r}"
        )
    if scope == "join" and operation == "union":
        return ValidationResult(1.0, 0, 0, 0, 0, 0, FDDiff.empty())
    if scope == "union" and operation == "join":
        return ValidationResult(1.0, 0, 0, 0, 0, 0, FDDiff.empty())

    strategy = str(config.VALIDATION_STRATEGY).strip().lower()
    if strategy not in {"fd", "distribution", "fd_and_distribution"}:
        raise ValueError(
            f"VALIDATION_STRATEGY must be 'fd', 'distribution', or 'fd_and_distribution'; "
            f"got {config.VALIDATION_STRATEGY!r}"
        )

    fd_result: Optional[ValidationResult] = None
    if strategy in {"fd", "fd_and_distribution"}:
        fd_result = _score_merge_fd(data, tracker, header, operation)

    dist_result = None
    if strategy in {"distribution", "fd_and_distribution"}:
        from merge_distribution_validation import (
            DistributionValidationResult,
            validate_merge_distribution,
        )

        if left_table and right_table and mapping is not None:
            dist_result = validate_merge_distribution(
                left_table,
                right_table,
                mapping,
                operation,
                data=data,
                tracker=tracker,
            )
        else:
            dist_result = DistributionValidationResult(passed=False)

    if strategy == "distribution":
        return _validation_result_from_distribution(dist_result)

    assert fd_result is not None
    if dist_result is not None:
        _apply_distribution_result(fd_result, dist_result, gate=strategy == "fd_and_distribution")
    return fd_result


def _validation_result_from_distribution(dist_result) -> ValidationResult:
    from merge_distribution_validation import DistributionValidationResult

    assert isinstance(dist_result, DistributionValidationResult)
    return ValidationResult(
        score=1.0 if dist_result.passed else 0.0,
        errors_in_merged=0,
        correctable_before=0,
        correctable_after=0,
        newly_correctable=0,
        correctable_lost=0,
        distribution_passed=dist_result.passed,
        distribution_max_tvd=dist_result.max_tvd,
        distribution_max_ks=dist_result.max_ks,
        distribution_checks=[check.to_row() for check in dist_result.checks],
    )


def _apply_distribution_result(
    result: ValidationResult,
    dist_result,
    *,
    gate: bool,
) -> None:
    from merge_distribution_validation import DistributionValidationResult

    assert isinstance(dist_result, DistributionValidationResult)
    result.distribution_passed = dist_result.passed
    result.distribution_max_tvd = dist_result.max_tvd
    result.distribution_max_ks = dist_result.max_ks
    result.distribution_checks = [check.to_row() for check in dist_result.checks]
    if gate and not dist_result.passed:
        result.score = 0.0


def _score_merge_fd(
    data: List[List[str]],
    tracker: List[List[str]],
    header: List[str],
    operation: str,
) -> ValidationResult:
    logical_col_names = (
        parse_union_logical_column_names(header) if operation == "union" else []
    )

    def _result(
        score: float,
        errors: int,
        before: int,
        after: int,
        newly: int,
        lost: int,
        fd_diff: FDDiff,
    ) -> ValidationResult:
        return ValidationResult(
            score,
            errors,
            before,
            after,
            newly,
            lost,
            fd_diff,
            operation=operation,
            logical_col_names=logical_col_names,
        )

    num_cols = len(data)
    if num_cols == 0 or len(data[0]) == 0:
        return _result(0.0, 0, 0, 0, 0, 0, FDDiff.empty())

    (
        col_sources,
        table_names,
        _profiles,
        present,
        errors,
        source_at,
    ) = build_merged_state(data, tracker, header)

    num_rows = len(data[0])
    all_rows = set(range(num_rows))
    merged_errors = all_merged_errors(errors, source_at)
    source_states = pre_merge_source_states(table_names)

    if operation == "join":
        if len(table_names) < 2:
            return _result(0.0, 0, 0, 0, 0, 0, FDDiff.empty())
        fds_before, fds_before_per_table = fds_before_join_from_states(table_names, source_states)
    elif operation == "union":
        fds_before, fds_before_per_table = fds_before_union_from_states(
            table_names, header, source_states
        )
    else:
        raise ValueError(f"Unknown operation: {operation}")

    fds_after_idx = discover_unary_fds(data, all_rows, present, errors)
    if operation == "union":
        fds_after = canonicalize_union_fds(fds_after_idx)
    else:
        fds_after = canonicalize_join_fds(fds_after_idx, col_sources)
    fd_diff = compare_fds(fds_before, fds_after, fds_before_per_table)

    if not merged_errors:
        return _result(0.0, 0, 0, 0, 0, 0, fd_diff)

    before_set = correctable_before_merge(source_states, merged_errors)

    if not fds_after_idx:
        return _result(
            0.0,
            len(merged_errors),
            len(before_set),
            0,
            0,
            len(before_set),
            fd_diff,
        )

    after_set = correctable_from_fds(fds_after_idx, data, present, errors, source_at) & merged_errors
    newly_set = after_set - before_set
    lost_set = before_set - after_set
    score = max(0.0, len(newly_set) / len(merged_errors))
    return _result(
        score,
        len(merged_errors),
        len(before_set),
        len(after_set),
        len(newly_set),
        len(lost_set),
        fd_diff,
    )


def validate_join(data, tracker, header) -> float:
    return score_merge(data, tracker, header, "join").score


def validate_union(data, tracker, header) -> float:
    return score_merge(data, tracker, header, "union").score


def blend_discovery_metrics(candidate: dict) -> Dict[str, float]:
    """
    BLEND discovery-time overlap metrics (used to shortlist candidates).

    Join: blend_score and blend_coverage are both the join tuple ratio in [0, 1].
    Union: blend_score is the sum of per-column overlap ratios (can exceed 1);
    blend_coverage is the fraction of aligned columns in [0, 1].
    """
    score = float(candidate["score"])
    if candidate["operation"] == "join":
        return {"blend_score": score, "blend_coverage": score}
    return {
        "blend_score": score,
        "blend_coverage": float(candidate.get("coverage") or 0.0),
    }


def validation_result_row(result: ValidationResult) -> Dict[str, object]:
    fd = result.fd_diff.counts()
    return {
        "validation_score": result.score,
        "errors_in_merged": result.errors_in_merged,
        "correctable_before": result.correctable_before,
        "correctable_after": result.correctable_after,
        "newly_correctable": result.newly_correctable,
        "correctable_lost": result.correctable_lost,
        "fds_before": fd["before"],
        "fds_after": fd["after"],
        "fds_introduced": fd["introduced"],
        "fds_broken": fd["broken"],
        "fds_retained": fd["retained"],
        "distribution_passed": result.distribution_passed,
        "distribution_max_tvd": f"{result.distribution_max_tvd:.6f}",
        "distribution_max_ks": f"{result.distribution_max_ks:.6f}",
    }


@dataclass
class CandidateReportRow:
    group_id: int
    candidate_idx: int
    operation: str
    left_table: str
    right_table: str
    blend_score: float
    blend_coverage: float
    validation_score: float
    selected: bool
    result: ValidationResult


def _fd_identity_fields(
    fd: FdKey,
    operation: str,
    logical_col_names: List[str],
) -> Dict[str, object]:
    if operation == "union":
        x, y = fd  # type: ignore[misc]
        return {
            "det_table": "",
            "det_col": x,
            "dep_table": "",
            "dep_col": y,
            "fd": format_union_fd((x, y), logical_col_names),
        }
    det_tab, det_col, dep_tab, dep_col = fd  # type: ignore[misc]
    return {
        "det_table": det_tab,
        "det_col": det_col,
        "dep_table": dep_tab,
        "dep_col": dep_col,
        "fd": format_join_fd((det_tab, det_col, dep_tab, dep_col)),
    }


def _sorted_fds(fds: Iterable[FdKey], operation: str) -> List[FdKey]:
    if operation == "union":
        return sorted(fds, key=lambda fd: (fd[0], fd[1]))  # type: ignore[index]
    return sorted(fds, key=lambda fd: (fd[0], fd[1], fd[2], fd[3]))  # type: ignore[index]


def append_candidate_report_rows(
    rows: List[dict],
    fd_rows: List[dict],
    distribution_rows: List[dict],
    entry: CandidateReportRow,
) -> None:
    rows.append({
        "group_id": entry.group_id,
        "candidate_idx": entry.candidate_idx,
        "operation": entry.operation,
        "left_table": entry.left_table,
        "right_table": entry.right_table,
        "blend_score": entry.blend_score,
        "blend_coverage": entry.blend_coverage,
        "validation_score": entry.validation_score,
        "selected": entry.selected,
        **validation_result_row(entry.result),
    })

    base = {
        "group_id": entry.group_id,
        "candidate_idx": entry.candidate_idx,
        "operation": entry.operation,
        "left_table": entry.left_table,
        "right_table": entry.right_table,
        "selected": entry.selected,
    }
    op = entry.result.operation or entry.operation
    logical_names = entry.result.logical_col_names
    for change_type, fds in (
        ("introduced", entry.result.fd_diff.introduced),
        ("broken", entry.result.fd_diff.broken),
        ("retained", entry.result.fd_diff.retained),
    ):
        for fd in _sorted_fds(fds, op):
            fd_rows.append({
                **base,
                "change_type": change_type,
                **_fd_identity_fields(fd, op, logical_names),
            })

    for tab_name, tab_fds in sorted(entry.result.fd_diff.before_per_table.items()):
        for fd in _sorted_fds(tab_fds, op):
            fd_rows.append({
                **base,
                "change_type": "before_per_table",
                **_fd_identity_fields(fd, op, logical_names),
                "source_table": tab_name,
            })

    for check in entry.result.distribution_checks:
        distribution_rows.append({
            "group_id": entry.group_id,
            "candidate_idx": entry.candidate_idx,
            "operation": entry.operation,
            "left_table": entry.left_table,
            "right_table": entry.right_table,
            "selected": entry.selected,
            **check,
        })


def write_validation_reports(
    candidate_rows: List[dict],
    fd_rows: List[dict],
    distribution_rows: List[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cand_path = output_dir / "validation_report.csv"
    cand_fields = [
        "group_id", "candidate_idx", "operation", "left_table", "right_table",
        "blend_score", "blend_coverage", "validation_score", "selected",
        "errors_in_merged", "correctable_before", "correctable_after",
        "newly_correctable", "correctable_lost",
        "fds_before", "fds_after", "fds_introduced", "fds_broken", "fds_retained",
        "distribution_passed", "distribution_max_tvd", "distribution_max_ks",
    ]
    with open(cand_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cand_fields)
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow({k: row.get(k, "") for k in cand_fields})

    fd_path = output_dir / "validation_fd_changes.csv"
    fd_fields = [
        "group_id", "candidate_idx", "operation", "left_table", "right_table", "selected",
        "change_type", "det_table", "det_col", "dep_table", "dep_col", "fd", "source_table",
    ]
    with open(fd_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fd_fields, extrasaction="ignore")
        writer.writeheader()
        for row in fd_rows:
            writer.writerow({k: row.get(k, "") for k in fd_fields})

    dist_path = output_dir / "validation_distribution_checks.csv"
    dist_fields = [
        "group_id", "candidate_idx", "operation", "left_table", "right_table", "selected",
        "check_kind", "left_col", "right_col", "value_type", "metric", "metric_value", "threshold",
        "passed", "left_clean_count", "right_clean_count",
    ]
    with open(dist_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dist_fields, extrasaction="ignore")
        writer.writeheader()
        for row in distribution_rows:
            writer.writerow({k: row.get(k, "") for k in dist_fields})

    summary_path = output_dir / "validation_summary.csv"
    summary_rows = _build_validation_summary_rows(candidate_rows, fd_rows)
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def _build_validation_summary_rows(
    candidate_rows: List[dict],
    fd_rows: List[dict],
) -> List[dict]:
    if not candidate_rows:
        return [{"metric": "candidates_evaluated", "value": 0}]

    selected = [r for r in candidate_rows if str(r.get("selected", "")).lower() in ("true", "1")]
    all_introduced = {r["fd"] for r in fd_rows if r.get("change_type") == "introduced"}
    all_broken = {r["fd"] for r in fd_rows if r.get("change_type") == "broken"}
    sel_introduced = {
        r["fd"] for r in fd_rows
        if r.get("change_type") == "introduced" and str(r.get("selected", "")).lower() in ("true", "1")
    }
    sel_broken = {
        r["fd"] for r in fd_rows
        if r.get("change_type") == "broken" and str(r.get("selected", "")).lower() in ("true", "1")
    }

    groups = {r["group_id"] for r in candidate_rows}
    rows = [
        {"metric": "comparison_groups", "value": len(groups)},
        {"metric": "candidates_evaluated", "value": len(candidate_rows)},
        {"metric": "candidates_selected", "value": len(selected)},
        {"metric": "total_fds_introduced_all_candidates", "value": len(all_introduced)},
        {"metric": "total_fds_broken_all_candidates", "value": len(all_broken)},
        {"metric": "total_fds_introduced_selected", "value": len(sel_introduced)},
        {"metric": "total_fds_broken_selected", "value": len(sel_broken)},
        {"metric": "sum_errors_in_merged_all", "value": sum(int(r.get("errors_in_merged") or 0) for r in candidate_rows)},
        {"metric": "sum_newly_correctable_all", "value": sum(int(r.get("newly_correctable") or 0) for r in candidate_rows)},
        {"metric": "sum_correctable_lost_all", "value": sum(int(r.get("correctable_lost") or 0) for r in candidate_rows)},
        {"metric": "sum_newly_correctable_selected", "value": sum(int(r.get("newly_correctable") or 0) for r in selected)},
        {
            "metric": "candidates_failed_distribution",
            "value": sum(
                1 for r in candidate_rows
                if str(r.get("distribution_passed", "true")).lower() in ("false", "0")
            ),
        },
        {
            "metric": "selected_failed_distribution",
            "value": sum(
                1 for r in selected
                if str(r.get("distribution_passed", "true")).lower() in ("false", "0")
            ),
        },
    ]
    return rows


def print_validation_summary(candidate_rows: List[dict], fd_rows: List[dict]) -> None:
    if not candidate_rows:
        return
    print("\n=== Validation summary ===")
    for row in _build_validation_summary_rows(candidate_rows, fd_rows):
        print(f"  {row['metric']}: {row['value']}")

    by_group: Dict[int, List[dict]] = {}
    for r in candidate_rows:
        by_group.setdefault(int(r["group_id"]), []).append(r)
    for group_id in sorted(by_group):
        print(f"\n--- Group {group_id} ---")
        for r in sorted(by_group[group_id], key=lambda x: int(x["candidate_idx"])):
            sel = " *" if str(r.get("selected", "")).lower() in ("true", "1") else ""
            print(
                f"  [{r['candidate_idx']}]{sel} {r['operation']} "
                f"{r['left_table']} + {r['right_table']}: "
                f"validation_score={float(r['validation_score']):.4f} "
                f"newly={r['newly_correctable']} lost={r.get('correctable_lost', 0)} "
                f"fds +{r['fds_introduced']}/-{r['fds_broken']}/={r['fds_retained']} "
                f"dist_passed={r.get('distribution_passed', '')} "
                f"max_tvd={r.get('distribution_max_tvd', '')} "
                f"max_ks={r.get('distribution_max_ks', '')}"
            )
            group_fds = [
                x for x in fd_rows
                if int(x["group_id"]) == group_id
                and int(x["candidate_idx"]) == int(r["candidate_idx"])
                and x.get("change_type") in ("introduced", "broken")
            ]
            for change in ("introduced", "broken"):
                labels = sorted(
                    x["fd"] for x in group_fds if x.get("change_type") == change
                )
                if labels:
                    print(f"    {change}: {', '.join(labels)}")
