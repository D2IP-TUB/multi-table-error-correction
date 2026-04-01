import copy
import json
import logging
import random
import unicodedata

from core.lake import Lake


def assign_budget_hard_zones(
    lake: Lake,
    zone_dict: dict,
    labeling_budget: int,
):
    for zone in zone_dict.values():
        if len(zone.cells) == 0:
            zone.labeling_budget = 0
            continue

        for cell in zone.cells.values():
            # Calculate cell influence
            cell_encoding_influence = set(
                lake.identity_encoding_dict.get(json.dumps(list(cell.value)), [])
            ).intersection(
                set(
                    lake.unicode_encoding_dict.get(
                        json.dumps([unicodedata.category(c) for c in cell.value]), []
                    )
                )
            )
            cell_influence = len(cell.row_error_cells) + len(cell_encoding_influence)
            cell.influence = cell_influence

    for zone in zone_dict.values():
        if "invalid_character" in zone.name or "invalid_pattern" in zone.name:
            # zone.labeling_budget = len(zone.columns)
            for column in zone.columns.values():
                column_clean_cells = [
                    cell for cell in column.cells.values() if not cell.is_error
                ]
                if len(column_clean_cells) != 0:
                    continue

                column_cells = [
                    cell
                    for cell in zone.cells.values()
                    if cell.column_idx == column.col_idx
                ]

                # pick two cells with the highest influence
                best_cell = max(
                    column_cells,
                    key=lambda c: c.influence,
                )
                zone.samples[
                    (best_cell.table_id, best_cell.column_idx, best_cell.row_idx)
                ] = best_cell
                column_cells.remove(best_cell)
                if len(column_cells) == 0:
                    continue
                second_best_cell = max(
                    column_cells,
                    key=lambda c: c.influence,
                )
                zone.samples[
                    (
                        second_best_cell.table_id,
                        second_best_cell.column_idx,
                        second_best_cell.row_idx,
                    )
                ] = second_best_cell
                labeling_budget -= 2
    if labeling_budget > 0:
        remained_zones = []
        for zone in zone_dict.values():
            if not ("invalid_character" in zone.name or "invalid_pattern" in zone.name):
                remained_zones.append(zone.name)
        for zone_name in zone_dict:
            if "invalid_character" in zone_name or "invalid_pattern" in zone_name:
                continue
            zone = zone_dict[zone_name]
            zone.labeling_budget = int(
                labeling_budget
                * (
                    len(zone.cells)
                    / sum(
                        len(z.cells)
                        for z in zone_dict.values()
                        if z.name in remained_zones
                    )
                )
            )
            uncovered_cells = copy.deepcopy(zone.cells)
            while zone.labeling_budget > 0:
                best_cells = max(
                    uncovered_cells.values(),
                    key=lambda c: c.influence,
                )
                zone.samples[
                    (best_cells.table_id, best_cells.column_idx, best_cells.row_idx)
                ] = best_cells
                zone.labeling_budget -= 1
                labeling_budget -= 1
                uncovered_cells.pop(
                    (best_cells.table_id, best_cells.column_idx, best_cells.row_idx)
                )
                cells_to_be_removed = []
                for cell in uncovered_cells.values():
                    if cell.column_idx == best_cells.column_idx:
                        cells_to_be_removed.append(
                            (cell.table_id, cell.column_idx, cell.row_idx)
                        )
                for cell_key in cells_to_be_removed:
                    uncovered_cells.pop(cell_key)
                if zone.labeling_budget > 0 and len(uncovered_cells) == 0:
                    uncovered_cells = copy.deepcopy(zone.cells)
                    for cell in zone.samples.values():
                        uncovered_cells.pop(
                            (cell.table_id, cell.column_idx, cell.row_idx)
                        )
                    if len(uncovered_cells) == 0:
                        break
    logging.info(f"Remained labeling budget: {labeling_budget}.")
    return zone_dict


def assign_budget_with_column_wise_zone_share(
    zone_dict: dict, labeling_budget: int, zones_share: dict
) -> dict:
    """
    Assign a labeling budget to a zone.

    Args:
        zone_dict (dict): Dictionary containing zones.
        labeling_budget (int): Total labeling budget to be distributed.
        zones_share (dict): Dictionary containing the share of budget for each zone.
    Returns:
        dict: Updated zone dictionary with assigned budgets.
    """
    zones_share_dict = {
        "syntactic_unique": zones_share[0],
        "syntactic_non_unique": zones_share[1],
        "semantic": zones_share[2],
    }
    spare_budget = 0
    for zone in zone_dict.values():
        if zone.name in zones_share_dict:
            zone.labeling_budget = int(labeling_budget * zones_share_dict[zone.name])
        else:
            zone.labeling_budget = 0
        if len(zone.cells) < zone.labeling_budget:
            spare_budget += zone.labeling_budget - len(zone.cells)
    zone_dict["syntactic_non_unique"].labeling_budget += spare_budget
    return zone_dict


def assign_budget_min_label_available(
    lake: Lake, zone_dict: dict, labeling_budget: int, output_dir: str
) -> dict:
    columns_to_zones = {}
    columns_to_cells = {}

    # Build mappings of columns to zones and cells
    for zone in zone_dict.values():
        if len(zone.cells) == 0:
            zone.labeling_budget = 0
            continue

        for cell in zone.cells.values():
            column_key = (cell.table_id, cell.column_idx)

            if column_key not in columns_to_zones:
                columns_to_zones[column_key] = set()
            if column_key not in columns_to_cells:
                columns_to_cells[column_key] = []

            columns_to_zones[column_key].add(zone.name)

            # Calculate cell influence
            cell_encoding_influence = set(
                lake.identity_encoding_dict.get(json.dumps(list(cell.value)), [])
            ).intersection(
                set(
                    lake.unicode_encoding_dict.get(
                        json.dumps([unicodedata.category(c) for c in cell.value]), []
                    )
                )
            )
            cell_influence = (
                len(cell.row_error_cells) + len(cell_encoding_influence)
                # + len(cell.column_error_cells)
            )
            cell.influence = cell_influence

            columns_to_cells[column_key].append(cell)

    # Track selected samples
    samples_selected = {column: [] for column in columns_to_zones}

    # Phase 1: Select one cell per column (initial coverage)
    for column in columns_to_zones:
        if len(columns_to_cells[column]) > 0 and labeling_budget > 0:
            # Find the best cell in this column
            best_cell = max(columns_to_cells[column], key=lambda c: c.influence)

            # Add to zone samples
            zone_dict[best_cell.zone].samples[
                (best_cell.table_id, best_cell.column_idx, best_cell.row_idx)
            ] = best_cell

            # Track selection
            samples_selected[column].append(best_cell)
            labeling_budget -= 1

    # Phase 2: Continue sampling with remaining budget (with zone reset)
    while labeling_budget > 0:
        best_candidate = None
        best_influence = -1
        best_column = None

        # Find the best unselected cell across all columns
        for column in columns_to_zones:
            if len(columns_to_cells[column]) > 0:
                # Get zones already sampled in this column
                zones_sampled_in_column = {
                    cell.zone for cell in samples_selected[column]
                }

                # Find unselected cells from unsampled zones in this column
                available_cells = [
                    cell
                    for cell in columns_to_cells[column]
                    if cell not in samples_selected[column]
                    and cell.zone not in zones_sampled_in_column
                ]

                if available_cells:
                    # Find the best cell among available ones
                    column_best = max(available_cells, key=lambda c: c.influence)

                    if column_best.influence > best_influence:
                        best_candidate = column_best
                        best_influence = column_best.influence
                        best_column = column

        # If we found a candidate, select it
        if best_candidate is not None:
            # Add to zone samples
            zone_dict[best_candidate.zone].samples[
                (
                    best_candidate.table_id,
                    best_candidate.column_idx,
                    best_candidate.row_idx,
                )
            ] = best_candidate

            # Track selection
            samples_selected[best_column].append(best_candidate)
            labeling_budget -= 1
        else:
            # No candidates found with zone restriction - try reset
            # Check if there are any unselected cells at all
            any_unselected = False
            for column in columns_to_zones:
                unselected_in_column = [
                    cell
                    for cell in columns_to_cells[column]
                    if cell not in samples_selected[column]
                ]
                if unselected_in_column:
                    any_unselected = True
                    break

            if any_unselected:
                # Reset: allow revisiting zones, find best unselected cell
                for column in columns_to_zones:
                    if len(columns_to_cells[column]) > 0:
                        # Find any unselected cells (no zone restriction)
                        available_cells = [
                            cell
                            for cell in columns_to_cells[column]
                            if cell not in samples_selected[column]
                        ]

                        if available_cells:
                            # Find the best cell among available ones
                            column_best = max(
                                available_cells, key=lambda c: c.influence
                            )

                            if column_best.influence > best_influence:
                                best_candidate = column_best
                                best_influence = column_best.influence
                                best_column = column

                # Select the best candidate found in reset phase
                if best_candidate is not None:
                    zone_dict[best_candidate.zone].samples[
                        (
                            best_candidate.table_id,
                            best_candidate.column_idx,
                            best_candidate.row_idx,
                        )
                    ] = best_candidate

                    samples_selected[best_column].append(best_candidate)
                    labeling_budget -= 1
                else:
                    break
            else:
                break
    logging.info(f"Remained labeling budget: {labeling_budget}.")

    return zone_dict


def assign_budget_automatic(
    zone_dict: dict, labeling_budget: int, output_dir: str, random_state: int = 42
) -> dict:
    rng = random.Random(random_state)
    zones_labeling_budget = {}
    all_err_cells_count = sum(len(zone.cells) for zone in zone_dict.values())
    for zone in zone_dict.values():
        if len(zone.cells) == 0:
            zones_labeling_budget[zone.name] = 0
            continue
        zones_labeling_budget[zone.name] = min(
            len(zone.cells),
            int(labeling_budget * (len(zone.cells) / all_err_cells_count)),
        )
        random_selected_cells = rng.sample(
            list(zone.cells.values()), zones_labeling_budget[zone.name]
        )
        for cell in random_selected_cells:
            zone.samples[(cell.table_id, cell.column_idx, cell.row_idx)] = cell
        zone.labeling_budget = zones_labeling_budget[zone.name]

    return zone_dict
