import logging

_flashfill = None


def _ensure_flashfill():
    global _flashfill
    if _flashfill is not None:
        return _flashfill
    from pyprose.transformation.text import flashfill
    _flashfill = flashfill
    return _flashfill


def generate_flash_fill_candidates(zones_dicst):
    for zone_name, zone in zones_dicst.items():
        if "invalid_pattern" not in zone_name:
            continue
        else:
            zone.flash_fill_samples = {}
            zone_error_cells_per_col = {}
            for cell_id, cell in zone.cells.items():
                if (cell.table_id, cell.column_idx) not in zone_error_cells_per_col:
                    zone_error_cells_per_col[(cell.table_id, cell.column_idx)] = []
                zone_error_cells_per_col[(cell.table_id, cell.column_idx)].append(cell)

        zone_samples_per_col = {}
        for sample_id, sample in zone.samples.items():
            for cell in zone_error_cells_per_col[(sample.table_id, sample.column_idx)]:
                if cell.row_idx == sample.row_idx:
                    continue
                if (cell.table_id, cell.column_idx) not in zone_samples_per_col:
                    zone_samples_per_col[(cell.table_id, cell.column_idx)] = []
                zone_samples_per_col[(cell.table_id, cell.column_idx)].append(sample)

        for col_id, samples in zone_samples_per_col.items():
            flash_fill_input = []
            table_id, col_idx = col_id
            if not samples:
                continue
            for sample in samples:
                if [sample.value, sample.ground_truth] not in flash_fill_input:
                    flash_fill_input.append([sample.value, sample.ground_truth])
            cell_id_to_flash_fill_idx = {}
            for cell in zone_error_cells_per_col[col_id]:
                if cell.row_idx not in [s.row_idx for s in samples]:
                    flash_fill_input.append([cell.value])
                    cell_id_to_flash_fill_idx[
                        (cell.table_id, cell.column_idx, cell.row_idx)
                    ] = len(flash_fill_input) - 1
            try:
                flash_filled = _ensure_flashfill()(flash_fill_input)
            except Exception as e:
                logging.error(
                    f"Error in flash fill for zone {zone_name}, column {col_id}: {e}"
                )
                continue
            for cell in zone_error_cells_per_col[col_id]:
                if cell.row_idx not in [s.row_idx for s in samples]:
                    flash_filled_value = flash_filled[
                        cell_id_to_flash_fill_idx[
                            (cell.table_id, cell.column_idx, cell.row_idx)
                        ]
                    ][1]
                    if flash_filled_value:
                        cell.flash_filled_value = flash_filled_value
                        zone.flash_fill_samples[
                            (
                                cell.table_id,
                                cell.column_idx,
                                cell.row_idx,
                            )
                        ] = cell
