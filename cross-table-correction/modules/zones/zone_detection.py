import logging
from typing import Dict


def detect_zones(lake, config, zones_dict: Dict = None):
    """
    Detect zones using rule-based strategy.

    Args:
        lake: The Lake object containing all tables and cells
        config: PipelineConfig with zoning settings
        zones_dict: Dictionary of zones to populate

    Returns:
        Dictionary of zones populated based on configured strategy
    """
    if zones_dict is None:
        zones_dict = {}

    detect_all_cells_zones_rule_based(lake, zones_dict)

    return zones_dict


def detect_all_cells_zones_rule_based(lake, zones_dict):
    """Detect zones using rule-based approach with pattern validation.
    
    Creates zones based on:
    1. Cardinality (unique/high vs non-unique/low)
    2. Pattern validity (valid vs invalid pattern)
    3. Semantic match (value in clean domain)
    """
    from pyprose.matching.text import learn_patterns
    
    for table in lake.tables:
        for column in lake.tables[table].columns.values():
            column_clean_values = column.get_unique_clean_values()
            column_error_cells = column.get_error_cells()
            if len(column_error_cells) == 0:
                logging.info(
                    f"Column {column.col_name} in table {table} has no error cells."
                )
                continue
            
            # Learn patterns from clean values for this column
            patterns = None
            if len(column_clean_values) >= 2:
                try:
                    patterns = learn_patterns(list(column_clean_values), include_outlier_patterns=True)
                except Exception as e:
                    logging.debug(f"Failed to learn patterns for column {column.col_name}: {e}")
                    patterns = None
            
            column.zones = set()
            for cell in column_error_cells:
                detect_cell_zone(
                    cell,
                    column,
                    column_clean_values,
                    zones_dict,
                    patterns,
                )
                column.zones.add(cell.zone)


def detect_cell_zone(
    cell,
    column,
    column_clean_values,
    zones_dict,
    patterns=None,
):
    """Detect the zone of a cell based on cardinality and pattern validity.

    Zones:
    - "unique_valid_pattern": high cardinality + value matches a valid pattern
    - "unique_invalid_pattern": high cardinality + value does NOT match any pattern
    - "non_unique_valid_pattern": low cardinality + (value in clean domain OR matches valid pattern)
    - "non_unique_invalid_pattern": low cardinality + value does NOT exist in clean domain AND does NOT match pattern

    Args:
        cell: The cell to be analyzed.
        column: The column to which the cell belongs.
        column_clean_values: Clean values of the column.
        zones_dict: Dictionary of zones to add the cell to.
        patterns: Learned PyProse patterns for this column (optional)
    """
    cell_value = cell.value
    column_key = (column.table_id, column.col_idx)

    # If column has no clean values, pattern cannot be learned -> mark as invalid_pattern by default
    # This is because patterns are learned from clean values
    if len(column_clean_values) == 0:
        if column.cardinality_type == "high":
            zone_name = "unique_invalid_pattern"
        else:
            zone_name = "non_unique_invalid_pattern"
    else:
        # Check if cell value is in clean domain
        value_in_clean_domain = cell_value in column_clean_values

        # Check if value matches any valid pattern
        pattern_is_valid = False
        if patterns:
            try:
                for pattern in patterns:
                    # Check if value matches this pattern
                    if hasattr(pattern, 'matches') and pattern.matches(cell_value):
                        pattern_is_valid = True
                        break
            except Exception as e:
                logging.debug(f"Pattern matching failed for cell {cell}: {e}")
                pattern_is_valid = False

        # Determine cardinality-based zone name
        if column.cardinality_type == "high":
            # Unique column: only pattern validity matters
            if pattern_is_valid:
                zone_name = "unique_valid_pattern"
            else:
                zone_name = "unique_invalid_pattern"
        else:
            # Non-unique column: value in clean domain OR pattern match = valid
            if value_in_clean_domain or pattern_is_valid:
                zone_name = "non_unique_valid_pattern"
            else:
                zone_name = "non_unique_invalid_pattern"

    cell.zone = zone_name
    target_zone = zones_dict[zone_name]
    target_zone.add_cell(cell)

    if target_zone.columns is None:
        target_zone.columns = {}
    target_zone.columns[column_key] = column
