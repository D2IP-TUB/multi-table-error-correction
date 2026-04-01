import csv
import pandas as pd
import shutil
from io import StringIO
from pathlib import Path
import config
from recreate_as_strings import (
    _read_provenance_with_exact_headers,
    _disambiguate_headers,
    recreate_merged_table_from_provenance,
)


def _find_rows_to_keep(dirty_df, clean_df):
    """
    Determine which rows to keep after UNION-style deduplication.

    Deduplication is driven entirely by the dirty table: rows with identical
    dirty values are candidates for removal. However, any row that contains
    at least one error (dirty != clean) is always kept, because we cannot
    safely discard it. Among a group of duplicate *error-free* rows, only
    the first occurrence is kept.

    Returns:
        Sorted list of row indices (0-based) to keep.
    """
    has_error = (dirty_df != clean_df).any(axis=1)

    seen_clean_tuples = set()
    keep = []
    for idx in range(len(dirty_df)):
        if has_error.iloc[idx]:
            keep.append(idx)
        else:
            row_tuple = tuple(dirty_df.iloc[idx])
            if row_tuple not in seen_clean_tuples:
                seen_clean_tuples.add(row_tuple)
                keep.append(idx)
    return keep


def _format_cell_id(table_id, column_id, row_id):
    return f'{table_id}.{column_id}.{row_id}'


def _infer_error_type_from_clean_changes_codes(error_codes):
    normalized = (error_codes or '').strip()
    if normalized == '[]':
        return 'RANDOM_TYPO'
    elif normalized.startswith('[') and normalized.endswith(']') and normalized != '[]':
        # Any non-empty error code list is FD_VIOLATION
        return 'FD_VIOLATION'
    else:
        return 'UNKNOWN'


def _resolve_error_type(error_type, violated_dependencies=''):
    """
    Resolve UNKNOWN labels using dependency/code hints.

    Rules:
    - Preserve known labels as-is.
    - UNKNOWN with empty/[]/[],[] style dependencies -> RANDOM_TYPO.
    - UNKNOWN with non-empty code list (contains e<digit>) -> FD_VIOLATION.
    """
    normalized_type = (error_type or '').strip().upper()
    if normalized_type and normalized_type != 'UNKNOWN':
        return normalized_type

    deps = (violated_dependencies or '').strip()
    if not deps:
        return 'RANDOM_TYPO'

    # Treat repeated empty-list patterns like [],[] or [] , [] as typo-like.
    deps_compact = deps.replace(' ', '')
    if deps_compact and set(deps_compact) <= set('[],'):
        return 'RANDOM_TYPO'

    if deps.startswith('[') and deps.endswith(']'):
        tokens = [token.strip() for token in deps[1:-1].split(',') if token.strip()]
        if not tokens:
            return 'RANDOM_TYPO'
        if any(token.lower().startswith('e') and token[1:].isdigit() for token in tokens):
            return 'FD_VIOLATION'

    return 'UNKNOWN'


def recreate_merged_tables_as_strings_union(disambiguate_columns=True):
    """
    Same as recreate_merged_tables_as_strings but applies UNION semantics:
    exact duplicate rows in the dirty file are removed, except rows that
    contain at least one error (dirty != clean) are always retained.

    The deduplication is performed on the dirty reconstruction; the clean
    file and provenance are then filtered to match the same kept rows.
    """

    # Derive the output leaf name from MERGED_PATH (e.g. "merged_union_0.5")
    output_dir = Path('merged_strings_default_set_union') / config.CORPUS / config.MERGED_PATH.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load merge summary to know which tables are union/join/none
    merge_summary_path = config.MERGED_PATH / 'merge_summary.csv'
    operation_by_id = {}
    if merge_summary_path.exists():
        summary_df = pd.read_csv(merge_summary_path, dtype=str, keep_default_na=False)
        for _, row in summary_df.iterrows():
            operation_by_id[str(row['output_id'])] = row['operation'].strip().lower()

    provenance_files = list(config.MERGED_PATH.glob('*_provenance.csv'))

    print(f"Found {len(provenance_files)} provenance files to process.\n")
    print(f"Output directory: {output_dir}\n")

    for prov_file in provenance_files:
        merged_id = prov_file.stem.replace('_provenance', '')
        table_dir = output_dir / merged_id
        table_dir.mkdir(exist_ok=True)

        print(f"Processing {merged_id}...")

        df_prov = _read_provenance_with_exact_headers(prov_file)

        # Collect all unique source tables mentioned in the provenance
        source_tables = set()
        for col in df_prov.columns:
            for cell_value in df_prov[col]:
                if cell_value and cell_value != '':
                    parts = cell_value.split(' § ')
                    if len(parts) == 3:
                        source_tables.add(parts[0])

        print(f"  Found {len(source_tables)} source tables: {source_tables}")

        # Load all source tables (both dirty and clean) as strings
        dirty_tables = {}
        clean_tables = {}
        error_provenance_tables = {}

        for table_name in source_tables:
            dirty_path = config.DIR_PATH / table_name / 'dirty.csv'
            clean_path = config.DIR_PATH / table_name / 'clean.csv'
            error_prov_path = config.DIR_PATH / table_name / 'clean_changes_provenance.csv'
            error_changes_path = config.DIR_PATH / table_name / 'clean_changes.csv'

            if clean_path.exists():
                clean_df = pd.read_csv(clean_path, dtype=str, keep_default_na=False, encoding='latin1')
                clean_tables[table_name] = clean_df
                print(f"  Loaded {table_name}/clean.csv ({len(clean_df)} rows)")
            else:
                print(f"  Warning: {clean_path} not found!")

            if dirty_path.exists():
                dirty_df = pd.read_csv(dirty_path, dtype=str, keep_default_na=False, encoding='latin1')
                if table_name in clean_tables:
                    dirty_df.columns = clean_tables[table_name].columns
                dirty_tables[table_name] = dirty_df
                print(f"  Loaded {table_name}/dirty.csv ({len(dirty_df)} rows)")
            else:
                print(f"  Warning: {dirty_path} not found!")

            if error_prov_path.exists():
                error_prov_df = pd.read_csv(error_prov_path, dtype=str, keep_default_na=False)
                error_map = {}
                for _, row in error_prov_df.iterrows():
                    # Parse cell_id to get correct column name (format: "row.column")
                    # The cell_id and column_name fields may be corrupted with data values appended
                    # e.g., "1261.full_name,\"Kimball, Richard W\"" instead of "1261.full_name"
                    if 'cell_id' in row and row['cell_id']:
                        cell_id = row['cell_id']
                        parts = cell_id.split('.', 1)  # Split on first dot only
                        if len(parts) == 2:
                            row_num = int(parts[0]) - 1  # cell_id uses 1-based indexing
                            # Extract column name before any comma (removes appended data values)
                            col_name = parts[1].split(',')[0]
                        else:
                            # Fallback to row_number and column_name if cell_id format unexpected
                            row_num = int(row['row_number']) - 1
                            col_name = row['column_name'].split(',')[0]
                    else:
                        # Fallback if cell_id doesn't exist
                        row_num = int(row['row_number']) - 1
                        col_name = row['column_name'].split(',')[0]
                    
                    error_type = _resolve_error_type(
                        row.get('error_type', ''),
                        row.get('violated_dependencies', ''),
                    )
                    error_map[(row_num, col_name)] = error_type
                error_provenance_tables[table_name] = error_map
                print(f"  Loaded {table_name}/clean_changes_provenance.csv ({len(error_prov_df)} errors)")
            elif error_changes_path.exists():
                error_map = {}
                num_errors = 0
                with open(error_changes_path, encoding='latin1', newline='') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row:
                            continue
                        if len(row) < 4:
                            continue

                        cell_id = row[0]
                        error_codes = ','.join(row[3:]).strip()
                        parts = cell_id.split('.')
                        if len(parts) < 2:
                            continue
                        # Isolated clean_changes* inputs use 1-based row indexing.
                        row_num = int(parts[0]) - 1
                        col_name = '.'.join(parts[1:])
                        error_map[(row_num, col_name)] = _infer_error_type_from_clean_changes_codes(error_codes)
                        num_errors += 1
                error_provenance_tables[table_name] = error_map
                print(f"  Loaded {table_name}/clean_changes.csv ({num_errors} errors, with inferred types)")

        # Reconstruct full dirty and clean tables (UNION ALL)
        merged_dirty_df = recreate_merged_table_from_provenance(df_prov, dirty_tables, 'dirty', debug=True)
        merged_clean_df = recreate_merged_table_from_provenance(df_prov, clean_tables, 'clean', debug=True)

        # --- UNION deduplication (only for union-operation tables) ---
        operation = operation_by_id.get(merged_id, 'unknown')
        if operation == 'union':
            n_before = len(merged_dirty_df)
            keep_indices = _find_rows_to_keep(merged_dirty_df, merged_clean_df)
            n_after = len(keep_indices)
            n_removed = n_before - n_after
            print(f"  Deduplication: {n_before} rows -> {n_after} rows ({n_removed} exact clean duplicates removed)")
            merged_dirty_df = merged_dirty_df.iloc[keep_indices].reset_index(drop=True)
            merged_clean_df = merged_clean_df.iloc[keep_indices].reset_index(drop=True)
            df_prov_filtered = df_prov.iloc[keep_indices].reset_index(drop=True)
        else:
            print(f"  Skipping deduplication (operation='{operation}')")
            df_prov_filtered = df_prov.reset_index(drop=True)

        # Write dirty CSV
        out_dirty = merged_dirty_df.copy()
        if disambiguate_columns:
            out_dirty.columns = _disambiguate_headers(list(merged_dirty_df.columns))
        out_dirty.to_csv(table_dir / 'dirty.csv', index=False)
        print(f"  Created dirty.csv with {len(merged_dirty_df)} rows and {len(merged_dirty_df.columns)} columns")

        # Write clean CSV
        out_clean = merged_clean_df.copy()
        if disambiguate_columns:
            out_clean.columns = _disambiguate_headers(list(merged_clean_df.columns))
        out_clean.to_csv(table_dir / 'clean.csv', index=False)
        print(f"  Created clean.csv with {len(merged_clean_df)} rows and {len(merged_clean_df.columns)} columns")

        # Map full header -> display name for error provenance
        col_to_display = None
        if disambiguate_columns:
            display_names = _disambiguate_headers(list(df_prov.columns))
            col_to_display = dict(zip(df_prov.columns, display_names))

        # Create merged-cell -> source-cell map (based on filtered provenance)
        merged_cell_source_records = []
        for row_idx in range(len(df_prov_filtered)):
            for col_idx, col_name in enumerate(df_prov_filtered.columns):
                prov_value = df_prov_filtered.loc[row_idx, col_name]
                if not prov_value:
                    continue

                parts = prov_value.split(' § ')
                if len(parts) != 3:
                    continue

                source_table = parts[0]
                source_col_idx = int(parts[1])
                source_row_idx = int(parts[2])

                if source_table in dirty_tables:
                    source_col_name = dirty_tables[source_table].columns[source_col_idx]
                elif source_table in clean_tables:
                    source_col_name = clean_tables[source_table].columns[source_col_idx]
                else:
                    continue

                error_type = ''
                if source_table in error_provenance_tables:
                    source_error_map = error_provenance_tables[source_table]
                    if (source_row_idx, source_col_name) in source_error_map:
                        error_type = source_error_map[(source_row_idx, source_col_name)]

                out_col = col_to_display[col_name] if col_to_display else col_name
                cell_id = _format_cell_id(merged_id, col_idx, row_idx)
                merged_cell_source_records.append({
                    'cell_id': cell_id,
                    'table_id': merged_id,
                    'column_id': col_idx,
                    'row_number': row_idx,
                    'column_name': out_col,
                    'source_table': source_table,
                    'source_row': source_row_idx,
                    'source_column': source_col_name,
                    'error_type': error_type
                })

        if merged_cell_source_records:
            merged_cell_source_df = pd.DataFrame(merged_cell_source_records)
            merged_cell_source_df.to_csv(table_dir / 'merged_cell_source_map.csv', index=False)
            print(f"  Created merged_cell_source_map.csv with {len(merged_cell_source_records)} mappings")

        # Reconstruct error provenance using the filtered provenance
        if error_provenance_tables:
            error_records = []
            # Track source cell occurrences with their locations in merged table
            source_cell_occurrences = {}  # Key: (source_table, source_row, source_col), Value: list of merged row indices

            for row_idx in range(len(df_prov_filtered)):
                for col_idx, col_name in enumerate(df_prov_filtered.columns):
                    prov_value = df_prov_filtered.loc[row_idx, col_name]

                    if prov_value and prov_value != '':
                        parts = prov_value.split(' § ')
                        if len(parts) == 3:
                            source_table = parts[0]
                            source_col_idx = int(parts[1])
                            source_row_idx = int(parts[2])

                            if source_table in dirty_tables:
                                source_col_name = dirty_tables[source_table].columns[source_col_idx]
                            elif source_table in clean_tables:
                                source_col_name = clean_tables[source_table].columns[source_col_idx]
                            else:
                                continue

                            # Track occurrence of this source cell with its location
                            source_cell_key = (source_table, source_row_idx, source_col_name)
                            if source_cell_key not in source_cell_occurrences:
                                source_cell_occurrences[source_cell_key] = []
                            source_cell_occurrences[source_cell_key].append(row_idx)

                            if source_table in error_provenance_tables:
                                error_map = error_provenance_tables[source_table]
                                if (source_row_idx, source_col_name) in error_map:
                                    error_type = error_map[(source_row_idx, source_col_name)]

                                    old_value = merged_dirty_df.iloc[row_idx][col_name]
                                    new_value = merged_clean_df.iloc[row_idx][col_name]

                                    out_col = col_to_display[col_name] if col_to_display else col_name
                                    cell_id = _format_cell_id(merged_id, col_idx, row_idx)
                                    error_records.append({
                                        'cell_id': cell_id,
                                        'table_id': merged_id,
                                        'column_id': col_idx,
                                        'row_number': row_idx,
                                        'column_name': out_col,
                                        'old_value': old_value,
                                        'new_value': new_value,
                                        'error_type': error_type,
                                        'source_table': source_table,
                                        'source_row': source_row_idx,
                                        'source_column': source_col_name
                                    })

            if error_records:
                error_prov_df = pd.DataFrame(error_records)
                error_prov_df.to_csv(table_dir / 'clean_changes_provenance.csv', index=False)
                print(f"  Created clean_changes_provenance.csv with {len(error_records)} errors tracked")

            # Create isolated files error map: source cell -> occurrence count + error type
                        # Create isolated files error map: source cell -> occurrence locations + error type
            if source_cell_occurrences:
                isolated_error_records = []
                for (source_table, source_row, source_col), merged_row_indices in source_cell_occurrences.items():
                    if source_table in error_provenance_tables:
                        error_map = error_provenance_tables[source_table]
                        if (source_row, source_col) in error_map:
                            error_type = error_map[(source_row, source_col)]
                            isolated_error_records.append({
                                'source_table': source_table,
                                'source_row': source_row,
                                'source_column': source_col,
                                'occurrence_count': len(merged_row_indices),
                                'merged_row_indices': ','.join(map(str, merged_row_indices)),
                                'error_type': error_type
                            })

                if isolated_error_records:
                    isolated_errors_df = pd.DataFrame(isolated_error_records)
                    isolated_errors_df.to_csv(table_dir / 'isolated_error_map.csv', index=False)
                    print(f"  Created isolated_error_map.csv with {len(isolated_error_records)} source cells tracked")

        # Save filtered provenance
        df_prov_filtered.to_csv(table_dir / 'provenance.csv', index=False, encoding='latin1')
        print(f"  Created provenance.csv ({len(df_prov_filtered)} rows)\n")


def main():
    print("=" * 80)
    print("Recreating merged tables (UNION — deduplicated) from provenance files...")
    print("=" * 80 + "\n")
    recreate_merged_tables_as_strings_union()
    print("\nDone!")


if __name__ == '__main__':
    main()
