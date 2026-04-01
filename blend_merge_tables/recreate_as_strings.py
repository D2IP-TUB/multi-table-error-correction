import csv
import pandas as pd
import shutil
from io import StringIO
from pathlib import Path
import config


# Merged headers are "table::col_id::col_name" (join) or "table::col_id::name | table2::col_id::name" (union).
# They are unique as written by merge_tables. If the CSV had duplicate header strings, pandas would
# make them unique on read (e.g. id, id.1), so we read the provenance header row explicitly to
# preserve exact column names and any intentional duplicates.


def _read_provenance_with_exact_headers(prov_path):
    """Read provenance CSV preserving the exact header row (no pandas duplicate renaming)."""
    with open(prov_path, encoding='latin1') as f:
        first_line = f.readline()
    header_row = next(csv.reader(StringIO(first_line)))
    df = pd.read_csv(
        prov_path, dtype=str, keep_default_na=False, encoding='latin1',
        names=header_row, skiprows=1
    )
    return df


def _disambiguate_headers(column_names):
    """
    When several columns share the same semantic name (e.g. join key 'id' from two tables),
    return a list of unique display names so downstream tools don't see duplicate names.
    Format: "table::col_id::name" -> "name" or "name_<table>" when name repeats.
    Union columns "left::0::x | right::1::x" use the first table for the suffix.
    """
    def semantic_name(h):
        part = h.split('|')[0].strip() if '|' in h else h
        return part.split('::')[-1].strip() if '::' in part else part

    def table_prefix(h):
        part = h.split('|')[0].strip() if '|' in h else h
        return part.split('::')[0].strip() if '::' in part else ''

    name_counts = {semantic_name(h): 0 for h in column_names}
    for h in column_names:
        name_counts[semantic_name(h)] += 1

    result = []
    used = {}
    for h in column_names:
        base = semantic_name(h)
        tbl = table_prefix(h)
        if name_counts[base] <= 1:
            result.append(base)
        else:
            key = (base, tbl)
            used[key] = used.get(key, 0) + 1
            n = used[key]
            result.append(f"{base}_{tbl}" if n == 1 else f"{base}_{tbl}_{n}")
    return result


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


def recreate_merged_table_from_provenance(df_prov, source_tables_dict, csv_type='dirty', debug=False):
    """
    Reconstruct a merged table from provenance information.
    
    Args:
        df_prov: DataFrame with provenance information
        source_tables_dict: Dict mapping table_name to DataFrame
        csv_type: 'dirty' or 'clean' to specify which CSV to read from
        debug: Print debug information for failed lookups
    
    Returns:
        DataFrame with reconstructed data as strings
    """
    merged_string_data = {col_name: [] for col_name in df_prov.columns}
    lookup_failures = {'table_not_found': 0, 'column_not_found': 0, 'row_out_of_bounds': 0, 'empty_prov': 0}
    
    for row_idx in range(len(df_prov)):
        for col_name in df_prov.columns:
            prov_value = df_prov.iloc[row_idx][col_name]
            
            if not prov_value or prov_value == '':
                merged_string_data[col_name].append('')
                lookup_failures['empty_prov'] += 1
            else:
                # Parse provenance: 'table_name § col_index § row_idx'
                parts = prov_value.split(' § ')
                if len(parts) == 3:
                    source_table = parts[0]
                    source_col_idx = int(parts[1])
                    source_row_idx = int(parts[2])
                    
                    if source_table not in source_tables_dict:
                        merged_string_data[col_name].append('')
                        lookup_failures['table_not_found'] += 1
                        if debug and lookup_failures['table_not_found'] <= 5:
                            print(f"    DEBUG: Table '{source_table}' not found in source_tables_dict")
                    else:
                        source_df = source_tables_dict[source_table]
                        if source_col_idx < 0 or source_col_idx >= len(source_df.columns):
                            merged_string_data[col_name].append('')
                            lookup_failures['column_not_found'] += 1
                            if debug and lookup_failures['column_not_found'] <= 5:
                                print(f"    DEBUG: Column index {source_col_idx} out of bounds in table '{source_table}' ({len(source_df.columns)} cols)")
                        elif source_row_idx < 0 or source_row_idx >= len(source_df):
                            merged_string_data[col_name].append('')
                            lookup_failures['row_out_of_bounds'] += 1
                            if debug and lookup_failures['row_out_of_bounds'] <= 5:
                                print(f"    DEBUG: Row {source_row_idx} out of bounds in table '{source_table}' (len={len(source_df)})")
                        else:
                            original_value = source_df.iloc[source_row_idx, source_col_idx]
                            merged_string_data[col_name].append(original_value)
                else:
                    merged_string_data[col_name].append('')
    
    if debug or sum(lookup_failures.values()) > len(df_prov) * len(df_prov.columns) * 0.01:  # More than 1% failures
        print(f"  Lookup statistics for {csv_type}:")
        for key, count in lookup_failures.items():
            if count > 0:
                print(f"    {key}: {count}")
    
    return pd.DataFrame(merged_string_data)


def recreate_merged_tables_as_strings(disambiguate_columns=True):
    """
    Iterate over provenance files and recreate merged CSV files with all columns as strings.
    This preserves the original data representation for data cleaning purposes.
    Creates both dirty.csv and clean.csv for each merged table.

    Headers in the merged files are exactly as in the provenance (table::col_id::col_name).
    They are unique per column. For joins, the same semantic name (e.g. id) can appear twice
    (FK and PK); both are kept with distinct full headers.

    Args:
        disambiguate_columns: If True, rename columns so that repetitive semantic names
            get a _<table> suffix (e.g. id -> id_fk_table, id_pk_table) for clearer output.
    """
    
    output_dir = Path('merged_strings_default_union_all') / config.CORPUS / config.MERGED_PATH.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    provenance_files = list(config.MERGED_PATH.glob('*_provenance.csv'))
    
    print(f"Found {len(provenance_files)} provenance files to process.\n")
    print(f"Output directory: {output_dir}\n")
    
    for prov_file in provenance_files:
        merged_id = prov_file.stem.replace('_provenance', '')
        table_dir = output_dir / merged_id
        table_dir.mkdir(exist_ok=True)
        
        print(f"Processing {merged_id}...")
        
        # Read provenance preserving exact headers (no pandas duplicate renaming)
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
                # Normalize column names to match clean columns (dirty has type suffixes)
                if table_name in clean_tables:
                    dirty_df.columns = clean_tables[table_name].columns
                dirty_tables[table_name] = dirty_df
                print(f"  Loaded {table_name}/dirty.csv ({len(dirty_df)} rows)")
            else:
                print(f"  Warning: {dirty_path} not found!")
            
            # Try to load error provenance file, then fallback to clean_changes.csv
            if error_prov_path.exists():
                # Load error provenance file - it contains error information per cell
                error_prov_df = pd.read_csv(error_prov_path, dtype=str, keep_default_na=False)
                # Create a mapping of (row, column) -> error_type for quick lookup
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
                # Load fallback format: cell_id, old_value, new_value, error_codes (no explicit error type)
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
        
        # Reconstruct dirty merged table
        merged_dirty_df = recreate_merged_table_from_provenance(df_prov, dirty_tables, 'dirty', debug=True)
        out_dirty = merged_dirty_df.copy()
        if disambiguate_columns:
            out_dirty.columns = _disambiguate_headers(list(merged_dirty_df.columns))
        out_dirty.to_csv(table_dir / 'dirty.csv', index=False)
        print(f"  Created dirty.csv with {len(merged_dirty_df)} rows and {len(merged_dirty_df.columns)} columns")
        
        # Reconstruct clean merged table
        merged_clean_df = recreate_merged_table_from_provenance(df_prov, clean_tables, 'clean', debug=True)
        out_clean = merged_clean_df.copy()
        if disambiguate_columns:
            out_clean.columns = _disambiguate_headers(list(merged_clean_df.columns))
        out_clean.to_csv(table_dir / 'clean.csv', index=False)
        print(f"  Created clean.csv with {len(merged_clean_df)} rows and {len(merged_clean_df.columns)} columns")
        
        # Optional: map full header -> display name for error provenance
        col_to_display = None
        if disambiguate_columns:
            display_names = _disambiguate_headers(list(df_prov.columns))
            col_to_display = dict(zip(df_prov.columns, display_names))

        # Create merged-cell -> source-cell map (based on provenance)
        merged_cell_source_records = []
        for row_idx in range(len(df_prov)):
            for col_idx, col_name in enumerate(df_prov.columns):
                prov_value = df_prov.loc[row_idx, col_name]
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

        # Reconstruct error provenance if any source tables have error information
        if error_provenance_tables:
            error_records = []
            # Track source cell occurrences with their locations in merged table
            source_cell_occurrences = {}  # Key: (source_table, source_row, source_col), Value: list of merged row indices
            
            for row_idx in range(len(df_prov)):
                for col_idx, col_name in enumerate(df_prov.columns):
                    prov_value = df_prov.loc[row_idx, col_name]
                    
                    if prov_value and prov_value != '':
                        parts = prov_value.split(' § ')
                        if len(parts) == 3:
                            source_table = parts[0]
                            source_col_idx = int(parts[1])
                            source_row_idx = int(parts[2])
                            
                            # Map column index to column name for error lookup
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
                            
                            # Check if this cell has an error in the source table
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
        
        # Copy provenance file to the table directory
        shutil.copy(prov_file, table_dir / 'provenance.csv')
        print(f"  Copied provenance.csv\n")


def main():
    print("="*80)
    print("Recreating merged tables with string types from provenance files...")
    print("="*80 + "\n")
    recreate_merged_tables_as_strings()
    print("\nDone!")


if __name__ == '__main__':
    main()
