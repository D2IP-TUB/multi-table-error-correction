import csv
import duckdb
import networkx as nx
import polars as pl
import random
import shutil
from tqdm import tqdm

import config


def find_joinable_tables(db_conn, tab_id, col_id, values, is_numeric, tab_lengths, tab_ids, top_k, min_rows, threshold):
    """
    Given a primary key, find its top_k foreign keys in the corpus.
    :param db_conn: connection to the DuckDB database storing the BLEND index
    :param tab_id: table identifier
    :param col_id: column index
    :param values: set of column values
    :param is_numeric: column type
    :param tab_lengths: table lengths
    :param tab_ids: identifiers of the corpus tables to consider
    :param top_k: number of columns to return
    :param min_rows: minimum ratio of tuples joined for each table
    :param threshold: minimum ratio of joined tuples over the length of the joined table
    :return: a list with the top_k overlapping columns in the corpus
    """

    query = f"""
        SELECT x.tab_id, x.col_id, COUNT(DISTINCT x.value), COUNT(x.value)
        FROM cell_idx x JOIN col_idx y ON x.tab_id = y.tab_id AND x.col_id = y.col_id
        WHERE x.tab_id IN ({', '.join(repr(x) for x in tab_ids)})
        AND y.is_numeric = {is_numeric}
        AND x.is_clean = True
        AND x.value IN ({', '.join(['?'] * len(values))})
        GROUP BY x.tab_id, x.col_id
        ORDER BY COUNT(DISTINCT x.value) DESC
        LIMIT {10 * top_k}
    """

    top_cols = list()
    for x in db_conn.execute(query, values).fetchall():
        pk_joined = x[2]
        pk_dangling = tab_lengths[tab_id] - pk_joined
        fk_joined = x[3]
        fk_dangling = tab_lengths[x[0]] - fk_joined
        if pk_joined < min_rows * tab_lengths[tab_id] or fk_joined < min_rows * tab_lengths[x[0]]:
            continue
        score = fk_joined / (fk_joined + pk_dangling + fk_dangling)
        if score >= threshold:
            top_cols.append({
                'l_tab_id': tab_id,
                'r_tab_id': x[0],
                'operation': 'join',
                'mapping': {col_id: x[1]},
                'joined_tuples': fk_joined,
                'dangling_tuples': pk_dangling + fk_dangling,
                'score': score
            })

    top_joins = sorted(top_cols, key=lambda x: x['score'], reverse=True)[:min(top_k, len(top_cols))]

    return top_joins


def find_unionable_tables(db_conn, tab_id, clean_values, is_numeric, tab_lengths, tab_ids, top_k, min_cols, threshold):
    """
    Given a table, find unionable tables in the corpus.
    :param db_conn: connection to the DuckDB database storing the BLEND index
    :param tab_id: table identifier
    :param clean_values: list of clean cell values in each column
    :param is_numeric: list of column types
    :param tab_lengths: table lengths
    :param tab_ids: identifiers of the corpus tables to consider
    :param top_k: number of tables to return
    :param min_cols: minimum ratio of matching columns over the total of both tables
    :param threshold: minimum cell overlap to consider two columns as unionable
    :return: a list with the top_k unionable tables in the corpus
    """

    # Find the most overlapping columns in the corpus
    # Column overlap: number of (clean) tuples with common values / total number of tuples
    top_tables = dict()
    for col_id in range(len(clean_values)):
        values = set(clean_values[col_id])
        query = f"""
            SELECT x.tab_id, x.col_id, COUNT(x.value), LIST(DISTINCT value)
            FROM cell_idx x JOIN col_idx y ON x.tab_id = y.tab_id AND x.col_id = y.col_id
            WHERE x.tab_id IN ({', '.join(repr(x) for x in tab_ids)})
            AND y.is_numeric = {is_numeric[col_id]}
            AND x.is_clean = True
            AND x.value IN ({', '.join(['?'] * len(values))})
            GROUP BY x.tab_id, x.col_id
            ORDER BY COUNT(x.value) DESC
            LIMIT {10 * top_k}
        """
        for (r_tab_id, r_col_id, r_tab_matched, r_tab_values) in db_conn.execute(query, values).fetchall():
            l_tab_matched = sum([clean_values[col_id].count(x) for x in r_tab_values])
            score = (l_tab_matched + r_tab_matched) / (tab_lengths[tab_id] + tab_lengths[r_tab_id])
            if score < threshold:
                continue
            if r_tab_id not in top_tables:
                top_tables[r_tab_id] = list()
            top_tables[r_tab_id].append((col_id, r_col_id, score))

    # Maximum weighted bipartite matching (with check on column number)
    top_unions = list()
    for r_tab_id in top_tables:
        r_tab_cols = db_conn.execute(f"SELECT COUNT(*) FROM col_idx WHERE tab_id = {r_tab_id}").fetchone()[0]
        num_cols = len(clean_values) + r_tab_cols
        g = nx.Graph()
        g.add_weighted_edges_from([(f'l_{x[0]}', f'r_{x[1]}', x[2]) for x in top_tables[r_tab_id]])
        mapping = [(x[0], x[1]) if x[0] <= x[1] else (x[1], x[0]) for x in list(nx.max_weight_matching(g))]
        mapping = {(int(x[0].lstrip('l_')), int(x[1].lstrip('r_'))) for x in mapping}
        if len(mapping) == 1 or 2 * len(mapping) < min_cols * num_cols:
            continue
        coverage = (2 * len(mapping)) / num_cols
        score = 0
        for x in top_tables[r_tab_id]:
            if (x[0], x[1]) in mapping:
                score += x[2]
        top_unions.append({
            'l_tab_id': tab_id,
            'r_tab_id': r_tab_id,
            'operation': 'union',
            'mapping': {x[0]: x[1] for x in mapping},
            'coverage': coverage,
            'score': score
        })

    top_unions = sorted(top_unions, key=lambda x: x['score'], reverse=True)[:min(top_k, len(top_unions))]

    return top_unions


def union_tables(l_tab_name, r_tab_name, l_tab_path, r_tab_path, mapping):
    """
    Union two tables.
    :param l_tab_name: name of the left table
    :param r_tab_name: name of the right table
    :param l_tab_path: path to the directory storing the left table
    :param r_tab_path: path to the directory storing the right table
    :param mapping: column mapping
    :return: the unioned table together with the corresponding cell tracker and header
    """

    # Load the left table
    with open(l_tab_path / 'dirty.csv', encoding='latin1') as file:
        csv_reader = csv.reader(file)
        l_header = next(csv_reader)
        l_data = [list(col) for col in zip(*list(csv_reader))]
        l_num_cols = len(l_data)
        l_num_rows = len(l_data[0])

    # Load the right table
    with open(r_tab_path / 'dirty.csv', encoding='latin1') as file:
        csv_reader = csv.reader(file)
        r_header = next(csv_reader)
        r_data = [list(col) for col in zip(*list(csv_reader))]
        r_num_cols = len(r_data)
        r_num_rows = len(r_data[0])

    # Generate the cell trackers for the two tables
    # Each cell stores a string 'table_name § column_name § row_idx'
    l_tracker = [[f'{l_tab_name} § {col_id} § {row_id}' for row_id in range(l_num_rows)] for col_id in range(l_num_cols)]
    r_tracker = [[f'{r_tab_name} § {col_id} § {row_id}' for row_id in range(r_num_rows)] for col_id in range(r_num_cols)]

    # Initialize the unioned table (hence the corresponding cell tracker) as a list of empty lists, one per column
    tot_cols = l_num_cols + r_num_cols - len(mapping)
    union_data = [list() for _ in range(tot_cols)]
    union_tracker = [list() for _ in range(tot_cols)]

    # Start filling the unioned table with the rows from the left table
    union_header = [f'{l_tab_name}::{col_id}::{l_header[col_id]}' for col_id in range(l_num_cols)] + ['' for _ in range(r_num_cols - len(mapping))]
    for row_id in range(l_num_rows):
        row = [l_data[col_id][row_id] for col_id in range(l_num_cols)] + ['' for _ in range(r_num_cols - len(mapping))]
        t_row = [l_tracker[col_id][row_id] for col_id in range(l_num_cols)] + ['' for _ in range(r_num_cols - len(mapping))]
        for col_id in range(tot_cols):
            union_data[col_id].append(row[col_id])
            union_tracker[col_id].append(t_row[col_id])

    # Continue filling the unioned table with the rows from the right table (following the column mappings)
    r2l = {mapping[l_col_id]: l_col_id for l_col_id in mapping}
    r_mapping = [mapping[col_id] if col_id in mapping else None for col_id in range(l_num_cols)]
    r_mapping += [col_id for col_id in range(r_num_cols) if col_id not in r2l]
    for col_id in range(tot_cols):
        if r_mapping[col_id] is None:
            continue
        if union_header[col_id] == '':
            union_header[col_id] = f'{r_tab_name}::{r_mapping[col_id]}::{r_header[r_mapping[col_id]]}'
        else:
            union_header[col_id] += f' | {r_tab_name}::{r_mapping[col_id]}::{r_header[r_mapping[col_id]]}'
    for row_id in range(r_num_rows):
        row = [r_data[col_id][row_id] if col_id is not None else '' for col_id in r_mapping]
        t_row = [r_tracker[col_id][row_id] if col_id is not None else '' for col_id in r_mapping]
        for col_id in range(tot_cols):
            union_data[col_id].append(row[col_id])
            union_tracker[col_id].append(t_row[col_id])

    return union_data, union_tracker, union_header


def join_tables(l_tab_name, r_tab_name, l_tab_path, r_tab_path, mapping):
    """
    Join two tables.
    :param l_tab_name: name of the left table
    :param r_tab_name: name of the right table
    :param l_tab_path: path to the directory storing the left table
    :param r_tab_path: path to the directory storing the right table
    :param mapping: column mapping
    :return: the joined table together with the corresponding cell tracker and header
    """

    # Load the left table
    with open(l_tab_path / 'dirty.csv', encoding='latin1') as file:
        csv_reader = csv.reader(file)
        l_header = next(csv_reader)
        l_data = [list(col) for col in zip(*list(csv_reader))]
        l_num_cols = len(l_data)
        l_num_rows = len(l_data[0])

    # Load the clean version of the left table
    with open(l_tab_path / 'clean.csv', encoding='latin1') as file:
        csv_reader = csv.reader(file)
        l_clean_header = next(csv_reader)
        l_clean_data = [list(col) for col in zip(*list(csv_reader))]

    # Load the right table
    with open(r_tab_path / 'dirty.csv', encoding='latin1') as file:
        csv_reader = csv.reader(file)
        r_header = next(csv_reader)
        r_data = [list(col) for col in zip(*list(csv_reader))]
        r_num_cols = len(r_data)
        r_num_rows = len(r_data[0])

    # Load the clean version of the right table
    with open(r_tab_path / 'clean.csv', encoding='latin1') as file:
        csv_reader = csv.reader(file)
        r_clean_header = next(csv_reader)
        r_clean_data = [list(col) for col in zip(*list(csv_reader))]

    # Generate the cell trackers for the two tables
    # Each cell stores a string 'table_name § column_name § row_idx'
    l_tracker = [[f'{l_tab_name} § {col_id} § {row_id}' for row_id in range(l_num_rows)] for col_id in range(l_num_cols)]
    r_tracker = [[f'{r_tab_name} § {col_id} § {row_id}' for row_id in range(r_num_rows)] for col_id in range(r_num_cols)]

    # Track the row index of each clean value in the primary key (left table)
    pk_id = list(mapping.keys())[0]
    fk_id = mapping[pk_id]
    pk_values = {l_data[pk_id][row_id]: row_id for row_id in range(l_num_rows) if l_data[pk_id][row_id] == l_clean_data[pk_id][row_id]}

    # Initialize the joined table (hence the corresponding cell tracker) as a list of empty lists, one per column
    tot_cols = l_num_cols + r_num_cols
    join_header = [f'{r_tab_name}::{col_id}::{r_header[col_id]}' for col_id in range(r_num_cols)]
    join_header += [f'{l_tab_name}::{col_id}::{l_header[col_id]}' for col_id in range(l_num_cols)]
    join_data = [list() for _ in range(tot_cols)]
    join_tracker = [list() for _ in range(tot_cols)]
    join_pk_row_ids = set()
    for row_id in range(r_num_rows):
        row = [r_data[col_id][row_id] for col_id in range(r_num_cols)]
        t_row = [r_tracker[col_id][row_id] for col_id in range(r_num_cols)]
        if r_data[fk_id][row_id] == r_clean_data[fk_id][row_id] and r_data[fk_id][row_id] in pk_values:
            l_row_id = pk_values[r_data[fk_id][row_id]]
            row += [l_data[col_id][l_row_id] for col_id in range(l_num_cols)]
            t_row += [l_tracker[col_id][l_row_id] for col_id in range(l_num_cols)]
            join_pk_row_ids.add(l_row_id)
        else:
            row += ['' for _ in range(l_num_cols)]
            t_row += ['' for _ in range(l_num_cols)]
        for col_id in range(tot_cols):
            join_data[col_id].append(row[col_id])
            join_tracker[col_id].append(t_row[col_id])

    for row_id in range(l_num_rows):
        if row_id in join_pk_row_ids:
            continue
        row = ['' for _ in range(r_num_cols)] + [l_data[col_id][row_id] for col_id in range(l_num_cols)]
        t_row = ['' for _ in range(r_num_cols)] + [l_tracker[col_id][row_id] for col_id in range(l_num_cols)]
        for col_id in range(tot_cols):
            join_data[col_id].append(row[col_id])
            join_tracker[col_id].append(t_row[col_id])

    return join_data, join_tracker, join_header


def save_table(tab_id, header, data, tracker):

    with open(config.MERGED_PATH / f'{tab_id}.csv', 'w', encoding='latin1') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(header)
        csv_writer.writerows(zip(*data))

    with open(config.MERGED_PATH / f'{tab_id}_provenance.csv', 'w', encoding='latin1') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(header)
        csv_writer.writerows(zip(*tracker))


def validate_union(data, tracker, header):
    """
    Validate the candidate unions and assign them a goodness score (0 for useless unions)
    :param data: cell values as a list of lists (one list for each column)
    :param tracker: cell provenance as a list of lists (one list for each column)
    :param header: header (for both data and tracker)
    :return: a goodness score between 0 and 1 (0 for useless unions)
    """

    return 1.0


def validate_join(data, tracker, header):
    """
    Validate the candidate joins and assign them a goodness score (0 for useless joins).

    The score measures the fraction of errors in fully-joined rows that are
    correctable via cross-table functional dependencies discovered in the join.
    Both directions are considered: FK columns can help correct PK errors and
    vice versa.  An error cell is "correctable" when:
      - It belongs to a dependent column Y of a new cross-table FD  X -> Y
      - The determinant X is clean in that row and appears in at least one other
        row where Y is also clean (i.e., the determinant has duplicates providing
        redundancy for majority-vote correction).

    :param data: cell values as a list of lists (one list for each column)
    :param tracker: cell provenance as a list of lists (one list for each column)
    :param header: header (for both data and tracker)
    :return: a goodness score between 0 and 1 (0 for useless joins)
    """

    if not config.JOIN_VALIDATION:
        return 1.0

    num_cols = len(data)
    if num_cols == 0:
        return 0.0
    num_rows = len(data[0])
    if num_rows == 0:
        return 0.0

    # --- 1. Parse header to identify source tables and original column ids ---
    # Header format: '{tab_name}::{col_id}::{col_name}'
    col_sources = []
    table_names_ordered = []
    table_names_set = set()
    for h in header:
        parts = h.split('::')
        tab_name, orig_col_id = parts[0], int(parts[1])
        col_sources.append((tab_name, orig_col_id))
        if tab_name not in table_names_set:
            table_names_ordered.append(tab_name)
            table_names_set.add(tab_name)

    if len(table_names_ordered) < 2:
        return 0.0

    # --- 2. Load clean versions of source tables (only for error detection) ---
    clean_tables = {}
    for tab_name in table_names_set:
        with open(config.DIR_PATH / tab_name / 'clean.csv', encoding='latin1') as f:
            reader = csv.reader(f)
            next(reader)
            clean_tables[tab_name] = [list(col) for col in zip(*list(reader))]

    # --- 3. Single-pass: build per-column present / error row-sets ---
    present = [set() for _ in range(num_cols)]
    errors = [set() for _ in range(num_cols)]

    for c in range(num_cols):
        src_tab, src_col = col_sources[c]
        clean_col = clean_tables[src_tab][src_col]
        col_tracker = tracker[c]
        col_data = data[c]
        for r in range(num_rows):
            if col_tracker[r] != '':
                present[c].add(r)
                src_row = int(col_tracker[r].rsplit(' § ', 1)[1])
                if col_data[r] != clean_col[src_row]:
                    errors[c].add(r)

    del clean_tables

    # Fully-joined rows via set intersection
    fully_joined = set.intersection(*present) if present else set()

    total_errors = sum(len(errors[c] & fully_joined) for c in range(num_cols))
    if total_errors == 0:
        return 0.0

    # Clean rows: fully-joined with no error in any column
    clean_rows = fully_joined - set().union(*errors)
    if len(clean_rows) < 2:
        return 0.0

    # --- 4. Find cross-table 1→1 FDs on clean rows ---
    # Only consider FDs where determinant and dependent are from different tables.
    # Same-table FDs are not from the join (they’re in the original table), so we skip them.
    cross_fds = []
    for x in range(num_cols):
        x_tab = col_sources[x][0]
        x_col = data[x]
        for y in range(num_cols):
            if col_sources[y][0] == x_tab:
                continue
            val_map = {}
            holds = True
            for r in clean_rows:
                xv = x_col[r]
                yv = data[y][r]
                prev = val_map.get(xv)
                if prev is None:
                    val_map[xv] = yv
                elif prev != yv:
                    holds = False
                    break
            if holds:
                cross_fds.append((x, y))

    if not cross_fds:
        return 0.0

    # --- 5. Score via set operations ---
    correctable = set()
    for x, y in cross_fds:
        # Candidate rows: y present, x present and clean
        candidates = (present[y] & present[x]) - errors[x]

        det_groups = {}
        x_col = data[x]
        for r in candidates:
            det_groups.setdefault(x_col[r], []).append(r)

        y_errs = errors[y]
        for group_rows in det_groups.values():
            if len(group_rows) < 2:
                continue
            errs_in_group = y_errs.intersection(group_rows)
            if len(errs_in_group) == len(group_rows):
                continue
            for r in errs_in_group:
                correctable.add((y, r))

    return len(correctable) / total_errors


def merge_tables():

    db_conn = duckdb.connect(config.DB_PATH)

    # Create the directory to store merged tables
    shutil.rmtree(config.MERGED_PATH, ignore_errors=True)
    config.MERGED_PATH.mkdir(parents=True, exist_ok=True)

    # Map each table identifier to the corresponding name
    tab_names = {x[0]: x[1] for x in db_conn.execute("SELECT * FROM tab_idx").fetchall()}

    # Remove tables that are not part of the considered subset
    subset_names = {dir_name.name for dir_name in config.DIR_PATH.iterdir() if dir_name.is_dir()}
    for tab_id in list(tab_names.keys()):
        if tab_names[tab_id] not in subset_names:
            del tab_names[tab_id]

    # Map each table identifier to the corresponding number of tuples
    tab_lenghts = dict()
    for tab_id in sorted(list(tab_names.keys())):
        query = f"""
            SELECT COUNT(*)
            FROM cell_idx
            WHERE tab_id = {tab_id}
            AND col_id = 0
        """
        tab_lenghts[tab_id] = db_conn.execute(query).fetchone()[0]

    """
    Iterate on all tables to find candidate joins and unions
    """

    joins = list()
    unions = list()

    for tab_id in tqdm(sorted(list(tab_names.keys()))):

        tab_name = tab_names[tab_id]
        tab_ids = set(tab_names.keys()).difference({tab_id})  # identifiers of the corpus tables to consider in the search

        # Load the table in its dirty version
        with open(config.DIR_PATH / tab_name / 'dirty.csv', encoding='latin1') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            data = [list(col) for col in zip(*list(csv_reader))]
            num_cols = len(data)
            num_rows = len(data[0])

        # Get information on which columns are numeric
        is_numeric = {x[1]: x[-1] for x in db_conn.execute(f"SELECT * FROM col_idx WHERE tab_id = {tab_id}").fetchall()}

        # Visualize the table as a Polars dataframe
        # print(pl.DataFrame(data, schema=header))

        # Load the table in its clean version to detect erroneous cells
        with open(config.DIR_PATH / tab_name / 'clean.csv', encoding='latin1') as file:
            csv_reader = csv.reader(file)
            clean_data = [list(col)[1:] for col in zip(*list(csv_reader))]

        # Determine the clean values and the cardinality for each column
        clean_values = [[data[col_id][row_id] for row_id in range(num_rows) if data[col_id][row_id] == clean_data[col_id][row_id]]
                        for col_id in range(num_cols)]
        cardinalities = [len(set(clean_values[col_id]).difference({''})) / len(clean_values[col_id]) for col_id in range(num_cols)]

        # Find candidate joins for the table's primary keys
        top_joins = list()
        if config.JOIN:
            for col_id in range(num_cols):
                if cardinalities[col_id] == 1.0 and (True if config.JOIN_NUMERIC else not is_numeric[col_id]):
                    values = set(clean_values[col_id]).difference({''})
                    top_joins += find_joinable_tables(db_conn, tab_id, col_id, values, is_numeric[col_id], tab_lenghts, tab_ids, config.TOP_JOIN, config.JOIN_ROWS, config.JOIN_THRESHOLD)
        joins += top_joins

        # Find candidate unions
        top_unions = list()
        if config.UNION:
            top_unions += find_unionable_tables(db_conn, tab_id, clean_values, is_numeric, tab_lenghts, tab_ids, config.TOP_UNION, config.UNION_COLS, config.UNION_THRESHOLD)
        unions += top_unions

    # Sort candidate joins
    joins = sorted(joins, key=lambda x: x['score'], reverse=True)

    # Remove redundant candidate unions then sort them
    union_pairs = set()
    unions_dedup = list()
    for x in unions:
        tab_pair = (x['l_tab_id'], x['r_tab_id']) if x['l_tab_id'] <= x['r_tab_id'] else (x['r_tab_id'], x['l_tab_id'])
        if tab_pair in union_pairs:
            continue
        union_pairs.add(tab_pair)
        unions_dedup.append(x)
    unions = sorted(unions_dedup, key=lambda x: x['score'], reverse=True)

    tab_count = 0
    to_merge = set(tab_names.keys())
    merge_summary = []

    print(f'\n{len(unions)} candidate unions.')
    print(f'{len(joins)} candidate joins.')
    print(f'{len(to_merge)} tables to merge.')

    while to_merge:

        # If all joins and unions have already been performed, save the remaining tables as they are
        if not joins and not unions:
            for tab_id in list(to_merge):
                tab_name = tab_names[tab_id]
                with open(config.DIR_PATH / tab_name / 'dirty.csv', encoding='latin1') as file:
                    csv_reader = csv.reader(file)
                    header = next(csv_reader)
                    data = [list(col) for col in zip(*list(csv_reader))]
                    num_cols = len(data)
                    num_rows = len(data[0])
                tracker = [[f'{tab_name} § {col_id} § {row_id}' for row_id in range(num_rows)] for col_id in range(num_cols)]
                header = [f'{tab_name}::{col_id}::{header[col_id]}' for col_id in range(num_cols)]
                save_table(tab_count, header, data, tracker)
                merge_summary.append({
                    'output_id': tab_count,
                    'operation': 'none',
                    'left_table': tab_name,
                    'right_table': '',
                    'candidate_score': '',
                    'validation_score': '',
                    'why': 'Single table (no merge applied).',
                    'details': ''
                })
                tab_count += 1
                to_merge = to_merge.difference({tab_id})

        cand_merges = list()

        # If there are candidate unions, pick the top union and compare it against the top join for each involved table...
        if unions:
            cand_merges.append(unions[0])
            for tab_id in [unions[0]['l_tab_id'], unions[0]['r_tab_id']]:
                picked_join = None
                for x in joins:
                    if x['l_tab_id'] == tab_id or x['r_tab_id'] == tab_id:
                        tab_ids = (x['l_tab_id'], x['r_tab_id']) if x['l_tab_id'] < x['r_tab_id'] else (x['r_tab_id'], x['l_tab_id'])
                        if tab_ids != picked_join:
                            cand_merges.append(x)
                            picked_join = tab_ids
                        break
        # ...otherwise, pick the top join
        elif joins:
            cand_merges.append(joins[0])

        print(f'\n{len(cand_merges)} candidates to compare:\n')
        for x in cand_merges:
            print(x)

        # Materialize the candidate merges
        merged_tables = list()
        for x in cand_merges:
            l_tab_name = tab_names[x['l_tab_id']]
            r_tab_name = tab_names[x['r_tab_id']]
            l_tab_path = config.DIR_PATH / l_tab_name
            r_tab_path = config.DIR_PATH / r_tab_name
            if x['operation'] == 'union':
                merged_data, merged_tracker, merged_header = union_tables(l_tab_name, r_tab_name, l_tab_path, r_tab_path, x['mapping'])
            elif x['operation'] == 'join':
                merged_data, merged_tracker, merged_header = join_tables(l_tab_name, r_tab_name, l_tab_path, r_tab_path, x['mapping'])
            merged_tables.append({
                'candidate': x,
                'data': merged_data.copy(),
                'tracker': merged_tracker.copy(),
                'header': merged_header.copy()
            })

        # Validate the candidate merges and assign them a goodness score (0 for useless merges)
        scores = list()
        for x in merged_tables:
            if x['candidate']['operation'] == 'union':
                scores.append(validate_union(x['data'].copy(), x['tracker'].copy(), x['header'].copy()))  # remove copy if do not edit
            elif x['candidate']['operation'] == 'join':
                scores.append(validate_join(x['data'].copy(), x['tracker'].copy(), x['header'].copy()))  # remove copy if do not edit

        print(f'\nCandidate scores: {scores}')

        # Pick the best merge, save the merged table, remove all candidate unions and joins with those tables
        if any(x > 0 for x in scores):
            win_idx = max(enumerate(scores), key=lambda x: x[1])[0]
            cand = merged_tables[win_idx]['candidate']
            save_table(tab_count, merged_tables[win_idx]['header'], merged_tables[win_idx]['data'], merged_tables[win_idx]['tracker'])

            if cand['operation'] == 'join':
                why = (f"Join coverage (candidate)={cand['score']:.4f}; "
                       f"validation_score = fraction of errors correctable via cross-table 1→1 FDs.")
                merge_summary.append({
                    'output_id': tab_count,
                    'operation': 'join',
                    'left_table': tab_names[cand['l_tab_id']],
                    'right_table': tab_names[cand['r_tab_id']],
                    'candidate_score': cand['score'],
                    'validation_score': scores[win_idx],
                    'why': why,
                    'details': f"mapping={cand['mapping']} joined_tuples={cand.get('joined_tuples','')} dangling={cand.get('dangling_tuples','')}"
                })
            else:
                why = (f"Union coverage (candidate)={cand['score']:.4f}; "
                       f"validation_score = goodness of union (0 = useless).")
                merge_summary.append({
                    'output_id': tab_count,
                    'operation': 'union',
                    'left_table': tab_names[cand['l_tab_id']],
                    'right_table': tab_names[cand['r_tab_id']],
                    'candidate_score': cand['score'],
                    'validation_score': scores[win_idx],
                    'why': why,
                    'details': f"mapping={cand['mapping']} coverage={cand.get('coverage','')}"
                })

            tab_count += 1
            merged_ids = {cand['l_tab_id'], cand['r_tab_id']}
            to_merge = to_merge.difference(merged_ids)
            unions = [x for x in unions if x['l_tab_id'] not in merged_ids and x['r_tab_id'] not in merged_ids]
            joins = [x for x in joins if x['l_tab_id'] not in merged_ids and x['r_tab_id'] not in merged_ids]

            print(f'\nWinning candidate:')
            print(cand)

            print(f'\n{len(unions)} candidate unions.')
            print(f'{len(joins)} candidate joins.')
            print(f'{len(to_merge)} tables to merge.')

        # Remove useless merges
        for i in range(len(scores)):
            if scores[i] == 0:
                l_tab_id = merged_tables[i]['candidate']['l_tab_id']
                r_tab_id = merged_tables[i]['candidate']['r_tab_id']
                mapping = merged_tables[i]['candidate']['mapping']
                if merged_tables[i]['candidate']['operation'] == 'union':
                    unions = [x for x in unions if not (x['l_tab_id'] == l_tab_id and x['r_tab_id'] == r_tab_id and x['mapping'] == mapping)]
                elif merged_tables[i]['candidate']['operation'] == 'join':
                    joins = [x for x in joins if not (x['l_tab_id'] == l_tab_id and x['r_tab_id'] == r_tab_id and x['mapping'] == mapping)]

        print(f'\nRemoving {len([x for x in scores if x == 0])} further candidates which scored zero.')

        print(f'\n{len(unions)} candidate unions.')
        print(f'{len(joins)} candidate joins.')
        print(f'{len(to_merge)} tables to merge.')

    # Write merge summary to CSV
    if merge_summary:
        summary_path = config.MERGED_PATH / 'merge_summary.csv'
        with open(summary_path, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['output_id', 'operation', 'left_table', 'right_table', 'candidate_score', 'validation_score', 'why', 'details'])
            w.writeheader()
            for row in merge_summary:
                w.writerow({k: str(row.get(k, '')) for k in ['output_id', 'operation', 'left_table', 'right_table', 'candidate_score', 'validation_score', 'why', 'details']})
        print(f'\nWrote merge summary to {summary_path}')


def main():
    merge_tables()


if __name__ == '__main__':
    main()

