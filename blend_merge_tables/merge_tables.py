import csv
import duckdb
import networkx as nx
import polars as pl
import random
import shutil
from typing import Tuple
from tqdm import tqdm

import config
from error_cells import get_error_profile, is_gt_mode
from merge_validation import (
    CandidateReportRow,
    append_candidate_report_rows,
    blend_discovery_metrics,
    print_validation_summary,
    score_merge,
    validation_result_row,
    write_validation_reports,
)


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

    if not values:
        return []

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


def _union_matching_edge(node_a: str, node_b: str) -> Tuple[int, int]:
    """Normalize a bipartite matching edge to (left_col_id, right_col_id)."""

    def parse_node(node: str) -> Tuple[str, int]:
        if node.startswith('l_'):
            return 'l', int(node.removeprefix('l_'))
        if node.startswith('r_'):
            return 'r', int(node.removeprefix('r_'))
        raise ValueError(f'unexpected union matching node: {node!r}')

    side_a, col_a = parse_node(node_a)
    side_b, col_b = parse_node(node_b)
    if side_a == 'l' and side_b == 'r':
        return col_a, col_b
    if side_a == 'r' and side_b == 'l':
        return col_b, col_a
    raise ValueError(f'union matching edge has same side twice: {node_a!r}, {node_b!r}')


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
        values = set(clean_values[col_id]).difference({''})
        if not values:
            continue
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
        mapping = {
            _union_matching_edge(node_a, node_b)
            for node_a, node_b in nx.max_weight_matching(g)
        }
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

    l_clean_data = None
    if is_gt_mode():
        with open(l_tab_path / 'clean.csv', encoding='latin1') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            l_clean_data = [list(col) for col in zip(*list(csv_reader))]

    # Load the right table
    with open(r_tab_path / 'dirty.csv', encoding='latin1') as file:
        csv_reader = csv.reader(file)
        r_header = next(csv_reader)
        r_data = [list(col) for col in zip(*list(csv_reader))]
        r_num_cols = len(r_data)
        r_num_rows = len(r_data[0])

    r_clean_data = None
    if is_gt_mode():
        with open(r_tab_path / 'clean.csv', encoding='latin1') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            r_clean_data = [list(col) for col in zip(*list(csv_reader))]

    l_profile = get_error_profile(l_tab_path, clean_data=l_clean_data)
    r_profile = get_error_profile(r_tab_path, clean_data=r_clean_data)

    # Generate the cell trackers for the two tables
    # Each cell stores a string 'table_name § column_name § row_idx'
    l_tracker = [[f'{l_tab_name} § {col_id} § {row_id}' for row_id in range(l_num_rows)] for col_id in range(l_num_cols)]
    r_tracker = [[f'{r_tab_name} § {col_id} § {row_id}' for row_id in range(r_num_rows)] for col_id in range(r_num_cols)]

    # Track the row index of each clean value in the primary key (left table)
    pk_id = list(mapping.keys())[0]
    fk_id = mapping[pk_id]
    pk_values = {
        l_data[pk_id][row_id]: row_id
        for row_id in range(l_num_rows)
        if l_profile.is_clean(row_id, pk_id, dirty_value=l_data[pk_id][row_id])
    }

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
        if r_profile.is_clean(row_id, fk_id, dirty_value=r_data[fk_id][row_id]) and r_data[fk_id][row_id] in pk_values:
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


MERGE_SUMMARY_FIELDS = [
    "output_id", "operation", "left_table", "right_table",
    "blend_score", "blend_coverage", "validation_score",
    "errors_in_merged", "correctable_before", "correctable_after",
    "newly_correctable", "correctable_lost",
    "fds_before", "fds_after", "fds_introduced", "fds_broken", "fds_retained",
    "mapping", "joined_tuples", "dangling_tuples", "notes",
]


def _tab_pair_key(cand: dict) -> tuple[int, int]:
    l_tab_id, r_tab_id = cand['l_tab_id'], cand['r_tab_id']
    return (l_tab_id, r_tab_id) if l_tab_id < r_tab_id else (r_tab_id, l_tab_id)


def _first_alternate_for_table(alternates: list, tab_id: int) -> dict | None:
    for cand in alternates:
        if cand['l_tab_id'] == tab_id or cand['r_tab_id'] == tab_id:
            return cand
    return None


def build_candidate_merges(joins: list, unions: list) -> list:
    """
    Build the candidate merge group for the next comparison round.

    With union priority, seed the group with the top union and add competing
    joins for each involved table. With join priority, do the converse.

    When MERGE_VALIDATION is False (paper default / Algorithm 1), only the
    priority-side top candidate is returned — no competing alternates.
    """
    priority = str(config.MERGE_PRIORITY).strip().lower()
    if priority not in {'union', 'join'}:
        raise ValueError(f"MERGE_PRIORITY must be 'union' or 'join', got {config.MERGE_PRIORITY!r}")

    if priority == 'union':
        primary, alternates = unions, joins
    else:
        primary, alternates = joins, unions

    cand_merges: list = []
    if primary:
        cand_merges.append(primary[0])
        if config.MERGE_VALIDATION:
            picked_pairs: set[tuple[int, int]] = set()
            for tab_id in [primary[0]['l_tab_id'], primary[0]['r_tab_id']]:
                alternate = _first_alternate_for_table(alternates, tab_id)
                if alternate is None:
                    continue
                tab_ids = _tab_pair_key(alternate)
                if tab_ids not in picked_pairs:
                    cand_merges.append(alternate)
                    picked_pairs.add(tab_ids)
    elif alternates:
        cand_merges.append(alternates[0])

    return cand_merges


def _merge_summary_notes(cand: dict, vresult, n_candidates: int) -> str:
    blend = blend_discovery_metrics(cand)
    if cand["operation"] == "join":
        blend_desc = (
            f"BLEND join coverage={blend['blend_coverage']:.4f} "
            f"(joined_tuples={cand.get('joined_tuples', '')}, "
            f"dangling={cand.get('dangling_tuples', '')})"
        )
    else:
        blend_desc = (
            f"BLEND union column-coverage={blend['blend_coverage']:.4f}, "
            f"overlap-sum={blend['blend_score']:.4f} "
            f"(sum of per-column overlap ratios; not capped at 1)"
        )
    return (
        f"Selected as best of {n_candidates} compared candidate(s) by validation_score. "
        f"{blend_desc}. "
        f"Validation on full merged table: "
        f"{vresult.newly_correctable}/{vresult.errors_in_merged} erroneous source cells "
        f"newly FD-correctable after merge "
        f"(score={vresult.score:.4f}); "
        f"{vresult.correctable_lost} previously correctable cells lost on merged table."
    )


def _build_merge_summary_row(
    output_id: int,
    cand: dict,
    left_table: str,
    right_table: str,
    vresult,
    n_candidates: int,
) -> dict:
    blend = blend_discovery_metrics(cand)
    row = {
        "output_id": output_id,
        "operation": cand["operation"],
        "left_table": left_table,
        "right_table": right_table,
        "blend_score": blend["blend_score"],
        "blend_coverage": blend["blend_coverage"],
        "mapping": cand.get("mapping", ""),
        "joined_tuples": cand.get("joined_tuples", ""),
        "dangling_tuples": cand.get("dangling_tuples", ""),
        "notes": _merge_summary_notes(cand, vresult, n_candidates),
        **validation_result_row(vresult),
    }
    return row


def save_table(tab_id, header, data, tracker):

    with open(config.MERGED_PATH / f'{tab_id}.csv', 'w', encoding='latin1') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(header)
        csv_writer.writerows(zip(*data))

    with open(config.MERGED_PATH / f'{tab_id}_provenance.csv', 'w', encoding='latin1') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(header)
        csv_writer.writerows(zip(*tracker))


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

        # Determine which cell values are trusted for overlap / join discovery
        clean_data = None
        if is_gt_mode():
            with open(config.DIR_PATH / tab_name / 'clean.csv', encoding='latin1') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)
                clean_data = [list(col) for col in zip(*list(csv_reader))]

        error_profile = get_error_profile(config.DIR_PATH / tab_name, clean_data=clean_data)
        clean_values = [
            [
                data[col_id][row_id]
                for row_id in range(num_rows)
                if error_profile.is_clean(row_id, col_id, dirty_value=data[col_id][row_id])
            ]
            for col_id in range(num_cols)
        ]
        cardinalities = [
            len(set(clean_values[col_id]).difference({''})) / len(clean_values[col_id])
            if clean_values[col_id] else 0.0
            for col_id in range(num_cols)
        ]

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
    validation_candidate_rows: list = []
    validation_fd_rows: list = []
    validation_distribution_rows: list = []
    comparison_group_id = 0

    print(f'\n{len(unions)} candidate unions.')
    print(f'{len(joins)} candidate joins.')
    print(f'{len(to_merge)} tables to merge.')
    print(f'MERGE_PRIORITY = {config.MERGE_PRIORITY}')

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
                    'notes': 'Single table (no merge applied).',
                })
                tab_count += 1
                to_merge = to_merge.difference({tab_id})

        cand_merges = build_candidate_merges(joins, unions)

        print(f'\n{len(cand_merges)} candidates to compare:\n')
        for x in cand_merges:
            print(x)

        if not cand_merges:
            continue

        comparison_group_id += 1

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
        validation_results = list()
        for x in merged_tables:
            cand = x['candidate']
            vresult = score_merge(
                x['data'].copy(),
                x['tracker'].copy(),
                x['header'].copy(),
                cand['operation'],
                left_table=tab_names[cand['l_tab_id']],
                right_table=tab_names[cand['r_tab_id']],
                mapping=cand['mapping'],
            )
            validation_results.append(vresult)
            scores.append(vresult.score)

        print(f'\nCandidate scores: {scores}')
        for i, vresult in enumerate(validation_results):
            print(f"  [{i}] {vresult.summary()}")

        win_idx = None
        if any(x > 0 for x in scores):
            blend_scores = [
                blend_discovery_metrics(x['candidate'])['blend_score']
                for x in merged_tables
            ]
            win_idx = max(
                range(len(scores)),
                key=lambda i: (scores[i], blend_scores[i]),
            )

        for i, x in enumerate(merged_tables):
            cand = x['candidate']
            blend = blend_discovery_metrics(cand)
            append_candidate_report_rows(
                validation_candidate_rows,
                validation_fd_rows,
                validation_distribution_rows,
                CandidateReportRow(
                    group_id=comparison_group_id,
                    candidate_idx=i,
                    operation=cand['operation'],
                    left_table=tab_names[cand['l_tab_id']],
                    right_table=tab_names[cand['r_tab_id']],
                    blend_score=blend['blend_score'],
                    blend_coverage=blend['blend_coverage'],
                    validation_score=scores[i],
                    selected=(i == win_idx),
                    result=validation_results[i],
                ),
            )

        # Pick the best merge, save the merged table, remove all candidate unions and joins with those tables
        if win_idx is not None:
            cand = merged_tables[win_idx]['candidate']
            save_table(tab_count, merged_tables[win_idx]['header'], merged_tables[win_idx]['data'], merged_tables[win_idx]['tracker'])

            vresult = validation_results[win_idx]
            merge_summary.append(_build_merge_summary_row(
                tab_count,
                cand,
                tab_names[cand['l_tab_id']],
                tab_names[cand['r_tab_id']],
                vresult,
                len(cand_merges),
            ))

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
            w = csv.DictWriter(f, fieldnames=MERGE_SUMMARY_FIELDS, extrasaction='ignore')
            w.writeheader()
            for row in merge_summary:
                w.writerow({k: str(row.get(k, '')) for k in MERGE_SUMMARY_FIELDS})
        print(f'\nWrote merge summary to {summary_path}')

    if validation_candidate_rows:
        write_validation_reports(
            validation_candidate_rows,
            validation_fd_rows,
            validation_distribution_rows,
            config.MERGED_PATH,
        )
        print_validation_summary(validation_candidate_rows, validation_fd_rows)
        print(f'\nWrote validation reports to {config.MERGED_PATH}/validation_*.csv')


def main():
    merge_tables()


if __name__ == '__main__':
    main()

