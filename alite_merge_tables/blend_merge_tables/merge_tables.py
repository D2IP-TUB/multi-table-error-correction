"""BLEND join/union discovery helpers used by discover_clusters (ALITE prep)."""

from typing import Tuple

import duckdb
import networkx as nx

import config


def find_joinable_tables(db_conn, tab_id, col_id, values, is_numeric, tab_lengths, tab_ids, top_k, min_rows, threshold):
    """
    Given a primary key, find its top_k foreign keys in the corpus.
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

    return sorted(top_cols, key=lambda x: x['score'], reverse=True)[:min(top_k, len(top_cols))]


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
    """
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

    return sorted(top_unions, key=lambda x: x['score'], reverse=True)[:min(top_k, len(top_unions))]
