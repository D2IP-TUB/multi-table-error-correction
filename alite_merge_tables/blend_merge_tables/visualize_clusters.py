"""
visualize_clusters.py

Lightweight visualization of the BLEND-discovered clusters.
Run after discover_clusters.py has been executed.

Produces two plots saved to the output directory:
  1. cluster_graph.png    — graph of table relationships, nodes colored by cluster
  2. cluster_sizes.png    — bar chart of cluster size distribution
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
from pathlib import Path
from collections import Counter

import config
from discover_clusters import discover_clusters, UnionFind

import csv
import duckdb
from tqdm import tqdm
from merge_tables import find_joinable_tables, find_unionable_tables


def build_pair_graph():
    """Re-run discovery and return (graph, clusters, tab_names)."""

    db_conn = duckdb.connect(config.DB_PATH)
    tab_names = {x[0]: x[1] for x in db_conn.execute("SELECT * FROM tab_idx").fetchall()}
    subset_names = {d.name for d in config.DIR_PATH.iterdir() if d.is_dir()}
    tab_names = {k: v for k, v in tab_names.items() if v in subset_names}

    tab_lengths = {}
    for tab_id in sorted(tab_names):
        tab_lengths[tab_id] = db_conn.execute(
            f"SELECT COUNT(*) FROM cell_idx WHERE tab_id = {tab_id} AND col_id = 0"
        ).fetchone()[0]

    G = nx.Graph()
    G.add_nodes_from(tab_names.keys())

    for tab_id in tqdm(sorted(tab_names), desc="Building graph"):
        tab_name = tab_names[tab_id]
        other_ids = set(tab_names.keys()) - {tab_id}

        with open(config.DIR_PATH / tab_name / 'dirty.csv', encoding='latin1') as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [list(col) for col in zip(*list(reader))]
            num_cols = len(data)
            num_rows = len(data[0]) if data else 0

        if num_rows == 0:
            continue

        with open(config.DIR_PATH / tab_name / 'clean.csv', encoding='latin1') as f:
            reader = csv.reader(f)
            next(reader)
            clean_data = [list(col) for col in zip(*list(reader))]

        is_numeric = {
            x[1]: x[-1]
            for x in db_conn.execute(f"SELECT * FROM col_idx WHERE tab_id = {tab_id}").fetchall()
        }
        clean_values = [
            [data[c][r] for r in range(num_rows) if data[c][r] == clean_data[c][r]]
            for c in range(num_cols)
        ]
        cardinalities = [
            len(set(clean_values[c]) - {''}) / max(len(clean_values[c]), 1)
            for c in range(num_cols)
        ]

        if config.JOIN:
            for col_id in range(num_cols):
                if cardinalities[col_id] == 1.0 and (config.JOIN_NUMERIC or not is_numeric.get(col_id, False)):
                    values = list(set(clean_values[col_id]) - {''})
                    if not values:
                        continue
                    for match in find_joinable_tables(
                        db_conn, tab_id, col_id, values, is_numeric.get(col_id, False),
                        tab_lengths, other_ids, config.TOP_JOIN, config.JOIN_ROWS, config.JOIN_THRESHOLD
                    ):
                        G.add_edge(match['l_tab_id'], match['r_tab_id'],
                                   operation='join', score=match['score'])

        if config.UNION:
            for match in find_unionable_tables(
                db_conn, tab_id, clean_values, is_numeric, tab_lengths, other_ids,
                config.TOP_UNION, config.UNION_COLS, config.UNION_THRESHOLD
            ):
                G.add_edge(match['l_tab_id'], match['r_tab_id'],
                           operation='union', score=match['score'])

    # Build clusters via connected components
    clusters = {node: list(comp) for comp in nx.connected_components(G) for node in comp}
    # Map each node to its component id
    comp_id = {}
    for cid, comp in enumerate(nx.connected_components(G)):
        for node in comp:
            comp_id[node] = cid

    return G, comp_id, tab_names


def plot_graph(G, comp_id, tab_names, out_path: Path):
    """Draw the relationship graph, nodes colored by cluster."""

    n_comps = max(comp_id.values()) + 1 if comp_id else 1
    cmap = cm.get_cmap('tab20', max(n_comps, 2))
    node_colors = [cmap(comp_id[n] % 20) for n in G.nodes()]

    # Separate edge types for styling
    join_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get('operation') == 'join']
    union_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('operation') == 'union']

    fig, ax = plt.subplots(figsize=(14, 10))

    # Use spring layout; scale down for large graphs
    n = G.number_of_nodes()
    k = 2.0 / max(n ** 0.5, 1)
    pos = nx.spring_layout(G, k=k, seed=42)

    node_size = max(20, 600 // max(n, 1))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=join_edges,  edge_color='steelblue',  width=1.2, alpha=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=union_edges, edge_color='coral',      width=1.2, alpha=0.6, style='dashed', ax=ax)

    # Label only small graphs
    if n <= 40:
        labels = {node: tab_names[node][:15] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)

    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color='steelblue', lw=2,  label='join'),
        Line2D([0], [0], color='coral',     lw=2, linestyle='dashed', label='union'),
    ]
    ax.legend(handles=legend, loc='upper left', fontsize=9)

    n_components = len(set(comp_id.values()))
    ax.set_title(f'BLEND Table Relationship Graph\n'
                 f'{n} tables · {G.number_of_edges()} edges · {n_components} components',
                 fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved graph → {out_path}")


def plot_size_distribution(comp_id, out_path: Path):
    """Bar chart of cluster size distribution."""

    sizes = Counter(Counter(comp_id.values()).values())  # size → count
    xs = sorted(sizes.keys())
    ys = [sizes[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([str(x) for x in xs], ys, color='steelblue', edgecolor='white')

    for bar, count in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(count), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Cluster size (number of tables)', fontsize=11)
    ax.set_ylabel('Number of clusters', fontsize=11)
    ax.set_title('BLEND Cluster Size Distribution', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved size chart → {out_path}")


if __name__ == '__main__':
    out_dir = config.CLUSTERS_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building relationship graph from BLEND index...")
    G, comp_id, tab_names = build_pair_graph()

    plot_graph(G, comp_id, tab_names, out_dir / 'cluster_graph.png')
    plot_size_distribution(comp_id, out_dir / 'cluster_sizes.png')

    n_multi = sum(1 for c in Counter(comp_id.values()).values() if c > 1)
    print(f"\nSummary: {len(set(comp_id.values()))} total clusters "
          f"({n_multi} multi-table, {len(set(comp_id.values())) - n_multi} singletons)")
