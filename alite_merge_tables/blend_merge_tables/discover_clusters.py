"""
discover_clusters.py

Uses the BLEND index to discover joinable/unionable table pairs, groups
them into clusters via union-find (connected components), and writes each
cluster as a folder of dirty.csv files — ready to feed directly into ALITE.

Usage:
    python discover_clusters.py

Output:
    One folder per cluster under config.CLUSTERS_PATH.
    Each folder contains copies of the source dirty.csv files named <table_name>.csv.
    Singleton tables (no discovered relation) get their own single-table folder.
"""

import csv
import shutil
import duckdb
from pathlib import Path
from tqdm import tqdm

import config
from merge_tables import find_joinable_tables, find_unionable_tables


# ---------------------------------------------------------------------------
# Union-Find (path-compressed)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_sanitized(tab_dir: Path, dst: Path):
    """
    Copy dirty.csv to dst, replacing erroneous cells (dirty != clean) with
    empty string so ALITE treats them as nulls rather than real values.
    """
    with open(tab_dir / 'dirty.csv', encoding='latin1', newline='') as fd, \
         open(tab_dir / 'clean.csv', encoding='latin1', newline='') as fc:
        dirty_rows = list(csv.reader(fd))
        clean_rows = list(csv.reader(fc))

    header = dirty_rows[0]
    out_rows = [header]
    n_dirty, n_clean = len(dirty_rows) - 1, len(clean_rows) - 1
    if n_dirty != n_clean:
        raise ValueError(
            f"{tab_dir.name}: dirty.csv has {n_dirty} data rows but clean.csv has {n_clean}"
        )
    for d_row, c_row in zip(dirty_rows[1:], clean_rows[1:]):
        sanitized = [
            cell if cell == c_row[i] else ''   # blank out erroneous cells
            for i, cell in enumerate(d_row)
        ]
        out_rows.append(sanitized)

    with open(dst, 'w', encoding='latin1', newline='') as f:
        csv.writer(f).writerows(out_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_clusters(output_dir: Path):

    db_conn = duckdb.connect(config.DB_PATH)

    # Map table id → name; restrict to tables present in DIR_PATH
    tab_names = {x[0]: x[1] for x in db_conn.execute("SELECT * FROM tab_idx").fetchall()}
    subset_names = {d.name for d in config.DIR_PATH.iterdir() if d.is_dir()}
    tab_names = {k: v for k, v in tab_names.items() if v in subset_names}

    # Map table id → row count
    tab_lengths = {}
    for tab_id in sorted(tab_names):
        tab_lengths[tab_id] = db_conn.execute(
            f"SELECT COUNT(*) FROM cell_idx WHERE tab_id = {tab_id} AND col_id = 0"
        ).fetchone()[0]

    # -----------------------------------------------------------------------
    # Discover joinable / unionable pairs
    # -----------------------------------------------------------------------
    all_pairs = []

    for tab_id in tqdm(sorted(tab_names), desc="Discovering pairs"):
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

        # Join discovery
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
                        all_pairs.append((match['l_tab_id'], match['r_tab_id']))

        # Union discovery
        if config.UNION:
            for match in find_unionable_tables(
                db_conn, tab_id, clean_values, is_numeric, tab_lengths, other_ids,
                config.TOP_UNION, config.UNION_COLS, config.UNION_THRESHOLD
            ):
                all_pairs.append((match['l_tab_id'], match['r_tab_id']))

    print(f"\nFound {len(all_pairs)} raw pairs ({sum(1 for p in all_pairs)} total).")

    # -----------------------------------------------------------------------
    # Build clusters via union-find (connected components over all pairs)
    # -----------------------------------------------------------------------
    uf = UnionFind(tab_names.keys())
    for a, b in all_pairs:
        uf.union(a, b)

    clusters: dict[int, list[int]] = {}
    for tab_id in tab_names:
        root = uf.find(tab_id)
        clusters.setdefault(root, []).append(tab_id)

    n_multi = sum(1 for v in clusters.values() if len(v) > 1)
    sizes = sorted([len(v) for v in clusters.values()], reverse=True)
    print(f"Formed {len(clusters)} clusters ({n_multi} multi-table, {len(clusters) - n_multi} singletons).")
    print(f"Top cluster sizes: {sizes[:20]}{'...' if len(sizes) > 20 else ''}")

    # -----------------------------------------------------------------------
    # Write clusters to output_dir — one subfolder per cluster
    # -----------------------------------------------------------------------
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    for cluster_id, (root, members) in enumerate(sorted(clusters.items())):
        if len(members) == 1:
            folder_name = tab_names[members[0]]        # singleton: use table name
        else:
            folder_name = f"cluster_{cluster_id:04d}"  # multi-table cluster
        cluster_dir = output_dir / folder_name
        cluster_dir.mkdir()
        for tab_id in members:
            tab_name = tab_names[tab_id]
            dst = cluster_dir / f"{tab_name}.csv"
            _copy_sanitized(config.DIR_PATH / tab_name, dst)

    print(f"\nClusters written to: {output_dir}")
    return clusters


if __name__ == '__main__':
    discover_clusters(config.CLUSTERS_PATH)
