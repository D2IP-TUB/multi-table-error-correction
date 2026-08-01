"""
visualize_fd_clusters.py

Visualize ALITE merged clusters from fd_output and clusters_for_alite.

Produces:
  1. connected_components_grid.png   — one panel per multi-table cluster (readable)
  2. connected_components.png        — all multi-table nodes in one graph (singletons omitted)
  3. component_table_counts.png      — how many tables per connected component
  4. fd_cluster_dimensions.png       — rows and columns of each FD output cluster

Usage (from codes/):
    python visualize_fd_clusters.py
    python visualize_fd_clusters.py --fd-output ../results/alite/mit_dwh/fd_output
    python visualize_fd_clusters.py --corpus open_data_uk
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D

from alite_paths import CORPORA, get_paths

_CORPUS_DEFAULTS = {
    corpus: {
        "fd_output": get_paths(corpus)["fd_output"],
        "clusters": get_paths(corpus)["clusters"],
    }
    for corpus in CORPORA
}


def resolve_paths(args) -> tuple[Path, Path, Path]:
    """Return (fd_output, clusters_dir, out_dir)."""
    if args.corpus:
        fd_output = _CORPUS_DEFAULTS[args.corpus]["fd_output"]
        clusters_dir = _CORPUS_DEFAULTS[args.corpus]["clusters"]
    else:
        fd_output = Path(args.fd_output)
        clusters_dir = (
            Path(args.clusters_dir)
            if args.clusters_dir
            else fd_output.parent / "clusters_for_alite"
        )

    out_dir = Path(args.out_dir) if args.out_dir else fd_output / "plots"
    return fd_output.resolve(), clusters_dir.resolve(), out_dir.resolve()


def load_cluster_membership(clusters_dir: Path) -> dict[str, str]:
    """Map table stem (no .csv) -> cluster folder name."""
    table_to_cluster: dict[str, str] = {}
    if not clusters_dir.is_dir():
        return table_to_cluster

    for cluster_path in sorted(clusters_dir.iterdir()):
        if not cluster_path.is_dir():
            continue
        for csv_path in cluster_path.glob("*.csv"):
            table_to_cluster[csv_path.stem] = cluster_path.name
    return table_to_cluster


def load_fd_results(fd_output: Path) -> pd.DataFrame:
    """One row per FD result file with row/column counts."""
    records = []
    for csv_path in sorted(fd_output.glob("*.csv")):
        name = csv_path.name
        if name.endswith("_merged_cell_source_map.csv") or name.endswith(
            "_subsumption_map.csv"
        ):
            continue
        try:
            df = pd.read_csv(csv_path, encoding="latin1", on_bad_lines="skip", dtype=str)
            records.append(
                {
                    "cluster": csv_path.stem,
                    "rows": len(df),
                    "cols": len(df.columns),
                    "is_multi": csv_path.stem.startswith("cluster_"),
                }
            )
        except Exception as exc:
            print(f"  Warning: could not read {name}: {exc}")
    return pd.DataFrame(records)


def cluster_table_counts(clusters_dir: Path) -> pd.DataFrame:
    """Tables per cluster folder (input side, before FD)."""
    records = []
    for cluster_path in sorted(clusters_dir.iterdir()):
        if not cluster_path.is_dir():
            continue
        n_tables = len(list(cluster_path.glob("*.csv")))
        records.append(
            {
                "cluster": cluster_path.name,
                "n_tables": n_tables,
                "is_multi": cluster_path.name.startswith("cluster_"),
            }
        )
    return pd.DataFrame(records)


def build_graph_from_blend(corpus: str | None):
    """Reuse BLEND discovery graph if index + config are available."""
    blend_dir = Path(__file__).resolve().parent.parent / "blend_merge_tables"
    sys.path.insert(0, str(blend_dir))
    try:
        import config  # noqa: WPS433

        if corpus:
            config.CORPUS = corpus
            config.DIR_PATH = config._ISOLATED_DIRS[corpus]
            config.REAL_LAKES = config._BASE / "real_lakes" / corpus
            config.DB_PATH = config._BASE / f"{corpus}_blend_index.duckdb"

        from visualize_clusters import build_pair_graph  # noqa: WPS433

        if not config.DB_PATH.exists():
            return None, None, None
        G, comp_id, tab_names = build_pair_graph()
        return G, comp_id, tab_names
    except Exception as exc:
        print(f"  BLEND graph unavailable ({exc}); using cluster-folder edges.")
        return None, None, None


def build_graph_from_clusters(clusters_dir: Path) -> tuple[nx.Graph, dict[str, int], dict[int, str]]:
    """Fallback graph: connect all tables that share a multi-table cluster folder."""
    G = nx.Graph()
    comp_id: dict[str, int] = {}
    tab_names: dict[int, str] = {}
    next_id = 0

    for cluster_path in sorted(clusters_dir.iterdir()):
        if not cluster_path.is_dir():
            continue
        tables = sorted(p.stem for p in cluster_path.glob("*.csv"))
        if not tables:
            continue

        cid = next_id
        next_id += 1
        for t in tables:
            G.add_node(t)
            comp_id[t] = cid
            tab_names[t] = t

        if len(tables) > 1:
            hub = tables[0]
            for other in tables[1:]:
                G.add_edge(hub, other, operation="cluster")

    return G, comp_id, tab_names


def relabel_graph_to_table_names(
    G: nx.Graph, tab_names: dict
) -> nx.Graph:
    """BLEND graphs use integer tab_id nodes; relabel to table name strings."""
    mapping = {}
    for node in G.nodes():
        name = tab_names.get(node, node)
        mapping[node] = name if isinstance(name, str) else str(name)
    return nx.relabel_nodes(G, mapping)


def cluster_groups(clusters_dir: Path) -> dict[str, list[str]]:
    """cluster folder name -> sorted table stems in that folder."""
    groups: dict[str, list[str]] = {}
    for cluster_path in sorted(clusters_dir.iterdir()):
        if not cluster_path.is_dir():
            continue
        tables = sorted(p.stem for p in cluster_path.glob("*.csv"))
        if tables:
            groups[cluster_path.name] = tables
    return groups


def _draw_cluster_subgraph(ax, G: nx.Graph, tables: list[str], title: str):
    """Draw one connected component (one merged cluster) on ax."""
    sub = G.subgraph(tables).copy()
    n = sub.number_of_nodes()
    if n == 0:
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        return

    join_edges = [(u, v) for u, v, d in sub.edges(data=True) if d.get("operation") == "join"]
    union_edges = [(u, v) for u, v, d in sub.edges(data=True) if d.get("operation") == "union"]
    other_edges = [
        (u, v) for u, v, d in sub.edges(data=True)
        if d.get("operation") not in ("join", "union")
    ]

    k = 1.2 if n <= 4 else 0.8
    pos = nx.spring_layout(sub, k=k, seed=42)
    node_size = max(80, 500 // max(n, 1))

    nx.draw_networkx_nodes(sub, pos, node_color="steelblue", node_size=node_size, alpha=0.9, ax=ax)
    if join_edges:
        nx.draw_networkx_edges(sub, pos, edgelist=join_edges, edge_color="steelblue", width=1.5, ax=ax)
    if union_edges:
        nx.draw_networkx_edges(
            sub, pos, edgelist=union_edges, edge_color="coral", width=1.5, style="dashed", ax=ax
        )
    if other_edges:
        nx.draw_networkx_edges(sub, pos, edgelist=other_edges, edge_color="gray", width=1.0, ax=ax)

    labels = {t: t[:14] + ("…" if len(t) > 14 else "") for t in sub.nodes()}
    nx.draw_networkx_labels(sub, pos, labels, font_size=5, ax=ax)
    ax.set_title(f"{title}\n{n} tables · {sub.number_of_edges()} edges", fontsize=8)
    ax.axis("off")


def plot_connected_components_grid(
    G: nx.Graph,
    groups: dict[str, list[str]],
    out_path: Path,
    edge_source: str,
):
    """
    One panel per multi-table merged cluster — the readable connected-component view.
    Singletons are omitted (they have no internal edges).
    """
    multi = [(name, tables) for name, tables in groups.items() if len(tables) > 1]
    if not multi:
        print("  No multi-table clusters to plot.")
        return

    n_panels = len(multi)
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes_flat = [axes] if n_panels == 1 else list(axes.flat)

    for ax, (cluster_name, tables) in zip(axes_flat, multi):
        _draw_cluster_subgraph(ax, G, tables, cluster_name)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    n_singletons = sum(1 for tables in groups.values() if len(tables) == 1)
    fig.suptitle(
        f"Connected components — multi-table clusters only ({edge_source})\n"
        f"{n_panels} merged clusters shown · {n_singletons} singletons omitted",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def plot_connected_components_multi_combined(
    G: nx.Graph,
    groups: dict[str, list[str]],
    out_path: Path,
    edge_source: str,
):
    """Single graph of all multi-table cluster nodes only (no singleton clutter)."""
    multi_tables = [t for tables in groups.values() if len(tables) > 1 for t in tables]
    if not multi_tables:
        return

    sub = G.subgraph(multi_tables).copy()
    n = sub.number_of_nodes()

    # Color by cluster folder
    table_to_cluster = {t: c for c, ts in groups.items() for t in ts}
    cluster_ids = {c: i for i, c in enumerate(sorted({table_to_cluster[t] for t in sub.nodes()}))}
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(cluster_ids), 2))
    node_colors = [cmap(cluster_ids[table_to_cluster[n]] % 20) for n in sub.nodes()]

    join_edges = [(u, v) for u, v, d in sub.edges(data=True) if d.get("operation") == "join"]
    union_edges = [(u, v) for u, v, d in sub.edges(data=True) if d.get("operation") == "union"]

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(sub, k=1.5 / max(n**0.25, 1), seed=42)
    node_size = max(40, 800 // max(n, 1))

    nx.draw_networkx_nodes(sub, pos, node_color=node_colors, node_size=node_size, alpha=0.9, ax=ax)
    if join_edges:
        nx.draw_networkx_edges(sub, pos, edgelist=join_edges, edge_color="steelblue", width=1.2, alpha=0.7, ax=ax)
    if union_edges:
        nx.draw_networkx_edges(
            sub, pos, edgelist=union_edges, edge_color="coral", width=1.2, alpha=0.7, style="dashed", ax=ax
        )

    labels = {t: t[:16] for t in sub.nodes()}
    nx.draw_networkx_labels(sub, pos, labels, font_size=5, ax=ax)

    legend = [
        Line2D([0], [0], color="steelblue", lw=2, label="join"),
        Line2D([0], [0], color="coral", lw=2, linestyle="dashed", label="union"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    n_singletons = sum(1 for tables in groups.values() if len(tables) == 1)
    ax.set_title(
        f"Multi-table connected components ({edge_source})\n"
        f"{n} tables in {len(cluster_ids)} clusters · {sub.number_of_edges()} edges "
        f"· {n_singletons} singletons omitted",
        fontsize=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_connected_components(
    G: nx.Graph,
    groups: dict[str, list[str]],
    out_path: Path,
    edge_source: str,
):
    """Legacy single-file output → now points to the multi-table combined view."""
    plot_connected_components_multi_combined(G, groups, out_path, edge_source)


def plot_component_table_counts(counts_df: pd.DataFrame, out_path: Path):
    """Bar chart: number of tables per connected component."""
    if counts_df.empty:
        return

    sizes = Counter(counts_df["n_tables"].tolist())
    xs = sorted(sizes.keys())
    ys = [sizes[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([str(x) for x in xs], ys, color="steelblue", edgecolor="white")
    for bar, count in zip(bars, ys):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Tables per cluster", fontsize=11)
    ax.set_ylabel("How many clusters have this size", fontsize=11)
    n_singletons = sizes.get(1, 0)
    n_multi = sum(v for k, v in sizes.items() if k > 1)
    max_size = max(xs) if xs else 0
    ax.set_title(
        f"Cluster size distribution\n"
        f"{n_singletons} singletons (1 table) · {n_multi} multi-table clusters · max size = {max_size}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_fd_dimensions(fd_df: pd.DataFrame, out_path: Path):
    """Grouped bar chart: rows and columns per FD output cluster."""
    if fd_df.empty:
        print("  No FD result CSVs found.")
        return

    fd_df = fd_df.sort_values("cluster").reset_index(drop=True)
    x = range(len(fd_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, len(fd_df) * 0.35), 6))
    ax.bar([i - width / 2 for i in x], fd_df["rows"], width, label="rows", color="steelblue")
    ax.bar([i + width / 2 for i in x], fd_df["cols"], width, label="columns", color="coral")

    ax.set_xticks(list(x))
    ax.set_xticklabels(fd_df["cluster"], rotation=90, fontsize=7)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("FD output size per merged cluster", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_fd_dimensions_scatter(fd_df: pd.DataFrame, counts_df: pd.DataFrame, out_path: Path):
    """Scatter: input tables vs FD output rows (multi-table clusters only)."""
    if fd_df.empty or counts_df.empty:
        return

    merged = fd_df.merge(counts_df, on="cluster", how="inner")
    multi = merged[merged["is_multi_x"]]

    if multi.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(multi["n_tables"], multi["rows"], alpha=0.75, s=60, c="steelblue")
    for _, row in multi.iterrows():
        ax.annotate(
            row["cluster"].replace("cluster_", ""),
            (row["n_tables"], row["rows"]),
            fontsize=6,
            alpha=0.8,
        )
    ax.set_xlabel("Input tables in component", fontsize=11)
    ax.set_ylabel("FD output rows", fontsize=11)
    ax.set_title("Input size vs FD output rows (multi-table clusters)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ALITE FD merged clusters")
    parser.add_argument("--corpus", choices=_CORPUS_DEFAULTS, default=None)
    parser.add_argument("--fd-output", default=None, help="Path to fd_output directory")
    parser.add_argument("--clusters-dir", default=None, help="Path to clusters_for_alite")
    parser.add_argument("--out-dir", default=None, help="Where to save plots")
    parser.add_argument(
        "--no-blend",
        action="store_true",
        help="Skip BLEND index; use cluster-folder edges only",
    )
    args = parser.parse_args()

    if args.corpus:
        fd_output, clusters_dir, out_dir = resolve_paths(args)
    elif args.fd_output:
        fd_output = Path(args.fd_output).resolve()
        clusters_dir = (
            Path(args.clusters_dir).resolve()
            if args.clusters_dir
            else fd_output.parent / "clusters_for_alite"
        )
        out_dir = Path(args.out_dir).resolve() if args.out_dir else fd_output / "plots"
    else:
        fd_output, clusters_dir, out_dir = resolve_paths(
            argparse.Namespace(
                corpus="open_data_uk",
                fd_output=None,
                clusters_dir=None,
                out_dir=None,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"FD output:  {fd_output}")
    print(f"Clusters:   {clusters_dir}")
    print(f"Plots dir:  {out_dir}\n")

    counts_df = cluster_table_counts(clusters_dir)
    fd_df = load_fd_results(fd_output)

    edge_source = "cluster folders"
    G, comp_id, tab_names = None, None, None
    if not args.no_blend:
        corpus_key = args.corpus or (
            "mit_dwh" if "mit_dwh" in str(fd_output) else "open_data_uk"
        )
        G, comp_id, tab_names = build_graph_from_blend(corpus_key)
        if G is not None:
            edge_source = "BLEND join/union pairs"

    if G is None:
        G, comp_id, tab_names = build_graph_from_clusters(clusters_dir)
    else:
        G = relabel_graph_to_table_names(G, tab_names)
        tab_names = {n: n for n in G.nodes()}

    groups = cluster_groups(clusters_dir)

    plot_connected_components_grid(
        G, groups, out_dir / "connected_components_grid.png", edge_source
    )
    plot_connected_components(
        G, groups, out_dir / "connected_components.png", edge_source
    )
    plot_component_table_counts(counts_df, out_dir / "component_table_counts.png")
    plot_fd_dimensions(fd_df, out_dir / "fd_cluster_dimensions.png")
    plot_fd_dimensions_scatter(fd_df, counts_df, out_dir / "fd_input_vs_output_rows.png")

    # Summary table
    if not fd_df.empty:
        summary_path = out_dir / "fd_cluster_sizes.csv"
        summary = fd_df.merge(counts_df, on="cluster", how="left", suffixes=("_fd", "_input"))
        summary.to_csv(summary_path, index=False)
        print(f"Saved → {summary_path}")

    n_multi = int(counts_df["is_multi"].sum()) if not counts_df.empty else 0
    print(
        f"\nSummary: {len(counts_df)} components "
        f"({n_multi} multi-table), {len(fd_df)} FD outputs"
    )


if __name__ == "__main__":
    main()
