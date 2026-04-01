#!/usr/bin/env python3
"""
Export averaged F1 (and P/R) per labeling budget for each ablation series,
write PGFPlots-ready CSVs + a LaTeX snippet (tikz/pgfplots).
The `.tex` file is **one** figure: feature panels include drop/only leave-one-out curves; pattern
enforcement (perfect-detector ablation) is its own panel. Layout is two rows (3+2) when five
panels are present, otherwise one row.

Expects output folder names:
  output_<dataset>_<exec_idx>_<budget>_<variant>
e.g. output_Quintet_3_1_10_all_features

Run from repo root:
  python export_ablation_plots.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_OUT_BASE = _SCRIPT_DIR / "latex_ablation"
_DATA_DIR = _OUT_BASE / "data"
_TEX_PATH = _OUT_BASE / "quintet_ablations_f1.tex"

# ---------------------------------------------------------------------------
# Parse output_Quintet_3_1_10_all_features  ->  dataset, iter, budget, variant
# ---------------------------------------------------------------------------

_OUTPUT_RE = re.compile(r"^output_(.+)_(\d+)_(\d+)_(.+)$")


def parse_output_dir_name(dirname: str) -> Optional[Tuple[str, int, int, str]]:
    m = _OUTPUT_RE.match(dirname)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)


def load_f1(output_dir: Path) -> Optional[float]:
    p = output_dir / "aggregated_results.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        f1 = data.get("aggregate_stats", {}).get("overall_f1")
        if f1 is None:
            return None
        return float(f1)
    except Exception:
        return None


def collect_series(
    results_root: Path,
    dataset_filter: str = "Quintet_3",
    series_key: Optional[Callable[[str], str]] = None,
) -> Dict[str, Dict[int, List[float]]]:
    """
    Returns {series_name: {budget: [f1_run1, f1_run2, ...]}}.
    series_key(variant) -> display key; default is variant string.
    """
    series_key = series_key or (lambda v: v)
    acc: DefaultDict[str, DefaultDict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    if not results_root.is_dir():
        print(f"  [skip] missing directory: {results_root}")
        return {}

    for child in sorted(results_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("output_"):
            continue
        parsed = parse_output_dir_name(child.name)
        if not parsed:
            print(f"  [skip] bad name: {child.name}")
            continue
        dataset, _iter, budget, variant = parsed
        if dataset != dataset_filter:
            continue
        f1 = load_f1(child)
        if f1 is None:
            print(f"  [skip] no JSON: {child.name}")
            continue
        key = series_key(variant)
        acc[key][budget].append(f1)

    # convert to plain dict
    return {k: dict(v) for k, v in acc.items()}


def average_series(
    raw: Dict[str, Dict[int, List[float]]],
) -> Dict[str, List[Tuple[int, float, float, float]]]:
    """
    For each series, list of (budget, mean_f1, mean_precision placeholder, mean_recall placeholder).
    We only load f1 from JSON for speed; optional extend to load P/R.
    """
    out: Dict[str, List[Tuple[int, float, float, float]]] = {}
    for series, by_budget in raw.items():
        rows: List[Tuple[int, float, float, float]] = []
        for budget in sorted(by_budget.keys()):
            vals = by_budget[budget]
            if not vals:
                continue
            mean_f1 = sum(vals) / len(vals)
            rows.append((budget, mean_f1, float("nan"), float("nan")))
        if rows:
            out[series] = rows
    return out


def load_metrics_full(output_dir: Path) -> Optional[Dict[str, float]]:
    p = output_dir / "aggregated_results.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        agg = data.get("aggregate_stats", {})
        return {
            "f1": float(agg.get("overall_f1", 0)),
            "precision": float(agg.get("overall_precision", 0)),
            "recall": float(agg.get("overall_recall", 0)),
        }
    except Exception:
        return None


def collect_series_metrics(
    results_root: Path,
    dataset_filter: str = "Quintet_3",
    series_key: Optional[Callable[[str], str]] = None,
) -> Dict[str, Dict[int, List[Dict[str, float]]]]:
    series_key = series_key or (lambda v: v)
    acc: DefaultDict[str, DefaultDict[int, List[Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    if not results_root.is_dir():
        return {}
    for child in sorted(results_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("output_"):
            continue
        parsed = parse_output_dir_name(child.name)
        if not parsed:
            continue
        dataset, _it, budget, variant = parsed
        if dataset != dataset_filter:
            continue
        m = load_metrics_full(child)
        if m is None:
            continue
        acc[series_key(variant)][budget].append(m)
    return {k: dict(v) for k, v in acc.items()}


def average_metrics(
    raw: Dict[str, Dict[int, List[Dict[str, float]]]],
) -> Dict[str, List[Tuple[int, float, float, float]]]:
    out: Dict[str, List[Tuple[int, float, float, float]]] = {}
    for series, by_budget in raw.items():
        rows = []
        for budget in sorted(by_budget.keys()):
            reps = by_budget[budget]
            if not reps:
                continue
            f1 = sum(r["f1"] for r in reps) / len(reps)
            pr = sum(r["precision"] for r in reps) / len(reps)
            rc = sum(r["recall"] for r in reps) / len(reps)
            rows.append((budget, f1, pr, rc))
        if rows:
            out[series] = rows
    return out


def write_series_csv(
    series_name: str,
    rows: List[Tuple[int, float, float, float]],
    data_dir: Path,
    file_stem: str,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{file_stem}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["labeling_budget", "f1_score", "precision", "recall"])
        for budget, f1, pr, rc in rows:
            w.writerow([budget, f"{f1:.6f}", f"{pr:.6f}", f"{rc:.6f}"])
    return path


def safe_filename(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", s).strip("_").lower()


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

# Labeling budgets used in experiments (exact x-axis ticks in figures)
BUDGET_TICKS = [10, 22, 96, 173]

# Total horizontal fill per row; each minipage width = ROW_WIDTH_FRAC / (#panels in that row).
ROW_WIDTH_FRAC = 0.98
# Axis height inside each minipage, as fraction of that minipage's \linewidth.
# Taller than width inside each minipage so $F_1$ curves stay readable in a 1×4 row.
AXIS_HEIGHT_RATIO = 0.82
# Slightly smaller marks when four panels sit side-by-side
MARK_SIZE_PT = 1.0

# (file_stem, legend entry, color, mark)
STYLE_CYCLE = [
    ("blue", "o"),
    ("teal", "triangle"),
    ("cyan", "oplus"),
    ("lime!80!black", "triangle*"),
    ("orange", "*"),
    ("gray", "square"),
    ("violet", "+"),
    ("black", "star"),
    ("green!50!black", "diamond*"),
    ("red!70!black", "x"),
    ("brown", "pentagon*"),
    ("magenta", "otimes"),
]


def latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


# Feature-group order for ablation folder names (drop_*, only_*).
FEATURE_GROUPS_CLUSTERING = [
    "value_based",
    "vicinity_based",
    "domain_based",
    "levenshtein",
    "pattern_based",
]
FEATURE_GROUPS_RULE = [
    "value_based",
    "vicinity_based",
    "domain_based",
    "levenshtein",
]


def feature_variant_order(groups: List[str]) -> List[str]:
    """all_features, then all drop_*, then all only_*."""
    names = ["all_features"]
    names.extend(f"drop_{g}" for g in groups)
    names.extend(f"only_{g}" for g in groups)
    return names


# Panel = (short title, plot tuples, optional \label for \ref{} without subcaption pkg)
PanelSpec = Tuple[str, List[Tuple[str, str, str, str]], Optional[str]]


def build_single_panel_axis_tex(
    rel_data_dir: str,
    plots: List[Tuple[str, str, str, str]],
    panel_title: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xticks: List[int],
    legend_columns: int = 2,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    axis_height_ratio: float = AXIS_HEIGHT_RATIO,
    mark_size_pt: float = MARK_SIZE_PT,
) -> str:
    """
    One pgfplots axis (multi-series), sized to \\linewidth of enclosing minipage.
    Exact x ticks. Legend inside the plot (NW).
    """
    ticks_csv = ",".join(str(t) for t in xticks)
    h = f"{axis_height_ratio:.3f}"
    ms = f"{mark_size_pt:.2f}"
    axis_open = [
        r"\begin{tikzpicture}",
        r"  \begin{axis}[",
        r"    width=\linewidth,",
        rf"    height={h}\linewidth,",
        r"    scale only axis,",
        r"    clip mode=individual,",
        f"    title={{{latex_escape(panel_title)}}},",
        r"    title style={font=\tiny, align=center, text width=\linewidth},",
    ]
    if show_xlabel:
        axis_open.append(r"    xlabel={Labeling budget},")
    if show_ylabel:
        axis_open.append(r"    ylabel={$F_1$},")
    lines = axis_open + [
        r"    label style={font=\tiny},",
        r"    xmode=log,",
        r"    log basis x={10},",
        f"    xmin={xmin}, xmax={xmax},",
        f"    ymin={ymin}, ymax={ymax},",
        f"    xtick={{{ticks_csv}}},",
        f"    xticklabels={{{ticks_csv}}},",
        r"    scaled x ticks=false,",
        r"    xticklabel style={font=\tiny, inner sep=1pt},",
        r"    yticklabel style={font=\tiny},",
        r"    grid=both,",
        r"    legend pos=north west,",
        f"    legend columns={legend_columns},",
        r"    legend style={",
        r"      font=\tiny,",
        r"      fill=white,",
        r"      fill opacity=0.92,",
        r"      draw=gray!40,",
        r"      line width=0.2pt,",
        r"      column sep=0.15em,",
        r"      row sep=0.02em,",
        r"      inner sep=1pt,",
        r"    },",
        r"  ]",
    ]
    for stem, leg, color, mark in plots:
        csv_path = f"{rel_data_dir}/{stem}.csv".replace("//", "/")
        lines.append(
            f"    \\addplot[color={color}, mark={mark}, mark size={ms}pt, thick] "
            f"table[x=labeling_budget, y=f1_score, col sep=comma]{{{csv_path}}};"
        )
        lines.append(f"    \\addlegendentry{{{latex_escape(leg)}}}")
    lines.extend(
        [
            r"  \end{axis}",
            r"\end{tikzpicture}",
        ]
    )
    return "\n".join(lines)


def build_multrow_figure_tex(
    rel_data_dir: str,
    panel_rows: List[List[PanelSpec]],
    figure_caption: str,
    figure_label: str,
    xmin: float = 8,
    xmax: float = 220,
    ymin: float = 0,
    ymax: float = 1.0,
    xticks: Optional[List[int]] = None,
) -> str:
    """
    One figure: each inner list is one row of minipages (\\hfill between columns).
    Y-axis label on the first panel of each row; x-axis label on the bottom row only.
    """
    xticks = list(xticks or BUDGET_TICKS)
    if not panel_rows or not any(panel_rows):
        return ""

    nrows = len(panel_rows)
    lines = [
        r"\begin{figure}[t]",
        r"  \centering",
        r"  \footnotesize",
    ]
    for r, row in enumerate(panel_rows):
        if not row:
            continue
        ncols = len(row)
        frac = ROW_WIDTH_FRAC / max(ncols, 1)
        for c, (title, plot_list, panel_label) in enumerate(row):
            show_ylabel = c == 0
            show_xlabel = r == nrows - 1
            if c > 0:
                lines.append(r"  \hfill")
            lines.append(f"  \\begin{{minipage}}[t]{{{frac:.3f}\\columnwidth}}")
            lines.append(r"    \centering")
            if panel_label:
                lines.append(f"    \\label{{{panel_label}}}")
            n = len(plot_list)
            leg_cols = 1 if n > 6 else (2 if n > 3 else 1)
            axis_tex = build_single_panel_axis_tex(
                rel_data_dir,
                plot_list,
                title,
                xmin,
                xmax,
                ymin,
                ymax,
                xticks,
                legend_columns=leg_cols,
                show_ylabel=show_ylabel,
                show_xlabel=show_xlabel,
            )
            for ln in axis_tex.split("\n"):
                lines.append("    " + ln if ln.strip() else "")
            lines.append(r"  \end{minipage}")
        if r < nrows - 1:
            lines.append(r"  \par\vspace{0.85ex}")
    lines.extend(
        [
            r"  \par\vspace{0.35ex}",
            f"  \\caption{{{figure_caption}}}",
            f"  \\label{{{figure_label}}}",
            r"\end{figure}",
            "",
        ]
    )
    return "\n".join(lines)


def panel_rows_from_flat(panels: List[PanelSpec]) -> List[List[PanelSpec]]:
    """Use a 3+2 grid for five panels; otherwise one row."""
    if len(panels) == 5:
        return [panels[:3], panels[3:]]
    return [panels]


def main():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    rel = "data"  # relative to where .tex lives (same folder as data/)

    sections: List[str] = []
    sections.append(
        "% Auto-generated by export_ablation_plots.py\n"
        "% Preamble: \\usepackage{pgfplots}  and  \\pgfplotsset{compat=1.18}\n"
        "% Figure: feature panels = all_features + drop_* + only_*; includes pattern enforcement.\n"
        "% Layout: one row, or two rows (3+2) when five panels.\n"
        "% Per-panel \\label{} kept for \\ref{fig:ablation-feature-clustering} etc.\n"
    )

    panels: List[PanelSpec] = []

    # --- 1) Feature ablation: clustering (all_features + drop_* + only_*)
    root_cl = _SCRIPT_DIR / "results_feature_ablation_clustering_based"
    raw = collect_series_metrics(root_cl, "Quintet_3", lambda v: v)
    avg = average_metrics(raw)
    order_cl = feature_variant_order(FEATURE_GROUPS_CLUSTERING)
    plots_cl: List[Tuple[str, str, str, str]] = []
    for i, name in enumerate(order_cl):
        if name not in avg:
            continue
        stem = safe_filename(f"cl_feat_{name}")
        write_series_csv(name, avg[name], _DATA_DIR, stem)
        c, m = STYLE_CYCLE[i % len(STYLE_CYCLE)]
        leg = name.replace("_", " ")
        plots_cl.append((stem, leg, c, m))
    if plots_cl:
        panels.append(
            (
                "Feature ablation (clustering)",
                plots_cl,
                "fig:ablation-feature-clustering",
            )
        )

    # --- 2) Feature ablation: rule-based (all_features + drop_* + only_*)
    root_rb = _SCRIPT_DIR / "results_feature_ablation_rule_based"
    raw = collect_series_metrics(root_rb, "Quintet_3", lambda v: v)
    avg = average_metrics(raw)
    order_rb = feature_variant_order(FEATURE_GROUPS_RULE)
    plots_rb: List[Tuple[str, str, str, str]] = []
    for i, name in enumerate(order_rb):
        if name not in avg:
            continue
        stem = safe_filename(f"rb_feat_{name}")
        write_series_csv(name, avg[name], _DATA_DIR, stem)
        c, m = STYLE_CYCLE[i % len(STYLE_CYCLE)]
        plots_rb.append((stem, name.replace("_", " "), c, m))
    if plots_rb:
        panels.append(
            (
                "Feature ablation (rule-based)",
                plots_rb,
                "fig:ablation-feature-rule",
            )
        )

    # --- 3) Negative pruning: RB vs CL x off/on ---
    roots_np = [
        ("rule", _SCRIPT_DIR / "results_negative_pruning_ablation_rule_based"),
        ("clust", _SCRIPT_DIR / "results_negative_pruning_ablation_clustering_based"),
    ]
    np_series: Dict[str, List[Tuple[int, float, float, float]]] = {}
    for tag, rroot in roots_np:
        raw = collect_series_metrics(
            rroot,
            "Quintet_3",
            lambda v, t=tag: f"{t}_{v}",
        )
        avg = average_metrics(raw)
        for composite, rows in avg.items():
            stem = safe_filename(f"np_{composite}")
            leg = (
                f"{'Rule-based' if composite.startswith('rule_') else 'Clustering'}, "
                f"neg. prune {'on' if composite.endswith('_on') else 'off'}"
            )
            write_series_csv(composite, rows, _DATA_DIR, stem)
            np_series[stem] = rows

    plots_np: List[Tuple[str, str, str, str]] = []
    np_order = [
        ("np_rule_negprune_off", "Rule-based, neg. prune off"),
        ("np_rule_negprune_on", "Rule-based, neg. prune on"),
        ("np_clust_negprune_off", "Clustering, neg. prune off"),
        ("np_clust_negprune_on", "Clustering, neg. prune on"),
    ]
    for j, (stem, leg) in enumerate(np_order):
        if stem not in np_series:
            print(f"  [warn] missing negative-pruning series: {stem}")
            continue
        c, m = STYLE_CYCLE[j % len(STYLE_CYCLE)]
        plots_np.append((stem, leg, c, m))
    if plots_np:
        panels.append(
            (
                "Negative pruning",
                plots_np,
                "fig:ablation-negprune",
            )
        )

    # --- 4) Cluster sampling ---
    root_cs = _SCRIPT_DIR / "results_cluster_sampling_ablation_clustering_based"

    def cs_key(v: str) -> str:
        if v.startswith("sampling_"):
            return v[len("sampling_") :]
        return v

    raw = collect_series_metrics(root_cs, "Quintet_3", cs_key)
    avg = average_metrics(raw)
    plots_cs: List[Tuple[str, str, str, str]] = []
    for i, name in enumerate(["column_coverage", "kmeans_pp"]):
        if name not in avg:
            print(f"  [warn] missing cluster-sampling series: {name}")
            continue
        stem = safe_filename(f"cs_{name}")
        write_series_csv(name, avg[name], _DATA_DIR, stem)
        c, m = STYLE_CYCLE[i % len(STYLE_CYCLE)]
        leg = "Column coverage" if name == "column_coverage" else "k-means++"
        plots_cs.append((stem, leg, c, m))
    if plots_cs:
        panels.append(
            (
                "Cluster sampling",
                plots_cs,
                "fig:ablation-cluster-sampling",
            )
        )

    # --- 5) Pattern enforcement (``perfect detector'' / invalid-zone modes)
    root_pe = _SCRIPT_DIR / "results_pattern_enforcement_ablation_rule_based"

    def pe_key(v: str) -> str:
        p = "pattern_enforcement_"
        return v[len(p) :] if v.startswith(p) else v

    raw_pe = collect_series_metrics(root_pe, "Quintet_3", pe_key)
    avg_pe = average_metrics(raw_pe)
    pe_order = ["check", "always_accept", "disabled"]
    pe_legend = {
        "check": "Oracle check",
        "always_accept": "Always accept",
        "disabled": "Disabled",
    }
    plots_pe: List[Tuple[str, str, str, str]] = []
    for i, mode in enumerate(pe_order):
        if mode not in avg_pe:
            print(f"  [warn] missing pattern-enforcement series: {mode}")
            continue
        stem = safe_filename(f"pe_{mode}")
        write_series_csv(mode, avg_pe[mode], _DATA_DIR, stem)
        c, m = STYLE_CYCLE[i % len(STYLE_CYCLE)]
        plots_pe.append((stem, pe_legend.get(mode, mode), c, m))
    if plots_pe:
        panels.append(
            (
                "Pattern enforcement",
                plots_pe,
                "fig:ablation-pattern-enforcement",
            )
        )

    if panels:
        cap_parts = [
            "Quintet-3 mean $F_1$ vs. labeling budget (ticks $10,22,96,173$). "
        ]
        rows_cap = panel_rows_from_flat(panels)
        row_desc = "; ".join(
            ", ".join(t for t, _, _ in row) for row in rows_cap
        )
        cap_parts.append(f"Panels (reading order): {row_desc}.")
        combined_caption = "".join(cap_parts)
        sections.append(
            build_multrow_figure_tex(
                rel,
                panel_rows_from_flat(panels),
                combined_caption,
                "fig:ablation-quintet-row",
                xmin=8,
                xmax=220,
            )
        )

    _TEX_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"[saved] {_TEX_PATH}")
    print(f"[saved] CSVs under {_DATA_DIR}")


if __name__ == "__main__":
    main()
