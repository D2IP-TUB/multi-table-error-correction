"""
Clustering and Sampling Module for Global Error Correction.
Uses MiniBatchKMeans for scalable clustering with random proportional sampling.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus
from sklearn.preprocessing import StandardScaler

from core.cell import Cell


class ErrorClusteringSampler:
    """Clusters dirty cells by unusualness and performs proportional random sampling."""

    def __init__(
        self, n_clusters: int = None, batch_size: int = 1000, random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.scaled_features = None
        self.kmeans = None  # fitted MiniBatchKMeans (needed for centroid sampling)

    def fit_clusters(
        self, features_matrix: np.ndarray
    ) -> np.ndarray:
        """Cluster cells using MiniBatchKMeans."""
        if self.n_clusters is None:
            self.n_clusters = min(50, len(features_matrix) // 10)

        logging.info(
            f"Clustering {features_matrix.shape[0]} cells into {self.n_clusters} clusters"
        )

        self.scaled_features = self.scaler.fit_transform(features_matrix)

        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=min(self.batch_size, len(self.scaled_features)),
            random_state=self.random_state,
            n_init="auto",
        )
        self.cluster_labels = kmeans.fit_predict(self.scaled_features)
        self.kmeans = kmeans

        cluster_counts = np.bincount(self.cluster_labels)
        logging.info(
            f"Created {self.n_clusters} clusters - sizes: min={np.min(cluster_counts)}, "
            f"max={np.max(cluster_counts)}, avg={np.mean(cluster_counts):.1f}"
        )

        return self.cluster_labels

    def sample_from_clusters(
        self,
        dirty_cells: List[Cell],
        cluster_labels: np.ndarray,
        labeling_budget: int,
        lake=None,
        strategy: str = "column_coverage",
    ) -> List[Cell]:
        """
        Sample cells from clusters.

        strategy:
          - "column_coverage" (default): allocate budget across clusters, then within
            each cluster prefer column coverage + random fill (existing behavior).
          - "centroid": same cluster budget split; within each cluster pick the cells
            whose scaled features are closest to the KMeans cluster centroid (medoid-like).
          - "kmeans_pp": same cluster budget split; within each cluster pick k cells using
            k-means++ seeding (D² to nearest already-chosen point) in scaled feature space.
        """
        strategy = (strategy or "column_coverage").strip().lower()
        if strategy == "centroid":
            return self._sample_from_clusters_centroid(
                dirty_cells, cluster_labels, labeling_budget
            )
        if strategy == "kmeans_pp":
            return self._sample_from_clusters_kmeans_pp(
                dirty_cells, cluster_labels, labeling_budget
            )

        clusters = self._group_cells_by_cluster(dirty_cells, cluster_labels)
        cluster_sizes = {cid: len(cells) for cid, cells in clusters.items()}
        num_clusters = len(cluster_sizes)

        if num_clusters == 0:
            return []

        # --- Allocate budget to clusters proportional to size ---
        cluster_budgets = self._allocate_budget(cluster_sizes, labeling_budget)

        # --- Sample within each cluster: column coverage first, then random ---
        rng = np.random.RandomState(self.random_state)
        sampled_cells: List[Cell] = []

        for cid, cluster_budget in cluster_budgets.items():
            if cluster_budget == 0:
                continue

            cells_in_cluster = [cell for cell, _ in clusters[cid]]
            cluster_budget = min(cluster_budget, len(cells_in_cluster))

            cells_by_col: Dict[tuple, List[Cell]] = {}
            for cell in cells_in_cluster:
                col_key = (cell.table_id, cell.column_idx)
                if col_key not in cells_by_col:
                    cells_by_col[col_key] = []
                cells_by_col[col_key].append(cell)

            cluster_sample: List[Cell] = []
            sampled_set: set = set()

            # Phase 1: 1 sample per column, largest columns first
            cols_sorted = sorted(cells_by_col.keys(), key=lambda k: len(cells_by_col[k]), reverse=True)
            for col_key in cols_sorted:
                if len(cluster_sample) >= cluster_budget:
                    break
                col_cells = cells_by_col[col_key]
                cell = col_cells[rng.randint(len(col_cells))]
                cluster_sample.append(cell)
                sampled_set.add(id(cell))

            # Phase 2: randomly fill remaining budget
            if len(cluster_sample) < cluster_budget:
                available = [c for c in cells_in_cluster if id(c) not in sampled_set]
                remaining = cluster_budget - len(cluster_sample)
                if available:
                    remaining = min(remaining, len(available))
                    indices = rng.choice(len(available), size=remaining, replace=False)
                    cluster_sample.extend(available[i] for i in indices)

            sampled_cells.extend(cluster_sample)

        return self._finalize_sample_list(sampled_cells, dirty_cells, labeling_budget)

    def _finalize_sample_list(
        self,
        sampled_cells: List[Cell],
        dirty_cells: List[Cell],
        labeling_budget: int,
    ) -> List[Cell]:
        """
        Deduplicate by (table, col, row); top up to min(budget, #unique dirty) so
        len(result) matches unique cells assigned in create_zones_from_clusters.
        """
        unique_dirty: List[Cell] = []
        seen_d = set()
        for c in dirty_cells:
            k = (c.table_id, c.column_idx, c.row_idx)
            if k not in seen_d:
                seen_d.add(k)
                unique_dirty.append(c)

        cap = min(labeling_budget, len(unique_dirty))
        out: List[Cell] = []
        seen = set()
        for c in sampled_cells:
            k = (c.table_id, c.column_idx, c.row_idx)
            if k not in seen:
                seen.add(k)
                out.append(c)

        if len(out) >= cap:
            return out[:cap]

        rng = np.random.RandomState((self.random_state + 7919) % (2**31))
        perm = rng.permutation(len(unique_dirty))
        for i in perm:
            if len(out) >= cap:
                break
            c = unique_dirty[i]
            k = (c.table_id, c.column_idx, c.row_idx)
            if k not in seen:
                seen.add(k)
                out.append(c)
        return out

    def _sample_from_clusters_centroid(
        self,
        dirty_cells: List[Cell],
        cluster_labels: np.ndarray,
        labeling_budget: int,
    ) -> List[Cell]:
        """
        Pick cells nearest to each cluster centroid in scaled feature space.

        Uses the same inter-cluster budget allocation as column_coverage mode.
        """
        if self.kmeans is None or self.scaled_features is None:
            raise RuntimeError(
                "Centroid sampling requires fit_clusters() to be called first."
            )

        clusters = self._group_cells_by_cluster(dirty_cells, cluster_labels)
        cluster_sizes = {cid: len(cells) for cid, cells in clusters.items()}
        if not cluster_sizes:
            return []

        cluster_budgets = self._allocate_budget(cluster_sizes, labeling_budget)
        centers = self.kmeans.cluster_centers_

        sampled_cells: List[Cell] = []

        for cid, cluster_budget in cluster_budgets.items():
            if cluster_budget == 0:
                continue

            cluster_entries = clusters[cid]
            cells_in_cluster = [cell for cell, _ in cluster_entries]
            row_indices = [idx for _, idx in cluster_entries]
            cluster_budget = min(cluster_budget, len(cells_in_cluster))

            Xc = self.scaled_features[row_indices]
            center = centers[cid]
            dists = np.linalg.norm(Xc - center, axis=1)

            k = cluster_budget
            pick_idx = np.argpartition(dists, k - 1)[:k]
            pick_idx = pick_idx[np.argsort(dists[pick_idx])]

            for j in pick_idx:
                sampled_cells.append(cells_in_cluster[int(j)])

        return self._finalize_sample_list(sampled_cells, dirty_cells, labeling_budget)

    def _sample_from_clusters_kmeans_pp(
        self,
        dirty_cells: List[Cell],
        cluster_labels: np.ndarray,
        labeling_budget: int,
    ) -> List[Cell]:
        """
        Within each cluster, select k cells using k-means++ (scaled features).

        Same inter-cluster budget as column_coverage / centroid. Per cluster, k is
        cluster_budget; picks use sklearn.cluster.kmeans_plusplus (same rule as
        k-means++ initialization).
        """
        if self.scaled_features is None:
            raise RuntimeError(
                "k-means++ sampling requires fit_clusters() to be called first."
            )

        clusters = self._group_cells_by_cluster(dirty_cells, cluster_labels)
        cluster_sizes = {cid: len(cells) for cid, cells in clusters.items()}
        if not cluster_sizes:
            return []

        cluster_budgets = self._allocate_budget(cluster_sizes, labeling_budget)
        sampled_cells: List[Cell] = []

        for cid, cluster_budget in cluster_budgets.items():
            if cluster_budget == 0:
                continue

            cluster_entries = clusters[cid]
            cells_in_cluster = [cell for cell, _ in cluster_entries]
            row_indices = [idx for _, idx in cluster_entries]
            cluster_budget = min(cluster_budget, len(cells_in_cluster))

            Xc = self.scaled_features[row_indices]
            # Per-cluster RNG derived from config seed (reproducible)
            rng = np.random.RandomState(
                (self.random_state * 1000003 + int(cid)) % (2**31)
            )
            n_local = Xc.shape[0]
            if cluster_budget >= n_local:
                pick_local = np.arange(n_local, dtype=int)
            else:
                _, pick_local = kmeans_plusplus(
                    Xc,
                    n_clusters=cluster_budget,
                    random_state=rng,
                )

            for j in pick_local:
                sampled_cells.append(cells_in_cluster[int(j)])

        return self._finalize_sample_list(sampled_cells, dirty_cells, labeling_budget)

    def _allocate_budget(
        self, cluster_sizes: Dict[int, int], labeling_budget: int
    ) -> Dict[int, int]:
        """
        Allocate labeling budget to clusters proportional to their size.

        Uses largest-remainder so sum(budgets) == min(labeling_budget, total_cells)
        when labeling_budget >= num_clusters (avoids int() truncation losing slots).
        """
        num_clusters = len(cluster_sizes)
        if num_clusters == 0:
            return {}
        total_cells = sum(cluster_sizes.values())
        labeling_budget = min(labeling_budget, total_cells)

        if labeling_budget < num_clusters:
            sorted_by_size = sorted(
                cluster_sizes.items(), key=lambda x: x[1], reverse=True
            )
            budgets = {cid: 0 for cid in cluster_sizes}
            for cid, _ in sorted_by_size[:labeling_budget]:
                budgets[cid] = 1
            return budgets

        # At least one sample per non-empty cluster, then proportional + largest remainder
        budgets = {cid: 1 for cid in cluster_sizes}
        remaining = labeling_budget - num_clusters
        if remaining <= 0:
            return budgets

        fracs: List[Tuple[float, int, int]] = []
        allocated_extra = 0
        for cid, size in cluster_sizes.items():
            exact = remaining * size / total_cells
            flo = int(exact)
            budgets[cid] += flo
            allocated_extra += flo
            fracs.append((exact - flo, cid, size))

        leftover = remaining - allocated_extra
        fracs.sort(key=lambda x: (-x[0], -x[2], x[1]))
        for i in range(leftover):
            budgets[fracs[i][1]] += 1

        return budgets

    def _group_cells_by_cluster(
        self, dirty_cells: List[Cell], cluster_labels: np.ndarray
    ) -> Dict[int, List[Tuple[Cell, int]]]:
        """Group cells by their cluster labels."""
        clusters = {}
        for i, (cell, cluster_id) in enumerate(zip(dirty_cells, cluster_labels)):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append((cell, i))
        return clusters

    def print_sampling_summary(
        self,
        sampled_cells: List[Cell],
        all_dirty_cells: List[Cell],
        cluster_labels: np.ndarray,
    ):
        """Print detailed summary of sampling results."""

        logging.info("=== SAMPLING SUMMARY ===")

        logging.info(f"Total cells sampled: {len(sampled_cells)}")
        logging.info(f"Total dirty cells: {len(all_dirty_cells)}")
        logging.info(
            f"Sampling ratio: {len(sampled_cells) / len(all_dirty_cells) * 100:.2f}%"
        )

        sampled_columns = set()
        sampled_tables = set()
        total_columns = set()
        total_tables = set()

        for cell in sampled_cells:
            sampled_columns.add((cell.table_id, cell.column_idx))
            sampled_tables.add(cell.table_id)

        for cell in all_dirty_cells:
            total_columns.add((cell.table_id, cell.column_idx))
            total_tables.add(cell.table_id)

        logging.info(
            f"Column coverage: {len(sampled_columns)}/{len(total_columns)} ({len(sampled_columns) / len(total_columns) * 100:.1f}%)"
        )
        logging.info(
            f"Table coverage: {len(sampled_tables)}/{len(total_tables)} ({len(sampled_tables) / len(total_tables) * 100:.1f}%)"
        )

        clusters = self._group_cells_by_cluster(all_dirty_cells, cluster_labels)
        sampled_clusters = set()

        for cell in sampled_cells:
            for i, original_cell in enumerate(all_dirty_cells):
                if original_cell is cell:
                    sampled_clusters.add(cluster_labels[i])
                    break

        logging.info(
            f"Cluster coverage: {len(sampled_clusters)}/{len(clusters)} ({len(sampled_clusters) / len(clusters) * 100:.1f}%)"
        )

        if hasattr(sampled_cells[0], "influence"):
            influences = [getattr(cell, "influence", 0) for cell in sampled_cells]
            total_influence = sum(influences)
            avg_influence = total_influence / len(influences) if influences else 0
            max_influence = max(influences) if influences else 0
            min_influence = min(influences) if influences else 0

            logging.info("Influence statistics:")
            logging.info(f"  - Total: {total_influence}")
            logging.info(f"  - Average: {avg_influence:.2f}")
            logging.info(f"  - Range: {min_influence} - {max_influence}")

        logging.info("\nPer-cluster sampling:")
        sorted_clusters = sorted(
            clusters.items(), key=lambda x: len(x[1]), reverse=True
        )

        for cluster_id, cluster_cells in sorted_clusters[:10]:
            cluster_size = len(cluster_cells)

            sampled_from_cluster = 0
            for cell in sampled_cells:
                for original_cell, _ in cluster_cells:
                    if original_cell is cell:
                        sampled_from_cluster += 1
                        break

            if sampled_from_cluster > 0:
                logging.info(
                    f"  Cluster {cluster_id}: {sampled_from_cluster}/{cluster_size} cells ({sampled_from_cluster / cluster_size * 100:.1f}%)"
                )

    def get_cluster_statistics(
        self,
        dirty_cells: List[Cell],
        cluster_labels: np.ndarray,
        include_feature_stats: bool = False,
        features_matrix: np.ndarray = None,
        feature_names: List[str] = None,
    ) -> Dict:
        """Get detailed statistics for each cluster."""
        clusters = self._group_cells_by_cluster(dirty_cells, cluster_labels)

        cluster_stats = {}

        for cluster_id, cluster_cells in clusters.items():
            cells_in_cluster = [cell for cell, _ in cluster_cells]
            cluster_size = len(cells_in_cluster)

            tables_in_cluster = set()
            columns_in_cluster = set()
            table_column_pairs = set()

            for cell in cells_in_cluster:
                tables_in_cluster.add(cell.table_id)
                columns_in_cluster.add(cell.column_idx)
                table_column_pairs.add((cell.table_id, cell.column_idx))

            error_types = {}
            if hasattr(cells_in_cluster[0], "error_type"):
                for cell in cells_in_cluster:
                    error_type = getattr(cell, "error_type", "Unknown")
                    error_types[error_type] = error_types.get(error_type, 0) + 1

            influences = []
            for cell in cells_in_cluster:
                if hasattr(cell, "influence"):
                    influences.append(cell.influence)

            influence_stats = {}
            if influences:
                influence_stats = {
                    "min": min(influences),
                    "max": max(influences),
                    "mean": np.mean(influences),
                    "std": np.std(influences),
                    "total": sum(influences),
                }

            value_lengths = [
                len(str(cell.value)) for cell in cells_in_cluster if cell.value
            ]
            length_stats = {}
            if value_lengths:
                length_stats = {
                    "min": min(value_lengths),
                    "max": max(value_lengths),
                    "mean": np.mean(value_lengths),
                    "std": np.std(value_lengths),
                }

            feature_stats = {}
            if include_feature_stats and features_matrix is not None:
                cluster_indices = [idx for _, idx in cluster_cells]
                cluster_features = features_matrix[cluster_indices]

                feature_stats = {
                    "feature_means": np.mean(cluster_features, axis=0).tolist(),
                    "feature_stds": np.std(cluster_features, axis=0).tolist(),
                    "feature_mins": np.min(cluster_features, axis=0).tolist(),
                    "feature_maxs": np.max(cluster_features, axis=0).tolist(),
                }

                if feature_names:
                    named_feature_stats = {}
                    for i, name in enumerate(feature_names):
                        named_feature_stats[name] = {
                            "mean": feature_stats["feature_means"][i],
                            "std": feature_stats["feature_stds"][i],
                            "min": feature_stats["feature_mins"][i],
                            "max": feature_stats["feature_maxs"][i],
                        }
                    feature_stats["named_features"] = named_feature_stats

            cluster_stats[cluster_id] = {
                "cluster_size": cluster_size,
                "num_tables": len(tables_in_cluster),
                "num_columns": len(columns_in_cluster),
                "num_table_column_pairs": len(table_column_pairs),
                "tables": list(tables_in_cluster),
                "columns": list(columns_in_cluster),
                "table_column_pairs": list(table_column_pairs),
                "error_type_distribution": error_types,
                "influence_stats": influence_stats,
                "value_length_stats": length_stats,
                "feature_stats": feature_stats,
                "diversity_metrics": {
                    "table_diversity": len(tables_in_cluster) / cluster_size,
                    "column_diversity": len(columns_in_cluster) / cluster_size,
                    "table_column_diversity": len(table_column_pairs) / cluster_size,
                },
            }

        total_cells = len(dirty_cells)
        total_tables = len(set(cell.table_id for cell in dirty_cells))
        total_columns = len(set(cell.column_idx for cell in dirty_cells))
        total_table_column_pairs = len(
            set((cell.table_id, cell.column_idx) for cell in dirty_cells)
        )

        overall_stats = {
            "total_clusters": len(cluster_stats),
            "total_cells": total_cells,
            "total_tables": total_tables,
            "total_columns": total_columns,
            "total_table_column_pairs": total_table_column_pairs,
            "avg_cluster_size": total_cells / len(cluster_stats)
            if cluster_stats
            else 0,
            "cluster_size_distribution": {
                "min": min(stats["cluster_size"] for stats in cluster_stats.values())
                if cluster_stats
                else 0,
                "max": max(stats["cluster_size"] for stats in cluster_stats.values())
                if cluster_stats
                else 0,
                "std": np.std(
                    [stats["cluster_size"] for stats in cluster_stats.values()]
                )
                if cluster_stats
                else 0,
            },
        }

        return {
            "cluster_statistics": cluster_stats,
            "overall_statistics": overall_stats,
        }

    def print_detailed_cluster_summary(
        self,
        dirty_cells: List[Cell],
        cluster_labels: np.ndarray,
        top_n_clusters: int = 10,
        include_feature_stats: bool = False,
        features_matrix: np.ndarray = None,
        feature_names: List[str] = None,
    ):
        """Print detailed summary of cluster characteristics."""
        stats = self.get_cluster_statistics(
            dirty_cells,
            cluster_labels,
            include_feature_stats,
            features_matrix,
            feature_names,
        )

        cluster_stats = stats["cluster_statistics"]
        overall_stats = stats["overall_statistics"]

        logging.info("=== DETAILED CLUSTER ANALYSIS ===")

        logging.info(f"Total clusters: {overall_stats['total_clusters']}")
        logging.info(f"Total cells: {overall_stats['total_cells']}")
        logging.info(f"Total tables: {overall_stats['total_tables']}")
        logging.info(f"Total columns: {overall_stats['total_columns']}")
        logging.info(
            f"Total table-column pairs: {overall_stats['total_table_column_pairs']}"
        )
        logging.info(f"Average cluster size: {overall_stats['avg_cluster_size']:.1f}")

        size_dist = overall_stats["cluster_size_distribution"]
        logging.info(
            f"Cluster size range: {size_dist['min']} - {size_dist['max']} (std: {size_dist['std']:.1f})"
        )

        sorted_clusters = sorted(
            cluster_stats.items(), key=lambda x: x[1]["cluster_size"], reverse=True
        )

        logging.info(f"\n=== TOP {top_n_clusters} CLUSTERS BY SIZE ===")

        for i, (cluster_id, cstats) in enumerate(sorted_clusters[:top_n_clusters]):
            logging.info(f"\n--- Cluster {cluster_id} (Rank #{i + 1}) ---")
            logging.info(f"Size: {cstats['cluster_size']} cells")
            logging.info(
                f"Tables: {cstats['num_tables']} ({cstats['tables'][:3]}{'...' if len(cstats['tables']) > 3 else ''})"
            )
            logging.info(
                f"Columns: {cstats['num_columns']} ({cstats['columns'][:5]}{'...' if len(cstats['columns']) > 5 else ''})"
            )
            logging.info(f"Table-Column pairs: {cstats['num_table_column_pairs']}")

            div = cstats["diversity_metrics"]
            logging.info(
                f"Diversity - Tables/cell: {div['table_diversity']:.3f}, Columns/cell: {div['column_diversity']:.3f}"
            )

            if cstats["error_type_distribution"]:
                error_types_str = ", ".join(
                    [f"{k}: {v}" for k, v in cstats["error_type_distribution"].items()]
                )
                logging.info(f"Error types: {error_types_str}")

            if cstats["influence_stats"]:
                inf = cstats["influence_stats"]
                logging.info(
                    f"Influence - Total: {inf['total']}, Avg: {inf['mean']:.2f}, Range: {inf['min']}-{inf['max']}"
                )

            if cstats["value_length_stats"]:
                length = cstats["value_length_stats"]
                logging.info(
                    f"Value lengths - Avg: {length['mean']:.1f}, Range: {length['min']}-{length['max']}"
                )

            if include_feature_stats and cstats["feature_stats"] and feature_names:
                logging.info("Top feature characteristics:")
                named_features = cstats["feature_stats"].get("named_features", {})
                top_features = sorted(
                    named_features.items(), key=lambda x: x[1]["mean"], reverse=True
                )[:3]
                for feat_name, feat_stats in top_features:
                    logging.info(
                        f"  {feat_name}: {feat_stats['mean']:.3f} ± {feat_stats['std']:.3f}"
                    )
