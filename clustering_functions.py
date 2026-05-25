from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import HDBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from data_science_functions import filter_and_clean_races, TOP_TIER_CLASSIFICATIONS, MAX_PROFILE_SCORE

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    PLOTLY_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    umap = None
    UMAP_AVAILABLE = False


DEFAULT_DATA_CANDIDATES = [
    # "data_test/normalized_races_df.parquet",
    # "data/normalized_races_df.parquet",
    "data_v2/races_df.parquet",
]

DATA_PATH: str | None = None
CATEGORY_CLUSTER_FEATURE_WEIGHT = 0.3
HDBSCAN_MIN_CLUSTER_SIZE = 25
HDBSCAN_MIN_SAMPLES = 5
HIERARCHICAL_LINKAGE_METHOD = "average"
HIERARCHICAL_ANALYSIS_CLUSTER_COUNTS = [3, 4, 5, 6, 8, 10, 12]
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.05
PLOT_INTERACTIVE = True
PLOT_SHOW = True
PLOT_SAVE_PATH: str | None = None

CLUSTER_FEATURES = [
    "profile_score",
    "final_km_percentage",
    "profile_score_last_25k",
    "distance_km",
    "elevation_m",
    # "won_how_clean", both features get one-hot-encoded
    # "classification", and expanded to 16 separate features, 
    # messing up the clustering distance
    "avg_speed_kmh"
]
CATEGORY_CLUSTER_FEATURES = [
    "won_how_clean",
    # "classification",
]

class ClusteringConfigError(ValueError):
    pass


WON_HOW_COLOR_MAP = {
    "large_sprint": "#1f77b4",
    "small_sprint": "#ff7f0e",
    "duo_sprint": "#2ca02c",
    "solo": "#d62728",
    "time_trial": "#9467bd",
    "unknown": "#7f7f7f",
}

CLUSTER_PLOT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

CLUSTER_NAME_MAP = {
    # 0: "sprints men 2",
    # 1: "puncheur stages",
    # 2: "mountain stage",
    # 3: "sprints women",
    # 4: "time trials",
    # 5: "sprints men 1",
    # 6: "spring-classics women",
    # 7: "spring-classics men",
}


def cluster_label_to_name(cluster_label: int) -> str:
    if cluster_label == -1:
        return "noise"
    return CLUSTER_NAME_MAP.get(cluster_label, f"cluster_{cluster_label}")


def normalize_won_how_clean_values(values: np.ndarray) -> np.ndarray:
    return np.array(
        [
            (str(value) if value is not None else "unknown")
            if str(value) in WON_HOW_COLOR_MAP
            else "unknown"
            for value in values
        ]
    )


def resolve_data_path() -> Path:
    if DATA_PATH is not None:
        path = Path(DATA_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Data path does not exist: {path}")
        return path

    for candidate in DEFAULT_DATA_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not auto-find race data parquet. Set DATA_PATH explicitly."
    )




def resolve_features(df: pl.DataFrame, requested_features: Iterable[str]) -> list[str]:
    requested = [feature.strip() for feature in requested_features if feature.strip()]
    resolved = [feature for feature in requested]

    missing = [feature for feature in resolved if feature not in df.columns]
    if missing:
        raise ClusteringConfigError(
            "Missing features in race data: "
            f"{missing}. Available columns include: {df.columns}"
        )

    return resolved


def load_race_data() -> pl.DataFrame:
    path = resolve_data_path()
    return pl.read_parquet(path)


def build_mixed_feature_space(
    df: pl.DataFrame,
    features: list[str],
) -> dict[str, object]:
    feature_df = df.select(features)

    numeric_dtypes = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Decimal,
    }
    numeric_cols = [
        column_name
        for column_name, dtype in feature_df.schema.items()
        if dtype in numeric_dtypes and column_name not in CATEGORY_CLUSTER_FEATURES
    ]
    categorical_cols = [
        column_name
        for column_name in feature_df.columns
        if column_name not in numeric_cols
    ]

    numeric_matrix = np.empty((feature_df.height, 0), dtype=float)
    numeric_normalized = np.empty((feature_df.height, 0), dtype=float)
    if numeric_cols:
        numeric_df = feature_df.select(
            [pl.col(column_name).cast(pl.Float64, strict=False).alias(column_name) for column_name in numeric_cols]
        )
        numeric_matrix = numeric_df.to_numpy().astype(float)
        numeric_min = np.nanmin(numeric_matrix, axis=0)
        numeric_max = np.nanmax(numeric_matrix, axis=0)
        numeric_ranges = numeric_max - numeric_min
        numeric_ranges[numeric_ranges == 0] = 1.0
        numeric_normalized = (numeric_matrix - numeric_min) / numeric_ranges

    categorical_matrix = np.empty((feature_df.height, 0), dtype=object)
    if categorical_cols:
        categorical_expressions = []
        for column_name in categorical_cols:
            column_expr = pl.col(column_name).cast(pl.Utf8, strict=False)
            if column_name == "won_how_clean":
                column_expr = (
                    pl.when(column_expr.is_null() | (column_expr == "unknown"))
                    .then(None)
                    .otherwise(column_expr)
                )
            else:
                column_expr = (
                    pl.when(column_expr.is_null())
                    .then(None)
                    .otherwise(column_expr)
                )
            categorical_expressions.append(column_expr.alias(column_name))
        categorical_df = feature_df.select(categorical_expressions)
        categorical_matrix = categorical_df.to_numpy()

    return {
        "numeric_matrix": numeric_matrix,
        "numeric_normalized": numeric_normalized,
        "numeric_features": numeric_cols,
        "categorical_matrix": categorical_matrix,
        "categorical_features": categorical_cols,
        "selected_features": features,
    }


def _is_missing_object_array(values: np.ndarray) -> np.ndarray:
    return np.equal(values, None) | np.equal(values, "")


def gower_distance_matrix(
    feature_space: dict[str, object],
    categorical_weight: float = CATEGORY_CLUSTER_FEATURE_WEIGHT,
) -> np.ndarray:
    numeric_normalized = feature_space["numeric_normalized"]
    categorical_matrix = feature_space["categorical_matrix"]
    n_points = 0
    if numeric_normalized.size:
        n_points = numeric_normalized.shape[0]
    elif categorical_matrix.size:
        n_points = categorical_matrix.shape[0]
    else:
        return np.empty((0, 0), dtype=float)

    dist_matrix = np.zeros((n_points, n_points), dtype=float)
    for i in range(n_points):
        total_distance = np.zeros(n_points, dtype=float)
        total_weight = np.zeros(n_points, dtype=float)

        if numeric_normalized.size:
            row_numeric = numeric_normalized[i]
            valid_numeric = ~(np.isnan(row_numeric) | np.isnan(numeric_normalized))
            total_distance += np.where(valid_numeric, np.abs(row_numeric - numeric_normalized), 0.0).sum(axis=1)
            total_weight += valid_numeric.sum(axis=1).astype(float)

        if categorical_matrix.size:
            for column_index in range(categorical_matrix.shape[1]):
                column_values = categorical_matrix[:, column_index]
                row_value = column_values[i]
                row_missing = (row_value is None) or (row_value == "")
                if row_missing:
                    continue
                valid_categorical = ~_is_missing_object_array(column_values)
                mismatch = column_values != row_value
                total_distance += np.where(valid_categorical, mismatch.astype(float) * categorical_weight, 0.0)
                total_weight += valid_categorical.astype(float) * categorical_weight

        with np.errstate(invalid="ignore", divide="ignore"):
            dist_matrix[i] = total_distance / total_weight

    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.where(np.isnan(dist_matrix), np.inf, dist_matrix)
    return dist_matrix


def cluster_distance_matrix(dist_matrix: np.ndarray) -> np.ndarray:
    model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="precomputed",
        allow_single_cluster=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return model.fit_predict(dist_matrix)


def compute_umap_embedding(
    dist_matrix: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    if not UMAP_AVAILABLE:
        raise ClusteringConfigError(
            "UMAP inspection requested but `umap-learn` is not installed."
        )
    reducer = umap.UMAP(
        n_components=n_components,
        metric="precomputed",
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return reducer.fit_transform(dist_matrix)


def summarize_clusters(labels: np.ndarray) -> dict[str, int | dict[int, int]]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    counts_map = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    n_clusters = len([label for label in unique_labels if label != -1])
    n_noise = counts_map.get(-1, 0)
    return {
        "n_clusters": int(n_clusters),
        "n_noise": int(n_noise),
        "cluster_sizes": counts_map,
    }


def summarize_partition(labels: np.ndarray) -> dict[str, int | dict[int, int] | float]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    counts_map = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    largest_cluster_share = float(np.max(counts) / labels.shape[0]) if labels.shape[0] else 0.0
    return {
        "n_clusters": int(unique_labels.shape[0]),
        "cluster_sizes": counts_map,
        "largest_cluster_share": largest_cluster_share,
    }


def compute_hierarchical_linkage_matrix(dist_matrix: np.ndarray) -> np.ndarray:
    finite_matrix = dist_matrix.copy()
    finite_mask = np.isfinite(finite_matrix)
    if np.any(~finite_mask):
        finite_values = finite_matrix[finite_mask]
        positive_values = finite_values[finite_values > 0]
        fallback_distance = float(np.max(positive_values)) if positive_values.size else 1.0
        finite_matrix = np.where(finite_mask, finite_matrix, fallback_distance)
    np.fill_diagonal(finite_matrix, 0.0)
    condensed = squareform(finite_matrix, checks=False)
    return linkage(condensed, method=HIERARCHICAL_LINKAGE_METHOD)


def hierarchical_labels_from_linkage(linkage_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    return labels.astype(int) - 1


def evaluate_hierarchical_quality(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    embedding: np.ndarray,
) -> dict[str, float | int | str | None]:
    n_points = int(labels.shape[0])
    unique_labels = np.unique(labels)
    n_clusters = int(unique_labels.shape[0])

    largest_cluster_share = 0.0
    if n_points > 0 and n_clusters > 0:
        _, counts = np.unique(labels, return_counts=True)
        largest_cluster_share = float(np.max(counts) / n_points)

    silhouette = None
    davies_bouldin = None
    calinski_harabasz = None
    if n_clusters >= 2 and n_points > n_clusters:
        dist_for_metric = dist_matrix.copy()
        finite_mask = np.isfinite(dist_for_metric)
        if np.any(~finite_mask):
            finite_values = dist_for_metric[finite_mask]
            positive_finite_values = finite_values[finite_values > 0]
            fallback_distance = float(np.max(positive_finite_values)) if positive_finite_values.size else 1.0
            dist_for_metric = np.where(finite_mask, dist_for_metric, fallback_distance)
        np.fill_diagonal(dist_for_metric, 0.0)
        silhouette = float(silhouette_score(dist_for_metric, labels, metric="precomputed"))
        davies_bouldin = float(davies_bouldin_score(embedding, labels))
        calinski_harabasz = float(calinski_harabasz_score(embedding, labels))

    return {
        "n_points": n_points,
        "n_assigned_clusters": n_clusters,
        "largest_cluster_share": float(largest_cluster_share),
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "silhouette_interpretation": _interpret_silhouette(silhouette),
        "davies_bouldin_interpretation": _interpret_davies_bouldin(davies_bouldin),
        "calinski_harabasz_interpretation": _interpret_calinski_harabasz(calinski_harabasz),
        "largest_cluster_share_interpretation": _interpret_largest_cluster_share(float(largest_cluster_share)),
    }


def analyze_hierarchical_clustering(
    dist_matrix: np.ndarray,
    embedding: np.ndarray,
    cluster_counts: list[int] | None = None,
) -> list[dict[str, object]]:
    linkage_matrix = compute_hierarchical_linkage_matrix(dist_matrix)
    analyses: list[dict[str, object]] = []
    for n_clusters in cluster_counts or HIERARCHICAL_ANALYSIS_CLUSTER_COUNTS:
        labels = hierarchical_labels_from_linkage(linkage_matrix, n_clusters=n_clusters)
        analyses.append(
            {
                "n_clusters_requested": int(n_clusters),
                "labels": labels,
                "summary": summarize_partition(labels),
                "quality": evaluate_hierarchical_quality(dist_matrix=dist_matrix, labels=labels, embedding=embedding),
            }
        )
    return analyses


def select_best_hierarchical_analysis(analyses: list[dict[str, object]]) -> dict[str, object]:
    scored = []
    for analysis in analyses:
        silhouette = analysis["quality"]["silhouette"]
        score = float(silhouette) if silhouette is not None else float("-inf")
        scored.append((score, analysis))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def prepare_clustering_inputs() -> tuple[pl.DataFrame, pl.DataFrame, list[str], dict[str, object], np.ndarray]:
    raw_df = load_race_data()
    try:
        df = filter_and_clean_races(raw_df)
    except ValueError as e:
        raise ClusteringConfigError(str(e))
    if df.height == 0:
        raise ClusteringConfigError(
            "No races left after applying the XGBoost classification filter."
        )

    combined_requested_features = list(dict.fromkeys(CLUSTER_FEATURES + CATEGORY_CLUSTER_FEATURES))
    selected_features = resolve_features(df, combined_requested_features)
    if len(selected_features) < 2:
        raise ClusteringConfigError(
            "Pick at least 2 features for clustering. "
            f"You passed {len(selected_features)}: {selected_features}"
        )

    feature_space = build_mixed_feature_space(df, selected_features)
    dist_matrix = gower_distance_matrix(
        feature_space,
        categorical_weight=CATEGORY_CLUSTER_FEATURE_WEIGHT,
    )
    return raw_df, df, selected_features, feature_space, dist_matrix


def _interpret_silhouette(score: float | None) -> str:
    if score is None or np.isnan(score):
        return "n/a (need >=2 non-noise clusters)"
    if score >= 0.50:
        return "very good (compact and well-separated)"
    if score >= 0.30:
        return "good"
    if score >= 0.10:
        return "weak"
    return "poor"


def _interpret_davies_bouldin(score: float | None) -> str:
    if score is None or np.isnan(score):
        return "n/a (need >=2 non-noise clusters)"
    if score < 0.80:
        return "very good"
    if score < 1.50:
        return "good"
    if score < 2.50:
        return "weak"
    return "poor"


def _interpret_calinski_harabasz(score: float | None) -> str:
    if score is None or np.isnan(score):
        return "n/a (need >=2 non-noise clusters)"
    if score >= 500:
        return "strong"
    if score >= 200:
        return "decent"
    return "weak"


def _interpret_noise_ratio(noise_ratio: float) -> str:
    if noise_ratio < 0.20:
        return "low noise (good)"
    if noise_ratio < 0.50:
        return "moderate noise"
    return "high noise (often too strict clustering)"


def _interpret_largest_cluster_share(share: float) -> str:
    if share < 0.60:
        return "balanced"
    if share < 0.80:
        return "somewhat dominated"
    return "highly dominated by one cluster"


def evaluate_clustering_quality(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    embedding: np.ndarray | None = None,
) -> dict[str, float | int | str | None]:
    n_points = int(labels.shape[0])
    noise_mask = labels == -1
    assigned_mask = ~noise_mask

    assigned_points = int(np.sum(assigned_mask))
    noise_points = int(np.sum(noise_mask))
    noise_ratio = (noise_points / n_points) if n_points else 0.0

    assigned_labels = labels[assigned_mask]
    unique_assigned_labels = np.unique(assigned_labels)
    n_assigned_clusters = int(unique_assigned_labels.shape[0])

    largest_cluster_share = 0.0
    if assigned_points > 0 and n_assigned_clusters > 0:
        _, assigned_counts = np.unique(assigned_labels, return_counts=True)
        largest_cluster_share = float(np.max(assigned_counts) / assigned_points)

    silhouette = None
    davies_bouldin = None
    calinski_harabasz = None
    if n_assigned_clusters >= 2 and assigned_points > n_assigned_clusters:
        dist_assigned = dist_matrix[np.ix_(assigned_mask, assigned_mask)]
        finite_mask = np.isfinite(dist_assigned)
        if np.any(~finite_mask):
            finite_values = dist_assigned[finite_mask]
            positive_finite_values = finite_values[finite_values > 0]
            fallback_distance = float(np.max(positive_finite_values)) if positive_finite_values.size else 1.0
            dist_assigned = np.where(finite_mask, dist_assigned, fallback_distance)
        np.fill_diagonal(dist_assigned, 0.0)
        silhouette = float(silhouette_score(dist_assigned, assigned_labels, metric="precomputed"))

        if embedding is not None:
            embedding_assigned = embedding[assigned_mask]
            unique_complete_labels = np.unique(assigned_labels)
            if unique_complete_labels.shape[0] >= 2 and assigned_labels.shape[0] > unique_complete_labels.shape[0]:
                davies_bouldin = float(davies_bouldin_score(embedding_assigned, assigned_labels))
                calinski_harabasz = float(calinski_harabasz_score(embedding_assigned, assigned_labels))

    return {
        "n_points": n_points,
        "assigned_points": assigned_points,
        "noise_points": noise_points,
        "noise_ratio": float(noise_ratio),
        "n_assigned_clusters": n_assigned_clusters,
        "largest_cluster_share": float(largest_cluster_share),
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "silhouette_interpretation": _interpret_silhouette(silhouette),
        "davies_bouldin_interpretation": _interpret_davies_bouldin(davies_bouldin),
        "calinski_harabasz_interpretation": _interpret_calinski_harabasz(calinski_harabasz),
        "noise_ratio_interpretation": _interpret_noise_ratio(float(noise_ratio)),
        "largest_cluster_share_interpretation": _interpret_largest_cluster_share(float(largest_cluster_share)),
    }


def get_cluster_top_races_by_startlist_score(
    result_df: pl.DataFrame,
    top_n: int = 3,
) -> dict[int, list[dict[str, object]]]:
    required_cols = ["cluster_label", "cluster_name", "startlist_score", "name", "year", "date"]
    missing_cols = [column_name for column_name in required_cols if column_name not in result_df.columns]
    if missing_cols:
        raise ClusteringConfigError(
            "Cannot compute top races per cluster. Missing columns: "
            f"{missing_cols}"
        )

    ranked_df = result_df.with_columns(
        pl.col("startlist_score").cast(pl.Float64, strict=False).alias("_startlist_score_float")
    )
    cluster_labels = (
        ranked_df.select("cluster_label")
        .unique()
        .to_series()
        .to_list()
    )

    top_races_by_cluster: dict[int, list[dict[str, object]]] = {}
    for cluster_label in sorted([int(label) for label in cluster_labels if int(label) != -1]):
        top_rows = (
            ranked_df
            .filter(pl.col("cluster_label") == cluster_label)
            .sort("_startlist_score_float", descending=True, nulls_last=True)
            .head(top_n)
            .select("cluster_name", "name", "year", "date", "_startlist_score_float")
            .rename({"_startlist_score_float": "startlist_score"})
            .to_dicts()
        )
        top_races_by_cluster[cluster_label] = top_rows

    return top_races_by_cluster


def get_cluster_most_central_races(
    result_df: pl.DataFrame,
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    top_n: int = 3,
) -> dict[int, list[dict[str, object]]]:
    required_cols = ["cluster_label", "cluster_name", "name", "year", "date"]
    missing_cols = [column_name for column_name in required_cols if column_name not in result_df.columns]
    if missing_cols:
        raise ClusteringConfigError(
            "Cannot compute central races per cluster. Missing columns: "
            f"{missing_cols}"
        )
    if dist_matrix.shape[0] != result_df.height or labels.shape[0] != result_df.height:
        raise ClusteringConfigError(
            "Cannot compute central races per cluster. Length mismatch between features, labels, and rows."
        )

    row_names = (
        result_df.select(pl.col("name").cast(pl.Utf8, strict=False))
        .fill_null("unknown")
        .to_series()
        .to_list()
    )
    row_years = (
        result_df.select(pl.col("year").cast(pl.Utf8, strict=False))
        .fill_null("unknown")
        .to_series()
        .to_list()
    )
    row_dates = (
        result_df.select(pl.col("date").cast(pl.Utf8, strict=False))
        .fill_null("unknown")
        .to_series()
        .to_list()
    )

    unique_labels = sorted([int(value) for value in np.unique(labels) if int(value) != -1])
    central_races_by_cluster: dict[int, list[dict[str, object]]] = {}

    for cluster_label in unique_labels:
        cluster_indices = np.flatnonzero(labels == cluster_label)
        if cluster_indices.size == 0:
            central_races_by_cluster[cluster_label] = []
            continue

        dist_cluster = dist_matrix[np.ix_(cluster_indices, cluster_indices)]

        finite_mask = np.isfinite(dist_cluster)
        if np.any(~finite_mask):
            finite_values = dist_cluster[finite_mask]
            positive_finite_values = finite_values[finite_values > 0]
            fallback_distance = float(np.max(positive_finite_values)) if positive_finite_values.size else 1.0
            dist_cluster = np.where(finite_mask, dist_cluster, fallback_distance)

        mean_distance_to_cluster = np.mean(dist_cluster, axis=1)
        top_local_indices = np.argsort(mean_distance_to_cluster)[:top_n]

        central_rows: list[dict[str, object]] = []
        for local_index in top_local_indices.tolist():
            global_index = int(cluster_indices[local_index])
            central_rows.append(
                {
                    "cluster_name": cluster_label_to_name(cluster_label),
                    "name": row_names[global_index],
                    "year": row_years[global_index],
                    "date": row_dates[global_index],
                    "mean_distance_to_cluster": float(mean_distance_to_cluster[local_index]),
                }
            )
        central_races_by_cluster[cluster_label] = central_rows

    return central_races_by_cluster


def plot_clusters_static(
    x_plot: np.ndarray,
    labels: np.ndarray,
    won_how_clean_values: np.ndarray,
    features: list[str],
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    dims = x_plot.shape[1]
    if dims not in (2, 3):
        raise ClusteringConfigError(
            f"Expected 2 or 3 features for plotting, got {dims}."
        )

    if won_how_clean_values.shape[0] != x_plot.shape[0]:
        raise ClusteringConfigError(
            "Length mismatch between plot points and won_how_clean values."
        )

    categories = normalize_won_how_clean_values(won_how_clean_values)
    unique_categories = [
        category
        for category in [
            "large_sprint",
            "small_sprint",
            "duo_sprint",
            "solo",
            "time_trial",
            "unknown",
        ]
        if np.any(categories == category)
    ]

    if dims == 2:
        fig, ax = plt.subplots(figsize=(10, 7))
        for category in unique_categories:
            mask = categories == category
            ax.scatter(
                x_plot[mask, 0],
                x_plot[mask, 1],
                s=22,
                alpha=0.8,
                color=WON_HOW_COLOR_MAP[category],
                label=category,
            )
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        ax.set_title(title or "Density Clustering (2D) - colored by won_how_clean")
    else:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        for category in unique_categories:
            mask = categories == category
            ax.scatter(
                x_plot[mask, 0],
                x_plot[mask, 1],
                x_plot[mask, 2],
                s=22,
                alpha=0.8,
                color=WON_HOW_COLOR_MAP[category],
                label=category,
            )
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        ax.legend(loc="best")
        ax.set_title(title or "Density Clustering (3D) - colored by won_how_clean")

    fig.tight_layout()
    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        print(f"Saved cluster plot to {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_clusters_interactive(
    x_plot: np.ndarray,
    labels: np.ndarray,
    won_how_clean_values: np.ndarray,
    features: list[str],
    race_names: np.ndarray,
    race_years: np.ndarray,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    if not PLOTLY_AVAILABLE:
        raise ClusteringConfigError(
            "Interactive plotting requested but plotly is not installed."
        )
    dims = x_plot.shape[1]
    if dims not in (2, 3):
        raise ClusteringConfigError(
            f"Expected 2 or 3 features for plotting, got {dims}."
        )
    if won_how_clean_values.shape[0] != x_plot.shape[0]:
        raise ClusteringConfigError(
            "Length mismatch between plot points and won_how_clean values."
        )

    categories = normalize_won_how_clean_values(won_how_clean_values)
    plot_df = pl.DataFrame(
        {
            features[0]: x_plot[:, 0],
            features[1]: x_plot[:, 1],
            "won_how_clean": categories.tolist(),
            "race_name": race_names.tolist(),
            "year": race_years.tolist(),
            "cluster_label": labels.tolist(),
            "cluster_name": [cluster_label_to_name(int(label)) for label in labels.tolist()],
        }
    )
    if dims == 3:
        plot_df = plot_df.with_columns(pl.Series(features[2], x_plot[:, 2].tolist()))

    plot_pd = plot_df.to_pandas()
    hover_data = {
        "race_name": True,
        "year": True,
        "cluster_label": True,
        "cluster_name": True,
    }

    if dims == 2:
        fig = px.scatter(
            plot_pd,
            x=features[0],
            y=features[1],
            color="won_how_clean",
            color_discrete_map=WON_HOW_COLOR_MAP,
            title=title or "Interactive Density Clustering (2D)",
            hover_data=hover_data,
            category_orders={"won_how_clean": list(WON_HOW_COLOR_MAP.keys())},
        )
    else:
        fig = px.scatter_3d(
            plot_pd,
            x=features[0],
            y=features[1],
            z=features[2],
            color="won_how_clean",
            color_discrete_map=WON_HOW_COLOR_MAP,
            title=title or "Interactive Density Clustering (3D)",
            hover_data=hover_data,
            category_orders={"won_how_clean": list(WON_HOW_COLOR_MAP.keys())},
        )

    fig.update_traces(marker={"size": 6, "opacity": 0.8})
    fig.update_layout(legend_title_text="won_how_clean")

    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() != ".html":
            output = output.with_suffix(".html")
        fig.write_html(str(output), include_plotlyjs="cdn")
        print(f"Saved interactive cluster plot to {output}")

    if show:
        fig.show()


def plot_cluster_labels_static(
    x_plot: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    plot_umap_clusters_static(
        x_plot=x_plot,
        labels=labels,
        title=title,
        save_path=None,
        show=PLOT_SHOW,
    )


def _cluster_label_color_map(labels: np.ndarray) -> dict[int, str]:
    unique_labels = sorted([int(label) for label in np.unique(labels)])
    color_map: dict[int, str] = {-1: "#b0b0b0"}
    color_index = 0
    for cluster_label in unique_labels:
        if cluster_label == -1:
            continue
        color_map[cluster_label] = CLUSTER_PLOT_COLORS[color_index % len(CLUSTER_PLOT_COLORS)]
        color_index += 1
    return color_map


def plot_umap_clusters_static(
    x_plot: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    if x_plot.shape[1] != 2:
        raise ClusteringConfigError(
            f"UMAP inspection expects 2 dimensions, got {x_plot.shape[1]}."
        )
    unique_labels = np.unique(labels)
    color_map = _cluster_label_color_map(labels)
    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster_label in unique_labels:
        mask = labels == cluster_label
        cluster_label_int = int(cluster_label)
        label_name = cluster_label_to_name(cluster_label_int)
        ax.scatter(
            x_plot[mask, 0],
            x_plot[mask, 1],
            s=22,
            alpha=0.8,
            color=color_map.get(cluster_label_int, "#333333"),
            label=label_name,
        )
    ax.set_xlabel("umap_1")
    ax.set_ylabel("umap_2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        print(f"Saved UMAP cluster plot to {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_umap_clusters_interactive(
    x_plot: np.ndarray,
    labels: np.ndarray,
    race_names: np.ndarray,
    race_years: np.ndarray,
    title: str,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    if not PLOTLY_AVAILABLE:
        raise ClusteringConfigError(
            "Interactive plotting requested but plotly is not installed."
        )
    if x_plot.shape[1] != 2:
        raise ClusteringConfigError(
            f"UMAP inspection expects 2 dimensions, got {x_plot.shape[1]}."
        )

    label_values = [int(label) for label in labels.tolist()]
    label_names = [cluster_label_to_name(label) for label in label_values]
    color_map = _cluster_label_color_map(labels)
    plot_df = pl.DataFrame(
        {
            "umap_1": x_plot[:, 0],
            "umap_2": x_plot[:, 1],
            "race_name": race_names.tolist(),
            "year": race_years.tolist(),
            "cluster_label": label_values,
            "cluster_name": label_names,
        }
    )
    plot_pd = plot_df.to_pandas()
    fig = px.scatter(
        plot_pd,
        x="umap_1",
        y="umap_2",
        color="cluster_name",
        color_discrete_map={cluster_label_to_name(label): color for label, color in color_map.items()},
        title=title,
        hover_data={
            "race_name": True,
            "year": True,
            "cluster_label": True,
            "cluster_name": True,
        },
    )
    fig.update_traces(marker={"size": 6, "opacity": 0.8})
    fig.update_layout(legend_title_text="cluster")
    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() != ".html":
            output = output.with_suffix(".html")
        fig.write_html(str(output), include_plotlyjs="cdn")
        print(f"Saved interactive UMAP cluster plot to {output}")
    if show:
        fig.show()


def run_clustering() -> tuple[pl.DataFrame, dict[str, object]]:
    raw_df, df, selected_features, feature_space, dist_matrix = prepare_clustering_inputs()
    noise_labels = cluster_distance_matrix(dist_matrix)
    noise_summary = summarize_clusters(noise_labels)
    embedding = compute_umap_embedding(dist_matrix, n_components=2)
    hierarchical_analyses = analyze_hierarchical_clustering(
        dist_matrix=dist_matrix,
        embedding=embedding,
    )
    best_analysis = select_best_hierarchical_analysis(hierarchical_analyses)
    labels = best_analysis["labels"]
    summary = best_analysis["summary"]
    quality = best_analysis["quality"]

    race_name_values = (
        df.select(pl.col("name").cast(pl.Utf8, strict=False))
        .fill_null("unknown")
        .to_series()
        .to_numpy()
        if "name" in df.columns
        else np.array(["unknown"] * df.height)
    )
    race_year_values = (
        df.select(pl.col("year").cast(pl.Utf8, strict=False))
        .fill_null("unknown")
        .to_series()
        .to_numpy()
        if "year" in df.columns
        else np.array(["unknown"] * df.height)
    )
    print("Data path:", resolve_data_path())
    print("Rows before XGBoost filter:", raw_df.height)
    print("Rows after XGBoost filter:", df.height)
    print("Classification filter:", TOP_TIER_CLASSIFICATIONS)
    print("Profile score filter:", f"<= {MAX_PROFILE_SCORE} (NaN kept)")
    print("Requested features:", selected_features)
    print("Numeric features:", feature_space["numeric_features"])
    print("Category features:", CATEGORY_CLUSTER_FEATURES)
    print("Category feature weight:", CATEGORY_CLUSTER_FEATURE_WEIGHT)
    print("Categorical features used in mixed distance:", feature_space["categorical_features"])
    print()
    print("=" * 80)
    print("CLUSTERING METHOD: HIERARCHICAL AGGLOMERATIVE CLUSTERING")
    print("=" * 80)
    print("Step 1. Distance Matrix: Gower-style mixed numeric/categorical distance")
    print("        - Numeric features normalized min-max, L1 distance, 4 features")
    print("        - Categorical features: mismatch penalty with weight", CATEGORY_CLUSTER_FEATURE_WEIGHT)
    print("        - Missing values treated specially (no contribution to distance)")
    print()
    print("Step 2. Linkage: Hierarchical agglomerative clustering with", HIERARCHICAL_LINKAGE_METHOD, "linkage")
    print("        - Builds dendrogram from distance matrix")
    print("        - No hardcoded parameters (unlike HDBSCAN)")
    print()
    print("Step 3. Cut Selection: Test multiple cluster counts (k) and select best by silhouette")
    print("        - Candidate k values:", HIERARCHICAL_ANALYSIS_CLUSTER_COUNTS)
    print("        - Best selected k:", best_analysis["n_clusters_requested"], "clusters")
    print()
    print("Step 4. Noise Diagnostic: HDBSCAN run separately to count outliers")
    print("        - HDBSCAN points marked as noise (-1):", noise_summary["n_noise"])
    print("        - Hierarchical clustering assigns ALL points to clusters (no noise)")
    print("=" * 80)
    print()
    print("Noise diagnostic (HDBSCAN only):", noise_summary["n_noise"], "points")
    print("Noise diagnostic cluster sizes:", noise_summary["cluster_sizes"])
    print()
    print("Hierarchical output:")
    print("- Best cut selected:", best_analysis["n_clusters_requested"], "clusters")
    print("- Assigned clusters:", summary["n_clusters"])
    print("- Cluster sizes:", summary["cluster_sizes"])
    print()
    print("\nHierarchical quality metrics by candidate cut:")
    print("(Silhouette: -1=bad, 0=no structure, 1=perfect; Davies-Bouldin: lower=better; Calinski-Harabasz: higher=better)")
    for analysis in hierarchical_analyses:
        candidate_summary = analysis["summary"]
        candidate_quality = analysis["quality"]
        print(
            f"- k={analysis['n_clusters_requested']}: "
            f"silhouette={candidate_quality['silhouette']:.4f} "
            f"db={candidate_quality['davies_bouldin']:.4f} "
            f"ch={candidate_quality['calinski_harabasz']:.1f} "
            f"largest_share={candidate_quality['largest_cluster_share']:.4f} "
            f"sizes={candidate_summary['cluster_sizes']}"
        )
    print("\nSelected hierarchical cut metrics:")
    print(
        "- silhouette:",
        quality["silhouette"],
        "->",
        quality["silhouette_interpretation"],
        "(good: >=0.30, bad: <0.10)",
    )
    print(
        "- davies_bouldin:",
        quality["davies_bouldin"],
        "->",
        quality["davies_bouldin_interpretation"],
        "(on UMAP embedding; heuristic only)",
    )
    print(
        "- calinski_harabasz:",
        quality["calinski_harabasz"],
        "->",
        quality["calinski_harabasz_interpretation"],
        "(on UMAP embedding; heuristic only)",
    )
    print(
        "- largest_cluster_share:",
        quality["largest_cluster_share"],
        "->",
        quality["largest_cluster_share_interpretation"],
        "(good: <0.60, bad: >0.80)",
    )
    print(
        "won_how_clean counts:",
        df.group_by("won_how_clean").agg(pl.len().alias("count")).sort("count", descending=True),
    )

    label_values = [int(label) for label in labels.tolist()]
    result_df = df.with_columns(
        pl.Series(name="cluster_label", values=label_values),
        pl.Series(name="cluster_name", values=[cluster_label_to_name(label) for label in label_values]),
    )
    cluster_top_races = get_cluster_top_races_by_startlist_score(result_df, top_n=3)
    cluster_central_races = get_cluster_most_central_races(
        result_df=result_df,
        dist_matrix=dist_matrix,
        labels=labels,
        top_n=3,
    )
    print("\nTop 3 races per cluster by startlist_score (name | year | date):")
    if not cluster_top_races:
        print("- No clusters available.")
    for cluster_label, races in cluster_top_races.items():
        cluster_name = cluster_label_to_name(cluster_label)
        print(f"- cluster {cluster_label} ({cluster_name}):")
        for index, race in enumerate(races, start=1):
            print(
                f"  {index}. {race.get('name')} | {race.get('year')} | {race.get('date')} "
                f"(startlist_score={race.get('startlist_score')})"
            )

    print("\nTop 3 most central races per cluster (name | year | date):")
    if not cluster_central_races:
        print("- No clusters available.")
    for cluster_label, races in cluster_central_races.items():
        cluster_name = cluster_label_to_name(cluster_label)
        print(f"- cluster {cluster_label} ({cluster_name}):")
        for index, race in enumerate(races, start=1):
            print(
                f"  {index}. {race.get('name')} | {race.get('year')} | {race.get('date')} "
                f"(mean_distance_to_cluster={race.get('mean_distance_to_cluster')})"
            )

    plot_title = (
        f"Hierarchical clustering on Gower-style distance - "
        f"k={best_analysis['n_clusters_requested']} hdbscan_noise={noise_summary['n_noise']}"
    )
    if PLOT_INTERACTIVE and PLOTLY_AVAILABLE:
        plot_umap_clusters_interactive(
            x_plot=embedding,
            labels=labels,
            race_names=race_name_values,
            race_years=race_year_values,
            title=plot_title,
            save_path=PLOT_SAVE_PATH,
            show=PLOT_SHOW,
        )
    else:
        if PLOT_INTERACTIVE and not PLOTLY_AVAILABLE:
            print("Plotly not installed; falling back to static matplotlib plot.")
        plot_umap_clusters_static(
            x_plot=embedding,
            labels=labels,
            title=plot_title,
            save_path=PLOT_SAVE_PATH,
            show=PLOT_SHOW,
        )

    if "race_id" not in result_df.columns:
        raise ClusteringConfigError(
            "Cannot create race_cluster_features: missing required column 'race_id'."
        )
    race_cluster_features = result_df.select(
        pl.col("race_id").alias("race_id"),
        pl.col("cluster_label").cast(pl.Int64, strict=False).alias("cluster_number"),
        pl.col("cluster_name").cast(pl.Utf8, strict=False).alias("cluster_name"),
    )
    race_cluster_features_output_path = Path("data_v2") / "race_cluster_features.parquet"
    race_cluster_features_output_path.parent.mkdir(parents=True, exist_ok=True)
    race_cluster_features.write_parquet(race_cluster_features_output_path)
    print(f"Saved race cluster features to {race_cluster_features_output_path}")

    summary["quality"] = quality
    summary["cluster_top_races"] = cluster_top_races
    summary["cluster_central_races"] = cluster_central_races
    summary["cluster_name_map"] = CLUSTER_NAME_MAP
    summary["race_cluster_features_path"] = str(race_cluster_features_output_path)
    summary["noise_diagnostic"] = noise_summary
    summary["hierarchical_analyses"] = hierarchical_analyses
    summary["selected_cut"] = int(best_analysis["n_clusters_requested"])
    return result_df, summary


def inspect_cluster_possibilites() -> None:
    raw_df, df, selected_features, feature_space, dist_matrix = prepare_clustering_inputs()
    noise_labels = cluster_distance_matrix(dist_matrix)
    noise_summary = summarize_clusters(noise_labels)
    embedding = compute_umap_embedding(dist_matrix, n_components=2)
    hierarchical_analyses = analyze_hierarchical_clustering(
        dist_matrix=dist_matrix,
        embedding=embedding,
    )
    best_analysis = select_best_hierarchical_analysis(hierarchical_analyses)
    labels = best_analysis["labels"]
    race_name_values = (
        df.select(pl.col("name").cast(pl.Utf8, strict=False))
        .fill_null("unknown")
        .to_series()
        .to_numpy()
        if "name" in df.columns
        else np.array(["unknown"] * df.height)
    )
    race_year_values = (
        df.select(pl.col("year").cast(pl.Utf8, strict=False))
        .fill_null("unknown")
        .to_series()
        .to_numpy()
        if "year" in df.columns
        else np.array(["unknown"] * df.height)
    )
    print("UMAP inspection for clustering possibilities")
    print("Data path:", resolve_data_path())
    print("Rows before XGBoost filter:", raw_df.height)
    print("Rows after XGBoost filter:", df.height)
    print("Selected features:", selected_features)
    print("Numeric features:", feature_space["numeric_features"])
    print("Categorical features:", feature_space["categorical_features"])
    print("Noise points (HDBSCAN diagnostic):", noise_summary["n_noise"])
    print("Best hierarchical cut:", best_analysis["n_clusters_requested"])
    for analysis in hierarchical_analyses:
        quality = analysis["quality"]
        print(
            f"- k={analysis['n_clusters_requested']}: "
            f"silhouette={quality['silhouette']} "
            f"db={quality['davies_bouldin']} "
            f"ch={quality['calinski_harabasz']} "
            f"largest_cluster_share={quality['largest_cluster_share']}"
        )
    if PLOT_INTERACTIVE and PLOTLY_AVAILABLE:
        plot_umap_clusters_interactive(
            x_plot=embedding,
            labels=labels,
            race_names=race_name_values,
            race_years=race_year_values,
            title="UMAP inspection - colored by hierarchical cluster label",
            save_path=None,
            show=PLOT_SHOW,
        )
    else:
        plot_umap_clusters_static(
            x_plot=embedding,
            labels=labels,
            title="UMAP inspection - colored by hierarchical cluster label",
            save_path=None,
            show=PLOT_SHOW,
        )


def compare_hierarchical_cuts(cut_k_values: list[int] | None = None) -> None:
    raw_df, df, selected_features, feature_space, dist_matrix = prepare_clustering_inputs()
    noise_labels = cluster_distance_matrix(dist_matrix)
    noise_summary = summarize_clusters(noise_labels)
    embedding = compute_umap_embedding(dist_matrix, n_components=2)
    hierarchical_analyses = analyze_hierarchical_clustering(
        dist_matrix=dist_matrix,
        embedding=embedding,
    )

    if cut_k_values is None:
        cut_k_values = [3, 12]

    selected_analyses = {
        analysis["n_clusters_requested"]: analysis
        for analysis in hierarchical_analyses
        if analysis["n_clusters_requested"] in cut_k_values
    }

    race_name_values = (
        df.select(pl.col("name").cast(pl.Utf8, strict=False))
        .fill_null("unknown")
        .to_series()
        .to_numpy()
        if "name" in df.columns
        else np.array(["unknown"] * df.height)
    )
    race_year_values = (
        df.select(pl.col("year").cast(pl.Utf8, strict=False))
        .fill_null("unknown")
        .to_series()
        .to_numpy()
        if "year" in df.columns
        else np.array(["unknown"] * df.height)
    )

    print("=" * 80)
    print("COMPARING HIERARCHICAL CUTS")
    print("=" * 80)
    print("Data path:", resolve_data_path())
    print("Rows:", df.height)
    print("Distance: Gower-style mixed numeric/categorical")
    print("Linkage: ", HIERARCHICAL_LINKAGE_METHOD)
    print("Noise diagnostic (HDBSCAN count only):", noise_summary["n_noise"])
    print("Comparing k values:", sorted(cut_k_values))
    print()
    print("Metric Explanations:")
    print("  Silhouette:       Range [-1, 1]. Higher = better. -1=bad, 0=no structure, 1=perfect.")
    print("  Davies-Bouldin:   Lower = better. Ratio of within to between cluster distances.")
    print("  Calinski-Harabasz: Higher = better. Ratio of between to within cluster variance.")
    print("  Largest share:    Fraction of points in biggest cluster. Lower = more balanced.")
    print("=" * 80)
    print()

    for k in sorted(cut_k_values):
        if k not in selected_analyses:
            print(f"⚠ k={k} not found in hierarchical analyses. Skipping.")
            continue
        analysis = selected_analyses[k]
        quality = analysis["quality"]
        summary = analysis["summary"]
        print(f"Cut k={k}:")
        print(f"  Clusters assigned: {summary['n_clusters']}")
        print(f"  Cluster sizes: {summary['cluster_sizes']}")
        print(f"  Silhouette: {quality['silhouette']:.4f} ({quality['silhouette_interpretation']})")
        print(f"  Davies-Bouldin: {quality['davies_bouldin']:.4f} ({quality['davies_bouldin_interpretation']})")
        print(f"  Calinski-Harabasz: {quality['calinski_harabasz']:.1f} ({quality['calinski_harabasz_interpretation']})")
        print(f"  Largest cluster share: {quality['largest_cluster_share']:.4f} ({quality['largest_cluster_share_interpretation']})")
        print()

    for k in sorted(cut_k_values):
        if k not in selected_analyses:
            continue
        analysis = selected_analyses[k]
        labels = analysis["labels"]

        plot_title = f"Hierarchical clustering k={k} on Gower-style distance (noise={noise_summary['n_noise']})"
        if PLOT_INTERACTIVE and PLOTLY_AVAILABLE:
            plot_umap_clusters_interactive(
                x_plot=embedding,
                labels=labels,
                race_names=race_name_values,
                race_years=race_year_values,
                title=plot_title,
                save_path=None,
                show=PLOT_SHOW,
            )
        else:
            if PLOT_INTERACTIVE and not PLOTLY_AVAILABLE:
                print("Plotly not installed; falling back to static matplotlib plot.")
            plot_umap_clusters_static(
                x_plot=embedding,
                labels=labels,
                title=plot_title,
                save_path=None,
                show=PLOT_SHOW,
            )


def main() -> None:
    run_clustering()
    # inspect_cluster_possibilites()
    # compare_hierarchical_cuts(cut_k_values=[3, 12])


if __name__ == "__main__":
    main()
