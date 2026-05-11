from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import OPTICS
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    PLOTLY_AVAILABLE = False


DEFAULT_DATA_CANDIDATES = [
    # "data_test/normalized_races_df.parquet",
    # "data/normalized_races_df.parquet",
    "data_v2/races_df.parquet",
]

DEFAULT_FEATURES = [
    "profile_score",
    # "startlist_score",
    "final_km_percentage",
    "profile_score_last_25k",
    "distance_km",
    "elevation_m",
    "won_how_clean",
    # "temp",#
    "classification",
    "avg_speed_kmh"
]

TOP_TIER_CLASSIFICATIONS = [
    "1.UWT",
    "1.Pro",
    "1.HC",
    "2.UWT",
    "2.Pro",
    "2.HC",
    "WT",
    "WC",
    "NC",
]
MAX_PROFILE_SCORE_FOR_MODEL = 500.0


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

CLUSTER_NAME_MAP = {
    0: "sprints men 2",
    1: "puncheur stages",
    2: "mountain stage",
    3: "sprints women",
    4: "time trials",
    5: "sprints men 1",
    6: "spring-classics women",
    7: "spring-classics men",
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


def resolve_data_path(data_path: str | None = None) -> Path:
    if data_path is not None:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data path does not exist: {path}")
        return path

    for candidate in DEFAULT_DATA_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not auto-find race data parquet. Pass --data-path explicitly."
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


def load_race_data(data_path: str | None = None) -> pl.DataFrame:
    path = resolve_data_path(data_path)
    return pl.read_parquet(path)


def filter_races_for_xgboost_input(df: pl.DataFrame) -> pl.DataFrame:
    if "classification" not in df.columns:
        raise ClusteringConfigError(
            "Missing required column 'classification' for filtering."
        )
    if "profile_score" not in df.columns:
        raise ClusteringConfigError(
            "Missing required column 'profile_score' for filtering."
        )
    if "elevation_m" not in df.columns:
        raise ClusteringConfigError(
            "Missing required column 'elevation_m' for filtering."
        )
    profile_score = pl.col("profile_score").cast(pl.Float64, strict=False)
    elevation_m = pl.col("elevation_m").cast(pl.Float64, strict=False)
    profile_unknown = profile_score.is_null() | profile_score.is_nan()
    elevation_unknown = elevation_m.is_null() | elevation_m.is_nan()
    return df.filter(
        pl.col("classification").is_in(TOP_TIER_CLASSIFICATIONS)
        & (
            profile_unknown
            | (profile_score <= MAX_PROFILE_SCORE_FOR_MODEL)
        )
        & ~(profile_unknown & elevation_unknown)
    )


def build_feature_matrix(
    df: pl.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
    categorical_cols = [
        column_name
        for column_name, dtype in feature_df.schema.items()
        if dtype not in numeric_dtypes
    ]

    if categorical_cols:
        feature_df = feature_df.with_columns(
            [
                pl.col(column_name)
                .cast(pl.Utf8, strict=False)
                .fill_null("unknown")
                .alias(column_name)
                for column_name in categorical_cols
            ]
        ).to_dummies(columns=categorical_cols)

    feature_df = feature_df.with_columns(
        [
            pl.col(column_name).cast(pl.Float64, strict=False).alias(column_name)
            for column_name in feature_df.columns
        ]
    )

    matrix = feature_df.to_numpy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix_imputed = imputer.fit_transform(matrix)
    matrix_scaled = scaler.fit_transform(matrix_imputed)
    return matrix_imputed, matrix_scaled, feature_df.columns


def density_cluster(
    x_scaled: np.ndarray,
    min_samples: int = 12,
    xi: float = 0.05,
    min_cluster_size: float = 0.05,
) -> np.ndarray:
    model = OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        cluster_method="xi",
    )
    return model.fit_predict(x_scaled)


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


def evaluate_clustering_quality(x_scaled: np.ndarray, labels: np.ndarray) -> dict[str, float | int | str | None]:
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
        x_assigned = x_scaled[assigned_mask]
        silhouette = float(silhouette_score(x_assigned, assigned_labels))
        davies_bouldin = float(davies_bouldin_score(x_assigned, assigned_labels))
        calinski_harabasz = float(calinski_harabasz_score(x_assigned, assigned_labels))

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


def run_clustering(
    data_path: str | None = None,
    features: list[str] | None = None,
    min_samples: int = 12,
    xi: float = 0.05,
    min_cluster_size: float = 0.05,
    save_path: str | None = None,
    interactive: bool = True,
    show: bool = True,
) -> tuple[pl.DataFrame, dict[str, object]]:
    raw_df = load_race_data(data_path)
    df = filter_races_for_xgboost_input(raw_df)
    if df.height == 0:
        raise ClusteringConfigError(
            "No races left after applying the XGBoost classification filter."
        )
    selected_features = resolve_features(df, features or DEFAULT_FEATURES)
    if len(selected_features) < 2:
        raise ClusteringConfigError(
            "Pick at least 2 features for clustering. "
            f"You passed {len(selected_features)}: {selected_features}"
        )

    x_actual, x_scaled, encoded_feature_names = build_feature_matrix(df, selected_features)
    labels = density_cluster(
        x_scaled,
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
    )

    summary = summarize_clusters(labels)
    quality = evaluate_clustering_quality(x_scaled=x_scaled, labels=labels)
    if "won_how_clean" not in df.columns:
        df = df.with_columns(pl.lit("unknown").alias("won_how_clean"))
    won_how_clean_values = (
        df.select(pl.col("won_how_clean").cast(pl.Utf8, strict=False))
        .fill_null("unknown")
        .to_series()
        .to_numpy()
    )
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
    print("Data path:", resolve_data_path(data_path))
    print("Rows before XGBoost filter:", raw_df.height)
    print("Rows after XGBoost filter:", df.height)
    print("Classification filter:", TOP_TIER_CLASSIFICATIONS)
    print("Profile score filter:", f"<= {MAX_PROFILE_SCORE_FOR_MODEL} (NaN kept)")
    print("Requested features:", selected_features)
    print("Encoded feature count:", len(encoded_feature_names))
    if len(encoded_feature_names) <= 20:
        print("Encoded features:", encoded_feature_names)
    else:
        print("Encoded features (first 20):", encoded_feature_names[:20], "...")
    print("Clustering space:", "normalized (z-score)")
    print("Plot space:", "actual feature values")
    print("Clusters found:", summary["n_clusters"])
    print("Noise points:", summary["n_noise"])
    print("Cluster sizes:", summary["cluster_sizes"])
    print("\nQuality metrics (for high-dimensional analysis):")
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
        "(good: <1.50, bad: >2.50)",
    )
    print(
        "- calinski_harabasz:",
        quality["calinski_harabasz"],
        "->",
        quality["calinski_harabasz_interpretation"],
        "(higher is better; rough guide: >500 strong, <200 weak)",
    )
    print(
        "- noise_ratio:",
        quality["noise_ratio"],
        "->",
        quality["noise_ratio_interpretation"],
        "(good: <0.20, bad: >0.50)",
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
    print("\nTop 3 races per cluster by startlist_score (name | year | date):")
    if not cluster_top_races:
        print("- No non-noise clusters available.")
    for cluster_label, races in cluster_top_races.items():
        cluster_name = cluster_label_to_name(cluster_label)
        print(f"- cluster {cluster_label} ({cluster_name}):")
        for index, race in enumerate(races, start=1):
            print(
                f"  {index}. {race.get('name')} | {race.get('year')} | {race.get('date')} "
                f"(startlist_score={race.get('startlist_score')})"
            )

    plot_title = (
        f"OPTICS clustering ({len(selected_features)}D) - "
        f"clusters={summary['n_clusters']} noise={summary['n_noise']}"
    )
    if x_actual.shape[1] in (2, 3):
        if interactive and PLOTLY_AVAILABLE:
            plot_clusters_interactive(
                x_plot=x_actual,
                labels=labels,
                won_how_clean_values=won_how_clean_values,
                features=encoded_feature_names,
                race_names=race_name_values,
                race_years=race_year_values,
                title=plot_title,
                save_path=save_path,
                show=show,
            )
        else:
            if interactive and not PLOTLY_AVAILABLE:
                print("Plotly not installed; falling back to static matplotlib plot.")
            plot_clusters_static(
                x_plot=x_actual,
                labels=labels,
                won_how_clean_values=won_how_clean_values,
                features=encoded_feature_names,
                title=plot_title,
                save_path=save_path,
                show=show,
            )
    else:
        print(
            f"Skipping plot for {x_actual.shape[1]} encoded dimensions (visualization supports only 2D/3D)."
        )
        print("Use the quality metrics above to assess high-dimensional clustering quality.")

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
    summary["cluster_name_map"] = CLUSTER_NAME_MAP
    summary["race_cluster_features_path"] = str(race_cluster_features_output_path)
    return result_df, summary


def parse_feature_list(raw: str | None) -> list[str]:
    if not raw:
        return DEFAULT_FEATURES.copy()
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster race data with density-based OPTICS and visualize in 2D/3D."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional parquet path. If omitted, auto-detects a known race parquet.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=",".join(DEFAULT_FEATURES),
        help=(
            "Comma-separated features (2 or more). "
            "Aliases supported: ps_25k->profile_score_last_25k, "
            "height_meters->elevation_m. "
            "Plotting is only available for 2D/3D; higher dimensions print quality metrics."
        ),
    )
    parser.add_argument("--min-samples", type=int, default=12)
    parser.add_argument("--xi", type=float, default=0.05)
    parser.add_argument("--min-cluster-size", type=float, default=0.05)
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static matplotlib plot instead of interactive plotly.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to save the plot (.html for interactive, image for static).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    features = parse_feature_list(args.features)
    run_clustering(
        data_path=args.data_path,
        features=features,
        min_samples=args.min_samples,
        xi=args.xi,
        min_cluster_size=args.min_cluster_size,
        save_path=args.save_path,
        interactive=not args.static,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
