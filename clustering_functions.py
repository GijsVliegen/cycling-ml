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
from sklearn.preprocessing import StandardScaler

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    PLOTLY_AVAILABLE = False

FEATURE_ALIASES: dict[str, str] = {
    "ps_25k": "profile_score_last_25k",
    "height_meters": "elevation_m",
}

DEFAULT_DATA_CANDIDATES = [
    # "data_test/normalized_races_df.parquet",
    # "data/normalized_races_df.parquet",
    "data_v2/races_df.parquet",
]

DEFAULT_FEATURES = [
    # "profile_score",
    # "startlist_score",
    # "final_km_percentage",
    "ps_25k",
    "distance_km",
    "height_meters",
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


def resolve_feature_name(feature_name: str) -> str:
    return FEATURE_ALIASES.get(feature_name, feature_name)


def resolve_features(df: pl.DataFrame, requested_features: Iterable[str]) -> list[str]:
    requested = [feature.strip() for feature in requested_features if feature.strip()]
    resolved = [resolve_feature_name(feature) for feature in requested]

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


def build_feature_matrix(df: pl.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    matrix = df.select(features).to_numpy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix_imputed = imputer.fit_transform(matrix)
    matrix_scaled = scaler.fit_transform(matrix_imputed)
    return matrix_imputed, matrix_scaled


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
        }
    )
    if dims == 3:
        plot_df = plot_df.with_columns(pl.Series(features[2], x_plot[:, 2].tolist()))

    plot_pd = plot_df.to_pandas()
    hover_data = {
        "race_name": True,
        "year": True,
        "cluster_label": True,
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
) -> tuple[pl.DataFrame, dict[str, int | dict[int, int]]]:
    raw_df = load_race_data(data_path)
    df = filter_races_for_xgboost_input(raw_df)
    if df.height == 0:
        raise ClusteringConfigError(
            "No races left after applying the XGBoost classification filter."
        )
    selected_features = resolve_features(df, features or DEFAULT_FEATURES)
    if len(selected_features) not in (2, 3):
        raise ClusteringConfigError(
            "Pick exactly 2 or 3 features for visual inspection. "
            f"You passed {len(selected_features)}: {selected_features}"
        )

    x_actual, x_scaled = build_feature_matrix(df, selected_features)
    labels = density_cluster(
        x_scaled,
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
    )

    summary = summarize_clusters(labels)
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
    print("Features:", selected_features)
    print("Clustering space:", "normalized (z-score)")
    print("Plot space:", "actual feature values")
    print("Clusters found:", summary["n_clusters"])
    print("Noise points:", summary["n_noise"])
    print("Cluster sizes:", summary["cluster_sizes"])
    print(
        "won_how_clean counts:",
        df.group_by("won_how_clean").agg(pl.len().alias("count")).sort("count", descending=True),
    )

    result_df = df.with_columns(pl.Series(name="cluster_label", values=labels.tolist()))
    plot_title = (
        f"OPTICS clustering ({len(selected_features)}D) - "
        f"clusters={summary['n_clusters']} noise={summary['n_noise']}"
    )
    if interactive and PLOTLY_AVAILABLE:
        plot_clusters_interactive(
            x_plot=x_actual,
            labels=labels,
            won_how_clean_values=won_how_clean_values,
            features=selected_features,
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
            features=selected_features,
            title=plot_title,
            save_path=save_path,
            show=show,
        )

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
            "Comma-separated features (exactly 2 or 3). "
            "Aliases supported: ps_25k->profile_score_last_25k, "
            "height_meters->elevation_m"
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
