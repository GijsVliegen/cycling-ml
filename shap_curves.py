"""
SHAP analysis for RaceModel (NDCG ranking model).

Run with:
    python shap_curves.py
"""

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import shap
from data_science.data_science_functions import (
    EMBEDDING_SIZE,
    filter_data,
)
from xgboost_functions import (
    RaceModel,
    load_rider_personal_data,
    load_result_features_with_pre_embed,
    data_path,
    DEFAULT_DATA_DIR,
)

# ── feature name list ─────────────────────────────────────────────────────────
# Must mirror the exact hstack order in RaceModel.to_xgboost_format.

def make_unique_names(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        if name not in counts:
            counts[name] = 0
            unique.append(name)
            continue
        counts[name] += 1
        unique.append(f"{name}__dup{counts[name]}")
    return unique


def find_identical_columns(X: np.ndarray, feature_names: list[str], max_report: int = 20) -> list[tuple[str, str]]:
    hash_buckets: dict[int, list[int]] = {}
    for col_idx in range(X.shape[1]):
        column_hash = hash(X[:, col_idx].tobytes())
        hash_buckets.setdefault(column_hash, []).append(col_idx)

    identical_pairs: list[tuple[str, str]] = []
    for indices in hash_buckets.values():
        if len(indices) < 2:
            continue
        for left_i in range(len(indices)):
            for right_i in range(left_i + 1, len(indices)):
                col_left = indices[left_i]
                col_right = indices[right_i]
                if np.array_equal(X[:, col_left], X[:, col_right]):
                    identical_pairs.append((feature_names[col_left], feature_names[col_right]))
                    if len(identical_pairs) >= max_report:
                        return identical_pairs
    return identical_pairs


def group_importance(mean_abs_shap: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    def sum_for(predicate):
        idxs = [i for i, name in enumerate(feature_names) if predicate(name)]
        return float(mean_abs_shap[idxs].sum()) if idxs else 0.0

    return {
        "embed_diff": sum_for(lambda n: n.startswith("embed_diff_")),
        "yearly": sum_for(lambda n: "_yr" in n),
        "personal": sum_for(lambda n: n in {"age", "height", "weight"}),
        "team_rank": sum_for(lambda n: n == "predicted_rank_in_team"),
        "race_stats": sum_for(
            lambda n: n in {
                "distance_km",
                "elevation_m",
                "profile_score",
                "profile_score_last_25k",
                "final_km_percentage",
                "year",
                "startlist_score",
            }
        ),
    }


def compute_feature_health(X: np.ndarray, feature_names: list[str]) -> list[dict]:
    rows = []
    for col_idx, name in enumerate(feature_names):
        col = X[:, col_idx]
        finite = np.isfinite(col)
        finite_ratio = float(finite.mean())
        if finite.any():
            nonzero_ratio = float((np.abs(col[finite]) > 1e-12).mean())
            col_mean = float(np.mean(col[finite]))
            col_std = float(np.std(col[finite]))
        else:
            nonzero_ratio = 0.0
            col_mean = float("nan")
            col_std = float("nan")
        rows.append(
            {
                "feature": name,
                "finite_ratio": finite_ratio,
                "nonzero_ratio": nonzero_ratio,
                "mean": col_mean,
                "std": col_std,
            }
        )
    return rows


def build_feature_names(model: RaceModel) -> list[str]:
    nr_years = 3
    embed_diff_names = [f"embed_diff_{i}" for i in range(1, EMBEDDING_SIZE + 1)]

    return (
        model.rider_result_features                                                 # stats + cosine + l1
        + [f"{f}_yr{i+1}" for i in range(nr_years) for f in model.rider_yearly_features]  # 3 years × 4 yearly feats
        + model.rider_personal_features                                             # age, height, weight
        + [model.team_model_rank_feature]                                           # predicted_rank_in_team
        + model.race_features                                                       # race stats
        + embed_diff_names                                                          # signed rider−race embedding diff
    )


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: str = DEFAULT_DATA_DIR):
    result_features_df = load_result_features_with_pre_embed(data_dir)
    results_embedded_df = pl.read_parquet(data_path(data_dir, "results_embedded_df.parquet"))
    races_inference_embedded_df = pl.read_parquet(data_path(data_dir, "races_inference_embedded_df.parquet"))
    riders_yearly_data = pl.read_parquet(data_path(data_dir, "rider_yearly_stats_df.parquet"))
    riders_personal_data = load_rider_personal_data(data_dir)
    races_df = pl.read_parquet(data_path(data_dir, "races_df.parquet"))

    riders_yearly_data = riders_yearly_data.with_columns(pl.all().replace(-1, 0))
    necessary_races, necessary_results = filter_data(races_df, result_features_df)

    results_features = results_embedded_df.join(
        necessary_results, on=["race_id", "name"], how="right"
    ).drop("year", "year_right")
    races_features = races_inference_embedded_df.join(
        necessary_races, on=["race_id"], how="right"
    )

    return results_features, riders_yearly_data, riders_personal_data, races_features


# ── SHAP plots ────────────────────────────────────────────────────────────────

def run_shap(data_dir: str = DEFAULT_DATA_DIR, max_samples: int = 5_000):
    print("Loading data …")
    results_features, riders_yearly_data, riders_personal_data, races_features = load_data(data_dir)

    print("Building feature matrix …")
    model = RaceModel(data_dir=data_dir, test_mode=False)
    model.load_model()

    X_train, y_train_ndcg, y_train_binary, train_groups, X_test, y_test_ndcg, y_test_binary, test_groups = (
        model.to_xgboost_format(
            result_features_df=results_features,
            riders_yearly_data=riders_yearly_data,
            riders_personal_data=riders_personal_data,
            races_features_df=races_features,
        )
    )

    feature_names = build_feature_names(model)
    n_features = X_train.shape[1]
    if len(feature_names) != n_features:
        print(
            f"WARNING: feature_names has {len(feature_names)} entries but X has {n_features} columns. "
            "Falling back to generic names."
        )
        feature_names = [f"f{i}" for i in range(n_features)]
    else:
        feature_names = make_unique_names(feature_names)

    # Use a random subsample to keep SHAP tractable
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(max_samples, len(X_test)), replace=False)
    X_sample = X_test[idx]

    identical_pairs = find_identical_columns(X_sample, feature_names)
    if identical_pairs:
        print("Potentially duplicated feature columns detected (identical values):")
        for left_name, right_name in identical_pairs:
            print(f"  - {left_name} == {right_name}")
    else:
        print("No identical feature columns found in sampled test data.")

    print(f"Computing SHAP values on {len(X_sample)} test samples …")
    explainer = shap.TreeExplainer(model.bst)
    shap_values_raw = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value
    if np.isscalar(expected_value):
        base_values = np.full(X_sample.shape[0], expected_value)
    else:
        base_values = np.asarray(expected_value)
        if base_values.ndim == 0:
            base_values = np.full(X_sample.shape[0], float(base_values))
        elif base_values.shape[0] != X_sample.shape[0]:
            base_values = np.full(X_sample.shape[0], float(base_values.reshape(-1)[0]))

    shap_values = shap.Explanation(
        values=shap_values_raw,
        base_values=base_values,
        data=X_sample,
        feature_names=feature_names,
    )

    # ── 1. Summary bar plot (mean |SHAP|) ─────────────────────────────────
    plt.figure()
    shap.plots.bar(shap_values, max_display=30, show=False)
    plt.title("Mean |SHAP| – feature importance")
    plt.tight_layout()
    plt.savefig("shap_bar.png", dpi=150)
    print("Saved shap_bar.png")

    # ── 2. Beeswarm plot (value × direction) ──────────────────────────────
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=30, show=False)
    plt.title("SHAP beeswarm")
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=150)
    print("Saved shap_beeswarm.png")

    # ── 3. Dependence plots for top-5 features ────────────────────────────
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top5_idx = np.argsort(mean_abs)[::-1][:5]
    for rank, fi in enumerate(top5_idx):
        fname = feature_names[fi]
        fig, ax = plt.subplots()
        shap.plots.scatter(shap_values[:, fi], ax=ax, show=False)
        ax.set_title(f"SHAP dependence – {fname}")
        ax.set_xlabel(fname)
        ax.set_ylabel("SHAP value")
        fig.tight_layout()
        safe_name = fname.replace("/", "_")
        fig.savefig(f"shap_dep_{rank+1}_{safe_name}.png", dpi=150)
        print(f"Saved shap_dep_{rank+1}_{safe_name}.png")
        plt.close(fig)

    # ── 4. Automated report files (CSV + TXT) ────────────────────────────
    order = np.argsort(mean_abs)[::-1]
    importance_df = pl.DataFrame(
        {
            "rank": np.arange(1, len(feature_names) + 1),
            "feature": [feature_names[i] for i in order],
            "mean_abs_shap": [float(mean_abs[i]) for i in order],
        }
    )
    importance_df.write_csv("shap_importance_report.csv")
    print("Saved shap_importance_report.csv")

    health_rows = compute_feature_health(X_sample, feature_names)
    health_df = pl.DataFrame(health_rows)
    health_df.write_csv("shap_feature_health.csv")
    print("Saved shap_feature_health.csv")

    grouped = group_importance(mean_abs, feature_names)
    dead_features = [
        row["feature"]
        for row in health_rows
        if row["finite_ratio"] == 0.0 or row["nonzero_ratio"] == 0.0
    ]

    with open("shap_report.txt", "w") as fh:
        fh.write("SHAP Automated Report\n")
        fh.write("=====================\n\n")
        fh.write(f"Samples used: {len(X_sample)}\n")
        fh.write(f"Total features: {len(feature_names)}\n\n")

        fh.write("Top 20 features by mean |SHAP|\n")
        fh.write("-----------------------------\n")
        for idx in order[:20]:
            fh.write(f"{feature_names[idx]}\t{float(mean_abs[idx]):.6f}\n")
        fh.write("\n")

        fh.write("Grouped contribution (sum mean |SHAP|)\n")
        fh.write("--------------------------------------\n")
        for key, value in grouped.items():
            fh.write(f"{key}\t{value:.6f}\n")
        fh.write("\n")

        fh.write("Identical column pairs (on sampled data)\n")
        fh.write("----------------------------------------\n")
        if identical_pairs:
            for left_name, right_name in identical_pairs:
                fh.write(f"{left_name} == {right_name}\n")
        else:
            fh.write("none\n")
        fh.write("\n")

        fh.write("Dead features (all NaN or all zero in sample)\n")
        fh.write("---------------------------------------------\n")
        if dead_features:
            for name in dead_features:
                fh.write(f"{name}\n")
        else:
            fh.write("none\n")
    print("Saved shap_report.txt")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    run_shap()
