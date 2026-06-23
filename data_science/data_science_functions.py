import math
import polars as pl
from typing import List, Dict
from pathlib import Path
from data_engineering.data_structure_functions import TOP_TIER_CLASSIFICATIONS

import numpy as np

from sklearn.discriminant_analysis import StandardScaler

from sklearn.decomposition import PCA

import gc

EMBEDDING_SIZE = 10 
DEFAULT_DATA_DIR = "data_v2"

# Classifications to include when filtering races for feature creation / inference.
# Covers WorldTour, ProSeries, Hors Catégorie and national/world championship tiers.
MAX_PROFILE_SCORE = 500.0
DEFAULT_TEST_DATA_DIR = "data_test"

RACE_SIMILARITY_COLS = [
    "distance_km", 
    "elevation_m", 
    "profile_score",
    "profile_score_last_25k",
    "final_km_percentage",
    "startlist_score",
]


def data_path(data_dir: str, filename: str) -> str:
    return str(Path(data_dir) / filename)


def ensure_data_dir(data_dir: str) -> Path:
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

def clean_won_how_column(races_df: pl.DataFrame, results_df: pl.DataFrame) -> pl.DataFrame:
    races_df = races_df.join(
        results_df.filter(pl.col("rank") == 1).select("race_id", "breakaway_km"),
        on = "race_id",
        how = "left"
    ).select(["race_id", pl.col("breakaway_km").alias("breakaway_km_first"), "won_how_clean"])
    won_how = pl.col("won_how_clean").cast(pl.Utf8, strict=False)
    sprint_group_size = won_how.str.extract(r"^Sprint of (\d+) riders$", 1).cast(pl.Int64, strict=False)

    return races_df.with_columns(
        pl.when(
            pl.col("breakaway_km_first").is_not_null()
        )
        .then(pl.lit("breakaway"))
        # .when(
        #     pl.col("breakaway_in_top_5") #TODO
        # )
        .when(
            won_how.is_null()
            | (won_how == "")
            | (won_how == "-")
            | (won_how == "Other")
        )
        .then(pl.lit("unknown"))
        .when(won_how == "Time trial")
        .then(pl.lit("time_trial"))
        .when(won_how == "Sprint à deux")
        .then(pl.lit("duo_sprint"))
        .when(won_how.str.contains(r"km solo$", literal=False))
        .then(pl.lit("solo"))
        .when(won_how == "Sprint of small group")
        .then(pl.lit("small_sprint"))
        .when(won_how == "Sprint of large group")
        .then(pl.lit("large_sprint"))
        .when(sprint_group_size.is_not_null() & (sprint_group_size < 10))
        .then(pl.lit("small_sprint"))
        .when(sprint_group_size.is_not_null())
        .then(pl.lit("large_sprint"))
        .otherwise(pl.lit("unknown"))
        .alias("won_how_clean")
    )

def filter_and_clean_races(races_df: pl.DataFrame, results_df: pl.DataFrame) -> pl.DataFrame:

    cols_to_sanitize = [
        "elevation_m",
        "profile_score",
        "profile_score_last_25k",
        "startlist_score",
    ]
    for col in cols_to_sanitize:
        if col not in races_df.columns:
            raise ValueError(f"Missing required column '{col}' for filtering.")
        
    unknown_expressions = [
        pl.when(pl.col(col) == -1)
        .then(float("nan"))
        .otherwise(pl.col(col))
        .alias(col)
        for col in cols_to_sanitize
    ]
    filtered = races_df.filter(
        pl.col("classification").is_in(TOP_TIER_CLASSIFICATIONS)
        & (pl.col("profile_score") <= MAX_PROFILE_SCORE)
    ).with_columns(unknown_expressions)
    clean_won_how = clean_won_how_column(
        filtered,
        results_df
    )
    filtered = filtered.join(
        clean_won_how,
        on = "race_id",
        how = "left"
    )
    return filtered



def normalize_race_data(races_df: pl.DataFrame) -> pl.DataFrame:
    #TODO: use z-scoring?
    return races_df.with_columns([
        (pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min())
        for c in RACE_SIMILARITY_COLS
    ])

def scores_to_probability_results(rider_scores: pl.DataFrame, max_rank_to_predict: int = 10, temperature = 1) -> pl.DataFrame:
    """convert rider scores to probability of getting each result for each rider

    rider_scores should have a col "score"
    
    """
    def compute_plackett_luce_probs(scores: List[float], max_rank_to_predict: int) -> Dict[int, List[float]]:
        """
        Approximate top-k Plackett-Luce probabilities.

        Args:
            scores: list or np.array of non-negative scores (length n)
            max_rank: integer, maximum rank to compute probabilities for

        Returns:
            rank_probs: dict mapping rank -> np.array of probabilities
                        rank_probs[1] is probability of each item finishing 1st, etc.
        """
        n = len(scores)
        if n == 0:
            return []
        scores = np.array(scores, dtype=float)
        n = len(scores)
        remaining = scores.copy()
        rank_probs = {}

        for k in range(1, max_rank_to_predict + 1):
            total = remaining.sum()
            if total == 0:
                probs = np.zeros(n)
            else:
                probs = remaining / total
            rank_probs[k] = list(probs.copy())
            # zero out the item with highest probability to simulate sequential removal
            # optional: if you want strictly sequential draws
            # remaining[np.argmax(probs)] = 0.0
            remaining = remaining * (1 - probs)

        return rank_probs

    # Usage
    scores = rider_scores["score"].head(max_rank_to_predict).to_list()
    scores = [math.exp(s / temperature) for s in scores] # apply temperature
    rank_probs = compute_plackett_luce_probs(scores, max_rank_to_predict)
    padding = [0.0] * (rider_scores.height - max_rank_to_predict)
    for k in range(1, len(rank_probs) + 1):
        rider_scores = rider_scores.with_columns(
            pl.Series(f"rank_{k}_prob", rank_probs[k] + padding), strict = False
        )

    return rider_scores

def scores_to_results(rider_scores: pl.DataFrame, participants: pl.DataFrame, race: pl.DataFrame) -> pl.DataFrame:
    #convert rider scores to results dataframe format
    return participants.join(
        rider_scores,
        on = "name",
        how = "left"
    ).sort("score", descending=True).with_row_index("predicted_result", offset=1)

def get_won_how_race_strength_counts(
    results: pl.DataFrame,
    races: pl.DataFrame,
):
    #for each race: windowed count of top-x within race-feat bounds and won-how
    #for new predictions: windowed count of top -x within race feat bounds and any won-how

    #TODO: somehow also filter on race similarity, not just won_how
    #Or adding a weight on race similarity between aggregated races and current race would also be fine
    #I dont see how, however, maybe using some categories again...
    #categories can work if they are overlapping, but can get messy


    won_how_possibilities = races.unique("won_how_clean").select("won_how_clean").to_series().to_list()
    ranks = [3, 10, 25, 999]
    windows = ["1110", "370", "40"]
    results = results.join(
        races.select(["race_id", "won_how_clean"]),
        on="race_id",
        how="left"
    )
    results = results.with_columns(
        [
            pl.when((pl.col("rank") <= r) & (pl.col("won_how_clean") == wh))
            .then(1).otherwise(0).alias(f"top_{r}_{wh}_bool")
            for r in ranks
            for wh in won_how_possibilities
        ]
    )
    count_cols = [
        f"top_{r}_{wh}_bool" 
        for r in ranks 
        for wh in won_how_possibilities
    ]
    results_w_counts = windowed_aggregation(
        results=results,
        races=races,
        features_to_aggregate=[*count_cols],
        windowed_expressions_to_aggregate={
            w: [
                pl.col(col).sum().alias(f"nr_{col[:-5]}_{w}")
                for col in count_cols
            ]
            for w in windows
        }
    ).select(
        "name",
        "race_id",
        "won_how_clean",
        *[f"nr_{col[:-5]}_{w}" for col in count_cols for w in windows]
    )
    won_how_counts_select_expressions = []
    r_w_list = [(r, w) for r in ranks for w in windows]
    for (r, w) in r_w_list:
        expr = None
        for wh in won_how_possibilities:
            name = f"nr_top_{r}_{wh}_{w}"
            r_wh_w_count_col = pl.col(name)
            if expr is None:
                expr = (
                    pl.when(pl.col("won_how_clean") == wh)
                    .then(r_wh_w_count_col)
                )
            else:
                expr = (
                    expr.when(pl.col("won_how_clean") == wh)
                        .then(r_wh_w_count_col)
                )
        won_how_counts_select_expressions.append(
            expr.alias(f"same_won_how_has_top_{r}_{w}")
        )
    results_w_won_how_counts = results_w_counts.with_columns(
        won_how_counts_select_expressions
    ).select(
        "race_id",
        "name",
        *[f"same_won_how_has_top_{r}_{w}"
         for r in ranks
         for w in windows]
    )
    #     [
    #         pl.when(pl.col("won_how_clean") == wh).then(
    #             pl.col("nr_top_{r}_{wh}_{w}").sum()
    #         ).when(pl.col("won_how_clean") == w)
    #         .then(1).otherwise(0).alias(f"has_top_{col[:-5]}_{w}")
    #         for col in count_cols
    #         for w in windows
    #     ]
    # )

    return results_w_won_how_counts

def get_overall_race_strength_counts(
    results: pl.DataFrame,
    races: pl.DataFrame,
) ->pl.DataFrame:
    
    ranks = [3, 10, 25]
    windows = ["1110", "370", "40"]
    results_w_counts = windowed_aggregation(
        results=results,
        races=races,
        features_to_aggregate=["rank"],
        windowed_expressions_to_aggregate={
            w: [
                (pl.when(
                    (pl.col("rank") <= r) 
                ).then(
                    1
                ).otherwise(0)).sum().alias(f"nr_top_{r}_{w}")
                for r in ranks
            ]
            for w in windows
        }
    )
    return results_w_counts.select(
        "name",
        "race_id",
        *[f"nr_top_{r}_{w}" for r in ranks for w in windows]
    )

def get_z_scores(
    results: pl.DataFrame,
    races: pl.DataFrame,
    rider_profiles: pl.DataFrame
) -> pl.DataFrame:
    dimensions = RACE_SIMILARITY_COLS #["profile_score", "startlist_score", "distance_km", "elevation_m"]
    ranks = [25, 10, 3]
    df = results.join(
        rider_profiles,
        on=["name", "race_id"],
        how="left"
    ).join(
        races.select(["race_id", *dimensions]),
        on = "race_id",
        how = "left"
    ).with_columns(
        ((pl.col(d) - pl.col(f"decay_weighted_{d}_{r}")) /
        pl.col(f"decay_weighted_{d}_{r}_variance").sqrt()).alias(f"z_score_{d}_{r}")
        for d in dimensions
        for r in ranks
    ).with_columns(
        [pl.sum_horizontal(
            [pl.col(f"z_score_{d}_{r}").pow(2) for d in dimensions]
        ).sqrt().alias(f"multivariate_z_score_{r}")
        for r in ranks]
    )
    df = df.select(
        ["race_id",
         "name",
         *[
             c
             for c in df.columns
             if "z_score" in c
         ]]
    )
    return df

# def create_result_features_pre_embed(results: pl.DataFrame, races: pl.DataFrame) -> pl.DataFrame:
    
#     races = races.with_columns(
#         pl.col("date").str.to_date()
#     )
#     races = races.with_columns(
#         pl.col("classification").cast(pl.Categorical),
#         pl.col("date").dt.month().alias("month")
#     )
    
#     discrete_race_features_to_bucket_on = [
#         "won_how_clean",
#     ]
#     continuous_race_features_to_bucket_on = [
#         "distance_km",
#         "elevation_m",
#         "profile_score",
#         "profile_score_last_25k",
#         "final_km_percentage",
#         # "classification",
#         "startlist_score",
#         # "team_rank"
#         # "month"
#         # "stage_or_one_day", #?
#     ]
#     continuous_rider_features_to_bucket_on = [
#         "team_rank",
#     ]
#     #TODO: use decay instead of hard windows -> group_by_dynamic not needed anymore and no weird discontinuities when window shifts
#     #TODO: 
#         # since features should represent type of rider, maybe go for average value of bucket-feature over best x races in last window, only coutning at least top-25.
#         # so strength is completly out of the picture
    
#      # - calculate exp-decay-weighted average for races with top-25, top-10 and top-3
#      # - so similiar to what we were doing before with hard windows and group_by_dynamic 
#     windows = {
#         "1110d": 1110,
#         # "370d": 370,
#         # "40d": 40,
#     }
#     rank_thresholds = [
#         25,
#         # 5,
#         3,
#         10,
#     ]
#     #create feature for every comb of window, win_count and race_features

#     #create buckets:
#     races_bucketed = races.with_columns(
#         [
#             pl.col(c).qcut(3, labels=["1", "2", "3"], allow_duplicates=True).alias(f"{c}_bucket").cast(pl.Categorical)
#             for c in continuous_race_features_to_bucket_on
#         ]
#     )
#     results_with_race_buckets = results.join(
#         races_bucketed,
#         on="race_id",
#         how="inner",
#     )
#     # Compute each rider position within their own team (relative value) for that race when team data is present.
#     # Keep this deterministic here so feature generation does not depend on a model trained later in the pipeline.
    
#     results_with_all_buckets = results_with_race_buckets.with_columns(
#         [
#             pl.col(c).qcut(3, labels=["1", "2", "3"], allow_duplicates=True).alias(f"{c}_bucket").cast(pl.Categorical)
#             for c in continuous_rider_features_to_bucket_on
#         ]
#     )

#     results_bucketed_window_dates = results_with_all_buckets.with_columns(
#         [
#             (pl.col("date") + pl.duration(days = (offset - 1))).alias(f"_window_date_{label}")
#             for label, offset in windows.items()
#         ]
#     ).drop(
#         [
#             c for c in results_with_all_buckets.columns
#             if c not in [
#                 "name",
#                 "date",
#                 "rank"
#             ] 
#             and not c.endswith("_bucket")
#             and not c.startswith("_window_date")
#             and c not in discrete_race_features_to_bucket_on
#         ]
#     )
#     results_bucketed_window_dates = results_bucketed_window_dates.sort("date")
#     lf = results_bucketed_window_dates.lazy()
    


#     continuous_race_bucket_top_x_expressions = [
#         (
#             ((pl.col("rank") <= rank_threshold) & (pl.col(f"{bucket_feature}_bucket") == bucket)).sum()
#             / (pl.col(f"{bucket_feature}_bucket") == bucket).sum()
#         )
#         .alias(f"top_{rank_threshold}_in_{bucket_feature}_{bucket}")
#         for bucket_feature in continuous_race_features_to_bucket_on
#         for bucket in ["1", "2", "3"]
#         for rank_threshold in rank_thresholds
#     ]
#     continuous_rider_bucket_top_x_expressions = [
#         (
#             ((pl.col("rank") <= rank_threshold) & (pl.col(f"{bucket_feature}_bucket") == bucket)).sum()
#             / (pl.col(f"{bucket_feature}_bucket") == bucket).sum()
#         )
#         .alias(f"top_{rank_threshold}_in_{bucket_feature}_{bucket}")
#         for bucket_feature in continuous_rider_features_to_bucket_on
#         for bucket in ["1", "2", "3"]
#         for rank_threshold in rank_thresholds
#     ]
#     discrete_race_bucket_top_x_expressions = [
#         (
#             ((pl.col("rank") <= rank_threshold) & (pl.col(bucket_feature) == bucket)).sum()
#             / (pl.col(bucket_feature) == bucket).sum()
#         ).alias(f"top_{rank_threshold}_in_{bucket_feature}_{bucket}")
#         for bucket_feature in discrete_race_features_to_bucket_on
#         for bucket in races.select(bucket_feature).unique().to_series().to_list()
#         for rank_threshold in rank_thresholds
#     ]

#     results_pre_embed_features = results.join(
#         races.select(["race_id", "date"]),
#         on = "race_id",
#         how = "left"
#     ).select(
#         "name",
#         "date",
#         "year"
#     ).filter(pl.col("date").is_not_null()).unique().with_columns(
#         (pl.col("date") - pl.duration(days=1)).alias("feature_date")
#     )
#     for label, _ in windows.items():
#         window_lf = lf.group_by_dynamic(
#                 index_column=f"_window_date_{label}",
#                 period=label,
#                 offset= f"-{label}",
#                 every="1d",
#                 group_by="name",
#                 start_by="window",
#             ).agg(
#                 [
#                 *continuous_race_bucket_top_x_expressions,
#                 *continuous_rider_bucket_top_x_expressions,
#                 *discrete_race_bucket_top_x_expressions
#                 ]
#             ).rename({
#                 f"_window_date_{label}": "date"
#             }).collect(engine="streaming")
#         window_lf = window_lf.rename({
#             col: f"f{col}_{label}"
#             for col in window_lf.columns
#             if col not in ["name", "date"]
#         }).rename({
#             "date": "feature_date"
#         })
#         results_pre_embed_features = results_pre_embed_features.join(
#             window_lf,
#             on=["name", "feature_date"],
#             how="left"
#         ).fill_nan(0).fill_null(0)
#         #TODO: nans arent the same as null occurences. neeeds to be changed. 

#     return results_pre_embed_features.drop("feature_date")


# def create_result_embeddings(pre_embed_features: pl.DataFrame, races_df: pl.DataFrame, results_df: pl.DataFrame) -> pl.DataFrame:
#     #create embeddings from pre-embed features
#     feature_cols = [
#         col for col in pre_embed_features.columns
#         if col not in ["name", "date", "year"]
#     ]
#     pre_embed_array = pre_embed_features.select(feature_cols).to_numpy()
#     pre_embed_array = StandardScaler().fit_transform(pre_embed_array)

#     pca = PCA(n_components=EMBEDDING_SIZE)
#     embedded_array = pca.fit_transform(pre_embed_array)

#     #INSPECT WITH
#     # - pca.explained_variance_
#     # - np.cumsum(pca.explained_variance_ratio_)
    
#     embedded_df = pl.DataFrame(embedded_array, schema=[f"embed_{i+1}" for i in range(embedded_array.shape[1])])
#     results_embedded = pl.concat([
#         pre_embed_features.select(["name", "date"]),
#         embedded_df
#     ], how="horizontal")
    
#     results_embedded_df = results_embedded.join(
#         results_df.select(["name", "race_id"]).join(
#             races_df.select("race_id", pl.col("date").str.to_date()),
#             on = "race_id",
#             how="left"
#         ),
#         on = ["name", "date"],
#         how = "inner"
#     )
#     return results_embedded_df


# def create_races_inference_embeddings(races_df: pl.DataFrame, base_embeddings_df: pl.DataFrame) -> pl.DataFrame:
#     #create embeddings based on previous races, like it will be done for predicting


#     duplicate_races = races_df.select("race_id", "name", "year", *RACE_SIMILARITY_COLS).join(
#         races_df.select("race_id", "name", "year", *RACE_SIMILARITY_COLS),
#         on = ["name"]
#     ).filter(
#         pl.col("year").cast(pl.Int64) <= pl.col("year_right").cast(pl.Int64) + 4
#     ).filter(
#         pl.col("year").cast(pl.Int64) >= pl.col("year_right").cast(pl.Int64)
#     ).filter(
#         ~ (pl.col("race_id") == pl.col("race_id_right"))
#     )

#     #TODO use Mahalanobis distance instead of euclidean and normalize features properly (also on the

#     duplicate_races_normalized = duplicate_races.with_columns(
#         *[
#             (
#                 (pl.col(c) - pl.col(f"{c}_right").min()) 
#                 / (pl.col(f"{c}_right").max() - pl.col(f"{c}_right").min())
#             ).over("name").alias(f"{c}_normalised")
#             for c in RACE_SIMILARITY_COLS
#         ],
#         *[
#             (
#                 (pl.col(f"{c}_right") - pl.col(f"{c}_right").min()) 
#                 / (pl.col(f"{c}_right").max() - pl.col(f"{c}_right").min())
#             ).over("name").alias(f"{c}_normalised_right")
#             for c in RACE_SIMILARITY_COLS
#         ]
#     )
#     duplicate_races_distance_part = duplicate_races_normalized.with_columns([
#         (
#             (pl.col(f"{c}_normalised") - pl.col(f"{c}_normalised_right")) ** 2
#         ).alias(f"{c}_distance")
#         for c in RACE_SIMILARITY_COLS
#     ])
#     duplicate_races_distance = duplicate_races_distance_part.with_columns(
#         (pl.sum_horizontal([
#             pl.col(f"{c}_distance") 
#             for c in RACE_SIMILARITY_COLS
#         ])
#         ).sqrt().alias("total_distance")
#     )
#     closest_race_ids = duplicate_races_distance.sort(
#         "total_distance", descending=False
#     ).group_by("race_id").head(3).select("race_id", "race_id_right")
#     closest_embedding = closest_race_ids.join(
#         base_embeddings_df.rename({"race_id": "race_id_right"}),
#         on="race_id_right",
#         how="left"
#     ).group_by(
#         "race_id",
#     ).agg(
#         *[
#             pl.col(c).mean()
#             for c in base_embeddings_df.columns if c != "race_id"
#         ]
#     )
#     return closest_embedding

# def create_races_base_embeddings(results_df, results_embedded_df: pl.DataFrame) -> pl.DataFrame:
#     df = results_embedded_df.join(
#         results_df.select(["name", "race_id", "rank"]).filter(pl.col("rank") < 25),
#         on = ["name", "race_id"],
#         how = "left"
#     )
#     df = df.with_columns(
#         pl.when(pl.col("rank") < 3).then(8.3333)
#         .otherwise(pl.when(pl.col("rank") < 10).then(2.5)
#         .otherwise(1)).alias("rank_weight")
#     )
    
#     races_embeddings = df.group_by("race_id").agg(
#         [
#             (   
#                 (pl.col(f"embed_{i+1}") * pl.col("rank_weight")).sum() 
#                 / pl.col("rank_weight").sum()
#             ).alias(f"embed_{i+1}")
#             for i in range(len([c for c in results_embedded_df.columns if c.startswith("embed_")]))
#         ]
#     )
#     return races_embeddings

# def calculate_cosine_similarity_polars(df, n_dims=EMBEDDING_SIZE):
#     # Build expressions for dot product
#     left_cols = [pl.col(f"embed_{i}") for i in range(1, n_dims + 1)]
#     right_cols = [pl.col(f"embed_{i}_right") for i in range(1, n_dims + 1)]

#     # Cosine similarity
#     dot_product = pl.sum_horizontal([l * r for l, r in zip(left_cols, right_cols)])
#     norm1 = pl.sum_horizontal([c ** 2 for c in left_cols]).sqrt()
#     norm2 = pl.sum_horizontal([c ** 2 for c in right_cols]).sqrt()
#     cosine_sim = dot_product / (norm1 * norm2)
    

#     # L1 distance
#     l1_distance = pl.sum_horizontal([ (l - r).abs() for l, r in zip(left_cols, right_cols) ])

#     return df.with_columns([
#         cosine_sim.alias("cosine_similarity"),
#         l1_distance.alias("l1_distance")
#     ])

# def add_embedding_similarity_to_results_df(
#     results_embedded_df: pl.DataFrame,
#     races_embedded_df: pl.DataFrame
# ) -> pl.DataFrame:
    
#     results_w_race_embeddings = results_embedded_df.join(
#         races_embedded_df,
#         on = "race_id",
#         how= "left",
#     )
#     embedding_similary = calculate_cosine_similarity_polars(results_w_race_embeddings)

#     return results_embedded_df.join(
#         embedding_similary.select(["name", 'race_id', "cosine_similarity", "l1_distance"]),
#         on = ["name", 'race_id'],
#         how="left"
#     )

def windowed_aggregation(
    results,
    races,
    features_to_aggregate,
    windowed_expressions_to_aggregate,
    # rank_thresholds: list[int],
):
    """
    Docstring for windowed_aggregation
    
    :param results: Description
    :param races: Description
    :param features_to_aggregate: Description
    :param aggregations: Description
    :param rank_thresholds: Description
    :param windows: Description
    """

    results = results.join(
        races.select(["race_id", pl.col("date").str.to_date(), *[
            f 
            for f in features_to_aggregate
            if f in races.columns
        ]]),
        on = "race_id",
        how = "left"
    )
    bare_results = results.select([
        "name",
        "date",
        *features_to_aggregate
    ]).sort("date")
    lf = bare_results.lazy()

    for window, expressions_to_aggregate in windowed_expressions_to_aggregate.items():
        offset = int(window)
        label = f"{offset}d"
        _window_df = (
            lf.with_columns(
                (pl.col("date") + pl.duration(days = (offset - 1))).alias(f"_window_date_{label}")
            ).group_by_dynamic(
                index_column=f"_window_date_{label}",
                period=label,
                offset= f"-{label}",
                every="1d",
                group_by="name",
                start_by="window",
            )
            .agg(
                *expressions_to_aggregate
                # pl.len().alias(f"nr_races_participated_{label}"),
                # *[
                #     (pl.when(pl.col("rank") < rank_threshold).then(1)
                #         .otherwise(None)
                #     ).sum().alias(f"nr_top_{rank_threshold}_{label}")
                #     for rank_threshold in rank_thresholds
                # ],
            )
            .rename({f"_window_date_{label}": "date"})
            .collect(engine="streaming")
        )
        results = results.join(_window_df, on=["name", "date"], how="left").fill_nan(0).fill_null(0)
    return results

def exp_decay_weighted_average(
    results: pl.DataFrame,
    races: pl.DataFrame,
    features_to_calculate_average_for: List[str],
    # half_times_in_days: List[int],
    top_rank_thresholds: List[int],
) -> pl.DataFrame:
    ### calculates average and coefficient of vairation
    #coefficient of variation is independted to size
    #since distributions for top-3 can be very low for
    #riders
    half_time_in_days = 1110
    #order 
    df = results.join(
        races.select([
            "race_id", 
            pl.col("date").str.to_date(), 
            *[
                f for f in features_to_calculate_average_for
                if f in races.columns and f not in ["date", "race_id"]
            ]
        ]),
        on = "race_id",
        how = "left"
    ).sort(["name", "date"]).with_columns(
        # pl.col("date").cast(pl.Int64).alias("date_int"),
        pl.col("name").cast(pl.Categorical).alias("rider_id")
    )
    rider_ids = df["rider_id"].to_physical().cast(pl.Int32).to_numpy()
    dates_days = df.select(pl.col("date").dt.epoch("d")).to_numpy().flatten()

    values = df[[
        feature for feature in features_to_calculate_average_for
    ]].to_numpy()
    # values = df[feature].to_numpy()
    n_rows = len(values)
    n_features = len(features_to_calculate_average_for) 
    n_ranks = len(top_rank_thresholds)
    n_decays = len([half_time_in_days])

    #n x (features x rank_thresholds) 
    #cant create mask in one pass due to intermediate polars step that gets rank as duplicate column
    #due to double for loop
    filter_masks = df.with_columns(
        [
            ((pl.col("rank") <= rank_threshold) & (pl.col(feature).is_not_null()))
            # (pl.col(f"{rank_threshold}_mask") & (pl.col(feature).is_not_null()))
            .alias(f"filter_mask_top_{rank_threshold}_{feature}")#_{half_time}" )
            for rank_threshold in top_rank_thresholds
            for feature in features_to_calculate_average_for
            # for half_time in half_times_in_days
        ]
    ).select([
        f"filter_mask_top_{rank_threshold}_{feature}"#_{half_time}" 
        for rank_threshold in top_rank_thresholds 
        for feature in features_to_calculate_average_for 
        # for half_time in half_times_in_days
    ]).to_numpy()

    weighted_averages = np.full((n_rows, n_ranks * n_features *  n_decays), np.nan)
    weighted_variances = np.full((n_rows, n_ranks * n_features *  n_decays), np.nan)
    # decay_consts = np.array([np.log(2) / half_time for half_time in half_times_in_days])
    decay_const = np.array([np.log(2) / half_time_in_days])
    weighted_sums_squared = np.zeros(n_ranks * n_features *  n_decays)
    weighted_sums = np.zeros(n_ranks * n_features *  n_decays)
    weight_sums = np.zeros(n_ranks * n_features *  n_decays)
    prev_rider = rider_ids[0]
    prev_date = dates_days[0]
    for i in range(n_rows):
        rider = rider_ids[i]
        date = dates_days[i]

        #new rider, reset sums
        if rider != prev_rider:
            weighted_sums *= 0.0
            weight_sums *= 0.0
            prev_rider = rider
            prev_date = date
        
        delta_days = date - prev_date
        decay = np.exp(-decay_const * delta_days)

        #TODO: if any features are null
        # This influences mask, and may cause that the application needs to be switched around
        # Or the for loop to create the mask may be switched around
        # to line up both

        #DECAY APPLICATION
        # - feat 0 multiplied with decay 0
        # - feat 1 multiplied with decay 1
        # ...
        # - feat n_decay multiplied decay 0

        weighted_sums_squared *= np.tile(decay, n_features * n_ranks)

        weighted_sums *= np.tile(decay, n_features * n_ranks)

        weight_sums *= np.tile(decay, n_features * n_ranks)

        with np.errstate(divide='ignore', invalid='ignore'):
            weighted_averages[i] = np.where(weight_sums > 0, weighted_sums / weight_sums, np.nan)
            weighted_variances[i] = np.where(weight_sums > 0, (
                weighted_sums_squared / weight_sums) - (weighted_averages[i] ** 2
            ), np.nan)  
        
        #VALUE APPLICATION
        # - feat 0 receives values 0
        # - feat 1 receives value 0
        # ...
        # - feat n_decay - 1 receives values 0
        # - feat n_decay receives value 1 
        # - feat n_decay * n_features - 1 receives value n_features
        weighted_sums_squared += filter_masks[i] * (
            np.tile(
                np.repeat(
                    values[i] ** 2, n_decays
                ), 
            n_ranks)
        )
        weighted_sums += filter_masks[i] * (
            np.tile(
                np.repeat(
                    values[i], n_decays
                ), 
            n_ranks)
        )
        weight_sums += filter_masks[i] * np.ones(n_ranks * n_features * n_decays)
        prev_date = date
    
    df = df.with_columns([
        *[
            pl.Series(f"decay_weighted_{feature}_{rank_threshold}", weighted_averages[:, idx])
            for idx, (
                feature, rank_threshold#, half_time
            ) in enumerate(
                [(f, r)#, h) 
                for r in top_rank_thresholds 
                for f in features_to_calculate_average_for 
                # for h in half_times_in_days
                ]
            )
        ],
        *[
            pl.Series(f"decay_weighted_{feature}_{rank_threshold}_variance", weighted_variances[:, idx])
            for idx, (feature, rank_threshold#, half_time
            ) in enumerate(
                [(f, r)#, h) 
                for r in top_rank_thresholds 
                for f in features_to_calculate_average_for 
                # for h in half_times_in_days
                ]
            )
        ]]
    )
    return df.select(
        ["race_id", 
         "name",
         *[
             c
             for c in df.columns
             if "decay_weighted" in c
         ]]
    )

def calculate_variance_aware_center_point(
    rider_weighted_race_features: pl.DataFrame,
    races_df: pl.DataFrame,
    features_to_calculate_for: List[str],
    rank_threshold = 25,
    half_time_in_days = 700
): 
    """ Riders has columns for average and variance of features
    
    calculate center point of top riders for each race

    -> measure between actual race features and riders who perform good on it is posssible
    """
    df = rider_weighted_race_features.filter(pl.col("rank") < 10)
    df = df.with_columns(
        [*[
            ((pl.col(f"decay_weighted_{feature}_{rank_threshold}")
            / pl.col(f"decay_weighted_{feature}_{rank_threshold}_variance"))
            ).alias(f"average_over_variance_{feature}")
            for feature in features_to_calculate_for
        ],*[
            (1
            / pl.col(f"decay_weighted_{feature}_{rank_threshold}_variance")
            ).alias(f"inverse_variance_{feature}")
            for feature in features_to_calculate_for
        ]]
    )
    df = df.group_by(
        "race_id"
    ).agg(
        [*[pl.sum(pl.col("average_over_variance_{feature}")).alias("sum_average_over_variance_{feature}")
          for feature in features_to_calculate_for
        ],
        *[pl.sum(pl.col("inverse_variance_{feature}")).alias("sum_inverse_variance_{feature}")
          for feature in features_to_calculate_for
        ]]
    )
    df = df.with_columns(
        [*[
            (pl.col(f"sum_average_over_variance_{feature}") /
              pl.col(f"sum_inverse_variance_{feature}")
            ).alias(f"center_point_{feature}")
            for feature in features_to_calculate_for
        ],*[
            (1 / pl.col(f"sum_inverse_variance_{feature}")).alias(f"center_point_variance_{feature}")
            for feature in features_to_calculate_for
        ]]
    ).select([
        "race_id",
        *[f"center_point_{feature}" for feature in features_to_calculate_for],
        *[f"center_point_variance_{feature}" for feature in features_to_calculate_for]
    ])
    return df

def create_rider_profiles(results: pl.DataFrame, races:pl.DataFrame) -> pl.DataFrame:
    # half_times_in_days = [370, 40]
    rank_thresholds = [25, 10, 3]
    features = RACE_SIMILARITY_COLS
    rider_profiles = exp_decay_weighted_average(
        results = results,
        races=races,        
        features_to_calculate_average_for=features,
        # half_times_in_days=half_times_in_days,
        top_rank_thresholds=rank_thresholds
    )
    return rider_profiles

def create_result_features_table(
    rider_profiles: pl.DataFrame,
    results: pl.DataFrame, 
    races:pl.DataFrame
) -> pl.DataFrame:
    #TODO: create strength scores for multiple types of races

    # -------- overall individual strength features ----------
    #TODO: use decay instead of hard windows -> group_by_dynamic not needed anymore and no weird discontinuities
    # - can use weighted average for features like
    # - average top-3 startlist score
    # - average top-25 startlist score

        # decayed_sum = compute_decay(group_codes, days, values, tau)
    #TODO: best result for each window, is with hard window tho so not sure
    #TODO: counting the nr of races for each window might give indications on injuries, fatigue, etc. 
    # and is also easier to do with hard windows

    # windows = {
    #     "1110d": 1110,
    #     "370d": 370,
    #     "40d": 40,
    # }
    # rank_thresholds = [
    #     25,
    #     10,
    #     3
    # ]


    results = results.join(
        races.select(["race_id", pl.col("date").str.to_date(), "startlist_score"]),
        on = "race_id",
        how = "left"
    )

    # bare_results = results.select([
    #     "name",
    #     "date",
    #     "rank",
    #     # "points",
    #     "year"
    # ]).sort("date")
    # lf = bare_results.lazy()

    # for label, offset in windows.items():
    #     _window_df = (
    #         lf.with_columns(
    #             (pl.col("date") + pl.duration(days = (offset - 1))).alias(f"_window_date_{label}")
    #         ).group_by_dynamic(
    #             index_column=f"_window_date_{label}",
    #             period=label,
    #             offset= f"-{label}",
    #             every="1d",
    #             group_by="name",
    #             start_by="window",
    #         )
    #         .agg(
    #             pl.len().alias(f"nr_races_participated_{label}"),
    #             *[
    #                 (pl.when(pl.col("rank") < rank_threshold).then(1)
    #                     .otherwise(None)
    #                 ).sum().alias(f"nr_top_{rank_threshold}_{label}")
    #                 for rank_threshold in rank_thresholds
    #             ],
    #         )
    #         .rename({f"_window_date_{label}": "date"})
    #         .collect(engine="streaming")
    #     )
    #     results = results.join(_window_df, on=["name", "date"], how="left").fill_nan(0).fill_null(0)
    #     del _window_df
    #     gc.collect()
    
    #TODO: delta change of strength score can be important, rapidly improving riders outperform their current strength score 


    # ---------- similar race strength features ----------
    race_strength_counts = get_won_how_race_strength_counts(
        results = results,
        races = races
    )
    overall_strength_counts = get_overall_race_strength_counts(
        results=results,
        races=races
    )
    z_scores = get_z_scores(
        results=results,
        races=races,
        rider_profiles=rider_profiles
    )
                                                            
    #TODO: use similarity embeddings to create weighted strength features
    #TODO: difference between overall strength and similar-race strength apparantly can be good

    # ---------- other individual features ----------
    sorted_results = results.sort("date")
    #TODO: join with spine using left-joins later on

    #days since last race
    days_since_last_race = (
        sorted_results
        .with_columns(
            (pl.col("date") - pl.col("date").shift(1).over("name")).alias("days_since_last_race")
        )
    ).select(["name", "race_id", "days_since_last_race"])
    #days since last top-3 race placement
    temp = sorted_results.with_columns(
        pl.when(pl.col("rank") <= 3)
        .then(pl.col("date"))
        .otherwise(None)
        .alias("top3_date")
    )
    temp = temp.with_columns(
        pl.col("top3_date")
        .reverse()
        .forward_fill()
        .reverse()
        .shift(1)
        .over("name")
        .alias("prev_top3")
    )
    days_since_last_top3 = (
        temp.with_columns(
            (pl.col("date") - pl.col("prev_top3")).alias("days_since_last_top3")
        )
    ).select(["name", "race_id", "days_since_last_top3"])

    #percentage of races in which team_rank was 1
    # - count race nr in reverse, per rider
    # - use reverse race nr as exponential decay factor of .1
    # - sum decay factors where team_rank was 1 and divide by sum of all decay factors
    #TODO: also with numba for loop
    # - current group-by only calculates for the msot recent ace, needs to be calculated for all races
    # - group-by verions:
    # half_time_nr_races = 10
    # decay_rate = np.log(2) / half_time_nr_races 
    # decay_weights = sorted_results.with_columns(
    #         pl.col("date").sort_by("date", descending=True).rank("dense").over("name").alias("reverse_race_nr")
    #     ).with_columns(
    #         # (0.1 ** pl.col("reverse_race_nr")).alias("decay_weight")
    #         ((-decay_rate * pl.col("reverse_race_nr")).exp()).alias("decay_weight")
    #     )
    # percentage_kopman_race_decay = (
    #     decay_weights.with_columns(
    #         pl.when(pl.col("team_rank") == 1).then(pl.col("decay_weight")).otherwise(0).alias("kopman_decay")
    #     ).group_by("name").agg(
    #         (pl.col("kopman_decay").sum() / pl.col("decay_weight").sum()).alias("percentage_kopman_race_decay")
    #     )
    # ).select(["name", "race_id", "percentage_kopman_race_decay"])

    #percentage of races in which team_rank was 3 or lower
    # percentage_team_top3_race_decay = (
    #     decay_weights.with_columns(
    #         pl.when(pl.col("team_rank") <= 3).then(pl.col("decay_weight")).otherwise(0).alias("team_top3_decay")
    #     ).group_by("name").agg(
    #         (pl.col("team_top3_decay").sum() / pl.col("decay_weight").sum()).alias("percentage_team_top3_race_decay")
    #     )
    # ).select(["name", "race_id", "percentage_team_top3_race_decay"])

    #TODO: overall rank on world ranking

    #nr days since first race day of this rider this year 
    days_since_season_start = (
        sorted_results.with_columns(
            (pl.col("date") - pl.datetime(pl.col("year"), 1, 1)).alias("day_nr_that_year")
        ).with_columns(
            pl.min("day_nr_that_year").over(["name", "year"]).alias("first_day_of_season")
        ).with_columns(
            (pl.col("day_nr_that_year") - pl.col("first_day_of_season")).alias("days_since_season_start")
        )
    ).select(["name", "race_id", "days_since_season_start"])

    #nr race_days this season
    # - use ordinal grouped over name and year, so it automatically resets each season and counts
    race_days_this_season = (
        sorted_results.with_columns(
            (pl.col("date").sort_by("date").rank("dense").over(["name", "year"]) - 1).alias("race_days_this_season")
        )
    ).select(["name", "race_id", "race_days_this_season"])

    # -------- team features ----------
    #nr of top-3, top-10, top-25, of teammembers in windows

    # race_winner_team = sorted_results.join(
    #     sorted_results.filter(pl.col("rank") == 1)
    #     .select(["race_id", "team"]).rename({"team": "winning_team"}),
    #     on="race_id",
    #     how="left"
    # )
    # in_winning_team = race_winner_team.with_columns(
    #     (pl.col("team") == pl.col("winning_team")).alias("in_winning_team")
    # ).drop("winning_team")
    # features_to_aggregate = ["in_winning_team", "rank"]
    # expressions_to_aggregate = {
    #     w: [
    #         # pl.sum("in_winning_team").alias("nr_wins_in_last_year"),
    #         *[
    #             (pl.when(pl.col("rank") < r).then(1)
    #                 .otherwise(None)
    #             ).sum().alias(f"individual_nr_top_{r}_{w}")
    #             for r in ranks
    #         ],
    #     ]
    #     for w in windows#,
        # "1100": [
        # ]
    # }
    # df_nr_of_individual_wins = windowed_aggregation(
    #     results = in_winning_team,
    #     races=races,
    #     features_to_aggregate=features_to_aggregate,
    #     windowed_expressions_to_aggregate=expressions_to_aggregate,
    # )
    ranks = [3, 10, 25]
    windows = ["1110", "370", "40"]
    teams = sorted_results.select("race_id", "name", "team").unique()
    df = overall_strength_counts.join(
        race_strength_counts,
        on=["race_id", "name"],
        how="left"
    ).join(
        teams,
        on=["race_id", "name"],
        how="left"
    ) 
    df = df.group_by("race_id", "team").agg(
        *[
            pl.sum(f"nr_top_{r}_{w}").alias(f"team_nr_top_{r}_{w}")
            for r in ranks for w in windows
        ],
        *[
            pl.sum(f"same_won_how_has_top_{r}_{w}").alias(f"team_same_won_how_has_top_{r}_{w}")
            for r in ranks for w in windows
        ]
    ).select(
        ["race_id", "team", 
         *[f"team_nr_top_{r}_{w}" for r in ranks for w in windows]]
    )
    nr_of_team_results = teams.join(
        df,
        on=["race_id", "team"],
        how="left"
    ).select(
        ["race_id", "name", 
         *[f"team_nr_top_{r}_{w}" for r in ranks for w in windows]]
    )
    #nr of teammates
    df = sorted_results.group_by("race_id", "team").agg(
        pl.n_unique("name").alias("nr_teammates")
    ).select(["race_id", "team", "nr_teammates"])
    nr_of_teammates = results.join(
        df,
        on=["race_id", "team"],
        how="left"
    ).select(["race_id", "name", "nr_teammates"])

    #TODO: best teammate in world ranking
    #TODO: average teammate world ranking

    # -------- competition features ----------
    #TODO: density of sprinters? etc.
    #TODO: weather  

    #top 10 is 2.5 times harder than top 25, top 3 is 8.3 times harder than top 25, so we can weight them accordingly
    #but dont count doubles
    # results = results.with_columns(
    #     [(pl.col(f"nr_top25_{label}") + (2.5 - 1) * pl.col(f"nr_top10_{label}") + (8.33 - 1 - 1.5) * pl.col(f"nr_top3_{label}"))
    #     .alias(f"strength_{label}") for label in windows.keys()]
    # )


    # results = results.with_columns(
    #     pl.col("strength_370d").rank("dense", descending=True).over(["race_id", "team"]).alias("relative_team_strength_rank")
    # )

    #TODO: aggregation
    feats_to_aggregate = [
        nr_of_teammates,
        nr_of_team_results,
        race_days_this_season,
        days_since_season_start,
        days_since_last_top3,
        days_since_last_race,z_scores,
        overall_strength_counts,
        race_strength_counts
    ]
    #spine
    results_spine = results.select("race_id", "name")
    results_feats = results_spine
    for feat in feats_to_aggregate:
        results_feats = results_feats.join(
            feat,
            on = ["race_id", "name"],
            how = "left"
        )
    return results_feats

"""*****************************************************

MAIN Functions

*****************************************************"""

def prepare_test_data(
    source_dir: str = DEFAULT_DATA_DIR,
    target_dir: str = DEFAULT_TEST_DATA_DIR,
    max_races_per_year: int = 500,
    nr_years: int = 7,
) -> dict:
    """Old version: pick x races per year for y years"""
    ensure_data_dir(target_dir)
    races_df = pl.read_parquet(data_path(source_dir, "races_df.parquet"))
    results_df = pl.read_parquet(data_path(source_dir, "results_df.parquet"))
    riders_yearly_data = pl.read_parquet(data_path(source_dir, "rider_yearly_stats_df.parquet"))
    rider_stats_path = Path(data_path(source_dir, "rider_stats_df.parquet"))
    riders_personal_data = pl.read_parquet(str(rider_stats_path)) if rider_stats_path.exists() else pl.DataFrame(
        schema={
            "name": pl.Utf8,
            "age": pl.Float64,
            "height": pl.Float64,
            "weight": pl.Float64,
        }
    )

    available_years = (
        races_df.select(pl.col("year").cast(pl.Int64).alias("year"))
        .unique()
        .sort("year")
        .to_series()
        .to_list()
    )
    selected_years = available_years[-nr_years:]
    selected_years_str = [str(year) for year in selected_years]

    def filter_and_clean_races_piped(df: pl.DataFrame) -> pl.DataFrame:
        return filter_and_clean_races(df, results_df=results_df)

    selected_races = (
        races_df
        .filter(pl.col("year").is_in(selected_years_str))
        .filter(pl.col("stage").is_null())
        .pipe(filter_and_clean_races_piped)
        .sort(["year", "startlist_score"], descending=[False, True])
        .group_by("year")
        .head(max_races_per_year)
    )
    selected_results = results_df.join(
        selected_races.select("race_id"),
        on="race_id",
        how="inner"
    ).filter(pl.col("name").is_in([
        "mathieu-van-der-poel",
        "tadej-pogacar",
        "jonathan-milan",
        "mattias-skjelmose-jensen",
        "oliver-naesen",
        "victor-campenaerts",
    ]))
    selected_races = selected_races.join(
        selected_results.select("race_id").unique(),
        on="race_id",
        how="inner"
    )
    selected_riders = selected_results.select("name").unique()
    selected_rider_yearly_data = riders_yearly_data.join(
        selected_riders,
        on="name",
        how="inner"
    )
    selected_rider_personal_data = riders_personal_data.join(
        selected_riders,
        on="name",
        how="inner"
    ) if "name" in riders_personal_data.columns else pl.DataFrame(
        schema={
            "name": pl.Utf8,
            "age": pl.Float64,
            "height": pl.Float64,
            "weight": pl.Float64,
        }
    )

    selected_races.write_parquet(data_path(target_dir, "races_df.parquet"))
    selected_results.write_parquet(data_path(target_dir, "results_df.parquet"))
    selected_rider_yearly_data.write_parquet(data_path(target_dir, "rider_yearly_stats_df.parquet"))
    selected_rider_personal_data.write_parquet(data_path(target_dir, "rider_stats_df.parquet"))

    return {
        "years": selected_years,
        "nr_races": selected_races.height,
        "nr_results": selected_results.height,
        "nr_riders": selected_riders.height,
    }



def filter_data(
        races_df: pl.DataFrame, 
        results_df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Filters on
     -> top-tier classification (WorldTour, ProSeries, HC, WC, NC) for feature creation
     -> excludes races with profile_score > 2000 while keeping unknown profile scores
     -> startlist score > 200 for inference / prediction
     """

    profile_score = pl.col("profile_score").cast(pl.Float64, strict=False)
    elevation_m = pl.col("elevation_m").cast(pl.Float64, strict=False)
    profile_unknown = profile_score.is_null() | profile_score.is_nan()
    elevation_unknown = elevation_m.is_null() | elevation_m.is_nan()
    necessary_races = races_df.filter(
        pl.col("classification").is_in(TOP_TIER_CLASSIFICATIONS)
        & (
            profile_unknown
            | (profile_score <= MAX_PROFILE_SCORE)
        )
        & ~(profile_unknown & elevation_unknown)
    )
    necessary_results = results_df.join(
        necessary_races.select("race_id"),
        on = "race_id",
        how = "inner"
    )
    return necessary_races, necessary_results

def create_result_aggregations(
    normalized_races_df: pl.DataFrame, 
    necessary_results: pl.DataFrame,
    rider_profiles: pl.DataFrame,
    data_dir: str
) -> tuple[pl.DataFrame, pl.DataFrame]:

    years = normalized_races_df.select("year").unique().cast(pl.Int64).sort("year").to_series().to_list()
    necessary_results_with_years = necessary_results.join(
        normalized_races_df.select("race_id", "year"),
        on = "race_id",
        how = "left"
    )
    tmp_dir = Path(data_dir) / "_tmp_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    total_years = len(years) - 4
    for i in range(total_years):
        target_year = str(years[i+4])
        result_part_path = tmp_dir / f"result_features_{target_year}.parquet"
        # pre_embed_part_path = tmp_dir / f"pre_embed_features_{target_year}.parquet"
        if result_part_path.exists(): # and pre_embed_part_path.exists():
            print(f"Skipping year {target_year} ({i+1}/{total_years}) — part files already exist.")
            continue
        print(f"Processing year {target_year} ({i+1}/{total_years})...")
        allowed_years = [str(year) for year in years[i:i+5]]
        part_normalized_races = normalized_races_df.filter(pl.col("year").is_in(allowed_years))
        part_necessary_results  = necessary_results_with_years.filter(pl.col("year").is_in(allowed_years))

        #TODO: rename to rider_profiles
        # part_pre_embed_features_df = create_result_features_pre_embed(
        #     results=part_necessary_results, 
        #     races=normalized_races_df
        # ).filter(pl.col("year") == target_year)

        part_results_features_df = create_result_features_table(
            results=part_necessary_results, 
            races=part_normalized_races,
            rider_profiles = rider_profiles
        )
        part_results_features_df.join(
            part_normalized_races.select("race_id", "year", "won_how_clean"),
            on="race_id",
            how="left"
        ).filter(pl.col("year") == target_year)
        part_results_features_df.write_parquet(result_part_path)
        del part_results_features_df
        # part_pre_embed_features_df.write_parquet(pre_embed_part_path)
        # del part_pre_embed_features_df, 
        del part_normalized_races, part_necessary_results
        gc.collect()

    # Check all parts are present before proceeding to concat
    missing_years = [
        str(years[i+4]) for i in range(total_years)
        if not (tmp_dir / f"result_features_{str(years[i+4])}.parquet").exists()
        # or not (tmp_dir / f"pre_embed_features_{str(years[i+4])}.parquet").exists()
    ]
    if missing_years:
        print(f"\nIncomplete run — missing parts for years: {missing_years}")
        print("Re-run the script to resume from where it stopped.")
        raise RuntimeError("Missing part files for years: " + ", ".join(missing_years))

    # Free large frames that are no longer needed before the heavy concat step
    del necessary_results_with_years
    gc.collect()

    result_part_files = sorted(tmp_dir.glob("result_features_*.parquet"))
    # pre_embed_part_files = sorted(tmp_dir.glob("pre_embed_features_*.parquet"))
    print(f"Concatenating {len(result_part_files)} result feature DataFrames")
    # print(f"Concatenating {len(pre_embed_part_files)} pre-embed feature DataFrames")
    # Use lazy scan so Polars streams parts instead of materialising all at once
    result_features_df = pl.scan_parquet(tmp_dir / "result_features_*.parquet").collect(engine="streaming").unique(subset=["name", "race_id"])
    # pre_embed_features_df = pl.scan_parquet(tmp_dir / "pre_embed_features_*.parquet").collect(engine="streaming").unique(subset=["name", "date", "year"])
    for f in result_part_files:# + pre_embed_part_files:
        f.unlink()
    tmp_dir.rmdir()
    return result_features_df#, pre_embed_features_df

def results_add_team_rank(results_df: pl.DataFrame) -> pl.DataFrame:
    return results_df.with_columns(
        pl.when(
            pl.col("team").is_not_null()
        )
        .then(
            pl.col("rank").rank("ordinal").over(["race_id", "team"])
            / pl.col("team").len().over(["race_id", "team"])
        )
        .otherwise(None)
        .alias("team_rank")
    )

def main(data_dir: str = DEFAULT_DATA_DIR):
    """create features dataframe"""
    pl.Config.set_tbl_cols(-1)

    results_df = pl.read_parquet(data_path(data_dir, "results_df.parquet")).filter(pl.col("rank") != -1)#filter out DNF, DNS, OTL
    races_df = pl.read_parquet(data_path(data_dir, "races_df.parquet"))
    # gc_results_df = pl.read_parquet(data_path(data_dir, "gc_results_df.parquet"))

    necessary_races, necessary_results = filter_data(races_df, results_df)

    normalized_races_df = normalize_race_data(races_df=necessary_races)

    normalized_races_df.write_parquet(data_path(data_dir, "normalized_races_df.parquet"))
    necessary_results = results_add_team_rank(necessary_results)
    # gc_results_df = gc_results_df.join(
    #     normalized_races_df.select([
    #         "race_id", 
    #         "year", 
    #         pl.col("name").alias("race_name"), 
    #         pl.col("stage").cast(pl.Int64).alias("stage_nr")
    #     ]),
    #     on = ["year", "race_name", "stage_nr"],
    #     how = "left"
    # ).select("race_id", "gc_rank", "time_diff", pl.col("rider_name").alias("name"))
    # necessary_results = necessary_results.join(
    #     gc_results_df,
    #     on = ["race_id", "name"],
    #     how = "left"
    # )
    rider_profiles = create_rider_profiles(necessary_results, normalized_races_df)

    # result_features_df = create_result_features_table(
    #     results=necessary_results, 
    #     races=normalized_races_df,
    #     rider_profiles = rider_profiles
    # )
    result_features_df = create_result_aggregations(
        normalized_races_df=normalized_races_df, 
        necessary_results=necessary_results,
        rider_profiles = rider_profiles,
        data_dir=data_dir   
    )


    # results_embedded_df = create_result_embeddings(
    #     pre_embed_features = pre_embed_features_df, 
    #     races_df=normalized_races_df, 
    #     results_df=necessary_results
    # )
    # races_base_embedded_df = create_races_base_embeddings(
    #     results_df = necessary_results,
    #     results_embedded_df = results_embedded_df
    # )
    # races_inference_embedded_df = create_races_inference_embeddings(
    #     races_df = normalized_races_df,
    #     base_embeddings_df= races_base_embedded_df
    # )

    # results_embedded_df = add_embedding_similarity_to_results_df(
    #     results_embedded_df=results_embedded_df,
    #     races_embedded_df=races_inference_embedded_df,
    # )
    print(result_features_df.columns)
    print(len(result_features_df.columns))
    print(result_features_df)
    
    # pre_embed_features_df.write_parquet(data_path(data_dir, "pre_embed_features_df.parquet"))
    result_features_df.write_parquet(data_path(data_dir, "result_features_df.parquet"))

    # results_embedded_df.write_parquet(data_path(data_dir, "results_embedded_df.parquet"))
    # races_base_embedded_df.write_parquet(data_path(data_dir, "races_base_embedded_df.parquet"))
    # races_inference_embedded_df.write_parquet(data_path(data_dir, "races_inference_embedded_df.parquet"))
    return {
        "result_features_rows": result_features_df.height,
        # "pre_embed_rows": pre_embed_features_df.height,
        # "results_embedded_rows": results_embedded_df.height,
    }


def main_test(
    source_dir: str = DEFAULT_DATA_DIR,
    target_dir: str = DEFAULT_TEST_DATA_DIR,
    max_races_per_year: int = 20,
    nr_years: int = 7,
):
    ensure_data_dir(target_dir)
    subset_summary = prepare_test_data(
        source_dir=source_dir,
        target_dir=target_dir,
        max_races_per_year=max_races_per_year,
        nr_years=nr_years,
    )
    feature_summary = main(data_dir=target_dir)
    return {
        "subset": subset_summary,
        "features": feature_summary,
        "data_dir": target_dir,
    }

# def assert_embedding_dimension():
#     pre_embed_features_df = pl.read_parquet("data_v2/pre_embed_features_df.parquet")

#     results_df = pl.read_parquet("data_v2/results_df.parquet").filter(pl.col("rank") != -1)#filter out DNF, DNS, OTL
#     races_df = pl.read_parquet("data_v2/races_df.parquet")
#     results_embedded_df = create_result_embeddings(pre_embed_features = pre_embed_features_df, races_df=races_df, results_df=results_df)
#     breakpoint()

# def assert_embedding_similarity():
#     results_embedded_df = pl.read_parquet("data_v2/results_embedded_df.parquet")
#     races_embedded_df = pl.read_parquet("data_v2/races_inference_embedded_df.parquet")
    # results_embedded_with_similarity = add_embedding_similarity_to_results_df(
    #     results_embedded_df=results_embedded_df,
    #     races_embedded_df=races_embedded_df,
    # )
    # breakpoint()


if __name__ == "__main__":
    # result = main_test()
    result = main()
    # if result is None:
    #     print("Run the script again to continue from the last completed year.")
    # # assert_embedding_dimension()
    # # assert_embedding_similarity()
    # pass