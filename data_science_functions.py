import polars as pl
from datetime import datetime
import re
import asyncio
import rbo
import hashlib
import itertools
from pprint import pprint
from typing import List, Dict

from sklearn.discriminant_analysis import StandardScaler

""" Similarity based on type of race

-> average global scores of riders to determine for a race
    to obtain race type scores e.g.
     RVV 2016 ->
        - one_day_races = 2115
        - GC = 992
        - TT = 683
        - Sprint = 910
        - Climber = 472
        - Hills = 1099
-> sum race type scores * rank over 3y, 1y and 6 weeks (peak performance tracking) 
    to obtain rider type scores e.g.
     Peter Sagan (winner of RVV 2016) ->
        3 Years ---------------------------
            - one_day_races = 9.68 m
            - GC = 8.8 m
            - TT = 4.42 m
            - Sprint = 4.00 m
            - Climber = 6.49 m
            - Hills = 6.32 m
        1 Year ---------------------------
            - one_day_races = 4.155 m
            - GC = 3.69 m
            - TT = 1.95 m 
            - Sprint = 1.66 m
            - Climber = 2.66 m
            - Hills = 2.62 m
        6 Weeks ---------------------------
            - one_day_races = 247 k
            - GC = 115 k
            - TT = 80 k
            - Sprint = 109 k
            - Climber = 53 k
            - Hills = 127 k
"""

def find_most_similar_races(normalised_upcoming_race_df: pl.DataFrame, normalized_races_df: pl.DataFrame, k = 5) -> pl.DataFrame:
    """
    Find knn neighbors based on race info:
    """
    if k == -1:
        k = normalized_races_df.height
    
    knn_races = normalized_races_df.with_columns([
        (
            (pl.col(c) - normalised_upcoming_race_df[c]) ** 2
        ).alias(c)
        for c in RACE_SIMILARITY_COLS
    ])
    knn_races = knn_races.with_columns([
        (pl.sum_horizontal(
            pl.col([c for c in RACE_SIMILARITY_COLS])
        ) #/ len(RACE_SIMILARITY_COLS) #TODO: not needed? right? check formula
        ).sqrt().alias("knn_distance"),
    ]).sort("knn_distance").head(k)
    return knn_races


def normalize_race_data(races_df: pl.DataFrame) -> pl.DataFrame:

    return races_df.with_columns([
        (pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min())
        for c in RACE_SIMILARITY_COLS
    ])


# def create_results_similarity(races_df: pl.DataFrame) -> pl.DataFrame:
#     top25 = (
#         races_df.sort("rank")
#         .group_by("race_id")
#         .agg(
#             pl.col("name").filter(pl.col("rank") < 25).alias("top_25_names")
#         )
#     )
#     # print(top25)

#     df_pandas = top25.to_pandas() #TODO: use polars
     
#     # Get all unique race_ids and their corresponding lists
#     races = df_pandas["race_id"].tolist()
#     lists = df_pandas["top_25_names"].tolist()

#     # Calculate pairwise combinations
#     pairs = list(itertools.combinations(range(len(races)), 2))
#     # len(pairs) = len(races) * (len(races)-1) / 2

#     # print(len(pairs))
#     # print(len(races))
#     # Calculate RBO scores for all pairs
#     results = [0] * len(pairs) * 2
#     # print(len(results))
    
#     for index, (i, j) in enumerate(pairs):
#         # print( index * 2)
#         race_i = races[i]
#         race_j = races[j]
#         list_i = lists[i]
#         list_j = lists[j]

#             # Handle empty lists
#         # if list_i is None or list_j is None:
#         #     rbo_score = 0.0
#         #     print(f"empty list for {race_i} vs {race_j}")
#         rbo_score = rbo.RankingSimilarity(list_i, list_j).rbo(p=0.9) #TODO: check out the p value
#         results[index * 2] = { #pair (a, b)
#             'race_id_1': race_i,
#             'race_id_2': race_j,
#             'rbo_score': rbo_score,
#             'list_1_length': len(list_i) if list_i is not None else 0,
#             'list_2_length': len(list_j) if list_j is not None else 0
#         }
#         results[index * 2 + 1] = { #pair (b, a)
#             'race_id_1': race_j,
#             'race_id_2': race_i,
#             'rbo_score': rbo_score,
#             'list_1_length': len(list_j) if list_j is not None else 0,
#             'list_2_length': len(list_i) if list_i is not None else 0
#         }
#     assert not any([r == 0 for r in results])

#     # print(results)
#     return pl.DataFrame(results)




def interpolate_profile_scores(races_df: pl.DataFrame) -> pl.DataFrame:
    #interpolates missing profile scores based on weighted average using results-similarity as weight

    #do this before normalization
    print("nr races before filtering: ", races_df.height)
    races_df = (races_df
        .filter(pl.col("profile_score") != -1)
        .filter(pl.col("profile_score_last_25k") != -1)
        .filter(pl.col("final_km_percentage") != -1)
    )   
    print("nr races after filtering: ", races_df.height)
    return races_df

    # profile_score_races = races_df.filter((pl.col("profile_score") > 0) & (pl.col("profile_score_last_25k") > 0))

    # knn = 10
    # results_similarity_with_profile = (results_similarity
    #     .join(
    #         profile_score_races.select(
    #             [
    #                 pl.col("race_id").alias("race_id_2"), 
    #                 pl.col("profile_score").alias("profile_score_2"), 
    #                 pl.col("profile_score_last_25k").alias("profile_score_last_25k_2"), 
    #             ]
    #         ),
    #         on = "race_id_2",
    #         how = "inner"
    #     )
    #     .sort("rbo_score", descending=True)
    #     .group_by("race_id_1")
    #     .head(knn)
    #     .with_columns(
    #         pl.col("rbo_score").sum().over("race_id_1").alias("rbo_score_sum")
    #     )
    # )
    # interpolated_profile_scores = results_similarity_with_profile.group_by("race_id_1").agg(
    #     ((pl.col("rbo_score") / pl.col("rbo_score_sum")) * pl.col("profile_score_2"))
    #     .sum().alias("interpolated_profile_score"),
    #     ((pl.col("rbo_score") / pl.col("rbo_score_sum")) * pl.col("profile_score_last_25k_2"))
    #     .sum().alias("interpolated_profile_score_last_25k")
    # )

    # interpolated_races = races_df.join(
    #     interpolated_profile_scores.select([
    #         pl.col("race_id_1").alias("race_id"),
    #         "interpolated_profile_score",
    #         "interpolated_profile_score_last_25k"
    #     ]),
    #     on = "race_id",
    #     how = "left"
    # ).with_columns([
    #     pl.when(pl.col("profile_score") == -1)
    #     .then(pl.col("interpolated_profile_score"))
    #     .otherwise(pl.col("profile_score"))
    #     .alias("profile_score"),
    #     pl.when(pl.col("profile_score_last_25k") == -1)
    #     .then(pl.col("interpolated_profile_score_last_25k"))
    #     .otherwise(pl.col("profile_score_last_25k"))
    #     .alias("profile_score_last_25k"),
    # ]).drop("interpolated_profile_score", "interpolated_profile_score_last_25k")

    # interpolated_races = interpolated_races.filter(
    #     (pl.col("profile_score") > 0) & (pl.col("profile_score_last_25k") > 0)
    # ) #filter out if still missing profile scores

    # return interpolated_races

import numpy as np

def scores_to_probability_results(rider_scores: pl.DataFrame) -> pl.DataFrame:
    """convert rider scores to probability of getting each result for each rider

    rider_scores should have a col "score"
    
    """
    def compute_plackett_luce_probs(scores: List[float], max_rank_to_predict: int = 10) -> Dict[int, List[float]]:
        # edge cases
        n = len(scores)
        if n == 0:
            return []
        total_sum = sum([
            s if s > 0 else 0
            for s in scores
        ])
        # if total_sum == 0:
        #     return [[0.0] * n for _ in range(max_rank_to_predict)]
        # scores = np.array(scores)
        # weights = np.exp(scores)
        # current_probs = weights / weights.sum()
        current_probs = [
            s / total_sum if s > 0 else 0 
            for s in scores
        ]
        rank_probs = {1: current_probs.copy()}
        for pos in range(2, max_rank_to_predict + 1):
            new_probs = [0.0] * n
            for i in range(n):
                if scores[i] == 0:
                    continue
                for j in range(n):
                    if j != i:
                        denom = total_sum - scores[j]
                        if denom > 0:
                            new_probs[i] += current_probs[j] * (scores[i] / denom)
            rank_probs[pos] = new_probs
            current_probs = new_probs
        return rank_probs

    # Usage
    scores = rider_scores["score"].to_list()
    rank_probs = compute_plackett_luce_probs(scores)
    for k in range(1, len(rank_probs) + 1):
        rider_scores = rider_scores.with_columns(
            pl.Series(f"rank_{k}_prob", rank_probs[k])
        )

    return rider_scores

def scores_to_results(rider_scores: pl.DataFrame, participants: pl.DataFrame, race: pl.DataFrame) -> pl.DataFrame:
    #convert rider scores to results dataframe format
    return participants.join(
        rider_scores,
        on = "name",
        how = "left"
    ).sort("score", descending=True).with_row_index("predicted_result", offset=1)

RACE_SIMILARITY_COLS = [
    "distance_km", 
    "elevation_m", 
    "profile_score",
    "profile_score_last_25k",
    "final_km_percentage"
]
def get_closest_5_races_to_stage_race(races: pl.DataFrame) -> pl.DataFrame:
    """
    closest 5 races in the same stage-race over last 3 years

    """
    duplicate_races = races.select("race_id", "name", "year", *RACE_SIMILARITY_COLS).join(
        races.select("race_id", "name", "year", *RACE_SIMILARITY_COLS),
        on = ["name"]
    ).filter(
        pl.col("year").cast(pl.Int64) <= pl.col("year_right").cast(pl.Int64) + 5
    ).filter(
        pl.col("year").cast(pl.Int64) >= pl.col("year_right").cast(pl.Int64)
    ).filter(
        ~ (pl.col("race_id") == pl.col("race_id_right"))
    )
    #TODO years should be replaced by dates, so only earlier stages are taken
    # not critical tho
    duplicate_races_normalized = duplicate_races.with_columns(
        *[
            (
                (pl.col(c) - pl.col(f"{c}_right").min()) 
                / (pl.col(f"{c}_right").max() - pl.col(f"{c}_right").min())
            ).over("name").alias(f"{c}_normalised")
            for c in RACE_SIMILARITY_COLS
        ],
        *[
            (
                (pl.col(f"{c}_right") - pl.col(f"{c}_right").min()) 
                / (pl.col(f"{c}_right").max() - pl.col(f"{c}_right").min())
            ).over("name").alias(f"{c}_normalised_right")
            for c in RACE_SIMILARITY_COLS
        ]
    )
    duplicate_races_distance_part = duplicate_races_normalized.with_columns([
        (
            (pl.col(f"{c}_normalised") - pl.col(f"{c}_normalised_right")) ** 2
        ).alias(f"{c}_distance")
        for c in RACE_SIMILARITY_COLS
    ])
    duplicate_races_distance = duplicate_races_distance_part.with_columns(
        (pl.sum_horizontal([
            pl.col(f"{c}_distance") 
            for c in RACE_SIMILARITY_COLS
        ])
        ).sqrt().alias("total_distance")
    )
    closest_races_distance = duplicate_races_distance.sort(
        "total_distance", descending=False
    ).group_by("race_id").head(5).select("race_id", "race_id_right", "total_distance")

    return closest_races_distance


# def create_race_features_table(races: pl.DataFrame, results: pl.DataFrame, riders: pl.DataFrame) -> pl.DataFrame:
#     riders = riders.select([
#         "name",
#         "Onedayraces",
#         "GC",
#         "TT",
#         "Sprint",
#         "Climber",
#         "Hills",
#     ])
#     top_results = results.select(["name", "race_id", "rank"]).join(
#         riders,
#         on="name",
#     ).filter(pl.col("rank") < 25)
#     race_avg_rider_stats = top_results.group_by("race_id").agg([
#         pl.col("Onedayraces").mean().alias("avg_Onedayraces"),
#         pl.col("GC").mean().alias("avg_GC"),
#         pl.col("TT").mean().alias("avg_TT"),
#         pl.col("Sprint").mean().alias("avg_Sprint"),
#         pl.col("Climber").mean().alias("avg_Climber"),
#         pl.col("Hills").mean().alias("avg_Hills"),
#     ])
#     races = races.with_columns(
#         pl.col("date").str.to_date()
#     )
#     race_avg_rider_stats = race_avg_rider_stats.join(
#         races.select([
#             pl.col("name").alias("race_name"),
#             "race_id",
#             "date",
#             "classification"
#         ]),
#         on = "race_id"
#     )
#     """*****************************************************" 
    
#     Average rider stats over last 3 years for ONE DAY RACES

#     *****************************************************"""
#     race_avg_rider_stats_over_3y_ONE_DAY_RACES = race_avg_rider_stats.filter(
#         ~ pl.col("classification").is_in(["2.Pro", "2.UWT"])
#     ).with_columns(
#         (pl.col("date") + pl.duration(days = 1109)).alias("_window_date")
#     ).sort("_window_date").group_by_dynamic(
#         index_column="_window_date",
#         every="1d",
#         period="1110d",
#         offset="-1111d",
#         label="left",
#         group_by="race_name",
#         # closed="left", #Cannot used closed when doing inference
#         start_by="window"
#     ).agg(
#         pl.col("avg_Onedayraces").mean().alias("avg_Onedayraces"),
#         pl.col("avg_GC").mean().alias("avg_GC"),
#         pl.col("avg_TT").mean().alias("avg_TT"),
#         pl.col("avg_Sprint").mean().alias("avg_Sprint"),
#         pl.col("avg_Climber").mean().alias("avg_Climber"),
#         pl.col("avg_Hills").mean().alias("avg_Hills"),
#     )
#     """*****************************************************" 
    
#     5 closest races in terms of kms, profile score, elevation, profile last 25k
#     Within the same stage-race last 3 years. FOR STAGE RACES
    
#     *****************************************************"""
#     closest_stage_races = get_closest_5_races_to_stage_race(
#         races=races.filter(pl.col("classification").is_in(["2.Pro", "2.UWT"]))
#     )
#     race_avg_rider_stats_closest_stages = closest_stage_races.join(
#         race_avg_rider_stats.rename(
#             {"race_id": "race_id_right"}
#         ),
#         on = "race_id_right",
#         how="left"
#     )
#     race_avg_rider_stats_over_3y_STAGES = race_avg_rider_stats_closest_stages.group_by(
#         "race_id"
#     ).agg(
#         pl.col("avg_Onedayraces").mean().alias("avg_Onedayraces"),
#         pl.col("avg_GC").mean().alias("avg_GC"),
#         pl.col("avg_TT").mean().alias("avg_TT"),
#         pl.col("avg_Sprint").mean().alias("avg_Sprint"),
#         pl.col("avg_Climber").mean().alias("avg_Climber"),
#         pl.col("avg_Hills").mean().alias("avg_Hills"),
#     )

#     races_STAGES = races.join(
#         race_avg_rider_stats_over_3y_STAGES,
#         on = ["race_id"],
#         how="inner"
#     )
    
#     races_ONE_DAY_RACES = races.join(
#         race_avg_rider_stats_over_3y_ONE_DAY_RACES.rename({
#             "race_name": "name",
#             "_window_date": "date"
#         }),
#         on = ["name", "date"],
#         how="inner"
#     )
#     races_features = pl.concat([races_STAGES, races_ONE_DAY_RACES]).unique()
#     return races_features

def create_result_features_pre_embed(results: pl.DataFrame, races: pl.DataFrame) -> pl.DataFrame:
    
    races = races.with_columns(
        pl.col("date").str.to_date()
    )
    races = races.with_columns(
        pl.col("classification").cast(pl.Categorical),
        pl.col("date").dt.month().alias("month")
    )
    race_features_to_bucket_on = [
        "distance_km",
        "elevation_m",
        "profile_score",
        "profile_score_last_25k",
        "final_km_percentage",
        # "classification",
        "startlist_score",
        "month"
        # "stage_or_one_day", #?
    ]
    windows = {
        "1110d": 1110,
        # "370d": 370,
    }
    rank_thresholds = [
        25,
        5,
        # 3
    ]
    #create feature for every comb of window, win_count and race_features

    #create buckets:
    races_bucketed = races.with_columns(
        [
            pl.col(c).qcut(3, labels=["1", "2", "3"]).alias(f"{c}_bucket").cast(pl.Categorical)
            for c in race_features_to_bucket_on
        ]
    )
    # top_results = results.filter(pl.col("rank") <= 25)
    results_bucketed = results.join(
        races_bucketed,
        on = "race_id",
        how= "inner"
    )
    feature_creation_expressions = [
        # pl.when(pl.col("top_25_count").is_null() | (pl.col("top_25_count") == 0)).then(0).otherwise
        (
            (pl.col(f"{bucket_feature}_bucket") == bucket).sum()
            / (pl.len())
        )
        .alias(f"top_{rank_threshold}_in_{bucket_feature}_{bucket}")
        for bucket_feature in race_features_to_bucket_on
        for bucket in ["1", "2", "3"]
        for rank_threshold in rank_thresholds
    ] 

    results_bucketed_window_dates = results_bucketed.with_columns(
        [
            (pl.col("date") + pl.duration(days = (offset - 1))).alias(f"_window_date_{label}")
            for label, offset in windows.items()
        ]
    ).drop(
        "rank",
        "specialty",
        "team",
        "age",
        "uci_pts",
        "pcs_pts",
        "name_right",
        "year",
        "stage",
        "classification",
        "distance_km",
        "elevation_m",
        "avg_speed_kmh",
        "startlist_score",
        "won_how",
        "temp",
        "profile_score",
        "profile_score_last_25k",
    )
    results_bucketed_window_dates = results_bucketed_window_dates.sort("date")
    lf = results_bucketed_window_dates.lazy()
    
    results_pre_embed_features = results.join(
        races.select(["race_id", "date"]),
        on = "race_id",
        how = "left"
    ).select(
        "name",
        "date"
    ).filter(pl.col("date").is_not_null()).unique()
    for label, _ in windows.items():

        window_lf = lf.group_by_dynamic(
                index_column=f"_window_date_{label}",
                period=label,
                offset= f"-{label}",
                every="1d",
                group_by="name",
                start_by="window",
            ).agg(
                feature_creation_expressions
            ).rename({
                f"_window_date_{label}": "date"
            }).collect(streaming=True)
        window_lf = window_lf.rename({
            col: f"f{col}_{label}"
            for col in window_lf.columns
            if col not in ["name", "date"]
        })
        results_pre_embed_features = results_pre_embed_features.join(
            window_lf,
            on=["name", "date"],
            how="left"
        ).fill_nan(0).fill_null(0)
    
    return results_pre_embed_features


from sklearn.decomposition import PCA
def create_result_embeddings(pre_embed_features: pl.DataFrame, races_df: pl.DataFrame, results_df: pl.DataFrame) -> pl.DataFrame:
    #create embeddings from pre-embed features
    feature_cols = [
        col for col in pre_embed_features.columns
        if col not in ["name", "date"]
    ]
    pre_embed_array = pre_embed_features.select(feature_cols).to_numpy()
    pre_embed_array = StandardScaler().fit_transform(pre_embed_array)

    pca = PCA(n_components=10)
    embedded_array = pca.fit_transform(pre_embed_array)

    #INSPECT WITH
    # - pca.explained_variance_
    # - np.cumsum(pca.explained_variance_ratio_)
    
    embedded_df = pl.DataFrame(embedded_array, schema=[f"embed_{i+1}" for i in range(embedded_array.shape[1])])
    results_embedded = pl.concat([
        pre_embed_features.select(["name", "date"]),
        embedded_df
    ], how="horizontal")
    
    results_embedded_df = results_embedded.join(
        results_df.select(["name", "race_id"]).join(
            races_df.select("race_id", pl.col("date").str.to_date()),
            on = "race_id",
            how="left"
        ),
        on = ["name", "date"],
        how = "inner"
    )
    return results_embedded_df

def create_races_embeddings(results_df, results_embedded_df: pl.DataFrame) -> pl.DataFrame:
    #TODO: also average over multiple years?
    results_embedded_df = results_embedded_df.join(
        results_df.select(["name", "race_id", "rank"]).filter(pl.col("rank") < 25),
        on = ["name", "race_id"],
        how = "left"
    )
    races_embeddings = results_embedded_df.group_by("race_id").agg(
        [
            pl.col(f"embed_{i+1}").mean().alias(f"embed_{i+1}")
            for i in range(len(results_embedded_df.columns) - 4)
        ]
    )
    return races_embeddings

def calculate_cosine_similarity_polars(df, n_dims=10):
    # Build expressions for dot product
    dot_product = pl.lit(0.0)
    norm1_sq = pl.lit(0.0)
    norm2_sq = pl.lit(0.0)
    
    for i in range(1, n_dims + 1):
        col1 = f"embed_{i}"
        col2 = f"embed_{i}_right"
        
        dot_product = dot_product + (pl.col(col1) * pl.col(col2))
        norm1_sq = norm1_sq + (pl.col(col1) ** 2)
        norm2_sq = norm2_sq + (pl.col(col2) ** 2)
    
    # Cosine similarity = dot_product / (norm1 * norm2)
    cosine_sim = dot_product / (norm1_sq.sqrt() * norm2_sq.sqrt())
    
    return df.with_columns(cosine_sim.alias("cosine_similarity"))

def add_embedding_similarity_to_results_df(
    results_embedded_df: pl.DataFrame,
    races_embedded_df: pl.DataFrame
) -> pl.DataFrame:
    
    results_w_race_embeddings = results_embedded_df.join(
        races_embedded_df,
        on = "race_id",
        how= "left",
    )
    embedding_similary = calculate_cosine_similarity_polars(results_w_race_embeddings)

    return results_embedded_df.join(
        embedding_similary.select(["name", 'race_id', "cosine_similarity"]),
        on = ["name", 'race_id'],
        how="left"
    )

def create_result_features_table(results: pl.DataFrame, races:pl.DataFrame) -> pl.DataFrame:

    # -------- individual strength features ----------
    windows = {
        "1110d": 1110,
        "370d": 370,
        "40d": 40,
    }

    dfs = []
    results = results.join(
        races.select(["race_id", pl.col("date").str.to_date()]),
        on = "race_id",
        how = "left"
    )
    bare_results = results.select([
        "name",
        "date",
        "rank"
    ]).sort("date")

    for label, offset in windows.items():
        dfs.append(
            bare_results.with_columns(
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
                pl.len().alias(f"nr_races_participated_{label}"),
                (pl.when(pl.col("rank") < 25).then(1)
                    .otherwise(None)
                ).sum().alias(f"nr_top25_{label}"),
                (pl.when(pl.col("rank") < 10).then(1)
                    .otherwise(None)
                ).sum().alias(f"nr_top10_{label}"),
                (pl.when(pl.col("rank") < 3).then(1)
                    .otherwise(None)
                ).sum().alias(f"nr_top3_{label}"),
            ).rename({f"_window_date_{label}": "date"})
        )
    
    for d in dfs:
        results = results.join(d, on=["name", "date"], how="left").fill_nan(0).fill_null(0)

    # -------- team features ----------
    #top 10 is 2.5 times harder than top 25, top 3 is 8.3 times harder than top 25, so we can weight them accordingly
    #but dont count doubles
    results = results.with_columns(
        (pl.col("nr_top25_370d") + (2.5 - 1) * pl.col("nr_top10_370d") + (8.33 - 1 - 1.5) * pl.col("nr_top3_370d"))
        .alias("strength")
    )
    results = results.with_columns(
        pl.col("strength").rank("dense", descending=True).over(["race_id", "team"]).alias("relative_team_strength_rank")
    )
    return results

"""*****************************************************

MAIN Functions

*****************************************************"""

def check_features_stats():
    """
    Check statistics on result_features: rider entries, races per year, and races per classification.
    Computes everything from the data_v2/result_features_df.parquet table.
    """
    result_features_df = pl.read_parquet("data_v2/result_features_df.parquet")

    # Rider entries stats
    print(result_features_df.fetch(5))
    rider_entries = result_features_df.group_by("name").agg(pl.len().alias("num_entries"))
    avg_entries = rider_entries.select(pl.col("num_entries").mean()).item()
    min_entries = rider_entries.select(pl.col("num_entries").min()).item()
    max_entries = rider_entries.select(pl.col("num_entries").max()).item()
    print(f"Rider entries - Avg: {avg_entries:.2f}, Min: {min_entries}, Max: {max_entries}")

    # Races per year
    races_per_year = result_features_df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).dt.year().alias("year")
    ).group_by("year").agg(pl.col("race_id").n_unique().alias("num_races")).sort("year")
    print("Races per year:")
    print(races_per_year)

    # Races per classification
    races_per_classification = result_features_df.group_by("classification").agg(pl.col("race_id").n_unique().alias("num_races")).sort("classification")
    print("Races per classification:")
    print(races_per_classification)

    # Riders per race
    riders_per_race = result_features_df.group_by('race_id').agg(pl.
    col('name').n_unique().alias('num_riders'))
    min_riders = riders_per_race.select(
        pl.col('num_riders').min()
    ).item()
    max_riders = riders_per_race.select(pl.col('num_riders').
        max()).item()
    mean_riders = riders_per_race.select(pl.
    col('num_riders').mean()).item()
    print(f'Min riders per race: {min_riders}')
    print(f'Max riders per race: {max_riders}')
    print(f'Mean riders per race: {mean_riders:.2f}')

def check_races_stats():
    """
    Check statistics on races

    avg startlist score over classification
    """
    races_df = pl.read_parquet("data_v2/races_df.parquet")
    nr_of_classification_races = (races_df
        .group_by("classification")
        .agg(pl.count().alias("num_races"))
        .sort("classification")
    )
    print(nr_of_classification_races)
    avg_startlist_score = (races_df
        .with_columns(
            pl.col("startlist_score").cast(pl.Float64)
        )
        .group_by("classification")
        .agg(
            pl.col("startlist_score").mean().alias("avg_startlist_score")
        )
        .sort("classification")
    )
    print(avg_startlist_score)

def clean_rider_stats():
    """Cleans some of the cols of rider data.
    
    Cols:
    - classification: to ordinal values
    
    """
    pass

def check_rider_stats():
    pass

def create_normalized_race_data():
    races_df = pl.read_parquet("data_v2/races_df.parquet")
    results_df = pl.read_parquet("data_v2/results_df.parquet").filter(pl.col("rank") != -1)#filter out DNF, DNS, OTL
    # results_similarity = create_results_similarity(results_df)
    interpolated_races_df = interpolate_profile_scores(races_df=races_df)
    normalized_races_df = normalize_race_data(races_df=interpolated_races_df)
    normalized_races_df.write_parquet("data_v2/normalized_races_df.parquet")

def main():
    """create features dataframe"""
    pl.Config.set_tbl_cols(-1)

    results_df = pl.read_parquet("data_v2/results_df.parquet").filter(pl.col("rank") != -1)#filter out DNF, DNS, OTL
    races_df = pl.read_parquet("data_v2/races_df.parquet")
    # riders_df = pl.read_parquet("data_v2/rider_stats_df.parquet")
    normalized_races_df = pl.read_parquet("data_v2/normalized_races_df.parquet")

    # races_features_df = create_race_features_table(
    #     races=races_df, 
    #     results=results_df,
    #     riders=riders_df
    # )
    # races_features_df.write_parquet("data_v2/races_features_df.parquet") 

    result_features_df = create_result_features_table(results=results_df, races=races_df)
    pre_embed_features_df = create_result_features_pre_embed(results=results_df, races=normalized_races_df)
    results_embedded_df = create_result_embeddings(pre_embed_features = pre_embed_features_df, races_df=races_df, results_df=results_df)
    races_embedded_df = create_races_embeddings(
        results_df = results_df,
        results_embedded_df = results_embedded_df)
    results_embedded_df = add_embedding_similarity_to_results_df(
        results_embedded_df=results_embedded_df,
        races_embedded_df=races_embedded_df,
    )
    # print(result_features_df.columns)
    # print(len(result_features_df.columns))
    # print(result_features_df)
    pre_embed_features_df.write_parquet("data_v2/pre_embed_features_df.parquet")
    result_features_df.write_parquet("data_v2/result_features_df.parquet")

    results_embedded_df.write_parquet("data_v2/results_embedded_df.parquet")
    races_embedded_df.write_parquet("data_v2/races_embedded_df.parquet")

def assert_embedding_dimension():
    pre_embed_features_df = pl.read_parquet("data_v2/pre_embed_features_df.parquet")

    results_df = pl.read_parquet("data_v2/results_df.parquet").filter(pl.col("rank") != -1)#filter out DNF, DNS, OTL
    races_df = pl.read_parquet("data_v2/races_df.parquet")
    results_embedded_df = create_result_embeddings(pre_embed_features = pre_embed_features_df, races_df=races_df, results_df=results_df)
    
def assert_embedding_similarity():
    results_embedded_df = pl.read_parquet("data_v2/results_embedded_df.parquet")
    races_embedded_df = pl.read_parquet("data_v2/races_embedded_df.parquet")
    results_embedded_with_similarity = add_embedding_similarity_to_results_df(
        results_embedded_df=results_embedded_df,
        races_embedded_df=races_embedded_df,
    )


if __name__ == "__main__":
    create_normalized_race_data()
    main()
    # assert_embedding_dimension()
    # assert_embedding_similarity()
    # check_results_df()
    # check_races_stats()
    # check_features_stats()
    # check_rider_stats()
    pass