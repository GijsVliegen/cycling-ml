import polars as pl
from datetime import datetime
import re
import asyncio
import rbo
import hashlib
import itertools
from pprint import pprint
from typing import List, Dict



condition_similarity_cols = [
    # "temp", # TODO: not always present 
    # "classification", #TODO (also convert to numeric value)
    "distance_km", 
    "elevation_m", 
    # "avg_speed_kmh", 
    "startlist_score",
    "profile_score",
    "profile_score_last_25k",
]

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
        for c in condition_similarity_cols
    ])
    knn_races = knn_races.with_columns([
        (pl.sum_horizontal(
            pl.col([c for c in condition_similarity_cols])
        ) #/ len(condition_similarity_cols) #TODO: not needed? right? check formula
        ).sqrt().alias("knn_distance"),
    ]).sort("knn_distance").head(k)
    return knn_races

def normalize_race_data(races_df: pl.DataFrame) -> pl.DataFrame:

    return races_df.with_columns([
        (pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min())
        for c in condition_similarity_cols
    ])


def create_results_similarity(races_df: pl.DataFrame) -> pl.DataFrame:
    top25 = (
        races_df
        .group_by("race_id")
        .agg(
            pl.col("name").filter(pl.col("rank") < 25).alias("top_25_names")
        )
    )
    # print(top25)

    df_pandas = top25.to_pandas() #TODO: use polars
     
    # Get all unique race_ids and their corresponding lists
    races = df_pandas["race_id"].tolist()
    lists = df_pandas["top_25_names"].tolist()

    # Calculate pairwise combinations
    pairs = list(itertools.combinations(range(len(races)), 2))
    
    # Calculate RBO scores for all pairs
    results = []
    
    for i, j in pairs:
        race_i = races[i]
        race_j = races[j]
        list_i = lists[i]
        list_j = lists[j]

            # Handle empty lists
        if list_i is None or list_j is None:
            rbo_score = 0.0
        else:
            try:
                # Calculate RBO score
                rbo_score = rbo.RankingSimilarity(list_i, list_j).rbo(p=0.9) #TODO: check out the p value
            except Exception as e:
                print(f"Error calculating RBO for {race_i} vs {race_j}: {e}")
                rbo_score = 0.0
        
        results.append({ #pair (a, b)
            'race_id_1': race_i,
            'race_id_2': race_j,
            'rbo_score': rbo_score,
            'list_1_length': len(list_i) if list_i is not None else 0,
            'list_2_length': len(list_j) if list_j is not None else 0
        })
        results.append({ #pair (b, a)
            'race_id_1': race_j,
            'race_id_2': race_i,
            'rbo_score': rbo_score,
            'list_1_length': len(list_j) if list_j is not None else 0,
            'list_2_length': len(list_i) if list_i is not None else 0
        })

    # print(results)
    return pl.DataFrame(results)




def interpolate_profile_scores(races_df: pl.DataFrame, results_similarity: pl.DataFrame) -> pl.DataFrame:
    #interpolates missing profile scores based on weighted average using results-similarity as weight

    #do this before normalization
    
    profile_score_races = races_df.filter((pl.col("profile_score") > 0) & (pl.col("profile_score_last_25k") > 0))

    results_similarity_with_profile = results_similarity.join(
        profile_score_races.select(
            [
                pl.col("race_id").alias("race_id_2"), 
                pl.col("profile_score").alias("profile_score_2"), 
                pl.col("profile_score_last_25k").alias("profile_score_last_25k_2"), 
            ]
        ),
        on = "race_id_2",
        how = "inner"
    )
    results_similarity_with_profile = results_similarity_with_profile.with_columns(
        pl.col("rbo_score").sum().over("race_id_1").alias("rbo_score_sum")
    )
    interpolated_profile_scores = results_similarity_with_profile.group_by("race_id_1").agg(
        ((pl.col("rbo_score") / pl.col("rbo_score_sum")) * pl.col("profile_score_2"))
        .sum().alias("interpolated_profile_score"),
        ((pl.col("rbo_score") / pl.col("rbo_score_sum")) * pl.col("profile_score_last_25k_2"))
        .sum().alias("interpolated_profile_score_last_25k")
    )

    interpolated_races = races_df.join(
        interpolated_profile_scores.select([
            pl.col("race_id_1").alias("race_id"),
            "interpolated_profile_score",
            "interpolated_profile_score_last_25k"
        ]),
        on = "race_id",
        how = "left"
    ).with_columns([
        pl.when(pl.col("profile_score") == -1)
        .then(pl.col("interpolated_profile_score"))
        .otherwise(pl.col("profile_score"))
        .alias("profile_score"),
        pl.when(pl.col("profile_score_last_25k") == -1)
        .then(pl.col("interpolated_profile_score_last_25k"))
        .otherwise(pl.col("profile_score_last_25k"))
        .alias("profile_score_last_25k"),
    ]).drop("interpolated_profile_score", "interpolated_profile_score_last_25k")

    interpolated_races = interpolated_races.filter(
        (pl.col("profile_score") > 0) & (pl.col("profile_score_last_25k") > 0)
    ) #filter out if still missing profile scores

    return interpolated_races


def scores_to_probability_results(rider_scores: pl.DataFrame, participants: pl.DataFrame) -> pl.DataFrame:
    """convert rider scores to probability of getting each result for each rider

    rider_scores should have a col "score"
    
    """
    def compute_plackett_luce_probs(scores: List[float], max_rank_to_predict: int = 5) -> Dict[int, List[float]]:
        # edge cases
        n = len(scores)
        if n == 0:
            return []
        total_sum = sum(scores)
        if total_sum == 0:
            return [[0.0] * n for _ in range(max_rank_to_predict)]
        
        current_probs = [s / total_sum for s in scores]
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
    for k in range(len(rank_probs)):
        rider_scores = rider_scores.with_columns(
            pl.Series(f"rank{k+1}_prob", rank_probs[k])
        )

    return rider_scores

def scores_to_results(rider_scores: pl.DataFrame, participants: pl.DataFrame, race: pl.DataFrame) -> pl.DataFrame:
    #convert rider scores to results dataframe format
    return participants.join(
        rider_scores,
        on = "name",
        how = "left"
    ).sort("score", descending=True).with_row_index("predicted_result", offset=1)

def create_feature_table(results: pl.DataFrame, races: pl.DataFrame) -> pl.DataFrame:
    spine = ["name", "race_id"]
    race_distance_features = [
        "distance_km", 
        "elevation_m", 
        "profile_score", 
        "profile_score_last_25k",
        "classification",
        "date"
    ]
    result_features = [
        "rank",
        "startlist_score",
        "age"
    ]
    basic_features = results.join(
        races,
        on="race_id",
    ).select(spine + race_distance_features + result_features)

    basic_features = basic_features.with_columns(
        pl.when(pl.col("rank") <= 3).then(1)
        .when(pl.col("rank") <= 8).then(2)
        .when(pl.col("rank") <= 15).then(3)
        .when(pl.col("rank") <= 25).then(4)
        .when(pl.col("rank") <= 50).then(5)
        .otherwise(6)
        .alias("rank_bucket")
    ).with_columns(
        (
            (pl.col("rank") / pl.col("rank").max().over("race_id")) 
        ).alias("rank_norm")
    ).with_columns(
        (
            ((pl.col("age") - pl.col("age").min()) / pl.col("age").max()) 
        ).alias("age_norm")
    ).with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .dt.year().alias("year")
    )

    basic_features = basic_features.with_columns([
        pl.when(pl.col("rank") <= 3)
            .then((
                pl.when(pl.col("rank") <= 3)
                    .then(pl.struct(["name","year"])) .otherwise(None)
                .n_unique().over("name") - 1)
            ).when((pl.col("rank") > 3) & (pl.col("rank") <= 8))
            .then((
                pl.when((pl.col("rank") > 3) & (pl.col("rank") <= 8))
                    .then(pl.struct(["name", "year"])) .otherwise(None)
                .n_unique().over("name") - 1)
            ).when((pl.col("rank") > 8) & (pl.col("rank") <= 15))
            .then((
                pl.when((pl.col("rank") > 8) & (pl.col("rank") <= 15))
                    .then(pl.struct(["name", "year"])) .otherwise(None)
                .n_unique().over("name") - 1)
            ).when((pl.col("rank") > 15) & (pl.col("rank") <= 25))
            .then((
                pl.when((pl.col("rank") > 15) & (pl.col("rank") <= 25))
                    .then(pl.struct(["name", "year"])) .otherwise(None)
                .n_unique().over("name") - 1)
            ) 
            .otherwise((
                pl.when((pl.col("rank") > 25))
                    .then(pl.struct(["name", "year"])) .otherwise(None)
                .n_unique().over("name") - 1)
            )
        .alias("rank_bucket_year_count")
    ])
    basic_features = basic_features.with_columns([
        pl.when(pl.col("rank") <= 25).then(1).otherwise(0)
        .sum().over(["name", "year"]).alias("top25_count_year"),
        pl.when(pl.col("rank") <= 25).then(1).otherwise(0)
        .sum().over("name").alias("top25_count")
    ]).with_columns(
        (pl.col("date").rank("ordinal").over("name") - 1).alias("attended_races")
    )
    basic_features = basic_features.with_columns(
        pl.col("classification").cast(pl.Categorical).to_physical().alias("classification_encoded")
    )

    nr_riders = basic_features.group_by(
        "race_id"
    ).agg(
        pl.col("name").count().alias("nr_riders")
    )
    basic_features = basic_features.join(
        nr_riders,
        on="race_id",
        how="left"
    )

    return basic_features
    

def main():
    """create features dataframe"""
    pl.Config.set_tbl_cols(-1)

    results_df = pl.read_parquet("data_v2/results_df.parquet").filter(pl.col("rank") != -1)#filter out DNF, DNS, OTL
    results_similarity = create_results_similarity(results_df)
    results_similarity.write_parquet("data_v2/results_similarity.parquet")
    print(results_df)
    # print(results_similarity)

    races_df = pl.read_parquet("data_v2/races_df.parquet")
    interpolated_races_df = interpolate_profile_scores(races_df=races_df, results_similarity=results_similarity)
    
    normalized_races_df = normalize_race_data(races_df=interpolated_races_df)
    print(interpolated_races_df)


    features_df = create_feature_table(results=results_df, races=normalized_races_df)
    print(features_df)
    features_df.write_parquet("data_v2/features_df.parquet")

def check_features_stats():
    """
    Check statistics on features: rider entries, races per year, and races per classification.
    Computes everything from the data_v2/features_df.parquet table.
    """
    features_df = pl.read_parquet("data_v2/features_df.parquet")

    # Rider entries stats
    print(features_df.fetch(5))
    rider_entries = features_df.group_by("name").agg(pl.len().alias("num_entries"))
    avg_entries = rider_entries.select(pl.col("num_entries").mean()).item()
    min_entries = rider_entries.select(pl.col("num_entries").min()).item()
    max_entries = rider_entries.select(pl.col("num_entries").max()).item()
    print(f"Rider entries - Avg: {avg_entries:.2f}, Min: {min_entries}, Max: {max_entries}")

    # Races per year
    races_per_year = features_df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).dt.year().alias("year")
    ).group_by("year").agg(pl.col("race_id").n_unique().alias("num_races")).sort("year")
    print("Races per year:")
    print(races_per_year)

    # Races per classification
    races_per_classification = features_df.group_by("classification").agg(pl.col("race_id").n_unique().alias("num_races")).sort("classification")
    print("Races per classification:")
    print(races_per_classification)

    # Riders per race
    riders_per_race = features_df.group_by('race_id').agg(pl.
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
    rider_df = pl.read_parquet("data_v2/rider_stats_df.parquet")
    print(rider_df)

if __name__ == "__main__":
    # main()
    # check_results_df()
    # check_races_stats()
    # check_features_stats()
    # check_rider_stats()
    pass