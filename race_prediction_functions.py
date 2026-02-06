import polars as pl
from data_science_functions import RACE_SIMILARITY_COLS, scores_to_probability_results


# def normalize_new_race_data(new_race: pl.DataFrame, old_races: pl.DataFrame) -> pl.DataFrame:
#     limits = old_races.select(
#         *[
#             pl.col(c).min().alias(f"{c}_min")
#             for c in RACE_SIMILARITY_COLS
#         ],
#         *[
#             pl.col(c).max().alias(f"{c}_max")
#             for c in RACE_SIMILARITY_COLS
#         ],
#     )
#     new_race_limits = new_race.join(limits, how="cross")
#     new_race_normalized = new_race_limits.select(
#         *[
#             c for c in new_race.columns
#             if c not in RACE_SIMILARITY_COLS
#         ],
#         *[
#             (pl.col(c) - pl.col(f"{c}_min")) / (pl.col(f"{c}_max") - pl.col(f"{c}_min")).alias(c)
#             for c in RACE_SIMILARITY_COLS
#         ]
#     )
#     return new_race_normalized

def get_new_race_embedding(
    race_to_find_for: pl.DataFrame, 
    old_embeddings: pl.DataFrame, 
    old_races: pl.DataFrame
) -> pl.DataFrame:
    """
    closest 5 races in the same (stage-)race over last 5 years

    This function is total overkill since it can act on find for multiple races at once. 
    """
    duplicate_races = race_to_find_for.select("race_id", "name", "year", *RACE_SIMILARITY_COLS).join(
        old_races.select("race_id", "name", "year", *RACE_SIMILARITY_COLS),
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
    closest_race_ids = duplicate_races_distance.sort(
        "total_distance", descending=False
    ).group_by("race_id").head(5).select("race_id", "race_id_right")
    closest_embedding = closest_race_ids.join(
        old_embeddings.rename({"race_id": "race_id_right"}),
        on="race_id_right",
        how="left"
    ).group_by(
        "race_id",
    ).agg(
        *[
            pl.col(c).mean()
            for c in old_embeddings.columns if c != "race_id"
        ]
    )
    return closest_embedding

    

def get_new_race_features(old_races_features: pl.DataFrame, race_to_find_for: pl.DataFrame) -> pl.DataFrame:
    """
    closest 5 races in the same (stage-)race over last 5 years

    This function is total overkill since it can act on find for multiple races at once. 
    """
    duplicate_races = race_to_find_for.select("race_id", "name", "year", *RACE_SIMILARITY_COLS).join(
        old_races_features.select("race_id", "name", "year", *RACE_SIMILARITY_COLS),
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


    closest_race_type_stats = closest_races_distance.join(
        old_races_features.rename({"race_id": "race_id_right"}),
        on="race_id_right"
    )
    race_type_stats = closest_race_type_stats.group_by(
        "race_id"
    ).agg(
        pl.col("avg_Onedayraces").mean().alias("avg_Onedayraces"),
        pl.col("avg_GC").mean().alias("avg_GC"),
        pl.col("avg_TT").mean().alias("avg_TT"),
        pl.col("avg_Sprint").mean().alias("avg_Sprint"),
        pl.col("avg_Climber").mean().alias("avg_Climber"),
        pl.col("avg_Hills").mean().alias("avg_Hills"),
    )
    race_type_stats = race_type_stats.join(
        race_to_find_for,
        on = "race_id"
    )
    return race_type_stats

def get_rider_features(
        new_race: pl.DataFrame,
        startlist: pl.DataFrame, 
        results: pl.DataFrame, 
        races: pl.DataFrame
    ) -> pl.DataFrame:

    results_with_dates = results.join(
        races.select(["race_id", pl.col("date").str.to_date()]),
        on = "race_id",
        how="left"
    )
    startlist_results = startlist.rename({"rider": "name"}).join(
        results_with_dates,
        on = "name"
    )

    windows = {
        "1110d": 1110,
        "370d": 370,
        "40d": 40,
    }
    startlist_results = startlist_results.join(
        new_race.select(pl.col("date").str.to_date().alias("new_race_date")),
        how="cross"
    )

    dfs = []
    for label, offset in windows.items():
        dfs.append(
            startlist_results.filter(
                pl.col("date") >= (pl.col("new_race_date") - pl.duration(days = (offset - 1)))
            ).group_by(
                "name"
            ).agg(
                pl.count("race_id").alias(f"nr_races_participated_{label}"),
                (pl.when(pl.col("rank") < 25).then(1)
                    .otherwise(None)
                ).sum().alias(f"nr_top25_{label}"),
                (pl.when(pl.col("rank") < 10).then(1)
                    .otherwise(None)
                ).sum().alias(f"nr_top10_{label}"),
                (pl.when(pl.col("rank") < 3).then(1)
                    .otherwise(None)
                ).sum().alias(f"nr_top3_{label}"),
            )
        )
    rider_features = startlist.join(
        new_race.select("race_id"),
        how="cross"
    ).rename({"rider": "name"})
    for df in dfs:
        if len(df) == 0:
            continue
        rider_features = rider_features.join(
            df,
            on="name",
            how="left"
        ).fill_null(0)
    return rider_features

def get_rider_embeddings(startlist: pl.DataFrame, results_embedded_df: pl.DataFrame) -> pl.DataFrame:
    most_recent_embedding_date = results_embedded_df.sort("date").with_columns(
        pl.col("date").last().over("name").alias("most_recent_date")
    )
    most_recent_embedding = most_recent_embedding_date.filter(pl.col("date") == pl.col("most_recent_date"))
    
    return startlist.rename({"rider": "name"}).select("name").join(
        most_recent_embedding,
        on="name",
        how="left"
    ).drop("date", "most_recent_date", "cosine_similarity", "race_id")

def get_rider_pairs(rider_features_df: pl.DataFrame) -> list[list[str, str]]:
    """
    returns pairs of rider names
    """
    pairs = (
        rider_features_df
        .join(rider_features_df, how="cross")
        .filter(pl.col("name") < pl.col("name_right"))
        .select(
            pl.struct(pl.col("name"), pl.col("name_right")).alias("pair")
        )
    )
    pairs_list = (
        pairs
        .select(pl.col("pair").map_elements(lambda x: (x["name"], x["name_right"])))
        .to_series()
        .to_list()
    )
    return pairs_list

from data_science_functions import calculate_cosine_similarity_polars
def add_embedding_similarity_for_new_race(
    rider_embeddings: pl.DataFrame, 
    race_embedding: pl.DataFrame
) -> pl.DataFrame:

    results_w_race_embeddings = rider_embeddings.join(
        race_embedding,
        how= "cross",
    )
    embedding_similary = calculate_cosine_similarity_polars(results_w_race_embeddings)

    return rider_embeddings.join(
        embedding_similary.select(["name", "cosine_similarity"]),
        on = ["name"],
        how="left"
    )

import numpy as np

def new_race_to_xgboost_format(
        rider_features_df: pl.DataFrame, 
        riders_yearly_data: pl.DataFrame,
        new_race_features: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[list[str, str]]]:
        mock_model = RaceModel()

        race_id: int = new_race_features.select(
            ["race_id"] 
        ).unique().to_series().to_list()[0] #Take race_ids from results since only care about races with results
    
        rider_pairs = get_rider_pairs(
            rider_features_df = rider_features_df
        )

        race_Y = np.random.choice([-1, 1], size=(len(rider_pairs),))

        rider_pair_features = mock_model.get_rider_pair_features(
            rider_pairs=rider_pairs,
            result_features_df = rider_features_df,
            riders_yearly_data = riders_yearly_data,
            race_year=2026,
            race_id = race_id,
            race_Y=race_Y
        )

        race_features = new_race_features.filter(
            pl.col("race_id") == race_id
        ).select(mock_model.race_features).to_numpy()[0].astype(np.float32, copy=False)
        race_features = np.tile(race_features, (len(rider_pairs), 1))

        race_X = np.hstack([rider_pair_features, race_features])
        # X.append(race_X)
        # y.append(race_Y)

        # y = np.concatenate(y)
        # X = np.vstack(X)
        # X_weights = np.array(X_weights)
        # print(f"training on {len(y)} pairs")

        return race_X, race_Y, rider_pairs

from xgboost_functions import RaceModel
import xgboost as xgb

def main():
    startlist_df = pl.read_parquet("data_v2/new_race_startlist.parquet")
    race_stats_df = pl.read_parquet("data_v2/new_race_stats.parquet")
    races_df = pl.read_parquet("data_v2/races_df.parquet")
    results_df = pl.read_parquet("data_v2/results_df.parquet")
    riders_yearly_data = pl.read_parquet("data_v2/rider_yearly_stats_df.parquet")
    races_embedded_df = pl.read_parquet("data_v2/races_embedded_df.parquet")
    results_embedded_df = pl.read_parquet("data_v2/results_embedded_df.parquet")
    """Prepare data of race and riders to give to xgboost"""

    # new_race_normalized = normalize_new_race_data(
    #     new_race = race_stats_df,
    #     old_races = races_df #use non normalized to normalize
    # )

    # race_feats = get_new_race_features(
    #     old_races_features = races_features_df,
    #     race_to_find_for=race_stats_df
    # )
    race_embedding = get_new_race_embedding(
        race_to_find_for = race_stats_df,
        old_embeddings = races_embedded_df,
        old_races = races_df,
    )
    race_feats = race_embedding.join(
        race_stats_df,
        on = ["race_id"],
        how="left"
    )
    rider_feats = get_rider_features(
        new_race=race_stats_df,
        startlist=startlist_df,
        results = results_df,
        races=races_df
    )
    rider_embeddings = get_rider_embeddings(
        startlist = startlist_df,
        results_embedded_df = results_embedded_df
    )
    rider_embeddings = add_embedding_similarity_for_new_race(
        rider_embeddings=rider_embeddings,
        race_embedding=race_embedding,
    )
    rider_feats = rider_feats.join(
        rider_embeddings,
        on="name"
    )

    """Use prepared data to generate inter-rider results"""

    model = RaceModel()
    model.load_model()

    X, Y, pairs = new_race_to_xgboost_format(
        rider_features_df=rider_feats, 
        riders_yearly_data=riders_yearly_data, 
        new_race_features = race_feats 
    )
    riders = set(
        [rider for pair in pairs for rider in pair]
    )
    rider_scores = {
        rider: [] for rider in riders
    }

    dtest = xgb.DMatrix(X, label=Y)
    y_pred_proba = model.bst.predict(dtest)
    for (first, second), pred_y, y in zip(pairs, y_pred_proba, Y):
        if y == 1:
            rider_scores[first].append(pred_y)
            rider_scores[second].append( - pred_y)
        else:
            rider_scores[first].append( - pred_y)
            rider_scores[second].append(pred_y)
    rider_scores = {
        rider: np.sum(scores) for rider, scores in rider_scores.items()
    }
    rider_scores_df = pl.DataFrame(
        list(rider_scores.items()), schema=["name", "score"]
    )    
    rider_percentages_df = scores_to_probability_results(rider_scores_df)
    top10 = sorted(rider_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        # Print nicely
    for name, score in top10:
        print(f"{name}: {score:.4f}")
    return


if __name__ == "__main__":
    main()