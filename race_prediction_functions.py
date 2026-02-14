import polars as pl
from data_science_functions import RACE_SIMILARITY_COLS, scores_to_probability_results

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


import numpy as np
from xgboost_functions import RaceModel, get_rider_embeddings
import xgboost as xgb

from xgboost_functions import RaceModel, \
    add_embedding_similarity_for_new_race, \
    new_race_to_xgboost_format, \
    get_rider_features, get_rider_embeddings
from data_science_functions import filter_data

def main():
    startlist_df = pl.read_parquet("data_v2/new_race_startlist.parquet")
    race_stats_df = pl.read_parquet("data_v2/new_race_stats.parquet")
    races_df = pl.read_parquet("data_v2/races_df.parquet")
    results_df = pl.read_parquet("data_v2/results_df.parquet")
    riders_yearly_data = pl.read_parquet("data_v2/rider_yearly_stats_df.parquet")
    races_embedded_df = pl.read_parquet("data_v2/races_embedded_df.parquet")
    results_embedded_df = pl.read_parquet("data_v2/results_embedded_df.parquet")
    necessary_races, necessary_results = filter_data(races_df, results_df)
    """Prepare data of race and riders to give to xgboost"""

    race_embedding = get_new_race_embedding(
        race_to_find_for = race_stats_df,
        old_embeddings = races_embedded_df,
        old_races = necessary_races,
    )
    race_feats = race_embedding.join(
        race_stats_df,
        on = ["race_id"],
        how="left"
    )
    rider_feats = get_rider_features(
        new_race=race_stats_df,
        startlist=startlist_df,
        results = necessary_results,
        races=necessary_races
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
    y_pred_proba = model.bst.predict(dtest)#, output_margin=True)
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