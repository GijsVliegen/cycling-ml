import polars as pl
from data_science_functions import (
    RACE_SIMILARITY_COLS, 
    scores_to_probability_results,
    calculate_cosine_similarity_polars,
    filter_data
)


import numpy as np
import xgboost as xgb
from pathlib import Path

from xgboost_functions import RaceModel
from xgboost_functions import load_rider_personal_data


DEFAULT_DATA_DIR = "data_v2"


def data_path(data_dir: str, filename: str) -> str:
    return str(Path(data_dir) / filename)

def get_new_race_embedding(
    race_to_find_for: pl.DataFrame, 
    base_embeddings: pl.DataFrame, 
    old_races: pl.DataFrame
) -> pl.DataFrame:
    """
    closest 8 races in the same (stage-)race over last 8 years

    This function is total overkill since it can act on find for multiple races at once. 
    """
    duplicate_races = race_to_find_for.select("race_id", "name", "year", *RACE_SIMILARITY_COLS).join(
        old_races.select("race_id", "name", "year", *RACE_SIMILARITY_COLS),
        on = ["name"]
    ).filter(
        pl.col("year").cast(pl.Int64) <= pl.col("year_right").cast(pl.Int64) + 4
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
    ).group_by("race_id").head(3).select("race_id", "race_id_right")
    closest_embedding = closest_race_ids.join(
        base_embeddings.rename({"race_id": "race_id_right"}),
        on="race_id_right",
        how="left"
    ).group_by(
        "race_id",
    ).agg(
        *[
            pl.col(c).mean()
            for c in base_embeddings.columns if c != "race_id"
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


# def get_new_race_embedding(
#     race_to_find_for: pl.DataFrame, 
#     old_embeddings: pl.DataFrame, 
#     old_races: pl.DataFrame
# ) -> pl.DataFrame:
#     """
#     closest 5 races in the same (stage-)race over last 5 years

#     This function is total overkill since it can act on find for multiple races at once. 
#     """
#     duplicate_races = race_to_find_for.select("race_id", "name", "year", *RACE_SIMILARITY_COLS).join(
#         old_races.select("race_id", "name", "year", *RACE_SIMILARITY_COLS),
#         on = ["name"]
#     ).filter(
#         pl.col("year").cast(pl.Int64) <= pl.col("year_right").cast(pl.Int64) + 5
#     ).filter(
#         pl.col("year").cast(pl.Int64) >= pl.col("year_right").cast(pl.Int64)
#     ).filter(
#         ~ (pl.col("race_id") == pl.col("race_id_right"))
#     )
#     if len(duplicate_races) < 5:
#         breakpoint()
#         print(f"Warning: only {len(duplicate_races)} similar races found for race \
#               {race_to_find_for.select('name').to_numpy()[0][0]}")
#     #TODO years should be replaced by dates, so only earlier stages are taken
#     # not critical tho
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
#     ).group_by("race_id").head(5).select("race_id", "race_id_right")
#     closest_embedding = closest_race_ids.join(
#         old_embeddings.rename({"race_id": "race_id_right"}),
#         on="race_id_right",
#         how="left"
#     ).group_by(
#         "race_id",
#     ).agg(
#         *[
#             pl.col(c).mean()
#             for c in old_embeddings.columns if c != "race_id"
#         ]
#     )
#     return closest_embedding

def get_rider_features(
        new_race: pl.DataFrame,
        startlist: pl.DataFrame, 
        results: pl.DataFrame,
        races: pl.DataFrame,
        pre_embed_features_df: pl.DataFrame,
    ) -> pl.DataFrame:
    del results, races

    new_race_date = new_race.select(pl.col("date").str.to_date().alias("new_race_date")).to_series().to_list()[0]
    pre_embed_cols = [col for col in pre_embed_features_df.columns if col.startswith("ftop_")]

    latest_pre_embed_per_rider = (
        pre_embed_features_df
        .filter(pl.col("date") < new_race_date)
        .sort("date", descending=True)
        .group_by("name")
        .first()
        .select(["name", *pre_embed_cols])
    )

    rider_features = (
        startlist.join(
            new_race.select("race_id"),
            how="cross"
        )
        .rename({"rider": "name"})
        .join(
            latest_pre_embed_per_rider,
            on="name",
            how="left"
        )
        .with_columns([
            pl.col(col).fill_null(0.0)
            for col in pre_embed_cols
        ])
    )

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
    ).drop("date", "most_recent_date", "cosine_similarity", "l1_distance", "race_id")

def get_rider_pairs(race_id: int, result_features_df: pl.DataFrame, min_top_rank: int = None) -> list[list[str, str]]:
    """
    returns pairs of rider names
    """
    max_weight = 10
    nr_riders = result_features_df.filter(pl.col("race_id") == race_id).height

    if min_top_rank is None:
        min_top_rank = nr_riders

    top_riders = result_features_df.filter(
        pl.col("race_id") == race_id
    ).sort("race_id")
    all_riders = result_features_df.filter(
        pl.col("race_id") == race_id
    ).sort("race_id")         
    pairs = []
    for i, top_rider in enumerate(top_riders["name"]):
        for j, other_rider in enumerate(all_riders["name"]):
            if top_rider != other_rider:
                if (top_rider, other_rider) not in pairs:
                    pairs.append((top_rider, other_rider))
                    pairs.append((other_rider, top_rider))
    return pairs


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
        embedding_similary.select(["name", "cosine_similarity", "l1_distance"]),
        on = ["name"],
        how="left"
    )

def new_race_to_xgboost_format(
    rider_features_df: pl.DataFrame, 
    riders_yearly_data: pl.DataFrame,
    riders_personal_data: pl.DataFrame,
    new_race_features: pl.DataFrame,
) -> np.ndarray:
    assert new_race_features.shape[0] == 1
    mock_model = RaceModel()

    race_id: int = new_race_features.select(
        ["race_id"] 
    ).unique().to_series().to_list()[0] #Take race_ids from results since only care about races with results
    race_year: int = int(new_race_features.select("year").to_numpy()[0][0])

    riders_features, rider_embeddings, _ = mock_model.get_rider_feats(
        race_results = rider_features_df,
        riders_yearly_data = riders_yearly_data,
        riders_personal_data = riders_personal_data,
        race_id = race_id,
        race_year=race_year,
        ranks=False
    )


    race_features = new_race_features.select(mock_model.race_features).to_numpy()[0].astype(np.float32, copy=False)
    race_embeddings = new_race_features.select(mock_model.embed_features).to_numpy()[0].astype(np.float32, copy=False)
    race_features = np.tile(race_features, (len(riders_features), 1))
    race_embeddings = np.tile(race_embeddings, (len(riders_features), 1))
    embedding_diff = rider_embeddings - race_embeddings
    team_rank_feature = mock_model.add_team_rank_feature(
        rider_features=riders_features,
        race_features_tiled=race_features,
        race_results=rider_features_df,
    )
    riders_features = np.hstack([riders_features, team_rank_feature])

    race_X = np.hstack([riders_features, race_features, embedding_diff])

    return race_X

def predict_race(startlist_df, race_stats_df, data_dir: str = DEFAULT_DATA_DIR):
    """Compute scores and perform exp them (for softmax later on in placket-luce)
    """
    
    races_df = pl.read_parquet(data_path(data_dir, "races_df.parquet"))
    results_df = pl.read_parquet(data_path(data_dir, "results_df.parquet"))
    pre_embed_features_df = pl.read_parquet(data_path(data_dir, "pre_embed_features_df.parquet"))
    riders_yearly_data = pl.read_parquet(data_path(data_dir, "rider_yearly_stats_df.parquet"))
    riders_personal_data = load_rider_personal_data(data_dir)
    
    races_base_embedded_df = pl.read_parquet(data_path(data_dir, "races_base_embedded_df.parquet"))
    results_embedded_df = pl.read_parquet(data_path(data_dir, "results_embedded_df.parquet"))
    """Prepare data of race and riders to give to xgboost"""
    
    riders_yearly_data = riders_yearly_data.with_columns(
        pl.all().replace(-1, 0)
    )
    necessary_races, necessary_results = filter_data(races_df, results_df, feature_creation=True)

    if len(
        race_stats_df.filter(pl.col("name") == "classic-brugge-de-panne")
    ) == 1:
        "inspect embedding since it doensnt predict spinners AT ALL for this"
    race_embedding = get_new_race_embedding(
        race_to_find_for = race_stats_df,
        base_embeddings = races_base_embedded_df,
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
        races=necessary_races,
        pre_embed_features_df=pre_embed_features_df,
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

    model = RaceModel(data_dir=data_dir, test_mode=False)
    model.load_model()

    X = new_race_to_xgboost_format(
        rider_features_df=rider_feats, 
        riders_yearly_data=riders_yearly_data, 
        riders_personal_data=riders_personal_data,
        new_race_features = race_feats 
    )
    dtest = xgb.DMatrix(X)
    y_pred = model.bst.predict(dtest)
    scores = rider_feats.select("name").with_columns(
        pl.Series(y_pred).alias("score")
    ).sort("score", descending=True)
    return scores

def main():
    startlist_df = pl.read_parquet(data_path(DEFAULT_DATA_DIR, "new_race_startlist.parquet"))
    race_stats_df = pl.read_parquet(data_path(DEFAULT_DATA_DIR, "new_race_stats.parquet"))
    """Prepare data of race and riders to give to xgboost"""
    predict_race(
        startlist_df = startlist_df,
        race_stats_df = race_stats_df,
        data_dir=DEFAULT_DATA_DIR,
    )
    return


if __name__ == "__main__":
    main()