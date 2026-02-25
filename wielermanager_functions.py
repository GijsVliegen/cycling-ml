import json
import polars as pl
from data_engineering.data_structure_functions import (
    create_new_race_data
)
from race_prediction_functions import predict_race
from data_science_functions import scores_to_probability_results

with open("WIELERMANAGER_RULES.json") as f:
    rules = json.load(f)
    races_raw = rules["races"]
    races_to_predict = [
        (raw_race["pcs_name"], -1, raw_race["type"])
        for raw_race in races_raw
    ]
    points_per_race_type = rules["points_per_race"]

with open("WIELERMANAGER_BUDGETS.json") as f:
    budgets = json.load(f)
    riders_with_known_calender = budgets["riders_with_calender_known"]

def get_startlists():
    logs = create_new_race_data(races_to_predict)
    print(logs)

def extend_startlists():
    for race, stage, race_type in races_to_predict:
        startlist_df = pl.read_parquet(f"data_v2/wielermanager/startlist_{race}.parquet")
        riders_per_team = startlist_df.groupby("team").agg(pl.col("name").alias("riders"))
        riders_with_known_calender = None

def compute_rider_average_points():
    for race, stage, race_type in races_to_predict:
        startlist_df = pl.read_parquet(f"data_v2/wielermanager/startlist_{race}.parquet")
        race_stats_df = pl.read_parquet(f"data_v2/wielermanager/new_race_stats_{race}.parquet")
        print(f"startlist length = {len(startlist_df)}")
        scores = predict_race(
            startlist_df = startlist_df,
            race_stats_df = race_stats_df
        )
        rider_percentages = convert_scores_to_points(race_type, scores, race)
        print(f"race: {race}")
        print(f"nr participants: {len(rider_percentages)}")
        print(rider_percentages.head(10))

    pass

def convert_scores_to_points(race_type, scores_df, race):
    points_per_rank = points_per_race_type[race_type]
    
    #TODO: dont guesstimate temperature but minimize log loss on historical data to find best temperature for plackett-luce

    rider_percentages_df = scores_to_probability_results(scores_df, max_rank_to_predict=30, temperature=0.05)
    rider_percentages_df = rider_percentages_df.with_columns(
        pl.sum_horizontal([
            pl.col(f"rank_{k}_prob") for k in range(1, len(points_per_rank) + 1)
        ]).alias("total_probability"),
        pl.sum_horizontal([
            pl.col(f"rank_{k}_prob") * points_per_rank[str(k)] for k in range(1, len(points_per_rank) + 1)
        ]).alias("expected_points_overestimate"),
    )
    rider_percentages_df = rider_percentages_df.with_columns(
        (pl.col("expected_points_overestimate") / pl.col("total_probability")).alias("expected_points")
    )#.select(["name", "expected_points"])
    # df = df.with_row_index("row_nr")

    # Convert dict to DataFrame

    # Join
    # df = df.join(dict_df, on="row_nr", how="left").drop("row_nr")
    return rider_percentages_df

if __name__ == "__main__":
    get_startlists()
    compute_rider_average_points()