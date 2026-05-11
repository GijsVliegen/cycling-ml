import polars as pl
from pathlib import Path

from data_engineering.data_structure_functions import create_new_race_data
from wielermanager.race_prediction_functions import predict_race
from wielermanager.race_config import (
    load_wielermanager_rules,
    write_race_manifest,
)
from data_science_functions import scores_to_probability_results

rules = load_wielermanager_rules()
points_per_race_type = rules["points_per_race"]

# with open("wielermanager/WIELERMANAGER_BUDGETS.json") as f:
#     budgets = json.load(f)
#     riders_with_known_calender = budgets["riders_with_calender_known"]


DEFAULT_DATA_DIR = "data_v2"
DEFAULT_TEST_DATA_DIR = "data_test"


def data_path(data_dir: str, filename: str) -> str:
    return str(Path(data_dir) / filename)


def ensure_wielermanager_dir(data_dir: str) -> Path:
    target = Path(data_dir) / "wielermanager"
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_races_to_predict(data_dir: str = DEFAULT_DATA_DIR, year: int = 2026):
    return [
        (race_entry["pcs_name"], -1, race_entry["type"])
        for race_entry in rules.get("races_voorjaar", [])
    ]


def get_available_races(data_dir: str = DEFAULT_DATA_DIR, limit: int | None = None):
    available = []
    for race, stage, race_type in get_races_to_predict(data_dir=data_dir):
        startlist_path = Path(data_path(data_dir, f"wielermanager/startlist_{race}_{stage}.parquet"))
        stats_path = Path(data_path(data_dir, f"wielermanager/new_race_stats_{race}_{stage}.parquet"))
        if startlist_path.exists() and stats_path.exists():
            available.append((race, stage, race_type))
    if limit is not None:
        available = available[:limit]
    return available


def prepare_test_race_data(
    source_dir: str = DEFAULT_DATA_DIR,
    target_dir: str = DEFAULT_TEST_DATA_DIR,
    limit: int = 6,
):
    ensure_wielermanager_dir(target_dir)
    selected_races = get_available_races(source_dir, limit=limit)
    for race, stage, _ in selected_races:
        pl.read_parquet(data_path(source_dir, f"wielermanager/startlist_{race}_{stage}.parquet")).write_parquet(
            data_path(target_dir, f"wielermanager/startlist_{race}_{stage}.parquet")
        )
        pl.read_parquet(data_path(source_dir, f"wielermanager/new_race_stats_{race}_{stage}.parquet")).write_parquet(
            data_path(target_dir, f"wielermanager/new_race_stats_{race}_{stage}.parquet")
        )
    return selected_races


def get_startlists(data_dir: str = DEFAULT_DATA_DIR, year: int = 2026):
    ensure_wielermanager_dir(data_dir)
    races_to_predict = get_races_to_predict(data_dir=data_dir, year=year)
    write_race_manifest(data_dir, races_to_predict)
    logs = create_new_race_data(races_to_predict, data_dir=data_dir, year=year)
    print(logs)
    return races_to_predict



def compute_rider_average_points(
    data_dir: str = DEFAULT_DATA_DIR,
    selected_races: list[tuple[str, int, str]] | None = None,
):
    ensure_wielermanager_dir(data_dir)
    if selected_races is None:
        selected_races = get_available_races(data_dir)
    for race, stage, race_type in selected_races:
        startlist_df = pl.read_parquet(
            data_path(data_dir, f"wielermanager/startlist_{race}_{stage}.parquet")
        )
        race_stats_df = pl.read_parquet(
            data_path(data_dir, f"wielermanager/new_race_stats_{race}_{stage}.parquet")
        )
        print(f"startlist length = {len(startlist_df)}")
        scores = predict_race(startlist_df=startlist_df, race_stats_df=race_stats_df, data_dir=data_dir)
        rider_percentages = convert_scores_to_points(race_type, scores, race)
        print(f"race: {race}")
        print(f"nr participants: {len(rider_percentages)}")
        print(rider_percentages.head(10))
        rider_percentages.write_parquet(
            data_path(data_dir, f"wielermanager/rider_percentages_{race}.parquet")
        )

    return [race for race, _, _ in selected_races]


def convert_scores_to_points(race_type, scores_df, race):
    points_per_rank = points_per_race_type[race_type]
    points_per_kopman_rank = {"1": 30, "2": 25, "3": 20, "4": 15, "5": 10, "6": 5}

    # TODO: dont guesstimate temperature but minimize log loss on historical data to find best temperature for plackett-luce

    manually_boost_riders_list = [
        "jonas-vingegaard"
    ]
    manually_lower_riders_list = [
    ]
    scores_df = scores_df.with_columns(
        pl.when(pl.col("name").is_in(manually_boost_riders_list))
        .then(pl.col("score") + 0.8)
        .otherwise(pl.col("score"))
        .alias("score")
    ).sort("score", descending=True)
    scores_df = scores_df.with_columns(
        pl.when(pl.col("name").is_in(manually_lower_riders_list))
        .then(pl.col("score") - 0.8)
        .otherwise(pl.col("score"))
        .alias("score")
    ).sort("score", descending=True)

    rider_percentages_df = scores_to_probability_results(
        scores_df, max_rank_to_predict=30, temperature=0.2
    )
    rider_percentages_df = rider_percentages_df.with_columns(
        pl.sum_horizontal(
            [pl.col(f"rank_{k}_prob") for k in range(1, len(points_per_rank) + 1)]
        ).alias("total_probability"),
        pl.sum_horizontal(
            [
                pl.col(f"rank_{k}_prob") * points_per_rank[str(k)]
                for k in range(1, len(points_per_rank) + 1)
            ]
        ).alias("expected_points_overestimate"),
        pl.sum_horizontal(
            [
                pl.col(f"rank_{k}_prob") * points_per_kopman_rank[str(k)]
                for k in range(1, len(points_per_kopman_rank) + 1)
            ]
        ).alias("expected_kopman_points_overestimate"),
    )
    rider_percentages_df.with_columns(
        pl.when(pl.col("total_probability") < 1)
        .then(1)
        .otherwise(pl.col("total_probability"))
        .alias("total_probability"),
    )
    rider_percentages_df = rider_percentages_df.with_columns(
        (pl.col("expected_points_overestimate") / pl.col("total_probability")).alias(
            "expected_points"
        ),
        (
            pl.col("expected_kopman_points_overestimate") / pl.col("total_probability")
        ).alias("expected_kopman_points"),
    )  
    return rider_percentages_df


if __name__ == "__main__":
    get_startlists()
    compute_rider_average_points()


def main_test(
    source_dir: str = DEFAULT_DATA_DIR,
    target_dir: str = DEFAULT_TEST_DATA_DIR,
    limit: int = 6,
):
    selected_races = prepare_test_race_data(source_dir=source_dir, target_dir=target_dir, limit=limit)
    computed_races = compute_rider_average_points(data_dir=target_dir, selected_races=selected_races)
    return {
        "data_dir": target_dir,
        "races": computed_races,
    }
