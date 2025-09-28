
import polars as pl
from datetime import datetime
import re
import asyncio
import rbo
import itertools
from data_cleaning_functions import get_all_results, get_results_one_race




# def get_racer_df(races_data):
    
def transform_race_data(races_data: list[dict]) -> pl.DataFrame:
    """
    Transform race data using Polars with the following operations:
    1. Parse date to datetime
    2. Convert distance, heigh_meters, speed, startlist_score to numeric
    3. Clean blank/missing temp values to NaN or drop
    """

    # Create a Polars DataFrame
    races_df = pl.DataFrame([race_data["stats"] for race_data in races_data])

    if "distance_km" not in races_df.columns:
        print(f"Error: missing distance_km in this {races_data}")

    # Convert types
    races_df = races_df.with_columns([
        # pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("distance_km").cast(pl.Float64, strict=False),
        pl.col("elevation_m").cast(pl.Float64, strict=False),
        pl.col("avg_speed_kmh").cast(pl.Float64, strict=False),
        pl.col("startlist_score").cast(pl.Float64, strict=False),
        pl.col("temp").cast(pl.Float64, strict=False),
        pl.col("profile_score").cast(pl.Float64, strict=False),
        pl.col("profile_score_last_25k").cast(pl.Float64, strict=False),
        #TODO: add profile scores
    ])

    all_results = []
    for results in [race_data["results"] for race_data in races_data]:
        all_results += results
    
    results_df = pl.DataFrame(all_results)
    results_df = results_df.with_columns([

        pl.when(pl.col("rank").cast(pl.Float64, strict=False).is_not_null())
        .then(pl.col("rank").cast(pl.Float64, strict=False))
        .otherwise(-1.0)
        .alias("rank"), #Save casting from str to float
        pl.col("age").cast(pl.Float64, strict=False),
        pl.col("uci_pts").cast(pl.Float64, strict=False),
        pl.col("pcs_pts").cast(pl.Float64, strict=False),
    ])

    # print(results_df)
    return results_df, races_df


async def make_data_structure():
    """
    Transforms raw data into structured polars Dataframes and persists these
    """
    
    all_results = await get_all_results(k=500)
    results_df, races_df = transform_race_data(all_results)
    print(results_df)
    print(races_df)
    results_df.write_parquet("data/results_df.parquet")
    races_df.write_parquet("data/races_df.parquet")

async def test_data_structure(race_url):

    one_result = await get_results_one_race(race_url)
    results_df, races_df = transform_race_data([one_result])
    print(results_df)
    print(races_df)

async def main():
    await make_data_structure()
    # pl.Config(tbl_cols=-1)
    # await test_data_structure("https://www.procyclingstats.com/race/bretagne-classic/2012/result")

if __name__ == "__main__":
    asyncio.run(main())