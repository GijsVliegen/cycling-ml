import asyncio
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm.asyncio import tqdm
import polars as pl
import re
from pprint import pprint
import pytest 


from data_science_functions import *

def test_races_features():
    test_races_df = pl.DataFrame({
        "name": ["test_race"] * 5 + ["test_GC_race"] * 6,
        "date": ["2025-01-01", "2024-01-01", "2023-01-01", "2022-01-01", "2021-01-01"]
            + ["2025-02-01", "2025-02-02", "2023-02-01", "2023-02-02", "2021-02-01", "2021-02-02"] , 
        "classification": ["1.1"] * 5 + ["2.UWT"] * 6,
        "year": [2025, 2024, 2023, 2022, 2021] + [2025, 2025, 2023, 2023, 2021, 2021],
        "race_id": ["1" , "2", "3", "4", "5"] + ["11" , "12", "13", "14", "15", "16"],
        "distance_km": [150] * 11,
        "elevation_m": [2000] * 11,
        "profile_score": [25] * 11,
        "profile_score_last_25k": [10] * 11,
    })
    test_results_df = pl.DataFrame({
        "race_id": ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5"]
        + ["11", "11", "12", "12", "13", "13", "14", "14", "15", "15", "16", "16"],
        "name": ["Rider A", "Rider B"] * 11,
        "rank": [1, 2] * 4 + [1, 25] * 1 + [1, 2] * 3 + [1, 25] * 3,
    })
    test_riders_df = pl.DataFrame({
        "name": ["Rider A", "Rider B"],
        "Onedayraces": [10, 20],
        "GC": [5, 15],
        "TT": [2, 8],
        "Sprint": [3, 7],
        "Climber": [4, 6],
        "Hills": [1, 9],
    })
    result = create_race_features_table(
        races=test_races_df,
        results=test_results_df,
        riders=test_riders_df,
    )

    print(result)
    # assert df1.sort(df1.columns).frame_equal(df2.sort(df2.columns))

def test_results_features():
    test_results_df = pl.DataFrame({
        "race_id": ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5", "6", "6", "7", "7"],
        #+ ["11", "11", "12", "12", "13", "13", "14", "14", "15", "15", "16", "16"],
        "name": ["Rider A", "Rider B"] * 7,
        "rank": [1, 2] + [1, 25] + [15, 2] *2 + [1, 25] * 2 + [1, 12],# + [1, 2] * 3 + [1, 25] * 3,
    })
    race_features_df = pl.DataFrame({
        "race_id": ["1" , "2", "3", "4", "5", "6", "7"],# + ["11" , "12", "13", "14", "15", "16"],
        "avg_Onedayraces": [1] * 7,
        "avg_GC": [2] * 7,
        "avg_TT": [3] * 7,
        "avg_Sprint": [4] * 7,
        "avg_Climber": [5] * 7,
        "avg_Hills": [6] * 7,
        "date": [
            datetime(2025, 2, 1), 
            datetime(2025, 1, 12), 
            datetime(2025, 1, 1), 
            datetime(2024, 2, 1), 
            datetime(2024, 1, 1), 
            datetime(2022, 1, 1), 
            datetime(2021, 1, 1)],
    })
    result = create_result_features_table(
        results=test_results_df,
        races_features=race_features_df
    )

    print(result)

if __name__ == "__main__":
    # test_races_features()
    test_results_features()