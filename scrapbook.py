import asyncio
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm.asyncio import tqdm
import polars as pl
import re
from pprint import pprint
import pytest 


BASE_URL = "https://www.procyclingstats.com"

# ---- Config ----
START_YEAR = 2010
END_YEAR = 2025
CONCURRENT_REQUESTS = 10
HEADERS = {"User-Agent": "Mozilla/5.0"}

from web_scraping_functions import *
from data_cleaning_functions import *
from data_structure_functions import *
from data_science_functions import *
from test import *
from test_data import raw_sample_data_cleaned
import polars as pl


race_url = "https://www.procyclingstats.com/race/paris-roubaix/2025/result"

async def test_run():
    # sample_data = [
    #     {
    #         'stats': {
    #             'classification': 'WT',
    #             'date': datetime(2012, 3, 17, 0, 0),
    #             'distance': '298',
    #             'heigh_meters': '2380',
    #             'speed': '42.632 km/h',
    #             'startlist_score': '1079',
    #             'temp': '',
    #             'won_how': 'Sprint of small group',
    #             "name": "test_name",
    #             "year": "2012"
    #         },
    #         'results': [
    #             {'name': 'simon-gerrans', 'rank': '1', 'age': '31', 'team': 'Orica GreenEDGE', 'specialty': 'Classic', 'pcs_points': '275', 'uci_points': '100'},
    #             {'name': 'fabian-cancellara', 'rank': '2', 'age': '30', 'team': 'RadioShack - Nissan', 'specialty': 'TT', 'pcs_points': '200', 'uci_points': '80'},
    #             {'name': 'simon-gerrans', 'rank': '3', 'age': '31', 'team': 'Orica GreenEDGE', 'specialty': 'Classic', 'pcs_points': '275', 'uci_points': '100'},  # duplicate
    #         ]
    #     },
    #     {
    #         'results': [
    #             {'age': '29',
    #             'name': 'mathieu-van-der-poel',
    #             'pcs_points': '225',
    #             'rank': '1',
    #             'specialty': 'Classic',
    #             'team': 'Alpecin - Deceuninck',
    #             'uci_points': '400'},
    #             {'age': '31',
    #             'name': 'jasper-stuyven',
    #             'pcs_points': '150',
    #             'rank': '2',
    #             'specialty': 'Classic',
    #             'team': 'Lidl - Trek',
    #             'uci_points': '320'
    #             }
    #         ],
    #         'stats': {
    #             'classification': '1.UWT',
    #             'date': datetime(2024, 3, 22, 0, 0),
    #             'distance': '207.6',
    #             'heigh_meters': '2019',
    #             'name': 'e3-harelbeke',
    #             'speed': '44.571 km/h',
    #             'startlist_score': '734',
    #             'temp': '11 °C',
    #             'won_how': '43.7 km solo',
    #             'year': '2024'
    #         }
    #     }
    # ]

    pl.Config.set_tbl_cols(-1)
    sample_data = raw_sample_data_cleaned[:-1]
    test_race = raw_sample_data_cleaned[-1]["stats"]

    results_df, races_df = transform_race_data(sample_data)
    print(races_df)
    print(test_race)
    scaled_df = find_most_similar_races(test_race, races_df)
    print(scaled_df)
    # print(results_df)
    # print(races_df.columns)
    # print(races_df)
    # create_results_similarity(results_df)
    # print("\nTransformed DataFrame:")
    # 


    # test_stage_race_url = "https://www.procyclingstats.com/race/tour-de-france/2024/stage-2"
    # print(get_race_name_info(test_stage_race_url))
    # weird_urls = []
    # races_urls = await fetch_all_races()
    # for race_url in races_urls:
    #     last_part = race_url.split("/")[-1]
    #     if last_part != "result" and "stage" not in last_part:
    #         weird_urls.append(race_url)
    
    # for weird_url in weird_urls:
    #     okay = False
    #     for race_url in races_urls:
    #         if weird_url in race_url and len(weird_url) != race_url:
    #             okay = True
    #             print(f"Okay: race_url = {race_url}")
    #     if not okay:
    #         print(f"Not Okay: weird_url = {weird_url}")


    # async with httpx.AsyncClient(headers=HEADERS, timeout=30.0) as client:
    #     resp = await client.get(race_url)
    # soup = BeautifulSoup(resp.text, "lxml")
    # results = get_race_results(soup)
    # pprint(results[:5])

    pass

async def test_race_profile_scores():
    test_stage_race_url = "https://www.procyclingstats.com/race/tour-de-france/2024/stage-2"
    race_profile_url = await get_race_profile_url(race_url)

    async with httpx.AsyncClient(headers=HEADERS, timeout=30.0) as client:
        resp = await client.get(race_profile_url)

    soup = BeautifulSoup(resp, "lxml")
    print(get_race_profile(soup))

    # race_url = "https://www.procyclingstats.com/race/paris-roubaix/2025/result"

    # soup = BeautifulSoup(resp.text, "lxml")
    # results = get_race_results(soup)
    # pprint(results[:5])


async def main():

    pass
if __name__ == "__main__":
    asyncio.run(main())