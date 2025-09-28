import asyncio
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm.asyncio import tqdm
import polars as pl
import re
from pprint import pprint
import pytest 
import time
from datetime import datetime
import hashlib

BASE_URL = "https://www.procyclingstats.com"

# ---- Config ----
START_YEAR = 2010
END_YEAR = 2025
CONCURRENT_REQUESTS = 10
HEADERS = {"User-Agent": "Mozilla/5.0"}

from web_scraping_functions import *


def filter_stats(stats: dict, race_url) -> dict:
    stats = {
        "startlist_score": stats["Startlist quality score"],
        "heigh_meters": stats["Vertical meters"],
        "date": datetime.strptime(stats["Date"], "%d %B %Y"), #or Datename aswell sometimes?
        "distance": stats["Distance"].split(" ")[0],
        "classification": stats["Classification"],
        "won_how": stats["Won how"],
        "speed": stats["Avg. speed winner"],
        "temp": stats["Avg. temperature"],
        "profile_score": stats["ProfileScore"],
        "profile_score_last_25k": stats["PS_final_25k"],
    } | get_race_name_info(race_url)

    date_str = stats["date"].strftime("%Y-%m-%d")
    classification = stats["classification"]
    
    # Generate race_id: Hash of date+classification
    # race_id = "R" + hashlib.md5(f"{date_str}_{classification}".encode()).hexdigest()[:8]

    return {
        # "race_id": race_id,
        "race_id": "R" + hashlib.md5(f"{stats.get("name")}_{stats.get("year")}_{stats.get("stage")}".encode()).hexdigest()[:8],
        "date": date_str,
        "name": stats.get("name"),
        "year": stats.get("year"),
        "stage": stats.get("stage"),
        "classification": classification,
        "distance_km": stats.get("distance") if stats.get("distance") else "-1",
        "elevation_m": stats.get("heigh_meters") if stats.get("heigh_meters") else "-1",
        "avg_speed_kmh": stats.get("speed").replace(" km/h", "") if stats.get("speed") else "-1",
        "startlist_score": stats.get("startlist_score").split(" ")[1][1:-1] if " " in stats.get("startlist_score") else stats.get("startlist_score"), #in case of stage races, take current startlist score for that stage
        "won_how": stats.get("won_how"),
        "temp": stats.get("temp").replace(" °C", "") or "-1", #lets hope this is not a possible value
        "profile_score": stats.get("profile_score"),
        "profile_score_last_25k": stats.get("profile_score_last_25k"),
    }


def filter_results(results: list[dict], race_stats: dict) -> list[dict]:
    seen_rider_ids = set()

    new_results = []
    for rider in results:
        
        rider_name = rider["racer_url_index"].split("/")[-1]
        if rider_name in seen_rider_ids:
            print(f"⚠️ Duplicate entry for rider '{rider_name}' in race {race_stats.get("name")}")
            continue
        seen_rider_ids.add(rider_name)

        new_results.append({
            "rank": rider.get("Rnk") if (rider.get("Rnk") and rider.get("Rnk") not in {"DNF", "DNS", "OTL"}) else "-1", #TODO: Rnk can be "DNF"
            "name": rider_name,
            "specialty": rider["Specialty"],
            "team": rider["Team"],
            "age": rider.get("Age") if rider.get("Age") else "-1",
            "uci_pts": rider.get("UCI") if rider.get("UCI") else "-1",
            "pcs_pts": rider.get("Pnt") if rider.get("Pnt") else "-1",
            "race_id": race_stats.get("race_id"),#"R" + hashlib.md5(f"{race_stats.get("name")}_{race_stats.get("year")}_{race_stats.get("stage")}".encode()).hexdigest()[:8],
        })

    return new_results
        

def get_race_name_info(race_url: str) -> dict:
    race_name, race_year, extension = race_url.split("/")[-3:]
    if extension == "result":
        return {
            "name": race_name,
            "year": race_year,
        }
    else:
        return {
            "name": race_name,
            "year": race_year,
            "stage": extension.split("-")[1]
        }


async def get_all_results(k = -1, s = -1) -> list[dict]:
    """fetches data and filters to results and stats
    
    k: nr of races to fetch
    s: nr of results for each race to fetch"""
    
    races_urls = await fetch_all_races() #534 races
    race_datas = []
    async with httpx.AsyncClient(headers=HEADERS, timeout=30.0) as client:

        # for race_url in races_urls[0:2]: 
        index = 0
        while (k == -1 or len(race_datas) < k) and index < len(races_urls):
            race_url = races_urls[index]

            last_part = race_url.split("/")[-1]
            if last_part == "result" or "stage" in last_part:
                print(f"getting {race_url}")
                try:
                    race_profile_url = await get_race_profile_url(race_url)
                    profile_resp = await client.get(race_profile_url)
                    
                    resp = await client.get(race_url)

                    race_data = parse_race_page(resp.text, profile_resp.text, race_url)
                    if not race_data:
                        continue
                    race_data["stats"] = filter_stats(race_data["stats"], race_url=race_url)
                    if s != -1:
                        race_data["results"] = filter_results(race_data["results"], race_data["stats"])[:s]
                    else:
                        race_data["results"] = filter_results(race_data["results"], race_data["stats"])
                    # pprint(race_data)
                    race_datas.append(race_data)
                except Exception as e:
                    print(f"Error occured: {e} for {race_url}")
                    
                time.sleep(1)

            index += 1

    return race_datas

async def get_results_one_race(race_url) -> dict:
    # race_url_stage = "https://www.procyclingstats.com/race/tour-de-france/2024/stage-2"
    # race_url_normal = ""#

    last_part = race_url.split("/")[-1]
    if last_part == "result" or "stage" in last_part:
        async with httpx.AsyncClient(headers=HEADERS, timeout=30.0) as client:
            race_profile_url = await get_race_profile_url(race_url)
            profile_resp = await client.get(race_profile_url)
            
            resp = await client.get(race_url)
            race_data = parse_race_page(resp.text, profile_resp.text, race_url)

            # pprint(race_data)
            if race_data:
                race_data["stats"] = filter_stats(race_data["stats"], race_url=race_url)
                race_data["results"] = filter_results(race_data["results"], race_data["stats"])[:25]
                pprint(race_data)
    return race_data

async def main():
    # await get_all_results()
    pl.Config(tbl_cols=-1)
    await get_results_one_race("https://www.procyclingstats.com/race/tour-de-france/2024/stage-2")
    await get_results_one_race("https://www.procyclingstats.com/race/tour-de-france/2024/stage-5")

if __name__ == "__main__":
    asyncio.run(main())