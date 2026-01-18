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

from soup_parsing_functions import *

test_race_url = "https://www.procyclingstats.com/race/liege-bastogne-liege/2016/result"

test_stage_race_url = "https://www.procyclingstats.com/race/tour-de-france/2024/stage-2"

@pytest.mark.asyncio
async def test_fetch_races_for_year():
    async with httpx.AsyncClient(headers=HEADERS, timeout=30.0) as client:
        races = await fetch_races_for_year(client, 2024)
    assert races != []


@pytest.mark.asyncio
async def test_fetch_races():
    races = await fetch_all_races()
    print(races)
    assert races != []

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "race_url",
    [test_race_url, test_stage_race_url]
)
async def test_parse_race_page(race_url):
    async with httpx.AsyncClient(headers=HEADERS, timeout=30.0) as client:
        resp = await client.get(race_url)
    race_data = parse_race_page(resp.text, race_url)
    stats = race_data["stats"]
    results = race_data["results"]
    assert stats["Datename"] != ""
    assert len(stats.keys()) in {17, 18},  print(f"nr of stats = {len(stats.keys())}")
    assert len(results > 50)
    assert len(results[0].keys() == 11)