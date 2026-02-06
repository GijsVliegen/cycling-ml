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

from data_science_functions import *
from test import *
from test_data import raw_sample_data_cleaned
import polars as pl


stage_race_url = "https://www.procyclingstats.com/race/paris-nice/2025"

# def test_gc_parsing():
#     soup = load_soup_from_http(stage_race_url)
#     stage_urls = parse_gc_page(soup)
#     print(stage_urls)
from curl_cffi import requests
from bs4 import BeautifulSoup

def main():

    url = "https://www.procyclingstats.com/race/tour-de-france/2025"

    session = requests.Session()

    response = session.get(
        url,
        impersonate="chrome120",
        headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://google.com",
        },
    )

    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.select("a"):
        print(link.get("href"))


if __name__ == "__main__":
    main()