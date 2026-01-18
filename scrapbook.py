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
from data_cleaning_functions import *
from data_structure_functions import *
from data_science_functions import *
from test import *
from test_data import raw_sample_data_cleaned
import polars as pl


stage_race_url = "https://www.procyclingstats.com/race/paris-nice/2025"

def test_gc_parsing():
    soup = load_soup_from_http(stage_race_url)
    stage_urls = parse_gc_page(soup)
    print(stage_urls)

def main():
    test_gc_parsing()


if __name__ == "__main__":
    main()