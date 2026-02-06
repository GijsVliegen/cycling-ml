from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
import time
import re
import os
# --- Brave binary path (Ubuntu) ---
BRAVE_PATH = "/usr/bin/brave-browser"

options = Options()
options.binary_location = BRAVE_PATH
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--start-maximized")

html_storage_dir = "html_cache"
driver = webdriver.Chrome(service=Service(), options=options) #TODO:outcomment if no webscraping is done?

def url_to_filename(url: str) -> str:
    """
    Convert URL tail to a filesystem-safe filename
    """
    tail = url.split("procyclingstats.com/")[-1]  # keep meaningful parts
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", tail)
    return f"{name}.html"

def download_page(url: str, output_dir=html_storage_dir) -> str:
    global driver
    os.makedirs(output_dir, exist_ok=True)
    filename = url_to_filename(url)
    filepath = os.path.join(output_dir, filename)

    #skip if already exists
    if os.path.exists(filepath):
        log_messages = [f"File already exists, skipping download: {filepath}"]
        return filepath,log_messages
    
    driver.get(url)
    try:
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        html = driver.page_source

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        log_messages = [f"Error downloading page {url}: {e}"]
        return None, log_messages
    finally:
        time.sleep(1) 

        driver.delete_all_cookies()
        driver.execute_script("window.localStorage.clear(); window.sessionStorage.clear();")
        time.sleep(1) #be nice
        driver.close()
        driver.quit()
        driver = webdriver.Chrome(service=Service(), options=options)
        time.sleep(1) #be nice

    return filepath, []

def load_soups_from_http(url: str | list[str]) -> list[BeautifulSoup]:
    global driver
    # print(f"Loading page: {url}")
    soups = []
    if isinstance(url, str):
        url = [url]
    for single_url in url:
        driver.get(single_url)
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        soup = BeautifulSoup(driver.page_source, "lxml")
        soups.add(soup)
    # print(f"Souped page: {url}")
    # driver.close()
    # driver.switch_to.new_window("tab")
        driver.delete_all_cookies()
        driver.execute_script("window.localStorage.clear(); window.sessionStorage.clear();")
        time.sleep(1) #be nice
        driver.close()
        driver.quit()
        driver = webdriver.Chrome(service=Service(), options=options)
        time.sleep(1) #be nice
        # print(f"closed driver for page: {url}")
    return soups

def load_soup_from_file(url: str, output_dir=html_storage_dir) -> BeautifulSoup:
    filename = url_to_filename(url)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return BeautifulSoup(f.read(), "lxml")

# driver = webdriver.Chrome(service=Service(), options=options)

# url = "https://www.procyclingstats.com/rider/tadej-pogacar"
# driver.get(url)

# # Wait until page content is loaded
# WebDriverWait(driver, 15).until(
#     EC.presence_of_element_located((By.TAG_NAME, "body"))
# )

# # --- Parse HTML ---
# soup = BeautifulSoup(driver.page_source, "lxml")

# # -------------------------
# # Specialties
# # -------------------------
# specialties = {}

# ul = soup.find("ul", class_="pps list")
# if ul:
#     for li in ul.find_all("li"):
#         xtitle = li.find("div", class_="xtitle")
#         xvalue = li.find("div", class_="xvalue")
#         if xtitle and xvalue:
#             spec = xtitle.get_text(strip=True)
#             score = int(xvalue.get_text(strip=True))
#             specialties[spec] = score

# # -------------------------
# # Rankings
# # -------------------------
# rankings = {}

# h4 = soup.find("h4", string="PCS Ranking position per season")
# if h4:
#     table = h4.find_next("table")
#     if table:
#         tbody = table.find("tbody")
#         for tr in tbody.find_all("tr"):
#             tds = tr.find_all("td")
#             if len(tds) >= 3:
#                 year = tds[0].get_text(strip=True)

#                 score_div = tds[1].find("div", class_="title")
#                 score = int(score_div.get_text(strip=True)) if score_div else 0

#                 rank = int(tds[2].get_text(strip=True))

#                 rankings[year] = {
#                     "score": score,
#                     "rank": rank,
#                 }

# # -------------------------
# # Final merged output
# # -------------------------
# rider_name = soup.find("h1").get_text(strip=True)

# result = [
#     {"name": rider_name, "year": year} | yearly_ranking | specialties
#     for year, yearly_ranking in rankings.items()
# ]
# print(result)
# print("\n")

# time.sleep(2)
# driver.close()
# driver.quit()

# time.sleep(2)

# # # -------------------------------------------------
# # # Statistics by season table

# # # very sketchy generated by ChatGPT, without actually knowing the html
# # # -------------------------------------------------

# driver = webdriver.Chrome(service=Service(), options=options)
# url = "https://www.procyclingstats.com/rider/tadej-pogacar/statistics/season-statistics"
# driver.get(url)

# # Wait for tables to be present #------ 2 second delay! -------
# WebDriverWait(driver, 5).until( 
#     EC.presence_of_element_located((By.TAG_NAME, "table"))
# )

# soup = BeautifulSoup(driver.page_source, "lxml")

# stats_by_season = []


# h3 = soup.find(
#     lambda tag: tag.name in ("h2", "h3", "h4")
#     and "Statistics by season" in tag.get_text(strip=True)
# )

# if not h3:
#     raise RuntimeError("Could not find 'Statistics by season' header")

# table = h3.find_next("table")
# if not table:
#     raise RuntimeError("Statistics by season table not found")

# tbody = table.find("tbody")
# for tr in tbody.find_all("tr"):
#     tds = [td.get_text(strip=True) for td in tr.find_all("td")]

#     if len(tds) >= 7:
#         stats_by_season.append({
#             "season": tds[0],
#             "points": int(tds[1].replace(",", "")) if tds[1] != "-" else -1,
#             "racedays": int(tds[2]) if tds[2] != "-" else -1,
#             "kms": int(tds[3].replace(",", "")) if tds[3] != "-" else -1,
#             "wins": int(tds[4]) if tds[4] != "-" else -1,
#             "top_3s": int(tds[5]) if tds[5] != "-" else -1,
#             "top_10s": int(tds[6]) if tds[6] != "-" else -1,
#         })



# time.sleep(2)
# driver.close()
# driver.quit()
# print(stats_by_season)
