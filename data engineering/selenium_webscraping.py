from curl_cffi import requests
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
        print(f"skipped")
        log_messages = [f"File already exists, skipping download: {filepath}"]
        return filepath,log_messages
    
    session = requests.Session()

    response = session.get(
        url,
        impersonate="chrome120",
        headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://google.com",
        },
    )

    try:
        html = response.text

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        log_messages = [f"Error downloading page {url}: {e}"]
        return None, log_messages
    # finally:
    #     time.sleep(1) 

    #     driver.delete_all_cookies()
    #     driver.execute_script("window.localStorage.clear(); window.sessionStorage.clear();")
    #     time.sleep(1) #be nice
    #     driver.close()
    #     driver.quit()
    #     driver = webdriver.Chrome(service=Service(), options=options)
    #     time.sleep(1) #be nice

    return filepath, []

def load_soups_from_http(url: str | list[str]) -> list[BeautifulSoup]:
    global driver
    # print(f"Loading page: {url}")
    soups = []
    if isinstance(url, str):
        url = [url]
    for single_url in url:

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
        soups.add(soup)

        
    #     driver.get(single_url)
    #     WebDriverWait(driver, 5).until(
    #         EC.presence_of_element_located((By.TAG_NAME, "body"))
    #     )
    #     soup = BeautifulSoup(driver.page_source, "lxml")
    #     soups.add(soup)
    # # print(f"Souped page: {url}")
    # # driver.close()
    # # driver.switch_to.new_window("tab")
    #     driver.delete_all_cookies()
    #     driver.execute_script("window.localStorage.clear(); window.sessionStorage.clear();")
    #     time.sleep(1) #be nice
    #     driver.close()
    #     driver.quit()
    #     driver = webdriver.Chrome(service=Service(), options=options)
    #     time.sleep(1) #be nice
        # print(f"closed driver for page: {url}")
    return soups

def load_soup_from_file(url: str, output_dir=html_storage_dir) -> BeautifulSoup:
    filename = url_to_filename(url)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return BeautifulSoup(f.read(), "lxml")
