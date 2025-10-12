import asyncio
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm.asyncio import tqdm
import polars as pl
import re

BASE_URL = "https://www.procyclingstats.com"

# ---- Config ----
START_YEAR = 2020 #goes back to pre-1940
END_YEAR = 2024
CONCURRENT_REQUESTS = 10
HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---- Race Indexing ----
async def fetch_races_for_year(client, year):
    url = f"{BASE_URL}/races.php?year={year}"#&class=1.UWT" #which classes are available on pcs? 
    resp = await client.get(url)
    soup = BeautifulSoup(resp.text, "lxml")
    links = soup.select("a")
    races = [urljoin(BASE_URL, a["href"]) for a in links if a.has_attr("href") and a["href"].startswith("race/")]

    for race_url in races:
        if race_url.split("/")[-1] == "gc":
            base_race_url = race_url[:-3]
            resp = await client.get(base_race_url)
            soup = BeautifulSoup(resp.text, "lxml")
            stages_header = soup.find("h4", string="Stages")
            if stages_header:
                table = stages_header.find_next("table")
                stage_links = [base_race_url + "/" +  a["href"].split("/")[-1] for a in table.find_all("a", href=True)]
                races.extend(stage_links)
            else:
                print(f"Error: Stages header not found for race {base_race_url}")

            # races.extend([
            #     race_url[:-3] + "stage_{i}"
            #     for i in range(23) ###to many stages
            # ])
        else:
            pass
            # races.append(race_url)
    races = [
        race 
        for race in races
        if "stage" in race or "result" in race
    ]

    return list(set(races))

def get_race_statistics(soup):
    """Retrieves 'Race information' table from race/results page
    
    input: soup from url like 
        - one-day-race: https://www.procyclingstats.com/race/paris-roubaix/2025/result
        - stage-race: https://www.procyclingstats.com/race/tour-de-france/2024/stage-2
    output:
        'Arrival': 'Ans',
        'Avg. speed winner': '38.7 km/h',
        'Avg. temperature': '',
        'Classification': '1.UWT',
        'Datename': '24 April 2016',
        'Departure': 'Liège',
        'Distance': '248 km',
        'Parcours type': '',
        'Points scale': '1.WT.A',
        'ProfileScore': '182',
        'Race category': 'ME - Men Elite',
        'Race ranking': 'n/a',
        'Start time': '11:15',
        'Startlist quality score': '1075',
        'UCI scale': 'UCI.WR.C1',
        'Vertical meters': '4015',
        'Won how': 'Sprint of small group'
        
        'Timelimit': '14%, or 5:23:25 (+0:39:43)',   -> only for stage races
        
    """

    #TODO: startlist quality score in stage races drops over time due to dropouts, but considering the tiredness of peloton increases, ignore this for now.

    h4 = soup.find("h4", string="Race information")

    # Step 2: Find the next <ul> with the desired class after the h4
    ul = h4.find_next("ul", class_="list keyvalueList lineh16 fs12")
    # Step 3: Extract the <li> items into a dictionary
    race_info = {}

    for li in ul.find_all("li"):
        title_div = li.find("div", class_="title")
        value_div = li.find("div", class_="value")
        if title_div and value_div:
            key = title_div.get_text(strip=True).rstrip(":")
            value = value_div.get_text(strip=True)
            race_info[key] = value

    return race_info

def get_race_results(soup):
    """Retrieves 'Results' table from race/results page
    
    input: soup from url like 
        - one-day-race: https://www.procyclingstats.com/race/paris-roubaix/2025/result
        - stage-race: https://www.procyclingstats.com/race/tour-de-france/2024/stage-2
    output: \
        list in order, containing the dicts
        {
            'Age': '26',
            'BIB': '26',
            'H2H': '✔️',
            'Pnt': '100',
            'Rider': 'VermeerschFlorian',
            'Rnk': '5',
            'Specialty': 'Classic',
            'Team': 'UAE Team Emirates - XRG',
            'Time': ',,',
            'UCI': '360',
            'racer_url_index': 'rider/florian-vermeersch'
        }
    """

    results = []

    # Step 1: Find the results table
    table = soup.find("table", class_="results")
    headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]

    # Step 2: Go through each <tr> in the tbody
    for row in table.find("tbody").find_all("tr"):
        row_data_points = []
        racer_url = ""

        for cell in row.find_all("td"):
            # Handle Rider column (name is in <a> tags)
            if cell.find("a") and "ridername" in cell.get("class", []):
                rider_name = cell.find("a")
                full_name = rider_name.get_text(strip=True)
                row_data_points.append(full_name)
                racer_url = rider_name["href"]

            # Handle Team column
            elif cell.find("a") and "cu600" in cell.get("class", []):
                row_data_points.append(cell.get_text(strip=True))

            # Handle Specialty column (text is in <span>)
            elif "specialty" in cell.get("class", []):
                span = cell.find("span")
                row_data_points.append(span.get_text(strip=True) if span else "")

            # H2H checkbox (no text)
            elif "h2h" in cell.get("class", []):
                row_data_points.append("✔️" if cell.find("input", {"type": "checkbox"}) else "")

            # Default: just get text content
            else:
                row_data_points.append(cell.get_text(strip=True))

        if "relegated" in row_data_points[0]:
            #filter out relegation messages
            continue
        if len(headers) != len(row_data_points):
            print(f"check data for {full_name}")
        row_data_dict = {
            key: val for (key, val) in zip(headers, row_data_points, strict=True)
        } | {
            "racer_url_index": racer_url
        }
        results.append(row_data_dict)

    return results

async def get_race_profile_url(race_url):
    return "/".join(race_url.split("/")[:-1]) + "/route/stage-profiles"

#TODO: add dynamic programming aka logging

logs = {} #(race_name, race_year: profile_scores)
def get_race_profile(soup, race_name, year, stage_nr = 1): #if no stage race -> stage 1
    global logs
    if (race_name, year, stage_nr) in logs:
        return logs[(race_name, year, stage_nr)]
    
    stage_data = {}
    stages_header = soup.find("h2", string="All stage profiles")
    stage_num = 1
    try:
        if stages_header:
            ul = stages_header.find_next("ul", class_="list dashed pad4 keyvalueList")
            if ul:
                items = ul.find_all("li")
                i = 0
                while i < len(items):
                    title_div = items[i].find("div", class_="title")
                    if title_div and "Stage:" in title_div.text:

                        # Initialize data
                        profile_score = None
                        ps_25k = None

                        # Find next few entries (up to 5 ahead)
                        for j in range(i + 1, min(i + 5, len(items))):
                            title = items[j].find("div", class_="title")
                            value = items[j].find("div", class_="value")
                            if not title or not value:
                                continue
                            if "ProfileScore" in title.text:
                                profile_score = int(value.text.strip())
                            elif "PS final 25k" in title.text:
                                ps_25k = int(value.text.strip())

                        stage_data[stage_num] = {
                            "ProfileScore": profile_score,
                            "PS_final_25k": ps_25k,
                        }
                        stage_num += 1
                    i += 1
        else:
            print("Stages header not found")
    except Exception as e:
        print(f"Encountered error {e} for {race_name}, {year}, {stage_nr}")
        return {
            "ProfileScore": -1,
            "PS_final_25k": -1,
        }
        
    if stage_nr > len(stage_data) - 1:
        # raise IndexError #sometimes just missing data
        return {
            "ProfileScore": -1,
            "PS_final_25k": -1,
        }
    logs |= {
        (race_name, year, i): profile
        for i, profile in stage_data.items()
    }  
    return stage_data[stage_nr]

# ---- Race Parsing ----
def parse_race_page(html, profile_html, race_url) -> dict:

     
    race_name, year, last_part = race_url.split("/")[-3:]
    if "stage" in last_part:
        stage_nr = int(last_part.split("-")[-1])
    else:
        stage_nr = 1

    soup = BeautifulSoup(html, "lxml")
    profile_soup = BeautifulSoup(profile_html, "lxml")
    h4 = soup.find("h4", string="Race information")
    if not h4:
        print(f"Error: no information for {str(soup)}")
        return None

    try:
        profile = get_race_profile(profile_soup, race_name, year, stage_nr)
    except IndexError:
        raise IndexError("on stage_nr")

    return {
        "stats": get_race_statistics(soup) | profile,
        "results": get_race_results(soup)
    }


async def fetch_all_races() -> list:
    async with httpx.AsyncClient(headers=HEADERS, timeout=30.0) as client:
        all_races = []
        for year in range(START_YEAR, END_YEAR + 1):
            races = await fetch_races_for_year(client, year)
            all_races.extend(races)
        return list(set(all_races))

# ---- Run ----
async def main():
    print("Fetching race URLs...")
    race_urls = await fetch_all_races()

    print(f"Found {len(race_urls)} races")

    print(race_urls[:10])


if __name__ == "__main__":
    asyncio.run(main())
