import asyncio
from http import client
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

# ---- Race Indexing ---

def parse_gc_page(soup) -> list[str]:
    """
    Example page: https://www.procyclingstats.com/race/tour-down-under/2025

    returns list of all urls to stages of this gc-race
    e.g [
        https://www.procyclingstats.com/race/tour-down-under/2025/stage-1,
        https://www.procyclingstats.com/race/tour-down-under/2025/stage-2,
        ...
    ]
    """
    stages_header = soup.find("h4", string="Stages")
    if stages_header:
        table = stages_header.find_next("table")
        stage_urls = [urljoin(BASE_URL, a["href"]) for a in table.find_all("a", href=True)]
    else:
        print(f"Error: Stages header not found for gc race")
        return []
    return stage_urls

def parse_calendar_page(soup) -> tuple[list[str], list[str]]:
    """
    Example page: https://www.procyclingstats.com/races.php?year=2025

    returns list of all urls to races that year
    e.g [
        https://www.procyclingstats.com/race/great-ocean-road-race/2025/result
    ],
    [
        https://www.procyclingstats.com/race/tour-down-under/2025
    ]
    """
    links = soup.select("a")
    race_urls = [urljoin(BASE_URL, a["href"]) for a in links if a.has_attr("href") and a["href"].startswith("race/")]
    gc_races = []

    for i in range(len(race_urls)-1, -1, -1):
        race_url = race_urls[i]
        if race_url.split("/")[-1] == "gc":
            base_race_url = "/".join(race_url.split("/")[:-1])
            gc_races.append(base_race_url)
            race_urls.pop(i)
    

    return [a for a in race_urls if a.endswith("result")], gc_races

# async def fetch_races_for_year(client, year):
#     """
#     Example page: https://www.procyclingstats.com/races.php?year=2025


#     """
#     url = f"{BASE_URL}/races.php?year={year}"#&class=1.UWT" #which classes are available on pcs? 
#     resp = await client.get(url)
#     soup = BeautifulSoup(resp.text, "lxml")
    

def parse_race_page(soup) -> dict:
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

def parse_race_result_page(soup) -> list[dict]:
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
    try:
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

        return results, []
    except Exception as e:
        return [], [f"Exception while parsing race results: {e}"]

def get_race_profile_url(race_url):
    return "/".join(race_url.split("/")[:-1]) + "/route/stage-profiles"


race_profile_logs = {} #(race_name, race_year: profile_scores)
def parse_race_profile_page(soup, race_name, year, stage_nr = 1): #if no stage race -> stage 1
    """Parse race profile page and return logs
    """
    global race_profile_logs
    if (race_name, year, stage_nr) in race_profile_logs:
        return race_profile_logs[(race_name, year, stage_nr)]
    
    stage_data = {}
    stages_header = soup.find("h2", string="All stage profiles")
    stage_num = 1
    try:
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
        return stage_data[stage_nr], []
    except Exception as e:
        return {
            "ProfileScore": -1,
            "PS_final_25k": -1,
        }, [f"Exception for {race_name}, {year}, {stage_nr}: {e}"]
        


def url_to_race_info(race_url) -> dict:
    """
    page example: https://www.procyclingstats.com/race/tour-de-france/2025/stage-1
    returns {
        "race_name": "tour-de-france",
        "year": "2025",
    }

    """
    race_name, year, _ = race_url.split("/")[-3:]
    return {
        "race_name": race_name,
        "year": year,
    }

def parse_rider_statistics_page(soup) -> list[dict]:
    """
    Page example: https://www.procyclingstats.com/rider/tadej-pogacar/statistics/season-statistics
    Results example: [
        {
            'season': '2023',
            'points': 2730,
            'racedays': 65,
            'kms': 11234,
            'wins': 11,
            'top_3s': 25,
            'top_10s': 40
        },
        ...
    ]
    """
    stats_by_season = []
    h3 = soup.find(
        lambda tag: tag.name in ("h2", "h3", "h4")
        and "Statistics by season" in tag.get_text(strip=True)
    )

    if not h3:
        raise RuntimeError("Could not find 'Statistics by season' header")
        

    table = h3.find_next("table")
    if not table:
        raise RuntimeError("Statistics by season table not found")

    tbody = table.find("tbody")
    for tr in tbody.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]

        if len(tds) >= 7:
            stats_by_season.append({
                "season": tds[0],
                "points": int(tds[1].replace(",", "")) if tds[1] != "-" else -1,
                "racedays": int(tds[2]) if tds[2] != "-" else -1,
                "kms": int(tds[3].replace(",", "")) if tds[3] != "-" else -1,
                "wins": int(tds[4]) if tds[4] != "-" else -1,
                "top_3s": int(tds[5]) if tds[5] != "-" else -1,
                "top_10s": int(tds[6]) if tds[6] != "-" else -1,
            })
    return stats_by_season

def parse_rider_page(soup) -> dict:
    """
    Page example: https://www.procyclingstats.com/rider/tadej-pogacar
    Results example: {
        'Onedayraces': 8733, 
        'GC': 7344, 
        'TT': 3227, 
        'Sprint': 222, 
        'Climber': 9559, 
        'Hills': 4118
    }
    """
    # Specialties
    specialties = {}
    ul = soup.find("ul", class_="pps list")
    if ul:
        for li in ul.find_all("li"):
            xtitle = li.find("div", class_="xtitle")
            xvalue = li.find("div", class_="xvalue")
            if xtitle and xvalue:
                spec = xtitle.get_text(strip=True)
                score = int(xvalue.get_text(strip=True))
                specialties[spec] = score
        return specialties  
    else:
        raise RuntimeError("Could not find specialties list")

def parse_startlist_page(soup) -> list[dict]:
    """
    Page example: 
    https://www.procyclingstats.com/race/alula-tour/2026/startlist
    """
    riders = []

    # Each <li> under startlist_v4 is a team block
    for team_li in soup.select("ul.startlist_v4 > li"):
        # Get the team slug from the <a class="team">
        team_tag = team_li.select_one(".ridersCont > div > a.team")
        if not team_tag or "href" not in team_tag.attrs:
            continue
        team_href = team_tag["href"]
        if not team_href.startswith("team/"):
            continue
        team_slug = team_href.split("team/")[1].strip()

        # Loop over each rider in this team
        for rider_li in team_li.select("ul li"):
            rider_a = rider_li.find("a")
            if rider_a and "href" in rider_a.attrs:
                href = rider_a["href"]
                if href.startswith("rider/"):
                    rider_slug = href.split("rider/")[1].strip()
                    riders.append({
                        "rider": rider_slug,
                        "team": team_slug
                    })

    return riders



# ---- Run ----
async def main():
    print("Fetching race URLs...")
    # race_urls = await fetch_all_races()

    # print(f"Found {len(race_urls)} races")

    # print(race_urls[:10])


if __name__ == "__main__":
    asyncio.run(main())
