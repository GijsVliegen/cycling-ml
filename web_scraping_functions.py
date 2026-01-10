from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from bs4 import BeautifulSoup
from bs4.element import Tag
from urllib.parse import urljoin

from download_html_functions import (
    BASE_URL,
    fetch_html,
    fetch_race_index_html,
    fetch_rider_html,
)

# ---- Config ----
START_YEAR = 2020  # goes back to pre-1940
END_YEAR = 2024


# ---- Helpers ----
def _as_tag(node: Any) -> Tag | None:
    return node if isinstance(node, Tag) else None


def _is_stage_or_result(url: str) -> bool:
    last_part = url.split("/")[-1]
    return last_part == "result" or "stage" in last_part


def _extract_stage_links(base_race_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    stages_header = _as_tag(soup.find("h4", string="Stages"))
    if stages_header is None:
        print(f"Error: Stages header not found for race {base_race_url}")
        return []

    table = _as_tag(stages_header.find_next("table"))
    if table is None:
        return []

    stage_links: List[str] = []
    for anchor in table.find_all("a", href=True):
        anchor_tag = _as_tag(anchor)
        if anchor_tag is None:
            continue
        href = anchor_tag.get("href")
        if not isinstance(href, str):
            continue
        stage_suffix = href.split("/")[-1]
        stage_links.append(f"{base_race_url}/{stage_suffix}")
    return stage_links


# ---- Race Indexing ----
async def fetch_races_for_year(year: int, *, force: bool = False) -> List[str]:
    html = await fetch_race_index_html(year, force=force)
    soup = BeautifulSoup(html, "lxml")

    race_urls: set[str] = set()
    stage_base_urls: set[str] = set()

    for anchor in soup.select("a"):
        anchor_tag = _as_tag(anchor)
        if anchor_tag is None:
            continue
        href = anchor_tag.get("href")
        if not isinstance(href, str) or not href.startswith("race/"):
            continue
        race_url = urljoin(BASE_URL, href)
        race_urls.add(race_url)
        if race_url.endswith("/gc"):
            stage_base_urls.add(race_url[:-3])

    for base_race_url in stage_base_urls:
        base_html = await fetch_html(base_race_url, force=force)
        race_urls.update(_extract_stage_links(base_race_url, base_html))

    filtered_races = [url for url in race_urls if _is_stage_or_result(url)]
    return sorted(filtered_races)


def get_race_statistics(soup: BeautifulSoup) -> Dict[str, str]:
    """Retrieves 'Race information' table from race/results page"""

    h4 = _as_tag(soup.find("h4", string="Race information"))
    if h4 is None:
        return {}

    ul = _as_tag(h4.find_next("ul", class_="list keyvalueList lineh16 fs12"))
    if ul is None:
        return {}

    race_info: Dict[str, str] = {}
    for li in ul.find_all("li"):
        title_div = _as_tag(li.find("div", class_="title"))
        value_div = _as_tag(li.find("div", class_="value"))
        if title_div is None or value_div is None:
            continue
        key = title_div.get_text(strip=True).rstrip(":")
        value = value_div.get_text(strip=True)
        race_info[key] = value

    return race_info


def get_race_results(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """Retrieves 'Results' table from race/results page"""

    table = _as_tag(soup.find("table", class_="results"))
    if table is None:
        return []

    thead = _as_tag(table.find("thead"))
    tbody = _as_tag(table.find("tbody"))
    if thead is None or tbody is None:
        return []

    headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    results: List[Dict[str, str]] = []

    for row in tbody.find_all("tr"):
        row_tag = _as_tag(row)
        if row_tag is None:
            continue
        row_data_points: List[str] = []
        racer_url = ""
        full_name = ""

        for cell in row_tag.find_all("td"):
            cell_tag = _as_tag(cell)
            if cell_tag is None:
                continue
            classes = cell_tag.get("class") or []
            link = _as_tag(cell_tag.find("a"))

            if link and "ridername" in classes:
                full_name = link.get_text(strip=True)
                row_data_points.append(full_name)
                racer_url = link.get("href", "") or ""
            elif link and "cu600" in classes:
                row_data_points.append(cell_tag.get_text(strip=True))
            elif "specialty" in classes:
                span = _as_tag(cell_tag.find("span"))
                row_data_points.append(span.get_text(strip=True) if span else "")
            elif "h2h" in classes:
                has_checkbox = cell_tag.find("input", {"type": "checkbox"}) is not None
                row_data_points.append("✔️" if has_checkbox else "")
            else:
                row_data_points.append(cell_tag.get_text(strip=True))

        if not row_data_points:
            continue
        if "relegated" in row_data_points[0]:
            continue
        if len(headers) != len(row_data_points):
            print("check data for row", row_data_points)
            continue

        row_data_dict = {
            key: val for (key, val) in zip(headers, row_data_points, strict=True)
        }
        row_data_dict["racer_url_index"] = racer_url
        results.append(row_data_dict)

    return results


async def get_race_profile_url(race_url: str) -> str:
    return "/".join(race_url.split("/")[:-1]) + "/route/stage-profiles"


logs: Dict[tuple[str, str, int], Dict[str, int]] = {}


def get_race_profile(soup: BeautifulSoup, race_name: str, year: str, stage_nr: int = 1) -> Dict[str, int]:
    global logs
    if (race_name, year, stage_nr) in logs:
        return logs[(race_name, year, stage_nr)]

    stage_data: Dict[int, Dict[str, int]] = {}
    stages_header = _as_tag(soup.find("h2", string="All stage profiles"))
    if stages_header is None:
        print("Stages header not found")
        return {"ProfileScore": -1, "PS_final_25k": -1}

    ul = _as_tag(stages_header.find_next("ul", class_="list dashed pad4 keyvalueList"))
    if ul is None:
        return {"ProfileScore": -1, "PS_final_25k": -1}

    items = ul.find_all("li")
    stage_num = 1
    i = 0
    while i < len(items):
        title_div = _as_tag(items[i].find("div", class_="title"))
        if title_div and "Stage:" in title_div.text:
            profile_score: int | None = None
            ps_25k: int | None = None
            for j in range(i + 1, min(i + 5, len(items))):
                title = _as_tag(items[j].find("div", class_="title"))
                value = _as_tag(items[j].find("div", class_="value"))
                if title is None or value is None:
                    continue
                text = title.text
                val = value.text.strip()
                if "ProfileScore" in text:
                    profile_score = int(val)
                elif "PS final 25k" in text:
                    ps_25k = int(val)
            stage_data[stage_num] = {
                "ProfileScore": profile_score or -1,
                "PS_final_25k": ps_25k or -1,
            }
            stage_num += 1
        i += 1

    if stage_nr not in stage_data:
        return {"ProfileScore": -1, "PS_final_25k": -1}

    for idx, profile in stage_data.items():
        logs[(race_name, year, idx)] = profile

    return stage_data[stage_nr]


# ---- Race Parsing ----
def parse_race_page(html: str, profile_html: str, race_url: str) -> Dict[str, List[Dict[str, str]]]:
    race_name, year, last_part = race_url.split("/")[-3:]
    stage_nr = int(last_part.split("-")[-1]) if "stage" in last_part else 1

    soup = BeautifulSoup(html, "lxml")
    profile_soup = BeautifulSoup(profile_html, "lxml")

    stats = get_race_statistics(soup)
    if not stats:
        print(f"Error: no information for {race_url}")
        return {}

    profile = get_race_profile(profile_soup, race_name, year, stage_nr)

    return {
        "stats": stats | profile,
        "results": get_race_results(soup),
    }


async def scrape_rider_data(rider_name: str, *, force: bool = False) -> List[Dict[str, int]]:
    """Scrape data for a single rider."""

    html = await fetch_rider_html(rider_name, force=force)
    soup = BeautifulSoup(html, "lxml")

    specialties: Dict[str, int] = {}
    ul = _as_tag(soup.find("ul", class_="pps list"))
    if ul:
        for li in ul.find_all("li"):
            xtitle = _as_tag(li.find("div", class_="xtitle"))
            xvalue = _as_tag(li.find("div", class_="xvalue"))
            if xtitle is None or xvalue is None:
                continue
            spec = xtitle.get_text(strip=True)
            score = int(xvalue.get_text(strip=True))
            specialties[spec] = score

    rankings: Dict[str, Dict[str, int]] = {}
    h4 = _as_tag(soup.find("h4", string="PCS Ranking position per season"))
    if h4:
        table = _as_tag(h4.find_next("table"))
        tbody = _as_tag(table.find("tbody")) if table else None
        if tbody:
            for tr in tbody.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 3:
                    continue
                year = tds[0].get_text(strip=True)
                score_div = _as_tag(tds[1].find("div", class_="title"))
                score = int(score_div.get_text(strip=True)) if score_div else 0
                rank = int(tds[2].get_text(strip=True))
                rankings[year] = {
                    "score": score,
                    "rank": rank,
                }

    return [
        {"name": rider_name, "year": year} | yearly_ranking | specialties
        for year, yearly_ranking in rankings.items()
    ]


async def fetch_all_riders(
    rider_names: List[str],
    *,
    force: bool = False,
    concurrency: int = 5,
) -> List[Dict[str, int]]:
    async def _scrape_single(name: str) -> List[Dict[str, int]]:
        return await scrape_rider_data(name, force=force)

    results: List[Dict[str, int]] = []
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _bounded(name: str) -> List[Dict[str, int]]:
        async with semaphore:
            return await _scrape_single(name)

    tasks = [_bounded(name) for name in rider_names]
    for task in tqdm(asyncio.as_completed(tasks), total=len(rider_names)):
        rider_data = await task
        results.extend(rider_data)
    return results


async def fetch_all_races(force: bool = False) -> List[str]:
    all_races: List[str] = []
    for year in range(START_YEAR, END_YEAR + 1):
        races = await fetch_races_for_year(year, force=force)
        all_races.extend(races)
    return sorted(set(all_races))


# ---- Run ----
async def main() -> None:
    print("Fetching race URLs...")
    race_urls = await fetch_all_races()
    print(f"Found {len(race_urls)} races")
    print(race_urls[:10])


if __name__ == "__main__":
    asyncio.run(main())
