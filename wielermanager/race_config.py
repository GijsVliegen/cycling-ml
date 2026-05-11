import json
import re
from pathlib import Path

from data_engineering.selenium_webscraping import load_soups_from_http
from data_engineering.soup_parsing_functions import parse_gc_page

BASE_URL = "https://www.procyclingstats.com"
DEFAULT_RULES_PATH = Path("wielermanager/WIELERMANAGER_RULES.json")
DEFAULT_MANIFEST_RELATIVE_PATH = Path("wielermanager/races_manifest.json")
DEFAULT_YEAR = 2026


RaceTuple = tuple[str, int, str]


def load_wielermanager_rules(rules_path: Path | str = DEFAULT_RULES_PATH) -> dict:
    with open(rules_path, encoding="utf-8") as file_handle:
        return json.load(file_handle)


def race_manifest_path(data_dir: str) -> Path:
    return Path(data_dir) / DEFAULT_MANIFEST_RELATIVE_PATH


def load_race_manifest(data_dir: str) -> list[RaceTuple] | None:
    manifest_path = race_manifest_path(data_dir)
    if not manifest_path.exists():
        return None

    with open(manifest_path, encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    return [
        (item["race_key"], int(item["stage"]), item["race_type"])
        for item in payload.get("races", [])
    ]


def write_race_manifest(data_dir: str, races: list[RaceTuple]) -> Path:
    manifest_path = race_manifest_path(data_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "races": [
            {
                "race_key": race_key,
                "stage": stage,
                "race_type": race_type,
            }
            for race_key, stage, race_type in races
        ]
    }
    with open(manifest_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=4)
    return manifest_path


def get_base_race_name(race_key: str) -> str:
    return race_key.split("__", 1)[0]


def build_stage_race_key(pcs_name: str, stage: int) -> str:
    if stage == 0:
        return f"{pcs_name}__prologue"
    if stage > 0:
        return f"{pcs_name}__stage_{stage}"
    return pcs_name


def _extract_stage_number_from_tail(tail: str) -> int | None:
    if tail == "prologue":
        return 0
    if tail == "result":
        return -1

    match = re.match(r"stage-(\d+)", tail)
    if not match:
        return None
    return int(match.group(1))


def expand_stage_race_entry(race_entry: dict, default_year: int = DEFAULT_YEAR) -> list[RaceTuple]:
    pcs_name = race_entry["pcs_name"]
    race_type = race_entry["type"]
    year = int(race_entry.get("year", default_year))

    gc_url = f"{BASE_URL}/race/{pcs_name}/{year}"
    gc_soup = load_soups_from_http(gc_url)[0]
    stage_urls = parse_gc_page(gc_soup)

    expanded_races: list[RaceTuple] = []
    for stage_url in stage_urls:
        tail = stage_url.rstrip("/").split("/")[-1]
        stage = _extract_stage_number_from_tail(tail)
        if stage is None:
            continue
        expanded_races.append((build_stage_race_key(pcs_name, stage), stage, race_type))

    expanded_races.sort(key=lambda item: (999 if item[1] < 0 else item[1], item[0]))
    return expanded_races


def resolve_races_to_predict(rules: dict, default_year: int = DEFAULT_YEAR) -> list[RaceTuple]:
    resolved_races: list[RaceTuple] = []
    for race_entry in rules.get("races_voorjaar", []) + rules.get("races", []):
        if race_entry.get("load_all_stages"):
            resolved_races.extend(expand_stage_race_entry(race_entry, default_year=default_year))
        else:
            resolved_races.append((race_entry["pcs_name"], -1, race_entry["type"]))
    return resolved_races
