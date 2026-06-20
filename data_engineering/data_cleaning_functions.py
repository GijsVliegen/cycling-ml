import asyncio
import polars as pl
from pprint import pprint
from datetime import datetime
import hashlib



def clean_won_how(won_how: str | None) -> str:
    if won_how in {None, "", "-", "Other"}:
        return "unknown"
    if won_how == "Time trial":
        return "time_trial"
    if won_how == "Sprint à deux":
        return "duo_sprint"
    if won_how.endswith("km solo"):
        return "solo"
    if won_how == "Sprint of small group":
        return "small_sprint"
    if won_how == "Sprint of large group":
        return "large_sprint"
    if won_how.startswith("Sprint of ") and won_how.endswith(" riders"):
        try:
            rider_count = int(won_how.removeprefix("Sprint of ").removesuffix(" riders"))
            return "small_sprint" if rider_count < 10 else "large_sprint"
        except ValueError:
            return "unknown"
    return "unknown"


def filter_stats(stats: dict, race_url) -> dict:
    # stats = {
    #     "startlist_score": stats["Startlist quality score"],
    #     "final_km_percentage": stats["Gradient final km"],
    #     "heigh_meters": stats["Vertical meters"],
    #     "date": , #or Datename aswell sometimes?
    #     "distance": ,
    #     "classification": stats["Classification"],
    #     "won_how": stats["Won how"],
    #     "speed": stats["Avg. speed winner"],
    #     "temp": stats["Avg. temperature"],
    #     "profile_score": stats["ProfileScore"],
    #     "profile_score_last_25k": stats["PS_final_25k"],
    # }
    stats |= get_race_name_info(race_url)

    final_km_percentage_str = stats.get("Gradient final km", "-1%")
    dist = stats["Distance"].split(" ")[0]
    speed_str = stats.get("Avg. speed winner")
    startlist_score_str = stats.get("Startlist quality score")
    temp_str = stats.get("Avg. temperature")
    
    # Generate race_id: Hash of date+classification
    # race_id = "R" + hashlib.md5(f"{date_str}_{classification}".encode()).hexdigest()[:8]


    return {
        # "race_id": race_id,
        "race_id": "R" + hashlib.md5(f"{stats.get("name")}_{stats.get("year")}_{stats.get("stage")}".encode()).hexdigest()[:8],
        "date": datetime.strptime(stats["Date"], "%d %B %Y").strftime("%Y-%m-%d"),
        "name": stats.get("name"),
        "year": stats.get("year"),
        "stage": stats.get("stage"),
        "classification": stats["Classification"],
        "race_category": stats["Race category"],
        "final_km_percentage":  float(final_km_percentage_str[:-1]) if final_km_percentage_str != "" else -1 ,
        "distance_km": float(dist) if dist else -1,
        "elevation_m": stats.get("Vertical meters", "-1"),
        "avg_speed_kmh": speed_str.replace(" km/h", "") if speed_str else "-1",
        "startlist_score": startlist_score_str.split(" ")[1][1:-1] if " " in startlist_score_str else startlist_score_str, #in case of stage races, take current startlist score for that stage
        "won_how": stats.get("Won how"),
        "temp": temp_str.replace(" °C", "") if temp_str else "-1", #lets hope this is not a possible value
        "profile_score": stats.get("ProfileScore"),
        "profile_score_last_25k": stats.get("PS_final_25k"),
        "won_how_clean": clean_won_how(stats.get("Won how")),
        "climbs": stats.get("climbs", None),
    }


def filter_results(results: list[dict], race_stats: dict) -> list[dict]:
    seen_rider_ids = set()

    new_results = []
    for rider in results:
        
        rider_name = rider["racer_url_index"].split("/")[-1]
        if rider_name in seen_rider_ids:
            if rider_name != "":
                print(f"⚠️ Duplicate entry for rider '{rider_name}' in race {race_stats.get("name")}")
            continue
        seen_rider_ids.add(rider_name)

        team_value = rider.get("Team")
        if isinstance(team_value, str):
            team_value = team_value.strip()
        if team_value == "":
            team_value = None

        new_results.append({
            "rank": rider.get("Rnk") if (rider.get("Rnk") and rider.get("Rnk") not in {"DNF", "DNS", "OTL"}) else "-1",
            "name": rider_name,
            "specialty": rider["Specialty"],
            "team": team_value,
            "age": rider.get("Age") if rider.get("Age") else "-1",
            "uci_pts": rider.get("UCI") if rider.get("UCI") else "-1",
            "pcs_pts": rider.get("Pnt") if rider.get("Pnt") else "-1",
            "race_id": race_stats.get("race_id"),
            "breakaway_km": rider.get("breakaway_km") if rider.get("breakaway_km") else "-1",
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