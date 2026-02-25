import asyncio
import polars as pl
from pprint import pprint
from datetime import datetime
import hashlib



def filter_stats(stats: dict, race_url) -> dict:
    stats = {
        "startlist_score": stats["Startlist quality score"],
        "final_km_percentage": stats["Gradient final km"],
        "heigh_meters": stats["Vertical meters"],
        "date": datetime.strptime(stats["Date"], "%d %B %Y"), #or Datename aswell sometimes?
        "distance": stats["Distance"].split(" ")[0],
        "classification": stats["Classification"],
        "won_how": stats["Won how"],
        "speed": stats["Avg. speed winner"],
        "temp": stats["Avg. temperature"],
        "profile_score": stats["ProfileScore"],
        "profile_score_last_25k": stats["PS_final_25k"],
    } | get_race_name_info(race_url)

    date_str = stats["date"].strftime("%Y-%m-%d")
    classification = stats["classification"]
    
    # Generate race_id: Hash of date+classification
    # race_id = "R" + hashlib.md5(f"{date_str}_{classification}".encode()).hexdigest()[:8]


    return {
        # "race_id": race_id,
        "race_id": "R" + hashlib.md5(f"{stats.get("name")}_{stats.get("year")}_{stats.get("stage")}".encode()).hexdigest()[:8],
        "date": date_str,
        "name": stats.get("name"),
        "year": stats.get("year"),
        "stage": stats.get("stage"),
        "classification": classification,
        "final_km_percentage": float(stats.get("final_km_percentage", "-1%")[:-1]) if stats.get("final_km_percentage") != "" else -1 ,
        "distance_km": stats.get("distance") if stats.get("distance") else "-1",
        "elevation_m": stats.get("heigh_meters") if stats.get("heigh_meters") else "-1",
        "avg_speed_kmh": stats.get("speed").replace(" km/h", "") if stats.get("speed") else "-1",
        "startlist_score": stats.get("startlist_score").split(" ")[1][1:-1] if " " in stats.get("startlist_score") else stats.get("startlist_score"), #in case of stage races, take current startlist score for that stage
        "won_how": stats.get("won_how"),
        "temp": stats.get("temp").replace(" °C", "") if stats.get("temp") else "-1", #lets hope this is not a possible value
        "profile_score": stats.get("profile_score"),
        "profile_score_last_25k": stats.get("profile_score_last_25k"),
    }


def filter_results(results: list[dict], race_stats: dict) -> list[dict]:
    seen_rider_ids = set()

    new_results = []
    for rider in results:
        
        rider_name = rider["racer_url_index"].split("/")[-1]
        if rider_name in seen_rider_ids:
            print(f"⚠️ Duplicate entry for rider '{rider_name}' in race {race_stats.get("name")}")
            continue
        seen_rider_ids.add(rider_name)

        new_results.append({
            "rank": rider.get("Rnk") if (rider.get("Rnk") and rider.get("Rnk") not in {"DNF", "DNS", "OTL"}) else "-1", #TODO: Rnk can be "DNF"
            "name": rider_name,
            "specialty": rider["Specialty"],
            "team": rider["Team"],
            "age": rider.get("Age") if rider.get("Age") else "-1",
            "uci_pts": rider.get("UCI") if rider.get("UCI") else "-1",
            "pcs_pts": rider.get("Pnt") if rider.get("Pnt") else "-1",
            "race_id": race_stats.get("race_id"),#"R" + hashlib.md5(f"{race_stats.get("name")}_{race_stats.get("year")}_{race_stats.get("stage")}".encode()).hexdigest()[:8],
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