
import polars as pl

try:
    # Try imports for when running from root directory
    from data_engineering.data_cleaning_functions import filter_results, filter_stats
    from data_engineering.soup_parsing_functions import (
        get_race_profile_url,
        parse_calendar_page,
        parse_gc_page,
        parse_race_page,
        parse_race_profile_page,
        parse_race_result_page,
        parse_rider_page,
        parse_rider_statistics_page,
        parse_startlist_page
    )
    from data_engineering.selenium_webscraping import (
        load_soup_from_file, 
        download_page,
        load_soups_from_http, 
        url_to_filename
    )
except ImportError:
    # Fall back to local imports for when running from data_engineering/ directory
    from data_cleaning_functions import filter_results, filter_stats
    from soup_parsing_functions import (
        get_race_profile_url,
        parse_calendar_page,
        parse_gc_page,
        parse_race_page,
        parse_race_profile_page,
        parse_race_result_page,
        parse_rider_page,
        parse_rider_statistics_page,
        parse_startlist_page
    )
    from selenium_webscraping import (
        load_soup_from_file, 
        download_page,
        load_soups_from_http, 
        url_to_filename
    )


BASE_URL = "https://www.procyclingstats.com"


def transform_new_race_data(new_race_dict: dict) -> pl.DataFrame:
    races_df = pl.DataFrame(new_race_dict)
    races_df = races_df.with_columns([
        # pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("distance_km").cast(pl.Float64, strict=False),
        pl.col("elevation_m").cast(pl.Float64, strict=False),
        pl.col("avg_speed_kmh").cast(pl.Float64, strict=False),
        pl.col("startlist_score").cast(pl.Float64, strict=False),
        pl.col("final_km_percentage").cast(pl.Float64, strict=False),
        pl.col("temp").cast(pl.Float64, strict=False),
        pl.col("profile_score").cast(pl.Float64, strict=False),
        pl.col("profile_score_last_25k").cast(pl.Float64, strict=False),
        #TODO: add profile scores
    ])
    return races_df

def transform_race_data(all_race_stats: list[dict], all_results: list[dict]) -> pl.DataFrame:
    """
    Transform race data using Polars with the following operations:
    1. Parse date to datetime
    2. Convert distance, heigh_meters, speed, startlist_score to numeric
    3. Clean blank/missing temp values to NaN or drop
    """

    # Create a Polars DataFrame

    # Convert types

    races_df = pl.DataFrame(all_race_stats)
    races_df = races_df.with_columns([
        # pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("distance_km").cast(pl.Float64, strict=False),
        pl.col("elevation_m").cast(pl.Float64, strict=False),
        pl.col("avg_speed_kmh").cast(pl.Float64, strict=False),
        pl.col("startlist_score").cast(pl.Float64, strict=False),
        pl.col("final_km_percentage").cast(pl.Float64, strict=False),
        pl.col("temp").cast(pl.Float64, strict=False),
        pl.col("profile_score").cast(pl.Float64, strict=False),
        pl.col("profile_score_last_25k").cast(pl.Float64, strict=False),
        #TODO: add profile scores
    ])
    
    results_df = pl.DataFrame(all_results)
    results_df = results_df.with_columns([

        pl.when(pl.col("rank").cast(pl.Float64, strict=False).is_not_null())
        .then(pl.col("rank").cast(pl.Float64, strict=False))
        .otherwise(-1.0)
        .alias("rank"), #Save casting from str to float
        pl.col("age").cast(pl.Float64, strict=False),
        pl.col("uci_pts").cast(pl.Float64, strict=False),
        pl.col("pcs_pts").cast(pl.Float64, strict=False),
    ])

    # print(results_df)
    return races_df, results_df


def fetch_year_race_urls(year: int) -> list[str]:
    """fetch all urls to races in a given year
    
    saves them to file
    """

    classifications_to_search = [
        "1.UWT",
        "2.UWT",
        "1.Pro",
        # "2.1",
        # "2.Pro",
        "1.1", #GP samyn is 1.1
    ]
    all_race_urls = []
    # for classification in classifications_to_search:
    calender_url = f"{BASE_URL}/races.php?s=&year={year}&circuit=&class=&filter=Filter"
    try:
        calender_soup = load_soups_from_http(calender_url)[0]
    except Exception as e:
        print(f"exception for retrieving calendar page {calender_url}: {e}")
        return []
    race_urls, gc_urls = parse_calendar_page(calender_soup)

    for gc_url in gc_urls:
        print(f"Fetching GC page: {gc_url}")
        try:
            gc_soup = load_soups_from_http(gc_url)[0]
        except:
            print(f"exception for retrieving gc page {gc_url}")
            continue
        stage_urls = parse_gc_page(gc_soup)
        race_urls.extend(stage_urls)
    all_race_urls.extend(race_urls)
    
    with open(f"urls/races_{year}.txt", "w") as f:
        f.write("\n".join(all_race_urls))
    return all_race_urls


def download_rider_pages() -> list[dict]:
    """Downloads rider pages and season statistics pages for the saved list of rider names
    
    returns logs
    """
    downloaded_riders = pl.read_parquet("data_v2/rider_yearly_stats_df.parquet")
    races_df = pl.read_parquet("data_v2/races_df.parquet").filter(
        pl.col("startlist_score") > 200
    )
    results_not_downloaded = pl.read_parquet("data_v2/results_df.parquet").join(
        races_df.select("race_id"),
        on="race_id",
        how="inner"
    )
    
    #     .group_by("name")
    #     .agg(pl.count())
    #     # .filter(pl.col("count") >= 5)).select("name").unique().join(
    #     #     downloaded_riders.select("name").unique(),
    #     #     on="name",
    #     #     how="anti"
    #     # )
    rider_names = results_not_downloaded["name"].to_list()
    print(f"Downloading pages for {len(rider_names)} riders")
    log_messages = []

    for rider in rider_names:
        # print(f"Downloading rider: {rider}")
        rider_url = f"{BASE_URL}/rider/{rider}"
        try:
            _, logs = download_page(rider_url)
            log_messages.extend(logs)
        except Exception as e:
            log_messages.append(f"exception for downloading rider page {rider_url}: {e}")
            continue
        season_statistics_url = f"{rider_url}/statistics/season-statistics"
        try:
            _, logs = download_page(season_statistics_url)
            log_messages.extend(logs)   
        except Exception as e:
            log_messages.append(f"exception for downloading rider stats page {season_statistics_url}: {e}")
            continue

    return log_messages

def get_already_downloaded_races(downloaded_races: pl.DataFrame) -> list[tuple[str, str]]:
    
        # f"{BASE_URL}/race/{race}/2026/startlist"
    keys_df = (
        downloaded_races
        .select(
            [pl.concat_str([
                pl.lit(f"{BASE_URL}/race/"),
                pl.col("name").cast(pl.Utf8),
                pl.lit("/"),
                pl.col("year").cast(pl.Utf8),
                pl.lit("/startlist"),
            ]).alias("key_startlist"),
            pl.concat_str([
                pl.lit(f"{BASE_URL}/race/"),
                pl.col("name").cast(pl.Utf8),
                pl.lit("/"),
                pl.col("year").cast(pl.Utf8),
                pl.lit("/result")
            ]).alias("key_result"),
            pl.concat_str([
                pl.lit(f"{BASE_URL}/race/"),
                pl.col("name").cast(pl.Utf8),
                pl.lit("/"),
                pl.col("year").cast(pl.Utf8),
                pl.lit("/stage-"),
                pl.col("stage").cast(pl.Utf8),
            ]).alias("key_stage"),
            
            ]
        )
    )
    valid_keys_startlist = set(keys_df["key_startlist"].to_list())
    valid_keys_result = set(keys_df["key_result"].to_list())
    valid_keys_stage = set(keys_df["key_stage"].to_list())
    return valid_keys_startlist, valid_keys_result, valid_keys_stage


def download_year_races(year: int, downloaded_races: pl.DataFrame) -> list[dict]:
    """Downloads all races for a given year + profiles
    
    returns logs
    """
    year_races_file = f"urls/races_{year}.txt"
    with open(year_races_file, "r") as f:
        race_urls = f.read().splitlines()

    valid_keys_startlist, valid_keys_result, valid_keys_stage = get_already_downloaded_races(downloaded_races)
    keys = set([*valid_keys_startlist, *valid_keys_result, *valid_keys_stage])
    race_urls_filtered = [s for s in race_urls if s not in keys]
    print(f"downloading {len(race_urls_filtered)} for races in year {year}, coming from {len(race_urls)}")
    
    log_messages = []
    for race_url in race_urls_filtered:
        # print(race_url)
        tail = race_url.split("/")[-1]
        if "result" not in tail and "stage" not in tail and "prologue" not in tail:
            continue
        try: 
            _, logs = download_page(race_url)
            log_messages.extend(logs)
        except Exception as e:
            log_messages.append(f"exception for downloading race page {race_url}: {e}")
        race_profile_url = get_race_profile_url(race_url)
        try:
            _, logs = download_page(race_profile_url)
            log_messages.extend(logs)
        except Exception as e:
            log_messages.append(f"exception for downloading race profile page {race_profile_url}: {e}")

    return log_messages

def parse_riders_to_polars() -> tuple[pl.DataFrame, pl.DataFrame]:
    
    results = pl.read_parquet("data_v2/results_df.parquet")
    rider_names = results.select("name").unique()["name"].to_list()


    log_messages = []
    all_rider_stats = []
    all_rider_yearly_stats = []

    for rider in rider_names:
        print(f"parsing rider: {rider}")
        rider_url = f"{BASE_URL}/rider/{rider}"
        try:
            rider_soup = load_soup_from_file(rider_url)
        except Exception as e:
            log_messages.append(f"exception for loading rider file {rider_url}: {e}")
            continue
        try:
            rider_stats, team_name = parse_rider_page(rider_soup)
            rider_stats["name"] = rider
            rider_stats["team_name"] = team_name
            all_rider_stats.append(rider_stats)
        except Exception as e:
            log_messages.append(f"exception for parsing rider page {rider_url}: {e}")
            continue

        rider_statistics_url = f"{rider_url}/statistics/season-statistics"
        try:
            rider_statistics_soup = load_soup_from_file(rider_statistics_url)
        except Exception as e:
            log_messages.append(f"exception for loading rider statistics file {rider_statistics_url}: {e}")
            continue
        try:
            rider_yearly_stats = parse_rider_statistics_page(rider_statistics_soup)
        except Exception as e:
            log_messages.append(f"exception for parsing rider statistics page {rider_statistics_url}: {e}")
            rider_yearly_stats = []

        for rider_year_stat in rider_yearly_stats:
            rider_year_stat["name"] = rider
        all_rider_yearly_stats.extend(rider_yearly_stats)
    
    all_rider_stats_df = pl.DataFrame(all_rider_stats)
    all_rider_yearly_stats_df = pl.DataFrame(all_rider_yearly_stats)

    return all_rider_stats_df, all_rider_yearly_stats_df, log_messages
                               

def parse_races_to_polars(year: int, already_in_polars: list[tuple[str, str]]) -> list[dict]:
    """Parse races for a certain year from saved html files to polars DataFrames
    
    races already_in_polars (race_name, year) are excluded from parsing
    """


    year_races_file = f"urls/races_{year}.txt"
    with open(year_races_file, "r") as f:
        race_urls = f.read().splitlines()
    
    logs = []
    all_race_stats = []
    all_results = []
    for race_url in race_urls:
        try:
            race_name, year, tail = race_url.split("/")[-3:]
        except Exception as e:
            logs.append(f"exception for splitting race url {race_url}: {e}")
            continue
        if (race_name, year) in already_in_polars:
            print(f"Skipping already parsed race {race_url}")
            continue
        if "result" not in tail and "stage" not in tail and "prologue" not in tail:
            continue
        # print(f"parsing {race_url}")
        try: 
            race_soup = load_soup_from_file(race_url)
        except Exception as e:
            logs.append(f"exception for loading race file {race_url}: {e}")
            continue
        try:
            race_stats = parse_race_page(race_soup)
        except Exception as e:
            logs.append(f"exception for parsing race page {race_url}: {e}")
            continue

        race_profile_url = get_race_profile_url(race_url)
        race_profile_soup = load_soup_from_file(race_profile_url)
        race_name, year, tail = race_url.split("/")[-3:]
        mapping = {
            "result": 1,
            "prologue": 0,
        }
        if tail not in mapping:
            try: 
                int(tail.split("-")[1])
            except Exception as e:
                continue 
        stage_nr = mapping[tail] if tail in mapping else int(tail.split("-")[1])
        race_profile, parse_logs = parse_race_profile_page(
            race_profile_soup,
            race_name=race_name,
            year=year,
            stage_nr=stage_nr
        )
        logs.extend(parse_logs)

        try:
            race_stats = filter_stats(race_stats | race_profile, race_url)
            all_race_stats.append(race_stats)
        except Exception as e:
            logs.append(f"exception for filtering stats: {e}")
            continue
        

        race_results, parse_logs = parse_race_result_page(race_soup)
        logs.extend(parse_logs)
        
        try:
            race_results = filter_results(race_results, race_stats)
            all_results.extend(race_results)
        except Exception as e:
            logs.append(f"exception for filtering stats: {e}")
            continue
    
    try:
        races_df, results_df = transform_race_data(
            all_race_stats=all_race_stats,
            all_results=all_results
        )
    except Exception as e:
        races_df, results_df = None, None
        logs.append(f"exception for filtering stats: {e}")
    

    return races_df, results_df, logs

def parse_new_races_to_dict(races: list[tuple[str, int, str]]) -> tuple[dict, list[str]]:
    logs = []
    for race_name, stage_nr, race_type in races:
        if stage_nr > 0:
            race_url = f"{BASE_URL}/race/{race_name}/2026/stage-{stage_nr}"
        else:
            race_url = f"{BASE_URL}/race/{race_name}/2026/result"
    
        tail = race_url.split("/")[-1]
        race_soup = load_soups_from_http(race_url)[0]
        try:
            race_stats = parse_race_page(race_soup)
        except Exception as e:
            logs.append(f"exception for parsing race page {race_url}: {e}")
            continue

        race_profile_url = get_race_profile_url(race_url)
        race_profile_soup = load_soups_from_http(race_profile_url)[0]
        race_name, year, tail = race_url.split("/")[-3:]
        mapping = {
            "result": 1,
            "prologue": 0,
        }
        stage_nr = mapping[tail] if tail in mapping else int(tail.split("-")[1])
        race_profile, parse_logs = parse_race_profile_page(
            race_profile_soup,
            race_name=race_name,
            year=year,
            stage_nr=stage_nr
        )
        logs.extend(parse_logs)

        try:
            race_stats = filter_stats(race_stats | race_profile, race_url)
        except Exception as e:
            logs.append(f"exception for filtering stats: {e}")
            continue
        
        try:
            race_df = transform_new_race_data(
                new_race_dict=race_stats,
            )
        except Exception as e:
            logs.append(f"exception for filtering stats: {e}")
            continue
    
    # race_results, parse_logs = parse_race_result_page(race_soup)
    # logs.extend(parse_logs)
    
    # try:
    #     race_results = filter_results(race_results, race_stats)
    # except Exception as e:
    #     logs.append(f"exception for filtering stats: {e}")
    #     return None, logs
    return race_df, logs


def make_races_results_df():
    current_races_df = pl.read_parquet("data_v2/races_df.parquet")
    current_results_df = pl.read_parquet("data_v2/results_df.parquet")
    races_in_polars = [
        (race["name"], race["year"]) 
        for race in current_races_df.select(["name", "year"]).unique().to_dicts()
    ]
    year_range = range(2005, 2014)
    logs = []
    for year in year_range:
        races_df, results_df, parse_logs = parse_races_to_polars(
            year = year,
            already_in_polars = races_in_polars
        )
        logs.extend(parse_logs)
        
        if races_df is not None and results_df is not None:
            current_races_df = pl.concat([current_races_df, races_df]).unique(subset=["name", "race_id"])
            current_results_df = pl.concat([current_results_df, results_df]).unique(subset=["name", "race_id"])

    current_races_df.write_parquet("data_v2/races_df.parquet")
    current_results_df.write_parquet("data_v2/results_df.parquet")

    return logs

def make_riders_stats_df():
    # current_rider_stats_df = pl.read_parquet("data_v2/rider_stats_df.parquet")
    current_rider_yearly_stats_df = pl.read_parquet("data_v2/rider_yearly_stats_df.parquet")
    logs = []
    all_rider_stats_df, all_rider_yearly_stats_df, parse_logs = parse_riders_to_polars()
    logs.extend(parse_logs)
    # current_rider_stats_df = pl.concat([current_rider_stats_df, all_rider_stats_df]).unique(subset=["name"])
    current_rider_yearly_stats_df = pl.concat([current_rider_yearly_stats_df, all_rider_yearly_stats_df]).unique(subset=["name", "season"])
    # current_rider_stats_df.write_parquet("data_v2/rider_stats_df.parquet")
    current_rider_yearly_stats_df.write_parquet("data_v2/rider_yearly_stats_df.parquet")

    return logs

def get_missing_data_overview():
    results = pl.read_parquet("data_v2/results_df.parquet")
    riders_in_results = results.select("name").unique()["name"].to_list()

    rider_stats = pl.read_parquet("data_v2/rider_stats_df.parquet")
    riders_in_stats = rider_stats.select("name").unique()["name"].to_list()
    missing_riders = set(riders_in_results) - set(riders_in_stats)
    print(f"Number of riders in results: {len(riders_in_results)}")
    print(f"Number of riders in stats: {len(riders_in_stats)}")
    print(f"Number of missing riders: {len(missing_riders)}")

    for year in range(2014, 2026):
        print("-----------------------------\n")
        get_missing_races_overview(year)

def get_missing_races_overview(year: int):

    year_races_file = f"urls/races_{year}.txt"
    with open(year_races_file, "r") as f:
        race_urls = f.read().splitlines()
    race_urls = [
        race_url for race_url in race_urls
        if "result" in race_url or "stage" in race_url or "prologue" in race_url
    ]
    nr_of_races_in_index = len(race_urls)
    results = pl.read_parquet("data_v2/results_df.parquet")
    races = pl.read_parquet("data_v2/races_df.parquet")
    races_in_races_df = races.filter(pl.col("year") == str(year)).select("race_id").unique()["race_id"].to_list()
    print(f"Number of races in index for year {year}: {nr_of_races_in_index}")
    print(f"Number of races in races df for year {year}: {len(races_in_races_df)}") 

def create_new_race_data(race_name_stages: tuple[str, int]):
    startlist_urls = [
        f"{BASE_URL}/race/{race}/2026/startlist"
        for (race, stage, race_type) in race_name_stages
    ]

    startlist_soups = load_soups_from_http(startlist_urls)
    all_startlists = []
    logs = []
    for soup, (race, stage, race_type) in zip(startlist_soups, race_name_stages):
        print(f"parsing race {race}")
        startlist, team_list = parse_startlist_page(soup)
        race_df, race_logs = parse_new_races_to_dict(
            races = race_name_stages
        )
        logs.append(race_logs)
        startlist_df = pl.DataFrame(startlist)
        startlist_df.write_parquet(f"data_v2/wielermanager/startlist_{race}.parquet")
        race_df.write_parquet(f"data_v2/wielermanager/new_race_stats_{race}.parquet")
    return logs

def main_new():
    """Get data for prediction of future race"""
    races = [("omloop-het-nieuwsblad", -1)]
    logs = create_new_race_data(races)
    return logs


def main():
    logs = []
    # fetch_year_race_urls(2025)
    # fetch_year_race_urls(2024)
    # fetch_year_race_urls(2023)
    # fetch_year_race_urls(2022)
    # fetch_year_race_urls(2021) #still needs to happen for 2.1 rraces

    # for i in range(2005, 2013):
    #     fetch_year_race_urls(i)
    # downloaded_races = pl.read_parquet("data_v2/races_df.parquet")
    # for i in range(2005, 2013):
    #     more_logs = download_year_races(i, downloaded_races)
    #     logs += more_logs

    more_logs = make_races_results_df()
    logs += more_logs
    
    # more_logs = download_rider_pages()
    # logs += more_logs
    more_logs = make_riders_stats_df()
    logs += more_logs

    # logs = main_new()

    with open(f"logs.txt", "w") as f:
        f.write("\n".join(logs))
    # print(logs)


    

if __name__ == "__main__":
    # asyncio.run(main())
    main()