from __future__ import annotations

from datetime import datetime
import argparse
from pathlib import Path
import polars as pl

try:
    from data_engineering.data_structure_functions import (
        fetch_year_race_urls,
        download_year_races,
        parse_races_to_polars,
        get_already_downloaded_races,
    )
except ImportError:
    from data_structure_functions import (
        fetch_year_race_urls,
        download_year_races,
        parse_races_to_polars,
        get_already_downloaded_races,
    )


def _default_years(include_previous_year: bool = False) -> list[int]:
    current_year = datetime.now().year
    years = [current_year]
    if include_previous_year:
        years.insert(0, current_year - 1)
    return years


def _load_current_tables(data_dir: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    races_path = Path(data_dir) / "races_df.parquet"
    results_path = Path(data_dir) / "results_df.parquet"
    if not races_path.exists() or not results_path.exists():
        raise FileNotFoundError(
            f"Expected parquet files at `{races_path}` and `{results_path}`."
        )
    return pl.read_parquet(races_path), pl.read_parquet(results_path)


def _count_missing_urls_for_year(year: int, current_races_df: pl.DataFrame) -> tuple[int, int]:
    fetched_urls = fetch_year_race_urls(year)
    valid_keys_startlist, valid_keys_result, valid_keys_stage = get_already_downloaded_races(current_races_df)
    existing_keys = set([*valid_keys_startlist, *valid_keys_result, *valid_keys_stage])

    candidate_urls = []
    for race_url in fetched_urls:
        tail = race_url.split("/")[-1]
        if "result" in tail or "stage" in tail or "prologue" in tail:
            candidate_urls.append(race_url)

    missing_urls = [url for url in candidate_urls if url not in existing_keys]
    return len(candidate_urls), len(missing_urls)


def _select_missing_urls_for_year(
    year: int,
    current_races_df: pl.DataFrame,
    max_missing_urls: int | None,
) -> tuple[list[str], int]:
    fetched_urls = fetch_year_race_urls(year)
    valid_keys_startlist, valid_keys_result, valid_keys_stage = get_already_downloaded_races(current_races_df)
    existing_keys = set([*valid_keys_startlist, *valid_keys_result, *valid_keys_stage])

    candidate_urls = []
    for race_url in fetched_urls:
        tail = race_url.split("/")[-1]
        if "result" in tail or "stage" in tail or "prologue" in tail:
            candidate_urls.append(race_url)

    missing_urls = [url for url in candidate_urls if url not in existing_keys]
    if max_missing_urls is not None:
        missing_urls = missing_urls[:max_missing_urls]
    return missing_urls, len(candidate_urls)


def update_latest_finished_races(
    years: list[int],
    data_dir: str = "data_v2",
    dry_run: bool = False,
    max_missing_urls_per_year: int | None = 120,
) -> dict:
    current_races_df, current_results_df = _load_current_tables(data_dir)
    logs: list[str] = []

    summary = {
        "years": years,
        "dry_run": dry_run,
        "max_missing_urls_per_year": max_missing_urls_per_year,
        "candidate_urls": 0,
        "missing_urls": 0,
        "selected_missing_urls": 0,
        "new_race_rows": 0,
        "new_result_rows": 0,
    }

    races_in_polars = [
        (race["name"], race["year"])
        for race in current_races_df.select(["name", "year"]).unique().to_dicts()
    ]

    for year in years:
        selected_missing_urls, candidate_count = _select_missing_urls_for_year(
            year=year,
            current_races_df=current_races_df,
            max_missing_urls=max_missing_urls_per_year,
        )
        summary["candidate_urls"] += candidate_count
        summary["selected_missing_urls"] += len(selected_missing_urls)

        valid_keys_startlist, valid_keys_result, valid_keys_stage = get_already_downloaded_races(current_races_df)
        existing_keys = set([*valid_keys_startlist, *valid_keys_result, *valid_keys_stage])
        missing_count = sum(1 for url in selected_missing_urls if url not in existing_keys)
        summary["missing_urls"] += missing_count

        print(
            f"year={year}: candidate_urls={candidate_count}, "
            f"selected_missing_urls={len(selected_missing_urls)}"
        )

        if dry_run or len(selected_missing_urls) == 0:
            continue

        print(f"\n=== Updating year {year} ===")
        year_races_file = Path("urls") / f"races_{year}.txt"
        year_races_file.parent.mkdir(parents=True, exist_ok=True)
        year_races_file.write_text("\n".join(selected_missing_urls))

        download_logs = download_year_races(year=year, downloaded_races=current_races_df)
        logs.extend(download_logs)

        races_df, results_df, parse_logs = parse_races_to_polars(
            year=year,
            already_in_polars=races_in_polars,
        )
        logs.extend(parse_logs)

        if races_df is not None and results_df is not None and races_df.height > 0 and results_df.height > 0:
            summary["new_race_rows"] += races_df.height
            summary["new_result_rows"] += results_df.height

            current_races_df = pl.concat([current_races_df, races_df]).unique(subset=["name", "race_id"])
            current_results_df = pl.concat([current_results_df, results_df]).unique(subset=["name", "race_id"])

            races_in_polars = [
                (race["name"], race["year"])
                for race in current_races_df.select(["name", "year"]).unique().to_dicts()
            ]

    if not dry_run:
        races_path = Path(data_dir) / "races_df.parquet"
        results_path = Path(data_dir) / "results_df.parquet"
        current_races_df.write_parquet(races_path)
        current_results_df.write_parquet(results_path)

        logs_path = Path("logs_latest_race_update.txt")
        logs_path.write_text("\n".join(logs))

        print(f"\nSaved updated races to `{races_path}`")
        print(f"Saved updated results to `{results_path}`")
        print(f"Saved logs to `{logs_path}`")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update DB with latest finished PCS races not yet in parquet tables."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Years to check (default: current year; or current+previous with --include-previous-year).",
    )
    parser.add_argument(
        "--include-previous-year",
        action="store_true",
        help="When no --years are provided, include previous year as well.",
    )
    parser.add_argument(
        "--data-dir",
        default="data_v2",
        help="Directory holding races_df.parquet and results_df.parquet (default: data_v2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report how many race URLs are missing, do not download/parse/write parquet.",
    )
    parser.add_argument(
        "--max-missing-urls-per-year",
        type=int,
        default=120,
        help="Max number of missing race URLs to process per year (default: 120).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    years = args.years if args.years is not None else _default_years(args.include_previous_year)
    summary = update_latest_finished_races(
        years=years,
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        max_missing_urls_per_year=args.max_missing_urls_per_year,
    )
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()
