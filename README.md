selenium: Starting driver for brave
 - location: Downloads/chromedriver-linux64
 - start: $ ./chromedriver


copilot cli in terminal: $ gh copilot


Data engineering:
  - fetch all race urls -> fetch_year_race_urls(year) -> txt file with race urls
  - download all race urls -> download_year_races(year) -> downloads html for all race urls
  - scrape downloaded htmls to polars -> make_races_results_df()

  -> retrieve riders from results_df
  - download all rider urls -> download_rider_pages()
  - scrape downloaded htmls to polars -> make_riders_stats_df()