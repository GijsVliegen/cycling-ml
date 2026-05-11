import math
import polars as pl
import pulp
import json
from pathlib import Path

from wielermanager.race_config import (
    load_race_manifest,
    load_wielermanager_rules,
    resolve_races_to_predict,
)

# ==============================
# INPUTS
# ==============================

EMERGENCY_TRANSFERS = 3
FREE_TRANSFERS = 3 - EMERGENCY_TRANSFERS
BUDGET = 100# - ((ET) * (ET + 1) // 2)  # adjust budget for emergency transfers, using triangular number cost
TEAM_SIZE = 16
TOP_K = 12

rules = load_wielermanager_rules()
points_per_race_type = rules["points_per_race"]

BUDGETS_PATH = Path("wielermanager/WIELERMANAGER_BUDGETS_voorjaar_2026.json")

with open(BUDGETS_PATH, encoding="utf-8") as f:
    budgets = json.load(f)
    # riders_with_known_calender = budgets["riders_with_calender_known"]
    budgets_list = budgets["budgets"]


DEFAULT_DATA_DIR = "data_v2"
DEFAULT_TEST_DATA_DIR = "data_test"


def data_path(data_dir: str, filename: str) -> str:
    return str(Path(data_dir) / filename)


def get_races_to_predict(data_dir: str = DEFAULT_DATA_DIR, year: int = 2026):
    manifest_races = load_race_manifest(data_dir)
    if manifest_races is not None:
        return manifest_races
    return resolve_races_to_predict(rules, default_year=year)


def get_available_prediction_races(data_dir: str = DEFAULT_DATA_DIR, limit: int | None = None):
    available = []
    for race, stage, race_type in get_races_to_predict(data_dir=data_dir):
        prediction_path = Path(data_path(data_dir, f"wielermanager/rider_percentages_{race}.parquet"))
        if prediction_path.exists():
            available.append((race, stage, race_type))
    if limit is not None:
        available = available[:limit]
    return available


def transfer_cost_formula(n):
    if n <= 0:
        return 0
    return n * (n + 1) // 2  # triangular number


def solve_team_selection(
    race_dfs: list[pl.DataFrame],
    cost_df: pl.DataFrame,
    enforce_seed_team: bool = True,
):

    #TODO: assumed to be ordered eactually, so no todo
    # sorted_race_dfs = sorted(race_dfs, key=lambda df: df["expected_points"].max(), reverse=True)
    
    race_tables = []
    race_tables_kopman = []
    for i, df in enumerate(race_dfs):
        race_tables.append(
            df.select(["name", "expected_points"]).rename({"expected_points": f"race_{i+1}"})
        )
        race_tables_kopman.append(
            df.select(["name", "expected_kopman_points"]).rename({"expected_kopman_points": f"race_{i+1}"})
        )

    all_points = race_tables[0]
    all_kopman_points = race_tables_kopman[0]
    for df in race_tables[1:]:
        all_points = all_points.join(df, on="name", how="full", coalesce=True)
    for df in race_tables_kopman[1:]:
        all_kopman_points = all_kopman_points.join(df, on="name", how="full", coalesce=True)

    all_points = all_points.fill_null(0)
    all_kopman_points = all_kopman_points.fill_null(0)
    data = all_points.join(cost_df, on="name", how="left").fill_null(2)
    data_kopman = all_kopman_points.join(cost_df, on="name", how="left").fill_null(2)

    riders = data["name"].to_list()
    races = [col for col in data.columns if col.startswith("race_")]

    cost = dict(zip(data["name"], data["cost"]))

    expected_points = {
        r: {
            race: data.filter(pl.col("name") == r)[race][0]
                if not math.isnan(data.filter(pl.col("name") == r)[race][0]) else 0
            for race in races
        }
        for r in riders
    }
    expected_kopman_points = {
        r: {
            race: data_kopman.filter(pl.col("name") == r)[race][0]
                if not math.isnan(data_kopman.filter(pl.col("name") == r)[race][0]) else 0
            for race in races
        }
        for r in riders
    }

    # ===============================
    # MODEL
    # ===============================
    model = pulp.LpProblem("SeasonOptimization", pulp.LpMaximize)

    #kopman at race t
    k = pulp.LpVariable.dicts("select_kopman",
                              [(r, t) for r in riders for t in races
                               if expected_kopman_points[r][t] > 0
                            ],
                              cat="Binary")
    #in team at race t
    x = pulp.LpVariable.dicts("select",
                              [(r, t) for r in riders for t in races],
                              cat="Binary")
    # not on bench at race t
    y = pulp.LpVariable.dicts("score",
                              [(r, t) for r in riders for t in races],
                              cat="Binary")

    # transfer in at race t
    z = pulp.LpVariable.dicts("transfer_in",
                              [(r, t) for r in riders for t in races],
                              cat="Binary")
    # transfer out at race t
    o = pulp.LpVariable.dicts("transfer_out",
                              [(r, t) for r in riders for t in races],
                              cat="Binary")
    # binary ladder: nr_transfers_atleast_n_at_t[n] = 1 if extra_transfers == n
    nr_transfers_atleast_n_at_t = pulp.LpVariable.dicts(
        "extra_transfer_choice_at_race_t",
        [(t, n) for t in races for n in list(range(20))],  # we won't have more than 20 extra transfers, can adjust if needed
        cat="Binary"
    )
    nr_transfers_exact_n_at_t = pulp.LpVariable.dicts(
        "extra_transfer_choice_at_race_t_single",
        [(t, n) for t in races for n in list(range(20))],  # we won't have more than 20 extra transfers, can adjust if needed
        cat="Binary"
    )

    # ===============================
    # Team selection tot op heden
    # ===============================
    team_selection = {
        "race_1": [
            "arnaud-de-lie",
            "paul-magnier",
            "jasper-philipsen",
            "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
        ],
        "race_2": [
            "arnaud-de-lie",
            "paul-magnier",
            "jasper-philipsen",
            "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
        ],
        "race_3": [
            "arnaud-de-lie",
            # "paul-magnier", 
            "jasper-philipsen",
            "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
        ],
        "race_4": [
            "arnaud-de-lie",
            # "paul-magnier",
            "jasper-philipsen",
            "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
        ],
        "race_5": [
            "arnaud-de-lie",
            # "paul-magnier",
            "jasper-philipsen",
            "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
        ],
        "race_6": [
            "arnaud-de-lie",
            # "paul-magnier",
            "jasper-philipsen",
            "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
        ],
        "race_7": [
            "arnaud-de-lie",
            # "paul-magnier",
            "jasper-philipsen",
            # "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
            "mads-pedersen",
        ],
        "race_8": [
            "arnaud-de-lie",
            # "paul-magnier",
            "jasper-philipsen",
            # "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
            "mads-pedersen",
        ],
        "race_9": [
            "arnaud-de-lie",
            # "paul-magnier",
            "jasper-philipsen",
            # "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
            "mads-pedersen",
        ],
        "race_10": [
            "arnaud-de-lie",
            # "paul-magnier",
            "jasper-philipsen",
            # "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
            "mads-pedersen",
        ],
        "race_11": [
            "arnaud-de-lie",
            # "paul-magnier",
            "jasper-philipsen",
            # "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
            "mads-pedersen",
        ],
        "race_12": [
            "arnaud-de-lie",
            # "paul-magnier",
            "jasper-philipsen",
            # "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
            "mads-pedersen",
        ],
        "race_13": [
            "arnaud-de-lie",
            # "paul-magnier",
            "jasper-philipsen",
            # "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
            "mads-pedersen",
        ],
        "race_14": [
            "arnaud-de-lie",
            # "paul-magnier",
            "jasper-philipsen",
            # "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
            "mads-pedersen",
        ],
        "race_15": [
            "arnaud-de-lie",
            # "paul-magnier",
            "jasper-philipsen",
            # "kaden-groves",
            "jordi-meeus",
            "christophe-laporte",
            # "tim-wellens",
            "davide-ballerini",
            "jelte-krijnsen",
            "thomas-pidcock",
            "dylan-teuns",
            "mathieu-van-der-poel",
            "alex-kirsch",
            "stanislaw-aniolkowski",
            "filippo-fiorelli",
            "tom-crabbe",
            "jordan-labrosse",
            "bastien-tronchon",
            "isaac-del-toro",
            "julian-alaphilippe",
            "romain-gregoire1",
            "tadej-pogacar",
            "mads-pedersen",
        ],
    }
    if enforce_seed_team:
        for t, team_riders in team_selection.items():
            if t not in races:
                continue
            unique_team_riders = list(dict.fromkeys(team_riders))
            pinned_riders = unique_team_riders[:TEAM_SIZE]
            for r in pinned_riders:
                if (r, t) in x:
                    model += x[(r, t)] == 1

    # ===============================
    # OBJECTIVE
    # ===============================

    # total points
    points = pulp.lpSum(
        expected_points[r][t] * y[(r, t)]
        for r in riders
        for t in races
    )
    kopman_points = pulp.lpSum(
        expected_kopman_points[r][t] * k[(r, t)]
        for r in riders
        for t in races
        if expected_kopman_points[r][t] > 0
    )

    
    # ===============================
    # TRANSFERS AND GLOBAL TRANSFER COST
    # ===============================

    for t in races:
        if t == "race_1":
            continue
        for r in riders: 
            # z=1 if rider enters squad at race t
            # also a ridern needs to be going out if one is coming in, so we link to the previous race 
            model += z[(r, t)] >= x[(r, t)] - x[(r, f"race_{int(t.split('_')[1]) - 1}")]
            model += o[(r, t)] >= -x[(r, t)] + x[(r, f"race_{int(t.split('_')[1]) - 1}")]
    free_transfers = FREE_TRANSFERS  # total free transfers for the season

    # total transfers across the season
    # for t in races:
    #     model += pulp.lpSum(nr_transfers_atleast_n_at_t[(t, n)] for n in range(20)) == 1  # exactly one choice for nr of extra transfers
    for t in races:
        for n in range(19):
            model += nr_transfers_atleast_n_at_t[(t, n)] >= nr_transfers_atleast_n_at_t[(t, n+1)]
            model += nr_transfers_atleast_n_at_t[(t, n)] <= 1
            model += nr_transfers_exact_n_at_t[(t, n)] == nr_transfers_atleast_n_at_t[(t, n)] - nr_transfers_atleast_n_at_t[(t, n+1)]
        for n in range(20):
            if t != "race_1":
                model += nr_transfers_atleast_n_at_t[(t, n)] >= nr_transfers_atleast_n_at_t[(f"race_{int(t.split('_')[1]) - 1}", n)]
        model += pulp.lpSum(nr_transfers_exact_n_at_t[(t, n)] for n in range(20)) == 1  # exactly one choice for nr of extra transfers

    for t in races:
        model += (
            pulp.lpSum(nr_transfers_atleast_n_at_t[(t, n)] for n in range(20)) 
            == 
            pulp.lpSum(
                z[(r, t_earlier)] for r in riders for t_earlier in races if int(t_earlier.split("_")[1]) <= int(t.split("_")[1])
            )
        )
    total_transfers_expr = pulp.lpSum(
        z[(r, t)] for r in riders for t in races if t != "race_1"
    )
    total_outgoing_expr = pulp.lpSum(
        o[(r, t)] for r in riders for t in races if t != "race_1"
    )
    model += total_transfers_expr == total_outgoing_expr  # ensure transfers in = transfers

    # extra transfers above the free ones
    #link to binary ladder
    # model += extra_transfers == pulp.lpSum(n * nr_transfers_exact_n_at_t[("race_19", n)] for n in range(20))
    # model += extra_transfers >= total_transfers_expr - free_transfers
    # model += extra_transfers >= 0  # ensure non-negative

    # linear approximation of transfer cost:
    # we cannot multiply extra_transfers * (extra_transfers + 1)/2 in MILP
    # instead, we approximate with a simple linear penalty
    # if you want exact triangular cost, you can postprocess after solving
    def replacement_cost(n):
        if n <= 0:
            return 0
        return n * (n + 1) // 2  # triangular numbers


    # ===============================
    # CONSTRAINTS
    # ===============================

    for t in races:
        transfer_cost_expr = pulp.lpSum(replacement_cost(n - free_transfers) * nr_transfers_exact_n_at_t[(t, n)] for n in range(20))

        # 20 riders
        model += pulp.lpSum(x[(r, t)] for r in riders) == TEAM_SIZE
        
        #one kopman
        model += pulp.lpSum(k[(r, t)] for r in riders if expected_kopman_points[r][t] > 0) == 1


        # make sure to have enough budget for future transfers, not just current one
        #TODO: how to?

        # budget
        model += pulp.lpSum(cost.get(r, 2) * x[(r, t)] for r in riders) + transfer_cost_expr <= BUDGET

        # 12 scoring
        model += pulp.lpSum(y[(r, t)] for r in riders) <= TOP_K

        for r in riders:
            model += y[(r, t)] <= x[(r, t)]
            if expected_kopman_points[r][t] > 0:
                model += k[(r, t)] <= x[(r, t)]

    # ===============================
    # FINAL OBJECTIVE
    # ===============================
    model += points + kopman_points
    # ===============================
    # SOLVE
    # ===============================

    model.solve(pulp.PULP_CBC_CMD(msg=True))
    solver_status = pulp.LpStatus.get(model.status, "Unknown")
    if solver_status not in {"Optimal", "Feasible"}:
        raise ValueError(f"Optimization failed with status: {solver_status}")
    
    # ==========================================
    # Extract Solution
    # ==========================================
    squads = {
        t: [r for r in riders if x[(r, t)].value() == 1]
        for t in races
    }
    #print all riders in the team who cost 2 mil
    for r in riders:
        if cost.get(r, 2) == 2 and any(x[(r, t)].value() == 1 for t in races):
            print(f"Rider {r} has unknown cost, treated as 2 million")
    selected_riders = squads["race_1"]
    selected_kopmannen = {
        t: next(
            (r for r in riders if expected_kopman_points[r][t] > 0 and k[(r, t)].value() == 1),
            None,
        )
        for t in races
    }
    # transfers per race
    # transfers = {}
    transfers_in = {
        r: t
        for t in races
        for r in riders 
        if z[(r, t)].value() is not None and z[(r, t)].value() > 0
    }
    transfers_out = {
        r: t
        for t in races
        for r in riders        
        if o[(r, t)].value() is not None and o[(r, t)].value() > 0
    }

    total_points = sum(
        expected_points.get(r, {}).get(t, 0) * y[(r, t)].value()
        # if y[(r, t)].value() is not None else 0
        for (r, t) in y
    )
    from collections import defaultdict

    rider_points = defaultdict(float)

    for (r, t) in y:
        if y[(r, t)].value() == 1:
            rider_points[r] += expected_points[r][t]

    # keep only riders with > 0
    rider_points = {
        r: pts for r, pts in rider_points.items()
        if pts > 0
    }
    rider_scoring_races = defaultdict(int)

    for (r, t) in y:
        if y[(r, t)].value() == 1:
            rider_scoring_races[r] += 1

    result_df = (
        pl.DataFrame({
            "name": list(rider_points.keys()),
            "total_points": list(rider_points.values()),
            "scoring_races": [
                rider_scoring_races[r]
                for r in rider_points.keys()
            ],
        })
        .join(cost_df, on="name", how="left").fill_null(2)
        .sort("total_points", descending=True)
    )
    total_cost_at_each_race = {
        t: sum(cost.get(r, 2) * x[(r, t)].value() for r in riders)
        for t in races
    }
    selected_riders_for_each_race = {
        t: [r for r in riders if y[(r, t)].value() == 1]
        for t in races
    }
    print("DEBUG")
    print("nr of registered transfers at each point:")
    for race_name in races:
        for transfer_nr in range(20):
            if (race_name, transfer_nr) in nr_transfers_exact_n_at_t and nr_transfers_exact_n_at_t[(race_name, transfer_nr)].value() == 1:
                print(f"at {race_name} we have {transfer_nr} extra transfers")
    
    print(f"selected_riders for each race: {selected_riders_for_each_race}")
    print("------------------------------")
    print(f"total_cost_at_each_race: {total_cost_at_each_race}")
    print(f" kopmannen: {selected_kopmannen}")
    return {
        "initial_selected_riders": selected_riders,
        # "initial_cost": total_cost,
        "total_expected_points": total_points,
        "squads_per_race": squads,
        "transfers_in": transfers_in,
        "transfers_out": transfers_out,
        "rider_summary_df": result_df,
    }

def create_rider_cost_df():
    ordered_budgets = [(item["name"], item["cost"]) 
        for item in budgets_list
    ]
    rider_cost_df = pl.DataFrame(
        {
            "name": [k for k, v in ordered_budgets],
            "cost": [v for k, v in ordered_budgets]
        }
    )
    return rider_cost_df

def main(
    data_dir: str = DEFAULT_DATA_DIR,
    selected_races: list[tuple[str, int, str]] | None = None,
    enforce_seed_team: bool = True,
):
    all_race_preds = []
    if selected_races is None:
        selected_races = get_available_prediction_races(data_dir)
    for race, stage, race_type in selected_races:
        race_prediction = pl.read_parquet(data_path(data_dir, f"wielermanager/rider_percentages_{race}.parquet"))
        all_race_preds.append(race_prediction)

    rider_cost_df = create_rider_cost_df()
    solution_dict = solve_team_selection(
        race_dfs = all_race_preds, 
        cost_df = rider_cost_df,
        enforce_seed_team=enforce_seed_team,
    )
    print("Selected Team:")
    print(f"transfers in: {solution_dict['transfers_in']}")
    print(f"transfers out: {solution_dict['transfers_out']}")
    print(f"rider summary: {solution_dict['rider_summary_df']}")
    print(solution_dict["initial_selected_riders"])
    print(f"Total Expected Points: {solution_dict['total_expected_points']}")
    return solution_dict


def main_test(data_dir: str = DEFAULT_TEST_DATA_DIR, limit: int = 6):
    selected_races = get_available_prediction_races(data_dir, limit=limit)
    return main(data_dir=data_dir, selected_races=selected_races, enforce_seed_team=False)
    


if __name__ == "__main__":
    main()