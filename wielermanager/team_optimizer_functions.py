import math
import polars as pl
import pulp
import json

# ==============================
# INPUTS
# ==============================

BUDGET = 120
TEAM_SIZE = 20
TOP_K = 12


with open("wielermanager/WIELERMANAGER_RULES.json") as f:
    rules = json.load(f)
    races_raw = rules["races"]
    races_to_predict = [
        (raw_race["pcs_name"], -1, raw_race["type"])
        for raw_race in races_raw
    ]
    points_per_race_type = rules["points_per_race"]

with open("wielermanager/WIELERMANAGER_BUDGETS.json") as f:
    budgets = json.load(f)
    riders_with_known_calender = budgets["riders_with_calender_known"]
    budgets_list = budgets["budgets"]


def transfer_cost_formula(n):
    if n <= 0:
        return 0
    return n * (n + 1) // 2  # triangular number


def solve_team_selection(race_dfs: list[pl.DataFrame], cost_df: pl.DataFrame):

    #TODO: assumed to be ordered eactually, so no todo
    # sorted_race_dfs = sorted(race_dfs, key=lambda df: df["expected_points"].max(), reverse=True)
    
    race_tables = []
    for i, df in enumerate(race_dfs):
        race_tables.append(
            df.select(["name", "expected_points"]).rename({"expected_points": f"race_{i+1}"})
        )

    all_points = race_tables[0]
    for df in race_tables[1:]:
        all_points = all_points.join(df, on="name", how="outer", coalesce=True)

    all_points = all_points.fill_null(0)
    data = all_points.join(cost_df, on="name", how="left").fill_null(2)

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

    # ===============================
    # MODEL
    # ===============================
    model = pulp.LpProblem("SeasonOptimization", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("select",
                              [(r, t) for r in riders for t in races],
                              cat="Binary")

    y = pulp.LpVariable.dicts("score",
                              [(r, t) for r in riders for t in races],
                              cat="Binary")

    z = pulp.LpVariable.dicts("transfer_in",
                              [(r, t) for r in riders for t in races],
                              cat="Binary")
        # binary ladder: u[n] = 1 if extra_transfers == n
    u = pulp.LpVariable.dicts(
        "extra_transfer_choice",
        list(range(20)),  # we won't have more than 20 extra transfers, can adjust if needed
        cat="Binary"
    )
    model += pulp.lpSum(u[n] for n in range(20)) == 1

    # transfer count selector
    # Integer variable for total extra transfers (above free 3 for the season)
    extra_transfers = pulp.LpVariable("extra_transfers", lowBound=0, cat="Integer")

    # ===============================
    # OBJECTIVE
    # ===============================

    # total points
    points = pulp.lpSum(
        expected_points[r][t] * y[(r, t)]
        for r in riders
        for t in races
    )

    
    # ===============================
    # TRANSFERS AND GLOBAL TRANSFER COST
    # ===============================

    for t in races:
        if t == "race_1":
            continue
        for r in riders:
            # z=1 if rider enters squad at race t
            model += z[(r, t)] >= x[(r, t)] - x[(r, f"race_{int(t.split('_')[1]) - 1}")]
    free_transfers = 3  # total free transfers for the season

    # total transfers across the season
    total_transfers_expr = pulp.lpSum(
        z[(r, t)] for r in riders for t in races if t != "race_1"
    )

    # extra transfers above the free ones
    #link to binary ladder
    model += extra_transfers == pulp.lpSum(n * u[n] for n in range(20))
    model += extra_transfers >= total_transfers_expr - free_transfers
    model += extra_transfers >= 0  # ensure non-negative

    # linear approximation of transfer cost:
    # we cannot multiply extra_transfers * (extra_transfers + 1)/2 in MILP
    # instead, we approximate with a simple linear penalty
    # if you want exact triangular cost, you can postprocess after solving
    def replacement_cost(n):
        if n <= 0:
            return 0
        return n * (n + 1) // 2  # triangular numbers

    transfer_cost_expr = pulp.lpSum(replacement_cost(n) * u[n] for n in range(20))
    model += pulp.lpSum(cost[r] * x[(r, "race_1")] for r in riders) + transfer_cost_expr <= BUDGET

    # ===============================
    # CONSTRAINTS
    # ===============================

    for t in races:

        # 20 riders
        model += pulp.lpSum(x[(r, t)] for r in riders) == TEAM_SIZE

        # budget
        model += pulp.lpSum(cost[r] * x[(r, t)] for r in riders) <= BUDGET

        # 12 scoring
        model += pulp.lpSum(y[(r, t)] for r in riders) <= TOP_K

        for r in riders:
            model += y[(r, t)] <= x[(r, t)]

    # ===============================
    # FINAL OBJECTIVE
    # ===============================
    model += points
    # ===============================
    # SOLVE
    # ===============================

    model.solve(pulp.PULP_CBC_CMD(msg=True))
    
    # ==========================================
    # Extract Solution
    # ==========================================
    squads = {
        t: [r for r in riders if x[(r, t)].value() == 1]
        for t in races
    }
    selected_riders = squads["race_1"]
    # transfers per race
    # transfers = {}
    total_transfers_so_far = 0
    transfers = {
        r: t
        for t in races
        for r in riders 
        if z[(r, t)].value() is not None and z[(r, t)].value() > 0
    }
    total_points = sum(
        expected_points[r][t] * y[(r, t)].value()
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
        .join(cost_df, on="name", how="left")
        .sort("total_points", descending=True)
    )


    return {
        "initial_selected_riders": selected_riders,
        # "initial_cost": total_cost,
        "total_expected_points": total_points,
        "squads_per_race": squads,
        "transfers": transfers,
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

def main():
    all_race_preds = []
    for race, stage, race_type in races_to_predict:
        race_prediction = pl.read_parquet(f"data_v2/wielermanager/rider_percentages_{race}.parquet")
        all_race_preds.append(race_prediction)
    rider_cost_df = create_rider_cost_df()
    solution_dict = solve_team_selection(
        race_dfs = all_race_preds, 
        cost_df = rider_cost_df
    )
    print("Selected Team:")
    print(f"transfers: {solution_dict['transfers']}")
    print(f"rider summary: {solution_dict['rider_summary_df']}")
    print(solution_dict["initial_selected_riders"])
    print(f"Total Expected Points: {solution_dict['total_expected_points']}")
    


if __name__ == "__main__":
    main()