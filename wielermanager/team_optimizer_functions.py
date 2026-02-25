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



def solve_team_selection(race_dfs: list[pl.DataFrame], cost_df: pl.DataFrame):

    
    
    # ==========================================
    # 1. Aggregate all race expected points
    # ==========================================
    
    race_tables = []
    
    for i, df in enumerate(race_dfs):
        race_tables.append(
            df.select(["name", "expected_points"]).rename({"expected_points": f"race_{i}"})
        )
    
    # Join all race dfs on name
    all_points = race_tables[0]
    for df in race_tables[1:]:
        all_points = all_points.join(
            df, 
            on="name", 
            how="outer",
            coalesce = True
        )
    
    # Fill missing values with 0
    all_points = all_points.fill_null(0)
    
    # Join with cost
    data = all_points.join(cost_df, on="name", how="left").fill_null(2) #default cost is 2
    
    riders = data["name"].to_list()
    races = [col for col in data.columns if col.startswith("race_")]
    
    # Create lookup dictionaries
    cost = dict(zip(data["name"], data["cost"]))
    
    expected_points = {
        r: {
            race: data.filter(pl.col("name") == r)[race][0]
                if not math.isnan(data.filter(pl.col("name") == r)[race][0]) else 0
            for race in races
        }
        for r in riders
    }
    
    # ==========================================
    # 2. Build MILP Model
    # ==========================================
    
    model = pulp.LpProblem("CyclingTeamOptimization", pulp.LpMaximize)
    
    # Binary variable: select rider
    x = pulp.LpVariable.dicts("select", riders, cat="Binary")
    
    # Binary variable: rider counts in race top 12
    y = pulp.LpVariable.dicts(
        "counted",
        [(r, race) 
         for r in riders for race in races],
        cat="Binary"
    )
    
    # ==========================================
    # 3. Objective
    # ==========================================
    
    model += pulp.lpSum(
        expected_points[r][race] * y[(r, race)]
        for r in riders
        for race in races
    )
    
    # ==========================================
    # 4. Constraints
    # ==========================================
    
    # Budget constraint
    model += pulp.lpSum(cost.get(r, 2) * x[r] for r in riders) <= BUDGET
    
    # Exactly 20 riders
    model += pulp.lpSum(x[r] for r in riders) == TEAM_SIZE
    
    # For each race: at most TOP_K riders count
    for race in races:
        model += pulp.lpSum(y[(r, race)] for r in riders) <= TOP_K
    
    # A rider can only count if selected
    for r in riders:
        for race in races:
            model += y[(r, race)] <= x[r]
    
    # ==========================================
    # 5. Solve
    # ==========================================
    
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # ==========================================
    # 6. Extract Solution
    # ==========================================
    
    selected_riders = [r for r in riders if x[r].value() == 1]
    
    total_cost = sum(cost[r] for r in selected_riders)
    
    # Calculate total expected score
    total_points = sum(
        expected_points[r][race] * y[(r, race)].value()
        for r in riders
        for race in races
    )
    
    # Create output DataFrame
    result_df = (
        data
        .filter(pl.col("name").is_in(selected_riders))
        .select(["name", "cost"])
        .sort("cost", descending=True)
    )
    
    return result_df, total_cost, total_points

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
    team_selection, total_cost, total_points = solve_team_selection(
        race_dfs = all_race_preds, 
        cost_df = rider_cost_df
    )
    print("Selected Team:")
    print(team_selection)
    print(f"Total Cost: {total_cost}")
    print(f"Total Expected Points: {total_points}")


if __name__ == "__main__":
    main()