from data_science.data_science_functions import main_test as features_main_test
from xgboost_functions import main_test as models_main_test
from wielermanager.wielermanager_functions import main_test as predictions_main_test
from wielermanager.team_optimizer_functions import main_test as team_main_test


def main():
    print("=== Stage 1: feature creation test ===")
    feature_summary = features_main_test(
        source_dir="data_v2",
        target_dir="data_test",
        max_races_per_year=12,
        nr_years=7,
    )
    print(feature_summary)

    print("=== Stage 2: model training test ===")
    model_summary = models_main_test(data_dir="data_test")
    print(model_summary)

    print("=== Stage 3: race prediction test ===")
    prediction_summary = predictions_main_test(
        source_dir="data_v2",
        target_dir="data_test",
        limit=6,
    )
    print(prediction_summary)

    print("=== Stage 4: team optimization test ===")
    team_summary = team_main_test(data_dir="data_test", limit=6)
    print({
        "initial_selected_riders": team_summary["initial_selected_riders"],
        "total_expected_points": team_summary["total_expected_points"],
        "transfers_in": team_summary["transfers_in"],
        "transfers_out": team_summary["transfers_out"],
    })


if __name__ == "__main__":
    main()
