import itertools
import numpy as np
import matplotlib.pyplot as plt
import math
import pprint
import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sympy as sp
from typing import Tuple

import random
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost.callback import EvaluationMonitor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

class RaceModel:
    """
    XGboost on pairs of riders predicting who will win
    """

    def __init__(self) -> None:
        """
        Initializes the RaceModel model.

        Args:
            X: Training data array (n_samples, n_features).
        """

        #### features_df ####
        # 0-4: name  ┆ race_id ┆ distance_km ┆ elevation_m ┆ profile_score ┆ 
        # 5-9: profile_score_last_25k ┆ classification ┆ date ┆ rank  ┆ startlist_score ┆ 
        # 10-14: age | rank_bucket | rank_normalized | age_normalized | year
        # 15-19: rank_bucket_year_count | top25_count_year | top25_count | attended_races | classification_encoded
        # 20-24: nr_riders

        #### riders_data ####
        # 0-4: name ┆ year ┆ score ┆ rank ┆ Onedayraces ┆ 
        # 5-9: GC ┆ TT ┆ Sprint ┆ Climber ┆ Hills

        self.rider_features = [
            # "score", 
            # "rank",
            "Onedayraces",
            "GC",
            "TT",
            "Sprint",
            "Climber",
            "Hills",
        ]
        self.rider_yearly_features = [
            "points",
            "racedays",
            "kms",
            "wins",
            "top_3s",
            "top_10s",
        ]
        self.race_features = [ 
            "distance_km",
            "elevation_m",
            "profile_score",
            "profile_score_last_25k",
            "classification_encoded",
            "year",
            "startlist_score", #Check present in pairs
        ]      

    def save_model(self) -> None:
        self.bst.save_model("data_v2/xgboost_model.json")

    def load_model(self) -> None:
        self.bst = xgb.Booster()
        self.bst.load_model("data_v2/xgboost_model.json")
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, train_weights: np.ndarray, test_weights: np.ndarray) -> None:


        dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
        dtest = xgb.DMatrix(X_test, label=y_test, weight=test_weights)
        del X_train, X_test #._test cannot be deleted

        epochs = []
        test_errors = []
        train_errors = []
        class PrintIterationCallback(xgb.callback.TrainingCallback):
            def after_iteration(xgb_self, model, epoch, evals_log):
                # if epoch % 1 == 0:
                test_y_pred_proba = model.predict(dtest)
                train_y_pred_proba = model.predict(dtrain)

                test_average_error = mean_absolute_error(y_test, test_y_pred_proba)
                train_average_error = mean_absolute_error(y_train, train_y_pred_proba)
                epochs.append(epoch)
                test_errors.append(test_average_error)
                train_errors.append(train_average_error)
                # print(f"Iteration {epoch}, F1 Score: {f1:.4f}")

                # Return False to continue training, True to stop
                return False
            
        bst = xgb.train(
            dtrain = dtrain,
            num_boost_round=1000,
            evals=[],
            params = {
                "eval_metric": "logloss",
                "min_child_weight": 10,
                "max_depth": 6,
                "tree_method": "hist",
                "max_bin": 128,
                # "subsample": 0.05,
            },
            callbacks=[PrintIterationCallback()],
        )
        self.bst = bst

        self.print_training_progress(epochs = epochs, test_errors = test_errors, train_errors = train_errors)
        return

    def print_training_progress(self, epochs, test_errors, train_errors):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, test_errors, label='Test Error', color='blue')
        plt.plot(epochs, train_errors, label='Train Error', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training Progress')
        plt.legend()
        plt.grid()
        plt.show()
        return
    
    def to_xgboost_format(
        self, 
        riders_data: pl.DataFrame, 
        result_features_df: pl.DataFrame, 
        races_features_df: pl.DataFrame,
        riders_yearly_data: pl.DataFrame
    ) -> np.ndarray:

        X = []
        y = []
        X_weights = []
        all_race_ids: list[int] = result_features_df.select(
            ["race_id"]
        ).unique().to_series().to_list()
        nr_races = len(all_race_ids) #100
        nr_pairs_per_race = 100

        #TODO: VECTORIZE THIS if possible?
        for race_id in all_race_ids[:nr_races]:
            race_year = result_features_df.filter(
                pl.col("race_id") == race_id
            ).select("year").to_numpy()[0][0]

            if race_year < 2016: #DO not take early data, since relies on data for 3 years back
                continue

            rider_pairs, pair_weights = self.get_rider_pairs_weights(
                race_id=race_id,
                result_features_df=result_features_df
            )
            rider_pairs = rider_pairs#[:min(nr_pairs_per_race, len(rider_pairs))]

            race_Y = np.random.choice([0, 1], size=(len(rider_pairs),))

            rider_pair_features = self.get_rider_pair_features(
                rider_pairs=rider_pairs,
                riders_data=riders_data,
                riders_yearly_data = riders_yearly_data,
                race_year=race_year,
                race_Y=race_Y
            )

            race_features = result_features_df.filter(
                pl.col("race_id") == race_id
            ).select(self.race_features).to_numpy()[0].astype(np.float32, copy=False)
            race_features = np.tile(race_features, (len(rider_pairs), 1))

            race_X = np.hstack([rider_pair_features, race_features])
            X.append(race_X)
            X_weights.extend(pair_weights)
            y.append(race_Y)

        y = np.concatenate(y)
        X = np.vstack(X)
        X_weights = np.array(X_weights)
        print(f"training on {len(y)} pairs")

        return X, y, X_weights

    def get_rider_pair_features(
        self, 
        rider_pairs: list[tuple[str, str]],
        riders_data: pl.DataFrame,
        riders_yearly_data: pl.DataFrame,
        race_year: int,
        race_Y: np.ndarray
    ) -> np.ndarray:    
        """
        rider_pairs, list of tuples of rider names, for a whole race (thousands of pairs)

        returns np.ndarray of shape (nr_pairs, nr_features + nr_years * nr_yearly_features)
        """

        nr_years = 3
        years_to_go_back = [
            str(race_year - i) for i in range(1, nr_years + 1)
        ]

        def get_rider_data(rider_filters: pl.DataFrame) -> tuple[np.array, np.array]:
            rider_0_data = riders_data.join(
                rider_filters.unique(subset=["name_0","name_1"]).select("name_0"),
                left_on=["name"],
                right_on=["name_0"],
                how="right"
            ).fill_null(
                np.NAN
            ).select(self.rider_features)
            rider_0_data: np.array = rider_0_data.to_numpy().astype(np.float32).astype(np.float32, copy=False)
            
            rider_1_data: pl.DataFrame = riders_data.join(
                rider_filters.unique(subset=["name_0","name_1"]).select("name_1"),
                left_on=["name"],
                right_on=["name_1"],
                how="right"
            ).fill_null(
                np.NAN
            ).select(self.rider_features)
            rider_1_data: np.array = rider_1_data.to_numpy().astype(np.float32).astype(np.float32, copy=False)
            return rider_0_data, rider_1_data

        def get_rider_yearly_data(rider_filters: pl.DataFrame) -> tuple[np.array, np.array]:
            rider_0_yearly_data: pl.DataFrame = riders_yearly_data.join(
                rider_filters,
                left_on=["name", "season"],
                right_on=["name_0", "year"],
                how="right" #preserve order of pairs
            ).fill_null(
                np.NAN
            ).select(self.rider_yearly_features)
            rider_0_yearly_data: np.array = rider_0_yearly_data.to_numpy().astype(np.float32, copy=False).reshape(len(rider_pairs), nr_years * len(self.rider_yearly_features))

            rider_1_yearly_data: pl.DataFrame = riders_yearly_data.join(
                rider_filters,
                left_on=["name", "season"],
                right_on=["name_1", "year"],
                how="right" #preserve order of pairs
            ).fill_null(
                np.NAN
            ).select(self.rider_yearly_features)
            rider_1_yearly_data: np.array = rider_1_yearly_data.to_numpy().astype(np.float32, copy=False).reshape(len(rider_pairs), nr_years * len(self.rider_yearly_features))
            return rider_0_yearly_data, rider_1_yearly_data

        rider_filters = pl.DataFrame(
            {
                "name_0": rider_pair_names[0],
                "name_1": rider_pair_names[1],
                "year": year
            } if y == 1 else {
                "name_0": rider_pair_names[1],
                "name_1": rider_pair_names[0],
                "year": year
            }
            for rider_pair_names, y in zip(rider_pairs, race_Y)
            for year in years_to_go_back
        )
        rider_0_data, rider_1_data = get_rider_data(rider_filters)
        rider_0_yearly_data, rider_1_yearly_data = get_rider_yearly_data(rider_filters)

        yearly_feature_cols = rider_0_yearly_data - rider_1_yearly_data

        feature_cols = rider_0_data - rider_1_data
        all_feat_cols = np.hstack([feature_cols, yearly_feature_cols])
        
        return all_feat_cols

    def get_rider_pairs_weights(self, race_id: int, result_features_df: pl.DataFrame, min_top_rank: int = 25) -> list[tuple[str, str, float]]:
        """
        returns pairs of rider names and weight of importance, rider 0 is the favorite

        selects pairs if first rider is in top 25

        returns 
            - names in the pair
            - weights: If a pair has weight w, then in the loss it behaves like it appears w times.
                - weight of highest rank rider
                - rank 1: w = max_weight, rank 25: w = 1
        """
        max_weight = 10
        nr_riders = result_features_df.filter(pl.col("race_id") == race_id).height

        if min_top_rank is None:
            min_top_rank = nr_riders

        top_riders = result_features_df.filter(
            pl.col("race_id") == race_id
        ).sort("rank").head(min_top_rank)
        all_riders = result_features_df.filter(
            pl.col("race_id") == race_id
        ).sort("rank")            
        pairs = []
        weights = []
        for i, top_rider in enumerate(top_riders["name"]):
            for j, other_rider in enumerate(all_riders["name"]):
                if i < j:
                    pairs.append((top_rider, other_rider))
                    """Weights explained:
                    - abs(i,j) -> larger distance should be easier to guess so more weight to these
                    - max_weight - i -> predicting first place more important"""
                    # if i == 10 and j :
                        # print(1 + (nr_riders - i)*(nr_riders - abs(i-j)))
                    weights.append(1 + (nr_riders - i)*(nr_riders - abs(i-j))) 
        return pairs, weights

    def get_rider_pair_win_diff(self, rider_0: str, rider_1: str, race_data: pl.DataFrame) -> int:
        #filter race_data on rider names

        #add boolean col if rider_0 present in that race
        #add boolean col if rider_1 present in

        #filter on both present
        #compare ranks, count relative nr of wins for rider_0

        race_data = race_data.filter(
            (pl.col("name") == rider_0) | (pl.col("name") == rider_1)
        )
        race_data_with_rank_0 = race_data.with_columns(
            pl.when(pl.col("name") == rider_0).then(pl.col("rank")).otherwise(-1).alias("rider_0_rank"),
        )

        #filter on both present
        race_data_with_both = race_data_with_rank_0.filter(
            (pl.col("name") == rider_1) & (pl.col("rider_0_rank") != -1)
        )

        #compare ranks, count relative nr of wins for rider_0
        rider_0_wins = race_data_with_both.filter(
            pl.col("rider_0_rank") < pl.col("rank")
        ).height

        return rider_0_wins / race_data_with_both.height

    def rerank_top_25(self, rider_score_dict: dict[str, float], race_data: pl.DataFrame) -> pl.DataFrame:
        """
        reranks top 25 riders taking into account their historical pairwise win rates
        """
        riders = list(rider_score_dict.keys())[:25]
        rider_pairs = itertools.product(riders, repeat=2)
        rider_win_diffs = {rider: [] for rider in riders}
        for rider_0, rider_1 in rider_pairs:
            win_diff = self.get_rider_pair_win_diff(rider_0=rider_0, rider_1=rider_1, race_data=race_data)
            rider_win_diffs[rider_0].append(win_diff)
            rider_win_diffs[rider_1].append(1 - win_diff)
        rider_avg_win_diffs = {
            rider: np.mean(win_diffs) for rider, win_diffs in rider_win_diffs.items()
        }
            # win_diffs.append((rider_0, rider_1, win_diff))
        return rider_avg_win_diffs

    def split_train_test(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        splits data based on year, year is second to last column

        last 20% of years as test set
        this is done by sorting the columns and splitting
        """
        years = X[:, -2]
        unique_years = np.unique(years)
        split_year = unique_years[int(len(unique_years) * 0.8)]

        train_mask = years < split_year
        test_mask = years >= split_year

        X_train = X[train_mask][:, :-2]
        y_train = y[train_mask]
        train_weights = weights[train_mask]

        X_test = X[test_mask][:, :-2]
        y_test = y[test_mask]
        test_weights = weights[test_mask]

        return X_train, y_train, train_weights, X_test, y_test, test_weights

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, test_weights: np.ndarray) -> None:
        dtest = xgb.DMatrix(X_test, label=y_test, weight=test_weights)
        y_pred_proba = self.bst.predict(dtest)
        # accuracy = np.mean(y_pred == y_test)
        # auc = roc_auc_score(y_test, self.bst.predict(dtest))
        # print(f"Test Accuracy: {accuracy:.4f}")
        # print(f"Test AUC: {auc:.4f}")


        """
        y_pred explained:
        - by default, threshold is 0.5
        - here, we set threshold to 0.2 to increase recall (catch more true positives)
        """
        y_pred = (y_pred_proba > 0.2).astype(int) 

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        # Print results
        print("=== Evaluation Metrics ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("\nDetailed classification report:")
        print(classification_report(y_test, y_pred, digits=4))

        return

    def train_model(self, riders_data: pl.DataFrame, result_features_df: pl.DataFrame, riders_yearly_data: pl.DataFrame) -> None:
        X, y, x_weights = self.to_xgboost_format(riders_data, result_features_df, riders_yearly_data)
        X_train, y_train, train_weights, X_test, y_test, test_weights = self.split_train_test(X, y, x_weights)
        print("Starting model fit")
        self.fit(
            X_train, 
            y_train, 
            X_test, 
            y_test, 
            train_weights, 
            test_weights,
            )
        print("Model fitted")
        self.evaluate_model(X_test, y_test, test_weights=test_weights)
        print("Model evaluated")

    def predict_race(self, riders_data: pl.DataFrame, result_features_df: pl.DataFrame, riders_yearly_data: pl.DataFrame,race_id: str):
        rider_pairs, _ = self.get_rider_pairs_weights(
            race_id = race_id,
            result_features_df = result_features_df,
            min_top_rank = None
        )
        race_year = result_features_df.filter(
            pl.col("race_id") == race_id
        ).select("year").to_numpy()[0][0]
        race_Y = np.ones(len(rider_pairs))  #dummy Y
        rider_features = self.get_rider_pair_features(
            rider_pairs = rider_pairs,
            riders_data = riders_data,
            riders_yearly_data=riders_yearly_data,
            race_year = race_year,
            race_Y = race_Y
        )
        dtest = xgb.DMatrix(rider_features, label=race_Y)
        y_pred_proba = self.bst.predict(dtest)
        # pair_outcomes = self.bst.predict(rider_features)  #probability that rider 0 wins
        riders = set(
            [rider for pair in rider_pairs for rider in pair]
        )
        rider_scores = {
            rider: [] for rider in riders
        }
        for (first, second), pred_y in zip(rider_pairs, y_pred_proba):
            rider_scores[first].append( 0 + pred_y)
            rider_scores[second].append(1 - pred_y)
        rider_scores = {
            rider: np.sum(scores) for rider, scores in rider_scores.items()
        }
        top10 = sorted(rider_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        # Print nicely
        for name, score in top10:
            print(f"{name}: {score:.4f}")

        pass

def train():
    riders_data = pl.read_parquet("data_v2/rider_stats_df.parquet")
    result_features_df = pl.read_parquet("data_v2/result_features_df.parquet")
    riders_yearly_data = pl.read_parquet("data_v2/rider_yearly_stats_df.parquet")
    model = RaceModel()
    model.train_model(riders_data, result_features_df, riders_yearly_data)
    model.save_model()

def predict():
    riders_data = pl.read_parquet("data_v2/rider_stats_df.parquet")
    result_features_df = pl.read_parquet("data_v2/result_features_df.parquet")
    riders_yearly_data = pl.read_parquet("data_v2/rider_yearly_stats_df.parquet")
    model = RaceModel()
    model.load_model()

    # races = pl.read_parquet("data/races_df.parquet")
    # print(races.filter(pl.col("date") == "2024-07-20").head(5))
    rvv_id = "R487e2808"
    tdf_2024_20_id = "R5e8c7e1a"
    model.predict_race(
        riders_data=riders_data,
        result_features_df=result_features_df,
        riders_yearly_data=riders_yearly_data,
        race_id=rvv_id
    )

    model.predict_race(
        riders_data=riders_data,
        result_features_df=result_features_df,
        riders_yearly_data=riders_yearly_data,
        race_id=tdf_2024_20_id
    )

def main():
    train()
    predict()

if __name__ == "__main__":
    main()