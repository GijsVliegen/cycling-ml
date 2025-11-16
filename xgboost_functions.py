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


        self.rider_feature_idxs = [2, 3, 4, 5, 6, 7, 8, 9]
        self.rider_features = [
            "score", 
            "rank",
            "Onedayraces",
            "GC",
            "TT",
            "Sprint",
            "Climber",
            "Hills",
        ]
        self.race_features_idxs = [2, 3, 4, 5, 6, 14, 10]
        self.race_features = [ 
            "distance_km",
            "elevation_m",
            "profile_score",
            "profile_score_last_25k",
            "classification_encoded",
            "year",
            "startlist_score", #Check present in pairs
        ]       


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, train_weights: np.ndarray, test_weights: np.ndarray) -> None:


        dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
        dtest = xgb.DMatrix(X_test, label=y_test, weight=test_weights)


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
            num_boost_round=200,
            evals=[],
            params = {
                "eval_metric": "logloss",
                "min_child_weight": 1,
                "max_depth": 6,
                "subsample": 0.5,
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
    
    def to_xgboost_format(self, riders_data: pl.DataFrame, race_data: pl.DataFrame) -> np.ndarray:
        X = []
        y = []
        X_weights = []
        all_race_ids: list[int] = race_data.select(
            ["race_id"]
        ).unique().to_series().to_list()
        nr_races = len(all_race_ids) #100
        nr_pairs_per_race = 100

        #TODO: VECTORIZE THIS
        for race_id in all_race_ids[:nr_races]:
            race_year = race_data.filter(
                pl.col("race_id") == race_id
            ).select("year").to_numpy()[0][0]

            if race_year < 2018:
                continue

            rider_pairs, pair_weights = self.get_rider_pairs(
                race_id=race_id,
                race_data=race_data
            )
            rider_pairs = rider_pairs#[:min(nr_pairs_per_race, len(rider_pairs))]

            race_Y = np.random.choice([0, 1], size=(len(rider_pairs),))

            rider_pair_features = self.get_rider_pair_features(
                rider_pairs=rider_pairs,
                riders_data=riders_data,
                race_year=race_year,
                race_Y=race_Y
            )

            race_features = race_data.filter(
                pl.col("race_id") == race_id
            ).select(self.race_features).to_numpy()[0]
            race_features = np.tile(race_features, (len(rider_pairs), 1))

            race_X = np.hstack([rider_pair_features, race_features])
            X.append(race_X)
            X_weights.append(pair_weights)
            y.append(race_Y)

        y = np.concatenate(y)
        X = np.vstack(X)
        X_weights = np.vstack(X_weights)
        print(f"training on {len(y)} pairs")

        return X, y

    def get_rider_pair_features(
        self, 
        rider_pairs: list[tuple[str, str]],
        riders_data: pl.DataFrame,
        race_year: int,
        race_Y: np.ndarray
    ) -> np.ndarray:    
        """
        TODO: if race_Y == 0, shift the pair since the first rider in data was the first in the race
        """
        pair_features = []
        years_to_go_back = 3
        years_to_go_back = [
            str(race_year - i) for i in range(1, years_to_go_back + 1)
        ]

        rider_0_filters = pl.DataFrame(
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

        rider_0_data: pl.DataFrame = riders_data.join(
            rider_0_filters,
            left_on=["name", "year"],
            right_on=["name_0", "year"],
            how="right" #preserve order of pairs
        ).fill_null(
            np.NAN
        ).select(self.rider_features)
        rider_0_data: np.array = rider_0_data.to_numpy().astype(np.float32).reshape(len(rider_pairs), 3 * len(self.rider_features))

        rider_1_data: pl.DataFrame = riders_data.join(
            rider_0_filters,
            left_on=["name", "year"],
            right_on=["name_1", "year"],
            how="right" #preserve order of pairs
        ).fill_null(
            np.NAN
        ).select(self.rider_features)
        rider_1_data: np.array = rider_1_data.to_numpy().astype(np.float32).reshape(len(rider_pairs), 3 * len(self.rider_features))

            # for rider_0_name, rider_1_name in rider_pairs: #TODO: make more efficient
            #     rider_0_data = riders_data.filter(
            #         pl.col("name") == rider_0_name
            #     ).filter(
            #         pl.col("year").is_in(years_to_go_back)
            #     ).select(self.rider_features).to_numpy().flatten().astype(np.float32)

            #     rider_1_data = riders_data.filter(
            #         pl.col("name") == rider_1_name
            #     ).filter(
            #         pl.col("year").is_in(years_to_go_back)
            #     ).select(self.rider_features).to_numpy().flatten().astype(np.float32)

            #     target_len = len(years_to_go_back) * len(self.rider_features)

            #     rider_0_data = np.pad(rider_0_data, (0, target_len - len(rider_0_data)), mode='constant', constant_values=np.NAN)
            #     rider_1_data = np.pad(rider_1_data, (0, target_len - len(rider_1_data)), mode='constant', constant_values=np.NAN)
            #     features = np.concatenate([rider_0_data, rider_1_data])
            #     pair_features.append(features)

            # pair_features = np.array(pair_features)
        # year_cols = [i * len(self.rider_features) for i in range(len(years_to_go_back) * 2)]
        feature_cols = rider_0_data - rider_1_data


        #shift pair depending on y value
        # X_shifted = pair_features.copy()
        # shift_mask = race_Y == 0
        # shift_length = pair_features.shape[1] // 2
        # pair_features[shift_mask, :shift_length] = X_shifted[shift_mask, shift_length:]
        # pair_features[shift_mask, shift_length:] = X_shifted[shift_mask, :shift_length]


        return feature_cols

    def get_rider_pairs(self, race_id: int, race_data: pl.DataFrame, min_top_rank: int = 25) -> list[tuple[str, str, float]]:
        """
        returns pairs of rider names and weight of importance, rider 0 is the favorite

        selects pairs if first rider is in top 25

        returns 
            - names in the pair
            - weights: If a pair has weight w, then in the loss it behaves like it appears w times.
                - weight of highest rank rider
                - rank 1: w = 2, rank 25: w = 1
        """
        if min_top_rank is None:
            nr_riders = race_data.filter(pl.col("race_id") == race_id).height
            min_top_rank = nr_riders

        top_riders = race_data.filter(
            pl.col("race_id") == race_id
        ).sort("rank").head(min_top_rank)
        all_riders = race_data.filter(
            pl.col("race_id") == race_id
        ).sort("rank")            
        pairs = []
        weights = []
        for i, top_rider in enumerate(top_riders["name"]):
            for j, other_rider in enumerate(all_riders["name"]):
                if i < j:
                    pairs.append((top_rider, other_rider))
                    weights.append(1 + (min_top_rank - (i + 1)) / min_top_rank)  #weight between 1 and 2
        return pairs, weights


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


        y_pred = (y_pred_proba > 0.5).astype(int)

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

    def train_model(self, riders_data: pl.DataFrame, race_data: pl.DataFrame) -> None:
        X, y, x_weights = self.to_xgboost_format(riders_data, race_data)
        X_train, y_train, train_weights, X_test, y_test, test_weights = self.split_train_test(X, y, x_weights)
        self.fit(X_train, y_train, X_test, y_test, train_weights, test_weights)
        print("Model fitted")
        self.evaluate_model(X_test, y_test, test_weights=test_weights)
        print("Model evaluated")

    def predict_race(self, riders_data: pl.DataFrame, race_data: pl.DataFrame, race_id: str):
        rider_pairs = self.get_rider_pairs(
            race_id = race_id,
            race_data = race_data,
            min_top_rank = None
        )
        #TODO: add some random switching of pairs 
        race_year = race_data.filter(
            pl.col("race_id") == race_id
        ).select("year").to_numpy()[0][0]
        race_Y = np.ones(len(rider_pairs))  #dummy
        rider_features = self.get_rider_pair_features(
            rider_pairs = rider_pairs,
            riders_data = riders_data,
            race_year = race_year,
            race_Y = race_Y
            # rider_pairs: list[tuple[str, str]],
            # riders_data: pl.DataFrame,
            # race_year: int,
            # race_Y: np.ndarray
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

def main():
    riders_data = pl.read_parquet("data/rider_data.parquet")
    race_data = pl.read_parquet("data/features_df.parquet")
    model = RaceModel()
    model.train_model(riders_data, race_data)

    # races = pl.read_parquet("data/races_df.parquet")
    # print(races.filter(pl.col("date") == "2024-07-20").head(5))
    rvv_id = "R487e2808"
    tdf_2024_20_id = "R5e8c7e1a"
    model.predict_race(
        riders_data=riders_data,
        race_data=race_data,
        race_id=rvv_id
    )

    model.predict_race(
        riders_data=riders_data,
        race_data=race_data,
        race_id=tdf_2024_20_id
    )

if __name__ == "__main__":
    main()