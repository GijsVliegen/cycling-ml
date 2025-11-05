import numpy as np
from pygam import s, LinearGAM
import matplotlib.pyplot as plt
import math
import pprint
import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sympy as sp
from typing import List, Dict, Tuple, Callable, Any, Optional
import mlflow
import mlflow.pyfunc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost.callback import EvaluationMonitor

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

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            early_stopping_rounds=100,
            eval_metric="logloss",
            callbacks=[EvaluationMonitor(show_stdv=True)]
        )

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

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss"
        )           

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss"
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtest, "test")],
            # early_stopping_rounds=50,
            verbose_eval=True
        )
        self.bst = bst
        return
    
    def to_xgboost_format(self, riders_data: pl.DataFrame, race_data: pl.DataFrame) -> np.ndarray:
        X = []
        y = []
        all_race_ids: list[int] = race_data.select(
            ["race_id"]
        ).unique().to_series().to_list()
        nr_races = 100 #len(all_race_ids) #100
        nr_pairs_per_race = 100

        #TODO: VECTORIZE THIS
        for race_id in all_race_ids[:nr_races]:
            rider_pairs = self.get_rider_pairs(
                race_id=race_id,
                race_data=race_data
            )
            rider_pairs = rider_pairs[:min(nr_pairs_per_race, len(rider_pairs))]
            race_year = race_data.filter(
                pl.col("race_id") == race_id
            ).select("year").to_numpy()[0][0]

            race_Y = np.random.choice([0, 1], size=(len(rider_pairs),))

            rider_pair_features = self.get_rider_pair_features(
                rider_pairs=rider_pairs,
                riders_data=riders_data,
                race_year=race_year,
                # race_Y=race_Y
            )

            race_features = race_data.filter(
                pl.col("race_id") == race_id
            ).select(self.race_features).to_numpy()[0]
            race_features = np.tile(race_features, (len(rider_pairs), 1))

            race_X = np.hstack([rider_pair_features, race_features])
            X.append(race_X)
            y.append(race_Y)

        y = np.vstack(y).flatten()
        X = np.vstack(X)
        print(f"training on {len(y)} pairs")

        return X, y

    def get_rider_pair_features(
        self, 
        rider_pairs: list[tuple[str, str]],
        riders_data: pl.DataFrame,
        race_year: int,
        # race_Y: np.ndarray
    ) -> np.ndarray:    
        """
        if race_Y == 0, shift the pair features since the first rider in data was the first in the race
        """
        pair_features = []
        years_to_go_back = 3
        years_to_go_back = [
            str(race_year - i) for i in range(1, years_to_go_back + 1)
        ]

        for rider_0_name, rider_1_name in rider_pairs: #TODO: make more efficient
            rider_0_data = riders_data.filter(
                pl.col("name") == rider_0_name
            ).filter(
                pl.col("year").is_in(years_to_go_back)
            ).select(self.rider_features).to_numpy().flatten().astype(np.float32)

            rider_1_data = riders_data.filter(
                pl.col("name") == rider_1_name
            ).filter(
                pl.col("year").is_in(years_to_go_back)
            ).select(self.rider_features).to_numpy().flatten().astype(np.float32)

            target_len = len(years_to_go_back) * len(self.rider_features)

            rider_0_data = np.pad(rider_0_data, (0, target_len - len(rider_0_data)), mode='constant', constant_values=np.NAN)
            rider_1_data = np.pad(rider_1_data, (0, target_len - len(rider_1_data)), mode='constant', constant_values=np.NAN)
            features = np.concatenate([rider_0_data, rider_1_data])
            pair_features.append(features)

        pair_features = np.array(pair_features)
        # year_cols = [i * len(self.rider_features) for i in range(len(years_to_go_back) * 2)]
        feature_cols = pair_features[:, 0:len(rider_0_data)] - pair_features[:, len(rider_0_data):] 


        #shift pair depending on y value
        # X_shifted = pair_features.copy()
        # shift_mask = race_Y == 0
        # shift_length = pair_features.shape[1] // 2
        # pair_features[shift_mask, :shift_length] = X_shifted[shift_mask, shift_length:]
        # pair_features[shift_mask, shift_length:] = X_shifted[shift_mask, :shift_length]


        return feature_cols

    def get_rider_pairs(self, race_id: int, race_data: pl.DataFrame) -> list[tuple[str, str]]:
        """
        pairs of rider names, rider 0 is the favorite

        selects pairs if first rider is in top 25

        returns names only
        """
        top_riders = race_data.filter(
            pl.col("race_id") == race_id
        ).sort("rank").head(25)
        all_riders = race_data.filter(
            pl.col("race_id") == race_id
        ).sort("rank")            
        pairs = []
        for i, top_rider in enumerate(top_riders["name"]):
            for j, other_rider in enumerate(all_riders["name"]):
                if i < j:
                    pairs.append((top_rider, other_rider))
        return pairs


    def split_train_test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        X_test = X[test_mask][:, :-2]
        y_test = y[test_mask]

        return X_train, y_train, X_test, y_test

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        dtest = xgb.DMatrix(X_test, label=y_test)
        y_pred = self.bst.predict(dtest)
        accuracy = np.mean(y_pred == y_test)
        auc = roc_auc_score(y_test, self.bst.predict(dtest))
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc:.4f}")


    # y_pred = (y_pred_proba > 0.5).astype(int)

    # # Compute metrics
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, zero_division=0)
    # recall = recall_score(y_test, y_pred, zero_division=0)
    # f1 = f1_score(y_test, y_pred, zero_division=0)
    # auc = roc_auc_score(y_test, y_pred_proba)
    # cm = confusion_matrix(y_test, y_pred)

    # # Print results
    # print("=== Evaluation Metrics ===")
    # print(f"Accuracy:  {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall:    {recall:.4f}")
    # print(f"F1 Score:  {f1:.4f}")
    # print(f"AUC:       {auc:.4f}")
    # print("Confusion Matrix:")
    # print(cm)
    # print("\nDetailed classification report:")
    # print(classification_report(y_test, y_pred, digits=4))

        return

    
    def train_model(self, riders_data: pl.DataFrame, race_data: pl.DataFrame) -> None:
        X, y = self.to_xgboost_format(riders_data, race_data)
        X_train, y_train, X_test, y_test = self.split_train_test(X, y)
        self.fit(X_train, y_train, X_test, y_test)
        print("Model fitted")
        self.evaluate_model(X_test, y_test)
        print("Model evaluated")

def main():
    riders_data = pl.read_parquet("data/rider_data.parquet")
    race_data = pl.read_parquet("data/features_df.parquet")
    model = RaceModel()
    model.train_model(riders_data, race_data)

if __name__ == "__main__":
    main()