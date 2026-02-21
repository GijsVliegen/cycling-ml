import math
from tkinter.font import names
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import json
import xgboost as xgb
from typing import Tuple
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    mean_absolute_error
)
from data_science_functions import (
    EMBEDDING_SIZE,
    RACE_SIMILARITY_COLS,
    calculate_cosine_similarity_polars,
    scores_to_probability_results,
    filter_data
)
from sklearn.metrics import ndcg_score
import numpy as np


with open("WIELERMANAGER_RULES.json") as f:
    rules = json.load(f)

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

        self.embed_features = [
            f"embed_{i}"
            for i in range(1, EMBEDDING_SIZE+1)
        ]
        self.rider_result_features = [
            'nr_races_participated_1110d',
            'nr_top25_1110d',
            'nr_top10_1110d',
            'nr_top3_1110d',
            "strength_1110d",
            'nr_races_participated_370d',
            'nr_top25_370d',
            'nr_top10_370d',
            'nr_top3_370d',
            "strength_370d",
            'nr_races_participated_40d',
            'nr_top25_40d',
            'nr_top10_40d',
            'nr_top3_40d',
            "strength_40d",
            "cosine_similarity",
            "l1_distance",
            "relative_team_strength_rank"
        ] + self.embed_features
        self.rider_yearly_features = [
            "points",
            "racedays",
            "kms",
            "wins",
        ]
        self.race_features = [ 
            "distance_km",
            "elevation_m",
            "profile_score",
            "profile_score_last_25k",
            "final_km_percentage",
            # "classification", #needs to be encoded
            "year",
            "startlist_score", #Check present in pairs
        ] + self.embed_features

    def save_model(self) -> None:
        self.bst.save_model("data_v2/xgboost_model.json")

    def load_model(self) -> None:
        self.bst = xgb.Booster()
        self.bst.load_model("data_v2/xgboost_model.json")
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, train_groups: np.ndarray, test_groups: np.ndarray) -> None:


        dtrain = xgb.DMatrix(X_train, label=y_train, group=train_groups)
        dtest = xgb.DMatrix(X_test, label=y_test, group=test_groups)
        del X_train, X_test #._test cannot be deleted

        epochs = []
        test_errors = []
        train_errors = []
        class PrintIterationCallback(xgb.callback.TrainingCallback):
            def after_iteration(xgb_self, model, epoch, evals_log):
                if epoch % 50 == 0:
                    test_y_pred = model.predict(dtest)
                    train_y_pred = model.predict(dtrain)

                    test_average_error = self.mean_ndcg(y_test, test_y_pred, test_groups)
                    train_average_error = self.mean_ndcg(y_train, train_y_pred, train_groups)
                    epochs.append(epoch)
                    test_errors.append(test_average_error)
                    train_errors.append(train_average_error)
                # print(f"Iteration {epoch}, F1 Score: {f1:.4f}")

                # Return False to continue training, True to stop
                return False
            
        bst = xgb.train(
            dtrain = dtrain,
            num_boost_round=1001,
            evals=[(dtrain, "train"), (dtest, "test")],
            params = {
                # "objective": "reg:squarederror",
                # "eval_metric": "rmse",
                # "objective": "binary:logistic",
                # "eval_metric": "logloss",
                "objective": "rank:ndcg",
                "eval_metric": "ndcg@25",
                "ndcg_exp_gain": False,
                "min_child_weight": 10,
                "max_depth": 6,
                "tree_method": "hist",
                # "max_bin": 3,
                "early_stopping_rounds": 50,
                "learning_rate": 0.1,   # ↓↓↓
                "subsample": 0.5,
            },
            callbacks=[PrintIterationCallback()], #If you want training progress plotted
        )
        self.bst = bst

        self.print_training_progress(epochs = epochs, test_errors = test_errors, train_errors = train_errors)
        return

    def print_training_progress(self, epochs, test_errors, train_errors):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, test_errors, label='Test ndcg', color='blue')
        plt.plot(epochs, train_errors, label='Train ndcg', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('ndcg')
        plt.title('Training Progress')
        plt.legend()
        plt.grid()
        plt.show()
        return
    
    def to_xgboost_format(
        self, 
        result_features_df: pl.DataFrame, 
        riders_yearly_data: pl.DataFrame,
        races_features_df: pl.DataFrame
    ) -> np.ndarray:

        X_train = []
        y_train = []
        train_groups = []
        X_test = []
        y_test = []
        test_groups = []
        result_features_df = result_features_df.unique(subset = ["race_id", "name"])
        all_race_ids: list[int] = result_features_df.join(
            races_features_df.select("race_id", "year"),
            on = "race_id",
            how = "inner"
        ).unique("race_id").sort("year").select(
            ["race_id"] 
        ).to_series().to_list() #Take race_ids from results since only care about races with results
       
        for race_id in all_race_ids:
            race_year = int(races_features_df.filter(
                pl.col("race_id") == race_id
            ).select("year").to_numpy()[0][0])
            race_results = result_features_df.filter(
                pl.col("race_id") == race_id
            ).sort("name")
            race_stats = races_features_df.filter(
                pl.col("race_id") == race_id
            )

            if race_year < 2016: #DO not take early data, since relies on data for 3 years back
                continue
            riders_features, rider_embeddings, ranks_to_predict = self.get_rider_feats(
                race_results = race_results,
                riders_yearly_data = riders_yearly_data,
                race_id = race_id,
                race_year=race_year,
            )

            race_features = race_stats.select(self.race_features).to_numpy()[0].astype(np.float32, copy=False)
            # race_embeddings = race_stats.select(self.embed_features).to_numpy()[0].astype(np.float32, copy=False)
            race_features = np.tile(race_features, (len(riders_features), 1))
            # race_embeddings = np.tile(race_embeddings, (len(riders_features), 1))
            # embedding_diff = np.abs(rider_embeddings - race_embeddings)
            race_X = np.hstack([riders_features, race_features])#, embedding_diff])
            if race_year < 2024:
                X_train.append(race_X)
                y_train.append(ranks_to_predict)
                train_groups.append(len(riders_features))
            else:
                X_test.append(race_X)
                y_test.append(ranks_to_predict)
                test_groups.append(len(riders_features))

        y_train = np.concatenate(y_train)
        X_train = np.vstack(X_train)
        y_test = np.concatenate(y_test)
        X_test = np.vstack(X_test)

        if any(y_test < 0) or any(y_train < 0):
            breakpoint()
            raise ValueError("Negative values found in ranks_to_predict, expected non-negative ranks")
        y_train = np.where(y_train <= 30, 1.0 / np.log2(y_train + 1), 0.0)
        y_test = np.where(y_test <= 30, 1.0 / np.log2(y_test + 1), 0.0)
        print(f"training on {len(y_train)} pairs")
        print(f"testing on {len(y_test)} pairs")

        return X_train, y_train, train_groups, X_test, y_test, test_groups
    
    def get_rider_feats(
        self, 
        race_results: pl.DataFrame,
        riders_yearly_data: pl.DataFrame,
        race_year: int,
        race_id: str,
        ranks = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:    
        """
        TODO
        """
        def get_rider_yearly_data(rider_filters: pl.DataFrame) -> tuple[np.array, np.array]:
            rider_yearly_data: pl.DataFrame = riders_yearly_data.join(
                rider_filters,
                left_on=["name", "season"],
                right_on=["name", "year"],
                how="right" #preserve order of pairs
            ).fill_null(
                np.NAN
            ).select(self.rider_yearly_features)
            rider_yearly_data: np.array = rider_yearly_data.to_numpy().astype(np.float32, copy=False).reshape(math.floor(len(rider_filters) / nr_years), nr_years * len(self.rider_yearly_features))

            return rider_yearly_data

        nr_years = 3
        years_to_go_back = [
            str(race_year - i) for i in range(1, nr_years + 1)
        ]
        rider_filters = race_results.select(
            "name"
        ).join(
            pl.DataFrame(
                {"year": years_to_go_back}
            ),
            how = "cross"
        ).sort("name")
        rider_yearly_data = get_rider_yearly_data(rider_filters)
        rider_data = race_results.select(self.rider_result_features).fill_null(
            np.NAN
        ).to_numpy().astype(np.float32, copy=False)
        rider_data[np.isnan(rider_data)] = 0.0
        rider_embeddings = race_results.select(self.embed_features).to_numpy().astype(np.float32, copy=False)
        if ranks:
            ranks_to_predict = race_results.select("rank").fill_null(
                np.NAN
            ).to_numpy().astype(np.float32, copy=False).flatten()
        else:
            ranks_to_predict = None

        if len(rider_filters) == 0:
            breakpoint()
        if rider_data.shape[0] != rider_yearly_data.shape[0]:
            raise ValueError("Feature columns shape does not match number of rider pairs")
        if ranks and any(np.isnan(ranks_to_predict)):
            raise ValueError("NaN values found in ranks_to_predict")
        
        rider_data = np.hstack([rider_data, rider_yearly_data])

        return rider_data, rider_embeddings, ranks_to_predict

    # def get_rider_pairs_values(
    #     self, 
    #     race_id: int, 
    #     result_features_df: pl.DataFrame, 
    #     min_top_rank: int = 25,
    #     all_pairs: bool = False
    # ) -> list[tuple[str, str, float]]:
    #     """
    #     returns pairs of rider names and weight of importance, rider 0 is the favorite

    #     selects pairs if first rider is in top 25

    #     returns 
    #         - names in the pair
    #         - value to predict (1 if first rider is better, -1 if second rider is better)
    #         - weights: If a pair has weight w, then in the loss it behaves like it appears w times.
    #             - weight = points for rank of first rider - points for rank of second rider, where points are defined by WEIGHTS_PER_RANK
    #     """
    #     nr_riders = result_features_df.filter(pl.col("race_id") == race_id).height

    #     if min_top_rank is None:
    #         min_top_rank = nr_riders
    #     if all_pairs:
    #         min_top_rank = nr_riders
    #     top_riders = result_features_df.filter(
    #         pl.col("race_id") == race_id
    #     ).sort("rank").head(min_top_rank)
    #     all_riders = result_features_df.filter(
    #         pl.col("race_id") == race_id
    #     ).sort("rank")        
    #     pairs = []
    #     values_to_predict = []
    #     weights = []
    #     for i, top_rider in enumerate(top_riders["name"]):
    #         for j, other_rider in enumerate(all_riders["name"]):
    #             if i < j:
    #                 if (top_rider, other_rider) not in pairs:
    #                     pairs.append((top_rider, other_rider))
    #                     pairs.append((other_rider, top_rider))
    #                     values_to_predict.append(1)
    #                     values_to_predict.append(-1)
    #                     weights.append(1)#WEIGHTS_PER_RANK.get(i+1, 1) - WEIGHTS_PER_RANK.get(j+1, 1))
    #                     weights.append(1)#WEIGHTS_PER_RANK.get(i+1, 1) - WEIGHTS_PER_RANK.get(j+1, 1))

                        
    #     return pairs, values_to_predict, weights

    def split_train_test(self, X: np.ndarray, y: np.ndarray, groups: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        splits data based on year, year is second to last column

        last 20% of years as test set
        this is done by sorting the columns and splitting
        """
        years = X[:, -2]
        unique_years = np.unique(years)
        split_year = unique_years[int(len(unique_years) * 0.9)]
        if split_year < 2015 or split_year > 2025:
            breakpoint()
            raise ValueError(f"Split year {split_year} is out of expected range")

        train_mask = years < split_year
        test_mask = years >= split_year
        cumsum_groups = np.cumsum(groups)
        groups_split_index = np.searchsorted(cumsum_groups, np.sum(train_mask))
        breakpoint()
        X_train = X[train_mask]
        y_train = y[train_mask]
        train_groups = groups[:groups_split_index]

        X_test = X[test_mask]
        y_test = y[test_mask]
        test_groups = groups[groups_split_index:]
        return X_train, y_train, train_groups, X_test, y_test, test_groups 


    def mean_ndcg(self,y_test, y_pred, test_groups) -> float:
        start = 0
        ndcg_scores = []

        for group_size in test_groups:
            end = start + group_size
            
            y_true_group = y_test[start:end]
            y_pred_group = y_pred[start:end]
            
            # reshape for sklearn (expects 2D arrays)
            ndcg = ndcg_score(
                [y_true_group],
                [y_pred_group],
                k=25
            )
            
            ndcg_scores.append(ndcg)
            start = end
        return np.mean(ndcg_scores)

    def precision_ndcg(self,y_test, y_pred, test_groups):
        precision_scores = []
        start = 0
        for group_size in test_groups:
            end = start + group_size
            
            y_true_group = y_test[start:end]
            y_pred_group = y_pred[start:end]
            
            # predicted top 25
            top25_idx = np.argsort(-y_pred_group)[:25]
            
            # true top 25
            true_top25 = set(np.where(y_true_group > 0)[0])
            
            hits = sum(idx in true_top25 for idx in top25_idx)
            precision_scores.append(hits / 25)
            
            start = end

        print("Mean Precision@25:", np.mean(precision_scores))
    
    def recall_ndcg(self,y_test, y_pred, test_groups):
        recall_scores = []
        start = 0
        for group_size in test_groups:
            end = start + group_size
            
            y_true_group = y_test[start:end]
            y_pred_group = y_pred[start:end]
            
            top25_idx = np.argsort(-y_pred_group)[:25]
            true_top25 = set(np.where(y_true_group > 0)[0])
            
            hits = sum(idx in top25_idx for idx in true_top25)
            recall_scores.append(hits / max(len(true_top25), 1))
            
            start = end

        print("Mean Recall@25:", np.mean(recall_scores))

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, test_groups: np.ndarray) -> None:
        dtest = xgb.DMatrix(X_test, label=y_test, group=test_groups)
        y_pred = self.bst.predict(dtest)
        
        """
        y_pred explained:
        - by default, threshold is 0
        """
        ndcg_scores = self.mean_ndcg(y_test, y_pred, test_groups)

        print("Mean NDCG@25:", np.mean(ndcg_scores))
        self.precision_ndcg(y_test, y_pred, test_groups)
        self.recall_ndcg(y_test, y_pred, test_groups)

    def train_model(
            self, 
            result_features_df: pl.DataFrame, 
            riders_yearly_data: pl.DataFrame,
            races_features_df: pl.DataFrame
        ) -> None:
        X_train, y_train, train_groups, X_test, y_test, test_groups = self.to_xgboost_format(
            result_features_df=result_features_df, 
            riders_yearly_data=riders_yearly_data, 
            races_features_df = races_features_df 
        )
        # X_train, y_train, train_groups, X_test, y_test, test_groups = self.split_train_test(X, y, groups)
        print("Starting model fit")
        self.fit(
            X_train, 
            y_train, 
            X_test, 
            y_test, 
            train_groups, 
            test_groups,
        )
        print("Model fitted")



def train():
    result_features_df = pl.read_parquet("data_v2/result_features_df.parquet")
    results_embedded_df = pl.read_parquet("data_v2/results_embedded_df.parquet")
    races_embedded_df = pl.read_parquet("data_v2/races_embedded_df.parquet")
    riders_yearly_data = pl.read_parquet("data_v2/rider_yearly_stats_df.parquet")
    races_df = pl.read_parquet("data_v2/races_df.parquet")


    riders_yearly_data = riders_yearly_data.with_columns(
        pl.all().replace(-1, 0)
    )
    necessary_races, necessary_results = filter_data(
        races_df = races_df,
        results_df = result_features_df,
    )

    results_features = results_embedded_df.join(
        necessary_results,
        on = ["race_id", "name"],
        how="left"
    )
    races_features = races_embedded_df.join(
        necessary_races,
        on = ["race_id"],
        how="left"
    )
    model = RaceModel()
    model.train_model(
        result_features_df=results_features, 
        riders_yearly_data=riders_yearly_data, 
        races_features_df=races_features)
    model.save_model()


def evaluate():
    result_features_df = pl.read_parquet("data_v2/result_features_df.parquet")
    results_embedded_df = pl.read_parquet("data_v2/results_embedded_df.parquet")
    races_embedded_df = pl.read_parquet("data_v2/races_embedded_df.parquet")
    riders_yearly_data = pl.read_parquet("data_v2/rider_yearly_stats_df.parquet")
    races_df = pl.read_parquet("data_v2/races_df.parquet")
    
    riders_yearly_data = riders_yearly_data.with_columns(
        pl.all().replace(-1, 0)
    )
    necessary_races, necessary_results = filter_data(races_df, result_features_df)

    results_features = results_embedded_df.join(
        necessary_results,
        on = ["race_id", "name"],
        how="left"
    )
    races_features = races_embedded_df.join(
        necessary_races,
        on = ["race_id"],
        how="left"
    )
    model = RaceModel()
    model.load_model()

    """Evaluate model"""
    X_train, y_train, train_groups, X_test, y_test, test_groups  = model.to_xgboost_format(
        result_features_df=results_features, 
        riders_yearly_data=riders_yearly_data, 
        races_features_df = races_features,
    )
    # X_train, y_train, train_weights, X_test, y_test, test_weights = model.split_train_test(X, y, x_weights)

    model.evaluate_model(X_test, y_test, test_groups=test_groups)

    print("Model evaluated")

    """Predict two examples"""

    rvv_id = "R487e2808"
    tdf_2024_20_id = "R5e8c7e1a"
    omloop_2025_id = "R5cbc284f"
    
    # predict_race(
    #     startlist_df=results_features.filter(
    #         pl.col("race_id") == rvv_id
    #     ).select(pl.col("name").alias("rider"), "team"),
    #     race_stats_df=races_df.filter(pl.col("race_id") == rvv_id)
    # )
    # print("-----------------")
    # predict_race(
    #     startlist_df=results_features.filter(
    #         pl.col("race_id") == omloop_2025_id
    #     ).select(pl.col("name").alias("rider"), "team"),
    #     race_stats_df=races_df.filter(pl.col("race_id") == omloop_2025_id)
    # )
    # predict_race(
    #     startlist_df=results_features.filter(
    #         pl.col("race_id") == tdf_2024_20_id
    #     ).select(pl.col("name").alias("rider"), "team"),
    #     race_stats_df=races_df.filter(pl.col("race_id") == tdf_2024_20_id)
    # )
    # actual_result = results_features.filter(pl.col("race_id") == tdf_2024_20_id).sort("rank")\
    #     .select("name", "rank").head(10)
    # print(actual_result)

def main():
    train()
    evaluate()

if __name__ == "__main__":
    main()