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


with open("WIELERMANAGER_RULES.json") as f:
    rules = json.load(f)

WEIGHTS_PER_RANK = rules["points_per_race"]["default"]
WEIGHTS_PER_RANK = {
    int(rank): weight for rank, weight in WEIGHTS_PER_RANK.items()
}
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
            # f"embed_{i}"
            # for i in range(1, EMBEDDING_SIZE+1)
        ]
        self.rider_diff_features = [
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
        ]
        self.rider_result_features = self.rider_diff_features + [
            "cosine_similarity",
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
            num_boost_round=500,
            evals=[(dtest, "test")],
            params = {
                # "objective": "reg:squarederror",
                # "eval_metric": "rmse",
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "min_child_weight": 10,
                "max_depth": 5,
                "tree_method": "hist",
                # "max_bin": 3,
                "early_stopping_rounds": 50,
                "learning_rate": 0.05,   # ↓↓↓
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
        result_features_df: pl.DataFrame, 
        riders_yearly_data: pl.DataFrame,
        races_features_df: pl.DataFrame,
        all_pairs: bool = False
    ) -> np.ndarray:

        X = []
        y = []
        X_weights = []
        result_features_df = result_features_df.unique(subset = ["race_id", "name"])
        all_race_ids: list[int] = result_features_df.select(
            ["race_id"] 
        ).unique().to_series().to_list() #Take race_ids from results since only care about races with results
       
        for race_id in all_race_ids:
            race_year = int(races_features_df.filter(
                pl.col("race_id") == race_id
            ).select("year").to_numpy()[0][0])

            if race_year < 2016: #DO not take early data, since relies on data for 3 years back
                continue

            rider_pairs, values_to_predict, weights = self.get_rider_pairs_values(
                race_id=race_id,
                result_features_df=result_features_df,
                all_pairs = all_pairs
            )
            rider_pairs = rider_pairs
            if len(rider_pairs) == 0:
                print(f"skipping because no pairs")
                continue
            ordering = np.array([1] * len(rider_pairs))
            values_to_predict = np.array(values_to_predict) * ordering

            try:
                rider_pair_features = self.get_rider_pair_features(
                    rider_pairs = rider_pairs,
                    result_features_df = result_features_df,
                    riders_yearly_data = riders_yearly_data,
                    race_year = race_year,
                    race_id = race_id,
                    ordering=ordering,
                )
            except ValueError as e:
                print(f"skipping race {races_features_df.filter(
                        pl.col("race_id") == race_id
                    ).select("name", "year").to_numpy()[0]}")
                continue

            race_features = races_features_df.filter(
                pl.col("race_id") == race_id
            ).select(self.race_features).to_numpy()[0].astype(np.float32, copy=False)
            race_features = np.tile(race_features, (len(rider_pairs), 1))

            race_X = np.hstack([rider_pair_features, race_features])
            X.append(race_X)
            y.append(values_to_predict)
            X_weights.extend(weights)

        y = np.concatenate(y)
        y = (y + 1) / 2 #convert -1,1 to 0,1
        X = np.vstack(X)
        X_weights = np.array(X_weights, dtype=np.float32)
        print(f"training on {len(y)} pairs")

        return X, y, X_weights

    def get_rider_pair_features(
        self, 
        rider_pairs: list[tuple[str, str]],
        result_features_df: pl.DataFrame,
        riders_yearly_data: pl.DataFrame,
        race_year: int,
        race_id: str,
        ordering: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:    
        """
        rider_pairs, list of tuples of rider names, for a whole race (thousands of pairs)

        returns np.ndarray of shape (nr_pairs, nr_features + nr_years * nr_yearly_features)
        """

        nr_years = 3
        years_to_go_back = [
            str(race_year - i) for i in range(1, nr_years + 1)
        ]
        race_results = result_features_df.filter(
            pl.col("race_id") == race_id
        )
        def get_riders_result_data(rider_filters: pl.DataFrame) -> tuple[np.array, np.array]:
            rider_0_data = race_results.filter().join(
                rider_filters.unique(subset=["name_0","name_1"], maintain_order=True).select("name_0"),
                left_on=["name"],
                right_on=["name_0"],
                how="right"
            ).fill_null(
                np.NAN
            ).select(self.rider_result_features)
            rider_0_data_for_diff = rider_0_data.select(self.rider_diff_features)
            rider_0_data_for_diff: np.array = rider_0_data_for_diff.to_numpy().astype(np.float32, copy=False)
            rider_0_data: np.array = rider_0_data.to_numpy().astype(np.float32, copy=False)
            rider_0_data[np.isnan(rider_0_data)] = 0.0
            rider_0_data_for_diff[np.isnan(rider_0_data_for_diff)] = 0.0
            rider_1_data: pl.DataFrame = race_results.join(
                rider_filters.unique(subset=["name_0","name_1"], maintain_order=True).select("name_1"),
                left_on=["name"],
                right_on=["name_1"],
                how="right"
            ).fill_null(
                np.NAN
            ).select(self.rider_result_features)
            rider_1_data_for_diff = rider_1_data.select(self.rider_diff_features)
            rider_1_data_for_diff: np.array = rider_1_data_for_diff.to_numpy().astype(np.float32, copy=False)
            rider_1_data: np.array = rider_1_data.to_numpy().astype(np.float32, copy=False)
            rider_1_data[np.isnan(rider_1_data)] = 0.0
            rider_1_data_for_diff[np.isnan(rider_1_data_for_diff)] = 0.0
            return rider_0_data, rider_1_data, rider_0_data_for_diff - rider_1_data_for_diff

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
            } if order == 1 else {
                "name_0": rider_pair_names[1],
                "name_1": rider_pair_names[0],
                "year": year
            }
            for rider_pair_names, order in zip(rider_pairs, ordering)
            for year in years_to_go_back
        )
        if len(rider_filters) == 0:
            breakpoint()
        rider_0_data, rider_1_data, rider_diff_data = get_riders_result_data(rider_filters)
        rider_0_yearly_data, rider_1_yearly_data = get_rider_yearly_data(rider_filters)

        yearly_feature_cols = rider_0_yearly_data - rider_1_yearly_data

        feature_cols = np.hstack([rider_0_data, rider_1_data, rider_diff_data])
        if feature_cols.shape[0] != yearly_feature_cols.shape[0]:
            raise ValueError("Feature columns shape does not match number of rider pairs")
            
        all_feat_cols = np.hstack([feature_cols, yearly_feature_cols])
        
        return all_feat_cols

    def get_rider_pairs_values(
        self, 
        race_id: int, 
        result_features_df: pl.DataFrame, 
        min_top_rank: int = 25,
        all_pairs: bool = False
    ) -> list[tuple[str, str, float]]:
        """
        returns pairs of rider names and weight of importance, rider 0 is the favorite

        selects pairs if first rider is in top 25

        returns 
            - names in the pair
            - value to predict (1 if first rider is better, -1 if second rider is better)
            - weights: If a pair has weight w, then in the loss it behaves like it appears w times.
                - weight = points for rank of first rider - points for rank of second rider, where points are defined by WEIGHTS_PER_RANK
        """
        nr_riders = result_features_df.filter(pl.col("race_id") == race_id).height

        if min_top_rank is None:
            min_top_rank = nr_riders
        if all_pairs:
            min_top_rank = nr_riders
        top_riders = result_features_df.filter(
            pl.col("race_id") == race_id
        ).sort("rank").head(min_top_rank)
        all_riders = result_features_df.filter(
            pl.col("race_id") == race_id
        ).sort("rank")        
        pairs = []
        values_to_predict = []
        weights = []
        for i, top_rider in enumerate(top_riders["name"]):
            for j, other_rider in enumerate(all_riders["name"]):
                if i < j:
                    if (top_rider, other_rider) not in pairs:
                        pairs.append((top_rider, other_rider))
                        pairs.append((other_rider, top_rider))
                        values_to_predict.append(1)
                        values_to_predict.append(-1)
                        weights.append(1)#WEIGHTS_PER_RANK.get(i+1, 1) - WEIGHTS_PER_RANK.get(j+1, 1))
                        weights.append(1)#WEIGHTS_PER_RANK.get(i+1, 1) - WEIGHTS_PER_RANK.get(j+1, 1))

                        
        return pairs, values_to_predict, weights

    def split_train_test(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        X_train = X[train_mask]
        y_train = y[train_mask]
        train_weights = weights[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]
        test_weights = weights[test_mask]

        return X_train, y_train, train_weights, X_test, y_test, test_weights

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, test_weights: np.ndarray) -> None:
        dtest = xgb.DMatrix(X_test, label=y_test, weight=test_weights)
        y_pred_proba = self.bst.predict(dtest)
        
        """
        y_pred explained:
        - by default, threshold is 0
        """
        y_pred = (y_pred_proba >= 0.1).astype(int) 
        y_test_class = (y_test == 1).astype(int)  #convert -1,1 to 0,1
        
        cm = confusion_matrix(y_test_class, y_pred)

        print("Confusion Matrix:")
        print(cm)
        print("\nDetailed classification report:")
        print(classification_report(y_test_class, y_pred, digits=4))
        precision, recall, thresholds = precision_recall_curve(y_test_class, y_pred_proba)
        plt.figure()

        plt.plot(thresholds, precision[:-1], label="Precision")
        plt.plot(thresholds, recall[:-1], label="Recall")

        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Precision and Recall vs Threshold")
        plt.legend()
        plt.grid(True)
        plt.show()
        return

    def train_model(
            self, 
            result_features_df: pl.DataFrame, 
            riders_yearly_data: pl.DataFrame,
            races_features_df: pl.DataFrame
        ) -> None:
        X, y, x_weights = self.to_xgboost_format(
            result_features_df=result_features_df, 
            riders_yearly_data=riders_yearly_data, 
            races_features_df = races_features_df 
        )
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

def get_new_race_embedding(
    race_to_find_for: pl.DataFrame, 
    old_embeddings: pl.DataFrame, 
    old_races: pl.DataFrame
) -> pl.DataFrame:
    """
    closest 5 races in the same (stage-)race over last 5 years

    This function is total overkill since it can act on find for multiple races at once. 
    """
    duplicate_races = race_to_find_for.select("race_id", "name", "year", *RACE_SIMILARITY_COLS).join(
        old_races.select("race_id", "name", "year", *RACE_SIMILARITY_COLS),
        on = ["name"]
    ).filter(
        pl.col("year").cast(pl.Int64) <= pl.col("year_right").cast(pl.Int64) + 5
    ).filter(
        pl.col("year").cast(pl.Int64) >= pl.col("year_right").cast(pl.Int64)
    ).filter(
        ~ (pl.col("race_id") == pl.col("race_id_right"))
    )
    #TODO years should be replaced by dates, so only earlier stages are taken
    # not critical tho
    duplicate_races_normalized = duplicate_races.with_columns(
        *[
            (
                (pl.col(c) - pl.col(f"{c}_right").min()) 
                / (pl.col(f"{c}_right").max() - pl.col(f"{c}_right").min())
            ).over("name").alias(f"{c}_normalised")
            for c in RACE_SIMILARITY_COLS
        ],
        *[
            (
                (pl.col(f"{c}_right") - pl.col(f"{c}_right").min()) 
                / (pl.col(f"{c}_right").max() - pl.col(f"{c}_right").min())
            ).over("name").alias(f"{c}_normalised_right")
            for c in RACE_SIMILARITY_COLS
        ]
    )
    duplicate_races_distance_part = duplicate_races_normalized.with_columns([
        (
            (pl.col(f"{c}_normalised") - pl.col(f"{c}_normalised_right")) ** 2
        ).alias(f"{c}_distance")
        for c in RACE_SIMILARITY_COLS
    ])
    duplicate_races_distance = duplicate_races_distance_part.with_columns(
        (pl.sum_horizontal([
            pl.col(f"{c}_distance") 
            for c in RACE_SIMILARITY_COLS
        ])
        ).sqrt().alias("total_distance")
    )
    closest_race_ids = duplicate_races_distance.sort(
        "total_distance", descending=False
    ).group_by("race_id").head(5).select("race_id", "race_id_right")
    closest_embedding = closest_race_ids.join(
        old_embeddings.rename({"race_id": "race_id_right"}),
        on="race_id_right",
        how="left"
    ).group_by(
        "race_id",
    ).agg(
        *[
            pl.col(c).mean()
            for c in old_embeddings.columns if c != "race_id"
        ]
    )
    return closest_embedding

def get_rider_features(
        new_race: pl.DataFrame,
        startlist: pl.DataFrame, 
        results: pl.DataFrame, 
        races: pl.DataFrame
    ) -> pl.DataFrame:

    results_with_dates = results.join(
        races.select(["race_id", pl.col("date").str.to_date(), "startlist_score"]),
        on = "race_id",
        how="left"
    )
    results_with_dates = results_with_dates.with_columns(
        (pl.when(pl.col("rank") <= 3).then(8.333).
        otherwise(pl.when(
            pl.col("rank") <= 10).then(2.5)
            .otherwise(pl.when(pl.col("rank") <= 25).then(1).otherwise(0)
        )) * pl.col("startlist_score")).alias("points"),
    )
    startlist_results = startlist.rename({"rider": "name"}).join(
        results_with_dates,
        on = "name"
    )

    windows = {
        "1110d": 1110,
        "370d": 370,
        "40d": 40,
    }
    startlist_results = startlist_results.join(
        new_race.select(pl.col("date").str.to_date().alias("new_race_date")),
        how="cross"
    )

    dfs = []
    for label, offset in windows.items():
        dfs.append(
            startlist_results.filter(
                pl.col("date") >= (pl.col("new_race_date") - pl.duration(days = (offset - 1)))
            ).group_by(
                "name"
            ).agg(
                pl.count("race_id").alias(f"nr_races_participated_{label}"),
                (pl.when(pl.col("rank") < 25).then(1)
                    .otherwise(None)
                ).sum().alias(f"nr_top25_{label}"),
                (pl.when(pl.col("rank") < 10).then(1)
                    .otherwise(None)
                ).sum().alias(f"nr_top10_{label}"),
                (pl.when(pl.col("rank") < 3).then(1)
                    .otherwise(None)
                ).sum().alias(f"nr_top3_{label}"),
                pl.col("points").sum().alias(f"strength_{label}"),
            )
        )
    rider_features = startlist.join(
        new_race.select("race_id"),
        how="cross"
    ).rename({"rider": "name"})
    for df in dfs:
        if len(df) == 0:
            rider_features = rider_features.with_columns(
                [pl.lit(0).alias(c)
                 for c in df.columns if c != "name"]
            )
            continue
        rider_features = rider_features.join(
            df,
            on="name",
            how="left"
        ).fill_null(0)
    
    rider_features = rider_features.with_columns(
        pl.col("strength_370d").rank("dense", descending=True).over(["race_id", "team"]).alias("relative_team_strength_rank")
    )

    return rider_features


def get_rider_embeddings(startlist: pl.DataFrame, results_embedded_df: pl.DataFrame) -> pl.DataFrame:
    most_recent_embedding_date = results_embedded_df.sort("date").with_columns(
        pl.col("date").last().over("name").alias("most_recent_date")
    )
    most_recent_embedding = most_recent_embedding_date.filter(pl.col("date") == pl.col("most_recent_date"))
    
    return startlist.rename({"rider": "name"}).select("name").join(
        most_recent_embedding,
        on="name",
        how="left"
    ).drop("date", "most_recent_date", "cosine_similarity", "race_id")

def get_rider_pairs(race_id: int, result_features_df: pl.DataFrame, min_top_rank: int = None) -> list[list[str, str]]:
    """
    returns pairs of rider names
    """
    max_weight = 10
    nr_riders = result_features_df.filter(pl.col("race_id") == race_id).height

    if min_top_rank is None:
        min_top_rank = nr_riders

    top_riders = result_features_df.filter(
        pl.col("race_id") == race_id
    ).sort("race_id")
    all_riders = result_features_df.filter(
        pl.col("race_id") == race_id
    ).sort("race_id")         
    pairs = []
    for i, top_rider in enumerate(top_riders["name"]):
        for j, other_rider in enumerate(all_riders["name"]):
            if top_rider != other_rider:
                if (top_rider, other_rider) not in pairs:
                    pairs.append((top_rider, other_rider))
                    pairs.append((other_rider, top_rider))
    return pairs


def add_embedding_similarity_for_new_race(
    rider_embeddings: pl.DataFrame, 
    race_embedding: pl.DataFrame
) -> pl.DataFrame:

    results_w_race_embeddings = rider_embeddings.join(
        race_embedding,
        how= "cross",
    )
    embedding_similary = calculate_cosine_similarity_polars(results_w_race_embeddings)

    return rider_embeddings.join(
        embedding_similary.select(["name", "cosine_similarity"]),
        on = ["name"],
        how="left"
    )

def new_race_to_xgboost_format(
    rider_features_df: pl.DataFrame, 
    riders_yearly_data: pl.DataFrame,
    new_race_features: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[list[str, str]]]:
    mock_model = RaceModel()

    race_id: int = new_race_features.select(
        ["race_id"] 
    ).unique().to_series().to_list()[0] #Take race_ids from results since only care about races with results

    rider_pairs = get_rider_pairs(
        result_features_df = rider_features_df,
        race_id = new_race_features.select("race_id").to_numpy()[0][0]
    )

    ordering = np.array([1] * len(rider_pairs))
    
    rider_pair_features = mock_model.get_rider_pair_features(
        rider_pairs=rider_pairs,
        result_features_df = rider_features_df,
        riders_yearly_data = riders_yearly_data,
        race_year=2026,
        race_id = race_id,
        ordering=ordering
    )

    race_features = new_race_features.filter(
        pl.col("race_id") == race_id
    ).select(mock_model.race_features).to_numpy()[0].astype(np.float32, copy=False)
    race_features = np.tile(race_features, (len(rider_pairs), 1))

    race_X = np.hstack([rider_pair_features, race_features])

    return race_X, ordering, rider_pairs

def predict_race(startlist_df, race_stats_df):
    races_df = pl.read_parquet("data_v2/races_df.parquet")
    results_df = pl.read_parquet("data_v2/results_df.parquet")
    riders_yearly_data = pl.read_parquet("data_v2/rider_yearly_stats_df.parquet")
    
    races_embedded_df = pl.read_parquet("data_v2/races_embedded_df.parquet")
    results_embedded_df = pl.read_parquet("data_v2/results_embedded_df.parquet")
    """Prepare data of race and riders to give to xgboost"""
    
    riders_yearly_data = riders_yearly_data.with_columns(
        pl.all().replace(-1, 0)
    )
    necessary_races, necessary_results = filter_data(races_df, results_df)

    race_embedding = get_new_race_embedding(
        race_to_find_for = race_stats_df,
        old_embeddings = races_embedded_df,
        old_races = necessary_races,
    )
    race_feats = race_embedding.join(
        race_stats_df,
        on = ["race_id"],
        how="left"
    )
    rider_feats = get_rider_features(
        new_race=race_stats_df,
        startlist=startlist_df,
        results = necessary_results,
        races=necessary_races
    )
    rider_embeddings = get_rider_embeddings(
        startlist = startlist_df,
        results_embedded_df = results_embedded_df
    )
    rider_embeddings = add_embedding_similarity_for_new_race(
        rider_embeddings=rider_embeddings,
        race_embedding=race_embedding,
    )
    rider_feats = rider_feats.join(
        rider_embeddings,
        on="name"
    )

    """Use prepared data to generate inter-rider results"""

    model = RaceModel()
    model.load_model()

    X, ordering, pairs = new_race_to_xgboost_format(
        rider_features_df=rider_feats, 
        riders_yearly_data=riders_yearly_data, 
        new_race_features = race_feats 
    )
    riders = set(
        [rider for pair in pairs for rider in pair]
    )
    rider_scores = {
        rider: [] for rider in riders
    }

    dtest = xgb.DMatrix(X)
    y_pred_proba = model.bst.predict(dtest)
    for (first, second), pred_y, x, order in zip(pairs, y_pred_proba, X, ordering):
        if order == 1:
            rider_scores[first].append(pred_y)
            rider_scores[second].append( - pred_y)
        else:
            rider_scores[first].append( - pred_y)
            rider_scores[second].append(pred_y)
    rider_scores = {
        rider: np.sum(scores) for rider, scores in rider_scores.items()
    }
    rider_scores_df = pl.DataFrame(
        list(rider_scores.items()), schema=["name", "score"]
    )    
    rider_percentages_df = scores_to_probability_results(rider_scores_df)

    top10 = sorted(rider_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, score in top10:
        print(f"{name}: {score:.4f}")
    return

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
    X, y, x_weights = model.to_xgboost_format(
        result_features_df=results_features, 
        riders_yearly_data=riders_yearly_data, 
        races_features_df = races_features,
        all_pairs = True 
    )
    X_train, y_train, train_weights, X_test, y_test, test_weights = model.split_train_test(X, y, x_weights)

    model.evaluate_model(X_test, y_test, test_weights=test_weights)

    print("Model evaluated")

    """Predict two examples"""

    rvv_id = "R487e2808"
    tdf_2024_20_id = "R5e8c7e1a"
    omloop_2025_id = "R5cbc284f"
    
    predict_race(
        startlist_df=results_features.filter(
            pl.col("race_id") == rvv_id
        ).select(pl.col("name").alias("rider"), "team"),
        race_stats_df=races_df.filter(pl.col("race_id") == rvv_id)
    )
    print("-----------------")
    predict_race(
        startlist_df=results_features.filter(
            pl.col("race_id") == omloop_2025_id
        ).select(pl.col("name").alias("rider"), "team"),
        race_stats_df=races_df.filter(pl.col("race_id") == omloop_2025_id)
    )
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