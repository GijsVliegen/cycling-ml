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


class NeuralNet(nn.Module):
    """
    Simple neural network with one input and one output for scoring.
    """

    def __init__(self, init_func: callable, lr: float = 0.01, nr_of_features = 1, correlated = False):
        super().__init__()
        
        if nr_of_features == 1:
            self.net = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        elif nr_of_features > 1 and not correlated:
            self.feature_input_layers = [
                nn.Sequential(
                    nn.Linear(1, 8 * nr_of_features),
                    nn.ReLU(),
                )] * nr_of_features
            self.merge_layer = nn.Sequential(
                nn.Linear(8 * nr_of_features, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )
        elif nr_of_features > 1 and correlated:
            self.net = nn.Sequential(
                nn.Linear(nr_of_features, 8 * nr_of_features),
                nn.ReLU(),
                nn.Linear(8 * nr_of_features, 16 * nr_of_features),
                nn.ReLU(),
                nn.Linear(16 * nr_of_features, 1)
            )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)
        self.criterion = nn.MarginRankingLoss(margin=1.0)

        self.separate_inputs = nr_of_features > 2 and not correlated

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.separate_inputs:
            x = x.view(-1, x.shape[-1]) 
            intermediate_vals = [
                f[x[:, i:i+1]]
                for i, f in enumerate(self.feature_input_layers)
            ]
            m = torch.cat(intermediate_vals, dim=1)
            return self.merge(m)

        else: 
            return self.net(x)

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def plot_learned_function(self):
        pass


    
class SplinesSGD:
    """
    Stochastic Gradient Descent model using splines for nonlinear feature transformations.

    Learns spline weights and distance weights to predict rider rankings based on
    historical performance and race characteristics.
    """

    nr_riders_per_race = 5

    def __init__(self, X: np.ndarray) -> None:
        """
        Initializes the SplinesSGD model.

        Args:
            X: Training data array (n_samples, n_features).
        """
        # 0-4: name  ┆ race_id ┆ distance_km ┆ elevation_m ┆ profile_score ┆ 
        # 5-9: profile_score_last_25k ┆ classification ┆ date ┆ rank  ┆ startlist_score ┆ 
        # 10-11: age | rank_bucket | rank_normalized

        self.init_feature_funcs = self._init_feature_funcs(X)
        self.feature_idxs = [9, 12]
        self.feature_names = ["startlist_score", "rank_normalized"]
        self.race_dist_idxs = [2, 3, 4, 5]#, 6]
        self.race_dist_metrics = [
            "distance_km",
            "elevation",
            "profile_score",
            "profile_score_last_25k",
            # "classification"
        ]
        self.distance_weights = np.full(
            len(self.race_dist_idxs),  #length
            1/len(self.race_dist_idxs) #init weights
        )
        self.lr_distance = 1e-4

        self.feature_functions: list[NeuralNet] = [
            NeuralNet(
                init_func = self.init_feature_funcs[feature_idx],
                lr = 0.01,
                nr_of_features=1,
                correlated=False
            ) for feature_idx in self.feature_idxs
        ]
    
    def _init_feature_funcs(self, X: np.ndarray) -> Dict[int, Callable]:
        """
        Initializes feature-specific spline functions based on data statistics.

        Args:
            X: Training data array.

        Returns:
            Dict mapping feature indices to initialization functions.
        """
        #init splines to some base functions
        rank_max = 40
        def rank_spline_init(x): #0-> 1,  40-> 0
            """exponential decay"""
            weight = math.log(2) / rank_max
            clipped_x = np.clip(x, 1, rank_max-1) #avoid too steep decline
            return -np.exp(weight * clipped_x) + 2
        
        a = min(X[:,9])
        b = np.mean(X[:,9])
        c = max(X[:,9])
        A = np.array([
            [a**2, a, 1],
            [b**2, b, 1],
            [c**2, c, 1]
        ])
        y = np.array([0.5, 1, 2])
        coeffs = np.linalg.solve(A, y)  # [alpha, beta, gamma]
        def startlist_score_init(x): #min -> 0.25, max -> 2
            """kwadratisch"""
            return coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]
        
        def rank_bucket_init(x):
            return -x
        
        return { #dict, safer for debuggin
            8: rank_spline_init,
            9: startlist_score_init,
            11: rank_bucket_init,
            12: lambda x: x
        }

        # self.grad_descend_pass(bucket_errors, Y_pred, Y_true, neighbor_feature_scores_3d)
    
    def get_closest_points(self, X: np.ndarray, y: np.ndarray, k: int) -> List[int]:
        """
        Finds k historical data points for the same rider in different races, prioritized by best ranks.

        Args:
            X: Full dataset.
            y: Current data point.
            k: Number of neighbors to return.

        Returns:
            List of indices of similar historical points, top k by rank.
        """
        neighbors = [
            i
            for i, data_point in enumerate(X)
            if (
                data_point[0] == y[0] and #same name
                data_point[1] != y[1] and      #different race id
                data_point[6] <= y[6]  #same or earlier year
            )
        ]
        # Sort neighbors by rank (index 8, lower is better) and take top k
        neighbors_sorted = sorted(neighbors, key=lambda i: X[i, 8])
        return neighbors_sorted[:k]


    def predict_ranking_for_race(
        self, 
        indices: List[int], 
        data: np.ndarray
    ) -> torch.tensor:
        """
        Predicts scores for multiple riders in a race.

        Args:
            indices: Indices of riders to predict.
            data: Full dataset.

        Returns:
            Tuple of predictions, 
            3D neighbor feature scores (len(indices) x n_features x k),
            list of neighbor indices,
            list of neighbor distances
        """
        Y_pred_scores = []
        for index in indices:

            rider_score = self.forward_rider(
                data, 
                rider = data[index],
                nr_neighbors=25
            )

            Y_pred_scores.append(rider_score)

        # Y_pred = Y.from_scores(Y_pred_scores)
        return Y_pred_scores

    def training_step(self, Y_true_ranking: np.ndarray, indices: List[int], data: np.ndarray) -> np.ndarray:
        """
        Performs one training step: predict, compute errors, and update weights.

        Args:
            Y_true: True rankings.
            indices: Rider indices.
            data: Full dataset.

        Returns:
            Array of errors for each rider.
        """
        self._zero_grad()
        # Y_true = Y.from_ranking(Y_true_ranking)
        Y_true_order = np.argsort(np.argsort(Y_true_ranking))
        Y_pred_scores = self.predict_ranking_for_race(
            indices,
            data
        )
        loss = self._list_preference_loss(Y_pred_scores, Y_true_order)
        loss.backward()
        self._step()
        # for f in self.feature_functions:
        #     total_norm = 0
        #     for p in f.net.parameters():
        #         if p.grad is not None:
        #             param_norm = p.grad.data.norm(2)
        #             total_norm += param_norm.item() ** 2
        #     print("Grad norm:", total_norm ** 0.5)


        return loss

    def _pairwise_preference_loss(self, score_preferred, score_rejected):
        """a pairwise logistic loss (aka Bradley-Terry or ranking loss)"""
        # print(f"{score_preferred:.3e}, {score_rejected:.3e}")
        return -F.logsigmoid(score_preferred - score_rejected).mean()

    def _list_preference_loss(self, scores, correct_order):
        pairs = [
            (i, j) 
            for idx, i in enumerate(correct_order)
            for j in correct_order[idx + 1:]
        ]
        losses = torch.vstack([
            self._pairwise_preference_loss(
                score_preferred = scores[pair[0]],
                score_rejected = scores[pair[1]]
            )
            for pair in pairs
        ])
        #TODO: vectorize
        
        return torch.sum(losses)
        #TODO: gather losses into one value

    def _step(self):
        
        # TODO: update distance weights
        # if x is not None and data is not None and neighbor_distances is not None and len(neighbor_distances) > 0:
        #     diffs = x[self.race_dist_idxs] - data[neighbor_idxs][:, self.race_dist_idxs]
        #     grad = np.sum(error * (diffs**2) / (2 * (neighbor_distances[:, None] + 0.001)), axis=0).astype(float)
        #     self.distance_weights -= self.lr_distance * grad
        #     self.distance_weights = np.maximum(self.distance_weights, 0)  # keep non-negative
        #     self.distance_weights /= np.sum(self.distance_weights)  # normalize

        for f in self.feature_functions:
            f.step()

    def _zero_grad(self):
        for f in self.feature_functions:
            f.zero_grad()

    def forward_rider(self,
        all_data: np.ndarray,
        rider: np.ndarray,
        nr_neighbors: int = 25,
    ) -> torch.tensor:
        """
        Computes feature scores for k-nearest neighbors of a rider in a race.

        Args:
            all_data: Full dataset.
            x: Current data point.
            k: Number of neighbors to base calculation on.

        Returns:
            np.ndarray: Array with each row containing the feature scores for a neighbor (shape: n_features x k).
        """
        # neighbor_idxs = self.get_closest_points(data, data[index], k=25)

        #TODO: multiply distance of neighbor with score

        neighbor_idxs = self.get_closest_points(
            X = all_data,
            y = rider,
            k = nr_neighbors
        )
        
        if len(neighbor_idxs) == 0:
            return -10

        # neighbor_data = data[neighbor_idxs]
        diffs = rider[self.race_dist_idxs] - all_data[neighbor_idxs][:, self.race_dist_idxs]
        neighbor_distances = np.sqrt(np.sum(self.distance_weights * diffs**2, axis=1).astype(float))
        
        neighbor_feature_vals = all_data[neighbor_idxs]

        neighbor_scores: torch.tensor = torch.prod(
            torch.stack([
                f.forward(
                    torch.from_numpy(neighbor_feature_vals[:, i : i + 1].astype(float)).float()
                ) 
                for i, f in zip(
                    self.feature_idxs,
                    self.feature_functions
                )
            ]),
            dim = 0
        ).squeeze(-1)
        if torch.isnan(neighbor_scores).any() or torch.isinf(neighbor_scores).any() or neighbor_scores.numel() == 0:
            print("debug here")
            print(neighbor_scores)
        # print(f"highest score = {neighbor_scores.max():.3e}, lowest score = {neighbor_scores.min():.3e}")
        #TODO: aggregate with prod or with sum???


        rider_score = neighbor_scores.mean()
        
        #TODO: add distance weights

        return rider_score

    def plot_learned_functions(self) -> None:
        for f in self.feature_functions:
            f.plot_learned_function()
            

class MLflowWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow wrapper for SplinesSGD model to enable logging and loading.
    """

    def __init__(self, model: SplinesSGD):
        self.model = model

    def predict(self, context, race_id: int) -> Dict[str, Any]:
        """
        Predicts top 25 riders for a given race.

        Args:
            context: MLflow context.
            race_id: int

        Returns:
            Dict with 'top_riders' and 'top_scores'.
        """
        All = np.load('data/features_df.npy')
        rider_idxs = np.where(All[:, 1] == race_id)[0]
        rider_names = All[rider_idxs, 0]
        
        # if len(rider_idxs) == 0:
        #     return {'riders': [], 'scores': []}
        
        Y_pred_scores = self.model.predict_ranking_for_race(rider_idxs.tolist(), All)

        return {'riders': rider_names, 'scores': Y_pred_scores.tolist()}


def split_train_test(All: pl.DataFrame, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a Polars DataFrame into train and test numpy arrays.

    Args:
        All: Full dataset DataFrame.
        test_ratio: Fraction for test set.

    Returns:
        Tuple of (train_array, test_array).
    """
    split_idx = int((1 - test_ratio) * All.height)
    return All[:split_idx].to_numpy(), All[split_idx:].to_numpy()

def get_random_riders(All: np.ndarray, race_id: Any, min_nr: int = 6, min_rank = -1) -> np.ndarray:
    """
    Selects random riders from a race, balancing top and bottom performers.

    Args:
        All: Full dataset.
        race_id: ID of the race.
        min_nr: Minimum number of riders to select.

    Returns:
        Array of selected rider indices.
    """
    if min_rank != -1:
        top_rider_idxs = np.where(
            (All[:,1] == race_id) & (All[:, 8] <= min_rank)
        )[0]
        n = min(min_nr, len(top_rider_idxs))
        return np.random.choice(top_rider_idxs, size=n, replace=False)

    top_rider_idxs = np.where(
        (All[:,1] == race_id) & (All[:, 8] <= 25)
    )[0] #top 25 results
    bottom_rider_idxs = np.where(
        (All[:,1] == race_id) & (All[:, 8] >= 25)
    )[0] #bottom results
    n = min(min_nr , min(len(top_rider_idxs), len(bottom_rider_idxs)))
    return np.concatenate((
        np.random.choice(top_rider_idxs, size=int(n/2), replace=False),
        np.random.choice(bottom_rider_idxs, size=int(n/2), replace=False)
    ))

def train_model(All: np.ndarray, X: np.ndarray, spline_model: SplinesSGD) -> None:
    """
    Trains the spline model using stochastic gradient descent.

    Args:
        All: Full dataset.
        X: Training subset.
        spline_model: Model instance to train.
    """
    nr_riders_per_race = 4
    epochs = 500
    total_loss = 0

    feat_1 = torch.from_numpy(All[:, 9].astype(float))
    feat_2 = torch.from_numpy(All[:, 12].astype(float))
    for f in (feat_1, feat_2):
        print(f.min(), f.max())
        print(torch.isnan(f).any(), torch.isinf(f).any())


    for epoch in range(epochs):
        random_race_id = np.random.choice(X[:,1])
        random_rider_idxs = get_random_riders(All, random_race_id, nr_riders_per_race)

        if len(random_rider_idxs) < nr_riders_per_race:
            print(f"not enough riders in this race, continued")
            continue

        Y_true_ranking = All[random_rider_idxs, 8]
        loss = spline_model.training_step(
            Y_true_ranking = Y_true_ranking, 
            indices = random_rider_idxs, 
            data = All
        )
        total_loss += loss

        if epoch % (epochs/10) == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")
            total_loss = 0

            # for f in spline_model.feature_functions:
            #     for name, param in f.net.named_parameters():
            #         if param.grad is not None:
            #             print(name, param.grad.abs().mean().item())


def compute_model_performance(All: np.ndarray, Y: np.ndarray, model: SplinesSGD) -> Dict[str, float]:
    """
    Evaluates model performance on test races.

    Args:
        All: Full dataset.
        Y: Test subset.
        model: Trained model.

    Returns:
        Dict of performance metrics (MSE, MAE, R2).
    """
    race_ids = np.unique(Y[:, 1])
    test_size = len(race_ids)
    test_race_ids = np.random.choice(race_ids, size = test_size, replace = False)

    nr_riders_per_race = 4

    from scipy.stats import spearmanr
    from scipy.stats import kendalltau
    import itertools
    import statistics

    def ranking_accuracy(scores: torch.Tensor, correct_order: list[int]) -> float:
        # 1 -> perfect order
        # 0.5 -> random order
        # 0 -> completely reversed
        true_ranks = {idx: rank for rank, idx in enumerate(correct_order)}

        total_pairs = 0
        correct_pairs = 0

        for i, j in itertools.combinations(range(len(scores)), 2):
            total_pairs += 1
            # Compare ground truth order
            gt = true_ranks[i] < true_ranks[j]
            # Compare model scores
            pred = scores[i] > scores[j]
            if gt == pred:
                correct_pairs += 1

        return correct_pairs / total_pairs

    def spearman_correlation(scores, correct_order):
        predicted_order = torch.argsort(scores, descending=True).tolist()
        return spearmanr(predicted_order, correct_order).correlation

    def kendall_tau(scores, correct_order):
        predicted_order = torch.argsort(scores, descending=True).tolist()
        return kendalltau(predicted_order, correct_order).correlation

    ra = []
    sc = []
    kt = []
    for test_race in test_race_ids:
        random_rider_idxs = get_random_riders(All, test_race, nr_riders_per_race)

        if len(random_rider_idxs) < nr_riders_per_race:
            continue

        Y_true_ranking = All[random_rider_idxs, 8]
        Y_true_order = np.argsort(np.argsort(Y_true_ranking))
        Y_pred_scores = model.predict_ranking_for_race(
            indices = random_rider_idxs,
            data = All
        )
        ra.append(ranking_accuracy(Y_pred_scores, Y_true_order))
        # sc.append(spearman_correlation(Y_pred_scores, Y_true_order))
        # kt.append(kendall_tau(Y_pred_scores, Y_true_order))


    print("Pairwise accuracy:", statistics.mean(ra))
    # print("Spearman:", statistics.mean(sc))
    # print("Kendall tau:", statistics.mean(kt))
    return {"Pairwise accuracy:": statistics.mean(ra)}


def main() -> None:
    race_result_features = pl.read_parquet("data/features_df.parquet")

    X_Y: tuple[np.ndarray, np.ndarray] = split_train_test(race_result_features)
    All = np.concatenate(X_Y)
    neural_net = SplinesSGD(All)

    with mlflow.start_run():
        # mlflow.log_param("lr", Spline.lr)
        mlflow.log_param("lr_distance", neural_net.lr_distance)
        mlflow.log_param("epochs", 1000)
        mlflow.log_param("n_splines", 15)

        train_model(All, X_Y[0], neural_net)

        model_perf_dict = compute_model_performance(All, X_Y[1], model=neural_net)
        for key, value in model_perf_dict.items():
            mlflow.log_metric(key, value)

        # Log the model
        mlflow.pyfunc.log_model(
            "model",
            python_model=MLflowWrapper(neural_net),
            registered_model_name="NeuralNetGoesBrrrr"
        )

    print("Model performance on test set:", model_perf_dict)


    random_race_id = race_result_features.select(pl.col("race_id").unique()).sample(1).row(0)[0]
    rider_idxs = get_random_riders(All, random_race_id, 10, min_rank = 10)
    rider_names = All[rider_idxs, 0]
    
    Y_true_ranking = All[rider_idxs, 8]
    Y_pred_scores = neural_net.predict_ranking_for_race(
        rider_idxs,
        All,
    )
    print({'riders': rider_names, 'scores': Y_pred_scores})


    
def predict_top25_for_race(race_id: Any) -> Dict[str, Any]:
    """
    Loads the trained model from MLflow and predicts top 25 riders for a race.

    Args:
        race_id: ID of the race to predict.

    Returns:
        Dict with 'top_riders' and 'top_scores'.
    """
    # Load the model; adjust the URI as needed (e.g., "models:/SplinesSGD/Production")
    model = mlflow.pyfunc.load_model("models:/SplinesSGD/None")
    return model.predict({'race_id': race_id})

def main2():
    race_result_features = pl.read_parquet("data/features_df.parquet")
    random_race_id = race_result_features.select(pl.col("race_id").unique()).sample(1).row(0)[0]

    predict_top25_for_race(random_race_id)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
    # main2()
    # race_result_features = pl.read_parquet("data/features_df.parquet")

    # X_Y: tuple[np.ndarray, np.ndarray] = split_train_test(race_result_features)
    # All = np.concatenate(X_Y)
    # spline_model = SplinesSGD(All)


#Training preidicting high results is more important:
# sklearn estimators usually accept sample_weight in .fit(X,y, sample_weight=...)


# class Y():
#     """holds the predictions for one race"""
#     def __init__(self, ranking = None, scores = None, order = []):
#         self.ranking: np.array = ranking
#         self.order: np.array = np.argsort(np.argsort(self.ranking))
#         if ranking is None and scores is None:
#             raise AssertionError("not both scores and ranking can be None")
        
#         self.ranking = ranking
#         self.scores: torch.tensor = scores

#     possible_buckets = [
#         [1, 3],
#         [4, 8],
#         [9, 15],
#         [16, 25],
#         [26, 99999]
#     ]
#     def ranks_to_buckets(self) -> np.ndarray:
#         """
#         Converts ranking positions to bucket categories.

#         Args:
#             ranks: Array of ranking positions.

#         Returns:
#             Array of bucket indices.
#         """
#         buckets = []
#         for r in self.ranking:
#             for i, b in enumerate(self.possible_buckets):
#                 if r >= b[0] and r <= b[1]:
#                     buckets.append(i)
#                     break
#         assert len(self.ranking) == len(buckets), f"error in mapping ranks to buckets: {self.ranking} -> {buckets}"
#         return np.array(buckets)
    
#     def complete_pred_with_true(self, Y_true = None):
#         if (self.ranking is None and Y_true is None) or \
#             (self.ranking is None and Y_true.ranking is None) or \
#             (self.ranking is not None and Y_true is not None):
#             raise AssertionError("either self should have a ranking, XOR Y_true should be given with ranking")
#         if self.ranking is None:
#             self.ranking = [
#                 Y_true.ranking[np.where(Y_true.order == order_to_find)[0]][0]
#                 for order_to_find in self.order
#             ]
#             self.buckets: np.array = self.ranks_to_buckets()
#         else:
#             self.buckets: np.array = self.ranks_to_buckets()
#             raise AssertionError("Already a ranking present")

#     @classmethod
#     def from_ranking(cls, ranking: np.array):
#         return cls(
#             ranking=ranking,
#             scores=None,
#             order=np.argsort(np.argsort(ranking))
#         )
        
#     @classmethod
#     def from_scores(cls, scores: torch.tensor):
#         return cls(
#             ranking = None,
#             scores = scores,
#             order = np.argsort(np.argsort( -1 * np.array(scores)))
#         )
    
#     def calculate_errors(self, Y_true) -> np.ndarray:
#         """
#         self: predicted Y
#         Y_true: actual Y

#         returns:
#         an error value for each rider, in the same order as y_pred

#         current error: nr of buckets to high or to low

#         # e.g. pred order = 26, true order = 1 -> 
#         #   pred_buckets = 6 and true_bucket = 1, error = -5, should predict lower score
#         #    
#         # e.g. -2 -> predicted order was 2 and true order was 4 -> should predict lower
#         # e.g. 6 -> predicited order was 8 and true order was 2 -> should predict higher
#         """
#         if self.buckets is None:
#             self.complete_pred_with_true(Y_true)

#         errors = Y_true.buckets - self.buckets
#         return errors

