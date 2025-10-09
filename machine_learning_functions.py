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


class NeuralNet(nn.Module):
    """
    Simple neural network with one input and one output for scoring.
    """

    def __init__(self, init_func: callable, lr: float = 0.01, nr_of_features = 1, correlated = False):
        super().__init__()
        
        if nr_of_features == 1:
            self.model = nn.Sequential(
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
        
    def preference_update(self, input1: float, input2: float, higher_is_1: bool = True) -> float:
        """
        Updates the network based on preference: whether input1 should score higher than input2.

        Args:
            input1: Tensor for first input.
            input2: Tensor for second input.
            higher_is_1: True if input1 should have higher score than input2.

        Returns:
            Loss value.
        """
        score1 = self.forward(torch.tensor(input1))
        score2 = self.forward(torch.tensor(input2))
        target = torch.tensor(1.0 if higher_is_1 else -1.0, dtype=torch.float32)
        loss = self.criterion(score1, score2, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def grad_descent_pass(self, 
        predicted_neigbor_scores: np.array,  #for each neighbor for each rider
        true_scores: np.array, 
        margin: float = 1.0
    ) -> float:
        """
        Updates the network based on a list of predicted scores and their true ordering.

        Args:
            predicted_scores: List of predicted score tensors.
            y: List of true rankings (lower y means higher rank, e.g., y=[3,1,2] means 2nd > 3rd > 1st).
            margin: Margin for the ranking loss. TODO: use margin parameter

        Returns:
            Total loss value.
        """
        pos_target = torch.tensor(1.0, dtype=torch.float32)
        neg_target = torch.tensor(-1.0, dtype=torch.float32)

        total_loss = 0.0
        num_pairs = 0
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if y[i] < y[j]:  # i should have higher score than j
                    loss = self.criterion(predicted_scores[i], predicted_scores[j], pos_target)
                elif y[j] < y[i]:  # j should have higher score than i
                    loss = self.criterion(predicted_scores[i], predicted_scores[j], neg_target)
                else:
                    continue  # equal, skip
                total_loss += loss
                num_pairs += 1
        if num_pairs > 0:
            total_loss /= num_pairs
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
class Spline():
    """
    Represents a spline for modeling nonlinear relationships in a single feature.

    This class uses Generalized Additive Models (GAM) to create spline basis functions
    and learns weights through gradient descent for feature transformation.
    """

    lr = 1e-3
    def __init__(
        self,
        spline_init_func: Callable,
        n_splines: int,
        feature_idx: int,
        feature_name: str,
        all_data: np.ndarray
    ):
        """
        Initializes the Spline object.

        Args:
            spline_init_func: Function to initialize spline weights.
            n_splines: Number of spline basis functions.
            feature_idx: Index of the feature in the data array.
            feature_name: Name of the feature for plotting.
            all_data: Full dataset array (n_samples, n_features).
        """  
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.gam = LinearGAM(s(0, n_splines=n_splines)).fit(all_data[:,[feature_idx]], all_data[:, [feature_idx]])
        self.weights_init = self._init_spline(
            data = all_data, 
            init_func = spline_init_func
        )
        self.weights = self.weights_init.copy()
        self.basis = self.gam._modelmat(all_data[:,[feature_idx]])

    def compute(self, indexes: List[int]) -> np.ndarray:
        """
        Computes spline values for given data indices.

        Args:
            indexes: List of row indices in the data.

        Returns:
            np.ndarray: Spline-transformed values for each index.
        """
        return self.basis[indexes] @ self.weights

    def _init_spline(self, data: np.ndarray, init_func: Callable) -> np.ndarray:
        """
        Initializes spline weights by fitting a least squares solution to the provided data and initialization function.

        Args:
            data (np.array): Input data array where features are accessed by self.feature_idx.
            init_func (callable): Function to generate target values for initialization, applied to the feature grid.

        Returns:
            np.ndarray: Array of spline weights computed from the least squares fit.
        """
        x_grid = np.linspace(data[:,self.feature_idx].min(), data[:,self.feature_idx].max(), 100).reshape(-1,1)
        B_grid = self.gam._modelmat(x_grid)
        y_target = init_func(x_grid.ravel())
        weights, *_ = np.linalg.lstsq(B_grid.toarray(), y_target, rcond=None)
        return weights
    
    def grad_descend_pass(self, neighbor_idxs: List[int], softmax_w: np.ndarray, error: np.ndarray) -> None:
        """
        Performs a gradient descent update on spline weights.

        Args:
            neighbor_idxs: Indices of neighboring data points.
            softmax_w: Softmax weights for gradient distribution.
            error: Error values for each neighbor.
        """
        weighted_basis_sum = softmax_w * self.basis[neighbor_idxs]
        gradient_pass = error * weighted_basis_sum
        self.weights -= self.lr * gradient_pass

    def plot_learned_spline(self) -> None:
        """
        Plots the learned spline curve against the initial spline.
        """
        XX = self.gam.generate_X_grid(term=0)   # only 1D in gam_f2 and gam_f3
        B = self.gam._modelmat(XX)              # spline basis on that grid
        y_hat = B @ self.weights                 # spline curve

        plt.figure()
        plt.plot(XX[:, 0], y_hat, label="learned spline")

        y_start = B @ self.weights_init
        plt.plot(XX[:, 0], y_start, "r--", label=str(self.feature_idx))

        plt.xlabel(f"f{self.feature_idx}: {self.feature_name}")
        plt.ylabel("spline value")
        plt.legend()
        plt.show()
        
    # def _create_penalty_matrix(self, n_basis):
    #     """
    #     Create penalty matrix for smoothness (penalizes second derivatives).
    #     This is the discrete approximation of the integral of squared second derivatives.
    #     """
    #     # Create second difference matrix
    #     D = np.diff(np.eye(n_basis), n=2, axis=0)
    #     # Penalty matrix is D.T @ D
    #     P = D.T @ D
    #     return P

class Y():
    """holds the predictions for one race"""
    def __init__(self, ranking = None, scores = None, order = []):
        self.ranking: np.array = ranking
        self.order: np.array = np.argsort(np.argsort(self.ranking))
        if ranking is None and scores is None:
            raise AssertionError("not both scores and ranking can be None")
        
        self.ranking = ranking
        self.scores = scores

    possible_buckets = [
        [1, 3],
        [4, 8],
        [9, 15],
        [16, 25],
        [26, 99999]
    ]
    def ranks_to_buckets(self) -> np.ndarray:
        """
        Converts ranking positions to bucket categories.

        Args:
            ranks: Array of ranking positions.

        Returns:
            Array of bucket indices.
        """
        buckets = []
        for r in self.ranking:
            for i, b in enumerate(self.possible_buckets):
                if r >= b[0] and r <= b[1]:
                    buckets.append(i)
                    break
        assert len(self.ranking) == len(buckets), f"error in mapping ranks to buckets: {self.ranking} -> {buckets}"
        return np.array(buckets)
    
    def complete_pred_with_true(self, Y_true = None):
        if (self.ranking is None and Y_true is None) or \
            (self.ranking is None and Y_true.ranking is None) or \
            (self.ranking is not None and Y_true is not None):
            raise AssertionError("either self should have a ranking, XOR Y_true should be given with ranking")
        if self.ranking is None:
            self.ranking = [
                Y_true.ranking[np.where(Y_true.order == order_to_find)[0]][0]
                for order_to_find in self.order
            ]
            self.buckets: np.array = self.ranks_to_buckets()
        else:
            self.buckets: np.array = self.ranks_to_buckets()
            raise AssertionError("Already a ranking present")

    @classmethod
    def from_ranking(cls, ranking: np.array):
        return cls(
            ranking=ranking,
            scores=None,
            order=np.argsort(np.argsort(ranking))
        )
        
    @classmethod
    def from_scores(cls, scores: np.array):
        return cls(
            ranking = None,
            scores = scores,
            order = np.argsort(np.argsort( -1 * np.array(scores)))
        )
    
    def calculate_errors(self, Y_true) -> np.ndarray:
        """
        self: predicted Y
        Y_true: actual Y

        returns:
        an error value for each rider, in the same order as y_pred

        current error: nr of buckets to high or to low

        # e.g. pred order = 26, true order = 1 -> 
        #   pred_buckets = 6 and true_bucket = 1, error = -5, should predict lower score
        #    
        # e.g. -2 -> predicted order was 2 and true order was 4 -> should predict lower
        # e.g. 6 -> predicited order was 8 and true order was 2 -> should predict higher
        """
        if self.buckets is None:
            self.complete_pred_with_true(Y_true)

        errors = Y_true.buckets - self.buckets
        return errors


    
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
        # 10-11: age | rank_bucket  

        self.init_feature_funcs = self._init_feature_funcs(X)
        self.feature_idxs = [9, 11]
        self.feature_names = ["startlist_score", "rank_bucket"]
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

        # self.feature_functions: List[Spline] = [
        #     Spline(
        #         spline_init_func = self.init_feature_funcs[feature_idx],
        #         n_splines = 15,
        #         feature_idx = feature_idx,
        #         feature_name = feature_name,
        #         all_data = X
        #     ) for feature_idx, feature_name in zip(
        #         self.feature_idxs,
        #         self.feature_names
        #     )
        # ]
        self.feature_functions: list[NeuralNet] = [
            NeuralNet(
                init_func = self.init_feature_funcs[feature_idx],
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
        }

        # self.grad_descend_pass(bucket_errors, Y_pred, Y_true, neighbor_feature_scores_3d)
    def grad_descend_pass(self, 
        errors: np.ndarray,
        Y_pred: Y,
        Y_true: Y,
        all_neighbor_feature_scores_3d: np.array
        ) -> None:
        """
        Performs gradient descent on feature_funcs and distance weights.

        Args:
            error: Error values.
            TODO: update this documentation
        """


        for i, feature_func in enumerate(self.feature_functions):
            all_neighbor_one_feature_score = all_neighbor_feature_scores_3d[
                :, :, i
            ]
            all_neighbor_true_estimated_scores = all_neighbor_one_feature_score[
                Y_true.ranking
            ]
            for rider_neighbors_one_feature_score, neighbor_true_estimated_scores, error in zip(
                all_neighbor_one_feature_score,
                all_neighbor_true_estimated_scores,
                errors
            ):
                average_true_estimate_one_feature_score = np.avg(neighbor_true_estimated_scores)
                for r_n_one_feature_score in rider_neighbors_one_feature_score:
                    feature_func.preference_update(
                        input_1 = r_n_one_feature_score,
                        input2 = average_true_estimate_one_feature_score,
                        higher_is_1= (error > 0)
                    )
            # feature_func.grad_descend_pass(
            #     pred
            #     neighbor_idxs = neighbor_idxs,
            #     softmax_w = softmax_w,
            #     error = error
            # )

        # update distance weights
        # if x is not None and data is not None and neighbor_distances is not None and len(neighbor_distances) > 0:
        #     diffs = x[self.race_dist_idxs] - data[neighbor_idxs][:, self.race_dist_idxs]
        #     grad = np.sum(error * (diffs**2) / (2 * (neighbor_distances[:, None] + 0.001)), axis=0).astype(float)
        #     self.distance_weights -= self.lr_distance * grad
        #     self.distance_weights = np.maximum(self.distance_weights, 0)  # keep non-negative
        #     self.distance_weights /= np.sum(self.distance_weights)  # normalize

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

    def compute_rider_scores_constituents(self,
        all_data: np.ndarray,
        rider: np.ndarray,
        k: int = 25,
    ) -> np.ndarray:
        """
        Computes feature scores for k-nearest neighbors of a rider in a race.

        Args:
            all_data: Full dataset.
            x: Current data point.
            k: Number of neighbors to base calculation on.

        Returns:
            np.ndarray: Array with each row containing the feature scores for a neighbor (shape: n_features x k).
        """
        neighbor_idxs = self.get_closest_points(
            X = all_data,
            y = rider,
            k = k
        )

        neighbor_feature_vals = all_data[neighbor_idxs]
        s_feature_vals: np.array = np.vstack([
            feature_func.forward(neighbor_feature_vals[: feature_idx])
            for feature_func, feature_idx in zip(
                self.feature_functions,
                self.feature_idxs
            )
        ])

        return s_feature_vals

    def predict_ranking_for_race(
        self, 
        indices: List[int], 
        data: np.ndarray
    ) -> Tuple[Y, np.ndarray, List[List[int]], List[np.ndarray]]:
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
        all_neighbor_feature_scores = []
        neighbor_idxs_s = []
        neighbor_distances_s = []
        for index in indices:
            neighbor_idxs = self.get_closest_points(data, data[index], k=25)
            neighbor_data = data[neighbor_idxs]
            diffs = data[index][self.race_dist_idxs] - neighbor_data[:, self.race_dist_idxs]
            neighbor_distances = np.sqrt(np.sum(self.distance_weights * diffs**2, axis=1).astype(float))

            neighbor_feature_scores = self.compute_rider_scores_constituents(
                data, 
                rider = data[index]
            )
            neighbor_scores = np.prod(neighbor_feature_scores, axis=0) / len(neighbor_idxs)

            #TODO: multiple distance of neighbor with score

            rider_pred = np.sum(neighbor_scores)

            Y_pred_scores.append(rider_pred)
            all_neighbor_feature_scores.append(neighbor_feature_scores)
            neighbor_idxs_s.append(neighbor_idxs)
            neighbor_distances_s.append(neighbor_distances)

        all_neighbor_feature_scores_3d = np.stack(all_neighbor_feature_scores, axis=0)
        Y_pred = Y.from_scores(Y_pred)
        return Y_pred, all_neighbor_feature_scores_3d, neighbor_idxs_s, neighbor_distances_s

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
        Y_true = Y.from_ranking(Y_true_ranking)
        Y_pred, neighbor_feature_scores_3d, _, _ = self.predict_ranking_for_race(
            indices,
            data
        )
        bucket_errors = Y_pred.calculate_errors(Y_true)


        self.grad_descend_pass(bucket_errors, Y_pred, Y_true, neighbor_feature_scores_3d)

        return bucket_errors


    def plot_learned_splines(self) -> None:
        for spline in self.feature_functions:
            spline.plot_learned_spline()
            

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
        if len(rider_idxs) == 0:
            return {'top_riders': [], 'top_scores': []}
        Y_pred, _, _, _ = self.model.predict_ranking_for_race(rider_idxs.tolist(), All)
        sorted_idxs = np.argsort(Y_pred.ranking)[::-1][:25]
        top_riders = All[rider_idxs[sorted_idxs], 0]
        top_scores = np.array(Y_pred.scores)[sorted_idxs]
        return {'top_riders': top_riders.tolist(), 'top_scores': top_scores.tolist()}


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

def get_random_riders(All: np.ndarray, race_id: Any, min_nr: int = 6) -> np.ndarray:
    """
    Selects random riders from a race, balancing top and bottom performers.

    Args:
        All: Full dataset.
        race_id: ID of the race.
        min_nr: Minimum number of riders to select.

    Returns:
        Array of selected rider indices.
    """
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
    print("training_model")
    nr_riders_per_race = 20
    epochs = 100
    total_loss = 0
    for epoch in range(epochs):
        random_race_id = np.random.choice(X[:,1]) #take id out of X
        random_rider_idxs = get_random_riders(All, random_race_id, nr_riders_per_race)

        if len(random_rider_idxs) < nr_riders_per_race:
            continue
            print(f"continued")

        Y_true_ranking = All[random_rider_idxs, 8]
        errors = spline_model.training_step(
            Y_true_ranking = Y_true_ranking, 
            indices = random_rider_idxs, 
            data = All
        )
        total_loss += sum([abs(error) for error in errors])

        if epoch % (epochs/10) == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")
            total_loss = 0


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
    test_size = 50
    race_ids = np.unique(Y[:, 1])
    test_race_ids = np.random.choice(race_ids, size = test_size, replace = False)

    all_Y_true_buckets = []
    all_Y_pred_buckets = []
    
    nr_riders_per_race = 10

    for test_race in test_race_ids:
        random_rider_idxs = get_random_riders(All, test_race, nr_riders_per_race)

        if len(random_rider_idxs) < nr_riders_per_race:
            continue

        race_pred = []

        Y_true_ranking = All[random_rider_idxs, 8]
        Y_true = Y.from_ranking(Y_true_ranking)
        Y_pred, neighbor_idxs_s, neighbor_scores_s, neighbor_distances_s = model.predict_ranking_for_race(
            Y_true_ranking = 
            random_rider_idxs
        )
        bucket_errors = Y_pred.calculate_errors(Y_true)
        all_Y_true_buckets.extend(Y_true.buckets)
        all_Y_pred_buckets.extend(Y_pred.buckets)


    mse = mean_squared_error(all_Y_true_buckets, all_Y_pred_buckets)
    mae = mean_absolute_error(all_Y_true_buckets, all_Y_pred_buckets)
    r2  = r2_score(all_Y_true_buckets, all_Y_pred_buckets)

    print(model.distance_weights)
    model.plot_learned_splines()
    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }


def main() -> None:
    race_result_features = pl.read_parquet("data/features_df.parquet")

    X_Y: tuple[np.ndarray, np.ndarray] = split_train_test(race_result_features)
    All = np.concatenate(X_Y)
    spline_model = SplinesSGD(All)

    with mlflow.start_run():
        mlflow.log_param("lr", Spline.lr)
        mlflow.log_param("lr_distance", spline_model.lr_distance)
        mlflow.log_param("epochs", 1000)
        mlflow.log_param("n_splines", 15)

        train_model(All, X_Y[0], spline_model)

        model_perf_dict = compute_model_performance(All, X_Y[1], model=spline_model)
        for key, value in model_perf_dict.items():
            mlflow.log_metric(key, value)

        # Log the model
        mlflow.pyfunc.log_model(
            "model",
            python_model=MLflowWrapper(spline_model),
            registered_model_name="SplinesSGD"
        )

    print("Model performance on test set:", model_perf_dict)
    
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
    main()
    # main2()

# Quick decision flow (what to try, in order)

# Baseline: simple model (logistic/regression) with minimal transforms + proper CV.

# Try GBDT (LightGBM/XGBoost/CatBoost) with basic hyperparameter tuning — often wins on small data with nonlinearities.

# If you need interpretability of specific nonlinear effects, try GAM or spline + Ridge.

# Use ensemble of GAM + GBDT if you need better predictive power and can afford complexity.




#Training preidicting high results is more important:
# sklearn estimators usually accept sample_weight in .fit(X,y, sample_weight=...)

