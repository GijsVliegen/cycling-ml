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

class WeightedNeighbourAggregator(nn.Module):
    def __init__(self, nr_of_features: int):



        super().__init__()
        self.net = nn.Sequential( #nr_neighbor_aggregator_net
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )

        self.raw_feature_weights = nn.Parameter(torch.ones(nr_of_features) / nr_of_features)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.history = []

    @property
    def feature_weights(self):
        # apply softplus on-the-fly to ensure positivity
        return torch.nn.functional.softmax(self.raw_feature_weights)
    
    # def baseline_history(self):
    #     x = torch.linspace(2, 25, 100).unsqueeze(1) #not to high y values for plot readability
    #     y_baseline = 1/x
    #     return ("1/x", x.numpy().flatten(), y_baseline.numpy().flatten())

    def get_time_weights(self, center_date, neighbor_dates) -> torch.tensor:
        months_diff = (center_date - neighbor_dates) / np.timedelta64(1, 'D')
        decay_rate = 1/(365*3)
        time_weights: np.array = np.exp( -1 * decay_rate * months_diff)
        return torch.from_numpy(time_weights)

    def forward(
        self, 
        center_features, 
        neighbor_features, 
        neighbor_scores,
        center_date,
        neighbor_dates
    ):
        """
        center_features: [nr_of_features]
        neighbor_features: [N, nr_of_features]
        neighbor_scores: [N] 
        """
        # Compute distances from center to each neighbor with learnable weights
        diff = center_features - neighbor_features  # [N, D]
        weighted_diff = (diff ** 2) * self.feature_weights  # [N, D]
        dist = torch.sqrt(weighted_diff.sum(dim=-1) + 1e-8)  # [N]
        weights = torch.exp(-dist)  # [N]
        time_weights = self.get_time_weights(
            center_date=center_date,
            neighbor_dates=neighbor_dates,
        )

        # Aggregate neighbor scores as weighted average
        aggregated_score = torch.sum(weights * time_weights * neighbor_scores) / torch.sum(weights)
        if neighbor_scores.numel() == 0:
            breakpoint()
        y = self.net(torch.tensor([neighbor_scores.numel()]).float())[0]
        nr_top_races_penalty = torch.clamp(y, min=0.1, max=10.0)
        aggregated_score *= nr_top_races_penalty

        return aggregated_score
    
    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def save_function(self, epoch):
        # self.history.append((epoch, self.feature_weights.detach().numpy()))
        x = torch.linspace(0, 25, 100).unsqueeze(1)
        with torch.no_grad():
            y = self.net(x)
            y_clamped = torch.clamp(y, min=0.1, max=10.0)
        self.history.append((epoch, x.numpy().flatten(), y_clamped.numpy().flatten()))

    def plot_model(self):
        print(f"feature weights: {self.feature_weights}")

        if hasattr(self, 'net') and self.net[0].in_features == 1:
            x = torch.linspace(0, 25, 100).unsqueeze(1)
            with torch.no_grad():
                y = self.net(x)
                y_baseline = 1/x
            plt.figure()
            plt.plot(x.numpy().flatten(), y.numpy().flatten(), label='Learned Function')
            plt.plot(x.numpy().flatten(), y_baseline.numpy().flatten(), label='1/x Baseline', linestyle='--')
            plt.title(f'Learned Function for nr_of_neighbors')
            plt.xlabel(f'nr_of_neighbors')
            plt.ylabel('Output')
            plt.show()
        else:
            print("Plotting not implemented for this configuration")

    def normalize_weights(self):
        pass
    
class BigNN(nn.Module):

    def __init__(self, lr: float = 0.01, nr_of_features = 1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(nr_of_features, 8 * nr_of_features),
            nn.ReLU(),
            nn.Linear(8 * nr_of_features, 16 * nr_of_features),
            nn.ReLU(),
            nn.Linear(16 * nr_of_features, 8 * nr_of_features),
            nn.ReLU(),
            nn.Linear(8 * nr_of_features, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            # nn.Softplus() #reason of low outputs after training?
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        #TODO: check weight initialization

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.net(x)

    def step(self):
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def save_function(self, epoch):
        if hasattr(self, 'net') and self.net[0].in_features == 1:
            feats = torch.stack([
                torch.linspace(0, 1, 100).unsqueeze(1)
            ])
            with torch.no_grad():
                y = self.net(feats)
            self.history.append((epoch, feats[:0].numpy().flatten(), y.numpy().flatten()))


    
class RaceModel:
    """
    Stochastic Gradient Descent model using splines for nonlinear feature transformations.

    Learns spline weights and distance weights to predict rider rankings based on
    historical performance and race characteristics.
    """

    def __init__(self, X: np.ndarray) -> None:
        """
        Initializes the RaceModel model.

        Args:
            X: Training data array (n_samples, n_features).
        """
        # 0-4: name  ┆ race_id ┆ distance_km ┆ elevation_m ┆ profile_score ┆ 
        # 5-9: profile_score_last_25k ┆ classification ┆ date ┆ rank  ┆ startlist_score ┆ 
        # 10-14: age | rank_bucket | rank_normalized | age_normalized | year
        # 15-18: rank_bucket_year_count | top25_count_year | top25_count | attended_races

        self.nr_riders_per_race = 16

        self.feature_idxs = [9, 12, 13, 15, 16, 17]
        self.feature_names = ["startlist_score", "rank_normalized", "age_normalized"]
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
        self.race_date_idx = 7
        self.lr_distance = 1e-4

        mins = [0, 0.5]
        maxs = [100, 10]
        self.features_function: BigNN = BigNN(
            lr = 0.01,
            nr_of_features=len(self.feature_idxs),
        )
        self.neighbor_aggregate_function = WeightedNeighbourAggregator(
            nr_of_features=len(self.race_dist_idxs)
        )

    
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
        
        a = min(X["startlist_score"])
        b = np.mean(X["startlist_score"])
        c = max(X["startlist_score"])
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
    
    def _pairwise_preference_loss(self, score_preferred: torch.tensor, score_rejected: torch.tensor):
        """a pairwise logistic loss (aka Bradley-Terry or ranking loss)"""
        # print(f"{score_preferred:.3e}, {score_rejected:.3e}")
        return -F.logsigmoid(score_preferred - score_rejected)

    def _get_pairs(self, correct_order: np.array, correct_ranking: np.array):
        """According to chatgpt very fast"""
        """only return pairs for which one result was top 25"""

        nr_of_top_25 = np.sum(correct_ranking <= 25)
        N = len(correct_order)
        num_pairs = (N - nr_of_top_25) * nr_of_top_25 + (nr_of_top_25 * (nr_of_top_25 - 1)) // 2

        # Preallocate arrays
        first_elements = np.empty(num_pairs, dtype=np.int32)
        second_elements = np.empty(num_pairs, dtype=np.int32)

        idx = 0
        for i in range(nr_of_top_25):
            length = N - i - 1
            first_elements[idx:idx+length] = correct_order[i]
            second_elements[idx:idx+length] = correct_order[i+1:]
            idx += length

        return first_elements, second_elements


    def _list_preference_loss(self, scores: torch.tensor, correct_order: np.array, correct_ranking: np.array):
        preferred_idxs, rejected_idxs = self._get_pairs(correct_order, correct_ranking)
        preferred_scores = scores[preferred_idxs]
        rejected_scores = scores[rejected_idxs]
        losses = self._pairwise_preference_loss(preferred_scores, rejected_scores)
        loss = torch.mean(losses)
        #TODO: weighted sum, top-3 more important.
        if loss.isnan():
            breakpoint()
        return loss

    def _step(self):
        # for f in self.feature_functions:
        #     f.step()
        self.features_function.step()
        self.neighbor_aggregate_function.step()

    def _zero_grad(self):
        # for f in self.feature_functions:
        #     f.zero_grad()
        self.features_function.zero_grad()
        self.neighbor_aggregate_function.zero_grad()

    def _get_closest_points_batch(self, X: np.ndarray, rider_idxs: List[int], k: int) -> List[List[int]]:
        """
        Finds k historical data points for multiple riders in different races, prioritized by best ranks.

        Args:
            X: Full dataset.
            rider_idxs: List of rider indices.
            k: Number of neighbors to return per rider.

        Returns:
            List of lists of indices of similar historical points, top k by rank per rider.
        """
        rider_data = X[rider_idxs]  # [n_riders, n_features]
        n_riders = len(rider_idxs)
        n_data = X.shape[0]

        # Vectorized masking using broadcasting
        # mask = (
        #     (rider_data[:, None, 0] == X[None, :, 0]) &  # same name
        #     (rider_data[:, None, 1] != X[None, :, 1]) &  # different race
        #     (X[None, :, 6] <= rider_data[:, None, 6]) &  # earlier or same date
        #     ((X[:, 8]) <= 25) #rank top-25
        # )
        mask = (
            (rider_data['name'][:, None] == X['name'][None, :]) &       # same rider
            (rider_data['race_id'][:, None] != X['race_id'][None, :]) &   # different race
            (X['date'][None, :] <= rider_data['date'][:, None]) &         # earlier or same date
            (X['rank'][None, :] <= 25)                                    # rank top-25 #TODO: parameterize, should be same as bucket count yearly definition in data_science_functions.py
        )

        neighbor_lists = []
        for rider_i in range(n_riders):
            indices = np.where(mask[rider_i])[0]
            if len(indices) == 0:
                neighbor_lists.append([])
                continue

            if len(indices) > k:
                # Sort by rank (ascending, lower is better)
                ranks = X["rank"][indices]
                sorted_order = np.argsort(ranks)
                indices = indices[sorted_order][:k]

            # if len(indices) > k:
            #     # Sort by date descending (most recent first)
            #     dates = X["date"][indices]
            #     sorted_order = np.argsort(dates)[::-1]  # descending
            #     indices = indices[sorted_order][:k]

            neighbor_lists.append(indices.tolist())

        return neighbor_lists

    def predict_ranking_for_race(
        self, 
        indices: List[int], 
        data: np.ndarray,
        torch_data: torch.tensor,
        nr_neighbors: int = 25,
        ranks_to_predict: int = 25,
    ) -> torch.tensor:
        """
        Predicts scores for multiple riders in a race.

        Args:
            indices: Indices of riders to predict.
            data: Full dataset.
            torch_data: Torch tensor of dataset.

        Returns:
            Tensor of predicted scores for each rider.
        """
        Y_pred_scores = self.forward_riders(
            all_data=data,
            torch_data=torch_data,
            rider_idxs=indices,
            nr_neighbors=nr_neighbors
        )

        return Y_pred_scores

    def training_step(
        self, 
        Y_true_ranking: np.ndarray, 
        indices: List[int], 
        data: np.ndarray,
        torch_data: torch.tensor
    ) -> np.ndarray:
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
        Y_true_order = np.argsort(np.argsort(Y_true_ranking))
        Y_pred_scores = self.predict_ranking_for_race(
            indices = indices,
            data = data,
            torch_data = torch_data
        )
        if Y_pred_scores.isnan().any():
            breakpoint()
        loss = self._list_preference_loss(scores = Y_pred_scores, correct_order= Y_true_order, correct_ranking=Y_true_ranking)

        if loss.requires_grad:
            loss.backward()
        else:
            print(f"""Loss does not require grad. Inspecting:
                loss={loss}, requires_grad={loss.requires_grad},
                grad_fn={loss.grad_fn}"""
            )
        self._step()


        return loss

    def forward_riders(self,
        all_data: np.ndarray,
        torch_data: torch.tensor,
        rider_idxs: List[int],
        nr_neighbors: int,
    ) -> torch.tensor:
        """
        Computes feature scores for k-nearest neighbors of multiple riders in a race.

        Args:
            all_data: Full dataset.
            torch_data: Torch tensor of dataset.
            rider_idxs: List of rider indices.
            nr_neighbors: Number of neighbors per rider.

        Returns:
            Tensor of rider scores.
        """
        neighbor_lists = self._get_closest_points_batch(
            X=all_data,
            rider_idxs=rider_idxs,
            k=nr_neighbors
        )
        all_neighbor_idxs = [idx for sublist in neighbor_lists for idx in sublist]
        rider_neighbor_counts = [len(sublist) for sublist in neighbor_lists]


        neighbor_data = torch_data[all_neighbor_idxs]

        neighbor_scores: torch.tensor = self.features_function.forward(
            neighbor_data[:, self.feature_idxs]
        ).squeeze(-1) 
        
        #TODO: squeeze needed?

        rider_scores = []
        offset = 0


        center_date: np.array = all_data["date"][rider_idxs[0]]
        center_race_features: torch.tensor = torch_data[rider_idxs[0]][self.race_dist_idxs]
        for i, rider_idx in enumerate(rider_idxs):
            count = rider_neighbor_counts[i]
            if count == 0:
                rider_scores.append(torch.tensor(0))
            else:
                neigh_scores = neighbor_scores[offset:offset + count]
                neighbor_race_features: torch.tensor = torch_data[all_neighbor_idxs[offset:offset + count]][:, self.race_dist_idxs]
                neighbor_dates: np.array = all_data["date"][all_neighbor_idxs[offset:offset + count]]

                if neigh_scores.isnan().any():
                    breakpoint()
                rider_score = self.neighbor_aggregate_function.forward(
                    center_race_features, 
                    neighbor_race_features, 
                    neigh_scores,
                    center_date, 
                    neighbor_dates
                )
                if rider_score.isnan().any():
                    breakpoint()
                rider_scores.append(rider_score)
            offset += count

        return torch.stack(rider_scores)

    def normalize_distance_weights(self):
        self.neighbor_aggregate_function.normalize_weights()

    def plot_all_learned_functions(self):

        #TODO: move code to their respective classes

        # Plot feature functions history in one grid
        # if self.feature_functions and self.feature_functions[0].history:
        #     fig, axes = plt.subplots(1, len(self.feature_functions) + 1, figsize=(15, 5))
        #     if len(self.feature_functions) == 1:
        #         axes = [axes]
        #     for i, ax in enumerate(axes):
        #         if i == len(axes) -1: #last element
        #             his = self.neighbor_aggregate_function.history
        #             for epoch, x_vals, y_vals in his[1:]: #skip first history
        #                 ax.plot(x_vals, y_vals, label=f'Epoch {epoch}')
        #             ax.set_title('Learned Function for nr_of_neighbors')
        #             ax.set_xlabel('nr_of_neighbors')
        #             ax.set_ylabel('Output')
        #             ax.legend()
        #             continue

        #         f = self.feature_functions[i]
        #         for epoch, x_vals, y_vals in f.history[1:]:
        #             ax.plot(x_vals, y_vals, label=f'Epoch {epoch}')
        #         ax.set_title(f'{self.feature_names[i]} Evolution')
        #         ax.set_xlabel(f'{self.feature_names[i]}')
        #         ax.set_ylabel('Output')
        #         ax.legend()

        # Plot neighbor aggregate weights history
        plt.title('Neighbor Aggregate Weights Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.show()


def polars_to_structured_array(df: pl.DataFrame, max_str_len=64):
    """
    Convert a Polars DataFrame to a NumPy structured array with appropriate dtypes.
    
    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame to convert.
    max_str_len : int
        Maximum length for string columns (for fixed-length Unicode arrays).
    
    Returns
    -------
    np.ndarray
        Structured NumPy array.
    """
    
    dtype = []

    for name, dtype_pol in zip(df.columns, df.dtypes):
        if dtype_pol == pl.Date or name == "date":
            dtype.append((name, 'datetime64[D]'))
        elif dtype_pol == pl.Datetime:
            dtype.append((name, 'datetime64[ns]'))
        elif dtype_pol == pl.Float64:
            dtype.append((name, np.float64))
        elif dtype_pol.is_integer():
            dtype.append((name, np.int64))
        elif dtype_pol == pl.Utf8:
            print("string column:", name)
            dtype.append((name, f'U{max_str_len}'))  # fixed-length Unicode
        else:
            print("value error for column:", name)
            raise ValueError
    
    # Preallocate structured array
    arr = np.empty(len(df), dtype=dtype)
    
    for name in df.columns:
        arr[name] = df[name].to_numpy()
    
    return arr

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

    X = polars_to_structured_array(All[:split_idx])
    Y = polars_to_structured_array(All[split_idx:])

    return X, Y

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

    distribution = {
        "top": 0.3,
        "bottom": 0.7
    }

    if min_rank != -1:
        top_rider_idxs = np.where(
            (All["race_id"] == race_id) & (All["rank"] <= min_rank)
        )[0]
        n = min(min_nr, len(top_rider_idxs))
        return np.random.choice(top_rider_idxs, size=n, replace=False)

    top_rider_idxs = np.where(
        (All["race_id"] == race_id) & (All["rank"] <= 25) #TODO: nr_riders needed in data
    )[0] #top 25 results
    bottom_rider_idxs = np.where(
        (All["race_id"] == race_id) & (All["rank"] > All["nr_riders"] * distribution["bottom"])
    )[0] #bottom results
    
    top = min(min_nr * distribution["top"], len(top_rider_idxs))
    bottom = min(min_nr * distribution["bottom"], len(bottom_rider_idxs))
                 
    return np.concatenate((
        np.random.choice(top_rider_idxs, size= math.ceil(top), replace=False),
        np.random.choice(bottom_rider_idxs, size= math.floor(bottom), replace=False)
    ))

from time import time
def train_model(All: np.ndarray, X: np.ndarray, model: RaceModel, torch_data: torch.tensor) -> None:
    """
    Trains the spline model using stochastic gradient descent.

    Args:
        All: Full dataset.
        X: Training subset.
        spline_model: Model instance to train.
    """
    nr_riders_per_race = model.nr_riders_per_race
    epochs = 5000
    total_loss = 0

    start_time = time()

    for epoch in range(epochs):
        not_early_date = X["date"] > np.datetime64("2014-01-01")
        random_race_id = np.random.choice(X[not_early_date]["race_id"])
        random_rider_idxs = get_random_riders(All, random_race_id, nr_riders_per_race)

        Y_true_ranking = All["rank"][random_rider_idxs]
        loss = model.training_step(
            Y_true_ranking = Y_true_ranking, 
            indices = random_rider_idxs, 
            data = All,
            torch_data = torch_data,
        )
        total_loss += loss

        if epoch % (epochs / 5) == 0:
            # for f in model.feature_functions:
            #     f.save_function(epoch)
            #TODO: combine in self.snapshot_model()
            #TODO: add for model.features_function
            model.neighbor_aggregate_function.save_function(epoch)
            print(f"Epoch {epoch}, Loss: {total_loss}")
            print(f"Feature weights: {model.neighbor_aggregate_function.feature_weights}")
            total_loss = 0
            model.normalize_distance_weights()

    training_time = time() - start_time
    print(f"training time (s) = {training_time}")
    # TODO: fix model.plot_all_learned_functions()
    return training_time

def predict_loss_diff_after_one_training_step(steps: int = 100) -> None:
    """
    Tests loss difference before and after one training step.
    
    and prints difference
    """

    

def compute_model_performance(All: np.ndarray, Y: np.ndarray, model: RaceModel, torch_data: torch.tensor) -> Dict[str, float]:
    """
    Evaluates model performance on test races.

    Args:
        All: Full dataset.
        Y: Test subset.
        model: Trained model.

    Returns:
        Dict of performance metrics (MSE, MAE, R2).
    """

    race_ids = np.unique(Y["race_id"])
    test_size = 50#len(race_ids)
    test_race_ids = np.random.choice(race_ids, size = test_size, replace = False)

    from scipy.stats import spearmanr
    from scipy.stats import kendalltau
    import itertools
    import statistics

    def ranking_accuracy(scores: torch.Tensor, correct_order: list[int]) -> float:
        "computed as pairwise accuracy over top-25 riders"

        # 1 -> perfect order
        # 0.5 -> random order
        # 0 -> completely reversed
        true_ranks = {idx: rank for rank, idx in enumerate(correct_order)}

        total_pairs = 0
        correct_pairs = 0

        for i, j in itertools.combinations(range(len(scores)), 2):
            if true_ranks[i] > 25 and true_ranks[j] > 25: #TODO: parameterize and align 25
                continue
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
    for test_id in test_race_ids:
        #Yeet
        nr_riders_per_race = 25
        top_25_rider_idxs = np.where(
            (All["race_id"] == test_id) & (All["rank"] <= nr_riders_per_race)
        )[0]


        # if len(random_rider_idxs) < nr_riders_per_race:
        #     continue

        Y_true_ranking = All["rank"][top_25_rider_idxs]
        Y_true_order = np.argsort(np.argsort(Y_true_ranking))
        Y_pred_scores = model.predict_ranking_for_race(
            indices = top_25_rider_idxs,
            data = All,
            torch_data = torch_data,
            nr_neighbors=25
        )
        ra.append(ranking_accuracy(Y_pred_scores, Y_true_order))
        # sc.append(spearman_correlation(Y_pred_scores, Y_true_order))
        # kt.append(kendall_tau(Y_pred_scores, Y_true_order))


    print("Pairwise accuracy:", statistics.mean(ra))
    # print("Spearman:", statistics.mean(sc))
    # print("Kendall tau:", statistics.mean(kt))
    return {"Pairwise accuracy:": statistics.mean(ra)}

def to_torch_data(np_data: np.ndarray) -> torch.tensor:
    """Convert structured numpy array to torch tensor."""
    
    #TODO: use All.shape + length of row, instead of converting to unstructured array

    unstructured = np.column_stack([np_data[name].astype(object) for name in np_data.dtype.names])
    torch_data = np.zeros(unstructured.shape, dtype=float)
    for i, col_name in enumerate(np_data.dtype.names):
        col = np_data[col_name]
        try:
            torch_data[:, i] = col.astype(float)
        except (ValueError, TypeError):
            torch_data[:, i] = 0.0
    return torch.from_numpy(torch_data).float()

def get_RVV_2024_id():
    name = "Ronde-Van-Vlaanderen"
    year = "2024"
    races = pl.read_parquet("data/races_df.parquet")
    race_id = races.filter(
        (pl.col("name").str.contains("vlaanderen")) & (pl.col("year") == year)
    ).select("race_id").to_numpy()[0][0]
    return race_id

def test_prediction(model: RaceModel, All: np.ndarray, torch_data: torch.tensor):
    rvv_id = get_RVV_2024_id()
    # rider_idxs = get_random_riders(All, rvv_id, min_nr=100, min_rank=100)
    ranks_to_consider = 100
    rider_idxs = np.where(
        (All["race_id"] == rvv_id) & (All["rank"] <= ranks_to_consider)
    )[0]

    Y_pred_scores = model.predict_ranking_for_race(
        indices = rider_idxs,
        data = All,
        torch_data = torch_data,
        nr_neighbors=25
    )
    predicted_ranks = torch.argsort(Y_pred_scores, descending=True)
    real_ranks = All["rank"][rider_idxs]
    print("Predicted ranking for RVV 2024:")
    # for rank, idx in enumerate(sorted_indices)[:25]:  #top 25
    #     rider_idx = rider_idxs[idx]
    #     rider_name = All["name"][rider_idx]
    #     rider_score = Y_pred_scores[idx].item()
    #     print(f"Rank {rank + 1}: {rider_name} (Score: {rider_score:.4f})")

    
    #only print top 25 real ranks
    sorted_indices_on_real_rank = np.argsort(real_ranks)
    for i in sorted_indices_on_real_rank[:25]:
        real_rank = real_ranks[i]
        rider_idx = rider_idxs[i]
        rider_name = All["name"][rider_idx]
        predicted_rank = predicted_ranks[i]

        rider_score = Y_pred_scores[i].item()
        print(f"{real_rank}: {rider_name}, predicted rank = {predicted_rank} (Score: {rider_score:.4f})")


def main() -> None:

    #TODO: generate test data

    race_result_features = pl.read_parquet("data/features_df.parquet")
    print(race_result_features.dtypes)
    X_Y: tuple[np.ndarray, np.ndarray] = split_train_test(race_result_features)
    All = np.concatenate(X_Y)

    print(All.dtype)
    neural_net = RaceModel(All)

    torch_data = to_torch_data(All)


    training_time = train_model(All, X_Y[0], neural_net, torch_data=torch_data)

    # model_perf_dict = compute_model_performance(
    #     All, X_Y[1], model=neural_net, torch_data=torch_data
    # ) | {
    #     "training_time_seconds": training_time
    # }
    # print("Model performance on test set:", model_perf_dict)

    test_prediction(neural_net, All, torch_data=torch_data)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
    # main2()
    # race_result_features = pl.read_parquet("data/features_df.parquet")

    # X_Y: tuple[np.ndarray, np.ndarray] = split_train_test(race_result_features)
    # All = np.concatenate(X_Y)
    # spline_model = RaceModel(All)


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

