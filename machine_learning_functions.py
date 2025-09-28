import numpy as np
from pygam import s, LinearGAM
import matplotlib.pyplot as plt
import math
import pprint
import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sympy as sp


class Spline():

    lr = 1e-3
    def __init__(
        self,
        spline_init_func: callable, 
        n_splines: int, 
        feature_idx: int, 
        feature_name: str, 
        all_data: np.array
    ):  
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.gam = LinearGAM(s(0, n_splines=n_splines)).fit(all_data[:,[feature_idx]], all_data[:, [feature_idx]])
        self.weights_init = self._init_spline(
            data = all_data, 
            init_func = spline_init_func
        )
        self.weights = self.weights_init
        self.basis = self.gam._modelmat(all_data[:,[feature_idx]])

    def compute(self, indexes: list[int]) -> np.array:
        return self.basis[indexes] @ self.weights

    def _init_spline(self, data: np.array, init_func: callable):
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
    
    def grad_descend_pass(self, neighbor_idxs, softmax_w, error):
        # lam2 = 10.0
        # smooth_penalties = [
        #     spline_weights.T @ penalty_matrix @ spline_weights
        #     for spline_weights, penalty_matrix in zip(self.weights, self.Ps) 
        # ]
        
        # Total loss
        # total_loss = mse_loss + self.lam_smooth * smooth_penalty
        
        # # Gradients
        # # ∂MSE/∂β = -2 * X^T @ (y - X @ β) / n
        # mse_grad = -2 * X_basis.T @ (y - y_pred) / len(y)
        
        # # ∂smooth_penalty/∂β = 2 * P @ β
        # smooth_grad = 2 * self.lam_smooth * P @ coefficients
        
        # # Total gradient
        # total_grad = mse_grad + smooth_grad

        weighted_basis_sum = softmax_w * self.basis[neighbor_idxs]
        gradient_pass = error * weighted_basis_sum
        self.weights -= self.lr * gradient_pass


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

class SplinesSGD:
    
    nr_riders_per_race = 5

    def __init__(self, X):
        # 0-4: name  ┆ race_id ┆ distance_km ┆ elevation_m ┆ profile_score ┆ 
        # 5-9: profile_score_last_25k ┆ classification ┆ date ┆ rank  ┆ startlist_score ┆ 
        # 10-11: age | rank_bucket  

        self.init_feature_funcs = self._init_feature_funcs(X)
        self.feature_idxs = [8, 9, 11]
        self.feature_names = ["rank", "startlist_score", "rank_bucket"]

        self.splines: list[Spline] = [
            Spline(
                spline_init_func = self.init_feature_funcs[feature_idx], 
                n_splines = 15, 
                feature_idx = feature_idx, 
                feature_name = feature_name,
                all_data = X
            ) for feature_idx, feature_name in zip(
                self.feature_idxs,
                self.feature_names
            )
        ] 
    
    def _init_feature_funcs(self, X):
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

    def grad_descend_pass(self, neighbor_idxs, softmax_w, error):
        for spline in self.splines:
            spline.grad_descend_pass(
                neighbor_idxs = neighbor_idxs,
                softmax_w = softmax_w,
                error = error
            )


    def get_closest_points(self, X, y) -> list[int]:
        """mock dist function: return all data points with the same id and distance class
        
        more entries noramlly for A than for B"""
        return [
            i
            for i, data_point in enumerate(X)
            if (
                data_point[0] == y[0] and #same name
                data_point[1] != y[1] and      #different race id
                data_point[6] <= y[6]  #same or earlier year   
            )
        ]

    def compute_y(self, 
        all_data: np.array, 
        x: np.array
    ) -> tuple[float, np.array]:
        neighbor_idxs = self.get_closest_points(
            X = all_data, 
            y = x
        )
        s_feature_vals: np.array = np.vstack([
            spline.compute(indexes = neighbor_idxs)
            for spline in self.splines
        ])
        neighbor_scores = np.prod(s_feature_vals, axis = 0) #only feature for rank*h
        neighbor_scores = np.array(neighbor_scores)/len(neighbor_idxs)  #  shape (10,)
        y_pred = np.sum(neighbor_scores)
        return y_pred, neighbor_idxs, neighbor_scores

    possible_buckets = [
        [1, 3],
        [4, 8],
        [9, 15],
        [16, 25],
        [26, 99999]
    ]
    def ranks_to_buckets(self, ranks: np.ndarray) -> np.ndarray:
        buckets = []
        for r in ranks:
            for i, b in enumerate(self.possible_buckets):
                if r >= b[0] and r <= b[1]:
                    buckets.append(i)
                    break
        assert len(ranks) == len(buckets), f"error in mapping ranks to buckets: {ranks} -> {buckets}"
        return np.array(buckets)
        
    def calculate_errors(self, y_pred: list, y_true: np.ndarray):
        """
        y_pred: predicted scores of each rider
        y_true: actual ranking

        returns:
        an error value for each rider, in the same order as y_pred

        current error: nr of buckets to high or to low

        # e.g. pred order = 4, true order = 1 -> error = 3, should predict higher score
        #    
        # e.g. -2 -> predicted order was 2 and true order was 4 -> should predict lower
        # e.g. 6 -> predicited order was 8 and true order was 2 -> should predict higher
        """
        y_true_order = np.argsort(np.argsort(y_true)) #higher rank -> later in order

        y_pred_order = np.argsort(np.argsort( -1 * np.array(y_pred))) #higher score -> early in order
        
        y_pred_ranks = [
            y_true[np.where(y_true_order == order_to_find)[0]][0]
            for order_to_find in y_pred_order
        ]


        errors = self.ranks_to_buckets(y_pred_ranks) - self.ranks_to_buckets(y_true)

        return errors

    def training_step(self, Y_true, indices, data):

        Y_pred = []
        neighbor_scores_s = []
        neighbor_idxs_s = []
        for index in indices:

        # compute log-terms for each neighbor
            new_y_pred, new_neighbor_idxs, new_neighbor_scores = self.compute_y(
                all_data = data, 
                x = data[index]
            )
            Y_pred.append(new_y_pred)
            neighbor_scores_s.append(new_neighbor_scores)
            neighbor_idxs_s.append(new_neighbor_idxs)

        errors = self.calculate_errors(Y_pred, Y_true)
        
        for error, neighbor_scores, neighbor_idxs in zip(errors, neighbor_scores_s, neighbor_idxs_s):
            # softmax weights for distributing gradient across neighbors
            softmax_w = np.exp(neighbor_scores) / np.sum(np.exp(neighbor_scores))
            self.grad_descend_pass(neighbor_idxs, softmax_w, error)
        return errors


    def plot_learned_splines(self):
        for gam, start_weights, weights, feature_idx in zip(self.gams, self.weights_init, self.weights, self.feature_idxs):

        # Generate a grid of X values for the feature
            XX = gam.generate_X_grid(term=0)   # only 1D in gam_f2 and gam_f3
            B = gam._modelmat(XX)              # spline basis on that grid
            y_hat = B @ weights                 # spline curve

            plt.figure()
            plt.plot(XX[:, 0], y_hat, label="learned spline")

            y_start = B @ start_weights
            plt.plot(XX[:, 0], y_start, "r--", label=str(feature_idx))

            plt.xlabel(f"f{feature_idx}")
            plt.ylabel("spline value")
            plt.legend()
            plt.show()
    
    def performance(self, All, Y):

        test_size = 50
        race_ids = np.unique(Y[:, 1])
        test_race_ids = np.random.choice(race_ids, size = test_size, replace = False)

        Y_true_buckets = []
        Y_pred_buckets = []
        for test_race in test_race_ids:
            random_race_idxs = np.where(All[:,1] == test_race)[0]

            n = min(self.nr_riders_per_race, len(random_race_idxs))
            random_rider_idxs = np.random.choice(random_race_idxs, size=n, replace=False)

            if len(random_rider_idxs) < 2:
                continue

            neighbor_idxs_s = [
                get_closest_points(All, All[rider_idx])
                for rider_idx in random_rider_idxs
            ]
            race_pred = []
            for neighbor_idxs in neighbor_idxs_s:

            # compute log-terms for each neighbor
                new_y_pred, _ = self.compute_y(neighbor_idxs)
                race_pred.append(new_y_pred)

            y_true_ranks = All[random_rider_idxs, 8]
            y_true_order = np.argsort(np.argsort(y_true_ranks)) 
            y_true_buckets = self.ranks_to_buckets(y_true_ranks)
            Y_true_buckets.extend(y_true_buckets)
            y_pred_order = np.argsort(np.argsort( -1 * np.array(race_pred))) #higher score -> early in order
            y_pred_ranks = [
                y_true_ranks[np.where(y_true_order == order_to_find)[0]][0]
                for order_to_find in y_pred_order
            ]
            y_pred_buckets = self.ranks_to_buckets(y_pred_ranks)
            Y_pred_buckets.extend(y_pred_buckets)


        mse = mean_squared_error(Y_true_buckets, Y_pred_buckets)
        mae = mean_absolute_error(Y_true_buckets, Y_pred_buckets)
        r2  = r2_score(Y_true_buckets, Y_pred_buckets)


        self.plot_learned_splines()
        return {
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }



def split_train_test(All: pl.DataFrame, test_ratio=0.2) -> tuple[np.ndarray, np.ndarray]:
    split_idx = int((1 - test_ratio) * All.height)
    return All[:split_idx].to_numpy(), All[split_idx:].to_numpy()

def train_model(All, X, spline_model: SplinesSGD):

    nr_riders_per_race = 8
    epochs = 1000
    total_loss = 0
    for epoch in range(epochs):
        random_race_id = np.random.choice(X[:,1]) #take id out of X
        random_top_race_idxs = np.where(
            (All[:,1] == random_race_id) & (All[:, 8] <= 25)
        )[0] #top 25 results
        random_bottom_race_idxs = np.where(
            (All[:,1] == random_race_id) & (All[:, 8] >= 25)
        )[0] #bottom 25 results


        n = min(nr_riders_per_race, len(random_top_race_idxs))
        random_rider_idxs = np.concatenate((
            np.random.choice(random_top_race_idxs, size=int(n/2), replace=False),
            np.random.choice(random_bottom_race_idxs, size=int(n/2), replace=False)
        ))

        if n < 2:
            continue

        # neighbor_idxs_s = [
        #     get_closest_points(All, All[rider_idxs])
        #     for rider_idxs in random_rider_idxs
        # ]

        Y_true = All[random_rider_idxs, 8]
        errors = spline_model.training_step(
            Y_true = Y_true, 
            indices = random_rider_idxs, 
            data = All
        )
        total_loss += sum([abs(error) for error in errors])

        if epoch % (epochs/10) == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")
            total_loss = 0


def compute_model_performance(All, Y, model: SplinesSGD):
    return model.performance(All, Y)

def main():
    race_result_features = pl.read_parquet("data/features_df.parquet")

    X_Y: tuple[np.ndarray, np.ndarray] = split_train_test(race_result_features)
    All = np.concatenate(X_Y)
    spline_model = SplinesSGD(All)
    
    train_model(All, X_Y[0], spline_model)


    # test_prediction = predict(X, test_race, B_f2, B_f3, B_rank, alpha, beta, gamma)
    # pprint.pprint(test_prediction)
    model_perf_dict = compute_model_performance(All, X_Y[1], model=spline_model)
    print("Model performance on test set:", model_perf_dict)
    

if __name__ == "__main__":
    main()

# Quick decision flow (what to try, in order)

# Baseline: simple model (logistic/regression) with minimal transforms + proper CV.

# Try GBDT (LightGBM/XGBoost/CatBoost) with basic hyperparameter tuning — often wins on small data with nonlinearities.

# If you need interpretability of specific nonlinear effects, try GAM or spline + Ridge.

# Use ensemble of GAM + GBDT if you need better predictive power and can afford complexity.


#nonlinearities: of fgetting the score of a rider for each race
# and then adding all the scores to get a total score:
#model training options


# Tree-based models (e.g. Gradient Boosted Trees, Random Forest):
# → Yes, they can approximate such nonlinearities, though not in closed form. They partition the feature space, so they’d learn "if feature_d is big, reduce score," etc. They approximate your formula but don’t give a clean multiplicative form.
# Let tree models approximate it:
# If interpretability isn’t critical, XGBoost / LightGBM will discover nonlinear interactions and approximate the shape of your formula automatically.


# Generalized Additive Models (GAMs):
# → They can learn separate nonlinear transformations per feature (e.g. the curve 
# 1/(x+1)
# 1/(x+1)), and then sum them. But they add them, not multiply.
# → You could log-transform your score and use GAM, because:

# log⁡(score)=−log⁡(a)−log⁡(b+1)+log⁡(1+c)−log⁡(d+1)
# log(score)=−log(a)−log(b+1)+log(1+c)−log(
# d



#Training, A energy-based model:
# Use NLL (cross-entropy) against the true label as the primary loss. You can add a small entropy regularizer if you want to penalize over-confidence (a confidence penalty), not minimize entropy.

# Regularization you should use

# Smoothing penalty on each spline term (limits wiggliness). Libraries usually provide/optimize this.

# L2 penalty on coefficients (ridge) — helps stability if you use basis expansions.

# Temperature / calibration: make 
# T
# T a hyperparameter or learn a scalar on a validation set.

# Diagnostics / tests

# Plot each learned 
# fj(xj)

# ) to check they are sensible (GAMs give these directly).

# Check cross-validated NLL and classification metrics.

# Check calibration curves (Brier score, reliability diagrams).

# Measure confidence / entropy on out-of-distribution examples — watch for overconfidence.




#Training preidicting high results is more important:
# sklearn estimators usually accept sample_weight in .fit(X,y, sample_weight=...)