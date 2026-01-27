# import numpy as np
# from pygam import s, LinearGAM
# import matplotlib.pyplot as plt
# import math
# import pprint
# """

# features
#  - f0: id of racer
#  - f1: distance class
#  - f2: race id 
#  - f7: year
#  - f3: startlist score
#  - f4: binned result year
#  - f5: rank
#  - f6: sick 0 or not sick 1 

#         {
#             {
#                 "id": racer_ids.index(racer_id),
#                 "distance_class": data_point_randomizer.choice([1, 2, 3]),
#                 "race_id": r["race_id"],
#                 "f2": r["startlist_score"],#startlist score 
#                 "f3": 1, #TODO: binned result year #more races in recent years should get an additional boost, but in a log scale
#                 "rank": np.where(r["results"] == racer_id)[0][0]
#             }
# """
# import numpy as np

# class SplinesSGD:
    
#     lr = 5e-3
#     epochs = 1000
#     total_loss = 0

#     def __init__(self, X, feature_idxs = [3, 4, 5, 6], n_splines=5):
#         # [3, 4, 5, 6] = f2, f3, rank, year
#         self.feature_names = ["f2", "f3", "rank", "year"]
#         self.feature_idxs = feature_idxs
#         self.n_splines = 5
#         self.gams = [
#             LinearGAM(s(0, n_splines=n_splines)).fit(X[:,[idx]], X[:, [5]])
#             for idx in feature_idxs
#         ]
#         self.weights = self.init_splines(X)
#         self.Bs = self.init_Bs(X)

#     def init_Bs(self, X):
#         return [
#             self.gams[idx]._modelmat(X[:,[feature_idx]])
#             for idx, feature_idx in enumerate(self.feature_idxs)
#         ]   
    
#     def init_splines(self, X):
#         #init splines to some base functions
#         spline_inits = [
#             lambda x: x/x,
#             lambda x: x/x,
#             lambda x: x/x,
#             lambda x: x/x,
#         ]
#         weights_init = []
#         for idx, feature_idx in enumerate(self.feature_idxs):
#             x_grid = np.linspace(X[:,feature_idx].min(), X[:,feature_idx].max(), 100).reshape(-1,1)
#             B_grid = self.gams[idx]._modelmat(x_grid)
#             y_target = spline_inits[idx](x_grid.ravel())
#             weights_init_new, *_ = np.linalg.lstsq(B_grid.toarray(), y_target, rcond=None)
#             weights_init.append(weights_init_new)

#         return weights_init


#     def grad_descend_pass(self, neighbor_idxs, softmax_w, error):
#         grads = [
#             np.zeros_like(spline_weights)
#             for spline_weights in self.weights
#         ]

#         for i, grad in enumerate(grads):
#             for neighbor_idx, w in zip(neighbor_idxs, softmax_w):
#                 grads[i] += error * w * self.Bs[i][neighbor_idx]

#         self.weights = [
#             w - self.lr * g
#             for w, g in zip(self.weights, grads)
#         ]

#     def compute_y(self, X, neighbor_idxs):

#         s_neighbors = []
#         for j in neighbor_idxs:
#             s_feature_vals = [
#                 B[j] @ w
#                 for B, w in zip(self.Bs, self.weights)
#             ]
#             neighbor_score = s_feature_vals[2] #only feature for rank*h
#             s_neighbors.append(neighbor_score)

#         s_neighbors = np.array(s_neighbors)/len(neighbor_idxs)  #  shape (10,)

#         # log-sum-exp for stability 
#         y_pred = np.sum(s_neighbors)  # = sum(exp(s))

#         return y_pred, s_neighbors

#     def training_step(self, X, x, idxs):
        
        
#         # compute log-terms for each neighbor
#         y_pred, s_neighbors = self.compute_y(X, idxs)

#         # loss derivative
#         y_true = x[5] #X[5] == rank
#         error = y_pred - y_true

#         # softmax weights for distributing gradient across neighbors
#         softmax_w = np.exp(s_neighbors) / np.sum(np.exp(s_neighbors))

#         self.grad_descend_pass(idxs, softmax_w, error)
#         return error

# racer_ids = ["A", "B", "C", "D"]

# def print_results(lists, labels, order_labels, orders):

#     for row, ord_lbl in zip(lists[:5], order_labels[:5]):
#         print(f"values: {row} -> order: {list(ord_lbl)}")

#     # if you want proportions of who comes 1st, 2nd, etc.
#     rank_counts = {label: np.zeros(4, dtype=int) for label in labels}
#     for ord_row in orders:
#         for rank, idx in enumerate(ord_row):
#             rank_counts[labels[idx]][rank] += 1

#     print("\nRank distribution:")
#     for lbl, counts in rank_counts.items():
#         print(f"{lbl}: {counts}")

# def compute_results(n_samples=200):
#     #A > B > C > D
#     A = np.random.randint(0, 15, n_samples)
#     B = np.random.randint(10, 25, n_samples)
#     C = np.random.randint(20, 35, n_samples)
#     D = np.random.randint(25, 40, n_samples)

#     # stack into a list of lists
#     lists = np.column_stack([A, B, C, D])

#     # compute the ordering (argsort gives rank positions per row)
#     orders = np.argsort(lists, axis=1)  # shape (n_samples, 4)

#     # convert to human-readable (e.g., "A before B before C before D")
#     labels = np.array(["A", "B", "C", "D"]) #[id 1, id 2, id 3, id 4]
#     order_labels = labels[orders] #transformed to result e.g. #[id 2, id 3, id 1, id 4]

#     print_results(lists, labels, order_labels, orders)

#     return list(order_labels)


# race_id_counter = 1
# def compute_races(data_point_randomizer, nr = 200):
#     result = compute_results(nr)
#     return [
#         {
#             "f0": data_point_randomizer.uniform(0, 10),#distance #TODO: implement in data some pattern
#             "f1": data_point_randomizer.uniform(0, 10),#height meters #TODO: implement in data some pattern 
#             "year": data_point_randomizer.choice([1, 2, 3], p=[0.4, 0.4, 0.2]),#todo
#             "race_id": x,
#             "startlist_score": data_point_randomizer.uniform(500, 1000),
#             "results": result[x] 
#         }
#         for x in range(nr)
#     ]

# def X_to_dict(x):
#     return {
#         "id": x[0],
#         "distance_class": x[1],
#         "race_id": x[2],
#         "year": x[6],  #f7
#         "f2": x[3],
#         "f3": x[4],
#         "rank": x[5]
#     }

# def compute_data_points(data_point_randomizer, nr_races = 200):
#     races = compute_races(data_point_randomizer=data_point_randomizer, nr=nr_races)

#     results = []
#     for racer_id in racer_ids:
#         results.extend([
#             {
#                 "id": racer_ids.index(racer_id),
#                 "distance_class": data_point_randomizer.choice([1, 2, 3]),
#                 "race_id": r["race_id"],
#                 "f2": r["startlist_score"],#startlist score 
#                 "f3": data_point_randomizer.choice([1, 2, 3]), #TODO: binned result year #more races in recent years should get an additional boost, but in a log scale
#                 "rank": np.where(r["results"] == racer_id)[0][0] + 1, #rank starts at 1
#                 "year": r["year"],  #f7
#                 "sick_or_not_sick": data_point_randomizer.choice([0, 1], p=[0.8, 0.2]) #90% chance of not being sick
#             }
#             for r in races
#         ])
#     return results




# def closest_points(X, y):
#     """mock dist function: return all data points with the same id and distance class
    
#     more entries noramlly for A than for B"""
#     return [
#         i
#         for i, data_point in enumerate(X)
#         if (
#             data_point[1] == int(y[1]) and #same dist class
#             data_point[0] == int(y[0]) and   #same rider id
#             data_point[2] != y[2] and      #different race id
#             data_point[6] <= y[6]  #same or earlier year   
#         )
#     ]


# def train_model(X, model: SplinesSGD):

#     lr = 5e-3
#     epochs = 1000
#     total_loss = 0
#     index_list = list(range(len(X)))
#     for epoch in range(epochs):
#         np.random.shuffle(index_list)
#         total_loss = 0
#         for i in index_list[:10]: # mini-batch of 10
#             # get neighbors for sample i
#             idxs = closest_points(X, X[i])

#             error = model.training_step(X, X[i], idxs)
#             total_loss += error**2

#         if epoch % 100 == 0:
#             print(f"Epoch {epoch}, Loss: {total_loss}")


# def predict(X, y, model):
#     idxs = closest_points(X, y)
#     y_pred = model.compute_y(X, idxs)
#     return y_pred


# def plot_learned_spline(gam, coeffs, feature_idx, title="Learned spline"):
#     # Generate a grid of X values for the feature
#     XX = gam.generate_X_grid(term=0)   # only 1D in gam_f2 and gam_f3
#     B = gam._modelmat(XX)              # spline basis on that grid
#     y_hat = B @ coeffs                 # spline curve

#     plt.figure()
#     plt.plot(XX[:, 0], y_hat, label="learned spline")

#     if feature_idx == 2:
#         # Plot y = 2/x (avoid division by zero)
#         safe_x = XX[:, 0].copy()
#         safe_x[safe_x == 0] = np.nan
#         plt.plot(XX[:, 0], np.log(2.0 / safe_x), "r--", label="y = 2/x")

#     elif feature_idx == 3:
#         # Plot y = x^2
#         plt.plot(XX[:, 0], np.log(XX[:, 0]**2), "g--", label="y = x^2")

#     elif feature_idx == 4:
#         # Plot y = x
#         plt.plot(XX[:, 0], np.log(XX[:, 0]+1), "g--", label="y = x")

#     plt.title(title)
#     plt.xlabel(f"f{feature_idx}")
#     plt.ylabel("spline value")
#     plt.legend()
#     plt.show()

# def split_train_test(All, test_ratio=0.2):
#     max_year = np.max(All[:, 6]) #f7, the year
#     mask = All[:, 6] == max_year
#     X = All[mask, :]
#     Y = All[~mask, :]
#     # np.random.shuffle(X)
#     # split_idx = int(len(X) * (1 - test_ratio))
#     return X, Y

# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# def compute_model_performance(All, Y, model: SplinesSGD):
    
#     y_pred = [model.compute_y(All, closest_points(All, y)[0]) for y in Y]

#     mse = mean_squared_error(Y[:, 5], y_pred)
#     mae = mean_absolute_error(Y[:, 5], y_pred)
#     r2  = r2_score(Y[:, 5], y_pred)

#     return {
#         "MSE": mse,
#         "MAE": mae,
#         "R2": r2
#     }



# def main():
#     data_point_randomizer = np.random.default_rng(42)

#     all_data_points = compute_data_points(data_point_randomizer, nr_races = 200) 
    
#     A_data_points = [a for a in all_data_points if a["id"] == 0 ][:40] #take 40 data points for A
#     B_data_points = [b for b in all_data_points if b["id"] == 1][:20] #take 20 data points for B
#     C_data_points = [c for c in all_data_points if c["id"] == 2][:40] #take 40 data points for C
#     D_data_points = [d for d in all_data_points if d["id"] == 3][:30] #take 30 data points for D
#     all_data_points = A_data_points + B_data_points + C_data_points + D_data_points

#     keys = ["id", "distance_class", "race_id", "f2", "f3", "rank", "year"] #needed for deterministic order

#     All = np.array([[d[k] for k in keys] for d in all_data_points])

#     X, Y = split_train_test(All)

#     spline_model = SplinesSGD(X)
    
#     test_race = 0
#     train_model(np.concatenate([X[:test_race],X[test_race:]]), spline_model)


#     # test_prediction = predict(X, test_race, B_f2, B_f3, B_rank, alpha, beta, gamma)
#     # pprint.pprint(test_prediction)
#     model_perf_dict = compute_model_performance(X, Y, model=spline_model)
#     print("Model performance on test set:", model_perf_dict)
#     pprint.pprint(X_to_dict(X[test_race]))

#     # Plot g(f2)
#     # plot_learned_spline(gam_f2, alpha, feature_idx=2, title="g(f2)")

#     # # Plot h(f3)
#     # plot_learned_spline(gam_f3, beta, feature_idx=3, title="h(f3)")

#     # # Plot rank(f4)
#     # plot_learned_spline(gam_rank, gamma, feature_idx=4, title="rank(f4)")

# def test():
    
#     data_point_randomizer = np.random.default_rng(42)

#     all_data_points = compute_data_points(data_point_randomizer, nr_races = 200)

#     A_data_points = [a for a in all_data_points if a["id"] == 0 ][:40] #take 40 data points for A
#     B_data_points = [b for b in all_data_points if b["id"] == 1][:20] #take 20 data points for B
#     C_data_points = [c for c in all_data_points if c["id"] == 2][:40] #take 40 data points for C
#     D_data_points = [d for d in all_data_points if d["id"] == 3][:30] #take 30 data points for D
#     all_data_points = A_data_points + B_data_points + C_data_points + D_data_points

#     keys = ["id", "distance_class", "race_id", "f2", "f3", "rank"] #needed for deterministic order

#     # Init weights
#     w0, w1, b = 1, 1, 0.0 #init by positive value

#     test_race = 5
#     X = np.array([[d[k] for k in keys] for d in all_data_points])
#     d = closest_points(X, test_race)
#     pprint.pprint(X[test_race])
#     pprint.pprint(d[:5]) #print first 5 closest points

# if __name__ == "__main__":
#     main()
#     # test()