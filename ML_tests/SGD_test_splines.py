# import numpy as np
# from pygam import s, LinearGAM
# import matplotlib.pyplot as plt
# import math
# """

# function to learn = (w0*f0 + w1*f1) * (spline_f2) * (spline_f3) + bias

# Tweights: 0.5, 0.75, functions: f_2 -> 2 * inverse, f_3 -> squared

# """
# # Dataset: 10 points, each with [f0, f1, f2, f3, target]

# def compute_value(f):
#     spline_f2 = 2 / f[2]  # 2 * inverse of f2
#     spline_f3 = f[3] ** 2  # squared of f3
#     return (0.5 * f[0] + 0.25 * f[1]) * spline_f2 * spline_f3 + 0

# data_point_generator = np.random.default_rng(42)
# X = data_point_generator.uniform(5, 12, (100, 4))

# # Compute function values
# y = [compute_value(data_point) for data_point in X]


# # Build spline terms for f2 and f3
# n_splines = 5
# gam_f2 = LinearGAM(s(0, n_splines=n_splines)).fit(X[:,[2]], y)
# gam_f3 = LinearGAM(s(0, n_splines=n_splines)).fit(X[:,[3]], y)

# # Extract spline design matrices
# B_f2 = gam_f2._modelmat(X[:,[2]])   # shape (N, n_splines)
# B_f3 = gam_f3._modelmat(X[:,[3]])

# # Init weights
# w0, w1, b = 1, 1, 0.0 #init by positive value

# def init_splines(X, gam_f2, gam_f3):
#     #init splines to some base functions

#     def g0(x):   # base guess for g(f2): inverse
#         return np.log(1 / x)

#     def h0(x):   # base guess for h(f3): exponential
#         return np.log(x ** 3)

#     x2_grid = np.linspace(X[:,2].min(), X[:,2].max(), 100).reshape(-1,1)
#     B_f2_grid = gam_f2._modelmat(x2_grid)
#     y2_target = g0(x2_grid.ravel())

#     alpha_init, *_ = np.linalg.lstsq(B_f2_grid.toarray(), y2_target, rcond=None)

#     x3_grid = np.linspace(X[:,3].min(), X[:,3].max(), 100).reshape(-1,1)
#     B_f3_grid = gam_f3._modelmat(x3_grid)
#     y3_target = h0(x3_grid.ravel())

#     beta_init, *_ = np.linalg.lstsq(B_f3_grid.toarray(), y3_target, rcond=None)

#     return alpha_init, beta_init

# alpha, beta = init_splines(X, gam_f2, gam_f3)

# lr  = 1e-3       # for w0, w1, b
# epochs = 2000

# index_list = list(range(len(X)))

# # SGD for the custom function: (w0*f0 + w1*f1) * (spline_f2) * (spline_f3)
# for epoch in range(epochs):
#     np.random.shuffle(index_list)
#     total_loss = 0
#     for i in index_list[:10]: # mini-batch of 10
#         f0, f1 = X[i,0], X[i,1]

#         y_true_log = np.log(y[i])

#         # Compute spline bases
#         f2_splined_logged = B_f2[i] @ alpha
#         f3_splined_logged = B_f3[i] @ beta

#         sum_part = w0*f0 + w1*f1
#         y_pred_log =  np.log(sum_part) + f2_splined_logged + f3_splined_logged + b

#         if math.isnan(y_pred_log) or math.isinf(y_pred_log):
#             print("Numerical issue encountered, stopping training.")
#             print("f_2:", X[i,2], "f_3:", X[i,3], "f2_splined:", f2_splined_logged, "f3_splined:", f3_splined_logged)
#             break

#         error = y_pred_log - y_true_log
#         total_loss += error**2

#         # Gradients
#         grad_w0 = 2 * error * f0
#         grad_w1 = 2 * error * f1
#         grad_b  = 2 * error

#         grad_alpha = 2 * error * B_f2[i]
#         grad_beta  = 2 * error * B_f3[i]

#         # Update
#         w0 -= lr * grad_w0
#         w1 -= lr * grad_w1
#         b  -= lr * grad_b
#         alpha -= lr * grad_alpha
#         beta  -= lr * grad_beta

#     if epoch % 500 == 0:
#         print(f"Epoch {epoch}, Loss: {total_loss}")

# # Predict on test point

# def predict(x):
#     f0, f1, f2, f3 = x
#     g_val_log = (gam_f2._modelmat([[f2]]) @ alpha)[0]
#     h_val_log = (gam_f3._modelmat([[f3]]) @ beta)[0]
#     sum_part = w0*f0 + w1*f1
#     y_pred_log = np.log(sum_part) + f2_splined_logged + f3_splined_logged + b


#     return np.exp(y_pred_log)

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

#     plt.title(title)
#     plt.xlabel(f"f{feature_idx}")
#     plt.ylabel("spline value")
#     plt.legend()
#     plt.show()


# # Plot g(f2)
# plot_learned_spline(gam_f2, alpha, feature_idx=2, title="g(f2)")

# # Plot h(f3)
# plot_learned_spline(gam_f3, beta, feature_idx=3, title="h(f3)")

# print("Learned weights: {w_0}, {w_1}, bias: {b}".format(w_0=w0, w_1=w1, b=b))
# test_point = np.array([7,8,9,10])  # = 122.9
# print("Prediction:", predict(test_point))
