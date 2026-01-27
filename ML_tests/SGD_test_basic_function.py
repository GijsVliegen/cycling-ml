# """

# write me example code in python, perform SGD on a dataset to estimate a simple function. Give a small database of 5
# points, 3 features each, and a test point, estimating the 3rd feature by the other 2
# gvliegen (01:5

# """
# import numpy as np

# # Dataset: 5 points, each with [X1, X2, Y]
# data = np.array([
#     [1, 2, 5], # 2*f_2 + 1 = 5
#     [2, 3, 7],
#     [3, 4, 9],
#     [4, 5, 11],
#     [5, 6, 13]
# ])

# # Test point: [X1, X2]
# test_point = np.array([6, 7])

# # Initialize weights and bias
# weights = np.zeros(2)  # for X1 and X2
# bias = 0.0
# learning_rate = 0.01
# epochs = 50 #1000

# # SGD for linear regression
# for epoch in range(epochs):
#     for point in data:
#         X = point[:2]  # X1, X2
#         y_true = point[2]  # Y
#         y_pred = np.dot(weights, X) + bias
#         error = y_pred - y_true
#         weights -= learning_rate * error * X
#         bias -= learning_rate * error

# # Predict on test point
# prediction = np.dot(weights, test_point) + bias
# print(f"Predicted Y for test point {test_point}: {prediction}")