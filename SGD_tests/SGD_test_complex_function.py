"""

write me another example code in python, perform SGD on a dataset to estimate a simple function. Give a small database of
10 points, 5 features each. The function to perform gradient descent on is SUM_x( w_x * f_x) * PRODUCT_k ( w_k * f_k +
b_k), optimizing w_x and w_k , f_j are the features.
gvliegen (02:03 AM)



TO GUESS: 0.5, 0.75, 0.25, 0.4

"""


import numpy as np

# Dataset: 10 points, each with [f0, f1, f2, f3, target]
data = np.array([
    [1.0, 2.0, 3.0, 4.0, 2.4],
    [2.0, 3.0, 4.0, 5.0, 6.5],
    [3.0, 4.0, 5.0, 6.0, 13.5],
    [4.0, 5.0, 6.0, 7.0, 24.15],
    [5.0, 6.0, 7.0, 8.0, 39.2],
    [6.0, 7.0, 8.0, 9.0, 59.4]
    # [7.0, 8.0, 9.0, 10.0, 5040.0],
    # [8.0, 9.0, 10.0, 11.0, 7920.0],
    # [9.0, 10.0, 11.0, 12.0, 11880.0],
    # [10.0, 11.0, 12.0, 13.0, 17160.0]
])

# Test point: [f0, f1, f2, f3]
test_point = np.array([7, 8, 9, 10]) # = 85.5


# Initialize weights: w0, w1, w2, w3

weights = np.ones(5)
learning_rate = 0.000001
epochs = 1000

# SGD for the custom function: (w0*f0 + w1*f1) * (w2*f2) * (w3*f3)
for epoch in range(epochs):
    np.random.shuffle(data)  # Shuffle data for stochasticity
    for point in data:
        f = point[:4]  # f0, f1, f2, f3
        y_true = point[4]  # target
        sum_part = weights[0]*f[0] + weights[1]*f[1]
        prod_part = weights[2]*f[2] * weights[3]*f[3]
        y_pred = sum_part * prod_part + weights[4]  # Adding bias term
        error = y_pred - y_true

        # Gradients
        grad_w0 = error * 2 * f[0] * prod_part
        grad_w1 = error * 2 * f[1] * prod_part
        grad_w2 = error * 2 * sum_part * f[2] * weights[3]*f[3]
        grad_w3 = error * 2 * sum_part * weights[2]*f[2] * f[3]

        weights[0] -= learning_rate * grad_w0
        weights[1] -= learning_rate * grad_w1
        weights[2] -= learning_rate * grad_w2
        weights[3] -= learning_rate * grad_w3

# Predict on test point
sum_part_test = weights[0]*test_point[0] + weights[1]*test_point[1]
prod_part_test = weights[2]*test_point[2] * weights[3]*test_point[3]
prediction = sum_part_test * prod_part_test
print(f"Predicted target for test point {test_point}: {prediction}")
print(f"Weights: {weights}")