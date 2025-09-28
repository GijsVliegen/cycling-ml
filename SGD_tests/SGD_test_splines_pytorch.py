"""

This time, assume the same training data and function to estimate as in
SGD_test_basic_function.py. Except, feature 3 and feature 4 are inputted to
splines, which need to be learned aswell. To do this, import a spline basis
generation form pyGAM and extract the design matrix, integrate this in the
solution you already had

use pytorch 


***********************************

TO GUESS: weights: 0.5, 0.75, 0.25, 0.4, functions: f_2 -> 2 * inverse, f_3, squared

"""
import torch
from pygam.terms import SplineTerm
import numpy as np

# Dataset: 10 points, each with [f0, f1, f2, f3, target]
data_np = np.array([
    [1.0, 2.0, 3.0, 4.0, 21.12],
    [2.0, 3.0, 4.0, 5.0, 40.65],
    [3.0, 4.0, 5.0, 6.0, 64.8],
    [4.0, 5.0, 6.0, 7.0, 92.97],
    [5.0, 6.0, 7.0, 8.0, 125.4],
    [6.0, 7.0, 8.0, 9.0, 167]
    # [7.0, 8.0, 9.0, 10.0, 5040.0],
    # [8.0, 9.0, 10.0, 11.0, 7920.0],
    # [9.0, 10.0, 11.0, 12.0, 11880.0],
    # [10.0, 11.0, 12.0, 13.0, 17160.0]
])
data = torch.tensor(data_np, dtype=torch.float32)

# Test point: [f0, f1, f2, f3]
test_point_np = np.array([7, 8, 9, 10]) # = 210.9
test_point = torch.tensor(test_point_np, dtype=torch.float32)

# Spline terms for f2 and f3
n_splines = 5
spline_term_f2 = SplineTerm(0, n_splines=n_splines, spline_order=1, lam=0.6)  # TODO: extract lam and spline_order
spline_term_f3 = SplineTerm(0, n_splines=n_splines, spline_order=1, lam=0.6) 

# Parameters
weights = torch.nn.Parameter(torch.zeros(2))  # w0, w1
w_spline2 = torch.nn.Parameter(torch.zeros(n_splines))
w_spline3 = torch.nn.Parameter(torch.zeros(n_splines))

learning_rate = 0.000001
epochs = 100
optimizer = torch.optim.SGD([weights, w_spline2, w_spline3], lr=learning_rate)

# SGD for the custom function: (w0*f0 + w1*f1) * (spline_f2) * (spline_f3)
for epoch in range(epochs):
    # Shuffle data
    indices = torch.randperm(data.size(0))
    data_shuffled = data[indices]

    for point in data_shuffled:
        f = point[:4]
        y_true = point[4]

        # Compute spline bases (numpy, then to torch)
        basis_f2_np = spline_term_f2.build_columns(f[2].item(), None)[0]
        basis_f3_np = spline_term_f3.build_columns(f[3].item(), None)[0]
        basis_f2 = torch.tensor(basis_f2_np, dtype=torch.float32)
        basis_f3 = torch.tensor(basis_f3_np, dtype=torch.float32)

        sum_part = weights[0]*f[0] + weights[1]*f[1]
        spline_f2_val = torch.dot(basis_f2, w_spline2)
        spline_f3_val = torch.dot(basis_f3, w_spline3)
        y_pred = sum_part * spline_f2_val * spline_f3_val
        loss = (y_pred - y_true)**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Predict on test point
basis_f2_test_np = spline_term_f2.build_columns(test_point[2].item(), None)[0]
basis_f3_test_np = spline_term_f3.build_columns(test_point[3].item(), None)[0]
basis_f2_test = torch.tensor(basis_f2_test_np, dtype=torch.float32)
basis_f3_test = torch.tensor(basis_f3_test_np, dtype=torch.float32)

sum_part_test = weights[0]*test_point[0] + weights[1]*test_point[1]
spline_f2_test = torch.dot(basis_f2_test, w_spline2)
spline_f3_test = torch.dot(basis_f3_test, w_spline3)
prediction = sum_part_test * spline_f2_test * spline_f3_test
print(f"Predicted target for test point {test_point_np}: {prediction.item()}")