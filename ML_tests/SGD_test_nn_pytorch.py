# """

# write me example code in python, perform SGD on a dataset to estimate a simple function. Give a small database of 5
# points, 3 features each, and a test point, estimating the 3rd feature by the other 2
# gvliegen (01:5

# """
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim


# class NeuralNetGoesBRRR():
#     class SimpleNet(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.net = nn.Sequential(
#                 nn.Linear(1, 16),
#                 nn.ReLU(),
#                 nn.Linear(16, 1),
#             )

#         def forward(self, x):
#             return self.net(x)
#             # return torch.stack([
#             #     self.net(x[i: i+1])
#             #     for i in len(x)
#             # ])
    
#     def __init__(self):
#         self.model = self.SimpleNet()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

#     def step(self):
#         self.optimizer.step()
    
#     def zero_grad(self):
#         self.optimizer.zero_grad()
    
#     def forward(self, x):
#         return self.model.forward(x)
    
# class CompleteModel():
#     def __init__(self, nr_of_features):
#         self.models = [NeuralNetGoesBRRR()] * nr_of_features
    
#     def step(self):
#         for m in self.models:
#             m.step()
    
#     def zero_grad(self):
#         for m in self.models:
#             m.zero_grad()

#     def forward(self, x):
#         x = x.view(-1, x.shape[-1]) 
#         return torch.sum(
#             torch.stack([
#                 m.forward(x[:, i:i+1] ** (2-i)) #first feature is squared
#                 for i, m in enumerate(self.models)
#             ]),
#             dim = 0
#         ).squeeze(-1)

# def custom_loss_function(y_pred, y_true):
#     # Mean Squared Error + small L1 regularization term
#     mse = torch.mean((y_pred - y_true) ** 2)
#     return mse
    

# # Dataset: 5 points, each with [X1, X2, Y]
# # data = torch.from_numpy(np.array([
# #     [1, 2, 5.5], # 0.5*f1^2 + 2*f_2 + 1 = 5
# #     [2, 3, 9],
# #     [3, 4, 13.5],
# #     [4, 5, 19],
# #     [5, 6, 25.5]
# # ], dtype=np.float32))

# x = torch.rand(100, 2) * 40 - 20
# y = 0.5 * (x[:, 0]**2) + 2*x[:, 1] + 1


# # Test point: [X1, X2]
# test_point = torch.from_numpy(np.array([6, 7], dtype=np.float32)) #-> 18 + 14 + 1 = 33
# y_test_point = 0.5 * (test_point[0]**2) + 2*test_point[1] + 1
# print(f"val to predict = {y_test_point}")

# epochs = 5000

# # model = NeuralNetGoesBRRR()

# # for epoch in range(epochs):
# #     model.zero_grad()

# #     for point in data:
# #         x = point[1:2]  # X1, X2
# #         y_true = point[2]  # Y
# #         y_pred = model.forward(x)
# #         loss = custom_loss_function(y_pred, y_true)
# #         loss.backward()
# #     model.optimizer.step()
# # prediction = model.forward(test_point[1:2])


# model = CompleteModel(nr_of_features=2)

# #go over a few of the points
# # for epoch in range(epochs):
# #     for point in data:
# #         model.zero_grad()
# #         x = point[:2]  # X1, X2
# #         y_true = point[2]  # Y
# #         y_pred = model.forward(x)
# #         loss = custom_loss_function(y_pred, y_true)
# #         loss.backward()
# #         model.step()
# #     if epoch % (epochs/10) == 0:
# #         prediction = model.forward(test_point)
# #         print(f"Predicted Y for test point {test_point}: {prediction}")

# #go over all points each epoch
# for epoch in range(epochs):

#     model.zero_grad()
#     y_pred = model.forward(x)
#     loss = custom_loss_function(y_pred, y)
#     loss.backward()
#     model.step()
#     if epoch % (epochs/10) == 0:
#         prediction = model.forward(test_point)
#         print(f"Predicted Y for test point {test_point}: {prediction}")

# # Predict on test point