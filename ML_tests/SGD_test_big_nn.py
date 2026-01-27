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
#             """Correlated features -> combined in input"""
#             super().__init__()
#             self.net = nn.Sequential(
#                 nn.Linear(2, 32),
#                 nn.ReLU(),
#                 nn.Linear(32, 1)
#             )

#             """Unrelated features -> later on combined"""
#             # super().__init__()
#             # self.f1 = nn.Sequential(
#             #     nn.Linear(1, 8),
#             #     nn.ReLU(),
#             # )
#             # self.f2 = nn.Sequential(
#             #     nn.Linear(1, 8),
#             #     nn.ReLU(),
#             # )
#             # self.merge = nn.Sequential(
#             #     nn.Linear(16, 8),
#             #     nn.ReLU(),
#             #     nn.Linear(8, 1)
#             # )

#         def forward(self, x):
#             return self.net(x)

#             # x = x.view(-1, x.shape[-1]) 
#             # m1 = self.f1(x[:, 0:1])
#             # m2 = self.f2(x[:, 1:2])
#             # m = torch.cat((m1, m2), dim=1)
#             # return self.merge(m)
    
#     def __init__(self):
#         self.model = self.SimpleNet()
#         self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)

#     def step(self):
#         self.optimizer.step()
    
#     def zero_grad(self):
#         self.optimizer.zero_grad()
    
#     def forward(self, x):
#         return self.model(x).squeeze(-1)
    
# def custom_loss_function(y_pred, y_true):
#     mse = torch.mean((y_pred - y_true) ** 2)
#     return mse
    

# x = torch.rand(100, 2) * 20 - 10
# y = 0.5 * (x[:, 0]**2) + 2*x[:, 1] + 1


# # Test point: [X1, X2]
# test_point = torch.from_numpy(np.array([6, 7], dtype=np.float32)) #-> 18 + 14 + 1 = 33
# y_test_point = 0.5 * (test_point[0]**2) + 2*test_point[1] + 1
# print(f"val to predict = {y_test_point}")

# epochs = 5000

# model = NeuralNetGoesBRRR()
# criterion = nn.MSELoss()

# for epoch in range(epochs):

#     model.zero_grad()
#     y_pred = model.forward(x)
#     # loss = custom_loss_function(y_pred, y)
#     loss = criterion(y_pred, y)
#     loss.backward()
#     model.step()
#     if epoch % (epochs/10) == 0:
#         prediction = model.forward(test_point)
#         print(f"Predicted Y for test point {test_point}: {prediction}")

# # Predict on test point