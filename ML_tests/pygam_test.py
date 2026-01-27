# from pygam.datasets import wage
# from pygam import LinearGAM, s, f
# import numpy as np

# x, y = wage()
# print(y)

# # X contains the year, age and education of each sampled person.
# #     The education category has been transformed to integers.
# #     y contains the wage.



# # basic example, lamda fedaulted to 0.6
# gam = LinearGAM(s(0) + s(1) + f(2)).fit(x, y)

# # gam.summary()

# #trying out multiple lambdas (penalties on function wiggliness)
# lam = np.logspace(-3, 5, 5)
# print(lam)
# lams = [lam] * 3
# gam.gridsearch(x, y, lam=lams)
# gam.summary()

# #GCV = generalized cross-validation score 

# # high-dimensional search-spaces, it is sometimes a good idea to try a randomized search.
# # We can acheive this by using numpy’s random module:
# random_lams = np.random.rand(100, 3)
# random_lams = random_lams * 6 - 3
# random_lams = 10 ** random_lams

# random_gam = LinearGAM(s(0) + s(1) + f(2)).fit(x, y)
# # random_gam.gridsearch(x, y, lam=random_lams)
# # random_gam.summary()
# print(gam.statistics_["GCV"] <= random_gam.statistics_["GCV"])

# #plotting splines + confidence intervals of 0.95
# import matplotlib.pyplot as plt

# for i, term in enumerate(gam.terms):
#     if term.isintercept:
#         continue

#     XX = gam.generate_X_grid(term=i)
#     pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

#     plt.figure()
#     plt.plot(XX[:, term.feature], pdep)
#     plt.plot(XX[:, term.feature], confi, c='r', ls='--')
#     plt.title(repr(term))
#     plt.show()

# # pyGAM can also fit interactions using tensor products via te()

# from pygam import PoissonGAM, s, te
# from pygam.datasets import chicago

# X, y = chicago(return_X_y=True)

# gam = PoissonGAM(s(0, n_splines=200) + te(3, 1) + s(2)).fit(X, y)


