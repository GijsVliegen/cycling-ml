from scipy.stats import qmc, beta,norm
import numpy as np
import matplotlib.pyplot as plt

env = "be-hematology"
env_nr = 1

de_hema_config = {
    "channels": [
        "pd", #darzalex
        "wd",
        "fd",
        "dd",
        "pi", #imbruvica
        "wi",
        "fi",
        "di",
        "pte", #tecvayli
        "wte",
        "fte",
        "dte",
        "pta", #talvey
        "wta",
        "fta",
        "dta",
    ],
    "l_bounds": [0] * 16,
    "u_bounds": [
        1, 1, 6, 4, 1, 1, 4, 3, 1, 1, 2, 2, 1, 1, 2, 2
    ]
}
be_hema_config = {
    "channels": [
        "phone darzalex", #darzalex
        "webcall darzalex",
        "f2f darzalex",
        "event virtual darzalex",
        "email darzalex",
        "event f2f darzalex",
        "phone imbruvica", #imbruvica
        "webcall imbruvica",
        "f2f imbruvica",
        "event virutal imbruvica",
        "email imbruvica",
        "event f2f imbruvica"
    ],
    "l_bounds": [0] * 12,
    "u_bounds": [
        3, 4, 9, 1, 7, 1, 3, 4, 9, 1, 7, 1
    ]
}
it_erl_config = {
    "channels": [
        "p", 
        "w",
        "f",
        "ev",
        "d",
        # "ef",
    ],
    "l_bounds": [0] * 5,
    "u_bounds": [
        1, 1, 11, 1, 11
    ]
}
es_darz_config = {
    "channels": [
        "p", 
        "w",
        "f",
        "ev",
        "d",
        "ef"
    ],
    "l_bounds": [0] * 6,
    "u_bounds": [
        1, 5, 23, 3, 7, 5
    ]
}
gb_tremfya_pso = {
    "channels": [
        "p", 
        "w",
        "f",
        "ev",
        "d",
        "ef"
    ],
    "l_bounds": [0, 1, 2, 0, 0, 0],
    "u_bounds": [
        3, 4, 5, 1, 5, 1
    ]
}
fr_tremfya_pso = {
    "channels": [
        "p", 
        "w",
        "f",
        # "ev",
        "d",
        "ef"
    ],
    "l_bounds": [0, 0, 0, 1, 0],
    "u_bounds": [
        1, 1, 8, 7, 1
    ]
}

configs = {
    "de-hematology": de_hema_config,
    "be-hematology": be_hema_config,
    "it-erleada-all": it_erl_config,
    "es-darzalex": es_darz_config,
    "gb-tremfya-pso": gb_tremfya_pso,
    "fr-tremfya-pso": fr_tremfya_pso
}

config = configs[env]
config = list(configs.values())[env_nr]

l_bounds_array = np.array(config["l_bounds"])
u_bounds_array_exclusive = np.array(config["u_bounds"])
u_bounds_array_inclusive = u_bounds_array_exclusive - 1
dims = len(config["channels"])

sampler = qmc.LatinHypercube(d=dims)
max_total = sum(u_bounds_array_inclusive)
print(f"max interactions = {max_total}")
print(f"total nr of combinations = {np.prod(u_bounds_array_exclusive - l_bounds_array)}")

X = sampler.random(n=3000)
lhs_samples = qmc.scale(X, l_bounds_array, u_bounds_array_exclusive) 
lhs_samples = np.floor(lhs_samples).astype(int)


# ------- add minimum and maximum edge cases -------
lhs_samples = np.unique(lhs_samples, axis=0)

#stats
non_zero_channels = np.sum((u_bounds_array_inclusive - l_bounds_array) > 0)
totals_stddev = np.std(lhs_samples.sum(axis=1))
shifts = totals_stddev * 2 / non_zero_channels
print(f"total shift summed stddev = {shifts * non_zero_channels:.6f}")

channel_ranges = u_bounds_array_exclusive - l_bounds_array
theoretical_channel_stddevs = np.sqrt(((channel_ranges - 1) * (channel_ranges + 1))/12)
shifts = theoretical_channel_stddevs * 2 / np.sqrt(non_zero_channels)
total_shift = np.sum(shifts)
print(f"total shift channel stddev = {total_shift:.6f}")

# ----- wider spread -----------
X_wide = sampler.random(n=1000)
lhs_samples_wide = qmc.scale(X_wide, l_bounds_array, u_bounds_array_exclusive) 
lhs_samples_min = l_bounds_array
lhs_samples_min_ones = lhs_samples_min + np.eye(dims, dtype=int)
lhs_samples_min_ones = np.minimum(u_bounds_array_inclusive, lhs_samples_min_ones)
lhs_samples_max = u_bounds_array_inclusive
lhs_samples_max_ones = lhs_samples_max - np.eye(dims, dtype=int)
lhs_samples_max_ones = np.maximum(l_bounds_array, lhs_samples_max_ones)
lhs_samples_lower = np.maximum(lhs_samples_wide - shifts, l_bounds_array)
lhs_samples_upper = np.minimum(lhs_samples_wide + shifts, u_bounds_array_inclusive)
lhs_samples_wide = np.vstack((lhs_samples_wide, lhs_samples_min, lhs_samples_min_ones, lhs_samples_max, lhs_samples_max_ones))
lhs_samples_wide = np.vstack((lhs_samples_wide, lhs_samples_lower, lhs_samples_upper))
lhs_samples_wide = np.floor(lhs_samples_wide).astype(int)
lhs_samples_wide = np.unique(lhs_samples_wide, axis=0)


# Plot distributions
plt.figure(figsize=(10, 5))
plt.hist(lhs_samples.sum(axis=1), bins=range(0, max_total + 2), alpha=0.5, label='Uniform')
plt.hist(lhs_samples_wide.sum(axis=1), bins=range(0, max_total + 2), alpha=0.5, label='Uniform')
plt.xlabel('Sum of samples')
plt.ylabel('Frequency')
plt.title('Distribution of summed samples')
plt.legend()
plt.show()

# Plot distributions for each column
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
for i in range(dims):
    ax = axes[i // 4, i % 4]
    ax.hist(lhs_samples[:, i], bins=range(l_bounds_array[i], u_bounds_array_exclusive[i] + 1), alpha=0.5, label='lhs_samples')
    ax.hist(lhs_samples_wide[:, i], bins=range(l_bounds_array[i], u_bounds_array_exclusive[i] + 1), alpha=0.5, label='lhs_samples_wide')
    ax.set_title(config["channels"][i])
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()
plt.tight_layout()
plt.show()

