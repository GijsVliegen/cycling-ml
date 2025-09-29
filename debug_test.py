import numpy as np

# Simulate the problematic code
neighbor_idxs = [0, 1, 2, 3, 4, 5]  # example list
neighbor_distances = np.array([0.5, 0.1, 0.8, 0.2, 0.9, 0.3])  # example distances

print("neighbor_idxs:", neighbor_idxs, type(neighbor_idxs))
print("neighbor_distances:", neighbor_distances, type(neighbor_distances))

if len(neighbor_idxs) > 3:  # simulate >25
    sorted_idx = np.argsort(neighbor_distances)[:3]
    print("sorted_idx:", sorted_idx, type(sorted_idx), sorted_idx.dtype)
    try:
        neighbor_idxs_new = np.array(neighbor_idxs)[sorted_idx].tolist()
        print("Success: neighbor_idxs_new =", neighbor_idxs_new)
    except Exception as e:
        print("Error:", e)