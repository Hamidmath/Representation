import json
import numpy as np
from itertools import combinations

# Load data
with open("data/nonstarShaped_signatures128HammingByArea.json", "r") as f:
    data = json.load(f)

keys = list(data.keys())
n = len(keys)

# Convert all signatures to a single (n, 15360) array
signatures = np.array([np.array(data[k]).flatten() for k in keys], dtype=np.float32)

# Compute pairwise Euclidean distances using broadcasting / efficient matrix math
# Using (a-b)^2 = a^2 + b^2 - 2ab trick
dot_prod = signatures @ signatures.T  # shape (n, n)
sq_norms = np.sum(signatures**2, axis=1, keepdims=True)  # shape (n, 1)
dists = np.sqrt(sq_norms + sq_norms.T - 2*dot_prod)  # shape (n, n)

# Save as dictionary with keys "key1_key2"
results = {}
for i in range(n):
    for j in range(n):  # only upper triangle
        results[f"{keys[i]}_{keys[j]}"] = float(dists[i, j])

with open("data/euclideanNonStarSignatures128HammingByArea.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Done! Fast vectorized computation finished.")
