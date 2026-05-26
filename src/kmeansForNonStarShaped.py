import json
import numpy as np
from scipy.cluster.vq import kmeans, vq

rep = "This report summarizes the results of applying k-means clustering to non-star-shaped signatures.\n\n"

# Load JSON
with open("data/nonstarShaped_signatures_rotatedSample.json", "r") as f:
    data = json.load(f)

# Prepare data: flatten each matrix to a 1D vector
keys = list(data.keys())
vectors = np.array([np.array(data[k]).flatten() for k in keys], dtype=float)
def repeat(n):
    report = ""
    for i in range(n):
        # Perform 10-means clustering using scipy kmeans (random init, not kmeans++)
        centroids, _ = kmeans(vectors, 10, iter=100)

        # Assign each vector to nearest centroid
        cluster_labels, distances = vq(vectors, centroids)

        # Add cluster keys to report
        for i in range(10):
            report += f"Cluster {i}: {[keys[j] for j in np.where(cluster_labels == i)[0]]}\n"

        # Function to compute Euclidean distance
        def euclidean_distance(a, b):
            return np.linalg.norm(a - b)

        # Function to compute angular distance
        def angular_distance(a, b):
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            return np.arccos(cos_sim)

        # Collect cluster info
        cluster_info = {}
        for i in range(10):
            idx = np.where(cluster_labels == i)[0]
            cluster_keys = [keys[j] for j in idx]
            cluster_vectors = vectors[idx]
            
            # Euclidean distances
            euclid_dists = np.linalg.norm(cluster_vectors - centroids[i], axis=1)
            euclid_max = euclid_dists.max()
            euclid_avg = euclid_dists.mean()
            
            # Angular distances
            ang_dists = np.array([angular_distance(v, centroids[i]) for v in cluster_vectors])
            ang_max = ang_dists.max()
            ang_avg = ang_dists.mean()
            
            cluster_info[i] = {
                "keys": cluster_keys,
                "euclidean_max": float(euclid_max),
                "euclidean_avg": float(euclid_avg),
                "angular_max": float(ang_max),
                "angular_avg": float(ang_avg)
            }

        report += "\n\nCluster Summary:\n"
        for i, info in cluster_info.items():
            report += f"  Cluster {i}: {len(info['keys'])} items\n"
            report += f"  Euclidean max/avg: {info['euclidean_max']:.4f} / {info['euclidean_avg']:.4f}\n"
            report += f"  Angular max/avg: {info['angular_max']:.4f} / {info['angular_avg']:.4f}\n"
        report += "\n" + "="*50 + "\n\n"
    return report
rep += repeat(100)
# Save report
with open("reports/kmeansForNonStarShapedRotation_report1024.txt", "w") as f:
    f.write(rep)
