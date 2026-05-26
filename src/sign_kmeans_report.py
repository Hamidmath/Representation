#!/usr/bin/env python3
"""
Process multiple rotatedSampleByArea*.json files, run signature computation + kmeans repeats,
append per-file results to a CSV (flushed after each file), free memory, then average across files
and plot summary graphs.

Output:
  - results CSV: "figures/kmeans_results_per_file.csv" (appended rows)
  - final PNG (averaged across files): "figures/kmeans_summary_avg.png"
"""

import os
import glob
import json
import csv
import gc
import numpy as np
from shapely import LineString, Polygon
from scipy.fft import fft, ifft
from scipy.cluster.vq import kmeans, vq
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt

# -------------------------
# Parameters (tweakable)
# -------------------------
INPUT_PATTERN = "data/rotatedSampleByArea*.json"   # matches base and numbered files
RESULTS_CSV = "figures/kmeans_results_per_file.csv"
OUT_PNG = "figures/kmeans_summary_avg.png"

N_THETA = [8,16,32,48,64,80,96,128,256,512,1024]
K = 10
REPEATS = 100     # keep as your original; lower if you want speed
LAMBDA_VAL = 1.0  # as before in compute_signature_fft

# make sure output folder exists
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

# -------------------------
# Helper functions (adapted from your code)
# -------------------------
def polygon_radial_function(vertices):
    """Given vertices of a polygon, returns its radial function r(theta)."""
    try:
        poly = Polygon(vertices)
        if not poly.is_valid:
            poly = poly.convex_hull
            if not poly.is_valid:
                print(f"Warning: Invalid polygon for vertices; convex hull invalid.")
                return lambda theta: None
    except Exception as e:
        print(f"Warning: Error creating polygon: {e}")
        return lambda theta: None

    def r(theta):
        dx, dy = np.cos(theta), np.sin(theta)
        ray = LineString([(0, 0), (1e6 * dx, 1e6 * dy)])
        inter = poly.boundary.intersection(ray)
        if inter.is_empty:
            return None
        pts = [inter] if inter.geom_type == "Point" else list(inter.geoms)
        dists = [np.hypot(p.x, p.y) for p in pts if p.x * dx + p.y * dy >= -1e-10]
        return min(dists) if dists else None
    return r

def compute_signature_fft(h_func, n_theta, lambda_val=LAMBDA_VAL):
    """
    Computes the S_h signature using FFT; returns real array length n_theta or None if missing samples.
    """
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    h_vals = []
    for th in thetas:
        v = h_func(th)
        h_vals.append(v)
    if any([v is None for v in h_vals]):
        return None
    h_vals = np.array(h_vals, dtype=float)
    h_vals_reversed = h_vals[::-1]
    p = np.exp(-lambda_val * h_vals_reversed)
    q = np.exp(lambda_val * h_vals)
    P, Q = fft(p), fft(q)
    R = P * Q
    S_unscaled = ifft(R)
    dtheta = 2 * np.pi / n_theta
    S = S_unscaled * dtheta
    return np.real(S)

def base_key(k):
    return k.split('_')[0] if isinstance(k, str) else str(k)

def evaluate_experiments(experiments):
    """
    Same evaluation logic as before.
    experiments: list of experiment, each experiment is list of K clusters (list of keys)
    Returns dict of metrics.
    """
    total_experiments = len(experiments)
    perfect_exps = 0
    imperfect_exps = 0
    imperfect_nmis = []
    imperfect_perfect_cluster_counts = []
    total_correct = 0
    total_items = 0

    for exp in experiments:
        all_labels_true = []
        all_labels_pred = []
        perfect_clusters = 0
        cluster_id = 0

        for cluster in exp:
            bases = [base_key(k) for k in cluster]
            if len(bases) == 0:
                cluster_id += 1
                continue
            most_common = max(set(bases), key=bases.count)
            correct = sum(1 for b in bases if b == most_common)
            total_correct += correct
            total_items += len(cluster)
            if len(set(bases)) == 1:
                perfect_clusters += 1
            all_labels_true += bases
            all_labels_pred += [cluster_id] * len(bases)
            cluster_id += 1

        if len(exp) > 0 and perfect_clusters == len(exp):
            perfect_exps += 1
        else:
            imperfect_exps += 1
            if len(all_labels_true) > 0:
                nmi = normalized_mutual_info_score(all_labels_true, all_labels_pred)
                imperfect_nmis.append(nmi)
                imperfect_perfect_cluster_counts.append(perfect_clusters)
            else:
                imperfect_nmis.append(0.0)
                imperfect_perfect_cluster_counts.append(0)

    avg_nmi = np.mean(imperfect_nmis) if imperfect_nmis else 1.0
    avg_perfect_clusters = (
        np.mean(imperfect_perfect_cluster_counts)
        if imperfect_perfect_cluster_counts
        else (len(experiments[0]) if experiments else 0)
    )
    avg_accuracy = total_correct / total_items if total_items > 0 else 0.0

    return {
        "total_experiments": total_experiments,
        "perfect_experiments": perfect_exps,
        "imperfect_experiments": imperfect_exps,
        "avg_nmi_imperfect": float(avg_nmi),
        "avg_perfect_clusters_in_imperfect": float(avg_perfect_clusters),
        "overall_accuracy": float(avg_accuracy),
    }

# -------------------------
# CSV helpers
# -------------------------
CSV_HEADER = ["input_file", "n_theta",
              "avg_nmi_imperfect", "avg_perfect_clusters_in_imperfect", "overall_accuracy",
              "valid_vectors", "ignored_vectors"]

def append_rows_to_csv(rows, csv_path=RESULTS_CSV):
    """Append list of dict rows to CSV; create header if not exists. Flush & fsync."""
    newfile = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if newfile:
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())
        for r in rows:
            writer.writerow(r)
        f.flush()
        os.fsync(f.fileno())

# -------------------------
# Main processing loop
# -------------------------
def process_one_file(input_path):
    """Process a single input JSON file, return list of per-n_theta result dicts."""
    print(f"\n=== Processing file: {input_path} ===")
    with open(input_path, "r") as f:
        raw_shapes = json.load(f)

    # build radial functions (keep in memory only while needed)
    radial_funcs = {}
    for key, vertices in raw_shapes.items():
        radial_funcs[key] = polygon_radial_function(vertices)

    results_for_file = []

    # For each n_theta, compute signatures in-memory and perform REPEATS kmeans
    for n_theta in N_THETA:
        print(f"  n_theta = {n_theta} ... computing signatures")
        signatures = {}
        valid_keys = []
        valid_vectors = []
        ignored_keys = []

        for key, h_func in radial_funcs.items():
            S = compute_signature_fft(h_func, n_theta=n_theta)
            signatures[key] = S
            if S is None or (not isinstance(S, (list, np.ndarray))) or len(S) == 0:
                ignored_keys.append(key)
            else:
                valid_keys.append(key)
                valid_vectors.append(np.array(S, dtype=float))

        print(f"    valid: {len(valid_keys)}, ignored: {len(ignored_keys)}")

        if len(valid_keys) == 0:
            # produce a row with zeros / NaNs
            row = {
                "input_file": os.path.basename(input_path),
                "n_theta": n_theta,
                "avg_nmi_imperfect": float("nan"),
                "avg_perfect_clusters_in_imperfect": float("nan"),
                "overall_accuracy": float("nan"),
                "valid_vectors": 0,
                "ignored_vectors": len(ignored_keys),
            }
            results_for_file.append(row)
            # free signature container for this n_theta
            signatures.clear()
            continue

        vectors = np.array(valid_vectors, dtype=float)

        # Run REPEATS times, collect clusters list-of-lists per repeat
        experiments = []
        for rep_i in range(REPEATS):
            k_for_run = min(K, vectors.shape[0])
            centroids, _ = kmeans(vectors, k_for_run, iter=100)
            cluster_labels, _ = vq(vectors, centroids)
            clusters = []
            for cid in range(K):
                idx = np.where(cluster_labels == cid)[0]
                cluster_keys = [valid_keys[j] for j in idx.tolist()]
                clusters.append(cluster_keys)
            experiments.append(clusters)
            # optional progress print
            if (rep_i + 1) % 25 == 0:
                print(f"      repeats done: {rep_i+1}/{REPEATS}")

        # evaluate
        metrics = evaluate_experiments(experiments)
        row = {
            "input_file": os.path.basename(input_path),
            "n_theta": n_theta,
            "avg_nmi_imperfect": metrics["avg_nmi_imperfect"],
            "avg_perfect_clusters_in_imperfect": metrics["avg_perfect_clusters_in_imperfect"],
            "overall_accuracy": metrics["overall_accuracy"],
            "valid_vectors": len(valid_keys),
            "ignored_vectors": len(ignored_keys),
        }
        results_for_file.append(row)

        # free large objects for this n_theta
        signatures.clear()
        del vectors
        del valid_vectors
        gc.collect()

    # free radial funcs and raw shapes
    radial_funcs.clear()
    del radial_funcs
    del raw_shapes
    gc.collect()

    return results_for_file

def main():
    # find all input files matching pattern
    input_files = sorted(glob.glob(INPUT_PATTERN))
    if not input_files:
        raise FileNotFoundError(f"No input files found with pattern: {INPUT_PATTERN}")
    print("Found input files:")
    for p in input_files:
        print("  ", p)

    # If CSV exists already, we will append to it. (User asked to append.)
    # Process each file one by one, append results and flush, then free memory.
    for input_path in input_files:
        rows = process_one_file(input_path)
        append_rows_to_csv(rows, RESULTS_CSV)
        print(f"  Appended {len(rows)} rows for {os.path.basename(input_path)} to {RESULTS_CSV}")
        # ensure memory freed
        del rows
        gc.collect()

    # After all files processed, compute averages across files for each n_theta
    # Read CSV and aggregate
    aggregated = {}  # n_theta -> list of metrics per file
    with open(RESULTS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n_theta = int(r["n_theta"])
            if n_theta not in aggregated:
                aggregated[n_theta] = {
                    "n_files": 0,
                    "nmi_list": [],
                    "perfect_list": [],
                    "accuracy_list": []
                }
            aggregated[n_theta]["n_files"] += 1
            # parse floats possibly nan
            def _safe_float(x):
                try:
                    return float(x)
                except:
                    return float("nan")
            aggregated[n_theta]["nmi_list"].append(_safe_float(r["avg_nmi_imperfect"]))
            aggregated[n_theta]["perfect_list"].append(_safe_float(r["avg_perfect_clusters_in_imperfect"]))
            aggregated[n_theta]["accuracy_list"].append(_safe_float(r["overall_accuracy"]))

    # compute averages (ignore NaNs using numpy nanmean)
    avg_nmi_list = []
    avg_perfect_clusters_list = []
    ntheta_list = []
    for n_theta in sorted(aggregated.keys()):
        ntheta_list.append(n_theta)
        nmi_arr = np.array(aggregated[n_theta]["nmi_list"], dtype=float)
        perf_arr = np.array(aggregated[n_theta]["perfect_list"], dtype=float)
        # use nanmean so files that didn't have valid vectors won't skew
        avg_nmi = float(np.nanmean(nmi_arr)) if nmi_arr.size > 0 else float("nan")
        avg_perf = float(np.nanmean(perf_arr)) if perf_arr.size > 0 else float("nan")
        avg_nmi_list.append(avg_nmi)
        avg_perfect_clusters_list.append(avg_perf)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax0, ax1 = axes

    ax0.plot(ntheta_list, avg_nmi_list, marker='o', linestyle='-')
    try:
        ax0.set_xscale('log', base=2)
    except TypeError:
        # older matplotlib compatibility fallback
        ax0.set_xscale('log', basex=2)
    ax0.set_xticks(ntheta_list)
    ax0.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax0.set_xlabel("m")
    ax0.set_ylabel("Average NMI")
    ax0.set_title("Avg NMI vs m")
    ax0.grid(True, which='both', linestyle='--', alpha=0.4)

    ax1.plot(ntheta_list, avg_perfect_clusters_list, marker='o', color='tab:orange')
    try:
        ax1.set_xscale('log', base=2)
    except TypeError:
        ax1.set_xscale('log', basex=2)
    ax1.set_xticks(ntheta_list)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("m")
    ax1.set_ylabel("Avg perfect clusters in imperfect")
    ax1.set_title("Avg perfect clusters in imperfect vs m")
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    print(f"\nSaved averaged summary plot to {OUT_PNG}")

if __name__ == "__main__":
    main()
