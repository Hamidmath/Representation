import re
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

def base_key(k):
    """Return base part before underscore (e.g., '20_3' -> '20')."""
    return k.split('_')[0]

def parse_clusters(report_text):
    """Parse clusters from the report text."""
    experiments = []
    blocks = report_text.split("=" * 50)
    for block in blocks:
        clusters = re.findall(r"Cluster\s+\d+:\s+\[([^\]]*)\]", block)
        if not clusters:
            continue
        exp_clusters = []
        for c in clusters:
            keys = re.findall(r"'([^']+)'", c)
            exp_clusters.append(keys)
        experiments.append(exp_clusters)
    return experiments

def evaluate_experiments(experiments):
    """Evaluate perfectness, NMI, and accuracy across experiments."""
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
            most_common = max(set(bases), key=bases.count)
            correct = sum(1 for b in bases if b == most_common)
            total_correct += correct
            total_items += len(cluster)

            # Check if cluster is perfect
            if len(set(bases)) == 1:
                perfect_clusters += 1

            # Add to label arrays for NMI
            all_labels_true += bases
            all_labels_pred += [cluster_id] * len(bases)
            cluster_id += 1

        # Evaluate experiment-level metrics
        if perfect_clusters == len(exp):
            perfect_exps += 1
        else:
            imperfect_exps += 1
            nmi = normalized_mutual_info_score(all_labels_true, all_labels_pred)
            imperfect_nmis.append(nmi)
            imperfect_perfect_cluster_counts.append(perfect_clusters)

    avg_nmi = np.mean(imperfect_nmis) if imperfect_nmis else 1.0
    avg_perfect_clusters = (
        np.mean(imperfect_perfect_cluster_counts)
        if imperfect_perfect_cluster_counts
        else len(experiments[0])
    )
    avg_accuracy = total_correct / total_items if total_items > 0 else 0

    return {
        "total_experiments": total_experiments,
        "perfect_experiments": perfect_exps,
        "imperfect_experiments": imperfect_exps,
        "avg_nmi_imperfect": avg_nmi,
        "avg_perfect_clusters_in_imperfect": avg_perfect_clusters,
        "overall_accuracy": avg_accuracy,
    }
N_theta = [8,16,32,64,128,256,512,1024,2048]
for n_theta in N_theta:
    with open(f"reports/kmeansForStarShapedRotation_report{n_theta}ByArea.txt", "r") as f:
        report_text = f.read()

    experiments = parse_clusters(report_text)
    results = evaluate_experiments(experiments)

    print("\n=== K-Means Report Analysis ===")
    print(f"Total experiments: {results['total_experiments']}")
    print(f"Perfect experiments: {results['perfect_experiments']}")
    print(f"Imperfect experiments: {results['imperfect_experiments']}")
    print(f"Average NMI (imperfect): {results['avg_nmi_imperfect']:.4f}")
    print(f"Avg perfect clusters in imperfect: {results['avg_perfect_clusters_in_imperfect']:.2f}")
    print(f"Overall accuracy: {results['overall_accuracy']*100:.2f}%")
