import csv
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "figures/kmeans_results_per_file.csv"
OUT_PNG = "figures/overall_accuracy_vs_ntheta_with_std.png"

def main():
    # 1. Read CSV and aggregate overall accuracies per n_theta
    aggregated = {}  # n_theta → list of accuracies

    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n_theta = int(r["n_theta"])
            acc = float(r["overall_accuracy"])
            if n_theta not in aggregated:
                aggregated[n_theta] = []
            if not np.isnan(acc):
                aggregated[n_theta].append(acc)

    # 2. Compute average and std accuracy per n_theta
    ntheta_list = sorted([n for n in aggregated.keys() if n != 80])
    avg_acc = np.array([
        np.mean(aggregated[n]) if aggregated[n] else np.nan
        for n in ntheta_list
    ])
    std_acc = np.array([
        np.std(aggregated[n]) if aggregated[n] else np.nan
        for n in ntheta_list
    ])

    # 3. Plot
    fig, ax = plt.subplots(figsize=(6, 5))

    # Shaded region for ±1 std deviation
    ax.fill_between(
        ntheta_list,
        np.clip(avg_acc - std_acc, 0, 1),
        np.clip(avg_acc + std_acc, 0, 1),
        color='purple', alpha=0.15, label='± 1 std dev'
    )

    # Mean curve
    ax.plot(ntheta_list, avg_acc, marker='o', linestyle='-', color='purple', label='Mean')

    # log scale for n_theta
    try:
        ax.set_xscale('log', base=2)
    except TypeError:
        ax.set_xscale('log', basex=2)

    xticks = [n for n in ntheta_list if n != 80]
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("m", fontsize=14)
    ax.set_ylabel("Average Accuracy", fontsize=14)
    ax.set_title("Average Accuracy vs m", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, which='both', alpha=0.6, linestyle='--', color='#666666')
    ax.legend(fontsize=12)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    print(f"Saved figure to: {OUT_PNG}")

if __name__ == "__main__":
    main()
