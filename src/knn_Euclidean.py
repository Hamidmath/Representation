"""
5-nearest-neighbor retrieval using the rotation-invariant star-shape signature.

Reads the precomputed Euclidean distance matrix, queries 10 sample fish,
and dumps the query + 5 closest neighbors as PNG copies under reports/.
"""

import json
import os
import shutil

INPUT_FILE = "data/euclideanStarSignatures128ByArea.json"
SHAPE_IMG_DIR = "data/ExtractedFromGifs"
OUTPUT_DIR = "reports/knn_EuclideanStar"
SET_PRE = ""

SAMPLE_INDICES = [102, 205, 309, 412, 518, 623, 734, 845, 956, 1070]
K = 5
N_SHAPES = 1100


def main():
    with open(INPUT_FILE, "r") as f:
        euclidean_data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx in SAMPLE_INDICES:
        res = []
        for i in range(1, N_SHAPES + 1):
            if i == idx:
                continue
            key = f"{idx}_{i}"
            if euclidean_data.get(key):
                res.append((i, euclidean_data[key]))
        res = sorted(res, key=lambda x: x[1])[:K]

        print(f"Top {K} closest items for kk{idx}:")
        for item in res:
            print(f"  kk{item[0]}  dist={item[1]:.6f}")

        sample_dir = os.path.join(OUTPUT_DIR, str(idx))
        os.makedirs(sample_dir, exist_ok=True)

        shutil.copy(
            os.path.join(SHAPE_IMG_DIR, f"kk{idx}.png"),
            os.path.join(sample_dir, f"{SET_PRE}cl_{idx}.png"),
        )
        with open(os.path.join(sample_dir, "euclidean_distance.txt"), "w") as fout:
            for item in res:
                shutil.copy(
                    os.path.join(SHAPE_IMG_DIR, f"kk{item[0]}.png"),
                    os.path.join(sample_dir, f"{SET_PRE}cl_{idx}_{item[0]}.png"),
                )
                fout.write(f"kk{item[0]}\t{item[1]}\n")
        print(f"Saved query + neighbors to {sample_dir}")


if __name__ == "__main__":
    main()
