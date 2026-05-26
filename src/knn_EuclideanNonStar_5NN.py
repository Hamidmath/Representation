"""
5-nearest-neighbor retrieval using the non-star (annulus) signature.

Reads the precomputed Euclidean distance matrix between annulus-based
indicator signatures, then for a fixed list of query indices copies the
query PNG + its 5 closest neighbors into reports/.
"""

import json
import os
import shutil

INPUT_FILE = "data/euclideanNonStarSignatures128ByArea.json"
SHAPE_IMG_DIR = "data/ExtractedFromGifs"
OUTPUT_DIR = "reports/knn_EuclideanNonStar"
SET_PRE = "Non_"

SAMPLE_INDICES = [102, 205, 309, 412, 518, 623, 734, 845, 956, 1070]
K = 5
N_SHAPES = 1100


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run euclideanOfNonStarSignatures.py first.")
        return

    with open(INPUT_FILE, "r") as f:
        euclidean_data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx in SAMPLE_INDICES:
        res = []
        for i in range(1, N_SHAPES + 1):
            if i == idx:
                continue
            dist = euclidean_data.get(f"{idx}_{i}")
            if dist is None:
                dist = euclidean_data.get(f"{i}_{idx}")
            if dist is not None:
                res.append((i, dist))
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
