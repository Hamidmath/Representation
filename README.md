# Rotation-Invariant Vectorized Shape Representations &mdash; SQUID Experiments (Star-Shaped Branch)

Code accompanying the paper
**_Rotation-Invariant Vectorized Shape Representations_**
by Hamid Shafieasl and Jeff M. Phillips (University of Utah).

This branch reproduces the **star-shaped** pipeline that produced every figure and
number in the paper:

1. Convert each fish outline into a star-shaped function on the discrete circle
   `S^m` (one radius per angular wedge).
2. Build the rotation-invariant signature `V_f in R^m` (computed in
   `O(m log m)` via the FFT convolution form).
3. Use the Euclidean distance between signatures for k-means clustering and
   k-nearest-neighbor retrieval.

A separate `non-star` branch contains the annulus extension.

---

## Repository layout

```
.
|-- data/
|   |-- SQUID/               # raw upstream SQUID dataset (1100 fish: .gif + .pts)
|   |-- ExtractedFromGifs/   # boundary points + visualisations recovered from the .gif files
|   |                          (.pts and .png per shape; the whole pipeline reads these)
|   `-- experiment_ids.json  # exact shape IDs used for the rotation, kNN and figure experiments
|-- src/                     # pipeline scripts
|-- figures/                 # scripts that reproduce the paper figures
|-- docs/                    # static images shown in this README
`-- README.md
```

After running the pipeline you will also see:

- `data/combinedPts.json`, `data/combinedPtsNormalizedByArea.json` &mdash; cached vertex data;
- `data/starShaped_signatures*ByArea.json`, `data/euclideanStarSignatures128ByArea.json` &mdash; cached signatures and pairwise distances;
- `data/rotatedSampleByArea*.json` &mdash; the random-rotation collections used for clustering;
- `figures/*.png`, `figures/*.pdf` &mdash; figures from the paper;
- `reports/knn_EuclideanStar/<query_id>/` &mdash; per-query 5-NN PNGs.

These outputs are git-ignored; everything in the repo is enough to regenerate them.

---

## Dataset

`data/SQUID/` is the upstream SQUID dataset (1100 fish outlines) of
Mokhtarian, Abbasi and Kittler. Each shape comes as a binary GIF plus a `.pts`
file with boundary samples.

`data/ExtractedFromGifs/` contains the same shapes after we re-extract their
boundaries from the GIFs (so that the point ordering is consistent across the
dataset) along with a PNG preview per shape; the whole pipeline reads from this
folder.

A sample preprocessed shape:

<p align="center">
  <img src="data/ExtractedFromGifs/kk102.png" alt="Example fish (kk102)" width="320">
</p>

A 4x6 grid of fish from the dataset (Figure 3 of the paper):

<p align="center">
  <img src="docs/sample_fish_grid.png" alt="Sample fish grid" width="640">
</p>

---

## Pipeline

The numbered scripts are the ones you actually run; the others are helpers
imported / invoked by them.

| # | Script                                        | What it does                                                                                                                              |
|---|-----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | `src/extractor.py`                            | Reads the raw `.gif` files, traces boundaries with OpenCV, and writes ordered `.pts` files into `data/ExtractedFromGifs/`.                |
| 2 | `src/combinePts.py`                           | Combines the per-shape `.pts` files into a single `data/combinedPts.json`.                                                                |
| 3 | `src/combinePtsNormalizeByArea.py`            | Translates each polygon so its **area-weighted centroid** is at the origin, then rescales so the maximum radius is 1. Writes `data/combinedPtsNormalizedByArea.json`. |
| 4 | `src/draw.py`                                 | (Optional) Renders a PNG preview of every `.pts` file in `data/ExtractedFromGifs/`.                                                       |
| 5 | `src/starShapedSignatureMaker.py`             | Approximates each polygon by a star-shaped function `f: S^m -> R+` and computes its FFT-based signature `V_f`. Writes `data/starShaped_signatures128ByArea.json`. |
| 6 | `src/sampleRotatedVersion.py`                 | Picks 10 random shape IDs, generates 9 random-rotation copies of each (100 shapes total), and writes one `data/rotatedSampleByArea*.json` file. Run multiple times to get multiple trials. |
| 7 | `src/euclideanOfStarSignatures.py`            | Computes the full 1100 x 1100 pairwise Euclidean distance matrix between signatures. Writes `data/euclideanStarSignatures128ByArea.json`. |
| 8 | `src/kmeansForStarShaped.py`                  | Simple stand-alone k-means clustering report on one rotated-sample file. Useful as a smoke test.                                          |
| 9 | `src/sign_kmeans_report.py`                   | The actual clustering experiment used in the paper &mdash; sweeps `m in {8,16,32,48,64,80,96,128,256,512,1024}` across every `data/rotatedSampleByArea*.json` and appends to `figures/kmeans_results_per_file.csv`. |
| 10 | `src/sign_kmeans_reportFromCSV_with_std.py`  | Reads the CSV produced by step 9 and renders Figure 4 left panel (`figures/overall_accuracy_vs_ntheta_with_std.png`).                     |
| 11 | `src/knn_Euclidean.py`                       | 5-NN retrieval for the 10 fixed query IDs (see `data/experiment_ids.json`). Writes `reports/knn_EuclideanStar/<idx>/`.                    |
| 12 | `src/resultOfReports.py`                     | Convenience report summariser used to inspect intermediate JSON outputs.                                                                  |

### Paper figures

| Script                              | Output                                                | Paper figure                                  |
|-------------------------------------|-------------------------------------------------------|-----------------------------------------------|
| `figures/shape_table.py`            | `figures/shape_table_4x6.{png,pdf}`                   | Figure 3 (sample fish grid)                   |
| `figures/shape8_triple.py`          | `figures/kk102_triple_styled.{png,pdf}`               | Figure 1 (effect of `m = 36, 120, 360`)       |
| `figures/shape8_styled.py`          | `figures/kk53_60_styled_COMBINED.{png,pdf}`           | Figure 2 (Reverse-of-Complement illustration) |
| `figures/knn_combined_figure.py`    | `figures/combined_knn_accuracy.png`                   | Figure 4 (clustering accuracy + 5-NN gallery) |

---

## Reproducing the paper experiments

Tested with Python 3.13. Required packages:
`numpy`, `scipy`, `opencv-python`, `Pillow`, `shapely`, `matplotlib`, `scikit-learn`.

```bash
# 0. Optional but recommended
python3 -m venv venv && source venv/bin/activate
pip install numpy scipy opencv-python Pillow shapely matplotlib scikit-learn
```

### One-shot preprocessing

If `data/ExtractedFromGifs/` is already populated (it ships with the repo) you
can skip step 1.

```bash
python3 src/extractor.py                  # GIF -> .pts (only needed once)
python3 src/combinePts.py                 # data/combinedPts.json
python3 src/combinePtsNormalizeByArea.py  # data/combinedPtsNormalizedByArea.json
python3 src/starShapedSignatureMaker.py   # data/starShaped_signatures128ByArea.json
python3 src/euclideanOfStarSignatures.py  # data/euclideanStarSignatures128ByArea.json
```

### Clustering experiment (Figure 4 left panel)

```bash
# 6 independent trials (each samples 10 shapes + 9 random rotations each)
for i in 0 1 2 3 4 5; do python3 src/sampleRotatedVersion.py; done

python3 src/sign_kmeans_report.py                       # builds figures/kmeans_results_per_file.csv
python3 src/sign_kmeans_reportFromCSV_with_std.py       # builds figures/overall_accuracy_vs_ntheta_with_std.png
```

`sampleRotatedVersion.py` writes to `data/rotatedSampleByArea<N>.json`; the
exact shape IDs we used are recorded in `data/experiment_ids.json` if you want
to reproduce them byte-for-byte.

### 5-NN search (Figure 4 right panel)

```bash
python3 src/knn_Euclidean.py
python3 figures/knn_combined_figure.py
```

This will write `figures/combined_knn_accuracy.png` &mdash; the combined panel from
the paper.

<p align="center">
  <img src="docs/accuracy_and_knn.png" alt="Clustering accuracy and 5-NN" width="800">
</p>

### Figures 1 and 2 (illustrations)

```bash
python3 figures/shape8_triple.py   # Figure 1, kk102 at m = 36, 120, 360
python3 figures/shape8_styled.py   # Figure 2, RoC illustration on kk53
python3 figures/shape_table.py     # Figure 3, 4x6 fish grid
```

<p align="center">
  <img src="docs/resolution_demo.png" alt="kk102 at three resolutions" width="800">
</p>

<p align="center">
  <img src="docs/roc_demo.png" alt="RoC illustration on kk53" width="640">
</p>

---

## Signature formula

For `f : Z_m -> R` (radial samples of a standardized star shape), the
rotation-invariant signature is

```
V_f[k] = (1/m) * sum_{j=0..m-1} exp(-(f[j] - f[(j+k) mod m]))
```

which is a circular convolution of two pointwise exponentials and is therefore
computed in `O(m log m)` via the FFT (see `src/starShapedSignatureMaker.py`).
The Euclidean distance `||V_f - V_g||` is invariant to rotations of `f` and
the Reverse-of-Complement operation `f -> M - f(2*pi - .)`.

---

## Citation

If you build on this code, please cite the paper:

```
@unpublished{shafieasl2025rotation,
  author  = {Hamid Shafieasl and Jeff M. Phillips},
  title   = {Rotation-Invariant Vectorized Shape Representations},
  year    = {2025},
  note    = {Preprint}
}
```

The SQUID dataset is from Mokhtarian, Abbasi and Kittler (Curvature Scale
Space) and Nasreddine et al.; please cite the original sources as well.
