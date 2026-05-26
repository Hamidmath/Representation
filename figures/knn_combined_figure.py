"""
Generate a combined figure:
  - Left  ~1/3 : overall_accuracy_vs_ntheta_with_std.png  (clustering accuracy)
  - Right ~2/3 : 5-NN table  (3 query rows × 6 columns: query + 5 neighbours)

Uses data/euclideanStarSignatures128ByArea.json for distances
(same as DeadlineNight/knn_Euclidean.py which generates knn_EuclideanStarNormalized).
Renders colored polar shapes from vertex data.
"""

import json, os, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ── paths ────────────────────────────────────────────────────────────────
VERTICES_JSON = "data/combinedPtsNormalizedByArea.json"
DISTANCES_JSON = "data/euclideanStarSignatures128ByArea.json"
ACC_IMG = "figures/overall_accuracy_vs_ntheta_with_std.png"
OUT_IMG = "figures/combined_knn_accuracy.png"

QUERY_IDS = [518, 734, 1070]
K = 5
N_SAMPLE = 180

# ── colors ───────────────────────────────────────────────────────────────
QUERY_FILL = "#4A90D9"
QUERY_EDGE = "#1B3A6B"
NEIGH_FILL = "#8BC34A"
NEIGH_EDGE = "#33691E"
GRID_COLOR = "#888888"


# ── helpers ──────────────────────────────────────────────────────────────

def find_knn(dist_data, query_id, k=K):
    neighbours = []
    for i in range(1, 1101):
        if i == query_id:
            continue
        key = f"{query_id}_{i}"
        if key in dist_data and dist_data[key] is not None:
            neighbours.append((i, float(dist_data[key])))
    neighbours.sort(key=lambda x: x[1])
    return neighbours[:k]


def polygon_radial_function(vertices):
    poly = Polygon(vertices)
    if not poly.is_valid:
        poly = poly.buffer(0)

    def r(theta):
        dx, dy = np.cos(theta), np.sin(theta)
        ray = LineString([(0, 0), (1e6 * dx, 1e6 * dy)])
        inter = poly.boundary.intersection(ray)
        if inter.is_empty:
            return np.nan
        pts = [inter] if inter.geom_type == "Point" else list(inter.geoms)
        dists = [np.hypot(p.x, p.y) for p in pts if p.x * dx + p.y * dy >= -1e-10]
        return min(dists) if dists else np.nan

    return r


def render_shape_to_image(vertices_data, shape_id, fill_color, edge_color,
                          is_query=False, size_px=250):
    """Render a polar shape and return as PIL Image."""
    pts = vertices_data[str(shape_id)]
    mx = max(p[0] ** 2 + p[1] ** 2 for p in pts) ** 0.5
    scaled = [(p[0] / mx, p[1] / mx) for p in pts]
    f = polygon_radial_function(scaled)
    thetas = np.linspace(0, 2 * np.pi, N_SAMPLE)
    r_vals = np.array([f(th) for th in thetas], dtype=float)
    rmax = np.nanmax(r_vals)
    if rmax > 0:
        r_vals /= rmax

    fig = plt.figure(figsize=(2.5, 2.5))
    ax = fig.add_subplot(projection="polar")
    ax.fill(thetas, r_vals, color=fill_color, alpha=0.35)
    ax.plot(thetas, r_vals, color=edge_color, linewidth=2.0)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=[''] * 5)
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
    ax.set_xticklabels([''] * 8)
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.set_ylim(0, 1)

    if is_query:
        ax.spines['polar'].set_edgecolor(QUERY_FILL)
        ax.spines['polar'].set_linewidth(3.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.resize((size_px, size_px), Image.LANCZOS)
    return img


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    with open(VERTICES_JSON) as f:
        vertices_data = json.load(f)
    with open(DISTANCES_JSON) as f:
        dist_data = json.load(f)

    # Pre-compute KNN for each query
    knn_info = {}
    for qid in QUERY_IDS:
        knn_info[qid] = find_knn(dist_data, qid, K)
        print(f"  Query {qid}: {[(n, round(d,3)) for n,d in knn_info[qid]]}")

    # ── Build the KNN table with PIL ─────────────────────────────────
    cell_size = 200
    query_border = 6
    neigh_border = 3
    header_h = 36
    pad = 6
    row_gap = 4

    n_rows = len(QUERY_IDS)
    q_cell = cell_size + 2 * query_border
    n_cell = cell_size + 2 * neigh_border

    col_widths = [q_cell] + [n_cell] * K
    total_w = sum(col_widths) + pad * (K + 2)
    row_h = max(q_cell, n_cell)
    total_h = header_h + pad + n_rows * row_h + (n_rows - 1) * row_gap + pad

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    # ── Title row ────────────────────────────────────────────────────
    col_titles = ["Query"] + [f"{i+1}-NN" for i in range(K)]
    hdr_colors = ["#2C5F8A"] + ["#5A8A5E"] * K
    x = pad
    for c in range(1 + K):
        w = col_widths[c]
        draw.rounded_rectangle(
            [x, pad // 2, x + w, pad // 2 + header_h],
            radius=8, fill=hdr_colors[c], outline="white", width=1)
        bbox = font.getbbox(col_titles[c])
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + (w - tw) // 2, pad // 2 + (header_h - th) // 2),
                  col_titles[c], fill="white", font=font)
        x += w + pad

    # ── Data rows ────────────────────────────────────────────────────
    print("Rendering shapes …")
    for row_i, qid in enumerate(QUERY_IDS):
        y = header_h + pad + row_i * (row_h + row_gap)

        # Query
        q_img = render_shape_to_image(
            vertices_data, qid, QUERY_FILL, QUERY_EDGE, is_query=True,
            size_px=cell_size)
        q_bordered = ImageOps.expand(q_img, border=query_border, fill="#4A90D9")
        x = pad
        canvas.paste(q_bordered, (x, y + (row_h - q_cell) // 2))

        # Neighbours (nearest to farthest, matching txt file order)
        neighbours = knn_info[qid]
        x = pad + q_cell + pad
        for col_j, (nid, dist) in enumerate(neighbours):
            n_img = render_shape_to_image(
                vertices_data, nid, NEIGH_FILL, NEIGH_EDGE, is_query=False,
                size_px=cell_size)
            n_bordered = ImageOps.expand(n_img, border=neigh_border, fill="#8BC34A")
            canvas.paste(n_bordered, (x, y + (row_h - n_cell) // 2))
            x += n_cell + pad

        print(f"  Row {row_i+1}/{n_rows} done (query {qid})")

    knn_tmp = OUT_IMG.replace(".png", "_knn_only.png")
    canvas.save(knn_tmp, dpi=(200, 200))
    print(f"  KNN table saved to {knn_tmp}")

    # ── Compose final side-by-side image ─────────────────────────────
    img_acc = Image.open(ACC_IMG)
    img_knn = Image.open(knn_tmp)

    # Use KNN table height as the target — both panels match this height
    final_h = img_knn.height

    # Scale KNN to ~2/3 of total width, keeping its aspect ratio
    knn_target_w = int(img_knn.width)  # keep original width
    img_knn_r = img_knn  # no resize needed

    # Scale accuracy image to match KNN height exactly
    acc_scale = final_h / img_acc.height
    acc_w = int(img_acc.width * acc_scale)
    img_acc_r = img_acc.resize((acc_w, final_h), Image.LANCZOS)

    gap = 30
    margin = 20
    total_w = margin + img_acc_r.width + gap + img_knn_r.width + margin
    total_h = margin + final_h + margin
    combined = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    combined.paste(img_acc_r, (margin, margin))
    combined.paste(img_knn_r, (margin + img_acc_r.width + gap, margin))

    combined.save(OUT_IMG, dpi=(200, 200))
    print(f"Final combined figure saved to: {OUT_IMG}")
    os.remove(knn_tmp)


if __name__ == "__main__":
    main()
