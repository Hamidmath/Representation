"""
Three-column figure for shape kk102 with n_sample = 36, 120, 360.
Each column: polar plot on top, cartesian on bottom.
Based on paperShapes/shape6.py logic (isolated steps, cartesian fill, no connected steps).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
import os
import json

# ── Style ────────────────────────────────────────────────────────────────
POLAR_FILL = "#4A90D9"
POLAR_EDGE = "#0D2240"
CART_LINE = "#0D2240"
CART_FILL = "#4A90D9"
FILL_ALPHA = 0.25
EDGE_ALPHA = 0.85
GRID_COLOR = "#888888"

FILE_PATH = "data/combinedPtsNormalizedByArea.json"
KEY_NUM = 102
N_SAMPLES = [36, 120, 360]
OUTPUT_DIR = "figures"


def readPointsJson(file_path, keyNum):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data[str(keyNum)]


def polygon_radial_function(vertices):
    try:
        poly = Polygon(vertices)
        if not poly.is_valid:
            return lambda theta: None
    except Exception:
        return lambda theta: None

    def r(theta):
        dx, dy = np.cos(theta), np.sin(theta)
        ray = LineString([(0, 0), (1e6 * dx, 1e6 * dy)])
        inter = poly.boundary.intersection(ray)
        if inter.is_empty:
            return None
        if inter.geom_type == "Point":
            pts = [inter]
        else:
            pts = list(inter.geoms)
        dists = [np.hypot(p.x, p.y) for p in pts if p.x * dx + p.y * dy >= -1e-10]
        return min(dists) if dists else None

    return r


def main():
    points = readPointsJson(FILE_PATH, KEY_NUM)
    _max = max(p[0] ** 2 + p[1] ** 2 for p in points) ** 0.5
    scaled = [(p[0] / _max, p[1] / _max) for p in points]
    f = polygon_radial_function(scaled)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.12, wspace=0.30,
                          left=0.04, right=0.96, top=0.92, bottom=0.06)

    for col, n_sample in enumerate(N_SAMPLES):
        thetas = np.linspace(0, 2 * np.pi, n_sample, endpoint=False)
        r_raw = [f(th) for th in thetas]
        r_vals = np.array([rv if rv is not None else np.nan for rv in r_raw])

        theta_edges = np.concatenate([thetas, [2 * np.pi]])

        # ── Polar plot ───────────────────────────────────────────────
        ax_p = fig.add_subplot(gs[0, col], projection='polar')

        # Isolated step segments (no connecting lines between bars)
        for i in range(len(thetas)):
            if np.isnan(r_vals[i]):
                continue
            th0, th1 = theta_edges[i], theta_edges[i + 1]
            rval = r_vals[i]
            # Step bar outline
            ax_p.plot([th0, th1], [rval, rval],
                      color=POLAR_EDGE, linewidth=1, alpha=EDGE_ALPHA)
            # Filled wedge from origin to bar
            ax_p.fill_between([th0, th1], [rval, rval], 0,
                              facecolor=POLAR_FILL, alpha=FILL_ALPHA,
                              edgecolor=POLAR_EDGE, linewidth=0.5)

        ax_p.set_rlabel_position(90)
        ax_p.set_ylim(0, 1.1)
        ax_p.set_rgrids([0.3, 0.6, 1.0])
        yticks = ax_p.get_yticks()
        labels = [''] * len(yticks)
        try:
            labels[list(yticks).index(1.0)] = '1'
        except ValueError:
            pass
        ax_p.set_yticklabels(labels)
        ax_p.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
        ax_p.set_xticklabels([
            '0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
            r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$'
        ], fontsize=12)
        ax_p.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)

        # ── Cartesian plot ───────────────────────────────────────────
        ax_c = fig.add_subplot(gs[1, col])

        # Isolated step segments + filled bars
        for i in range(len(thetas)):
            if np.isnan(r_vals[i]):
                continue
            th0, th1 = theta_edges[i], theta_edges[i + 1]
            rval = r_vals[i]
            # Step line
            ax_c.plot([th0, th1], [rval, rval],
                      color=CART_LINE, linewidth=1.2)
            # Filled bar
            ax_c.fill_between([th0, th1], 0, rval,
                              facecolor=CART_FILL, alpha=FILL_ALPHA,
                              edgecolor=CART_LINE, linewidth=0.5)

        ax_c.set_xlim(0, 2 * np.pi)
        ax_c.set_ylim(0, 1.1)
        ax_c.set_xlabel(r'$\theta$ (radians)', fontsize=13)
        ax_c.tick_params(labelsize=11)
        ax_c.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_png = os.path.join(OUTPUT_DIR, f"kk{KEY_NUM}_triple_styled.png")
    out_pdf = out_png.replace('.png', '.pdf')
    fig.savefig(out_png, dpi=400, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_png}")
    print(f"Saved → {out_pdf}")


if __name__ == "__main__":
    main()
