"""
Styled version of paperShapes/shape8.py
Generates a 2x2 figure: f(θ) and RoC(f)(θ) = 1 - f(2π - θ)
with improved colors and styling.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
import os
import json

# ── Style config ─────────────────────────────────────────────────────────
ORIG_POLAR_FILL = "#4A90D9"       # blue fill for f(θ)
ORIG_POLAR_EDGE = "#0D2240"       # darker blue edge
ORIG_CART_LINE = "#0D2240"        # darker blue curve
ORIG_CART_FILL = "#4A90D9"

TRANS_POLAR_FILL = "#8BC34A"      # green fill for RoC
TRANS_POLAR_EDGE = "#1A3A0E"      # darker green edge
TRANS_CART_LINE = "#1A3A0E"       # darker green curve
TRANS_CART_FILL = "#8BC34A"

FILL_ALPHA = 0.25
EDGE_ALPHA = 0.85
GRID_COLOR = "#888888"


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


def drawMineStyled(n_sample, file_path, keyNum):
    points_list = readPointsJson(file_path, keyNum)
    if not points_list:
        print("No points loaded. Exiting.")
        return

    points = points_list
    _max = max(p[0] ** 2 + p[1] ** 2 for p in points) ** 0.5
    if _max == 0:
        print("All points at origin. Exiting.")
        return
    scaled_points = [(p[0] / _max, p[1] / _max) for p in points]

    f = polygon_radial_function(scaled_points)

    n_sample = 60
    thetas = np.linspace(0, 2 * np.pi, n_sample, endpoint=False)
    r_raw = [f(th) for th in thetas]
    r_vals = np.array([rv if rv is not None else np.nan for rv in r_raw])

    # RoC transformation: 1 - f(2π - θ)
    theta_mapped = (2 * np.pi - thetas) % (2 * np.pi)
    r_mapped_raw = [f(th) for th in theta_mapped]
    r_mapped = np.array([rv if rv is not None else np.nan for rv in r_mapped_raw])
    r_trans_vals = 1.0 - r_mapped

    # Step edges for stepwise plotting
    theta_edges = np.concatenate([thetas, [2 * np.pi]])
    r_step_edges = np.concatenate([r_vals, [r_vals[0]]])
    r_trans_step_edges = np.concatenate([r_trans_vals, [r_trans_vals[0]]])

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    # ── Shared plotting function ────────────────────────────────────
    def plot_one(ax_polar, ax_cart, r_vals_local, r_step_edges_local,
                 polar_fill, polar_edge, cart_line, cart_fill):

        # === POLAR: isolated steps with colored fill ===
        for i in range(len(thetas)):
            if np.isnan(r_vals_local[i]):
                continue
            th0, th1 = theta_edges[i], theta_edges[i + 1]
            rval = r_vals_local[i]
            ax_polar.plot([th0, th1], [rval, rval],
                          color=polar_edge, linewidth=1, alpha=EDGE_ALPHA)
            ax_polar.fill_between([th0, th1], [rval, rval], 0,
                                  facecolor=polar_fill, alpha=FILL_ALPHA,
                                  edgecolor=polar_edge, linewidth=0.5)

        ax_polar.set_rlabel_position(90)
        ax_polar.set_ylim(0, 1.1)
        ax_polar.set_rgrids([0.3, 0.6, 1.0])
        yticks = ax_polar.get_yticks()
        labels = [''] * len(yticks)
        try:
            labels[list(yticks).index(1.0)] = '1'
        except ValueError:
            pass
        ax_polar.set_yticklabels(labels)
        ax_polar.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
        ax_polar.set_xticklabels([
            '0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
            r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$'
        ], fontsize=13)
        ax_polar.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)

        # === CARTESIAN: steps with fill ===
        for i in range(len(thetas)):
            if np.isnan(r_vals_local[i]):
                continue
            th0, th1 = theta_edges[i], theta_edges[i + 1]
            rval = r_vals_local[i]
            ax_cart.plot([th0, th1], [rval, rval],
                         color=cart_line, linewidth=1.2)
            ax_cart.fill_between([th0, th1], 0, rval,
                                 facecolor=cart_fill, alpha=FILL_ALPHA,
                                 edgecolor=cart_line, linewidth=0.5)

        ax_cart.set_xlim(0, 2 * np.pi)
        ax_cart.set_ylim(0, 1.1)
        ax_cart.set_xlabel(r'$\theta$ (radians)', fontsize=13)
        ax_cart.tick_params(labelsize=11)
        ax_cart.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)

    # ── Version A: original layout [polar, polar; cart, cart] ────────
    fig_a = plt.figure(figsize=(10, 8))
    ax_po_a = fig_a.add_subplot(2, 2, 1, projection='polar')
    ax_co_a = fig_a.add_subplot(2, 2, 3)
    ax_pt_a = fig_a.add_subplot(2, 2, 2, projection='polar')
    ax_ct_a = fig_a.add_subplot(2, 2, 4)

    plot_one(ax_po_a, ax_co_a, r_vals, r_step_edges,
             ORIG_POLAR_FILL, ORIG_POLAR_EDGE, ORIG_CART_LINE, ORIG_CART_FILL)
    plot_one(ax_pt_a, ax_ct_a, r_trans_vals, r_trans_step_edges,
             TRANS_POLAR_FILL, TRANS_POLAR_EDGE, TRANS_CART_LINE, TRANS_CART_FILL)

    ax_po_a.set_title(r"$f(\theta)$", pad=20, fontsize=17, fontweight='bold')
    ax_pt_a.set_title(r"$\mathrm{RoC}(f)(\theta) = 1 - f(2\pi - \theta)$",
                      pad=20, fontsize=17, fontweight='bold')
    fig_a.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=0.08,
                          wspace=0.35, hspace=0.1)

    path_a = os.path.join(output_dir, f"kk{keyNum}_{n_sample}_styled_COMBINED.png")
    fig_a.savefig(path_a, dpi=400, bbox_inches='tight')
    fig_a.savefig(path_a.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig_a)
    print(f"Version A (original layout) → {path_a}")

    # ── Version B: transposed layout [f: polar, cart; RoC: polar, cart] ──
    fig_b = plt.figure(figsize=(8, 10))
    ax_po_b = fig_b.add_subplot(2, 2, 1, projection='polar')
    ax_co_b = fig_b.add_subplot(2, 2, 2)
    ax_pt_b = fig_b.add_subplot(2, 2, 3, projection='polar')
    ax_ct_b = fig_b.add_subplot(2, 2, 4)

    plot_one(ax_po_b, ax_co_b, r_vals, r_step_edges,
             ORIG_POLAR_FILL, ORIG_POLAR_EDGE, ORIG_CART_LINE, ORIG_CART_FILL)
    plot_one(ax_pt_b, ax_ct_b, r_trans_vals, r_trans_step_edges,
             TRANS_POLAR_FILL, TRANS_POLAR_EDGE, TRANS_CART_LINE, TRANS_CART_FILL)

    ax_po_b.set_title(r"$f(\theta)$", pad=20, fontsize=17, fontweight='bold')
    ax_pt_b.set_title(r"$\mathrm{RoC}(f)(\theta) = 1 - f(2\pi - \theta)$",
                      pad=20, fontsize=17, fontweight='bold')
    fig_b.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06,
                          wspace=0.25, hspace=0.15)

    path_b = os.path.join(output_dir, f"kk{keyNum}_{n_sample}_styled_TRANSPOSED.png")
    fig_b.savefig(path_b, dpi=400, bbox_inches='tight')
    fig_b.savefig(path_b.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig_b)
    print(f"Version B (transposed layout) → {path_b}")


# Same parameters as the original: keyNum=53, n_sample=120
drawMineStyled(120, "data/combinedPtsNormalizedByArea.json", 53)
