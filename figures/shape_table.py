"""
Draw a 4×6 table of shape outlines from 2D Cartesian points.
No polar, no fill — just clean boundary outlines with table grid lines.
"""

import json
import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

FILE_PATH = "data/combinedPtsNormalizedByArea.json"
OUTPUT_DIR = "figures"

SHAPE_IDS = [
    [7, 17, 20, 38, 42, 46],
    [51, 57, 62, 75, 95, 110],
    [157, 159, 470, 517, 527, 528],
    [534, 539, 577, 617, 643, 672],
]

EDGE_COLOR = "#0D2240"
EDGE_WIDTH = 3.0
CELL_WIDTH = 300
CELL_HEIGHT = 200
LINE_COLOR = (153, 153, 153)
LINE_WIDTH = 2


def render_shape_cartesian(points):
    """Render shape outline from 2D Cartesian vertices, return PIL image."""
    pts = np.array(points, dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]

    # Close the polygon
    xs = np.append(xs, xs[0])
    ys = np.append(ys, ys[0])

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(xs, ys, color=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ax.set_aspect('equal')
    ax.axis('off')

    # Tight limits with small padding
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                pad_inches=0.01, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    with open(FILE_PATH, 'r') as f:
        all_data = json.load(f)

    n_rows = len(SHAPE_IDS)
    n_cols = len(SHAPE_IDS[0])

    total_w = n_cols * CELL_WIDTH + (n_cols + 1) * LINE_WIDTH
    total_h = n_rows * CELL_HEIGHT + (n_rows + 1) * LINE_WIDTH

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Draw grid lines
    for r in range(n_rows + 1):
        y = r * (CELL_HEIGHT + LINE_WIDTH)
        draw.rectangle([0, y, total_w, y + LINE_WIDTH - 1], fill=LINE_COLOR)
    for c in range(n_cols + 1):
        x = c * (CELL_WIDTH + LINE_WIDTH)
        draw.rectangle([x, 0, x + LINE_WIDTH - 1, total_h], fill=LINE_COLOR)

    # Render and paste shapes
    for row_i, row_ids in enumerate(SHAPE_IDS):
        for col_j, keyNum in enumerate(row_ids):
            points = all_data[str(keyNum)]
            shape_img = render_shape_cartesian(points)

            # Fit into cell while preserving aspect ratio
            img_w, img_h = shape_img.size
            scale = min((CELL_WIDTH - 10) / img_w, (CELL_HEIGHT - 10) / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            shape_resized = shape_img.resize((new_w, new_h), Image.LANCZOS)

            cell_x = col_j * (CELL_WIDTH + LINE_WIDTH) + LINE_WIDTH
            cell_y = row_i * (CELL_HEIGHT + LINE_WIDTH) + LINE_WIDTH
            paste_x = cell_x + (CELL_WIDTH - new_w) // 2
            paste_y = cell_y + (CELL_HEIGHT - new_h) // 2
            canvas.paste(shape_resized, (paste_x, paste_y))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_png = os.path.join(OUTPUT_DIR, "shape_table_4x6.png")
    out_pdf = out_png.replace('.png', '.pdf')
    canvas.save(out_png, dpi=(400, 400))
    canvas.save(out_pdf, dpi=(400, 400))
    print(f"Saved → {out_png}")
    print(f"Saved → {out_pdf}")


if __name__ == "__main__":
    main()
