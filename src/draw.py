# This code reads 2D points from a .pts file, plots them, and saves the plot as a PNG image.

import numpy as np
import matplotlib.pyplot as plt
import os

def draw_and_save_pts(file_path, output_path):
    # --- Step 1: Load points from .pts file ---
    points = np.loadtxt(file_path, dtype=int)

    # --- Step 2: Separate x and y coordinates ---
    x = points[:, 0]
    y = points[:, 1]

    # --- Step 3: Plot the shape ---
    plt.figure(figsize=(6, 3))
    plt.plot(x, y, 'k-', linewidth=1)
    plt.scatter(x, y, s=5, color='k')  # optional: show points
    plt.gca().invert_yaxis()  # match image coordinate system
    plt.axis('equal')
    plt.axis('off')  # remove axis lines and ticks
    plt.tight_layout()


    # --- Step 5: Save as PNG ---
    output_file = os.path.join(output_path, f"{os.path.splitext(os.path.basename(file_path))[0]}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved image: {output_file}")

# === Example usage ===
if __name__ == "__main__":
    dir="data/ExtractedFromGifs/"
    for i in range(1, 1101):
        draw_and_save_pts(f"{dir}kk{i}.pts", dir)


