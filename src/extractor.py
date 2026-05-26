# This file extract pixels from gif images and save them as ordered boundary points in .pts files.

from PIL import Image
import numpy as np
import cv2
import os

def gif_to_ordered_boundary_pts(image_path, output_dir):
    # --- Step 1: Load GIF and convert to grayscale ---
    img = Image.open(image_path).convert('L')

    # --- Step 2: Convert to numpy array and threshold to binary ---
    pixel_array = np.array(img)
    binary_matrix = (pixel_array > 128).astype(int)  # 1 = white, 0 = black (boundary)

    # --- Step 3: Convert to OpenCV format (invert: 255 = boundary) ---
    img_cv = (1 - binary_matrix).astype(np.uint8) * 255

    # --- Step 4: Find contours (ordered boundary points) ---
    contours, _ = cv2.findContours(img_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Choose the largest contour (main shape)
    contour = max(contours, key=cv2.contourArea)
    ordered_points = contour[:, 0, :]  # shape (N, 2)

    # --- Step 5: Save to .pts file ---
    output_path = os.path.splitext(os.path.basename(image_path))[0] + '.pts'
    np.savetxt(output_dir + output_path, ordered_points, fmt='%d')

    # --- Optional visualization ---
    # plt.imshow(binary_matrix, cmap='gray')
    # plt.plot(ordered_points[:, 0], ordered_points[:, 1], 'r-', linewidth=1)
    # plt.title('Detected Ordered Boundary')
    # plt.show()

    print(f"Saved ordered boundary points to: {output_path}")
    return ordered_points


# === Example usage ===
if __name__ == "__main__":
    for i in range(1, 1101):
        image_path = f"data/SQUID/kk{i}.gif"  # Replace with your GIF path
        output_dir="data/ExtractedFromGifs/"
        gif_to_ordered_boundary_pts(image_path, output_dir)
