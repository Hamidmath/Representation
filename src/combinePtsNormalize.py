import numpy as np
import json
import os

def center_points(points):
    """Translates points so their centroid is at the origin (0,0)."""
    if not points:
        return []
    points_array = np.array(points)
    centroid = np.mean(points_array, axis=0)
    centered_points = points_array - centroid
    return centered_points.tolist()

def normalizer(points):
    r = 0
    for p in points:
        rr = np.sqrt(p[0]**2 + p[1]**2)
        if rr > r:
            r = rr
    for p in points:
        p[0] = p[0] / r
        p[1] = p[1] / r
    return points

def generate_points(points):
    points = center_points(points)
    points = normalizer(points)
    return points

input_json = 'data/combinedPts.json'
with open(input_json, 'r') as file:
    data = json.load(file)

# Process each shape (each key)
normalized_data = {}
for key, pts in data.items():
    normalized_data[key] = generate_points(pts)

# Save result
output_json = 'data/combinedPtsNormalized.json'
with open(output_json, 'w') as file:
    json.dump(normalized_data, file, indent=4)

print(f"Normalized data saved to {output_json}")