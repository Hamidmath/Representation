import numpy as np
import json
import os
from typing import List, Tuple

EPS = 1e-12

def polygon_signed_area(points: List[List[float]]) -> float:
    """Compute signed area (shoelace). Positive if vertices are CCW."""
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return 0.5 * area

def polygon_centroid(points: List[List[float]]) -> Tuple[float, float]:
    """
    Compute polygon centroid (area-weighted centroid).
    If area is (near) zero, fall back to arithmetic mean of vertices.
    """
    A = polygon_signed_area(points)
    if abs(A) < EPS:
        # Degenerate polygon — use mean of vertices as fallback
        arr = np.array(points, dtype=float)
        return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))

    cx = 0.0
    cy = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    factor = 1.0 / (6.0 * A)
    cx *= factor
    cy *= factor
    return float(cx), float(cy)

def center_points_by_area(points: List[List[float]]) -> List[List[float]]:
    """Translate points so polygon centroid (area centroid) is at origin."""
    if not points:
        return []
    cx, cy = polygon_centroid(points)
    centered = [[p[0] - cx, p[1] - cy] for p in points]
    return centered

def normalizer(points: List[List[float]]) -> List[List[float]]:
    """Scale points so the maximum radial distance from origin is 1."""
    if not points:
        return []
    r = 0.0
    for p in points:
        rr = (p[0]**2 + p[1]**2)**0.5
        if rr > r:
            r = rr
    if r < EPS:
        return points  # all points at/near origin
    return [[p[0] / r, p[1] / r] for p in points]

def generate_points(points: List[List[float]], center_by: str = "area") -> List[List[float]]:
    """
    center_by:
      - "area": center each polygon by its area centroid (default)
      - "mean": center by arithmetic mean of vertices (old behavior)
    """
    if center_by == "area":
        points = center_points_by_area(points)
    elif center_by == "mean":
        arr = np.array(points, dtype=float)
        centroid = np.mean(arr, axis=0) if len(points) else np.array([0.0, 0.0])
        points = [[p[0] - centroid[0], p[1] - centroid[1]] for p in points]
    else:
        raise ValueError("center_by must be 'area' or 'mean'")
    points = normalizer(points)
    return points

def compute_global_area_weighted_centroid(all_polygons: dict) -> Tuple[float, float]:
    """
    Compute a single centroid from multiple polygons, weighted by polygon area (absolute area).
    Returns (cx, cy).
    """
    total_area = 0.0
    weighted_cx = 0.0
    weighted_cy = 0.0
    for pts in all_polygons.values():
        A = polygon_signed_area(pts)
        area = abs(A)
        if area < EPS:
            # For degenerate polygons, use small weight using vertex mean (optional)
            # Here we'll skip degenerate polygons (no contribution).
            continue
        cx, cy = polygon_centroid(pts)
        weighted_cx += cx * area
        weighted_cy += cy * area
        total_area += area
    if total_area < EPS:
        return 0.0, 0.0
    return weighted_cx / total_area, weighted_cy / total_area

if __name__ == "__main__":
    input_json = 'data/combinedPts.json'
    with open(input_json, 'r') as file:
        data = json.load(file)

    # Option A (default): center each polygon by its area centroid individually
    normalized_data = {}
    for key, pts in data.items():
        normalized_data[key] = generate_points(pts, center_by="area")

    # Option B (alternative): center all polygons by the same global area-weighted centroid:
    # Uncomment this block if you prefer a single shared center for all shapes.
    """
    global_cx, global_cy = compute_global_area_weighted_centroid(data)
    normalized_data = {}
    for key, pts in data.items():
        centered = [[p[0] - global_cx, p[1] - global_cy] for p in pts]
        normalized_data[key] = normalizer(centered)
    """

    output_json = 'data/combinedPtsNormalizedByArea.json'
    with open(output_json, 'w') as file:
        json.dump(normalized_data, file, indent=4)

    print(f"Normalized data saved to {output_json}")
