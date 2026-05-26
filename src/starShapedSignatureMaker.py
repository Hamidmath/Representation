import json
import numpy as np
from shapely import LineString, Polygon
from scipy.fft import fft, ifft

def polygon_radial_function(vertices):
    """Given vertices of a polygon, returns its radial function r(theta)."""
    try:
        poly = Polygon(vertices)
        if not poly.is_valid:
            poly = poly.convex_hull
            if not poly.is_valid:
                print(f"Error: Invalid polygon for vertices, convex hull also invalid.")
                return lambda theta: None
    except Exception as e:
        print(f"Error creating polygon: {e}")
        return lambda theta: None
    def r(theta):
        dx, dy = np.cos(theta), np.sin(theta)
        ray = LineString([(0, 0), (1e6 * dx, 1e6 * dy)])
        inter = poly.boundary.intersection(ray)
        if inter.is_empty: return None
        pts = [inter] if inter.geom_type == "Point" else list(inter.geoms)
        dists = [np.hypot(p.x, p.y) for p in pts if p.x * dx + p.y * dy >= -1e-10]
        return min(dists) if dists else None
    return r

def compute_signature_fft(h_func, n_theta, lambda_val=1.0):
    """
    Computes the S_h signature using the Fast Fourier Transform.
    This replaces the old loop-based compute_S_h.
    """
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    h_vals = np.array([h_func(th) for th in thetas])
    if np.any(h_vals == None): return None
    h_vals = h_vals.astype(float)

    h_vals_reversed = h_vals[::-1]
    p = np.exp(-lambda_val * h_vals_reversed)
    q = np.exp(lambda_val * h_vals)

    P, Q = fft(p), fft(q)
    R = P * Q
    S_unscaled = ifft(R)
    
    dtheta = 2 * np.pi / n_theta
    S = S_unscaled * dtheta
    return np.real(S)

n_theta = 128

input_file = "data/combinedPtsNormalizedByArea.json"
output_file = f"data/starShaped_signatures{n_theta}ByArea.json"




# input_file = "data/rotatedSampleByArea.json"
# output_file = f"data/starShaped_signatures_rotatedSample{n_theta}ByArea.json"


with open(input_file, "r") as f:
    data = json.load(f)
signatures = {}
for key, vertices in data.items():
    radial_func = polygon_radial_function(vertices)
    signature = compute_signature_fft(radial_func, n_theta=n_theta)
    signatures[key] = signature 
with open(output_file, "w") as f:
    json.dump({k: v.tolist() if v is not None else None for k, v in signatures.items()}, f, indent=4)