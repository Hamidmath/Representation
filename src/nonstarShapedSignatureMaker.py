import numpy as np
import json
from scipy.fft import fft, ifft

def get_radial_matrix(S, n, n_sample):
    """
    Returns a NumPy array where each column represents the indicator function f_i
    for the i-th annulus, evaluated over n_sample angular intervals.
    
    Parameters:
    - S: List of 2D points, e.g., [[x1, y1], [x2, y2], ...].
    - n: Number of concentric circles.
    - n_sample: Number of angular intervals in [0, 2pi).
    
    Returns:
    - A NumPy array of shape (n_sample, n) where entry (j, i) is 1 if there is at least
      one point in S in the i-th annulus and j-th angular interval, 0 otherwise.
    """
    # Validate inputs
    if n <= 0 or n_sample <= 0:
        raise ValueError("n and n_sample must be positive")
    
    # Convert S to NumPy array for vectorized operations
    points = np.array(S, dtype=np.float64)
    
    # Compute radii: sqrt(x^2 + y^2)
    radii = np.sqrt(np.sum(points**2, axis=1))
    
    # Filter out (0,0) points (where radius = 0)
    valid_mask = radii > 0
    points = points[valid_mask]
    radii = radii[valid_mask]
    
    # Compute angles using arctan2, normalized to [0, 2pi)
    angles = np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)
    
    # Initialize output matrix: n_sample rows, n columns
    matrix = np.zeros((n_sample, n), dtype=np.int8)
    
    # Compute angular interval size
    angle_step = 2 * np.pi / n_sample
    
    # Process each annulus
    for i in range(1, n + 1):
        # Define radius boundaries for the i-th annulus
        r_inner = (i - 1) / n if i > 1 else 0
        r_outer = i / n
        
        # Find points in the i-th annulus
        annulus_mask = (r_inner < radii) & (radii <= r_outer)
        annulus_angles = angles[annulus_mask]
        
        # Compute angular interval indices for these points
        if len(annulus_angles) > 0:
            interval_indices = np.floor(annulus_angles / angle_step).astype(int)
            # Ensure indices are within [0, n_sample-1]
            interval_indices = np.clip(interval_indices, 0, n_sample - 1)
            # Set matrix entries to 1 for intervals containing points
            matrix[interval_indices, i - 1] = 1

    return matrix.T



def compute_signature_fft(points_list, n_theta, annulus = 5, lambda_val=1.0):
    """
    Computes the S_h signature using the Fast Fourier Transform.
    This replaces the old loop-based compute_S_h.
    """
    radial_matrix = get_radial_matrix(points_list, annulus, n_theta)
    S_matrix = np.zeros((annulus, n_theta))
    for i in range(annulus):
        h_vals = radial_matrix[i]
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
        S_matrix[i] = np.real(S)
    return S_matrix


input_file = 'data/combinedPtsNormalizedByArea.json'
output_file = 'data/nonstarShaped_signatures128ByArea.json'

# input_file = 'data/rotatedSample.json'
# output_file = 'data/nonstarShaped_signatures_rotatedSample128.json'



with open(input_file, 'r') as f:
    points_list = json.load(f)

dic = {}
for key in points_list:
    points = points_list[key]
    signature = compute_signature_fft(points, n_theta=128, annulus=8, lambda_val=1.0)
    dic[key] = signature.tolist()

with open(output_file, 'w') as f:
    json.dump(dic, f, indent=4)
