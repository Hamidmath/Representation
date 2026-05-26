import json
import numpy as np
import os
import math
from typing import Dict, List, Tuple

# Tunable: bigger chunk -> less overhead but more memory usage.
# If you have a lot of RAM, set CHUNK_SIZE to a large value (e.g. 5000).
CHUNK_SIZE = 2000
OUT_FILE = "data/euclideanStarSignatures128ByArea.json"
IN_FILE = "data/starShaped_signatures128ByArea.json"


def load_signatures(path: str) -> Tuple[List[str], List]:
    """Load JSON and return parallel lists: keys, signatures (or None if invalid)."""
    with open(path, "r") as f:
        data: Dict[str, List] = json.load(f)

    keys = list(data.keys())
    signatures = []
    # Determine expected dimension from first non-None entry
    expected_dim = None
    for k in keys:
        sig = data[k]
        if sig is None:
            signatures.append(None)
            continue
        if not isinstance(sig, (list, tuple)):
            signatures.append(None)
            continue
        if expected_dim is None and len(sig) > 0:
            expected_dim = len(sig)
        # if shape mismatch, mark None
        if expected_dim is not None and len(sig) != expected_dim:
            signatures.append(None)
        else:
            signatures.append(sig)

    return keys, signatures


def build_array(signatures: List, dtype=np.float32):
    """
    Convert list of signatures to a numpy array.
    Returns:
      - X: numpy array of shape (N_valid, D)
      - idx_map: list mapping valid_index -> original index
      - invalid_mask: boolean list length N where True means invalid
    """
    invalid_mask = [s is None for s in signatures]
    valid_indices = [i for i, v in enumerate(signatures) if v is not None]

    if len(valid_indices) == 0:
        return np.zeros((0, 0), dtype=dtype), [], invalid_mask

    D = len(signatures[valid_indices[0]])
    X = np.empty((len(valid_indices), D), dtype=dtype)
    for out_i, orig_i in enumerate(valid_indices):
        X[out_i, :] = np.asarray(signatures[orig_i], dtype=dtype)
    return X, valid_indices, invalid_mask


def pairwise_distances_blockwise(keys: List[str],
                                 signatures: List,
                                 out_path: str,
                                 chunk_size: int = CHUNK_SIZE):
    """
    Compute NxN pairwise Euclidean distances. Writes a dict { "k1_k2": dist, ... } to out_path.
    Uses block/chunk computation when necessary.
    """
    N = len(keys)
    X, valid_indices, invalid_mask = build_array(signatures, dtype=np.float32)
    N_valid = X.shape[0]

    # Preallocate result dict
    result = {}

    # If there are invalid entries, set distances involving them to None
    if any(invalid_mask):
        for i, k1 in enumerate(keys):
            for j, k2 in enumerate(keys):
                if invalid_mask[i] or invalid_mask[j]:
                    result[f"{k1}_{k2}"] = None

    # If no valid signatures, we're done
    if N_valid == 0:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)
        return

    # Compute squared norms once (float64 for stable accumulation, then keep as float32)
    # Use float64 for norm sums to avoid precision issues with large D.
    sq_norms = np.einsum("ij,ij->i", X.astype(np.float64), X.astype(np.float64)).astype(np.float32)

    # Fast path: if the whole NxN valid-by-valid matrix fits comfortably, compute at once
    # Heuristic memory estimate (bytes) ~ N_valid^2 * 4 (float32)
    est_bytes = N_valid * N_valid * 4
    MEM_LIMIT = 1_000_000_000  # 1GB heuristic threshold; adjust to taste

    if est_bytes <= MEM_LIMIT:
        # compute full Gram matrix
        G = X.dot(X.T)  # shape (N_valid, N_valid), BLAS-backed
        # d2_ij = ||xi||^2 + ||xj||^2 - 2 G_ij
        # use broadcasting
        d2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * G
        # Numerical safety
        np.maximum(d2, 0.0, out=d2)
        Dmat = np.sqrt(d2, dtype=np.float32)

        # Fill results for pairs of valid indices
        for vi, orig_i in enumerate(valid_indices):
            k1 = keys[orig_i]
            row = Dmat[vi]
            for vj, orig_j in enumerate(valid_indices):
                k2 = keys[orig_j]
                result[f"{k1}_{k2}"] = float(row[vj])

    else:
        # Chunked approach: compute block rows
        # We'll compute for i_block rows vs all columns
        n = N_valid
        for i0 in range(0, n, chunk_size):
            i1 = min(i0 + chunk_size, n)
            Xi = X[i0:i1, :]  # shape (bi, D)
            # compute dot Xi @ X.T -> shape (bi, n)
            Gblock = Xi.dot(X.T)
            sq_i = sq_norms[i0:i1, None]  # shape (bi,1)
            d2block = sq_i + sq_norms[None, :] - 2.0 * Gblock
            np.maximum(d2block, 0.0, out=d2block)
            Dblock = np.sqrt(d2block, dtype=np.float32)

            # write block into result
            for local_i, vi in enumerate(range(i0, i1)):
                orig_i = valid_indices[vi]
                k1 = keys[orig_i]
                row = Dblock[local_i]
                for vj, orig_j in enumerate(valid_indices):
                    k2 = keys[orig_j]
                    result[f"{k1}_{k2}"] = float(row[vj])

    # Re-ensure entries involving invalids (in case we computed only for valids earlier)
    if any(invalid_mask):
        for i, k1 in enumerate(keys):
            for j, k2 in enumerate(keys):
                if invalid_mask[i] or invalid_mask[j]:
                    result[f"{k1}_{k2}"] = None

    # Write JSON
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"✅ Done! Results written to {out_path}")


if __name__ == "__main__":
    keys, signatures = load_signatures(IN_FILE)
    pairwise_distances_blockwise(keys, signatures, OUT_FILE, chunk_size=CHUNK_SIZE)
