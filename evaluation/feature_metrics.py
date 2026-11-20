"""RBF maximum mean discrepancy on L2-normalized feature vectors."""

import numpy as np


def l2_normalize(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not features.shape[0] or not np.isfinite(features).all():
        raise ValueError("Features must be a nonempty finite [N,C] array")
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def _kernel_mean(left, right, sigma, block_size):
    total = 0.0
    right_norm = (right * right).sum(1)[None, :]
    for start in range(0, len(left), block_size):
        block = left[start:start + block_size]
        distances = ((block * block).sum(1)[:, None] + right_norm - 2 * block @ right.T).clip(0)
        total += np.exp(-distances / (2 * sigma * sigma)).sum()
    return total / (len(left) * len(right))


def rbf_mmd_squared(source, target, sigma=1.0, block_size=256):
    """Biased empirical MMD^2, including diagonal terms, as in Table 10."""
    if sigma <= 0 or block_size < 1:
        raise ValueError("RBF sigma and block size must be positive")
    source, target = l2_normalize(source), l2_normalize(target)
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target feature dimensions differ")
    squared = (_kernel_mean(source, source, sigma, block_size)
               + _kernel_mean(target, target, sigma, block_size)
               - 2 * _kernel_mean(source, target, sigma, block_size))
    return max(float(squared), 0.0)


def rbf_mmd(source, target, sigma=1.0, block_size=256):
    """Return the RBF MMD distance for callers that need the unsquared value."""
    return float(np.sqrt(rbf_mmd_squared(source, target, sigma, block_size)))
