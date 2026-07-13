"""
Robust intensity normalisation for cross-domain cardiac MRI.

Per-volume mean/std z-scoring is dominated by background/FOV and does not
transfer across scanners or acquisition protocols. Percentile clipping to
[p_low, p_high] followed by a scale to a fixed range is far more stable
across domains (this is the nnU-Net choice for the same reason).

Use `robust_normalize_volume` to compute (lo, hi) from a whole volume once,
then `apply_robust` on each 2-D slice so every slice of a patient shares the
same scale. `robust_normalize_slice` is the fallback for the single-slice
inference path where no volume is available.
"""

import numpy as np

# Clip to the central 99% of intensities, then map [lo, hi] -> [0, 1].
P_LOW = 0.5
P_HIGH = 99.5


def robust_stats(volume: np.ndarray, p_low: float = P_LOW, p_high: float = P_HIGH):
    """Return (lo, hi) percentile intensities for a volume (or slice)."""
    lo = float(np.percentile(volume, p_low))
    hi = float(np.percentile(volume, p_high))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return lo, hi


def apply_robust(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip arr to [lo, hi] and scale to [0, 1]."""
    out = np.clip(arr, lo, hi)
    out = (out - lo) / (hi - lo)
    return out.astype(np.float32)


def robust_normalize_slice(slice_2d: np.ndarray,
                           p_low: float = P_LOW,
                           p_high: float = P_HIGH) -> np.ndarray:
    """Percentile-normalise a single 2-D slice to [0, 1]."""
    lo, hi = robust_stats(slice_2d, p_low, p_high)
    return apply_robust(slice_2d, lo, hi)
