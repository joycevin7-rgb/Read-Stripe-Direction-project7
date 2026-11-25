"""
Stage A: Measure intensity change across columns using Sobel Gx.
OWNER: A
"""

import cv2
import numpy as np


def run_stage_a(images, ksize):
    """
    Compute horizontal gradient (Sobel Gx) for each grayscale swatch.

    Parameters
    ----------
    images : dict[str, np.ndarray]
        Mapping from swatch filename to 2D grayscale image array.
    ksize : int
        Sobel kernel size (must match Stage B).

    Returns
    -------
    handoff_gx : dict[str, float]
        Mean absolute Gx score per swatch (the ONLY value passed to Stage C).
    trial_a_rows : list[dict]
        Comparison rows of mean signed Gx vs mean absolute Gx per swatch.
    abs_gx_maps : dict[str, np.ndarray]
        Absolute Gx gradient magnitude maps (for visualization only).
    """
    handoff_gx = {}
    trial_a_rows = []
    abs_gx_maps = {}

    for filename, gray in images.items():
        # Use CV_64F so negative gradients (bright-to-dark) are preserved, not clipped to 0
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)

        # Signed mean: positive and negative edges can cancel each other out
        mean_signed_gx = float(np.mean(gx))

        # Absolute mean: measures total edge strength across columns regardless of sign
        abs_gx = np.abs(gx)
        mean_abs_gx = float(np.mean(abs_gx))

        # Store only the full-image mean absolute Gx for Stage C
        handoff_gx[filename] = mean_abs_gx

        # Record comparison evidence for Stage A's role check
        trial_a_rows.append(
            {
                "swatch": filename,
                "mean_signed_Gx": mean_signed_gx,
                "mean_abs_Gx": mean_abs_gx,
            }
        )
        abs_gx_maps[filename] = abs_gx

    return handoff_gx, trial_a_rows, abs_gx_maps
