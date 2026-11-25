"""
Stage B: Measure intensity change across rows using Sobel Gy.
OWNER: B
"""

import cv2
import numpy as np


def run_stage_b(images, ksize, border):
    """
    Compute vertical gradient (Sobel Gy) for each grayscale swatch.

    Parameters
    ----------
    images : dict[str, np.ndarray]
        Mapping from swatch filename to 2D grayscale image array.
    ksize : int
        Sobel kernel size (must match Stage A).
    border : int
        Number of border pixels to crop on each side for the border comparison.

    Returns
    -------
    handoff_gy : dict[str, float]
        Mean absolute Gy score for the FULL image (the ONLY value passed to Stage C).
    trial_b_rows : list[dict]
        Comparison rows of full-image vs border-cropped mean absolute Gy.
    abs_gy_maps : dict[str, np.ndarray]
        Absolute Gy gradient magnitude maps (for visualization only).
    """
    handoff_gy = {}
    trial_b_rows = []
    abs_gy_maps = {}

    for filename, gray in images.items():
        # Use CV_64F so negative gradients are preserved, not clipped to 0
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        abs_gy = np.abs(gy)

        # Full-image mean absolute Gy (handed off to Stage C)
        mean_abs_gy_full = float(np.mean(abs_gy))

        # Cropped mean absolute Gy (excluding border pixels on all four sides)
        gy_cropped = abs_gy[border:-border, border:-border]
        mean_abs_gy_no_border = float(np.mean(gy_cropped))

        # Store only the full-image score for Stage C
        handoff_gy[filename] = mean_abs_gy_full

        # Record comparison evidence for Stage B's role check
        trial_b_rows.append(
            {
                "swatch": filename,
                "mean_abs_Gy_full": mean_abs_gy_full,
                "mean_abs_Gy_no_border": mean_abs_gy_no_border,
                "difference_full_minus_cropped": mean_abs_gy_full - mean_abs_gy_no_border,
            }
        )
        abs_gy_maps[filename] = abs_gy

    return handoff_gy, trial_b_rows, abs_gy_maps
