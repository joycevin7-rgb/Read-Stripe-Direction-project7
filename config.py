"""
Shared configuration for Classical Methods Lab - Group 14: Read Stripe Direction.
All tunable parameters and file paths live here as named constants.
"""

from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Input swatch filenames (order preserved across all tables and figures)
SWATCH_FILES = [
    "swatch_1.png",
    "swatch_2.png",
    "swatch_3.png",
]

# Sobel kernel size (shared between Stage A and Stage B)
KSIZE = 3

# Border width in pixels for Stage B's border-effect comparison
BORDER = 5

# Decision margins tested by Stage C (ratio of stronger gradient to weaker gradient)
MARGINS = [1.0, 2.5]
MAIN_MARGIN = 2.5
COMPARISON_MARGIN = 1.0
