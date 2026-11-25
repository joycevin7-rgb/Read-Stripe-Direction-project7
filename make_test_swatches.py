"""
Generate synthetic stripe swatches for testing when images/ is empty or missing files.
Never overwrites existing real swatch files.
"""

import cv2
import numpy as np

import config


def _make_rotated_stripe_image(
    size=200,
    orientation="vertical",
    period=20.0,
    mean_intensity=128.0,
    amplitude=80.0,
    angle_deg=4.0,
    noise_std=8.0,
    seed=42,
):
    """Create a synthetic grayscale stripe image with small rotation and Gaussian noise."""
    rng = np.random.default_rng(seed)

    # Generate on a larger canvas so rotation + center crop leaves no black corners
    pad = 60
    full_size = size + 2 * pad
    y_coords, x_coords = np.mgrid[0:full_size, 0:full_size]

    if orientation == "vertical":
        coord = x_coords
    else:
        coord = y_coords

    # Sinusoidal stripe pattern
    pattern = mean_intensity + amplitude * np.sin(2.0 * np.pi * coord / period)

    # Rotate around the center of the padded canvas
    center = (full_size / 2.0, full_size / 2.0)
    rot_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        pattern.astype(np.float32),
        rot_matrix,
        (full_size, full_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Center crop to the target size (200x200)
    cropped = rotated[pad : pad + size, pad : pad + size]

    # Add Gaussian noise and clip to valid 8-bit grayscale range [0, 255]
    noise = rng.normal(loc=0.0, scale=noise_std, size=(size, size))
    noisy = np.clip(cropped + noise, 0, 255).astype(np.uint8)
    return noisy


def main():
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    specs = {
        "swatch_1.png": dict(
            orientation="vertical",
            period=20.0,
            mean_intensity=128.0,
            amplitude=85.0,
            angle_deg=4.0,
            noise_std=8.0,
            seed=101,
        ),
        "swatch_2.png": dict(
            orientation="horizontal",
            period=22.0,
            mean_intensity=128.0,
            amplitude=85.0,
            angle_deg=-3.5,
            noise_std=8.0,
            seed=202,
        ),
        "swatch_3.png": dict(
            orientation="vertical",
            period=18.0,
            mean_intensity=128.0,
            amplitude=14.0,  # Low-contrast swatch
            angle_deg=5.0,
            noise_std=6.0,
            seed=303,
        ),
    }

    created = 0
    skipped = 0
    for filename in config.SWATCH_FILES:
        out_path = config.IMAGES_DIR / filename
        if out_path.exists():
            print(f"[SKIP] {out_path} already exists (will not overwrite).")
            skipped += 1
            continue

        img = _make_rotated_stripe_image(size=200, **specs[filename])
        cv2.imwrite(str(out_path), img)
        print(f"[CREATED] {out_path} ({img.shape[1]}x{img.shape[0]} grayscale)")
        created += 1

    print(f"Done: {created} created, {skipped} skipped.")


if __name__ == "__main__":
    main()
