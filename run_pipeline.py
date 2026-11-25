"""
End-to-end pipeline runner: Stage A -> Stage B -> Stage C.
Loads grayscale swatches, runs all three stages, and saves all required outputs.
"""

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import config
from stage_a import run_stage_a
from stage_b import run_stage_b
from stage_c import run_stage_c


def load_grayscale_swatches(images_dir, filenames):
    """Load all swatch images in grayscale and assert each one loaded."""
    images = {}
    for name in filenames:
        path = Path(images_dir) / name
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert img is not None, f"Failed to load grayscale image: {path}"
        images[name] = img
    return images


def write_csv(path, fieldnames, rows):
    """Write a list of dictionaries to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for k, v in row.items():
                formatted[k] = f"{v:.4f}" if isinstance(v, float) else v
            writer.writerow(formatted)


def write_results_markdown(path, rows):
    """Write the three-row results table in Markdown format."""
    headers = [
        "swatch",
        "mean_abs_Gx",
        "mean_abs_Gy",
        f"decision (margin {config.MAIN_MARGIN})",
        f"decision (margin {config.COMPARISON_MARGIN})",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = [
            str(row["swatch"]),
            f"{row['mean_abs_Gx']:.4f}",
            f"{row['mean_abs_Gy']:.4f}",
            str(row[f"decision (margin {config.MAIN_MARGIN})"]),
            str(row[f"decision (margin {config.COMPARISON_MARGIN})"]),
        ]
        lines.append("| " + " | ".join(vals) + " |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_labeled_strip(images, results_rows, out_path):
    """
    Save the three swatches side by side at the SAME display scale (vmin=0, vmax=255).
    Do NOT auto-normalize each swatch so the low-contrast one stays visibly low-contrast.
    """
    fig, axes = plt.subplots(1, len(results_rows), figsize=(12, 4.2))
    if len(results_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, results_rows):
        name = row["swatch"]
        decision = row[f"decision (margin {config.MAIN_MARGIN})"]
        comp_decision = row[f"decision (margin {config.COMPARISON_MARGIN})"]
        ax.imshow(images[name], cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"{name}\nDecision: {decision} (m={config.MAIN_MARGIN})\n"
            f"[m={config.COMPARISON_MARGIN}: {comp_decision}]",
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle(
        "Read Stripe Direction - Labeled Swatch Strip (fixed scale 0..255)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_trial_a_plot(trial_a_rows, out_path):
    """Plot Stage A comparison: mean signed Gx vs mean absolute Gx."""
    names = [r["swatch"] for r in trial_a_rows]
    signed_vals = [r["mean_signed_Gx"] for r in trial_a_rows]
    abs_vals = [r["mean_abs_Gx"] for r in trial_a_rows]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars1 = ax.bar(x - width / 2, signed_vals, width, label="Mean Signed Gx", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, abs_vals, width, label="Mean Absolute |Gx|", color="#dd8452")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Gradient Response")
    ax.set_title("Stage A (Owner A): Mean Signed Gx vs Mean Absolute |Gx|")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        offset = 3 if h >= 0 else -12
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_trial_b_plot(trial_b_rows, out_path):
    """Plot Stage B comparison: mean absolute Gy with vs without 5-px border."""
    names = [r["swatch"] for r in trial_b_rows]
    full_vals = [r["mean_abs_Gy_full"] for r in trial_b_rows]
    cropped_vals = [r["mean_abs_Gy_no_border"] for r in trial_b_rows]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars1 = ax.bar(x - width / 2, full_vals, width, label="Full Image |Gy|", color="#55a868")
    bars2 = ax.bar(
        x + width / 2,
        cropped_vals,
        width,
        label=f"Without {config.BORDER}-px Border |Gy|",
        color="#c44e52",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean Absolute Gy")
    ax.set_title(f"Stage B (Owner B): Full Image vs Cropped ({config.BORDER}-px Border) |Gy|")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_trial_c_plot(trial_c_rows, out_path):
    """Plot Stage C comparison: dominant-to-weaker gradient ratio vs margins 1.0 and 2.5."""
    names = [r["swatch"] for r in trial_c_rows]
    ratios = [r["dominant_to_weaker_ratio"] for r in trial_c_rows]
    colors = ["#8172b3" if r["is_uncertain_at_2.5"] else "#4c72b0" for r in trial_c_rows]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    bars = ax.bar(x, ratios, width=0.45, color=colors)

    ax.axhline(
        config.COMPARISON_MARGIN,
        color="#55a868",
        linestyle="--",
        linewidth=1.3,
        label=f"Margin = {config.COMPARISON_MARGIN}",
    )
    ax.axhline(
        config.MAIN_MARGIN,
        color="#c44e52",
        linestyle="-.",
        linewidth=1.5,
        label=f"Margin = {config.MAIN_MARGIN} (main)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Stronger / Weaker Gradient Ratio")
    ax.set_title("Stage C (Owner C): Margin 1.0 vs 2.5 Decisions")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    max_ratio = max(ratios) if ratios else 3.0
    ax.set_ylim(0, max(max_ratio * 1.25, config.MAIN_MARGIN * 1.25))

    for bar, row in zip(bars, trial_c_rows):
        h = bar.get_height()
        label = (
            f"ratio={h:.2f}\n"
            f"m1.0: {row['decision_margin_1.0']}\n"
            f"m2.5: {row['decision_margin_2.5']}"
        )
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_gradient_maps(abs_gx_maps, abs_gy_maps, out_path):
    """Save |Gx| and |Gy| maps for all swatches using one shared display scale."""
    names = list(abs_gx_maps.keys())
    global_max = max(
        max(float(np.max(abs_gx_maps[n])), float(np.max(abs_gy_maps[n])))
        for n in names
    )

    fig, axes = plt.subplots(2, len(names), figsize=(12, 7))
    for col, name in enumerate(names):
        im_x = axes[0, col].imshow(abs_gx_maps[name], cmap="magma", vmin=0, vmax=global_max)
        axes[0, col].set_title(f"{name} |Gx|")
        axes[0, col].axis("off")

        axes[1, col].imshow(abs_gy_maps[name], cmap="magma", vmin=0, vmax=global_max)
        axes[1, col].set_title(f"{name} |Gy|")
        axes[1, col].axis("off")

    fig.colorbar(im_x, ax=axes.ravel().tolist(), shrink=0.8, label="Sobel Magnitude (shared scale)")
    fig.suptitle("Sobel |Gx| and |Gy| Magnitude Maps (Shared Display Scale)", fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load grayscale swatches
    images = load_grayscale_swatches(config.IMAGES_DIR, config.SWATCH_FILES)

    # 2. Stage A (Owner A) -> produces handoff_gx
    handoff_gx, trial_a_rows, abs_gx_maps = run_stage_a(images, ksize=config.KSIZE)

    # 3. Stage B (Owner B) -> produces handoff_gy using the SAME KSIZE
    handoff_gy, trial_b_rows, abs_gy_maps = run_stage_b(
        images, ksize=config.KSIZE, border=config.BORDER
    )

    # 4. Stage C (Owner C) -> receives ONLY handoff_gx and handoff_gy
    results_rows, trial_c_rows, uncertain_cases = run_stage_c(
        handoff_gx,
        handoff_gy,
        margins=config.MARGINS,
        main_margin=config.MAIN_MARGIN,
        comparison_margin=config.COMPARISON_MARGIN,
    )

    # 5. Save handoffs.json
    handoffs_payload = {
        "shared_ksize": config.KSIZE,
        "stage_a_to_c_mean_abs_gx": handoff_gx,
        "stage_b_to_c_mean_abs_gy": handoff_gy,
    }
    with open(config.OUTPUTS_DIR / "handoffs.json", "w", encoding="utf-8") as f:
        json.dump(handoffs_payload, f, indent=2)

    # 6. Save results tables (CSV and Markdown)
    results_fields = [
        "swatch",
        "mean_abs_Gx",
        "mean_abs_Gy",
        f"decision (margin {config.MAIN_MARGIN})",
        f"decision (margin {config.COMPARISON_MARGIN})",
    ]
    write_csv(config.OUTPUTS_DIR / "results_table.csv", results_fields, results_rows)
    write_results_markdown(config.OUTPUTS_DIR / "results_table.md", results_rows)

    # 7. Save labeled strip image
    save_labeled_strip(images, results_rows, config.OUTPUTS_DIR / "labeled_strip.png")

    # 8. Save per-role comparison evidence (CSV + PNG)
    write_csv(
        config.OUTPUTS_DIR / "trial_A_signed_vs_abs.csv",
        ["swatch", "mean_signed_Gx", "mean_abs_Gx"],
        trial_a_rows,
    )
    save_trial_a_plot(trial_a_rows, config.OUTPUTS_DIR / "trial_A_signed_vs_abs.png")

    write_csv(
        config.OUTPUTS_DIR / "trial_B_border.csv",
        ["swatch", "mean_abs_Gy_full", "mean_abs_Gy_no_border", "difference_full_minus_cropped"],
        trial_b_rows,
    )
    save_trial_b_plot(trial_b_rows, config.OUTPUTS_DIR / "trial_B_border.png")

    write_csv(
        config.OUTPUTS_DIR / "trial_C_margins.csv",
        [
            "swatch",
            "mean_abs_Gx",
            "mean_abs_Gy",
            "dominant_to_weaker_ratio",
            "decision_margin_1.0",
            "decision_margin_2.5",
            "is_uncertain_at_2.5",
        ],
        trial_c_rows,
    )
    save_trial_c_plot(trial_c_rows, config.OUTPUTS_DIR / "trial_C_margins.png")

    # 9. Save optional gradient maps
    save_gradient_maps(abs_gx_maps, abs_gy_maps, config.OUTPUTS_DIR / "gradient_maps.png")

    # 10. Print summary table to terminal
    print("\n=== Classical Methods Lab (Group 14: Read Stripe Direction) ===")
    print(f"Shared KSIZE = {config.KSIZE}, Border = {config.BORDER} px, Margins = {config.MARGINS}")
    print("-" * 78)
    print(
        f"{'Swatch':<14} | {'mean |Gx|':>10} | {'mean |Gy|':>10} | "
        f"{'Dec (m=2.5)':<12} | {'Dec (m=1.0)':<12}"
    )
    print("-" * 78)
    for r in results_rows:
        print(
            f"{r['swatch']:<14} | {r['mean_abs_Gx']:>10.4f} | {r['mean_abs_Gy']:>10.4f} | "
            f"{r[f'decision (margin {config.MAIN_MARGIN})']:<12} | "
            f"{r[f'decision (margin {config.COMPARISON_MARGIN})']:<12}"
        )
    print("-" * 78)
    for m in config.MARGINS:
        u = uncertain_cases[m]
        print(f"Uncertain cases at margin {m}: {u if u else 'None'}")
    print(f"\nAll outputs written to: {config.OUTPUTS_DIR}\n")


if __name__ == "__main__":
    main()
