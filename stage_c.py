"""
Stage C: Decide stripe direction from Stage A and Stage B handoff scores.
OWNER: C
"""


def classify_direction(gx_score, gy_score, margin):
    """
    Classify a single swatch as 'vertical', 'horizontal', or 'uncertain'.

    Vertical stripes change intensity across columns -> strong Gx.
    Horizontal stripes change intensity across rows -> strong Gy.
    """
    if gx_score >= margin * gy_score:
        return "vertical"
    if gy_score >= margin * gx_score:
        return "horizontal"
    return "uncertain"


def run_stage_c(handoff_gx, handoff_gy, margins, main_margin=2.5, comparison_margin=1.0):
    """
    Decide stripe direction using ONLY the handoff scores from Stage A and Stage B.

    Parameters
    ----------
    handoff_gx : dict[str, float]
        Mean absolute Gx score per swatch from Stage A.
    handoff_gy : dict[str, float]
        Mean absolute Gy score per swatch from Stage B.
    margins : list[float]
        List of margin thresholds to evaluate (e.g., [1.0, 2.5]).
    main_margin : float
        Primary margin used for the final reported decision (default 2.5).
    comparison_margin : float
        Secondary margin used for comparison (default 1.0).

    Returns
    -------
    results_rows : list[dict]
        Main results table rows (one per swatch).
    trial_c_rows : list[dict]
        Margin comparison rows including uncertain flags.
    uncertain_cases : dict[float, list[str]]
        Swatches marked 'uncertain' for each tested margin.
    """
    results_rows = []
    trial_c_rows = []
    uncertain_cases = {m: [] for m in margins}

    for filename in handoff_gx:
        gx_score = handoff_gx[filename]
        gy_score = handoff_gy[filename]

        decisions_by_margin = {}
        for m in margins:
            decision = classify_direction(gx_score, gy_score, m)
            decisions_by_margin[m] = decision
            if decision == "uncertain":
                uncertain_cases[m].append(filename)

        # Ratio of stronger gradient to weaker gradient (helps explain margin sensitivity)
        weaker = min(gx_score, gy_score)
        stronger = max(gx_score, gy_score)
        ratio = stronger / weaker if weaker > 0 else float("inf")

        main_decision = decisions_by_margin[main_margin]
        comp_decision = decisions_by_margin[comparison_margin]

        results_rows.append(
            {
                "swatch": filename,
                "mean_abs_Gx": gx_score,
                "mean_abs_Gy": gy_score,
                f"decision (margin {main_margin})": main_decision,
                f"decision (margin {comparison_margin})": comp_decision,
            }
        )

        trial_c_rows.append(
            {
                "swatch": filename,
                "mean_abs_Gx": gx_score,
                "mean_abs_Gy": gy_score,
                "dominant_to_weaker_ratio": ratio,
                "decision_margin_1.0": comp_decision,
                "decision_margin_2.5": main_decision,
                "is_uncertain_at_2.5": main_decision == "uncertain",
            }
        )

    return results_rows, trial_c_rows, uncertain_cases
