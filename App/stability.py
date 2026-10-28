"""
Stability metrics for pairwise LLM preferences.

The original metrics (order, prompt and overall stability) are kept
unchanged. New metrics use the extra fields produced by
perturbations.py and are None when those fields are missing, so older
results and the existing tests keep working.

Note: the order stability rate is also the coverage, i.e. the share of
pairs where the model has a real, order-consistent preference.
"""


def _rate(part, whole):
    """Percentage rounded to 2 decimals, or None if whole is 0."""
    if not whole:
        return None
    return round(part / whole * 100, 2)


def analyze_stability(perturbation_results):
    """Calculate preference stability metrics."""

    total_pairs = len(perturbation_results)

    if total_pairs == 0:
        return {
            "total_pairs": 0,
            "order_stable_pairs": 0,
            "prompt_stable_pairs": 0,
            "stable_under_both": 0,
            "order_stability_rate": None,
            "prompt_stability_rate": None,
            "overall_stability_rate": None,
            "position_driven_pairs": 0,
            "position_bias_rate": None,
            "comparable_pairs": 0,
            "prompt_stable_comparable_pairs": 0,
            "comparable_prompt_stability_rate": None,
            "mean_paraphrase_stability_rate": None,
        }

    # ---------------- Original metrics ----------------

    order_stable = sum(
        result["order_stable"] for result in perturbation_results
    )
    prompt_stable = sum(
        result["prompt_stable"] for result in perturbation_results
    )
    stable_under_both = sum(
        result["order_stable"] and result["prompt_stable"]
        for result in perturbation_results
    )

    # ---------------- Position bias ----------------
    # Share of all LLM calls in which the first-shown option was picked.
    # 50% means no slot preference; far from 50% means position bias.

    calls = [
        result for result in perturbation_results
        if "first_slot_picks" in result and "total_calls" in result
    ]

    first_slot_picks = sum(result["first_slot_picks"] for result in calls)
    total_calls = sum(result["total_calls"] for result in calls)

    position_bias_rate = _rate(first_slot_picks, total_calls)

    # ---------------- Prompt stability on comparable pairs ----------------
    # Prompt stability is only meaningful when the baseline was not
    # position-driven; position-driven pairs are excluded here.

    comparable = [
        result for result in perturbation_results
        if result.get("prompt_comparable") is True
    ]
    prompt_stable_comparable = sum(
        result["prompt_stable"] for result in comparable
    )

    has_comparable_field = any(
        "prompt_comparable" in result for result in perturbation_results
    )

    # ---------------- Per-paraphrase agreement ----------------

    paraphrase_rates = [
        result["paraphrase_stability_rate"]
        for result in perturbation_results
        if result.get("paraphrase_stability_rate") is not None
    ]
    mean_paraphrase_rate = (
        round(sum(paraphrase_rates) / len(paraphrase_rates), 2)
        if paraphrase_rates else None
    )

    return {
        "total_pairs": total_pairs,
        "order_stable_pairs": order_stable,
        "prompt_stable_pairs": prompt_stable,
        "stable_under_both": stable_under_both,

        "order_stability_rate": _rate(order_stable, total_pairs),
        "prompt_stability_rate": _rate(prompt_stable, total_pairs),
        "overall_stability_rate": _rate(stable_under_both, total_pairs),

        "position_driven_pairs": total_pairs - order_stable,
        "position_bias_rate": position_bias_rate,

        "comparable_pairs": len(comparable),
        "prompt_stable_comparable_pairs": prompt_stable_comparable,
        "comparable_prompt_stability_rate": (
            _rate(prompt_stable_comparable, len(comparable))
            if has_comparable_field else None
        ),

        "mean_paraphrase_stability_rate": mean_paraphrase_rate,
    }
