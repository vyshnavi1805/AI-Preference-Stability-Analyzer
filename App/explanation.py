def _short_option(value, limit=60):
    """Keep long candidate responses readable in the text report."""
    text = str(value).replace("\n", " ").strip()

    if len(text) <= limit:
        return text

    return text[: limit - 3].rstrip() + "..."


def generate_explanation(
    consistency_result,
    stability_result,
    perturbation_results=None,
):
    """Generate a human-readable preference analysis report."""

    report = []

    report.append("=" * 65)
    report.append("AI PREFERENCE STABILITY ANALYSIS REPORT")
    report.append("=" * 65)

    # 1. Consistency analysis
    report.append("\n1. PREFERENCE CONSISTENCY")

    is_consistent = consistency_result.get("is_consistent")
    cycles = consistency_result.get("cycles", [])

    if is_consistent is True:
        report.append("Status: CONSISTENT")
        report.append("No directed preference cycles were detected.")

    elif is_consistent is False:
        report.append("Status: INCONSISTENT")
        report.append(
            "Cyclic triads (3-option cycles): "
            f"{consistency_result.get('cyclic_triads', 'N/A')}"
        )
        report.append("Example cycles (at most 10 shown):")

        for index, cycle in enumerate(cycles, start=1):
            if cycle:
                cycle_path = " -> ".join(map(str, cycle))
                cycle_path += f" -> {cycle[0]}"
                report.append(f"Cycle {index}: {cycle_path}")

    else:
        report.append("Status: UNKNOWN")
        report.append(
            "The consistency result is missing or invalid."
        )

    # 2. Stability metrics
    report.append("\n2. PREFERENCE STABILITY")

    total_pairs = stability_result.get("total_pairs", 0)

    if total_pairs == 0:
        report.append("Status: INSUFFICIENT DATA")
        report.append(
            "No pairwise perturbation results are available."
        )

    else:
        order_rate = stability_result.get("order_stability_rate")
        prompt_rate = stability_result.get("prompt_stability_rate")
        overall_rate = stability_result.get("overall_stability_rate")

        report.append(f"Pairs evaluated: {total_pairs}")

        if order_rate is not None:
            report.append(f"Order stability: {order_rate}%")

        if prompt_rate is not None:
            report.append(f"Prompt stability: {prompt_rate}%")

        if overall_rate is not None:
            report.append(
                f"Stability under both tests: {overall_rate}%"
            )

        # 3. Interpretation
        report.append("\n3. INTERPRETATION")

        if order_rate is not None:
            if order_rate == 100:
                report.append(
                    "- All tested pairs preserved their preferences "
                    "when the option order was reversed."
                )
            elif order_rate == 0:
                report.append(
                    "- Every tested pair changed its selected option "
                    "under reversed presentation."
                )
            else:
                report.append(
                    "- Some preferences changed when the option order "
                    "was reversed."
                )

        if prompt_rate is not None:
            if prompt_rate == 100:
                report.append(
                    "- All tested pairs were stable under the tested "
                    "question paraphrases."
                )
            elif prompt_rate == 0:
                report.append(
                    "- No tested pair satisfied the strict prompt-"
                    "stability criterion."
                )
            else:
                report.append(
                    "- Some pairs were unstable under the tested "
                    "question paraphrases."
                )

        report.append(
            "- Stability describes agreement with the baseline "
            "preferences; it does not establish correctness."
        )

    # 4. Pairwise instability details
    if perturbation_results:
        report.append("\n4. DETECTED PREFERENCE CHANGES")
        changes_found = 0

        for result in perturbation_results:
            option_a = _short_option(
                result.get("option_a", "Option A")
            )
            option_b = _short_option(
                result.get("option_b", "Option B")
            )

            if result.get("order_stable") is False:
                changes_found += 1
                report.append(
                    f"- Order sensitivity: {option_a} vs {option_b}"
                )

            if result.get("prompt_stable") is False:
                changes_found += 1
                report.append(
                    f"- Prompt sensitivity: {option_a} vs {option_b}"
                )

        if changes_found == 0:
            report.append(
                "No preference changes were recorded "
                "in the supplied perturbation results."
            )

    # 5. Limitations
    report.append("\n5. INTERPRETATION NOTES")
    report.append(
        "- Consistency checks detect directed cycles in the graph."
    )
    report.append(
        "- Stability metrics depend on the tested alternatives, "
        "prompts, and perturbations."
    )
    report.append(
        "- Results should not be generalised beyond the evaluated "
        "cases without further testing."
    )

    report.append("\n" + "=" * 65)
    report.append("END OF REPORT")
    report.append("=" * 65)

    return "\n".join(report)