from itertools import combinations

from .llm import generate_preference
RESPONSE_PARAPHRASES = (
    "Which response better answers the original request?",
    "Which answer is more appropriate for the user's request?",
    "Which response is the better answer to the request?",
)

DECISION_PARAPHRASES = (
    "Which option best fits the decision described in the question?",
    "Which alternative is the better choice for the stated decision?",
    "Which option would be preferable for the situation described?",
)


def _selected_option(preference, first_option, second_option):
    """Convert an A/B response into the selected option."""
    choice = str(preference).strip().upper()

    if choice == "A":
        return first_option
    if choice == "B":
        return second_option

    raise ValueError(
        f"Unexpected preference returned by LLM: {preference!r}"
    )


def _evaluation_question(question, paraphrase, question_style):
    """Frame a paraphrase around the original task."""
    label = (
        "Original request"
        if question_style == "response"
        else "Decision question"
    )

    return f"{label}:\n{question}\n\n{paraphrase}"


def _get_paraphrases(question_style, paraphrases):
    if paraphrases is not None:
        selected = tuple(paraphrases)
    elif question_style == "response":
        selected = RESPONSE_PARAPHRASES
    else:
        selected = DECISION_PARAPHRASES

    if not selected:
        raise ValueError("At least one paraphrase is required.")

    return selected


def _majority_verdict(verdicts):
    """
    Return a strict majority verdict.

    Ignore None values. Return None if there are no valid votes
    or if no verdict receives more than half of the valid votes.
    """
    votes = [verdict for verdict in verdicts if verdict is not None]

    if not votes:
        return None

    counts = {}
    for verdict in votes:
        counts[verdict] = counts.get(verdict, 0) + 1

    highest = max(counts.values())
    winners = [
        verdict
        for verdict, count in counts.items()
        if count == highest
    ]

    if len(winners) != 1:
        return None

    if highest <= len(votes) / 2:
        return None

    return winners[0]


def run_perturbation_tests(
    question,
    options,
    context="",
    baseline_results=None,
    paraphrases=None,
    question_style="decision",
):
    """
    Evaluate every unique pair under reversed order and paraphrases.

    A consistent choice exists only if baseline and reversed-baseline
    judgments select the same actual option.

    Every paraphrase is tested in both option orders. The aggregated
    verdict is the strict majority of the order-consistent baseline
    and paraphrase verdicts.

    question_style:
        "decision" - decision alternatives such as laptops.
        "response" - responses to a request, such as MT-Bench.
    """
    if len(options) < 2:
        raise ValueError("At least two options are required.")

    if len(set(options)) != len(options):
        raise ValueError("Each option must be unique.")

    if question_style not in {"decision", "response"}:
        raise ValueError(
            "question_style must be 'decision' or 'response'."
        )

    paraphrases = _get_paraphrases(question_style, paraphrases)

    baseline_lookup = {}

    if baseline_results is not None:
        for result in baseline_results:
            pair = (result["option_a"], result["option_b"])
            baseline_lookup[pair] = result["preference"]

    results = []

    for option_a, option_b in combinations(options, 2):
        pair = (option_a, option_b)

        # Baseline preference
        if baseline_results is None:
            baseline_preference = generate_preference(
                question, option_a, option_b, context
            )
        else:
            if pair not in baseline_lookup:
                raise ValueError(
                    f"Baseline result missing for pair: {pair}"
                )

            baseline_preference = baseline_lookup[pair]

        baseline_choice = _selected_option(
            baseline_preference, option_a, option_b
        )

        # Reverse the option order
        reversed_preference = generate_preference(
            question, option_b, option_a, context
        )

        reversed_choice = _selected_option(
            reversed_preference, option_b, option_a
        )

        order_stable = baseline_choice == reversed_choice

        consistent_choice = (
            baseline_choice if order_stable else None
        )

        # Track first-slot selections across all recorded judgments
        first_slot_picks = int(
            str(baseline_preference).strip().upper() == "A"
        )
        total_calls = 1

        first_slot_picks += int(
            str(reversed_preference).strip().upper() == "A"
        )
        total_calls += 1

        paraphrase_results = []

        for index, paraphrase in enumerate(paraphrases, start=1):
            perturbed_question = _evaluation_question(
                question, paraphrase, question_style
            )

            # Original option order
            forward_preference = generate_preference(
                perturbed_question,
                option_a,
                option_b,
                context,
            )

            forward_choice = _selected_option(
                forward_preference, option_a, option_b
            )

            # Reversed option order
            reverse_preference = generate_preference(
                perturbed_question,
                option_b,
                option_a,
                context,
            )

            reverse_choice = _selected_option(
                reverse_preference, option_b, option_a
            )

            paraphrase_order_stable = (
                forward_choice == reverse_choice
            )

            paraphrase_consistent_choice = (
                forward_choice
                if paraphrase_order_stable
                else None
            )

            comparable = (
                consistent_choice is not None
                and paraphrase_consistent_choice is not None
            )

            stable = (
                comparable
                and paraphrase_consistent_choice == consistent_choice
            )

            first_slot_picks += int(
                str(forward_preference).strip().upper() == "A"
            )
            first_slot_picks += int(
                str(reverse_preference).strip().upper() == "A"
            )
            total_calls += 2

            paraphrase_results.append({
                "paraphrase_id": index,
                "paraphrase": paraphrase,
                "forward_preference": forward_preference,
                "forward_choice": forward_choice,
                "reverse_preference": reverse_preference,
                "reverse_choice": reverse_choice,
                "order_stable": paraphrase_order_stable,
                "consistent_choice": paraphrase_consistent_choice,
                "prompt_comparable": comparable,
                "prompt_stable": bool(stable),
            })

        comparable_paraphrases = [
            item
            for item in paraphrase_results
            if item["prompt_comparable"]
        ]

        prompt_comparable = consistent_choice is not None

        # Strict definition: every paraphrase must be comparable and stable.
        prompt_stable = (
            bool(comparable_paraphrases)
            and len(comparable_paraphrases) == len(paraphrase_results)
            and all(
                item["prompt_stable"]
                for item in comparable_paraphrases
            )
        )

        # Aggregate only order-consistent verdicts.
        verdicts = (
            [consistent_choice]
            + [
                item["consistent_choice"]
                for item in paraphrase_results
            ]
        )

        aggregated_verdict = _majority_verdict(verdicts)

        valid_votes = [
            verdict for verdict in verdicts
            if verdict is not None
        ]

        if not valid_votes:
            aggregation_status = "no_stable_votes"
        elif aggregated_verdict is None:
            aggregation_status = "no_strict_majority"
        else:
            aggregation_status = "verdict_available"

        # Preserve fields expected by the existing dashboard.
        first_paraphrase = paraphrase_results[0]

        results.append({
            "option_a": option_a,
            "option_b": option_b,

            "baseline_preference": baseline_preference,
            "baseline_choice": baseline_choice,

            "reversed_preference": reversed_preference,
            "reversed_choice": reversed_choice,

            "reframed_preference":
                first_paraphrase["forward_preference"],
            "reframed_choice":
                first_paraphrase["forward_choice"],

            "order_stable": order_stable,
            "position_driven": not order_stable,
            "consistent_choice": consistent_choice,

            "aggregated_verdict": aggregated_verdict,
            "aggregated_verdict_votes": len(valid_votes),
            "aggregation_status": aggregation_status,

            "prompt_comparable": prompt_comparable,
            "prompt_stable": prompt_stable,
            "paraphrase_results": paraphrase_results,

            "paraphrase_stability_rate": (
                round(
                    sum(
                        item["prompt_stable"]
                        for item in comparable_paraphrases
                    )
                    / len(comparable_paraphrases) * 100,
                    2,
                )
                if comparable_paraphrases
                else None
            ),

            "first_slot_picks": first_slot_picks,
            "total_calls": total_calls,
        })

    return results