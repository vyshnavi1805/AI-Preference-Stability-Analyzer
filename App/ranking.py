"""Rank alternatives using aggregated LLM preferences."""

from itertools import combinations


def rank_alternatives(options, perturbation_results):
    """
    Rank alternatives using Copeland scoring.

    Unresolved/missing comparisons do not affect scores. Duplicate records for
    the same unordered pair are marked unresolved to avoid order-dependent
    rankings. Verification does not modify this ranking.
    """
    if not isinstance(options, (list, tuple)):
        raise TypeError("options must be a list or tuple.")
    if len(options) < 2:
        raise ValueError("At least two alternatives are required.")
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError("Every alternative must be a non-empty string.")

    options = [option.strip() for option in options]
    if len(set(options)) != len(options):
        raise ValueError("Each alternative must be unique after trimming whitespace.")
    if not isinstance(perturbation_results, (list, tuple)):
        raise TypeError("perturbation_results must be a list or tuple.")

    valid_options = set(options)
    stats = {
        option: {"wins": 0, "losses": 0, "score": 0, "evaluated_pairs": 0}
        for option in options
    }
    expected_pairs = {
        frozenset((a, b)) for a, b in combinations(options, 2)
    }

    # Group by unordered pair so reversed duplicates are detected.
    records_by_pair = {}
    for result in perturbation_results:
        if not isinstance(result, dict):
            continue
        option_a = result.get("option_a")
        option_b = result.get("option_b")
        if not isinstance(option_a, str) or not isinstance(option_b, str):
            continue
        if option_a not in valid_options or option_b not in valid_options:
            continue
        if option_a == option_b:
            continue
        records_by_pair.setdefault(frozenset((option_a, option_b)), []).append(result)

    pair_results = []
    for pair_key in sorted(
        expected_pairs,
        key=lambda pair: tuple(sorted(pair, key=options.index)),
    ):
        option_a, option_b = sorted(pair_key, key=options.index)
        records = records_by_pair.get(pair_key, [])

        if not records:
            pair_results.append({
                "option_a": option_a,
                "option_b": option_b,
                "baseline_choice": None,
                "aggregated_verdict": None,
                "aggregation_status": "unresolved",
                "source_aggregation_status": "missing comparison",
                "baseline_agrees_with_aggregate": None,
                "unresolved_reason": "Missing comparison result",
            })
            continue

        if len(records) > 1:
            pair_results.append({
                "option_a": option_a,
                "option_b": option_b,
                "baseline_choice": None,
                "aggregated_verdict": None,
                "aggregation_status": "unresolved",
                "source_aggregation_status": "duplicate comparison records",
                "baseline_agrees_with_aggregate": None,
                "unresolved_reason": (
                    f"Found {len(records)} records for the same unordered pair; "
                    "duplicates are excluded to avoid order-dependent scoring."
                ),
            })
            continue

        result = records[0]
        baseline = result.get("baseline_choice")
        aggregated_verdict = result.get("aggregated_verdict")
        source_status = result.get("aggregation_status")

        resolved = (
            isinstance(aggregated_verdict, str)
            and aggregated_verdict in (option_a, option_b)
            and source_status in ("verdict_available", "resolved")
        )

        if resolved:
            winner = aggregated_verdict
            loser = option_b if winner == option_a else option_a
            stats[winner]["wins"] += 1
            stats[loser]["losses"] += 1
            stats[winner]["score"] += 1
            stats[loser]["score"] -= 1
            stats[option_a]["evaluated_pairs"] += 1
            stats[option_b]["evaluated_pairs"] += 1
            pair_status = "resolved"
            reason = None
        else:
            winner = None
            pair_status = "unresolved"
            reason = source_status or "No valid aggregated verdict"

        pair_results.append({
            "option_a": option_a,
            "option_b": option_b,
            "baseline_choice": baseline,
            "aggregated_verdict": winner,
            "aggregation_status": pair_status,
            "source_aggregation_status": source_status,
            "baseline_agrees_with_aggregate": (
                baseline == winner
                if winner is not None and baseline in (option_a, option_b)
                else None
            ),
            "unresolved_reason": reason,
        })

    ordered = sorted(options, key=lambda option: stats[option]["score"], reverse=True)
    ranking = []
    previous_score = None
    current_rank = 0
    for index, option in enumerate(ordered, start=1):
        score = stats[option]["score"]
        if score != previous_score:
            current_rank = index
            previous_score = score
        ranking.append({
            "alternative": option,
            "rank": current_rank,
            **stats[option],
        })

    total_pairs = len(expected_pairs)
    resolved_pairs = sum(
        item["aggregation_status"] == "resolved" for item in pair_results
    )
    unresolved_pairs = [
        item for item in pair_results
        if item["aggregation_status"] != "resolved"
    ]
    coverage = resolved_pairs / total_pairs if total_pairs else 0.0

    if resolved_pairs == 0:
        status, final_choice = "no_reliable_winner", None
    else:
        highest_score = ranking[0]["score"]
        leaders = [
            row["alternative"] for row in ranking
            if row["score"] == highest_score
        ]
        if len(leaders) > 1:
            status, final_choice = "tie", None
        else:
            status, final_choice = "winner", leaders[0]

    comparable_pairs = [
        item for item in pair_results
        if item["aggregated_verdict"] is not None
        and item["baseline_choice"] in (item["option_a"], item["option_b"])
    ]
    baseline_agreement_rate = (
        sum(
            item["baseline_choice"] == item["aggregated_verdict"]
            for item in comparable_pairs
        ) / len(comparable_pairs)
        if comparable_pairs else None
    )

    explanation = _build_explanation(
        final_choice=final_choice,
        status=status,
        ranking=ranking,
        resolved_pairs=resolved_pairs,
        total_pairs=total_pairs,
        unresolved_pairs=unresolved_pairs,
        coverage=coverage,
        baseline_agreement_rate=baseline_agreement_rate,
    )

    return {
        "final_choice": final_choice,
        "status": status,
        "ranking": ranking,
        "pair_results": pair_results,
        "resolved_pairs": resolved_pairs,
        "unresolved_pairs": unresolved_pairs,
        "total_pairs": total_pairs,
        "coverage": coverage,
        "baseline_agreement_rate": baseline_agreement_rate,
        "explanation": explanation,
        "scoring_method": "Copeland score (wins minus losses)",
    }


def _build_explanation(
    final_choice,
    status,
    ranking,
    resolved_pairs,
    total_pairs,
    unresolved_pairs,
    coverage,
    baseline_agreement_rate,
):
    if status == "winner":
        winner = next(
            row for row in ranking if row["alternative"] == final_choice
        )
        parts = [
            f"{final_choice} ranks first with a Copeland score of {winner['score']}.",
            f"It has {winner['wins']} win(s) and {winner['losses']} loss(es) "
            "among resolved comparisons.",
        ]
    elif status == "tie":
        leaders = [
            row["alternative"] for row in ranking
            if row["score"] == ranking[0]["score"]
        ]
        parts = [
            "There is no unique winner. The top alternatives are tied: "
            + ", ".join(str(item) for item in leaders) + "."
        ]
    else:
        parts = [
            "No reliable winner can be determined because no "
            "pairwise comparisons were resolved."
        ]

    parts.append(
        f"Resolved comparisons: {resolved_pairs} of {total_pairs} "
        f"({coverage:.1%} coverage)."
    )
    if baseline_agreement_rate is not None:
        parts.append(
            f"Baseline agreement with aggregated verdicts: "
            f"{baseline_agreement_rate:.1%}."
        )
    if unresolved_pairs:
        descriptions = [
            f"{item['option_a']} vs {item['option_b']}: "
            f"{item.get('unresolved_reason') or 'unresolved'}"
            for item in unresolved_pairs
        ]
        parts.append("Excluded unresolved comparisons: " + "; ".join(descriptions) + ".")
    parts.append(
        "The ranking represents aggregated LLM preferences, not objective correctness."
    )
    return " ".join(parts)
