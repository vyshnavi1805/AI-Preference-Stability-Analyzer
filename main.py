import traceback
#from httpx import options
from App.preferences import evaluate_pairs
from App.graph import build_preference_graph
from App.consistency import check_consistency
from App.perturbations import run_perturbation_tests
from App.stability import analyze_stability
from App.explanation import generate_explanation
from App.ranking import rank_alternatives
from App.verification import verify_answer

def collect_input():
    """Collect a decision question and its alternatives."""

    print("\nAI Preference Stability Analyzer")
    print("=" * 50)

    question = input("Enter the decision question: ").strip()

    if not question:
        raise ValueError("The decision question cannot be empty.")

    context = input(
        "Enter additional context (press Enter to skip): "
    ).strip()

    print("\nEnter the alternatives.")
    print("Leave the input empty when you have entered all alternatives.")

    options = []

    while True:
        option = input(
            f"Alternative {len(options) + 1}: "
        ).strip()

        if not option:
            break

        options.append(option)

    if len(options) < 2:
        raise ValueError("At least two alternatives are required.")

    if len(set(options)) != len(options):
        raise ValueError("Each alternative must be unique.")

    return question, options, context




def run_pipeline(question, options, context="", question_style="decision"):
    """Execute the complete preference analysis pipeline."""

    # 1. Generate baseline pairwise preferences
    baseline_results = evaluate_pairs(
        question, options, context
    )

    # 2. Construct the preference graph
    graph = build_preference_graph(baseline_results)

    # 3. Analyse graph consistency
    consistency_result = check_consistency(graph)

    # 4. Test stability under controlled perturbations
    perturbation_results = run_perturbation_tests(
        question=question,
        options=options,
        context=context,
        baseline_results=baseline_results,
        question_style=question_style,
    )

    # 5. Calculate stability metrics
    stability_result = analyze_stability(
        perturbation_results
    )
    # 6. Aggregate stable pairwise preferences into a ranking
    ranking_result = rank_alternatives(
        options=options,
        perturbation_results=perturbation_results,
    )
    # 7. Verify the final ranked answer
    verification_result = verify_answer(
        question=question,
        selected_answer=ranking_result.get("final_choice"),
        options=options,
    )

    # 7. Generate a human-readable report
    explanation = generate_explanation(
        consistency_result=consistency_result,
        stability_result=stability_result,
        perturbation_results=perturbation_results,
    )

    return {
        "preferences": baseline_results,
        "graph": graph,
        "consistency": consistency_result,
        "perturbations": perturbation_results,
        "stability": stability_result,
        "ranking": ranking_result,
        "verification": verification_result,
        "explanation": explanation,
    }


def display_results(results):
    """Display the outputs of the analysis."""

    print("\n" + "=" * 65)
    print("BASELINE PREFERENCES")
    print("=" * 65)

    for result in results["preferences"]:
        print(f"\n{result['option_a']} VS {result['option_b']}")
        print(f"LLM preference: {result['preference']}")

    print("\n" + "=" * 65)
    print("PREFERENCE GRAPH")
    print("=" * 65)

    for source, target in results["graph"].edges():
        print(f"{source} -> {target}")

    print("\n" + "=" * 65)
    print("PERTURBATION RESULTS")
    print("=" * 65)

    for result in results["perturbations"]:
        print(f"\nPair: {result['option_a']} VS {result['option_b']}")
        print(f"Order stable: {result['order_stable']}")
        print(f"Prompt stable: {result['prompt_stable']}")

    
    print("\n" + results["explanation"])

    # Display final choice and ranking
    ranking_result = results.get("ranking", {})

    print("\n" + "=" * 65)
    print("FINAL CHOICE AND RANKING")
    print("=" * 65)

    status = ranking_result.get("status")
    final_choice = ranking_result.get("final_choice")

    if status == "winner" and final_choice is not None:
        print(f"\nFinal choice: {final_choice}")

    elif status == "tie":
        print("\nNo unique winner: the top alternatives are tied.")

    else:
        print("\nNo reliable winner could be determined.")

    print("\nComplete ranking:")

    for item in ranking_result.get("ranking", []):
        print(
            f"Rank {item['rank']}: {item['alternative']} | "
            f"Wins: {item['wins']} | "
            f"Losses: {item['losses']} | "
            f"Copeland score: {item['score']}"
        )

    print("\nReasoning:")
    print(
        ranking_result.get(
            "explanation",
            "No ranking explanation available.",
        )
    )
        # Display verification results
    verification = results.get("verification", {})

    print("\n" + "=" * 65)
    print("ANSWER VERIFICATION")
    print("=" * 65)

    print(f"Status: {verification.get('status', 'UNABLE_TO_VERIFY')}")
    print(f"Method: {verification.get('method', 'unknown')}")
    print(f"Selected answer: {verification.get('selected_answer')}")
    print(f"Reference/independent answer: {verification.get('expected_answer')}")
    print(f"Explanation: {verification.get('explanation', 'No explanation available.')}")


def main():
    # Handle user input separately.
    try:
        question, options, context = collect_input()

    except ValueError as error:
        print(f"\nInput error: {error}")
        return

    # Run the pipeline and display the results.
    try:
        results = run_pipeline(
            question=question,
            options=options,
            context=context,
        )
            
        display_results(results)
        print("\nAnalysis completed.")
    
    except RuntimeError as error:
        print(f"\n{error}")
    except Exception as error:
        print(
            f"\nPipeline execution failed: "
            f"{type(error).__name__}: {error}"
        )
        traceback.print_exc()


if __name__ == "__main__":
    main()

