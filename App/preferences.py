from itertools import combinations
from App.llm import generate_preference
def generate_pairs(options):
    """
    Generate all unique pairwise combinations.
    """
    return list(combinations(options, 2))
def evaluate_pairs(question, options, context=""):
    """
    Generate pairwise comparisons and ask the LLM
    to choose a preferred option for each pair.
    """
    pairs = generate_pairs(options)
    results = []
    for option_a, option_b in pairs:
        preference = generate_preference(
            question,
            option_a,
            option_b,
            context
        )
        results.append({
            "option_a": option_a,
            "option_b": option_b,
            "preference": preference
        })
    return results