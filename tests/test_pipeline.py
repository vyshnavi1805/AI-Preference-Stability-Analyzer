"""
Tests for reply parsing, cyclic triads, perturbations, the full
pipeline and dataset validation.

The LLM is replaced by small fake functions, so these tests run
without Ollama and always give the same result.
"""
import unittest
from unittest.mock import patch

import networkx as nx

from App.consistency import check_consistency, count_cyclic_triads
from App.llm import parse_preference
from App.perturbations import run_perturbation_tests
from experiments.run_experiments import validate_record
from main import run_pipeline


# ----------------------------------------------------------------
# Fake models: each takes the same arguments as generate_preference
# ----------------------------------------------------------------

# Rock beats Scissors, Scissors beats Paper, Paper beats Rock.
RPS_BEATS = {
    ("Rock", "Scissors"),
    ("Scissors", "Paper"),
    ("Paper", "Rock"),
}

LAPTOP_RANK = ["Laptop P", "Laptop Q", "Laptop R", "Laptop S"]


def rps_model(question, option_a, option_b, context=""):
    return "A" if (option_a, option_b) in RPS_BEATS else "B"


def ranked_model(question, option_a, option_b, context=""):
    better_is_a = LAPTOP_RANK.index(option_a) < LAPTOP_RANK.index(option_b)
    return "A" if better_is_a else "B"


def always_first_model(question, option_a, option_b, context=""):
    # Pure position bias: always picks whichever option is shown first.
    return "A"


def patch_model(fake):
    """Replace the LLM call everywhere the pipeline uses it."""
    return [
        patch("App.preferences.generate_preference", side_effect=fake),
        patch("App.perturbations.generate_preference", side_effect=fake),
    ]


def run_with_model(fake, question, options):
    patches = patch_model(fake)
    for p in patches:
        p.start()
    try:
        return run_pipeline(question=question, options=options)
    finally:
        for p in patches:
            p.stop()


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

class TestParsePreference(unittest.TestCase):

    def test_valid_replies(self):
        for reply, expected in [
            ("A", "A"), ("b", "B"), (" B ", "B"), ("A.", "A"),
            ("Option B", "B"), ("B - better battery", "B"),
        ]:
            with self.subTest(reply=reply):
                self.assertEqual(parse_preference(reply), expected)

    def test_invalid_replies(self):
        for reply in ["A or B", "I think", "", "a good one"]:
            with self.subTest(reply=reply):
                with self.assertRaises(ValueError):
                    parse_preference(reply)


class TestCyclicTriads(unittest.TestCase):

    def test_one_triad_in_rock_paper_scissors(self):
        graph = nx.DiGraph(list(RPS_BEATS))
        self.assertEqual(count_cyclic_triads(graph), 1)

    def test_no_triads_in_ranking(self):
        graph = nx.DiGraph([
            ("P", "Q"), ("P", "R"), ("P", "S"),
            ("Q", "R"), ("Q", "S"), ("R", "S"),
        ])
        self.assertEqual(count_cyclic_triads(graph), 0)

    def test_listed_cycles_are_capped_at_ten(self):
        # Full "everyone beats the next one, last beats first" graph
        # on 8 nodes has far more than 10 cycles.
        graph = nx.DiGraph()
        nodes = list(range(8))
        for i in nodes:
            for j in nodes:
                if i != j and (j - i) % 8 in (1, 2, 3):
                    graph.add_edge(i, j)

        result = check_consistency(graph)

        self.assertFalse(result["is_consistent"])
        self.assertEqual(len(result["cycles"]), 10)
        self.assertGreater(result["cyclic_triads"], 0)


class TestPerturbations(unittest.TestCase):

    def test_consistent_model_is_stable(self):
        with patch(
            "App.perturbations.generate_preference",
            side_effect=ranked_model,
        ):
            results = run_perturbation_tests("Q", LAPTOP_RANK)

        self.assertTrue(all(r["order_stable"] for r in results))
        self.assertTrue(all(r["prompt_stable"] for r in results))

    def test_position_bias_is_detected(self):
        with patch(
            "App.perturbations.generate_preference",
            side_effect=always_first_model,
        ):
            results = run_perturbation_tests("Q", LAPTOP_RANK)

        # Swapping the order always flips the winner.
        self.assertFalse(any(r["order_stable"] for r in results))


class TestPipelineEndToEnd(unittest.TestCase):

    def test_cycle_input(self):
        result = run_with_model(
            rps_model,
            "Which hand wins most often?",
            ["Rock", "Paper", "Scissors"],
        )

        self.assertFalse(result["consistency"]["is_consistent"])
        self.assertEqual(result["consistency"]["cyclic_triads"], 1)
        self.assertIn("INCONSISTENT", result["explanation"])

    def test_no_cycle_input(self):
        result = run_with_model(
            ranked_model,
            "Which laptop is best for a CS student?",
            LAPTOP_RANK,
        )

        self.assertTrue(result["consistency"]["is_consistent"])
        self.assertEqual(result["consistency"]["cyclic_triads"], 0)
        self.assertEqual(result["stability"]["overall_stability_rate"], 100.0)

    def test_position_bias_lowers_order_stability(self):
        result = run_with_model(
            always_first_model,
            "Which laptop is best for a CS student?",
            LAPTOP_RANK,
        )

        self.assertEqual(result["stability"]["order_stability_rate"], 0.0)

class TestPreferenceGraph(unittest.TestCase):

    def test_build_preference_graph(self):
        from App.graph import build_preference_graph

        results = [
            {
                "option_a": "Rock",
                "option_b": "Scissors",
                "preference": "A",
            },
            {
                "option_a": "Scissors",
                "option_b": "Paper",
                "preference": "A",
            },
            {
                "option_a": "Rock",
                "option_b": "Paper",
                "preference": "B",
            },
        ]

        graph = build_preference_graph(results)

        self.assertTrue(graph.has_edge("Rock", "Scissors"))
        self.assertTrue(graph.has_edge("Scissors", "Paper"))
        self.assertTrue(graph.has_edge("Paper", "Rock"))
        self.assertEqual(count_cyclic_triads(graph), 1)
class TestDatasetValidation(unittest.TestCase):

    def test_valid_record(self):
        question, options, context = validate_record(
            {"question": " Q ", "options": ["X", "Y"]}, 1
        )
        self.assertEqual((question, options, context), ("Q", ["X", "Y"], ""))

    def test_invalid_records(self):
        bad_records = [
            {"question": "", "options": ["X", "Y"]},
            {"question": "Q", "options": ["X"]},
            {"question": "Q", "options": ["X", "X "]},
            {"question": "Q", "options": ["X", ""]},
            "not a dict",
        ]
        for record in bad_records:
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    validate_record(record, 1)

if __name__ == "__main__":
    unittest.main()