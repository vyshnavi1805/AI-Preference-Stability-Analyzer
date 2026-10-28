
import unittest
import networkx as nx
from App.graph import build_preference_graph
from App.consistency import check_consistency
from App.stability import analyze_stability


class TestPreferenceGraph(unittest.TestCase):

    def test_acyclic_preferences_are_consistent(self):
        results = [
            {
                "option_a": "A",
                "option_b": "B",
                "preference": "A",
            },
            {
                "option_a": "A",
                "option_b": "C",
                "preference": "A",
            },
            {
                "option_a": "B",
                "option_b": "C",
                "preference": "A",
            },
        ]

        graph = build_preference_graph(results)
        result = check_consistency(graph)

        self.assertTrue(result["is_consistent"])
        self.assertEqual(result["cycles"], [])

    def test_directed_cycle_is_detected(self):
        results = [
            # A is preferred over B: A -> B
            {
                "option_a": "A",
                "option_b": "B",
                "preference": "A",
            },

            # B is preferred over C: B -> C
            {
                "option_a": "B",
                "option_b": "C",
                "preference": "A",
            },

            # C is preferred over A: C -> A
            {
                "option_a": "A",
                "option_b": "C",
                "preference": "B",
            },
        ]

        graph = build_preference_graph(results)

        result = check_consistency(graph)

        self.assertFalse(result["is_consistent"])
        self.assertTrue(result["cycles"])

        expected_edges = {
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
        }

        self.assertEqual(set(graph.edges()), expected_edges)


class TestStabilityAnalysis(unittest.TestCase):

    def test_stability_metrics(self):
        results = [
            {
                "order_stable": True,
                "prompt_stable": True,
            },
            {
                "order_stable": False,
                "prompt_stable": True,
            },
        ]

        metrics = analyze_stability(results)

        self.assertEqual(metrics["total_pairs"], 2)
        self.assertEqual(metrics["order_stable_pairs"], 1)
        self.assertEqual(metrics["prompt_stable_pairs"], 2)
        self.assertEqual(metrics["stable_under_both"], 1)

        self.assertEqual(metrics["order_stability_rate"], 50.0)
        self.assertEqual(metrics["prompt_stability_rate"], 100.0)
        self.assertEqual(metrics["overall_stability_rate"], 50.0)

    def test_empty_stability_results(self):
        metrics = analyze_stability([])

        self.assertEqual(metrics["total_pairs"], 0)
        self.assertIsNone(metrics["order_stability_rate"])
        self.assertIsNone(metrics["prompt_stability_rate"])
        self.assertIsNone(metrics["overall_stability_rate"])
    
    def test_cycle_detected_when_display_limit_is_zero(self):
        graph = build_preference_graph([
            {"option_a": "A", "option_b": "B", "preference": "A"},
            {"option_a": "B", "option_b": "C", "preference": "A"},
            {"option_a": "A", "option_b": "C", "preference": "B"},
        ])

        result = check_consistency(graph, max_cycles=0)

        self.assertFalse(result["is_consistent"])
        self.assertEqual(result["cycles"], [])
        self.assertEqual(result["cyclic_triads"], 1)

    def test_negative_cycle_limit_is_rejected(self):
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
        ])

        with self.assertRaises(ValueError):
            check_consistency(graph, max_cycles=-1)
    
    def test_position_bias_metrics(self):
        results = [
            {
                "order_stable": False,
                "prompt_stable": False,
                "first_slot_picks": 8,
                "total_calls": 10,
                "prompt_comparable": False,
                "paraphrase_stability_rate": None,
            },
            {
                "order_stable": True,
                "prompt_stable": True,
                "first_slot_picks": 2,
                "total_calls": 10,
                "prompt_comparable": True,
                "paraphrase_stability_rate": 100.0,
            },
        ]

        metrics = analyze_stability(results)

        self.assertEqual(metrics["position_bias_rate"], 50.0)
        self.assertEqual(metrics["position_driven_pairs"], 1)
        self.assertEqual(metrics["comparable_pairs"], 1)
        self.assertEqual(
            metrics["comparable_prompt_stability_rate"], 100.0
        )
        self.assertEqual(
            metrics["mean_paraphrase_stability_rate"], 100.0
        )

    def test_missing_new_metrics_remain_undefined(self):
        results = [
            {
                "order_stable": True,
                "prompt_stable": True,
            }
        ]

        metrics = analyze_stability(results)

        self.assertIsNone(metrics["position_bias_rate"])
        self.assertIsNone(
            metrics["comparable_prompt_stability_rate"]
        )
        self.assertIsNone(
            metrics["mean_paraphrase_stability_rate"]
        )




if __name__ == "__main__":
    unittest.main()
