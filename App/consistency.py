
from itertools import combinations, islice
import networkx as nx


def count_cyclic_triads(graph):
    """Count unique three-node directed cycles."""
    count = 0

    for a, b, c in combinations(graph.nodes, 3):
        for x, y, z in ((a, b, c), (a, c, b)):
            if (
                graph.has_edge(x, y)
                and graph.has_edge(y, z)
                and graph.has_edge(z, x)
            ):
                count += 1

    return count


def detect_cycles(graph, max_cycles=10):
    """Return at most max_cycles directed cycles."""
    if max_cycles < 0:
        raise ValueError("max_cycles cannot be negative.")

    return list(islice(nx.simple_cycles(graph), max_cycles))


def check_consistency(graph, max_cycles=10):
    """
    Check whether the directed graph contains any cycle.

    The cycle display limit affects only the returned examples,
    not the consistency verdict.
    """
    if max_cycles < 0:
        raise ValueError("max_cycles cannot be negative.")

    # Detect at least one cycle independently of the display limit.
    has_cycle = next(nx.simple_cycles(graph), None) is not None

    cycles = detect_cycles(graph, max_cycles)

    return {
        "is_consistent": not has_cycle,
        "cycles": cycles,
        "cyclic_triads": count_cyclic_triads(graph),
    }
