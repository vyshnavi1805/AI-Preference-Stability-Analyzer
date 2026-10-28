import networkx as nx


def build_preference_graph(results):
    graph = nx.DiGraph()

    for result in results:
        option_a = result["option_a"]
        option_b = result["option_b"]
        preference = result["preference"].strip().upper()

        if preference == "A":
            graph.add_edge(option_a, option_b)

        elif preference == "B":
            graph.add_edge(option_b, option_a)

        else:
            raise ValueError(
                f"Invalid preference: {preference!r}. "
                "Expected exactly 'A' or 'B'."
            )

    return graph