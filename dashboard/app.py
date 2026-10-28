import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import networkx as nx

# Add project root so this works when launched with:
# streamlit run dashboard/app.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import run_pipeline

st.set_page_config(
    page_title="AI Preference Stability Analyzer",
    page_icon="🔎",
    layout="wide",
)

st.title("AI Preference Stability Analyzer")
st.write(
    "Compare alternatives, inspect the model's pairwise choices, view the preference graph, "
    "and check logical consistency and stability."
)

with st.container(border=True):
    st.subheader("1 · Define the decision")
    question = st.text_input(
        "Decision question",
        placeholder="e.g., Which project is best for a computer science student?",
    )
    context = st.text_area(
        "Decision criteria / context (optional)",
        placeholder="e.g., Prioritize learning value, usefulness, and portfolio impact.",
        height=90,
    )
    options_text = st.text_area(
        "Alternatives — one per line",
        placeholder="Build a compiler\nBuild a campus-management app\nBuild an AI code-review tool",
        height=150,
    )
    st.caption("Every unique pair is evaluated. Three alternatives produce three pairwise comparisons.")
    run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

if run_clicked:
    st.session_state.pop("analysis_results", None)
    clean_question = question.strip()
    clean_context = context.strip()
    options = [line.strip() for line in options_text.splitlines() if line.strip()]

    if not clean_question:
        st.error("Enter a decision question first.")
    elif len(options) < 2:
        st.error("Enter at least two alternatives, one per line.")
    elif len(set(options)) != len(options):
        st.error("Duplicate alternatives found. Make each alternative unique.")
    else:
        try:
            with st.spinner("Comparing alternatives and analysing the results…"):
                st.session_state["analysis_results"] = run_pipeline(
                    question=clean_question,
                    options=options,
                    context=clean_context,
                )
            st.success("Analysis finished.")
        except Exception as error:
            st.error(f"Analysis failed: {type(error).__name__}: {error}")

results = st.session_state.get("analysis_results")
if results:
    st.divider()
    st.header("Results")
    
    # 1. Pairwise choices first: show the human-readable winner, not just A/B.
    st.subheader("2 · Pairwise choices")
    st.caption("Each row answers: between these two alternatives, which one did the model prefer?")
    preference_rows = []
    for item in results.get("preferences", []):
        preference = str(item.get("preference", "")).strip().upper()
        option_a = item.get("option_a")
        option_b = item.get("option_b")
        if preference == "A":
            selected = option_a
            status = "Valid choice"
        elif preference == "B":
            selected = option_b
            status = "Valid choice"
        else:
            selected = "No valid A/B choice"
            status = "Needs review"
        preference_rows.append({
            "Option A": option_a,
            "Option B": option_b,
            "Model selected": selected,
            "Raw response": preference or "EMPTY",
            "Status": status,
        })
    if preference_rows:
        st.dataframe(pd.DataFrame(preference_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No pairwise choices were returned.")

    # 2. Graph with selectable perturbation view and highlighted changed edges.
    st.subheader("3 · Preference graph")
    st.caption(
        "Each arrow points from the preferred alternative to the less-preferred one. "
        "Yellow edges indicate a preference direction that changed relative to baseline."
    )
    graph = results.get("graph")
    baseline_edges = list(graph.edges()) if graph is not None and hasattr(graph, "edges") else []
    nodes = list(graph.nodes()) if graph is not None and hasattr(graph, "nodes") else []
    perturbations = results.get("perturbations", []) or []

    # Build labels for each available graph view.
    view_labels = ["Baseline"]
    prompt_view_lookup = {}
    if perturbations:
        view_labels.append("Reversed option order")
        max_prompts = max(
            (len(item.get("paraphrase_results", []) or []) for item in perturbations),
            default=0,
        )
        for prompt_idx in range(max_prompts):
            paraphrase = next((
                (item.get("paraphrase_results", []) or [])[prompt_idx].get("paraphrase", "")
                for item in perturbations
                if len(item.get("paraphrase_results", []) or []) > prompt_idx
            ), "")
            label = f"Prompt variation {prompt_idx + 1}"
            if paraphrase:
                label += f": {paraphrase}"
            view_labels.append(label)
            prompt_view_lookup[label] = prompt_idx

    selected_view = st.selectbox(
        "Graph view",
        view_labels,
        help="Switch between original choices, reversed option order, and each tested rephrased prompt.",
        key="preference_graph_view",
    )

    # Use the per-pair perturbation records to rebuild the selected graph.
    baseline_winners = {}
    active_winners = {}
    pair_details = {}
    for item in perturbations:
        pair_key = (item.get("option_a"), item.get("option_b"))
        baseline_winners[pair_key] = item.get("baseline_choice")
        pair_details[pair_key] = item
        if selected_view == "Baseline":
            active_winners[pair_key] = item.get("baseline_choice")
        elif selected_view == "Reversed option order":
            active_winners[pair_key] = item.get("reversed_choice")
        else:
            p_idx = prompt_view_lookup.get(selected_view, 0)
            p_results = item.get("paraphrase_results", []) or []
            if len(p_results) > p_idx:
                # The forward-order judgment defines the displayed prompt graph;
                # the side panel below also exposes the reversed-order judgment.
                active_winners[pair_key] = p_results[p_idx].get("forward_choice")
            else:
                active_winners[pair_key] = item.get("baseline_choice")

    # If the pipeline did not return perturbation details, keep its baseline graph.
    if not active_winners:
        for item in results.get("preferences", []):
            a, b = item.get("option_a"), item.get("option_b")
            raw = str(item.get("preference", "")).strip().upper()
            active_winners[(a, b)] = a if raw == "A" else b if raw == "B" else None
            baseline_winners[(a, b)] = active_winners[(a, b)]
            if a not in nodes:
                nodes.append(a)
            if b not in nodes:
                nodes.append(b)

    active_edges = []
    edge_is_changed = {}
    for pair_key, winner in active_winners.items():
        a, b = pair_key
        if winner not in (a, b):
            continue
        loser = b if winner == a else a
        edge = (winner, loser)
        active_edges.append(edge)
        edge_is_changed[edge] = (
            selected_view != "Baseline" and winner != baseline_winners.get(pair_key)
        )

    # For a baseline view, use the actual graph edges to preserve all original graph details.
    if selected_view == "Baseline" and not active_winners:
        active_edges = baseline_edges

    if nodes:
        # Compact graph: short node labels, full alternative text on hover only.
        graph_for_layout = nx.DiGraph()
        graph_for_layout.add_nodes_from(nodes)
        graph_for_layout.add_edges_from(active_edges)
        positions = nx.spring_layout(graph_for_layout, seed=17, k=1.4)
        short_label = {node: f"Option {i + 1}" for i, node in enumerate(nodes)}
        fig = go.Figure()

        # Draw arrows without text labels; hover targets carry edge explanations.
        midpoint_x, midpoint_y, midpoint_hover = [], [], []
        for source, target in active_edges:
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            changed = edge_is_changed.get((source, target), False)
            edge_color = "#EAB308" if changed else "#475569"
            edge_width = 4.5 if changed else 2.3
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color=edge_color, width=edge_width),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.7, arrowwidth=edge_width,
                arrowcolor=edge_color, standoff=24, startstandoff=24,
            )
            midpoint_x.append((x0 + x1) / 2)
            midpoint_y.append((y0 + y1) / 2)
            status = "CHANGED vs baseline" if changed else "Unchanged vs baseline"
            midpoint_hover.append(
                f"Preferred: {source}<br>Less preferred: {target}<br>{status}"
            )

        # Invisible hover points near edge midpoints: details appear only on hover.
        if midpoint_x:
            fig.add_trace(go.Scatter(
                x=midpoint_x, y=midpoint_y, mode="markers",
                marker=dict(size=18, color="rgba(0,0,0,0.01)"),
                hovertext=midpoint_hover, hoverinfo="text", showlegend=False,
            ))

        node_x = [positions[node][0] for node in nodes]
        node_y = [positions[node][1] for node in nodes]
        node_hover = [f"{short_label[node]}<br><br>Alternative:<br>{node}" for node in nodes]
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            text=[short_label[node] for node in nodes], textposition="middle center",
            textfont=dict(size=12, color="#0F172A"),
            marker=dict(size=34, color="#DCE7F3", line=dict(width=1.8, color="#334155")),
            hovertext=node_hover, hoverinfo="text", showlegend=False,
        ))
        fig.update_layout(
            height=max(460, 110 * len(nodes)), margin=dict(l=30, r=30, t=30, b=30),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor="white", paper_bgcolor="white",
            hoverlabel=dict(bgcolor="#111827", font=dict(color="white", size=13)),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Hover over a node to read the full alternative. Hover near an arrow to inspect the preference and whether it changed.")
        st.markdown(
            "<span style='color:#475569;font-weight:600'>━━</span> Unchanged edge &nbsp;&nbsp; "
            "<span style='color:#EAB308;font-weight:700'>━━</span> Changed preference edge",
            unsafe_allow_html=True,
        )
        if active_edges:
            with st.expander("View edge details"):
                st.dataframe(pd.DataFrame([
                    {
                        "Preferred alternative (arrow starts here)": source,
                        "Less preferred alternative (arrow points here)": target,
                        "Changed vs baseline": "YES" if edge_is_changed.get((source, target), False) else "No",
                    }
                    for source, target in active_edges
                ]), use_container_width=True, hide_index=True)
        else:
            st.info("No valid directed edges were produced for this view.")
    else:
        st.info("The graph contains no preference nodes or edges.")

    if selected_view != "Baseline" and perturbations:
        changed_count = sum(bool(value) for value in edge_is_changed.values())
        if changed_count:
            st.warning(f"{changed_count} edge(s) changed direction compared with the baseline. Yellow arrows mark the changed preference(s).")
        else:
            st.success("No edge direction changed compared with the baseline in this view.")

    # 3. Explain cycle result in plain language.
    st.subheader("4 · Logical consistency")
    consistency = results.get("consistency") or {}
    is_consistent = consistency.get("is_consistent")
    cycles = consistency.get("cycles", []) or []
    if is_consistent is True:
        st.success("No cycle was detected. The current graph has no recorded directed preference loop.")
    elif is_consistent is False:
        st.warning("A cycle was detected. Following the preference arrows can lead back to an earlier alternative.")
    else:
        st.error("The cycle result is unavailable; inspect the returned consistency data.")
    if cycles:
        for index, cycle in enumerate(cycles, start=1):
            if cycle:
                st.write(f"Cycle {index}: " + " → ".join(map(str, [*cycle, cycle[0]])))
    st.caption("No cycle does not mean every choice is correct; it only describes the graph's structure.")

    # 4. Stability, with clear definitions.
    st.subheader("5 · Stability")
    st.caption("Stability measures whether the model repeats the same preference under a change. It does not measure correctness.")
    stability = results.get("stability") or {}
    metric_columns = st.columns(3)
    metrics = [
        ("Order stability", stability.get("order_stability_rate"), "Same alternative chosen after swapping the two options."),
        ("Prompt stability", stability.get("prompt_stability_rate"), "Same alternative chosen after rephrasing the question."),
        ("Overall stability", stability.get("overall_stability_rate"), "Same preference under the implemented perturbations."),
    ]
    for column, (label, value, help_text) in zip(metric_columns, metrics):
        with column:
            column.metric(label, "N/A" if value is None else f"{float(value):.2f}%", help=help_text)
    comparable_rate = stability.get("comparable_prompt_stability_rate")
    st.metric(
        "Prompt stability (comparable pairs)",
        "N/A" if comparable_rate is None else f"{float(comparable_rate):.2f}%",
        help="Calculated only for pairs where prompt comparisons can be meaningfully matched.",
    )

    # Show exactly how the question was rephrased and how the model responded.
    st.subheader("6 · Prompt sensitivity: see the changed prompt")
    st.caption(
        "Expand a comparison to see the original question, the exact rephrased prompt sent for each test, "
        "and the model's choices in both option orders. The alternatives and context are held constant."
    )
    perturbations = results.get("perturbations", []) or []
    if perturbations:
        for pair_index, item in enumerate(perturbations, start=1):
            pair_title = f"{pair_index}. {item.get('option_a')}  vs  {item.get('option_b')}"
            with st.expander(pair_title):
                st.markdown("**Original question**")
                st.code(question.strip(), language=None)
                st.markdown("**Baseline choice**")
                st.write(f"Selected: **{item.get('baseline_choice', 'Unavailable')}**")
                st.write(f"After swapping option order: **{item.get('reversed_choice', 'Unavailable')}**")
                paraphrase_results = item.get("paraphrase_results", []) or []
                if paraphrase_results:
                    for p_index, p_item in enumerate(paraphrase_results, start=1):
                        st.markdown(f"**Prompt variation {p_index}**")
                        # This mirrors App/perturbations.py::_evaluation_question.
                        label = "Original request" if "response" in str(results.get("question_style", "decision")).lower() else "Decision question"
                        changed_prompt = f"{label}:\n{question.strip()}\n\n{p_item.get('paraphrase', '')}"
                        st.markdown("Changed prompt sent to the model")
                        st.code(changed_prompt, language=None)
                        left, right = st.columns(2)
                        with left:
                            st.markdown("**A/B order**")
                            st.write(f"A = {item.get('option_a')}")
                            st.write(f"B = {item.get('option_b')}")
                            st.write(f"Raw response: `{p_item.get('forward_preference', 'N/A')}`")
                            st.write(f"Chosen: **{p_item.get('forward_choice', 'N/A')}**")
                        with right:
                            st.markdown("**Reversed order**")
                            st.write(f"A = {item.get('option_b')}")
                            st.write(f"B = {item.get('option_a')}")
                            st.write(f"Raw response: `{p_item.get('reverse_preference', 'N/A')}`")
                            st.write(f"Chosen: **{p_item.get('reverse_choice', 'N/A')}**")
                        if p_item.get("prompt_comparable"):
                            if p_item.get("prompt_stable"):
                                st.success("Prompt-stable: same underlying preference as baseline.")
                            else:
                                st.warning("Prompt-sensitive: preference changed from baseline.")
                        else:
                            st.info("Not comparable: baseline or perturbed choice changes with option order.")
                        st.divider()
                else:
                    st.info(
                        "This result contains only the first rephrased choice, not the individual prompt records. "
                        "The exact tested prompt variations are not available in this result object."
                    )
    else:
        st.info("No perturbation-level results were returned.")

    with st.expander("Read the explanation report"):
        explanation = results.get("explanation", "No explanation report was returned.")
        st.text(explanation)
        st.download_button(
            label="Download analysis report",
            data=str(explanation),
            file_name="preference_stability_report.txt",
            mime="text/plain",
        )

    # Final choice and ranking
    st.subheader("Final Choice and Ranking")

    ranking_result = results.get("ranking", {})
    final_choice = ranking_result.get("final_choice")
    ranking_status = ranking_result.get("status")

    if ranking_status == "winner" and final_choice is not None:
        st.success(f"Final choice: {final_choice}")

    elif ranking_status == "tie":
        st.warning(
            "No unique final choice: the highest-ranked "
            "alternatives are tied."
        )

    else:
        st.warning(
            "No reliable winner: no pairwise comparisons "
            "produced a stable aggregated verdict."
        )

    # Build the ranking table
    ranking_rows = []

    for row in ranking_result.get("ranking", []):
        ranking_rows.append({
            "Rank": row["rank"],
            "Alternative": row["alternative"],
            "Wins": row["wins"],
            "Losses": row["losses"],
            "Copeland Score": row["score"],
            "Resolved Comparisons": row["evaluated_pairs"],
        })

    if ranking_rows:
        st.dataframe(
            pd.DataFrame(ranking_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Copeland score = wins minus losses. "
            "Unresolved pairwise comparisons are excluded."
        )

        st.markdown("**Reasoning behind the final choice**")
        st.write(
            ranking_result.get(
                "explanation",
                "No ranking explanation is available.",
            )
        )

    else:
        st.info(
            "A ranking is not available for this analysis."
        )
    
    # Answer verification
    st.subheader("Answer Verification")
    verification = results.get("verification") or {}

    verification_status = verification.get(
        "status", "UNABLE_TO_VERIFY"
    )
    verification_method = verification.get("method", "unknown")
    selected_answer = verification.get("selected_answer")
    expected_answer = verification.get("expected_answer")
    verification_explanation = verification.get(
        "explanation", "No verification explanation available."
    )

    if verification_status == "CORRECT":
        st.success("Verification status: CORRECT")
    elif verification_status == "INCORRECT":
        st.error("Verification status: INCORRECT")
    else:
        st.warning("Verification status: UNABLE TO VERIFY")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Verification method**")
        st.write(verification_method)

        st.markdown("**Selected answer**")
        st.write(
            selected_answer
            if selected_answer is not None
            else "No final answer selected"
        )

    with col2:
        st.markdown("**Reference / independently derived answer**")
        st.write(
            expected_answer
            if expected_answer is not None
            else "Unavailable"
        )

    st.markdown("**Verification explanation**")
    st.write(verification_explanation)

    st.caption(
        "Verification evaluates the selected answer separately. "
        "It does not modify the Copeland ranking."
    )
