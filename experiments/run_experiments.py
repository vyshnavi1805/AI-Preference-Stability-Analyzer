import argparse
import csv
import json
import httpx
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import run_pipeline


SUMMARY_FIELDS = [
    "scenario_id",
    "status",
    "num_options",
    "num_pairs",
    "is_consistent",
    "cycle_triads",
    "order_stability_rate",
    "prompt_stability_rate",
    "overall_stability_rate",
    "human_winner",
    "consistent_predicted_winner",
    "consistent_human_agreement",
    "aggregated_predicted_winner",
    "aggregated_human_agreement",
    "predicted_winner",
    "human_agreement",
    "error",
]


def load_records(input_path):
    """Load a JSON array, JSON object, or JSONL dataset."""
    path = Path(input_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    content = path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError("The dataset file is empty.")

    try:
        data = json.loads(content)

    except json.JSONDecodeError:
        records = []

        for line_number, line in enumerate(
            content.splitlines(), start=1
        ):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

        return records

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "records" in data:
            records = data["records"]

            if not isinstance(records, list):
                raise ValueError(
                    "The 'records' field must contain a JSON array."
                )

            return records

        return [data]

    raise ValueError(
        "The dataset must be a JSON array, a JSON object, "
        "or a JSONL file."
    )


def validate_record(record, index):
    """Validate and normalize a dataset record."""
    if not isinstance(record, dict):
        raise ValueError(
            f"Record {index} must be a JSON object."
        )

    question = record.get("question")
    options = record.get("options")
    context = record.get("context", "")

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            f"Record {index} needs a non-empty question."
        )

    if not isinstance(options, list) or len(options) < 2:
        raise ValueError(
            f"Record {index} needs at least two options."
        )

    if any(
        not isinstance(option, str) or not option.strip()
        for option in options
    ):
        raise ValueError(
            f"Record {index} contains an empty or invalid option."
        )

    options = [option.strip() for option in options]

    if len(set(options)) != len(options):
        raise ValueError(
            f"Record {index} contains duplicate options."
        )

    if not isinstance(context, str):
        raise ValueError(
            f"Record {index}: context must be a string."
        )

    return question.strip(), options, context.strip()


def make_failed_summary(
    scenario_id,
    error,
    human_winner=None,
):
    """Create a consistent summary for a failed record."""
    return {
        "scenario_id": scenario_id,
        "status": "failed",
        "num_options": None,
        "num_pairs": None,
        "is_consistent": None,
        "cycle_triads": None,
        "order_stability_rate": None,
        "prompt_stability_rate": None,
        "overall_stability_rate": None,
        "human_winner": human_winner,
        "consistent_predicted_winner": None,
        "consistent_human_agreement": None,
        "aggregated_predicted_winner": None,
        "aggregated_human_agreement": None,
        "predicted_winner": None,
        "human_agreement": None,
        "error": str(error),
    }


def calculate_human_agreement(
    record,
    preferences,
    perturbations=None,
    options=None,
):
    """
    Calculate human agreement for binary preference records.

    Returns two separate results:
      1. Agreement with the order-consistent baseline verdict.
      2. Agreement with the aggregated verdict across prompts.

    Human labels model_a/model_b map to the first/second supplied option.
    Agreement is undefined for multi-option records because a binary
    human label does not encode a ranking over multiple candidates.
    """
    raw_human_label = record.get("human_winner")

    human_winner = {
        "model_a": "A",
        "model_b": "B",
    }.get(raw_human_label)

    if raw_human_label in ("A", "B"):
        human_winner = raw_human_label

    if options is None:
        options = record.get("options")

    if not isinstance(options, list) or len(options) != 2:
        return {
            "human_winner": human_winner,
            "consistent_predicted_winner": None,
            "consistent_human_agreement": None,
            "aggregated_predicted_winner": None,
            "aggregated_human_agreement": None,
        }

    first_option, second_option = options

    first_pair = (
        perturbations[0]
        if perturbations
        else None
    )

    consistent_choice = (
        first_pair.get("consistent_choice")
        if isinstance(first_pair, dict)
        else None
    )

    aggregated_choice = (
        first_pair.get("aggregated_verdict")
        if isinstance(first_pair, dict)
        else None
    )

    def option_to_label(choice):
        if choice == first_option:
            return "A"

        if choice == second_option:
            return "B"

        return None

    consistent_predicted = option_to_label(
        consistent_choice
    )

    aggregated_predicted = option_to_label(
        aggregated_choice
    )

    consistent_agreement = (
        consistent_predicted == human_winner
        if (
            human_winner in ("A", "B")
            and consistent_predicted in ("A", "B")
        )
        else None
    )

    aggregated_agreement = (
        aggregated_predicted == human_winner
        if (
            human_winner in ("A", "B")
            and aggregated_predicted in ("A", "B")
        )
        else None
    )

    return {
        "human_winner": human_winner,
        "consistent_predicted_winner": consistent_predicted,
        "consistent_human_agreement": consistent_agreement,
        "aggregated_predicted_winner": aggregated_predicted,
        "aggregated_human_agreement": aggregated_agreement,
    }


def run_experiments(input_path, output_dir):
    """Run the dataset and save summary CSV and detailed JSON."""
    records = load_records(input_path)

    if not records:
        raise ValueError("No dataset records were found.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summaries = []
    detailed_results = []

    for index, record in enumerate(records, start=1):
        scenario_id = str(index)

        if isinstance(record, dict):
            supplied_id = record.get("scenario_id")

            if supplied_id is not None and str(supplied_id).strip():
                scenario_id = str(supplied_id)

        raw_human_label = (
            record.get("human_winner")
            if isinstance(record, dict)
            else None
        )

        normalized_human_winner = {
            "model_a": "A",
            "model_b": "B",
            "A": "A",
            "B": "B",
        }.get(raw_human_label)

        print(
            f"[{index}/{len(records)}] Processing {scenario_id}"
        )

        try:
            question, options, context = validate_record(
                record, index
            )

            # MT-Bench-style records use response comparison;
            # ordinary decision inputs use decision-style prompts.
            question_style = record.get(
                "question_style",
                (
                    "response"
                    if (
                        isinstance(record, dict)
                        and record.get("human_winner")
                    )
                    else "decision"
                ),
            )

            result = run_pipeline(
                question=question,
                options=options,
                context=context,
                question_style=question_style,
            )

            consistency = result["consistency"]
            stability = result["stability"]
            cycles = consistency.get("cycles", [])
            preferences = result.get("preferences", [])

            agreement = calculate_human_agreement(
                record,
                preferences,
                result.get("perturbations"),
                options,
            )

            human_winner = agreement["human_winner"]

            consistent_predicted_winner = agreement[
                "consistent_predicted_winner"
            ]

            consistent_human_agreement = agreement[
                "consistent_human_agreement"
            ]

            aggregated_predicted_winner = agreement[
                "aggregated_predicted_winner"
            ]

            aggregated_human_agreement = agreement[
                "aggregated_human_agreement"
            ]

            summary = {
                "scenario_id": scenario_id,
                "status": "success",
                "num_options": len(options),
                "num_pairs": stability.get("total_pairs", 0),
                "is_consistent": consistency.get("is_consistent"),
                "cycle_triads": consistency.get("cyclic_triads"),
                "order_stability_rate":
                    stability.get("order_stability_rate"),
                "prompt_stability_rate":
                    stability.get("prompt_stability_rate"),
                "overall_stability_rate":
                    stability.get("overall_stability_rate"),

                "human_winner": human_winner,

                "consistent_predicted_winner":
                    consistent_predicted_winner,

                "consistent_human_agreement":
                    consistent_human_agreement,

                "aggregated_predicted_winner":
                    aggregated_predicted_winner,

                "aggregated_human_agreement":
                    aggregated_human_agreement,

                # Legacy fields now refer to the aggregated verdict.
                "predicted_winner":
                    aggregated_predicted_winner,

                "human_agreement":
                    aggregated_human_agreement,

                "error": "",
            }

            details = {
                **summary,
                "question": question,
                "context": context,
                "options": options,
                "question_style": question_style,
                "preferences": preferences,

                "graph_edges": [
                    {
                        "preferred": source,
                        "less_preferred": target,
                    }
                    for source, target in result["graph"].edges()
                ],

                "cycles": cycles,
                "perturbations": result["perturbations"],
                "explanation": result["explanation"],
            }

        except Exception as error:
            connection_error = isinstance(
                error,
                (
                    ConnectionError,
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                ),
            )

            wrapped_connection_error = (
                isinstance(error, RuntimeError)
                and "Could not connect to Ollama" in str(error)
            )

            if connection_error or wrapped_connection_error:
                raise RuntimeError(
                    "Ollama is unavailable. Start Ollama and "
                    "rerun the experiments."
                ) from error

            summary = make_failed_summary(
                scenario_id,
                error,
                normalized_human_winner,
            )

            details = summary.copy()

            print(
                f"  Failed: {type(error).__name__}: {error}"
            )

        summaries.append(summary)
        detailed_results.append(details)

    # Write summary CSV.
    csv_path = output_path / "experiment_summary.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS,
        )
        writer.writeheader()
        writer.writerows(summaries)

    # Write detailed JSON.
    json_path = output_path / "experiment_details.json"

    json_path.write_text(
        json.dumps(
            detailed_results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    successful = sum(
        row["status"] == "success"
        for row in summaries
    )

    failed = len(summaries) - successful

    def report_agreement(field, title):
        evaluated = [
            row
            for row in summaries
            if (
                row["status"] == "success"
                and row.get(field) is not None
            )
        ]

        matches = sum(
            row[field] is True
            for row in evaluated
        )

        mismatches = len(evaluated) - matches

        rate = (
            matches / len(evaluated) * 100
            if evaluated
            else None
        )

        print(f"\n{title}:")
        print(f"Eligible records: {len(evaluated)}")
        print(f"Matches: {matches}")
        print(f"Mismatches: {mismatches}")

        if rate is not None:
            print(f"Agreement rate: {rate:.2f}%")
        else:
            print("Agreement rate: N/A")

    print("\nExperiment run completed.")
    print(f"Records processed: {len(records)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    report_agreement(
        "consistent_human_agreement",
        "Agreement on order-consistent baseline pairs",
    )

    report_agreement(
        "aggregated_human_agreement",
        "Agreement using aggregated verdict across prompts",
    )

    print(f"\nSummary: {csv_path}")
    print(f"Details: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run preference stability experiments."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSON or JSONL dataset file.",
    )

    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "experiments" / "results"
        ),
        help="Directory for experiment output files.",
    )

    args = parser.parse_args()

    try:
        run_experiments(
            input_path=args.input,
            output_dir=args.output,
        )

    except (
        FileNotFoundError,
        ValueError,
        OSError,
        RuntimeError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()