
import csv
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

# Assumes data.csv is in the project root, alongside main.py.
DATASET_PATH = Path(__file__).resolve().parent.parent / "data.csv"

QUESTION_COLUMN = "Question"
OPTIONS_COLUMN = "Options"
ANSWER_COLUMN = "Answer"

GROQ_MODEL = "openai/gpt-oss-120b"


def normalize_text(value):
    """Normalize whitespace and case for text comparisons."""
    if value is None:
        return ""
    return " ".join(str(value).strip().casefold().split())


def normalize_answer(value):
    """Normalize simple numeric answers, including optional units."""
    text = normalize_text(value)

    if not text:
        return None

    match = re.fullmatch(
        r"([-+]?\d+(?:\.\d+)?)"
        r"(?:\s+[a-z][a-z\s/]*)?",
        text,
    )

    if match:
        try:
            return ("number", Decimal(match.group(1)).normalize())
        except InvalidOperation:
            pass

    return ("text", text)


def extract_option_label(value):
    """Extract labels from answers such as 'B', 'B) 48', or 'B. 48'."""
    if value is None:
        return None

    match = re.match(
        r"^\s*([A-Z])(?:\s*[\).:]|\s*$)",
        str(value),
        re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


def load_answer_key(dataset_path=DATASET_PATH):
    """Load the trusted question, options, and answer records from CSV."""
    answer_key = {}

    with open(dataset_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter="\t")

        required = {
            QUESTION_COLUMN,
            OPTIONS_COLUMN,
            ANSWER_COLUMN,
        }
        columns = set(reader.fieldnames or [])

        if not required.issubset(columns):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}. "
                f"Found: {reader.fieldnames}"
            )

        for line_number, row in enumerate(reader, start=2):
            question = (row.get(QUESTION_COLUMN) or "").strip()
            options = (row.get(OPTIONS_COLUMN) or "").strip()
            answer = (row.get(ANSWER_COLUMN) or "").strip()

            if not question or not answer:
                continue

            key = normalize_text(question)

            if key in answer_key:
                raise ValueError(
                    f"Duplicate question in CSV near line {line_number}: "
                    f"{question}"
                )

            answer_key[key] = {
                "question": question,
                "options": options,
                "answer": answer,
            }

    return answer_key



def compare_answers(selected_answer, expected_answer, options=""):
    """Compare answer text and multiple-choice option labels."""

    if selected_answer is None or expected_answer is None:
        return False

    selected = normalize_answer(selected_answer)
    expected = normalize_answer(expected_answer)

    if selected is not None and selected == expected:
        return True

    # Parse options such as: A) 36; B) 48; C) 52; D) 60
    option_map = {}
    for item in re.split(r"[;\n]+", options or ""):
        match = re.match(
            r"^\s*([A-Z])\s*[\).:]\s*(.*?)\s*$",
            item,
            re.IGNORECASE,
        )
        if match:
            option_map[match.group(1).upper()] = match.group(2).strip()

    selected_label = extract_option_label(selected_answer)
    expected_label = extract_option_label(expected_answer)

    # If both values include labels, verify both the label and its text.
    if selected_label and expected_label:
        if selected_label != expected_label:
            return False

        selected_text = option_map.get(selected_label)
        expected_text = option_map.get(expected_label)

        if selected_text and expected_text:
            return normalize_answer(selected_text) == normalize_answer(expected_text)

        return normalize_text(selected_answer) == normalize_text(expected_answer)

    # Resolve a single option label to its text when possible.
    selected_text = option_map.get(selected_label) if selected_label else None
    expected_text = option_map.get(expected_label) if expected_label else None

    selected_value = normalize_answer(selected_text or selected_answer)
    expected_value = normalize_answer(expected_text or expected_answer)

    return selected_value is not None and selected_value == expected_value


def verify_with_dataset(question, selected_answer, answer_key):
    """Return a verdict using a trusted CSV reference answer."""
    if not isinstance(question, str) or not question.strip():
        return {
            "status": "UNABLE_TO_VERIFY",
            "method": "dataset",
            "expected_answer": None,
            "selected_answer": selected_answer,
            "explanation": "The question is missing.",
        }

    record = answer_key.get(normalize_text(question))

    if record is None:
        return None  # Signal that Groq fallback may be used.

    expected = record["answer"]

    if selected_answer is None or not str(selected_answer).strip():
        return {
            "status": "UNABLE_TO_VERIFY",
            "method": "dataset",
            "expected_answer": expected,
            "selected_answer": selected_answer,
            "options": record["options"],
            "explanation": "The analyzer did not produce a final answer.",
        }

    correct = compare_answers(
        selected_answer,
        expected,
        record["options"],
    )

    return {
        "status": "CORRECT" if correct else "INCORRECT",
        "method": "dataset",
        "expected_answer": expected,
        "selected_answer": selected_answer,
        "options": record["options"],
        "explanation": (
            "The selected answer matches the trusted CSV reference."
            if correct
            else "The selected answer does not match the trusted CSV reference."
        ),
    }



def verify_with_groq(question, selected_answer, options=""):
    """Independently verify an answer using Groq."""

    if selected_answer is None or not str(selected_answer).strip():
        return {
            "status": "UNABLE_TO_VERIFY",
            "method": "groq",
            "expected_answer": None,
            "selected_answer": selected_answer,
            "explanation": "The analyzer did not produce a final answer.",
        }

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "status": "UNABLE_TO_VERIFY",
            "method": "groq",
            "expected_answer": None,
            "selected_answer": selected_answer,
            "explanation": "GROQ_API_KEY is not configured.",
        }

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        prompt = f"""
You are an independent answer verifier.

Solve the question independently before evaluating the proposed answer.
Do not assume the proposed answer is correct.

Question:
{question}

Options:
{options or "Not supplied"}

Proposed answer:
{selected_answer}

Return only a valid JSON object with:
- verdict: CORRECT, INCORRECT, or UNABLE_TO_VERIFY
- expected_answer: independently derived answer, or null
- explanation: brief reasoning

Use UNABLE_TO_VERIFY if the question is ambiguous or cannot be
answered reliably. Do not guess.
"""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or ""
        data = json.loads(content)

        verdict = str(data.get("verdict", "")).strip().upper()
        expected = data.get("expected_answer")

        if verdict not in {"CORRECT", "INCORRECT", "UNABLE_TO_VERIFY"}:
            verdict = "UNABLE_TO_VERIFY"

        if expected is None and verdict != "UNABLE_TO_VERIFY":
            verdict = "UNABLE_TO_VERIFY"

        return {
            "status": verdict,
            "method": "groq",
            "expected_answer": expected,
            "selected_answer": selected_answer,
            "explanation": str(
                data.get("explanation", "No explanation provided.")
            ),
        }

    except Exception as exc:
        return {
            "status": "UNABLE_TO_VERIFY",
            "method": "groq",
            "expected_answer": None,
            "selected_answer": selected_answer,
            "explanation": f"Groq verification failed: {exc}",
        }


def verify_answer(
    question,
    selected_answer,
    answer_key=None,
    options="",
):
    """
    Verify after ranking:
    1. Use the trusted CSV answer if the question exists.
    2. Otherwise, use GROQ as an independent fallback.
    """
    if answer_key is None:
        try:
            answer_key = load_answer_key()
        except (OSError, ValueError, csv.Error) as exc:
            return {
                "status": "UNABLE_TO_VERIFY",
                "method": "dataset",
                "expected_answer": None,
                "selected_answer": selected_answer,
                "explanation": f"Could not load the CSV answer key: {exc}",
            }

    try:
        dataset_result = verify_with_dataset(
            question,
            selected_answer,
            answer_key,
        )
    except (TypeError, AttributeError):
        dataset_result = None

    if dataset_result is not None:
        return dataset_result

    return verify_with_groq(
        question,
        selected_answer,
        options,
    )
