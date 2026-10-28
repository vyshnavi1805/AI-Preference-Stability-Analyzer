"""Tests for App/verification.py.

Run from the project root:
    python test_verification.py

Dataset tests use data.csv and do not require an API call.
Live Groq checks are optional:
    python test_verification.py --live-groq

Requires GROQ_API_KEY and the `groq` package for live checks.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from App.verification import (
    compare_answers,
    load_answer_key,
    normalize_text,
    verify_answer,
    verify_with_dataset,
)


def print_result(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"       {detail}")


def run_unit_checks():
    """Check deterministic helper behavior without calling Groq."""
    cases = [
        ("same numeric answers", "5", "5", "", True),
        ("numeric formatting", "5.0", "5", "", True),
        ("different answers", "4", "5", "", False),
        ("same text ignoring case", "Paris", "paris", "", True),
        ("different option labels", "A) 36", "B) 48",
         "A) 36; B) 48", False),
        ("label and answer text", "B) 48", "48",
         "A) 36; B) 48", True),
        ("missing selected answer", None, "5", "", False),
        ("missing expected answer", "5", None, "", False),
    ]

    passed_count = 0
    for name, selected, expected, options, wanted in cases:
        try:
            actual = compare_answers(selected, expected, options)
            passed = actual is wanted
            print_result(
                f"compare_answers: {name}",
                passed,
                f"expected={wanted}, actual={actual}",
            )
        except Exception as exc:
            passed = False
            print_result(f"compare_answers: {name}", False, str(exc))
        passed_count += int(passed)

    # Whitespace/case normalization is a deterministic helper check.
    try:
        actual = normalize_text("  What   IS  this? ")
        wanted = "what is this?"
        passed = actual == wanted
        print_result(
            "normalize_text: whitespace and case",
            passed,
            f"expected={wanted!r}, actual={actual!r}",
        )
    except Exception as exc:
        passed = False
        print_result("normalize_text: whitespace and case", False, str(exc))
    passed_count += int(passed)

    return passed_count, len(cases) + 1


def run_dataset_checks(answer_key):
    """Exercise verification against an answer already in the CSV."""
    if not answer_key:
        print_result(
            "dataset verification",
            False,
            "data.csv contains no usable question/answer records.",
        )
        return 0, 1

    # Use a real row from the loaded dataset rather than inventing a question.
    record = next(iter(answer_key.values()))
    question = record["question"]
    expected = record["answer"]
    options = record.get("options", "")

    passed_count = 0
    total = 2

    try:
        correct_result = verify_with_dataset(question, expected, answer_key)
        passed = (
            correct_result is not None
            and correct_result.get("status") == "CORRECT"
            and correct_result.get("method") == "dataset"
        )
        print_result(
            "CSV answer is recognized as correct",
            passed,
            f"status={correct_result.get('status') if correct_result else None}, "
            f"method={correct_result.get('method') if correct_result else None}",
        )
    except Exception as exc:
        passed = False
        print_result("CSV answer is recognized as correct", False, str(exc))
    passed_count += int(passed)

    try:
        # Choose a deliberately different value for the simple case where the
        # stored answer is numeric; otherwise use a distinct sentinel string.
        wrong_answer = "__deliberately_different_test_answer__"
        wrong_result = verify_with_dataset(question, wrong_answer, answer_key)
        passed = (
            wrong_result is not None
            and wrong_result.get("status") == "INCORRECT"
            and wrong_result.get("method") == "dataset"
        )
        print_result(
            "different answer is rejected by CSV",
            passed,
            f"status={wrong_result.get('status') if wrong_result else None}, "
            f"method={wrong_result.get('method') if wrong_result else None}",
        )
    except Exception as exc:
        passed = False
        print_result("different answer is rejected by CSV", False, str(exc))
    passed_count += int(passed)

    return passed_count, total


def run_fallback_checks(answer_key, live_groq=False):
    """Check that unknown questions take the Groq fallback path."""
    unknown_question = (
        "Test-only question not present in the CSV: "
        "what is 19 multiplied by 23?"
    )

    # Ensure this exact test question is not accidentally present in the file.
    if normalize_text(unknown_question) in answer_key:
        unknown_question += " Use the alternate wording for this test."

    if not live_groq:
        # Verify routing with no network call by temporarily removing the key.
        # The production function should return UNABLE_TO_VERIFY when no key
        # is configured. Restore the original environment afterward.
        original_key = os.environ.get("GROQ_API_KEY")
        try:
            os.environ.pop("GROQ_API_KEY", None)
            result = verify_answer(
                question=unknown_question,
                selected_answer="437",
                answer_key=answer_key,
                options="",
            )
            passed = (
                result.get("status") == "UNABLE_TO_VERIFY"
                and result.get("method") == "groq"
                and "GROQ_API_KEY" in result.get("explanation", "")
            )
            print_result(
                "unknown question uses Groq fallback gracefully without key",
                passed,
                f"status={result.get('status')}, method={result.get('method')}",
            )
            return int(passed), 1
        except Exception as exc:
            print_result(
                "unknown question uses Groq fallback gracefully without key",
                False,
                str(exc),
            )
            return 0, 1
        finally:
            if original_key is not None:
                os.environ["GROQ_API_KEY"] = original_key
    else:
        if not os.getenv("GROQ_API_KEY"):
            print_result(
                "live Groq fallback",
                False,
                "GROQ_API_KEY is not set in this terminal.",
            )
            return 0, 1

        try:
            result = verify_answer(
                question=unknown_question,
                selected_answer="437",
                answer_key=answer_key,
                options="",
            )
            status = result.get("status")
            passed = (
                result.get("method") == "groq"
                and status in {"CORRECT", "INCORRECT", "UNABLE_TO_VERIFY"}
            )
            print_result(
                "live Groq fallback returns a valid result structure",
                passed,
                f"status={status}, method={result.get('method')}, "
                f"expected_answer={result.get('expected_answer')!r}",
            )
            return int(passed), 1
        except Exception as exc:
            print_result("live Groq fallback", False, str(exc))
            return 0, 1


def main():
    parser = argparse.ArgumentParser(description="Test answer verification.")
    parser.add_argument(
        "--live-groq",
        action="store_true",
        help="Make a real Groq API call for an unknown question.",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("AI PREFERENCE STABILITY ANALYZER — VERIFICATION TESTS")
    print("=" * 64)

    try:
        answer_key = load_answer_key()
        print_result(
            "CSV answer key loads",
            bool(answer_key),
            f"{len(answer_key)} usable question(s) loaded.",
        )
        csv_passed = int(bool(answer_key))
        csv_total = 1
    except Exception as exc:
        print_result("CSV answer key loads", False, str(exc))
        print("\nCannot continue without a readable data.csv answer key.")
        raise SystemExit(1)

    unit_passed, unit_total = run_unit_checks()
    dataset_passed, dataset_total = run_dataset_checks(answer_key)
    fallback_passed, fallback_total = run_fallback_checks(
        answer_key,
        live_groq=args.live_groq,
    )

    passed = csv_passed + unit_passed + dataset_passed + fallback_passed
    total = csv_total + unit_total + dataset_total + fallback_total

    print("\n" + "=" * 64)
    print(f"RESULT: {passed}/{total} checks passed")
    if passed == total:
        print("All checks passed.")
    else:
        print("One or more checks failed. Review the failure messages above.")
    print("=" * 64)

    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
