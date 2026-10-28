
import os
import re

import ollama

MODEL = "qwen2.5:3b"
#GEMINI_MODEL = "gemini-2.5-flash"
MAX_ATTEMPTS = 2

PROVIDER = os.getenv("PREFERENCE_PROVIDER", "ollama").lower()

#_gemini_client = None

"""
def _get_gemini_client():
    global _gemini_client

    if _gemini_client is None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set the GEMINI_API_KEY environment variable."
            )

        _gemini_client = genai.Client(api_key=api_key)

    return _gemini_client
"""

def _parse_choice(response_text):
    text = response_text.strip().upper()

    # Accept A, B, Option A, Option B, and short explanations.
    match = re.fullmatch(
        r"\s*(?:OPTION\s*)?([AB])"
        r"(?:\s*[\.\):,\-].*)?\s*",
        text,
    )

    return match.group(1) if match else None


def _ask_ollama(prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0,
            "num_predict": 10,
        },
    )

    return response["message"]["content"]

"""
def _ask_gemini(prompt):
    from google.genai import types

    response = _get_gemini_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=10,
        ),
    )

    return response.text or ""
"""
def generate_preference(
    question,
    option_a,
    option_b,
    context="",
):
    prompt = f"""
You are evaluating two alternatives.

Question:
{question}

Context:
{context if context else "No additional context."}

Option A:
{option_a}

Option B:
{option_b}

Compare the alternatives using the question's criteria.
Judge their content, not their position or letter.
Apply the same standards regardless of option order.

Return exactly one letter: A or B.
Do not provide an explanation.
"""

    for attempt in range(MAX_ATTEMPTS):
        try:
            """if PROVIDER == "gemini":
                raw_response = _ask_gemini(prompt)
                """
            if PROVIDER == "ollama":
                raw_response = _ask_ollama(prompt)
            else:
                raise ValueError(
                    f"Unsupported PREFERENCE_PROVIDER: {PROVIDER}"
                )

            choice = _parse_choice(raw_response)

            if choice:
                return choice

            print(
                f"Invalid model response "
                f"(attempt {attempt + 1}): {raw_response!r}"
            )

        except Exception as exc:
            if attempt == MAX_ATTEMPTS - 1:
                raise RuntimeError(
                    f"{PROVIDER} preference evaluation failed: {exc}"
                ) from exc

    raise ValueError("Model failed to return a valid A/B choice.")