import csv
import json
import re
from pathlib import Path

source = Path("data.csv")
output = Path("experiments/input/dataset_58.json")
output.parent.mkdir(parents=True, exist_ok=True)

records = []
seen_questions = set()

with source.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    required = {"Question", "Options", "Answer"}
    if not required.issubset(set(reader.fieldnames or [])):
        raise ValueError(f"Unexpected columns: {reader.fieldnames}")

    for row_num, row in enumerate(reader, start=2):
        question = (row["Question"] or "").strip()
        raw_options = (row["Options"] or "").strip()
        answer = (row["Answer"] or "").strip()

        # Keep the original option text, including its A/B/C/D label.
        options = [
            item.strip()
            for item in re.split(r"[;\n]+", raw_options)
            if item.strip()
        ]

        if not question or not answer:
            raise ValueError(f"Missing question/answer on row {row_num}")
        if len(options) != 4:
            raise ValueError(
                f"Expected 4 options on row {row_num}; got {len(options)}"
            )

        key = question.casefold()
        if key in seen_questions:
            raise ValueError(f"Duplicate question on row {row_num}")
        seen_questions.add(key)

        # Save the answer key as metadata; the current runner may ignore it.
        records.append({
            "scenario_id": f"DATASET_{len(records)+1:03d}",
            "question": question,
            "options": options,
            "context": "",
            "question_style": "decision",
            "reference_answer": answer
        })

with output.open("w", encoding="utf-8", newline="\n") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
    f.write("\n")

# Re-read and validate the generated JSON.
with output.open("r", encoding="utf-8") as f:
    check = json.load(f)

assert len(check) == len(records)
assert all(len(item["options"]) == 4 for item in check)

print(f"Source records converted: {len(records)}")
print(f"Output file: {output}")
print(f"Validation: passed")
print("First record:")
print(json.dumps(check[0], ensure_ascii=False, indent=2))
