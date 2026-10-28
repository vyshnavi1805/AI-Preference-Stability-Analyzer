import json
from pathlib import Path

source = Path("experiments/input/dataset_58.json")
output = Path("experiments/input/dataset_20.json")

with source.open("r", encoding="utf-8") as f:
    records = json.load(f)

subset = records[:20]

with output.open("w", encoding="utf-8", newline="\n") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)
    f.write("\n")

print("Subset records:", len(subset))
print("First scenario:", subset[0]["scenario_id"])
print("Last scenario:", subset[-1]["scenario_id"])
print("Reference answers retained:", all("reference_answer" in r for r in subset))
