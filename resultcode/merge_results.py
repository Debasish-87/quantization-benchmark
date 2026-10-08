import csv
import json
from pathlib import Path

BASE = Path("merged-results")

performance_file = BASE / "performance" / "benchmark_results.csv"
accuracy_file = BASE / "accuracy" / "results" / "accuracy" / "gsm8k_final_summary.json"

output_dir = BASE / "combined"
output_dir.mkdir(parents=True, exist_ok=True)

# Load performance results
with open(performance_file, newline="", encoding="utf-8") as f:
    performance = list(csv.DictReader(f))

# Load GSM8K results
with open(accuracy_file, encoding="utf-8") as f:
    accuracy = json.load(f)

accuracy_map = {
    row["quantization"]: row
    for row in accuracy
}

# Merge accuracy into every performance row
combined = []

for row in performance:
    q = row["quantization"]
    acc = accuracy_map[q]

    merged = dict(row)
    merged["gsm8k_accuracy"] = acc["accuracy"]
    merged["gsm8k_accuracy_percent"] = acc["accuracy_percent"]
    merged["gsm8k_samples"] = acc["samples"]
    merged["gsm8k_correct"] = acc["correct"]
    merged["gsm8k_delta_vs_fp16_pp"] = acc["accuracy_delta_pp_vs_fp16"]

    combined.append(merged)

# Save CSV
csv_file = output_dir / "combined_results.csv"

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=combined[0].keys())
    writer.writeheader()
    writer.writerows(combined)

# Save JSON
json_file = output_dir / "combined_results.json"

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2)

print("=" * 80)
print("MERGE COMPLETE")
print("=" * 80)
print(f"Performance rows : {len(performance)}")
print(f"Combined rows    : {len(combined)}")
print()
print("GSM8K:")
for q, acc in accuracy_map.items():
    print(
        f"{q:>5} : "
        f"{acc['accuracy_percent']:.2f}% "
        f"({acc['correct']}/{acc['samples']})"
    )

print()
print("Created:")
print(csv_file)
print(json_file)
