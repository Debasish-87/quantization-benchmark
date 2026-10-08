import csv
from pathlib import Path
import matplotlib.pyplot as plt

INPUT = Path("merged-results/combined/combined_results.csv")
OUT = Path("merged-results/combined/plots")
OUT.mkdir(parents=True, exist_ok=True)

with open(INPUT, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

methods = ["fp16", "int8", "int4"]
labels = {"fp16": "FP16", "int8": "INT8", "int4": "INT4"}

def data_for(method):
    return [r for r in rows if r["quantization"] == method]

# Throughput
plt.figure()
for method in methods:
    data = data_for(method)
    x = [int(r["concurrency"]) for r in data]
    y = [float(r["output_throughput_tok_s"]) for r in data]
    plt.plot(x, y, marker="o", label=labels[method])

plt.xlabel("Concurrency")
plt.ylabel("Output Throughput (tok/s)")
plt.title("Throughput vs Concurrency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "throughput_vs_concurrency.png", dpi=200)
plt.close()

# P95 TTFT
plt.figure()
for method in methods:
    data = data_for(method)
    x = [int(r["concurrency"]) for r in data]
    y = [float(r["p95_ttft_ms"]) for r in data]
    plt.plot(x, y, marker="o", label=labels[method])

plt.xlabel("Concurrency")
plt.ylabel("P95 TTFT (ms)")
plt.title("P95 TTFT vs Concurrency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "p95_ttft_vs_concurrency.png", dpi=200)
plt.close()

print("First 2 plots created successfully.")

# P95 E2E latency
plt.figure()

for method in methods:
    data = data_for(method)
    x = [int(r["concurrency"]) for r in data]
    y = [float(r["p95_e2el_ms"]) / 1000 for r in data]

    plt.plot(x, y, marker="o", label=labels[method])

plt.xlabel("Concurrency")
plt.ylabel("P95 E2E Latency (seconds)")
plt.title("P95 E2E Latency vs Concurrency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "p95_e2e_vs_concurrency.png", dpi=200)
plt.close()


# VRAM
plt.figure()

vram = []

for method in methods:
    data = data_for(method)
    value = max(float(r["gpu_0_peak_vram_mb"]) for r in data)
    vram.append(value / 1024)

plt.bar([labels[m] for m in methods], vram)

plt.ylabel("Peak VRAM (GB)")
plt.title("Peak GPU VRAM Usage")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "vram_by_quantization.png", dpi=200)
plt.close()


# Performance vs accuracy
plt.figure()

for method in methods:
    data = data_for(method)

    throughput = max(
        float(r["output_throughput_tok_s"])
        for r in data
    )

    accuracy = float(data[0]["gsm8k_accuracy_percent"])

    plt.scatter(
        throughput,
        accuracy,
        s=100,
        label=labels[method],
    )

    plt.annotate(
        labels[method],
        (throughput, accuracy),
        xytext=(6, 6),
        textcoords="offset points",
    )

plt.xlabel("Maximum Output Throughput (tok/s)")
plt.ylabel("GSM8K Accuracy (%)")
plt.title("Performance vs Accuracy Trade-off")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "performance_vs_accuracy.png", dpi=200)
plt.close()

print("All 5 plots generated successfully.")
