import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def extract(raw, method, concurrency, repetition, memory):
    def g(k):
        return raw.get(k)

    row = {
        "quantization": method,
        "concurrency": concurrency,
        "repetition": repetition,
        "request_throughput_req_s": g("request_throughput"),
        "output_throughput_tok_s": g("output_throughput"),
        "total_token_throughput_tok_s": g("total_token_throughput"),
        "completed": g("completed"),
        "failed": g("failed"),
        "mean_ttft_ms": g("mean_ttft_ms"),
        "p50_ttft_ms": g("p50_ttft_ms"),
        "p95_ttft_ms": g("p95_ttft_ms"),
        "p99_ttft_ms": g("p99_ttft_ms"),
        "mean_tpot_ms": g("mean_tpot_ms"),
        "p50_tpot_ms": g("p50_tpot_ms"),
        "p95_tpot_ms": g("p95_tpot_ms"),
        "p99_tpot_ms": g("p99_tpot_ms"),
        "mean_itl_ms": g("mean_itl_ms"),
        "p50_itl_ms": g("p50_itl_ms"),
        "p95_itl_ms": g("p95_itl_ms"),
        "p99_itl_ms": g("p99_itl_ms"),
        "mean_e2el_ms": g("mean_e2el_ms"),
        "p50_e2el_ms": g("p50_e2el_ms"),
        "p95_e2el_ms": g("p95_e2el_ms"),
        "p99_e2el_ms": g("p99_e2el_ms"),
        "error_rate": (
            (g("failed") or 0) / ((g("completed") or 0) + (g("failed") or 0))
            if (g("completed") or 0) + (g("failed") or 0) else 0
        ),
    }
    row.update(memory)
    return row


def write_outputs(rows, env, accuracies, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "benchmark_results.csv", index=False)

    with open(out / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    with open(out / "accuracy.json", "w", encoding="utf-8") as f:
        json.dump(accuracies, f, indent=2)

    # Stable aggregate across repetitions.
    numeric = [
        c for c in df.columns
        if c not in {"quantization", "concurrency", "repetition"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not df.empty:
        agg = df.groupby(["quantization", "concurrency"])[numeric].agg(["mean", "std"])
        agg.to_csv(out / "benchmark_aggregate.csv")

    _plot(df, out, "output_throughput_tok_s", "Output throughput (tok/s)", "throughput.png")
    _plot(df, out, "p50_ttft_ms", "P50 TTFT (ms)", "ttft_p50.png")
    _plot(df, out, "p95_ttft_ms", "P95 TTFT (ms)", "ttft_p95.png")
    _plot(df, out, "p50_tpot_ms", "P50 TPOT (ms)", "tpot_p50.png")
    _plot(df, out, "p95_e2el_ms", "P95 E2E latency (ms)", "e2e_p95.png")
    _plot(df, out, "gpu_0_peak_vram_mb", "Peak GPU 0 VRAM (MB)", "vram.png")

    return df


def _plot(df, out, column, ylabel, filename):
    if df.empty or column not in df.columns:
        return
    plt.figure(figsize=(9, 5))
    for method, g in df.groupby("quantization"):
        g = g.sort_values("concurrency")
        plt.plot(g["concurrency"], g[column], marker="o", label=method)
    plt.xlabel("Concurrency")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs concurrency")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / filename, dpi=160)
    plt.close()
