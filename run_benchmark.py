import json
import time
from pathlib import Path

import yaml

from benchmark.accuracy import run_accuracy
from benchmark.memory import GPUMonitor
from benchmark.results import extract, write_outputs
from benchmark.server import VLLMServer
from benchmark.system import environment
from benchmark.vllm_bench import run_vllm_bench


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out = Path(cfg["output"]["directory"])
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    env = environment()
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("=" * 72)
    print("PRODUCTION-STYLE QUANTIZATION PERFORMANCE BENCHMARK")
    print("=" * 72)
    print(json.dumps(env, indent=2))

    rows = []
    accuracies = {}
    repetitions = int(cfg["benchmark"].get("repetitions", 1))

    for method in cfg["quantization"]["methods"]:
        name = method["name"]
        print("\n" + "=" * 72)
        print(f"CONFIGURATION: {name} | {method['model']}")
        print("=" * 72)

        server = VLLMServer(cfg, method)
        server_log = raw_dir / f"server_{name}.log"

        try:
            print("Starting vLLM...")
            server.start(server_log)
            print("vLLM ready.")

            if cfg["accuracy"]["enabled"]:
                acc = run_accuracy(cfg, method)
                accuracies[name] = acc
                print(f"Accuracy smoke score: {acc['score_percent']}%")

            for rep in range(1, repetitions + 1):
                for concurrency in cfg["benchmark"]["concurrency_levels"]:
                    print(f"\n[{name}] repetition={rep} concurrency={concurrency}")
                    raw_file = raw_dir / f"{name}_r{rep}_c{concurrency}.json"

                    monitor = GPUMonitor(interval=0.20)
                    monitor.start()
                    started = time.time()
                    try:
                        raw = run_vllm_bench(
                            cfg, method, raw_file, concurrency, rep
                        )
                    finally:
                        monitor.stop()

                    wall = time.time() - started
                    mem = monitor.summary()
                    row = extract(raw, name, concurrency, rep, mem)
                    row["wall_time_s"] = round(wall, 3)

                    # Approximate benchmark compute cost only; not full notebook cost.
                    gpu_count = max(1, len(env.get("gpu", [])))
                    row["estimated_gpu_cost_usd"] = (
                        wall / 3600 * cfg["cost"]["gpu_hourly_usd"] * gpu_count
                    )
                    rows.append(row)

                    print(f"  P50 TTFT: {row['p50_ttft_ms']} ms")
                    print(f"  P95 TTFT: {row['p95_ttft_ms']} ms")
                    print(f"  P50 TPOT: {row['p50_tpot_ms']} ms")
                    print(f"  P95 E2E : {row['p95_e2el_ms']} ms")
                    print(f"  Output  : {row['output_throughput_tok_s']} tok/s")
                    print(f"  Failed  : {row['failed']}")
                    print(f"  Peak VRAM: {mem}")

        finally:
            print("Stopping vLLM...")
            server.stop()

    df = write_outputs(rows, env, accuracies, out)

    # Human-readable summary.
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    if not df.empty:
        cols = [
            "quantization", "concurrency",
            "p50_ttft_ms", "p95_ttft_ms",
            "p50_tpot_ms", "p95_e2el_ms",
            "output_throughput_tok_s",
            "gpu_0_peak_vram_mb", "failed",
        ]
        print(df[cols].to_string(index=False))

    print(f"\nResults saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
