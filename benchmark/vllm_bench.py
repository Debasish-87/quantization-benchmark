import json
import subprocess
import sys
from pathlib import Path


def csv(value):
    return ",".join(str(x) for x in value) if isinstance(value, list) else str(value)


def run_vllm_bench(cfg, method, raw_file, concurrency, repetition):
    b = cfg["benchmark"]
    raw_file = Path(raw_file)
    raw_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "serve",
        "--backend", "openai-chat",
        "--base-url", f"http://{cfg['server']['host']}:{cfg['server']['port']}",
        "--endpoint", "/v1/chat/completions",
        "--model", cfg["model"]["served_name"],
        "--tokenizer", method["model"],
        "--dataset-name", "random",
        "--random-input-len", str(b["input_tokens"]),
        "--random-output-len", str(b["output_tokens"]),
        "--num-prompts", str(b["num_prompts"]),
        "--max-concurrency", str(concurrency),
        "--request-rate", str(b["request_rate"]),
        "--num-warmups", str(b["warmup_prompts"]),
        "--percentile-metrics", csv(b["percentile_metrics"]),
        "--metric-percentiles", csv(b["percentiles"]),
        "--seed", str(b["seed"]),
        "--temperature", str(b["temperature"]),
        "--save-result",
        "--result-dir", str(raw_file.parent),
        "--result-filename", raw_file.name,
        "--disable-tqdm",
    ]

    print("Running:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError("vLLM benchmark failed.")

    if not raw_file.exists():
        raise RuntimeError(f"Benchmark result not found: {raw_file}")

    with open(raw_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["_benchmark_meta"] = {
        "quantization": method["name"],
        "model": method["model"],
        "concurrency": concurrency,
        "repetition": repetition,
    }
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data
