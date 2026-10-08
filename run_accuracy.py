
import json
import sys
import time
from pathlib import Path

import yaml

from benchmark.server import VLLMServer
from accuracy_benchmark import evaluate_gsm8k


PROJECT_DIR = Path("/kaggle/working/quant-benchmark")
CONFIG_FILE = PROJECT_DIR / "config.yaml"
RESULTS_DIR = PROJECT_DIR / "results" / "accuracy"


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for method in cfg["quantization"]["methods"]:

        print("\n" + "=" * 80)
        print(f"STARTING ACCURACY EVALUATION: {method['name'].upper()}")
        print(f"MODEL: {method['model']}")
        print("=" * 80)

        server_log = (
            RESULTS_DIR / f"server_{method['name']}.log"
        )

        server = VLLMServer(
            cfg,
            method,
        )

        try:
            print("Starting vLLM server...")
            server.start(server_log)

            print("vLLM server is ready.")

            result = evaluate_gsm8k(
                cfg,
                method,
                num_samples=100,
            )

            all_results.append(result)

        except Exception as exc:
            print(
                f"\nERROR during {method['name']} evaluation:"
                f"\n{exc}"
            )

            all_results.append(
                {
                    "benchmark": "gsm8k",
                    "quantization": method["name"],
                    "model": method["model"],
                    "error": str(exc),
                }
            )

        finally:
            print(f"Stopping {method['name']} server...")
            server.stop()
            time.sleep(3)

    # ---------------------------------------------------------
    # Calculate degradation versus FP16
    # ---------------------------------------------------------

    valid_results = [
        r for r in all_results
        if "accuracy" in r
    ]

    fp16 = next(
        (
            r for r in valid_results
            if r["quantization"] == "fp16"
        ),
        None,
    )

    if fp16:
        baseline = fp16["accuracy"]

        for result in valid_results:
            result["accuracy_delta_vs_fp16"] = round(
                result["accuracy"] - baseline,
                6,
            )

            result["accuracy_delta_pp_vs_fp16"] = round(
                (result["accuracy"] - baseline) * 100,
                2,
            )

    summary_file = RESULTS_DIR / "gsm8k_final_summary.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    print("\n")
    print("=" * 80)
    print("FINAL GSM8K ACCURACY RESULTS")
    print("=" * 80)

    print(
        f"{'Model':<10}"
        f"{'Accuracy':>12}"
        f"{'Delta vs FP16':>18}"
    )

    print("-" * 80)

    for result in all_results:

        if "accuracy" not in result:
            print(
                f"{result['quantization']:<10}"
                f"{'FAILED':>12}"
            )
            continue

        delta = result.get(
            "accuracy_delta_pp_vs_fp16",
            0.0,
        )

        print(
            f"{result['quantization']:<10}"
            f"{result['accuracy_percent']:>10.2f}%"
            f"{delta:>15.2f} pp"
        )

    print("=" * 80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
