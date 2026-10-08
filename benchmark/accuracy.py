import requests


def run_accuracy(cfg, method):
    """
    Lightweight API-level quality smoke test.
    It is intentionally deterministic and is not a replacement for a
    formal benchmark such as MMLU/GSM8K/HellaSwag.
    """
    if not cfg["accuracy"]["enabled"]:
        return {"score": None, "passed": True}

    base = f"http://{cfg['server']['host']}:{cfg['server']['port']}"
    model = cfg["model"]["served_name"]

    cases = [
        ("What is 2 + 2? Answer with only the number.", "4"),
        ("What is the capital of France? Answer with one word.", "Paris"),
        ("What is 10 * 7? Answer with only the number.", "70"),
        ("Name the planet known as the Red Planet.", "Mars"),
        ("How many days are in a leap year? Answer with only the number.", "366"),
        ("What is H2O commonly called? Answer with one word.", "water"),
        ("What is 15 - 6? Answer with only the number.", "9"),
        ("Which language is primarily used to style web pages?", "CSS"),
        ("What is the opposite of hot?", "cold"),
        ("How many sides does a triangle have? Answer with only the number.", "3"),
    ]

    passed = 0
    for prompt, expected in cases:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": cfg["accuracy"]["max_tokens"],
        }
        r = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip().lower()
        if expected.lower() in answer:
            passed += 1

    score = passed / len(cases)
    return {
        "score": score,
        "score_percent": round(score * 100, 2),
        "passed": score >= cfg["accuracy"]["minimum_score"],
        "cases": len(cases),
        "correct": passed,
    }
