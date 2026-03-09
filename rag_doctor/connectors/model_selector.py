"""
Ollama model selection utility.
Benchmarks installed models on RAG tasks and recommends the best one.
"""
from __future__ import annotations
import time
import json
import urllib.request
import urllib.error
from typing import List, Dict, Optional, Tuple

OLLAMA_BASE_URL = "http://localhost:11434"

BENCHMARK_QUERY = "What is the maximum dose of acetaminophen for liver disease patients?"
BENCHMARK_CONTEXT = (
    "Acetaminophen dosing: Standard adults 4000mg max daily. "
    "For liver disease patients: maximum 2000mg per day due to reduced hepatic metabolism."
)
BENCHMARK_EXPECTED = "2000mg"


def _post(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _get(url: str, timeout: int = 10) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def check_ollama_running() -> bool:
    result = _get(f"{OLLAMA_BASE_URL}/api/tags")
    return "error" not in result


def list_models() -> List[Dict]:
    result = _get(f"{OLLAMA_BASE_URL}/api/tags")
    if "error" in result:
        return []
    return result.get("models", [])


def benchmark_model(model_name: str) -> Dict:
    """
    Run a quick RAG benchmark on a model.
    Returns: speed (tokens/s), quality (0-1), latency (s)
    """
    prompt = (
        f"Answer using ONLY this context:\n{BENCHMARK_CONTEXT}\n\n"
        f"Question: {BENCHMARK_QUERY}\nAnswer in one sentence:"
    )
    t0 = time.time()
    result = _post(f"{OLLAMA_BASE_URL}/api/generate", {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 64}
    }, timeout=90)

    latency = time.time() - t0

    if "error" in result:
        return {"model": model_name, "error": result["error"], "score": 0}

    response = result.get("response", "").lower()
    eval_count = result.get("eval_count", 1)
    eval_duration = result.get("eval_duration", 1e9)  # nanoseconds
    tokens_per_sec = (eval_count / eval_duration) * 1e9 if eval_duration > 0 else 0

    # Quality: does answer contain the expected value?
    quality = 1.0 if BENCHMARK_EXPECTED.lower() in response else 0.0
    # Partial credit for mentioning relevant numbers
    if "2000" in response or "two thousand" in response:
        quality = max(quality, 0.8)

    # Composite score: 60% quality, 40% speed (normalized)
    speed_score = min(tokens_per_sec / 30.0, 1.0)  # 30 t/s = perfect speed score
    composite = 0.6 * quality + 0.4 * speed_score

    return {
        "model": model_name,
        "response": result.get("response", "").strip()[:120],
        "quality": round(quality, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "latency_s": round(latency, 2),
        "composite_score": round(composite, 3),
    }


def select_and_benchmark(verbose: bool = True) -> Optional[str]:
    """
    Check Ollama, list installed models, benchmark each, return best model name.
    """
    if not check_ollama_running():
        print("❌  Ollama is not running. Start it with: ollama serve")
        return None

    models = list_models()
    if not models:
        print("❌  No models installed. Install one with:")
        print("      ollama pull llama3.2")
        print("      ollama pull mistral")
        return None

    model_names = [m["name"] for m in models]

    if verbose:
        print(f"\n{'='*58}")
        print(f"  Ollama Model Selector for rag-doctor")
        print(f"{'='*58}")
        print(f"  Found {len(model_names)} installed model(s):")
        for name in model_names:
            size = next((m.get("size", 0) for m in models if m["name"] == name), 0)
            size_gb = f"{size/1e9:.1f}GB" if size else "?"
            print(f"    • {name:40} {size_gb}")
        print(f"\n  Running RAG benchmark on each model...")
        print(f"  Task: Medical dosing question (requires precise retrieval)")
        print()

    results = []
    for name in model_names:
        if verbose:
            print(f"  Testing {name}...", end="", flush=True)
        bm = benchmark_model(name)
        results.append(bm)
        if verbose:
            if "error" in bm:
                print(f" ❌ error: {bm['error'][:50]}")
            else:
                q_icon = "✅" if bm["quality"] >= 0.8 else "⚠️ " if bm["quality"] > 0 else "❌"
                print(f" {q_icon} quality={bm['quality']:.1f} speed={bm['tokens_per_sec']:.0f}t/s latency={bm['latency_s']:.1f}s score={bm['composite_score']:.3f}")

    # Sort by composite score
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("❌  All model benchmarks failed")
        return None

    valid.sort(key=lambda x: -x["composite_score"])
    best = valid[0]

    if verbose:
        print(f"\n{'─'*58}")
        print(f"  🏆  Best model: {best['model']}")
        print(f"       Quality score : {best['quality']:.2f}")
        print(f"       Speed         : {best['tokens_per_sec']:.0f} tokens/sec")
        print(f"       Latency       : {best['latency_s']:.1f}s")
        print(f"       Composite     : {best['composite_score']:.3f}")
        if len(valid) > 1:
            print(f"\n  Full ranking:")
            for i, r in enumerate(valid, 1):
                print(f"    {i}. {r['model']:40} score={r['composite_score']:.3f}")
        print(f"{'='*58}\n")

    return best["model"]
