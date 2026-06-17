"""
Quick smoke + speed test for the model server (model_server/server.py).

Run it AFTER the server is up. It hits the same HTTP API the real project uses,
so a pass here means the pipeline will work too. Reports latency + a rough
tokens/sec, and runs the call twice (1st = cold, 2nd = warm).

Usage (on any box that can reach the server):
    python test_server.py
    python test_server.py "Summarize what Siemens does in one sentence."
    MODEL_SERVER_URL=http://localhost:8000 MODEL_SERVER_KEY=secret python test_server.py
"""

import os
import sys
import time

import requests

URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8000").rstrip("/")
KEY = os.getenv("MODEL_SERVER_KEY") or os.getenv("API_KEY")
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "In one sentence, what does Siemens do?"

headers = {"Content-Type": "application/json"}
if KEY:
    headers["Authorization"] = f"Bearer {KEY}"


def health() -> None:
    try:
        r = requests.get(f"{URL}/health", timeout=10)
        print(f"[health] {r.status_code} {r.json()}")
    except Exception as exc:
        print(f"[health] cannot reach {URL} -> {exc}")
        sys.exit(1)


def ask(prompt: str, label: str) -> None:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.3,
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{URL}/v1/chat/completions", headers=headers, json=body, timeout=300)
        r.raise_for_status()
    except Exception as exc:
        print(f"[{label}] request failed -> {exc}")
        sys.exit(1)
    dt = time.perf_counter() - t0
    text = r.json()["choices"][0]["message"]["content"].strip()
    approx_tok = max(1, len(text) // 4)  # ~4 chars/token, rough
    print(f"\n[{label}] {dt:.2f}s | ~{approx_tok} tokens | ~{approx_tok / dt:.1f} tok/s (rough)")
    print(f"[response]\n{text}")


if __name__ == "__main__":
    print(f"server: {URL}  auth: {'yes' if KEY else 'no'}")
    health()
    print(f"\nprompt: {PROMPT}")
    ask(PROMPT, "cold")   # first call includes any warmup
    ask(PROMPT, "warm")   # steady-state latency
