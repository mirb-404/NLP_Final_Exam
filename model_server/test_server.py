"""Smoke + speed test for server.py. Run it after the server is up.

    python test_server.py
    python test_server.py "What does Siemens do?"
"""

import os
import sys
import time

import requests

URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8000").rstrip("/")
KEY = os.getenv("MODEL_SERVER_KEY") or os.getenv("API_KEY")
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "In one sentence, what does Siemens do?"
HEADERS = {"Authorization": f"Bearer {KEY}"} if KEY else {}


def ask(prompt, label):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 128}
    t = time.perf_counter()
    r = requests.post(f"{URL}/v1/chat/completions", json=body, headers=HEADERS, timeout=300)
    r.raise_for_status()
    dt = time.perf_counter() - t
    text = r.json()["choices"][0]["message"]["content"].strip()
    print(f"\n[{label}] {dt:.2f}s | ~{len(text) // 4} tokens | ~{len(text) // 4 / dt:.1f} tok/s")
    print(text)


print("health:", requests.get(f"{URL}/health", timeout=10).json())
print(f"prompt: {PROMPT}")
ask(PROMPT, "cold")  # first call may include warmup
ask(PROMPT, "warm")  # steady-state speed
