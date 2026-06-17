"""Interactive test client for server.py. Type a prompt, get a reply + speed.

    python test_server.py        # then type prompts; blank line or Ctrl-C quits
"""

import os
import time

import requests

URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8000").rstrip("/")
KEY = os.getenv("MODEL_SERVER_KEY") or os.getenv("API_KEY")
HEADERS = {"Authorization": f"Bearer {KEY}"} if KEY else {}


def ask(prompt):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256}
    t = time.perf_counter()
    r = requests.post(f"{URL}/v1/chat/completions", json=body, headers=HEADERS, timeout=300)
    r.raise_for_status()
    dt = time.perf_counter() - t
    text = r.json()["choices"][0]["message"]["content"].strip()
    print(f"\n{text}\n[{dt:.2f}s | ~{len(text) // 4} tokens | ~{len(text) // 4 / dt:.1f} tok/s]")


print("health:", requests.get(f"{URL}/health", timeout=10).json())
print("Type a prompt (blank line or Ctrl-C to quit).")
while True:
    try:
        prompt = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not prompt:
        break
    ask(prompt)
