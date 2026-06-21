"""
Small shared helpers: timestamps, JSON/text IO, text cleaning, and the two
model factories (`get_embedder`, `get_llm`) used across the backend.

Keeping the model loaders here means every agent uses the *same* embedding model
and the *same* LLM, and we only load each heavy object once (cached).
"""

import html
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from src import config


# ----------------------------------------------------------------------------
# Time + IO
# ----------------------------------------------------------------------------
def now_iso() -> str:
    """ISO-8601 timestamp (used in the CEO briefing and trace, like the course agent output)."""
    return datetime.now(timezone.utc).astimezone().isoformat()


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ----------------------------------------------------------------------------
# Text cleaning (Task 3 — Module 2 idioms)
# ----------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
_HTML_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    """Unescape HTML entities, strip tags, collapse whitespace. Cheap, deterministic, explainable."""
    if not text:
        return ""
    text = html.unescape(text)   # &nbsp; &lt; &amp; -> real chars (news/StackExchange/research feeds)
    text = _HTML_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embedder():
    """Load the SentenceTransformer embedding model once (Module 10 pattern)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


class _ChatLLM:
    """
    Thin uniform wrapper over the HF Inference Providers chat endpoint.

    Exposes the same `.invoke(prompt) -> str` surface as the local
    HuggingFacePipeline, so `ask_llm` does not care which backend it gets.
    """

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def invoke(self, prompt: str) -> str:
        resp = self._client.chat_completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        return resp.choices[0].message.content


class _RemoteServerLLM:
    """
    Client for a self-hosted OpenAI-compatible chat server (see model_server/).

    Lets us run a strong open-weight model on a DataLab / GPU box and call it
    from here over HTTP — no local GPU, no paid API, exam-rule compliant.
    """

    def __init__(self, url: str, api_key: str | None):
        self._url = url.rstrip("/") + "/v1/chat/completions"
        self._key = api_key

    def invoke(self, prompt: str) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        resp = requests.post(
            self._url,
            headers=headers,
            timeout=180,
            json={
                "model": config.LLM_REPO_ID,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


@lru_cache(maxsize=1)
def get_llm():
    """
    Return the reasoning LLM.

    Priority order:
      1. Self-hosted server (MODEL_SERVER_URL set) — your DataLab box running
         model_server/server.py. Best for the exam: strong open model, no local
         GPU, no paid API. See model_server/README.md.
      2. HF Inference Providers endpoint (HUGGINGFACEHUB_API_TOKEN set) — free
         tier, credit-limited (Llama-3.1-8B-Instruct).
      3. Local Qwen2.5-0.5B-Instruct via HuggingFacePipeline (pattern from
         Module11/LangGraph.ipynb) — offline, no token, weak but never fails.
    """
    import os

    server_url = os.getenv("MODEL_SERVER_URL")
    if server_url:
        return _RemoteServerLLM(server_url, os.getenv("MODEL_SERVER_KEY"))

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        try:
            from huggingface_hub import InferenceClient

            return _ChatLLM(InferenceClient(token=token), config.LLM_REPO_ID)
        except Exception as exc:  # network / model gated -> fall through to local
            print(f"[utils] HF Inference unavailable ({exc}); using local fallback.")

    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline

    gen = pipeline(
        "text-generation",
        model=config.LLM_FALLBACK,
        max_new_tokens=400,
        do_sample=False,
        return_full_text=False,  # return only the generated continuation, not the prompt
    )
    return HuggingFacePipeline(pipeline=gen)


def ask_llm(prompt: str) -> str:
    """Send a prompt to the LLM and return clean text. One call site for all agents."""
    out = get_llm().invoke(prompt)
    # HuggingFacePipeline returns str; some wrappers return objects with .content
    return str(getattr(out, "content", out)).strip()
