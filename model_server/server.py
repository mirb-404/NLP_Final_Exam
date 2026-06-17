"""
Standalone model server for the AI CEO project.

Run this on your DataLab / GPU box. It downloads an open-weight instruct model
from Hugging Face and serves it over an OpenAI-compatible HTTP API
(POST /v1/chat/completions). The main project (src/utils.py) calls this endpoint
when the env var MODEL_SERVER_URL is set — so the exam machine needs no GPU and
no paid API; it just talks to your server.

Exam-rule note: the served model is open-weight (freely accessible), so this is
COMPLIANT — it is NOT a paid commercial LLM API.

--- Quick start (run ON the DataLab) ---------------------------------------
    pip install -r requirements.txt

    # pick a model that fits your DataLab hardware:
    #   GPU  >=16GB : meta-llama/Llama-3.1-8B-Instruct   (best, PDF-recommended)
    #   GPU  ~8GB   : Qwen/Qwen2.5-3B-Instruct           (good, default)
    #   CPU only    : Qwen/Qwen2.5-1.5B-Instruct         (slow but works)
    export MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
    export API_KEY="choose-a-secret"     # optional; clients must then send it
    python server.py                     # serves on 0.0.0.0:8000

--- Expose it (DataLab usually has no public IP) ---------------------------
    # easiest: cloudflared (no signup)
    cloudflared tunnel --url http://localhost:8000
    # -> prints https://<random>.trycloudflare.com   (this is your MODEL_SERVER_URL)

--- Point the project at it (on THIS machine, in .env) ---------------------
    MODEL_SERVER_URL=https://<random>.trycloudflare.com
    MODEL_SERVER_KEY=choose-a-secret      # only if you set API_KEY above
"""

import os
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "agent_models"))  # where weights are stored
API_KEY = os.getenv("API_KEY")            # if set, clients must send Authorization: Bearer <key>
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Files we never need (mistral ships a non-HF single-file dupe + .pth/.gguf variants).
# Tip: if a repo has BOTH *.bin and *.safetensors, add "*.bin" here to ~halve the download.
_IGNORE = ["*.pth", "*.gguf", "original/*", "consolidated.safetensors"]
_WEIGHT_GLOBS = ("*.safetensors", "*.bin")


def _model_path() -> Path:
    """Local folder for this model, e.g. agent_models/mistralai__Mistral-7B-Instruct-v0.2."""
    return MODELS_DIR / MODEL_ID.replace("/", "__")


def _is_downloaded(path: Path) -> bool:
    """True only if config + at least one weight file are already on disk."""
    if not (path / "config.json").exists():
        return False
    return any(next(path.glob(g), None) for g in _WEIGHT_GLOBS)


def ensure_model() -> str:
    """Download MODEL_ID into agent_models/ once; reuse it (offline) thereafter."""
    path = _model_path()
    if _is_downloaded(path):
        print(f"[server] model found at {path} — loading locally (no download)")
    else:
        print(f"[server] model not in {path} — downloading {MODEL_ID} (one time) ...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(path),
            token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),  # required for gated repos (Mistral)
            ignore_patterns=_IGNORE,
        )
        print(f"[server] download complete -> {path}")
    return str(path)


MODEL_PATH = ensure_model()
# 4-bit (NF4) drops VRAM ~15GB -> ~5GB. Only *faster* if fp16 would OOM/offload;
# on a GPU that already fits fp16 it is similar or slightly slower. Default off.
LOAD_4BIT = os.getenv("LOAD_4BIT", "0").lower() in ("1", "true", "yes")

print(f"[server] loading {MODEL_ID} from {MODEL_PATH} ... (4bit={LOAD_4BIT})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

model_kwargs = {"device_map": "auto", "local_files_only": True}
if LOAD_4BIT:
    import torch
    from transformers import BitsAndBytesConfig

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    model_kwargs["torch_dtype"] = "auto"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
print(f"[server] model ready on device: {model.device}")

app = FastAPI(title="AI CEO model server")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.3


def _check_auth(authorization: str | None) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid api key")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, authorization: str | None = Header(default=None)):
    """OpenAI-compatible chat completion (single, non-streaming response)."""
    _check_auth(authorization)

    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        do_sample=req.temperature > 0,
        temperature=max(req.temperature, 1e-5),
        pad_token_id=tokenizer.eos_token_id,
    )
    # decode only the newly generated tokens (drop the prompt)
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
