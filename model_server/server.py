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

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
API_KEY = os.getenv("API_KEY")            # if set, clients must send Authorization: Bearer <key>
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

print(f"[server] loading {MODEL_ID} ... (first run downloads weights)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
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
