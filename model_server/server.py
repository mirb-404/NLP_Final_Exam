"""Tiny OpenAI-compatible server for an open-weight chat model.

Downloads MODEL_ID into agent_models/ on first run, then serves it offline at
POST /v1/chat/completions. Point the project at it via MODEL_SERVER_URL in .env.

    MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2" python server.py
    LOAD_4BIT=1 python server.py        # small GPU (~5GB); needs bitsandbytes
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

MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
API_KEY = os.getenv("API_KEY")  # if set, clients must send: Authorization: Bearer <key>
LOAD_4BIT = os.getenv("LOAD_4BIT", "0").lower() in ("1", "true", "yes")
PATH = Path(os.getenv("MODELS_DIR", "agent_models")) / MODEL_ID.replace("/", "__")


def load_model():
    """Download into agent_models/ if missing, then load from disk (offline)."""
    if not ((PATH / "config.json").exists() and any(PATH.glob("*.safetensors"))):
        print(f"[server] downloading {MODEL_ID} -> {PATH}")
        snapshot_download(MODEL_ID, local_dir=str(PATH),
                          token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                          ignore_patterns=["*.pth", "*.bin", "*.gguf", "consolidated.safetensors"])

    kw = {"device_map": "auto", "local_files_only": True}
    if LOAD_4BIT:
        import torch
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["torch_dtype"] = "auto"

    tok = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
    mdl = AutoModelForCausalLM.from_pretrained(PATH, **kw)
    print(f"[server] {MODEL_ID} ready on {mdl.device} (4bit={LOAD_4BIT})")
    return tok, mdl


tokenizer, model = load_model()
app = FastAPI()


class Chat(BaseModel):
    messages: list[dict]
    max_tokens: int = 512
    temperature: float = 0.3


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/chat/completions")
def chat(req: Chat, authorization: str = Header(None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid api key")

    prompt = tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
    ins = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ins, max_new_tokens=req.max_tokens,
                         do_sample=req.temperature > 0,
                         temperature=max(req.temperature, 1e-5),
                         pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
            "created": int(time.time()), "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}]}


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
