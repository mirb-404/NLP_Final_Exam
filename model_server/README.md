# model_server — self-hosted reasoning LLM

Hosts an **open-weight** instruct model (from Hugging Face) on your own box
(DataLab / GPU server / HF Space) and serves it over an **OpenAI-compatible**
HTTP API. The main project calls it instead of a paid API — so the exam machine
needs no GPU and no credits.

**Exam-legal:** the model is open-weight / freely accessible → not a paid
commercial LLM API.

---

## 1. Run the server (on the DataLab)

```bash
pip install -r requirements.txt

export MODEL_ID="Qwen/Qwen2.5-3B-Instruct"   # see hardware table below
export API_KEY="some-secret"                 # optional bearer auth
python server.py                             # 0.0.0.0:8000
```

Pick `MODEL_ID` for your hardware:

| DataLab hardware | MODEL_ID | Notes |
|---|---|---|
| GPU ≥16 GB | `meta-llama/Llama-3.1-8B-Instruct` | best; PDF-recommended (gated → accept license on HF) |
| GPU ~8 GB | `Qwen/Qwen2.5-3B-Instruct` | good; **default** |
| CPU only | `Qwen/Qwen2.5-1.5B-Instruct` | works, slow |

Check it locally on the DataLab:
```bash
curl http://localhost:8000/health
```

## 2. Expose it publicly

A DataLab notebook usually has no public IP. Open a tunnel:

```bash
# cloudflared — no signup, prints an https URL
cloudflared tunnel --url http://localhost:8000
#  -> https://<random>.trycloudflare.com
```
(or `ngrok http 8000`). That HTTPS URL is your `MODEL_SERVER_URL`.

## 3. Point the project at it (on THIS machine)

Add to the project root `.env`:
```
MODEL_SERVER_URL=https://<random>.trycloudflare.com
MODEL_SERVER_KEY=some-secret      # only if you set API_KEY
```
`src/utils.get_llm()` auto-detects `MODEL_SERVER_URL` and routes all LLM calls
to your server. Then:
```bash
uv run python main.py
```

## 4. Verify end-to-end
```bash
curl -X POST "$MODEL_SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer some-secret" \
  -d '{"messages":[{"role":"user","content":"Reply: PONG"}],"max_tokens":10}'
```

---

## ⚠️ Exam-day reliability
- **Ephemeral URL:** `trycloudflare` URLs change on every restart → update `.env`
  if you restart the tunnel. (A named cloudflared tunnel or ngrok reserved domain
  is stable.)
- **Idle timeouts:** many free notebooks kill long-running processes. Keep the
  DataLab tab active; test the full `main.py` run the day before.
- **Plan B:** if the server is down, the project still falls back to the local
  `Qwen2.5-0.5B` model (weak but offline) — no crash.

## HF-native alternative (more reliable than a tunnel)
Deploy this same `server.py` logic as a **Hugging Face Space** (free CPU, or
ZeroGPU). A Space gets a stable public URL by default and is built to stay up —
use its URL as `MODEL_SERVER_URL`. Trade-off: free Space hardware is modest.
