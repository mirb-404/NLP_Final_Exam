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

export MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"   # see hardware table below
export API_KEY="some-secret"                           # optional bearer auth
python server.py                                       # 0.0.0.0:8000
```

**Weights are stored in `agent_models/`** (override with `MODELS_DIR`). On
startup the script:
- checks `agent_models/<model>/` — if the weights are there, it **loads them
  offline** (no download);
- if not, it **downloads once** into that folder, then loads.

So the first run downloads; every run after is offline. Gated models (Mistral)
need auth for that first download — run `huggingface-cli login` (or set
`HUGGINGFACEHUB_API_TOKEN`) **and** accept the model license on HF.

> Space tip: Mistral ships duplicate weight formats. The script already skips
> `consolidated.safetensors`; if the repo also has `*.bin` next to
> `*.safetensors`, add `"*.bin"` to `_IGNORE` in `server.py` to ~halve the download.

Pick `MODEL_ID` for your hardware:

| DataLab hardware | MODEL_ID | Notes |
|---|---|---|
| GPU ≥16 GB | `mistralai/Mistral-7B-Instruct-v0.2` | fp16, ~15 GB; gated → accept license + login |
| GPU 6–12 GB | `mistralai/Mistral-7B-Instruct-v0.2` + `LOAD_4BIT=1` | ~5 GB, fits smaller GPUs |
| GPU ~8 GB (no quant) | `Qwen/Qwen2.5-3B-Instruct` | good, lighter |
| CPU only | `Qwen/Qwen2.5-1.5B-Instruct` | works, slow |

### 4-bit quantization (`LOAD_4BIT=1`)
```bash
pip install bitsandbytes          # CUDA/Linux GPU only
LOAD_4BIT=1 MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2" python server.py
```
Cuts VRAM ~15 GB → ~5 GB. **Not automatically faster:** on a GPU that already
fits fp16 it's similar or slightly slower (dequant overhead). It's faster *only*
when fp16 would OOM and spill to CPU offload. Rule of thumb: GPU ≥16 GB → leave
it off (fp16); smaller GPU → turn it on so the model fits without offloading.

Check it locally on the DataLab:
```bash
curl http://localhost:8000/health
```

## 2. Expose it publicly

**Same machine?** If you run the whole pipeline ON the DataLab too, skip this —
just use `MODEL_SERVER_URL=http://localhost:8000`. No tunnel needed.

Otherwise (pipeline on a different machine), a DataLab notebook usually has no
public IP, so open a tunnel:

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

Easiest — the test script (health check + timed call, reports tokens/sec):
```bash
python test_server.py                                  # default localhost
python test_server.py "What does Siemens do?"          # custom prompt
MODEL_SERVER_URL=http://localhost:8000 MODEL_SERVER_KEY=some-secret python test_server.py
```
It calls twice (cold + warm) so you see steady-state speed.

Or raw curl:
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
