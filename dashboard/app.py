"""
Executive Intelligence Dashboard (Streamlit) — Deliverable 2.

Styled renderer of results/dashboard_data.json. Each PDF section is rendered by a
render_*(d) function so the same code shows the live data AND any historical
snapshot saved in the Activity log. Asking the agent re-runs the analysis, so a
question refreshes every tab and is saved with a full snapshot.
"""

import contextlib
import html
import io
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "dashboard_data.json"
ACTIVITY = ROOT / "results" / "activity.json"   # grows as questions are asked

# Load .env so the pipeline run from the dashboard uses MODEL_SERVER_URL (Mistral),
# not the local Qwen fallback.
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402  (model ids for the tech-stack panel)
from src.utils import strip_src_refs  # noqa: E402  (drop inline (src-N) markers from LLM text)

LLM_MODEL = config.LLM_REPO_ID
EMBED_MODEL = config.EMBEDDING_MODEL
RAG_TOOL = "search_knowledge_base — Hybrid BM25 + dense"

st.set_page_config(page_title="AI Strategy Consultant — Strategic Intelligence", layout="wide")

# ---------------------------------------------------------------- styling ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
#MainMenu, footer, header {visibility:hidden;}
.block-container{padding-top:1.4rem; max-width:1180px;}
.hero{font-size:1.9rem;font-weight:700;color:#cdd9f5;margin:.1rem 0 .2rem;}
.sub{color:#8a93a3;margin-bottom:1rem;}
.row{display:flex;gap:.8rem;flex-wrap:wrap;margin:.3rem 0 1rem;}
.metric{flex:1;min-width:140px;background:#1b2334;border:1px solid #2a3550;border-radius:14px;
        padding:.9rem 1.1rem;box-shadow:0 2px 6px rgba(0,0,0,.15);}
.m-val{font-size:1.45rem;font-weight:700;color:#eef2f9;}
.m-lbl{font-size:.74rem;color:#8a93a3;text-transform:uppercase;letter-spacing:.05em;}
.card{background:#1b2334;border:1px solid #2a3550;border-radius:14px;padding:1rem 1.2rem;
      margin-bottom:.8rem;box-shadow:0 2px 6px rgba(0,0,0,.15);}
.card-title{font-weight:600;font-size:1.04rem;color:#eef2f9;}
.muted{font-size:.82rem;color:#8a93a3;margin:.25rem 0 .55rem;}
.desc{color:#c7d0de;line-height:1.6;margin:.25rem 0 .5rem;}
.step{color:#7ddf9f;font-size:.88rem;font-weight:600;margin:.2rem 0 .45rem;}
.stack{font-size:.82rem;color:#aeb8c7;margin:.45rem 0;} .stack code{font-size:.76rem;color:#9fd0ff;}
.badge{display:inline-block;padding:.16rem .7rem;border-radius:999px;font-size:.74rem;font-weight:700;white-space:nowrap;}
.b-high{background:#3a1f24;color:#ff8a8a;} .b-med{background:#3a2f1c;color:#ffc46b;} .b-low{background:#1e3328;color:#7ddf9f;}
.evidence{font-size:.83rem;color:#aeb8c7;border-left:3px solid #34405d;padding-left:.7rem;margin:.3rem 0;}
.evidence a, .muted a{color:#9fd0ff;text-decoration:none;} .evidence a:hover, .muted a:hover{text-decoration:underline;}
.answer{background:#161e2e;border:1px solid #2a3550;border-radius:16px;
        padding:1.3rem 1.5rem;font-size:1.04rem;line-height:1.7;color:#dbe3f2;}
.brief-card{background:#161e2e;border:1px solid #243049;border-radius:14px;
            padding:1.1rem 1.35rem;margin-bottom:.85rem;box-shadow:0 2px 6px rgba(0,0,0,.12);}
.brief-q{font-weight:700;font-size:1.06rem;color:#ff8f8f;margin-bottom:.4rem;}
.brief-a{color:#ffc9c9;line-height:1.75;font-size:1.0rem;}
.flex-between{display:flex;justify-content:space-between;align-items:center;gap:.6rem;}
.stTabs [data-baseweb="tab"]{font-weight:600;}
</style>
""", unsafe_allow_html=True)

_PIE_COLORS = ["#4ade80", "#f87171", "#60a5fa", "#fbbf24", "#c084fc", "#94a3b8"]


# ----------------------------------------------------------------- helpers ---
def esc(x) -> str:
    return html.escape(str(x))


def badge(level) -> str:
    cls = {"high": "b-high", "medium": "b-med", "low": "b-low"}.get(str(level).lower(), "b-med")
    return f"<span class='badge {cls}'>{esc(level)}</span>"


def sentiment_label(score) -> str:
    """Signed sentiment score (-1..+1) from RoBERTa/TweetEval -> readable label + value. ±0.05 = neutral."""
    x = float(score or 0)
    word = "Positive" if x >= 0.05 else "Negative" if x <= -0.05 else "Neutral"
    return f"{word} ({x:+.2f})"


def metric(label, value) -> str:
    return f"<div class='metric'><div class='m-val'>{esc(value)}</div><div class='m-lbl'>{esc(label)}</div></div>"


def evidence_html(evidence) -> str:
    rows = []
    for e in evidence:
        title, url, src = esc(e.get("title", "")), e.get("url", ""), esc(e.get("source", ""))
        cited = f"<a href='{esc(url)}' target='_blank'>{title}</a>" if url else title
        link = f" · <a href='{esc(url)}' target='_blank'>source</a>" if url else ""
        rows.append(f"<div class='evidence'>[{esc(e.get('ref', 'src'))}] {cited} — <i>{src}</i>{link}</div>")
    return "".join(rows) or "<div class='evidence muted'>no evidence</div>"


def card(title_html: str, body_html: str) -> None:
    st.markdown(f"<div class='card'><div class='flex-between'>{title_html}</div>{body_html}</div>",
                unsafe_allow_html=True)


def _contrast(hex_color: str) -> str:
    """Black or white, whichever contrasts with the given slice colour (by luminance)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#0e1117" if luminance > 0.55 else "#ffffff"


def show_pie(values: dict, caption: str = "") -> None:
    """Dark-themed pie chart — bright wedges, each % label coloured opposite its slice."""
    values = {k: v for k, v in (values or {}).items() if v}
    if not values:
        return
    if caption:
        st.markdown(f"<div class='muted'>{esc(caption)}</div>", unsafe_allow_html=True)
    colors = _PIE_COLORS[:len(values)]
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    fig.patch.set_alpha(0)
    wedges, texts, autotexts = ax.pie(
        list(values.values()), labels=list(values.keys()), autopct="%1.0f%%",
        colors=colors, startangle=90, pctdistance=0.72, labeldistance=1.08,
        textprops={"color": "#eef2f9", "fontsize": 13, "fontweight": "600"},
        wedgeprops={"linewidth": 2, "edgecolor": "#0e1117"})
    for a, col in zip(autotexts, colors):     # % label contrasts with its own wedge
        a.set_color(_contrast(col))
        a.set_fontsize(12)
        a.set_fontweight("700")
    ax.set_aspect("equal")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_bar(pairs: list[tuple], caption: str = "") -> None:
    """Dark-themed horizontal bar chart — used for trend signal strength (keyword mentions)."""
    pairs = [(str(k), float(v)) for k, v in (pairs or []) if v]
    if not pairs:
        return
    if caption:
        st.markdown(f"<div class='muted'>{esc(caption)}</div>", unsafe_allow_html=True)
    pairs = pairs[:10][::-1]                       # biggest bar on top
    labels, values = [p[0] for p in pairs], [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(5.6, max(2.4, 0.42 * len(pairs))))
    fig.patch.set_alpha(0)
    ax.barh(labels, values, color="#60a5fa", edgecolor="#0e1117")
    ax.tick_params(colors="#aeb8c7", labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("none")
    for i, v in enumerate(values):
        ax.text(v, i, f" {int(v)}", va="center", color="#eef2f9", fontsize=10, fontweight="600")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


@st.cache_resource
def get_retriever():
    """Load the hybrid retriever once (reads the Chroma index)."""
    from src.retriever_hybrid import HybridRetriever
    return HybridRetriever()


def load_activity() -> list:
    return json.loads(ACTIVITY.read_text(encoding="utf-8")) if ACTIVITY.exists() else []


def log_activity(entry: dict) -> None:
    items = [entry] + load_activity()
    ACTIVITY.write_text(json.dumps(items[:20], indent=2, ensure_ascii=False), encoding="utf-8")


@contextlib.contextmanager
def live_timer(placeholder, label="⏱ Agent running"):
    """Tick an elapsed-seconds counter in `placeholder` from a background thread while a
    blocking call (the agent loop) runs in the main thread. Streamlit only lets a thread
    touch the UI if it carries the script-run context, so we attach it; if that internal
    API is missing on this Streamlit build we degrade gracefully to a final time only."""
    start = time.perf_counter()
    stop = threading.Event()
    thread = None
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        ctx = get_script_run_ctx()

        def _tick():
            while not stop.is_set():
                placeholder.markdown(
                    f"<div class='muted'>{label} {time.perf_counter() - start:.1f}s</div>",
                    unsafe_allow_html=True)
                time.sleep(0.1)

        thread = threading.Thread(target=_tick, daemon=True)
        add_script_run_ctx(thread, ctx)
        thread.start()
    except Exception:
        thread = None  # live ticking unsupported -> final elapsed still shown below
    try:
        yield lambda: time.perf_counter() - start
    finally:
        stop.set()
        if thread:
            thread.join(timeout=1)
        placeholder.markdown(
            f"<div class='muted'>{label} — done in {time.perf_counter() - start:.1f}s</div>",
            unsafe_allow_html=True)


# ---- analysis data: refreshed whenever the agent calls the LLM --------------
def regenerate_data(full: bool = False) -> None:
    """Run the analysis pipeline so every tab reflects the latest data. Called after each
    agent question (LLM call). full=True re-collects fresh documents first.
    Falls back to a full pipeline whenever the index OR the corpus is missing, so a
    half-built state (e.g. Chroma exists but corpus.csv was wiped) self-heals."""
    from src.knowledge_base import count as kb_count
    from src.orchestrator import run_analyze, run_pipeline
    try:
        ready = kb_count() > 0 and config.CORPUS_CSV.exists()
    except Exception:
        ready = False
    run_pipeline() if (full or not ready) else run_analyze()


def empty_data() -> dict:
    """Placeholder so the dashboard (and the Ask tab) still load before any corpus exists."""
    return {
        "company": {"name": config.COMPANY, "industry": config.INDUSTRY,
                    "n_documents": 0, "n_sources": 0, "source_breakdown": {}, "last_update": "—"},
        "market_intelligence": {}, "opportunities": [], "risks": [],
        "sentiment": {}, "recommendations": [], "briefing": {},
    }


def load_dashboard_data() -> dict:
    """Load the latest analysis if present; otherwise return an empty placeholder so the
    app never crashes — the agent / Re-analyse builds the corpus on first use."""
    if DATA.exists():
        try:
            return json.loads(DATA.read_text(encoding="utf-8"))
        except Exception:
            pass
    return empty_data()


def not_ready_note() -> None:
    st.info("No analysis yet. Ask the agent a question on the **Ask the Consultant** tab "
            "(it builds the knowledge base on first use), or click **Re-analyse** in the sidebar.")


# ------------------------------------------------------- section renderers ---
# Each takes a data dict so it renders the live data OR a saved Activity snapshot.
def render_overview(d: dict) -> None:
    co = d["company"]
    if not co.get("n_documents"):
        not_ready_note()
        return
    st.markdown("<div class='row'>" + "".join([
        metric("Documents", co.get("n_documents", 0)),
        metric("Data sources", co.get("n_sources", 0)),
        metric("Industry", co.get("industry", "")),
        metric("Last update", str(co.get("last_update", ""))[:16].replace("T", " ")),
    ]) + "</div>", unsafe_allow_html=True)
    mi = d.get("market_intelligence", {})
    left, right = st.columns([2, 1])
    with left:
        st.markdown("##### Recent news")
        for n in mi.get("recent_news", []):
            link = f"[{n['title']}]({n['url']})" if n.get("url") else n["title"]
            st.markdown(f"- {link}  ·  *{n.get('source', '')}*")
        st.markdown("##### Competitor activity")
        for e in mi.get("competitor_activity", []):
            title = f"[{e['title']}]({e['url']})" if e.get("url") else f"**{e.get('title', '')}**"
            st.markdown(f"- {title}  ·  *{e.get('source', '')}*")
        st.markdown("##### Important company announcements")
        anns = mi.get("company_announcements", [])
        if anns:
            for a in anns:
                link = f"[{a['title']}]({a['url']})" if a.get("url") else a["title"]
                st.markdown(f"- {link}  ·  *{a.get('source', '')}*")
        else:
            st.caption("No official announcements detected in the current corpus.")
        st.markdown("##### Emerging technologies")
        techs = mi.get("emerging_technologies", [])
        if techs:
            for t in techs:
                st.markdown(f"- **{strip_src_refs(t.get('title', ''))}**"
                            + (f"  ·  *confidence {t['confidence']}*" if t.get("confidence") else ""))
        else:
            st.caption("None surfaced yet — see the **Trends** tab.")
        kw = [k for k in mi.get("keywords", []) if not config.keyword_is_noise(k)]
        if kw:
            st.caption("Trending: " + " · ".join(kw))
    with right:
        show_pie(co.get("source_breakdown", {}), "Document mix by source")


def render_opportunities(d: dict) -> None:
    if not d.get("opportunities"):
        not_ready_note()
        return
    for opp in d.get("opportunities", []):
        desc = f"<div class='desc'>{esc(strip_src_refs(opp['description']))}</div>" if opp.get("description") else ""
        card(f"<span class='card-title'>{esc(opp['title'])}</span>{badge(opp.get('impact'))}",
             f"{desc}<div class='muted'>Confidence {esc(opp.get('confidence', 0))}</div>"
             f"{evidence_html(opp.get('evidence', []))}")


def render_risks(d: dict) -> None:
    risks = d.get("risks", [])
    if not risks:
        not_ready_note()
        return
    severity = {}
    for r in risks:
        key = str(r.get("severity", "Medium")).capitalize()
        severity[key] = severity.get(key, 0) + 1
    if severity:
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            show_pie(severity, "Risks by severity")
    for r in risks:
        desc = f"<div class='desc'>{esc(strip_src_refs(r['description']))}</div>" if r.get("description") else ""
        card(f"<span class='card-title'>{esc(r['title'])}</span>{badge(r.get('severity'))}",
             f"{desc}<div class='muted'>{esc(r.get('category', ''))} · confidence {esc(r.get('confidence', 0))}</div>"
             f"{evidence_html(r.get('evidence', []))}")


def render_trends(d: dict) -> None:
    """Task 4 — emerging trends management should monitor, with a signal-strength chart."""
    t = d.get("trends", {})
    items = t.get("items", [])
    # Drop domain noise (company name, ticker, publisher names, finance filler) so the
    # chart shows real topics — handles older data saved before the engine-side filter.
    signals = [s for s in t.get("signals", []) if not config.keyword_is_noise(s.get("keyword", ""))]
    if not items and not signals:
        not_ready_note()
        return
    if signals:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("##### Trend signal strength")
            st.caption("Documents in the corpus mentioning each top topic — the evidence weight behind a trend.")
        with right:
            show_bar([(s["keyword"], s["mentions"]) for s in signals])
    st.markdown("##### Emerging trends to monitor")
    st.caption("Grouped by lens: Technology trends · Customer behaviour shifts · Industry developments.")
    labels = {"Technology": "Technology trends",
              "Customer behaviour": "Customer behaviour shifts",
              "Industry": "Industry developments"}
    by_cat = {c: [] for c in labels}
    for it in items:                                  # legacy items lack a category
        cat = it.get("category")
        by_cat[cat if cat in by_cat else "Industry"].append(it)
    for cat, group in by_cat.items():
        if not group:
            continue
        st.markdown(f"**{labels[cat]}**")
        for it in group:
            desc = strip_src_refs(it.get("description", ""))
            desc_html = f"<div class='desc'>{esc(desc)}</div>" if desc else ""
            card(f"<span class='card-title'>{esc(strip_src_refs(it.get('title', '')))}</span>"
                 f"{badge('monitor')}",
                 f"{desc_html}<div class='muted'>Confidence {esc(it.get('confidence', 0))}</div>"
                 f"{evidence_html(it.get('evidence', []))}")


def render_sentiment(d: dict) -> None:
    s = d.get("sentiment", {})
    if not s:
        not_ready_note()
        return
    st.markdown("<div class='row'>" + "".join([
        metric("News sentiment", sentiment_label(s.get("news_sentiment", 0))),
        metric("Public sentiment", sentiment_label(s.get("public_sentiment", 0))),
        metric("Overall", sentiment_label(s.get("overall_sentiment", 0))),
    ]) + "</div>", unsafe_allow_html=True)
    st.caption("RoBERTa fine-tuned on TweetEval (negative/neutral/positive), mapped to −1 … +1 as P(pos)−P(neg); within ±0.05 counts as neutral.")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("##### Sentiment split")
        show_pie(s.get("distribution", {}))
    with v2:
        st.markdown("##### Sentiment trend")
        if s.get("trend"):
            st.line_chart(pd.DataFrame(s["trend"]).set_index("period")["sentiment"])


def render_recommendations(d: dict) -> None:
    if not d.get("recommendations"):
        not_ready_note()
        return
    for r in d.get("recommendations", []):
        rationale = f"<div class='desc'>{esc(strip_src_refs(r['rationale']))}</div>" if r.get("rationale") else ""
        first_step = f"<div class='step'>First step: {esc(strip_src_refs(r['first_step']))}</div>" if r.get("first_step") else ""
        risk_assess = r.get("risk_assessment", [])
        risk_html = (f"<div class='muted'>Risk assessment: {esc(', '.join(risk_assess))}</div>"
                     if risk_assess else "")
        body = (f"<div style='margin:.35rem 0'>{badge(r.get('priority'))} priority "
                f"&nbsp; {badge(r.get('risk_level'))} risk</div>{rationale}{first_step}"
                f"<div class='muted'>Expected impact: {esc(', '.join(r.get('expected_impact', [])))}</div>"
                f"{risk_html}{evidence_html(r.get('supporting_evidence', []))}")
        card(f"<span class='card-title'>{esc(r['recommendation'])}</span>", body)


def render_briefing(d: dict) -> None:
    b = d.get("briefing", {})
    if not b:
        not_ready_note()
        return
    for title, key in [("What happened?", "what_happened"),
                       ("Why does it matter?", "why_it_matters"),
                       ("What should management do next?", "what_next")]:
        st.markdown(f"<div class='brief-card'><div class='brief-q'>{title}</div>"
                    f"<div class='brief-a'>{esc(strip_src_refs(b.get(key, '')))}</div></div>", unsafe_allow_html=True)


# -------------------------------------------------------------------- data ---
data = load_dashboard_data()
c = data["company"]

st.markdown(f"<div class='hero'>AI Strategy Consultant — {esc(c['name'])}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub'>{esc(c['industry'])}</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### AI Strategy Consultant")
    st.caption(f"Last update {str(c['last_update'])[:16].replace('T', ' ')}")
    if st.button("Re-analyse now", use_container_width=True):
        with st.spinner("Re-analysing…"):
            try:
                regenerate_data()
                st.rerun()
            except Exception as exc:
                st.error(f"Re-analyse failed — is the model server up?\n\n{exc}")
    if st.button("Collect fresh data + re-analyse", use_container_width=True,
                 help="Pull new live documents, re-index, then re-analyse — updates every tab"):
        with st.spinner("Collecting fresh data and analysing…"):
            try:
                regenerate_data(full=True)
                st.rerun()
            except Exception as exc:
                st.error(f"Re-run failed — is the model server up?\n\n{exc}")
    with st.expander("Manage data"):
        st.caption("Delete the corpus, index and results. The agent rebuilds them on the next question.")
        if st.checkbox("Confirm delete", key="confirm_del") and \
                st.button("Delete all collected data", use_container_width=True):
            import shutil
            get_retriever.clear()
            shutil.rmtree(ROOT / "data", ignore_errors=True)
            for f in (ROOT / "results").glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            st.success("Deleted. Ask the agent to rebuild.")
            st.rerun()
    st.divider()
    st.markdown("**Models & Retrieval**")
    st.markdown(f"<div class='stack'><b>LLM</b><br><code>{esc(LLM_MODEL)}</code></div>"
                f"<div class='stack'><b>Embeddings</b><br><code>{esc(EMBED_MODEL)}</code></div>"
                f"<div class='stack'><b>RAG tool</b><br><code>{esc(RAG_TOOL)}</code></div>",
                unsafe_allow_html=True)

tabs = st.tabs(["Ask the Consultant", "Overview", "Opportunities", "Risks", "Trends",
                "Sentiment", "Recommendations", "Briefing", "Retrieval", "Activity"])

# ---- Tab: Ask the Consultant (interactive agent) ----------------------------
with tabs[0]:
    st.markdown("#### Ask the AI Strategy Consultant anything strategic")
    st.markdown("<div class='muted'>The agent reasons, calls tools for evidence, answers, then offers a "
                "menu of strategic options. Asking also re-runs the analysis, so every other tab updates "
                "from the same fresh evidence — and the whole state is saved to the Activity log.</div>",
                unsafe_allow_html=True)

    queued = st.session_state.pop("queued_q", None)          # set by a follow-up button
    q = st.text_input("question", value=queued or "", key="ask_q",
                      placeholder="e.g. How do we beat BYD in China?", label_visibility="collapsed")

    if (st.button("Ask the Consultant", type="primary") or queued) and q.strip():
        buf = io.StringIO()
        timer_box = st.empty()        # live elapsed counter, sits next to the spinner
        with st.spinner("Reasoning, calling tools, and refreshing every tab…"):
            try:
                from main import ask_ceo, strategic_options
                if not config.CORPUS_CSV.exists():       # first use -> agent builds the knowledge base
                    regenerate_data()
                with live_timer(timer_box):
                    t0 = time.perf_counter()
                    with contextlib.redirect_stdout(buf):
                        answer = ask_ceo(q.strip())
                    agent_secs = round(time.perf_counter() - t0, 1)   # the agent loop itself
                    opts = strategic_options(q.strip(), answer)
                    # full live trace: each tool call AND what it returned
                    trace = [l.rstrip() for l in buf.getvalue().splitlines()
                             if "calling tool:" in l or l.lstrip().startswith("->")]
                    try:
                        regenerate_data()       # LLM was called -> refresh every other tab
                    except Exception:
                        pass
                    snapshot = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else {}
                    log_activity({"time": datetime.now().isoformat(timespec="minutes"),
                                  "question": q.strip(), "answer": answer, "elapsed": agent_secs,
                                  "options": opts["options"], "snapshot": snapshot})
                    st.session_state["last_result"] = {"answer": answer, "trace": trace,
                                                       "elapsed": agent_secs, **opts}
                st.rerun()
            except Exception as exc:
                st.error(f"Agent unavailable — is the model server running?\n\n{exc}")

    res = st.session_state.get("last_result")
    if res:
        st.markdown(f"<div class='answer'>{esc(res['answer'])}</div>", unsafe_allow_html=True)
        if res.get("elapsed") is not None:
            st.caption(f"⏱ Agent loop ran in {res['elapsed']}s")
        if res.get("trace"):
            n_calls = sum(1 for l in res["trace"] if "calling tool:" in l)
            with st.expander(f"Tool calls & outputs ({n_calls})", expanded=True):
                st.code("\n".join(res["trace"]), language="text")
        if res["options"]:
            st.markdown("##### Your strategic options")
            for o in res["options"]:
                path, _, upside = o.partition("|")
                st.markdown(f"<div class='card'><span class='card-title'>{esc(path.strip())}</span>"
                            f"<div class='desc'>{esc(upside.strip())}</div></div>", unsafe_allow_html=True)
        if res["followups"]:
            st.markdown("##### Dig deeper")
            for i, fu in enumerate(res["followups"]):
                if st.button(fu, key=f"fu_{i}"):
                    st.session_state.queued_q = fu
                    st.rerun()

with tabs[1]:
    render_overview(data)
with tabs[2]:
    render_opportunities(data)
with tabs[3]:
    render_risks(data)
with tabs[4]:
    render_trends(data)
with tabs[5]:
    render_sentiment(data)
with tabs[6]:
    render_recommendations(data)
with tabs[7]:
    render_briefing(data)

# ---- Tab: Retrieval (Hybrid RAG — top-ranked results) -----------------------
with tabs[8]:
    st.markdown("#### Hybrid RAG Retrieval")
    st.markdown("<div class='muted'>BM25 (sparse) + dense embeddings, fused and min-max normalised "
                "(score = 0.5·dense + 0.5·sparse). Enter a query to see the top-ranked evidence.</div>",
                unsafe_allow_html=True)
    rq = st.text_input("retrieval", placeholder="e.g. Tesla battery supply chain risk",
                       label_visibility="collapsed", key="rq")
    if st.button("Retrieve top results", key="ret_btn") and rq.strip():
        with st.spinner("Hybrid retrieval…"):
            try:
                hits = get_retriever().retrieve(rq.strip())
                st.dataframe(
                    pd.DataFrame([{"rank": h["rank"], "fused": h["score"], "dense": h["dense"],
                                   "sparse": h["sparse"], "type": h["source_type"],
                                   "source": h["source"], "title": h["title"][:70]} for h in hits]),
                    hide_index=True, use_container_width=True)
                for h in hits:
                    st.markdown(
                        f"<div class='card'><div class='flex-between'>"
                        f"<span class='card-title'>#{h['rank']} · {esc(h['title'])}</span>"
                        f"<span class='badge b-low'>fused {h['score']}</span></div>"
                        f"<div class='muted'>{esc(h['source'])} · {esc(h['source_type'])} · "
                        f"dense {h['dense']} · sparse {h['sparse']}"
                        + (f" · <a href='{esc(h['url'])}' target='_blank'>source</a>" if h.get('url') else "")
                        + f"</div><div class='desc'>{esc(h['text'][:320])}</div></div>", unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Retrieval failed — is the index built? Run `python main.py ingest`.\n\n{exc}")

# ---- Tab: Activity (each entry stores a full dashboard snapshot) ------------
with tabs[9]:
    st.markdown("#### Activity log")
    st.markdown("<div class='muted'>Every question is saved with a full snapshot of every tab at that "
                "moment. Pick one to replay the dashboard exactly as it was.</div>", unsafe_allow_html=True)
    activity = load_activity()
    if not activity:
        st.info("No questions yet. Ask the Consultant something on the first tab to start the log.")
    else:
        labels = [f"{it.get('time', '')} — {it['question'][:60]}" for it in activity]
        idx = st.selectbox("Logged question", range(len(activity)), format_func=lambda i: labels[i])
        it = activity[idx]
        st.markdown(f"<div class='answer'>{esc(it.get('answer', ''))}</div>", unsafe_allow_html=True)
        if it.get("elapsed") is not None:
            st.caption(f"⏱ Agent loop ran in {it['elapsed']}s")
        for o in it.get("options", []):
            path, _, upside = o.partition("|")
            st.markdown(f"<div class='card'><span class='card-title'>{esc(path.strip())}</span>"
                        f"<div class='desc'>{esc(upside.strip())}</div></div>", unsafe_allow_html=True)
        snap = it.get("snapshot")
        if snap:
            st.markdown("##### Dashboard snapshot at that time")
            sub = st.tabs(["Overview", "Opportunities", "Risks", "Trends",
                           "Sentiment", "Recommendations", "Briefing"])
            with sub[0]:
                render_overview(snap)
            with sub[1]:
                render_opportunities(snap)
            with sub[2]:
                render_risks(snap)
            with sub[3]:
                render_trends(snap)
            with sub[4]:
                render_sentiment(snap)
            with sub[5]:
                render_recommendations(snap)
            with sub[6]:
                render_briefing(snap)
        else:
            st.caption("No snapshot stored for this entry.")
