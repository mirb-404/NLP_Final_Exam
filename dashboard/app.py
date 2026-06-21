"""
Executive Intelligence Dashboard (Streamlit) — Deliverable 2.

Thin, styled renderer of results/dashboard_data.json. Every PDF dashboard section
is a tab; the backend already shaped the data. Run the backend first:

    python main.py report                       # writes results/dashboard_data.json
    uv run streamlit run dashboard/app.py
"""

import contextlib
import html
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "dashboard_data.json"
ACTIVITY = ROOT / "results" / "activity.json"   # grows as the CEO asks questions


def load_activity() -> list:
    return json.loads(ACTIVITY.read_text(encoding="utf-8")) if ACTIVITY.exists() else []


def log_activity(entry: dict) -> None:
    items = load_activity()
    items.insert(0, entry)
    ACTIVITY.write_text(json.dumps(items[:50], indent=2, ensure_ascii=False), encoding="utf-8")

sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402  (model ids for the tech-stack panel)

LLM_MODEL = config.LLM_REPO_ID
EMBED_MODEL = config.EMBEDDING_MODEL
RAG_TOOL = "search_knowledge_base — Hybrid BM25 + dense"

st.set_page_config(page_title="AI CEO — Strategic Intelligence", layout="wide")

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
.metric{flex:1;min-width:150px;background:#1b2334;border:1px solid #2a3550;border-radius:14px;
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

# ----------------------------------------------------------------- helpers ---
def esc(x) -> str:
    return html.escape(str(x))


def badge(level) -> str:
    cls = {"high": "b-high", "medium": "b-med", "low": "b-low"}.get(str(level).lower(), "b-med")
    return f"<span class='badge {cls}'>{esc(level)}</span>"


def metric(label, value) -> str:
    return f"<div class='metric'><div class='m-val'>{esc(value)}</div><div class='m-lbl'>{esc(label)}</div></div>"


def evidence_html(evidence) -> str:
    return "".join(
        f"<div class='evidence'>[{esc(e.get('ref', 'src'))}] {esc(e.get('title', ''))} — <i>{esc(e.get('source', ''))}</i></div>"
        for e in evidence
    ) or "<div class='evidence muted'>no evidence</div>"


def card(title_html: str, body_html: str) -> None:
    st.markdown(f"<div class='card'><div class='flex-between'>{title_html}</div>{body_html}</div>",
                unsafe_allow_html=True)


@st.cache_resource
def get_retriever():
    """Load the hybrid retriever once (reads the Chroma index)."""
    sys.path.insert(0, str(ROOT))
    from src.retriever_hybrid import HybridRetriever
    return HybridRetriever()


# ---- analysis data: refreshed whenever the agent calls the LLM --------------
def regenerate_data(full: bool = False) -> None:
    """Run the analysis pipeline so every tab reflects the latest data. Called after each
    agent question (LLM call), so asking also refreshes opportunities / risks /
    recommendations / briefing. full=True re-collects fresh documents first."""
    from src.knowledge_base import count as kb_count
    from src.orchestrator import run_analyze, run_pipeline
    try:
        has_index = kb_count() > 0
    except Exception:
        has_index = False
    run_pipeline() if (full or not has_index) else run_analyze()


def load_dashboard_data() -> dict:
    """Load the latest analysis; generate it once if it does not exist yet."""
    if not DATA.exists():
        regenerate_data()
    return json.loads(DATA.read_text(encoding="utf-8"))


# -------------------------------------------------------------------- data ---
try:
    data = load_dashboard_data()
except Exception as exc:
    st.error("Could not load analysis. Make sure the model server is running and the index "
             f"is built (`python main.py ingest`).\n\n{exc}")
    st.stop()
c = data["company"]

st.markdown(f"<div class='hero'>AI CEO — {esc(c['name'])}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub'>{esc(c['industry'])}</div>", unsafe_allow_html=True)

# ---- Sidebar: tech stack (visible on every tab) -----------------------------
with st.sidebar:
    st.markdown("### AI CEO")
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
                st.success("All tabs updated with fresh data.")
                st.rerun()
            except Exception as exc:
                st.error(f"Re-run failed — is the model server up?\n\n{exc}")
    st.divider()
    st.markdown("**Models & Retrieval**")
    st.markdown(f"<div class='stack'><b>LLM</b><br><code>{esc(LLM_MODEL)}</code></div>"
                f"<div class='stack'><b>Embeddings</b><br><code>{esc(EMBED_MODEL)}</code></div>"
                f"<div class='stack'><b>RAG tool</b><br><code>{esc(RAG_TOOL)}</code></div>",
                unsafe_allow_html=True)

tabs = st.tabs(["Ask the CEO", "Overview", "Opportunities", "Risks",
                "Sentiment", "Recommendations", "Briefing", "Retrieval", "Activity"])

# ---- Tab: Ask the CEO (interactive agent) -----------------------------------
with tabs[0]:
    st.markdown("#### Ask the AI CEO anything strategic")
    st.markdown("<div class='muted'>The agent reasons, calls tools for evidence, answers, then offers a "
                "menu of strategic options. Asking also re-runs the analysis, so every other tab "
                "updates from the same fresh evidence.</div>", unsafe_allow_html=True)

    queued = st.session_state.pop("queued_q", None)          # set by a follow-up button
    q = st.text_input("question", value=queued or "", key="ask_q",
                      placeholder="e.g. How do we beat BYD in China?", label_visibility="collapsed")

    if (st.button("Ask the CEO", type="primary") or queued) and q.strip():
        buf = io.StringIO()
        with st.spinner("Reasoning, calling tools, and refreshing every tab…"):
            try:
                from main import ask_ceo, strategic_options
                with contextlib.redirect_stdout(buf):
                    answer = ask_ceo(q.strip())
                opts = strategic_options(q.strip(), answer)
                tool_lines = [l.strip() for l in buf.getvalue().splitlines() if "calling tool" in l]
                log_activity({"time": datetime.now().isoformat(timespec="minutes"),
                              "question": q.strip(), "answer": answer, "options": opts["options"]})
                # The LLM was called -> re-run the analysis so all other tabs update too.
                try:
                    regenerate_data()
                except Exception:
                    pass
                st.session_state["last_result"] = {"answer": answer, "tools": tool_lines, **opts}
                st.rerun()
            except Exception as exc:
                st.error(f"Agent unavailable — is the model server running?\n\n{exc}")

    res = st.session_state.get("last_result")
    if res:
        st.markdown(f"<div class='answer'>{esc(res['answer'])}</div>", unsafe_allow_html=True)
        if res["tools"]:
            with st.expander(f"{len(res['tools'])} tool calls"):
                st.code("\n".join(res["tools"]), language="text")
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

# ---- Tab: Overview (Section 1 + 2) ------------------------------------------
with tabs[1]:
    st.markdown("<div class='row'>" + "".join([
        metric("Documents", c["n_documents"]),
        metric("Data sources", c["n_sources"]),
        metric("Industry", c["industry"]),
        metric("Last update", str(c["last_update"])[:16].replace("T", " ")),
    ]) + "</div>", unsafe_allow_html=True)

    mi = data.get("market_intelligence", {})
    left, right = st.columns(2)
    with left:
        st.markdown("##### Recent news")
        for n in mi.get("recent_news", []):
            link = f"[{n['title']}]({n['url']})" if n.get("url") else n["title"]
            st.markdown(f"- {link}  ·  *{n.get('source', '')}*")
    with right:
        st.markdown("##### Competitor activity")
        for e in mi.get("competitor_activity", []):
            st.markdown(f"- **{e.get('title', '')}**  ·  *{e.get('source', '')}*")
    if mi.get("keywords"):
        st.caption("Trending: " + " · ".join(mi["keywords"]))

# ---- Tab: Opportunities (Section 3) -----------------------------------------
with tabs[2]:
    for opp in data.get("opportunities", []):
        desc = f"<div class='desc'>{esc(opp['description'])}</div>" if opp.get("description") else ""
        card(f"<span class='card-title'>{esc(opp['title'])}</span>{badge(opp.get('impact'))}",
             f"{desc}<div class='muted'>Confidence {esc(opp.get('confidence', 0))}</div>"
             f"{evidence_html(opp.get('evidence', []))}")

# ---- Tab: Risks (Section 4) -------------------------------------------------
with tabs[3]:
    for r in data.get("risks", []):
        desc = f"<div class='desc'>{esc(r['description'])}</div>" if r.get("description") else ""
        card(f"<span class='card-title'>{esc(r['title'])}</span>{badge(r.get('severity'))}",
             f"{desc}<div class='muted'>{esc(r.get('category', ''))} · confidence {esc(r.get('confidence', 0))}</div>"
             f"{evidence_html(r.get('evidence', []))}")

# ---- Tab: Sentiment (Section 5) ---------------------------------------------
with tabs[4]:
    s = data.get("sentiment", {})
    st.markdown("<div class='row'>" + "".join([
        metric("News sentiment", s.get("news_sentiment", 0)),
        metric("Public sentiment", s.get("public_sentiment", 0)),
        metric("Overall", s.get("overall_sentiment", 0)),
    ]) + "</div>", unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("##### Distribution")
        if s.get("distribution"):
            st.bar_chart(pd.Series(s["distribution"], name="documents"))
    with v2:
        st.markdown("##### Sentiment trend")
        if s.get("trend"):
            st.line_chart(pd.DataFrame(s["trend"]).set_index("period")["sentiment"])

# ---- Tab: Recommendations (Section 6) ---------------------------------------
with tabs[5]:
    for r in data.get("recommendations", []):
        rationale = f"<div class='desc'>{esc(r['rationale'])}</div>" if r.get("rationale") else ""
        first_step = f"<div class='step'>▶ First step: {esc(r['first_step'])}</div>" if r.get("first_step") else ""
        body = (f"<div style='margin:.35rem 0'>{badge(r.get('priority'))} priority "
                f"&nbsp; {badge(r.get('risk_level'))} risk</div>"
                f"{rationale}{first_step}"
                f"<div class='muted'>Expected impact: {esc(', '.join(r.get('expected_impact', [])))}</div>"
                f"{evidence_html(r.get('supporting_evidence', []))}")
        card(f"<span class='card-title'>{esc(r['recommendation'])}</span>", body)

# ---- Tab: Briefing (Section 7) ----------------------------------------------
with tabs[6]:
    b = data.get("briefing", {})
    for title, key in [("What happened?", "what_happened"),
                       ("Why does it matter?", "why_it_matters"),
                       ("What should management do next?", "what_next")]:
        st.markdown(f"<div class='brief-card'><div class='brief-q'>{title}</div>"
                    f"<div class='brief-a'>{esc(b.get(key, ''))}</div></div>",
                    unsafe_allow_html=True)

# ---- Tab: Retrieval (Hybrid RAG — top-ranked results) -----------------------
with tabs[7]:
    st.markdown("#### Hybrid RAG Retrieval")
    st.markdown("<div class='muted'>BM25 (sparse) + dense embeddings, fused and min-max normalised "
                "(score = 0.5·dense + 0.5·sparse). Enter a query to see the top-ranked evidence "
                "the agents retrieve.</div>", unsafe_allow_html=True)
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
                        f"dense {h['dense']} · sparse {h['sparse']}</div>"
                        f"<div class='desc'>{esc(h['text'][:320])}</div></div>", unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Retrieval failed — is the index built? Run `python main.py ingest`.\n\n{exc}")

# ---- Tab: Activity (grows as the CEO asks questions) ------------------------
with tabs[8]:
    st.markdown("#### Activity log")
    st.markdown("<div class='muted'>Every question you ask the agent is logged here, newest first — "
                "a running record of decisions explored.</div>", unsafe_allow_html=True)
    activity = load_activity()
    if not activity:
        st.info("No questions yet. Ask the CEO something on the first tab to start the log.")
    for it in activity:
        opts = "".join(f"<div class='evidence'>{esc(o)}</div>" for o in it.get("options", []))
        card(f"<span class='card-title'>{esc(it['question'])}</span>"
             f"<span class='muted'>{esc(it.get('time', ''))}</span>",
             f"<div class='desc'>{esc(it.get('answer', ''))}</div>{opts}")
