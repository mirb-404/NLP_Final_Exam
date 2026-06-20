"""
Executive Intelligence Dashboard (Streamlit) — Deliverable 2.

A thin renderer of results/dashboard_data.json: every PDF dashboard section maps
to one key in that file (the backend already shaped it). Run the backend first:

    python main.py report           # writes results/dashboard_data.json
    uv run streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "dashboard_data.json"

_LEVEL = {"high": "🔴 High", "medium": "🟡 Medium", "low": "🟢 Low"}


def badge(level: str) -> str:
    return _LEVEL.get(str(level).lower(), level)


def show_evidence(evidence: list[dict]) -> None:
    for e in evidence:
        line = f"`[{e.get('ref', 'src')}]` **{e.get('title', '')}** — {e.get('source', '')}"
        if e.get("score") is not None:
            line += f"  ·  score {e['score']}"
        st.markdown(line)
        if e.get("url"):
            st.caption(e["url"])


st.set_page_config(page_title="AI CEO — Strategic Intelligence", layout="wide")

if not DATA.exists():
    st.error(f"No data yet. Run `python main.py report` first.\nExpected: {DATA}")
    st.stop()

data = json.loads(DATA.read_text(encoding="utf-8"))

# ---- Sidebar: refresh + interactive agent -----------------------------------
with st.sidebar:
    st.header("🧠 AI CEO")
    if st.button("🔄 Reload data"):
        st.rerun()
    st.divider()
    st.subheader("Ask the agent")
    question = st.text_input("Strategic question", placeholder="What are our biggest risks?")
    if st.button("Ask") and question:
        sys.path.insert(0, str(ROOT))
        with st.spinner("Agent reasoning + calling tools…"):
            try:
                from main import ask_ceo
                st.success(ask_ceo(question))
            except Exception as exc:
                st.error(f"Agent unavailable (is the model server running?): {exc}")

# ---- Section 1: Company Overview --------------------------------------------
c = data["company"]
st.title(f"AI CEO — {c['name']}")
o = st.columns(4)
o[0].metric("Industry", c["industry"])
o[1].metric("Documents", c["n_documents"])
o[2].metric("Data sources", c["n_sources"])
o[3].metric("Last update", str(c["last_update"])[:16].replace("T", " "))
st.divider()

# ---- Section 2: Market Intelligence -----------------------------------------
st.header("📰 Market Intelligence")
mi = data.get("market_intelligence", {})
left, right = st.columns(2)
with left:
    st.subheader("Recent news")
    for n in mi.get("recent_news", []):
        st.markdown(f"- [{n['title']}]({n['url']}) — *{n['source']}*" if n.get("url")
                    else f"- {n['title']} — *{n['source']}*")
with right:
    st.subheader("Competitor activity")
    for e in mi.get("competitor_activity", []):
        st.markdown(f"- **{e.get('title', '')}** — {e.get('source', '')}")
st.caption("Trending keywords: " + ", ".join(mi.get("keywords", [])))
st.divider()

# ---- Section 3: Opportunity Monitor -----------------------------------------
st.header("🚀 Opportunity Monitor")
for opp in data.get("opportunities", []):
    with st.expander(f"{badge(opp.get('impact'))}  ·  {opp['title']}  ·  confidence {opp.get('confidence', 0)}"):
        show_evidence(opp.get("evidence", []))
st.divider()

# ---- Section 4: Risk Monitor ------------------------------------------------
st.header("⚠️ Risk Monitor")
for risk in data.get("risks", []):
    header = f"{badge(risk.get('severity'))}  ·  {risk['title']}  ·  {risk.get('category', '')}  ·  confidence {risk.get('confidence', 0)}"
    with st.expander(header):
        show_evidence(risk.get("evidence", []))
st.divider()

# ---- Section 5: Sentiment Analysis ------------------------------------------
st.header("📊 Sentiment Analysis")
s = data.get("sentiment", {})
sc = st.columns(3)
sc[0].metric("News sentiment", s.get("news_sentiment", 0))
sc[1].metric("Public sentiment", s.get("public_sentiment", 0))
sc[2].metric("Overall", s.get("overall_sentiment", 0))
viz = st.columns(2)
with viz[0]:
    st.caption("Distribution")
    dist = s.get("distribution", {})
    if dist:
        st.bar_chart(pd.Series(dist, name="documents"))
with viz[1]:
    st.caption("Sentiment trend")
    trend = s.get("trend", [])
    if trend:
        st.line_chart(pd.DataFrame(trend).set_index("period")["sentiment"])
st.divider()

# ---- Section 6: Strategic Recommendations -----------------------------------
st.header("✅ Strategic Recommendations")
for r in data.get("recommendations", []):
    with st.container(border=True):
        st.markdown(f"### {r['recommendation']}")
        m = st.columns(2)
        m[0].markdown(f"**Priority:** {badge(r.get('priority'))}")
        m[1].markdown(f"**Risk level:** {badge(r.get('risk_level'))}")
        st.markdown("**Expected impact:** " + ", ".join(r.get("expected_impact", [])))
        st.markdown("**Supporting evidence:**")
        show_evidence(r.get("supporting_evidence", []))
st.divider()

# ---- Section 7: CEO Briefing ------------------------------------------------
st.header("📝 CEO Briefing")
b = data.get("briefing", {})
st.subheader("What happened?")
st.write(b.get("what_happened", ""))
st.subheader("Why does it matter?")
st.write(b.get("why_it_matters", ""))
st.subheader("What should management do next?")
st.write(b.get("what_next", ""))
