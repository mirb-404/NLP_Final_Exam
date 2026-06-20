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
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "dashboard_data.json"

st.set_page_config(page_title="AI CEO — Strategic Intelligence", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------- styling ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
#MainMenu, footer, header {visibility:hidden;}
.block-container{padding-top:1.4rem; max-width:1180px;}
.hero{font-size:1.9rem;font-weight:700;color:#16243f;margin:.1rem 0 .2rem;}
.sub{color:#7a8595;margin-bottom:1rem;}
.row{display:flex;gap:.8rem;flex-wrap:wrap;margin:.3rem 0 1rem;}
.metric{flex:1;min-width:150px;background:#fff;border:1px solid #e7e9ee;border-radius:14px;
        padding:.9rem 1.1rem;box-shadow:0 1px 3px rgba(0,0,0,.04);}
.m-val{font-size:1.45rem;font-weight:700;color:#16243f;}
.m-lbl{font-size:.74rem;color:#8a93a3;text-transform:uppercase;letter-spacing:.05em;}
.card{background:#fff;border:1px solid #e7e9ee;border-radius:14px;padding:1rem 1.2rem;
      margin-bottom:.8rem;box-shadow:0 1px 3px rgba(0,0,0,.04);}
.card-title{font-weight:600;font-size:1.02rem;color:#16243f;}
.muted{font-size:.82rem;color:#8a93a3;margin:.15rem 0 .5rem;}
.badge{display:inline-block;padding:.14rem .65rem;border-radius:999px;font-size:.74rem;font-weight:600;white-space:nowrap;}
.b-high{background:#fde8e8;color:#c0392b;} .b-med{background:#fff3e0;color:#b9770e;} .b-low{background:#e6f6ec;color:#1e7e45;}
.evidence{font-size:.83rem;color:#5a6472;border-left:3px solid #d9dee7;padding-left:.7rem;margin:.26rem 0;}
.answer{background:linear-gradient(135deg,#f3f6ff,#eef2ff);border:1px solid #dbe2ff;border-radius:16px;
        padding:1.3rem 1.5rem;font-size:1.03rem;line-height:1.65;color:#1f2d45;}
.brief-q{font-weight:600;font-size:1.08rem;color:#16243f;margin:1.1rem 0 .25rem;}
.brief-a{color:#ff6b6b;line-height:1.7;font-size:1.0rem;}
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


# -------------------------------------------------------------------- data ---
if not DATA.exists():
    st.error(f"No data yet. Run `python main.py report` first.\n\nExpected: {DATA}")
    st.stop()

data = json.loads(DATA.read_text(encoding="utf-8"))
c = data["company"]

st.markdown(f"<div class='hero'>🧠 AI CEO — {esc(c['name'])}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub'>{esc(c['industry'])}</div>", unsafe_allow_html=True)

tabs = st.tabs(["💬 Ask the CEO", "📊 Overview", "🚀 Opportunities", "⚠️ Risks",
                "📈 Sentiment", "✅ Recommendations", "📝 Briefing"])

# ---- Tab: Ask the CEO (interactive agent) -----------------------------------
with tabs[0]:
    st.markdown("#### Ask the AI CEO anything strategic")
    st.markdown("<div class='muted'>The agent reasons, calls tools to fetch real evidence, then answers.</div>",
                unsafe_allow_html=True)
    q = st.text_input("question", placeholder="e.g. How do we beat Samsung in wearables?",
                      label_visibility="collapsed")
    if st.button("Ask the CEO  →", type="primary") and q.strip():
        sys.path.insert(0, str(ROOT))
        buf = io.StringIO()
        with st.spinner("Reasoning and calling tools…"):
            try:
                from main import ask_ceo
                with contextlib.redirect_stdout(buf):
                    answer = ask_ceo(q.strip())
                st.markdown(f"<div class='answer'>{esc(answer)}</div>", unsafe_allow_html=True)
                tool_lines = [l.strip() for l in buf.getvalue().splitlines() if "calling tool" in l]
                if tool_lines:
                    with st.expander(f"🔧 {len(tool_lines)} tool calls"):
                        st.code("\n".join(tool_lines), language="text")
            except Exception as exc:
                st.error(f"Agent unavailable — is the model server running?\n\n{exc}")

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
        st.markdown("##### 📰 Recent news")
        for n in mi.get("recent_news", []):
            link = f"[{n['title']}]({n['url']})" if n.get("url") else n["title"]
            st.markdown(f"- {link}  ·  *{n.get('source', '')}*")
    with right:
        st.markdown("##### 🏁 Competitor activity")
        for e in mi.get("competitor_activity", []):
            st.markdown(f"- **{e.get('title', '')}**  ·  *{e.get('source', '')}*")
    if mi.get("keywords"):
        st.caption("Trending: " + " · ".join(mi["keywords"]))

# ---- Tab: Opportunities (Section 3) -----------------------------------------
with tabs[2]:
    for opp in data.get("opportunities", []):
        card(f"<span class='card-title'>{esc(opp['title'])}</span>{badge(opp.get('impact'))}",
             f"<div class='muted'>Confidence {esc(opp.get('confidence', 0))}</div>{evidence_html(opp.get('evidence', []))}")

# ---- Tab: Risks (Section 4) -------------------------------------------------
with tabs[3]:
    for r in data.get("risks", []):
        card(f"<span class='card-title'>{esc(r['title'])}</span>{badge(r.get('severity'))}",
             f"<div class='muted'>{esc(r.get('category', ''))} · confidence {esc(r.get('confidence', 0))}</div>"
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
        body = (f"<div style='margin:.3rem 0'>{badge(r.get('priority'))} priority "
                f"&nbsp; {badge(r.get('risk_level'))} risk</div>"
                f"<div class='muted'>Expected impact: {esc(', '.join(r.get('expected_impact', [])))}</div>"
                f"{evidence_html(r.get('supporting_evidence', []))}")
        card(f"<span class='card-title'>{esc(r['recommendation'])}</span>", body)

# ---- Tab: Briefing (Section 7) ----------------------------------------------
with tabs[6]:
    b = data.get("briefing", {})
    for title, key in [("What happened?", "what_happened"),
                       ("Why does it matter?", "why_it_matters"),
                       ("What should management do next?", "what_next")]:
        st.markdown(f"<div class='brief-q'>{title}</div>"
                    f"<div class='brief-a'>{esc(b.get(key, ''))}</div>",
                    unsafe_allow_html=True)
