"""
Task 4 — Strategic Intelligence Engine.

Per theme (opportunities / risks / trends / competitor_activity): retrieve evidence
(hybrid BM25 + dense), ask the LLM for concrete items grounded in it, then attach the
cited evidence + a retrieval-derived confidence. Output -> results/intelligence.json
"""

import re

from src import config
from src.classical_agent import top_keywords
from src.preprocess import load_corpus
from src.retriever_hybrid import HybridRetriever
from src.utils import ask_llm, save_json, strip_src_refs

EVIDENCE_PER_ITEM = 3


def _evidence_block(docs: list[dict]) -> str:
    """Format retrieved docs as a numbered, citable context block."""
    lines = []
    for d in docs:
        preview = d["text"][: config.CHUNK_PREVIEW]
        lines.append(f"[src-{d['rank']}] ({d['source']}) {d['title']}. {preview}")
    return "\n".join(lines)


_SRC_REF_RE = re.compile(r"src-(\d+)", re.IGNORECASE)


def _evidence_list(docs: list[dict], cited_text: str = "") -> list[dict]:
    """Compact evidence objects for the dashboard / recommendations.

    Always shows the top EVIDENCE_PER_ITEM ranked docs, but extends the list far
    enough to cover the highest [src-N] the item text actually cites — so a
    sentence that references src-5 renders src-5 instead of dangling at src-3."""
    n = EVIDENCE_PER_ITEM
    refs = [int(m) for m in _SRC_REF_RE.findall(cited_text or "")]
    if refs:
        n = max(n, min(max(refs), len(docs)))
    return [
        {"ref": f"src-{d['rank']}", "title": d["title"], "source": d["source"],
         "url": d["url"], "score": d["score"]}
        for d in docs[:n]
    ]


def _confidence(docs: list[dict]) -> float:
    """Mean fused retrieval score of the supporting evidence (already in [0,1])."""
    top = docs[:EVIDENCE_PER_ITEM]
    return round(sum(d["score"] for d in top) / len(top), 3) if top else 0.0


_PREAMBLE_RE = re.compile(
    r"^(here (are|is)|based on|the following|sure|below are|these are|as an? )", re.IGNORECASE
)


def _parse_lines(raw: str) -> list[str]:
    """Pull clean items out of an LLM numbered/bulleted list."""
    items = []
    for line in raw.splitlines():
        # strip leading list markers: "1.", "1)", "[1]", "-", "*", "•"
        line = re.sub(r"^\s*(?:\[?\d+[\].)]|[-*•])\s*", "", line).strip()
        line = line.strip("*").strip()  # drop markdown bold
        # skip blank lines and lines that just echo the format template
        if len(line) <= 3:
            continue
        if re.fullmatch(r"(?:high|medium|low|\s|::|\|)+", line, flags=re.IGNORECASE):
            continue
        # skip LLM framing lines: "Here are 5 trends...:", "Based on the evidence:"
        if line.endswith(":") or _PREAMBLE_RE.match(line):
            continue
        items.append(line)
    return items[:5]


_LEVELS = ("high", "medium", "low")
_RISK_CATS = ("competitive", "regulatory", "sentiment", "supply chain", "financial")


def _split_fields(line: str) -> list[str]:
    """Split an LLM list item into atomic fields, tolerating both '::' and '|'
    separators — the model often merges 'category | severity' with a pipe instead
    of two '::', which a positional parser would misalign."""
    return [p.strip() for p in re.split(r"::|\|", line) if p.strip()]


def _pick_level(fields: list[str], default: str = "Medium") -> str:
    """First High/Medium/Low rating among the fields (exact field, else leading word)."""
    for f in fields:
        if f.lower() in _LEVELS:
            return f.capitalize()
    for f in fields:
        head = f.lower().split()
        if head and head[0] in _LEVELS:
            return head[0].capitalize()
    return default


def _pick_from(fields: list[str], vocab, default: str) -> str:
    """First field that names one of `vocab` (substring match)."""
    for f in fields:
        fl = f.lower()
        for v in vocab:
            if v in fl:
                return v
    return default


def _description(fields: list[str], tags) -> str:
    """Recombine the free-text fields (everything that isn't a rating/category tag)
    into the full sentence, so a description split by a stray separator stays whole."""
    free = [f for f in fields if f.lower() not in tags]
    return " ".join(free)


def detect_opportunities(retriever: HybridRetriever) -> list[dict]:
    docs = retriever.retrieve(config.ENGINE_QUERIES["opportunities"])
    prompt = (
        f"You are a strategy analyst for {config.COMPANY}. Using ONLY the evidence "
        f"below, list up to 5 concrete business OPPORTUNITIES.\nOne per line, format:\n"
        f"<short opportunity title> :: <High|Medium|Low> :: <one full sentence on the opportunity and why it matters>\n\n"
        f"EVIDENCE:\n{_evidence_block(docs)}\n\nOPPORTUNITIES:"
    )
    items = []
    for line in _parse_lines(ask_llm(prompt)):
        fields = _split_fields(line)
        if not fields:
            continue  # drop malformed lines that carry no actual opportunity
        title, rest = fields[0], fields[1:]
        items.append({
            "title": strip_src_refs(title),
            "impact": _pick_level(rest),
            "description": strip_src_refs(_description(rest, _LEVELS)),
            "confidence": _confidence(docs),
            "evidence": _evidence_list(docs, line),
        })
    return items


def detect_risks(retriever: HybridRetriever) -> list[dict]:
    docs = retriever.retrieve(config.ENGINE_QUERIES["risks"])
    prompt = (
        f"You are a risk analyst for {config.COMPANY}. Using ONLY the evidence below, "
        f"list up to 5 concrete RISKS.\nOne per line, format:\n"
        f"<short risk title> :: <competitive|regulatory|sentiment|supply chain|financial> :: "
        f"<High|Medium|Low> :: <one full sentence on the risk and its impact>\n\n"
        f"EVIDENCE:\n{_evidence_block(docs)}\n\nRISKS:"
    )
    items = []
    for line in _parse_lines(ask_llm(prompt)):
        fields = _split_fields(line)
        if not fields:
            continue  # drop malformed lines that carry no actual risk
        title, rest = fields[0], fields[1:]
        items.append({
            "title": strip_src_refs(title),
            "category": _pick_from(rest, _RISK_CATS, "competitive"),
            "severity": _pick_level(rest),
            "description": strip_src_refs(_description(rest, _LEVELS + _RISK_CATS)),
            "confidence": _confidence(docs),
            "evidence": _evidence_list(docs, line),
        })
    return items


# The three lenses the brief asks trends to cover. The LLM's free-text label is
# snapped onto one of these so the dashboard can group cleanly.
def _trend_category(raw: str) -> str:
    """Snap an LLM category label onto one of the three brief-defined buckets."""
    t = raw.lower()
    if any(w in t for w in ("custom", "behav", "consumer", "demand", "buyer", "adopt")):
        return "Customer behaviour"
    if any(w in t for w in ("tech", "ai", "battery", "software", "autonom", "product", "innovat")):
        return "Technology"
    return "Industry"   # market / regulatory / competitive / supply-chain developments


def detect_trends(retriever: HybridRetriever, df) -> list[dict]:
    docs = retriever.retrieve(config.ENGINE_QUERIES["trends"])
    prompt = (
        f"Using ONLY the evidence below, list up to 5 emerging TRENDS "
        f"{config.COMPANY} management should monitor. Cover all three lenses where the "
        f"evidence supports them: Technology trends, Customer behaviour shifts and "
        f"Industry developments.\nOne per line, format:\n"
        f"<Technology|Customer behaviour|Industry> :: <short trend title> :: "
        f"<one full sentence on the trend and why it matters>\n\n"
        f"EVIDENCE:\n{_evidence_block(docs)}\n\nTRENDS:"
    )
    items = []
    for line in _parse_lines(ask_llm(prompt)):
        fields = _split_fields(line)
        if not fields:
            continue  # drop malformed lines that carry no actual trend
        # category leads, then the title, then the sentence
        category = _trend_category(fields[0])
        title = fields[1] if len(fields) >= 2 else fields[0]
        desc = " ".join(fields[2:]) if len(fields) >= 3 else ""
        if not title:
            continue
        items.append({
            "title": strip_src_refs(title),
            "category": category,
            "description": strip_src_refs(desc),
            "confidence": _confidence(docs),
            "evidence": _evidence_list(docs, line),
        })
    return items


def competitor_activity(retriever: HybridRetriever) -> list[dict]:
    docs = retriever.retrieve(config.ENGINE_QUERIES["competitors"])
    return _evidence_list(docs)


def run() -> dict:
    df = load_corpus()
    retriever = HybridRetriever()
    intel = {
        "opportunities": detect_opportunities(retriever),
        "risks": detect_risks(retriever),
        "trends": detect_trends(retriever, df),
        "competitor_activity": competitor_activity(retriever),
        "keywords": top_keywords((df["title"] + " " + df["text"]).tolist()),
    }
    save_json(intel, config.RESULTS_DIR / "intelligence.json")
    print(f"[intelligence_engine] {len(intel['opportunities'])} opportunities, "
          f"{len(intel['risks'])} risks, {len(intel['trends'])} trends")
    return intel


if __name__ == "__main__":
    run()
