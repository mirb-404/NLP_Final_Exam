"""
LangChain tools for the AI Strategy Consultant agent (Module 11 — tool-calling agent).

Every capability the agent needs is exposed here as an @tool, so a LangGraph
agent can *decide* which to call and fetch real evidence from the knowledge base
instead of hallucinating. These are fetchers (retrieval + classical NLP); the
agent does the strategic reasoning over what they return.

Used by the agent graph in main.py.
"""

from functools import lru_cache

from langchain_core.tools import tool

from src import config
from src.classical_agent import corpus_sentiment, top_keywords
from src.preprocess import load_corpus
from src.retriever_hybrid import HybridRetriever


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    """Build the hybrid retriever once and reuse it across tool calls."""
    return HybridRetriever()


def _format(docs: list[dict]) -> str:
    return "\n".join(
        f"[src-{d['rank']}] ({d['source']}, {d['source_type']}) {d['title']}: {d['text'][:300]}"
        for d in docs
    ) or "No relevant documents found."


@tool
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base (news, finance, community, research) and return
    the most relevant evidence snippets for the query. Use this to ground any claim with
    cited [src-#] evidence about opportunities, risks, products, or markets."""
    return _format(_retriever().retrieve(query))


@tool
def get_competitor_activity() -> str:
    """Return recent evidence about what the company's competitors are doing."""
    return _format(_retriever().retrieve(config.ENGINE_QUERIES["competitors"]))


@tool
def get_sentiment() -> str:
    """Return news vs public sentiment for the company (VADER) and the
    positive/negative/neutral distribution across the corpus."""
    s = corpus_sentiment(load_corpus())
    return (f"news_sentiment={s['news_sentiment']}, public_sentiment={s['public_sentiment']}, "
            f"overall={s['overall_sentiment']}, distribution={dict(s['distribution'])}")


@tool
def get_trending_keywords() -> str:
    """Return the most important terms across the collected corpus (TF-IDF)."""
    df = load_corpus()
    return ", ".join(top_keywords((df["title"] + " " + df["text"]).tolist()))


@tool
def get_company_overview() -> str:
    """Return the company name, industry, number of collected documents and source types."""
    df = load_corpus()
    return (f"company={config.COMPANY}, industry={config.INDUSTRY}, "
            f"documents={len(df)}, source_types={sorted(df['source_type'].unique())}")


# The toolset the agent is allowed to call.
TOOLS = [
    search_knowledge_base,
    get_competitor_activity,
    get_sentiment,
    get_trending_keywords,
    get_company_overview,
]
