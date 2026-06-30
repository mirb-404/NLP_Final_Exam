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
from src.classical_agent import texts_sentiment, top_keywords
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
def get_competitor_activity(query: str) -> str:
    """Return evidence about what competitors are doing, focused on the query
    (e.g. a named rival or market such as 'BYD in China')."""
    return _format(_retriever().retrieve(f"competitors and rivals: {query}"))


@tool
def get_sentiment(query: str) -> str:
    """Return the sentiment (RoBERTa, 3-class TweetEval) of the evidence about the query —
    the mean signed score and the positive/negative/neutral breakdown."""
    docs = _retriever().retrieve(query)
    s = texts_sentiment([d["title"] + " " + d["text"] for d in docs])
    return f"mean_sentiment={s['mean_sentiment']}, distribution={s['distribution']}"


@tool
def get_trending_keywords(query: str) -> str:
    """Return the most important terms (TF-IDF) in the evidence about the query."""
    docs = _retriever().retrieve(query)
    return ", ".join(top_keywords([d["title"] + " " + d["text"] for d in docs]))


@tool
def get_company_overview() -> str:
    """Return the company name, industry, number of collected documents and source types.
    A corpus-wide status tool — takes no query."""
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
