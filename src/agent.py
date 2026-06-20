"""
AI CEO Agent — LangGraph ReAct loop (Module 11).

The DataLab model_server is a plain text chat endpoint (no native function
calling), so the agent uses the ReAct pattern: the model writes

    Thought: ...
    Action: <tool name>
    Action Input: <input>

the graph runs that tool, feeds back an `Observation:`, and loops until the model
writes `Final Answer: ...`. Tools come from src/tools.py; the LLM is the DataLab
Mistral via utils.ask_llm. Works with any text model — no vLLM, no server changes.

    from src.agent import ask_ceo
    print(ask_ceo("If you were the CEO of Apple today, what would you do next and why?"))
"""

import re
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src import config
from src.tools import TOOLS
from src.utils import ask_llm

MAX_STEPS = 6  # tool calls before the agent must answer

_TOOLS_BY_NAME = {t.name: t for t in TOOLS}
_TOOL_DESCS = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in TOOLS)
_ACTION_RE = re.compile(r"Action:\s*([^\n]+)\n+Action Input:\s*([^\n]*)", re.IGNORECASE)

SYSTEM = (
    f"You are the AI strategic advisor to the CEO of {config.COMPANY}. "
    "Give concrete, prioritised, evidence-based advice citing the [src-#] markers from tools.\n\n"
    f"Available tools:\n{_TOOL_DESCS}\n\n"
    "Reason step by step in EXACTLY this format:\n"
    "Thought: <reasoning>\n"
    "Action: <one tool name>\n"
    "Action Input: <input, or NONE>\n"
    "(You then receive an `Observation:`.) When you have enough evidence, output:\n"
    "Final Answer: <the executive answer>"
)


class AgentState(TypedDict, total=False):
    question: str
    scratchpad: str   # running Thought/Action/Observation transcript
    pending: str      # the model's latest output (routed on)
    steps: int


# ----------------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------------
def reason_node(state: AgentState) -> AgentState:
    """Ask the LLM for the next Thought/Action (or Final Answer)."""
    prompt = f"{SYSTEM}\n\nQuestion: {state['question']}\n{state.get('scratchpad', '')}"
    text = ask_llm(prompt).split("Observation:")[0].strip()  # stop if it hallucinates an Observation
    return {
        "pending": text,
        "scratchpad": state.get("scratchpad", "") + "\n" + text,
        "steps": state.get("steps", 0) + 1,
    }


def act_node(state: AgentState) -> AgentState:
    """Run the tool named in the latest Action and append its Observation."""
    m = _ACTION_RE.search(state["pending"])
    name = m.group(1).strip().strip("`").split()[0]
    arg = m.group(2).strip()

    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        obs = f"Unknown tool '{name}'. Choose one of: {list(_TOOLS_BY_NAME)}"
    else:
        try:
            obs = tool.invoke({"query": arg} if "query" in tool.args else {})
        except Exception as exc:
            obs = f"Tool error: {exc}"
    return {"scratchpad": state["scratchpad"] + f"\nObservation: {obs}\n"}


def _route(state: AgentState) -> str:
    """Stop on a Final Answer or step budget; otherwise run the requested tool."""
    if "final answer:" in state["pending"].lower() or state.get("steps", 0) >= MAX_STEPS:
        return "end"
    return "act" if _ACTION_RE.search(state["pending"]) else "end"


@lru_cache(maxsize=1)
def build_agent():
    """Compile the ReAct graph once:  reason --(action?)--> act --> reason --> ... --> END."""
    g = StateGraph(AgentState)
    g.add_node("reason", reason_node)
    g.add_node("act", act_node)
    g.set_entry_point("reason")
    g.add_conditional_edges("reason", _route, {"act": "act", "end": END})
    g.add_edge("act", "reason")
    return g.compile()


def ask_ceo(question: str) -> str:
    """Run the ReAct agent on one strategic question and return its final answer."""
    state = build_agent().invoke(
        {"question": question, "scratchpad": "", "steps": 0},
        {"recursion_limit": 2 * MAX_STEPS + 2},
    )
    m = re.search(r"Final Answer:\s*(.+)", state["scratchpad"], re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else state["scratchpad"].strip()


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or f"If you were the CEO of {config.COMPANY} today, what would you do next and why?"
    print(ask_ceo(q))
