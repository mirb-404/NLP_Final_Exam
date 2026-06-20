"""
AI CEO — Strategic Intelligence Agent  (main entry point).

main.py IS the agent graph: a LangGraph ReAct loop where the DataLab Mistral
decides which tool to call, fetches real evidence, and answers. Every capability
is a tool (src/tools.py); collection, retrieval, intelligence, etc. live in their
own modules. The graph is the brain; the modules are the hands.

    python main.py                     # agent answers the default CEO question
    python main.py ask "<question>"    # agent answers your question
    python main.py chat                # interactive: type questions in a loop
    python main.py ingest              # refresh data: collect -> corpus -> Chroma index
    python main.py report              # deterministic pipeline -> results/*.json deliverables

Tool calls are printed live so you can watch the agent fetch evidence.
"""

import re
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from src import config
from src.tools import TOOLS
from src.utils import ask_llm

load_dotenv()  # MODEL_SERVER_URL / token from .env

# ----------------------------------------------------------------------------
# THE AGENT GRAPH — LangGraph ReAct loop (Module 11)
#
#     reason --(Action?)--> act --> reason --> ... --> Final Answer (END)
#
# The model_server is a plain text endpoint (no native function calling), so the
# agent uses the ReAct pattern: the LLM writes `Action: <tool>` / `Action Input:`,
# we run that tool and feed back an `Observation:`, and loop until `Final Answer:`.
# ----------------------------------------------------------------------------
MAX_STEPS = 6  # tool calls before the agent must answer

_TOOLS_BY_NAME = {t.name: t for t in TOOLS}
_TOOL_DESCS = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in TOOLS)
_ACTION_RE = re.compile(r"Action:\s*([^\n]+)\n+Action Input:\s*([^\n]*)", re.IGNORECASE)

SYSTEM = (
    f"You are the AI strategic advisor to the CEO of {config.COMPANY}. "
    "Give concrete, prioritised, evidence-based advice citing the [src-#] markers from tools.\n\n"
    f"Available tools:\n{_TOOL_DESCS}\n\n"
    "Reason step by step in EXACTLY this format:\n"
    "Thought: <reasoning>\nAction: <one tool name>\nAction Input: <input, or NONE>\n"
    "(You then receive an `Observation:`.) When you have enough evidence, output:\n"
    "Final Answer: <the executive answer>"
)


class AgentState(TypedDict, total=False):
    question: str
    scratchpad: str   # running Thought/Action/Observation transcript
    pending: str      # the model's latest output (routed on)
    steps: int


def reason_node(state: AgentState) -> AgentState:
    """Ask the LLM for the next Thought/Action (or the Final Answer)."""
    prompt = f"{SYSTEM}\n\nQuestion: {state['question']}\n{state.get('scratchpad', '')}"
    text = ask_llm(prompt).split("Observation:")[0].strip()  # ignore any hallucinated Observation
    return {
        "pending": text,
        "scratchpad": state.get("scratchpad", "") + "\n" + text,
        "steps": state.get("steps", 0) + 1,
    }


def act_node(state: AgentState) -> AgentState:
    """Run the tool named in the latest Action and append its Observation (printed live)."""
    m = _ACTION_RE.search(state["pending"])
    name = m.group(1).strip().strip("`").split()[0]
    arg = m.group(2).strip()

    tool = _TOOLS_BY_NAME.get(name)
    print(f"  🔧 calling tool: {name}({arg!r})")
    if tool is None:
        obs = f"Unknown tool '{name}'. Choose one of: {list(_TOOLS_BY_NAME)}"
    else:
        try:
            obs = tool.invoke({"query": arg} if "query" in tool.args else {})
        except Exception as exc:
            obs = f"Tool error: {exc}"
    print(f"     ↳ {obs[:140].replace(chr(10), ' ')} ...")
    return {"scratchpad": state["scratchpad"] + f"\nObservation: {obs}\n"}


def _route(state: AgentState) -> str:
    """Stop on a Final Answer or step budget; otherwise run the requested tool."""
    if "final answer:" in state["pending"].lower() or state.get("steps", 0) >= MAX_STEPS:
        return "end"
    return "act" if _ACTION_RE.search(state["pending"]) else "end"


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
    """Run the agent on one strategic question and return its final answer."""
    state = build_agent().invoke(
        {"question": question, "scratchpad": "", "steps": 0},
        {"recursion_limit": 2 * MAX_STEPS + 2},
    )
    m = re.search(r"Final Answer:\s*(.+)", state["scratchpad"], re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else state["scratchpad"].strip()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ask"

    if cmd == "ingest":
        from src.orchestrator import run_ingest
        run_ingest()
    elif cmd == "report":
        from src.orchestrator import run_analyze
        state = run_analyze()
        print("\n=== CEO BRIEFING ===\n" + state.get("briefing", ""))
    elif cmd == "chat":
        print(f"AI CEO agent for {config.COMPANY}. Ask a question (or 'exit').")
        while True:
            q = input("\n> ").strip()
            if q.lower() in ("exit", "quit", "q", ""):
                break
            print("\n" + ask_ceo(q))
    else:  # "ask" or no command -> run the agent
        question = " ".join(sys.argv[2:]) if cmd == "ask" else ""
        question = question or f"If you were the CEO of {config.COMPANY} today, what would you do next and why?"
        print(f"\n=== AI CEO AGENT ===\nQ: {question}\n")
        print("\nANSWER:\n" + ask_ceo(question))


if __name__ == "__main__":
    main()
