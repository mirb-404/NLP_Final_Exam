"""
AI Strategy Consultant — main entry point.

main.py IS the agent graph: a LangGraph ReAct loop where the LLM picks a tool
(src/tools.py), fetches real evidence, and answers. Graph = brain, modules = hands.

    python main.py                     # agent answers the default CEO question
    python main.py ask "<question>"    # agent answers your question
    python main.py chat                # interactive loop
    python main.py ingest              # refresh data: collect -> corpus -> Chroma index
    python main.py report              # deterministic pipeline -> results/*.json
    python main.py collect|corpus|index   # run a single ingest stage
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
# tolerate the model putting "Action:" and "Action Input:" on one line or two
_ACTION_RE = re.compile(r"Action:\s*([^\n]+?)\s+Action Input:\s*([^\n]*)", re.IGNORECASE)

SYSTEM = (
    f"You are the AI strategic advisor to the CEO of {config.COMPANY}. "
    "Give concrete, prioritised, evidence-based advice grounded in the evidence the tools return.\n\n"
    f"Available tools (use ONLY these — never invent another tool):\n{_TOOL_DESCS}\n\n"
    "Prefer to start with search_knowledge_base for grounding evidence, then use the specialised "
    "tools (competitor/sentiment/keywords) to go deeper. Don't repeat a tool with near-identical input.\n\n"
    "Reason step by step in EXACTLY this format:\n"
    "Thought: <reasoning>\nAction: <one tool name>\nAction Input: <input, or NONE>\n"
    "Then STOP and wait for the real Observation — never write the Observation yourself "
    "or make up tool results. When you have enough evidence, output:\n"
    "Final Answer: <the executive answer>\n"
    "The Final Answer MUST be grounded in the Observations you actually received — refer to "
    "their real figures/keywords, invent no supporting data, and avoid generic advice that would "
    "fit any company. If the evidence is thin, say so plainly. Write it as clean prose for the "
    "CEO — do NOT print [src-#] markers."
)


class AgentState(TypedDict, total=False):
    question: str
    scratchpad: str   # running Thought/Action/Observation transcript
    pending: str      # the model's latest output (routed on)
    steps: int


def reason_node(state: AgentState) -> AgentState:
    """Ask the LLM for the next Thought/Action (or the Final Answer)."""
    prompt = f"{SYSTEM}\n\nQuestion: {state['question']}\n{state.get('scratchpad', '')}"
    text = ask_llm(prompt).split("Observation:")[0]   # never let the model invent an Observation
    m = _ACTION_RE.search(text)
    if m:
        text = text[: m.end()]   # keep only the first Thought + Action + Action Input; we run the tool
    text = text.strip()
    print(text)                  # show the agent's reasoning + chosen action live
    return {
        "pending": text,
        "scratchpad": state.get("scratchpad", "") + "\n" + text + "\n",
        "steps": state.get("steps", 0) + 1,
    }


def act_node(state: AgentState) -> AgentState:
    """Run the tool named in the latest Action and append its Observation (printed live)."""
    m = _ACTION_RE.search(state["pending"])
    name = m.group(1).strip().strip("`").split()[0]
    arg = m.group(2).strip()

    tool = _TOOLS_BY_NAME.get(name)
    print(f"  calling tool: {name}({arg!r})")
    if tool is None:
        obs = f"Unknown tool '{name}'. Choose one of: {list(_TOOLS_BY_NAME)}"
    else:
        try:
            obs = tool.invoke({"query": arg} if "query" in tool.args else {})
        except Exception as exc:
            obs = f"Tool error: {exc}"
    print(f"     -> {obs[:300].replace(chr(10), ' ')}")
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


_STEP_RE = re.compile(
    r"Thought:\s*(.*?)\n+Action:\s*([^\n]+?)\s*\n+Action Input:\s*([^\n]*)",
    re.IGNORECASE | re.DOTALL,
)


def agent_steps(scratchpad: str) -> list[dict]:
    """Split the agent's transcript into ordered steps — one per tool call — each as
    {thought, tool, input, observation}. This only READS what the ReAct loop already
    produced, so the CLI and dashboard can show *why* each tool was chosen."""
    observations = re.findall(
        r"Observation:\s*(.*?)(?=\n\s*Thought:|\Z)", scratchpad, re.IGNORECASE | re.DOTALL
    )
    steps = []
    for i, (thought, tool, arg) in enumerate(_STEP_RE.findall(scratchpad)):
        name = tool.strip().strip("`").split()
        steps.append({
            "thought": thought.strip(),
            "tool": name[0] if name else tool.strip(),
            "input": arg.strip(),
            "observation": observations[i].strip() if i < len(observations) else "",
        })
    return steps


def _final_answer(question: str, scratchpad: str) -> str:
    """The agent's Final Answer, cut off before any trailing Thought/Action the model
    tacked on. Falls back to one clean synthesis if the loop never concluded."""
    m = re.search(
        r"Final Answer:\s*(.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
        scratchpad, re.IGNORECASE | re.DOTALL,
    )
    if m and m.group(1).strip():
        return m.group(1).strip()
    # Ran out of steps without concluding -> force one clean synthesis from the evidence gathered.
    forced = ask_llm(f"{SYSTEM}\n\nQuestion: {question}\n{scratchpad}\n\n"
                     "Using ONLY the evidence in the Observations above (use the real figures/"
                     "keywords, invent nothing, no [src-#] markers), reply with ONLY:\nFinal Answer:")
    return forced.split("Final Answer:")[-1].strip()


_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|how are you|how'?s it going|good (morning|afternoon|evening)|"
    r"thanks?|thank you|who are you|what can you do)\b",
    re.IGNORECASE,
)


def _is_chitchat(question: str) -> bool:
    """Cheap gate: a short greeting / pleasantry rather than a strategic question.
    These get a one-line redirect instead of triggering the whole agent loop."""
    q = question.strip()
    return bool(_GREETING_RE.match(q)) and len(q.split()) <= 5


def run_agent(question: str) -> dict:
    """Run the ReAct loop once and return the final answer *and* the step-by-step
    reasoning trace (Thought -> Action -> Observation), so callers can show how the
    agent decided, not just its conclusion."""
    if _is_chitchat(question):   # don't fabricate a question + run tools for a greeting
        return {"answer": (f"I'm the AI strategy consultant for {config.COMPANY}. "
                           "Ask me a strategic question — e.g. 'How do we beat BYD in China?'"),
                "steps": []}
    state = build_agent().invoke(
        {"question": question, "scratchpad": "", "steps": 0},
        {"recursion_limit": 2 * MAX_STEPS + 2},
    )
    pad = state["scratchpad"]
    return {"answer": _final_answer(question, pad), "steps": agent_steps(pad)}


def ask_ceo(question: str) -> str:
    """The agent's final answer to one strategic question."""
    return run_agent(question)["answer"]


def strategic_options(question: str, answer: str) -> dict:
    """Turn the agent's answer into a decision menu: strategic moves (with upside) and
    sharper follow-up questions, so the CEO sees how much further they can push."""
    prompt = (
        f"You are advising the CEO of {config.COMPANY}.\n"
        f"Question: {question}\nAnswer so far: {answer}\n\n"
        "Give the CEO a decision menu. Use EXACTLY this format, nothing else:\n"
        "OPTION: <a bold strategic move> | <the upside: how much better this could make the company>\n"
        "OPTION: <a different move> | <upside>\n"
        "OPTION: <a different move> | <upside>\n"
        "FOLLOWUP: <a sharper question the CEO should ask next>\n"
        "FOLLOWUP: <another follow-up question>\n"
        "FOLLOWUP: <another follow-up question>\n"
    )
    text = ask_llm(prompt)
    # capture "move | upside" but stop the upside at the next | or newline so a
    # same-line FOLLOWUP never bleeds into it; pull FOLLOWUPs separately.
    options = [f"{p.strip()} | {u.strip()}"
               for p, u in re.findall(r"OPTION:\s*([^|\n]+)\|\s*([^|\n]+)", text)][:3]
    followups = [f.strip() for f in re.findall(r"FOLLOWUP:\s*([^\n|]+)", text)][:3]
    return {"options": options, "followups": followups}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ask"

    if cmd == "collect":
        from src.collector import collect_all
        collect_all()
    elif cmd == "corpus":
        from src.preprocess import build_corpus
        build_corpus()
    elif cmd == "index":
        from src.knowledge_base import build_index
        from src.preprocess import load_corpus
        build_index(load_corpus())
    elif cmd == "ingest":
        from src.orchestrator import run_ingest
        run_ingest()
    elif cmd == "report":
        from src.orchestrator import run_analyze
        state = run_analyze()
        print("\n=== CEO BRIEFING ===\n" + state.get("briefing", ""))
    elif cmd == "chat":
        print(f"AI Strategy Consultant agent for {config.COMPANY}. Ask a question (or 'exit').")
        while True:
            q = input("\n> ").strip()
            if q.lower() in ("exit", "quit", "q", ""):
                break
            print("\n" + ask_ceo(q))
    else:  # "ask" or no command -> run the agent
        question = " ".join(sys.argv[2:]) if cmd == "ask" else ""
        question = question or f"If you were the CEO of {config.COMPANY} today, what would you do next and why?"
        print(f"\n=== AI STRATEGY CONSULTANT AGENT ===\nQ: {question}\n")
        print("\nANSWER:\n" + ask_ceo(question))


if __name__ == "__main__":
    main()
