"""Subagents: the `task` tool runs a fresh agent loop on a focused job and returns only its final report.

Built-in types: explore (read-only search) and general (all tools). Custom agents live in
.zeusai/agents/*.md (project) or ~/.zeusai/agents/*.md (user):

    ---
    name: test-runner
    description: Runs the test suite and reports failures with likely causes
    tools: read_file, search_code, glob_files, run_shell
    model: laguna
    ---
    You run tests and explain failures. ...
"""
import pathlib
import threading

from .memory import _parse
from .tools import TOOLS, READ_ONLY_TOOLS

READ_TOOLS = sorted(READ_ONLY_TOOLS - {"shell_output"})
NO_NESTING = {"task", "ask_user", "todo_write"}  # subagents can't spawn agents, ask the user, or touch the parent's todos

BUILTIN = {
    "explore": {
        "description": "Fast read-only search of the codebase: finding files, definitions, usages, and answering 'where/how is X' questions.",
        "tools": READ_TOOLS,
        "prompt": ("You are a read-only codebase explorer working for another agent. Search efficiently: glob_files for names, "
                   "search_code for text, read_file only the relevant parts; batch independent lookups in one turn. "
                   "Never modify anything. Finish with a concise report: the answer, the exact file paths and line numbers "
                   "that support it, and anything you couldn't determine."),
    },
    "general": {
        "description": "General-purpose agent for multi-step research or implementation work that benefits from a separate context.",
        "tools": None,  # all except NO_NESTING
        "prompt": ("You are a capable coding agent handling one delegated task for another agent. Do the task completely "
                   "using tools, verify the result, then finish with a concise report of what you found or changed "
                   "(files, commands, outcomes). Your final message is all the other agent will see."),
    },
}


def _dirs():
    return [pathlib.Path.home() / ".zeusai" / "agents", pathlib.Path.cwd() / ".zeusai" / "agents"]


def custom_agents():
    out = {}
    for d in _dirs():  # project overrides user
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            meta, body = _parse(p)
            name = meta.get("name") or p.stem
            tools = [t.strip() for t in meta.get("tools", "").split(",") if t.strip()] or None
            out[name] = {"description": meta.get("description", ""), "tools": tools,
                         "model": meta.get("model") or None, "prompt": body, "path": str(p)}
    return out


def all_agents():
    return {**BUILTIN, **custom_agents()}


def tool_schema():
    """The task tool definition, listing the available agent types."""
    types = "\n".join(f"- {n}: {a['description']}" for n, a in all_agents().items())
    return {
        "type": "function",
        "function": {
            "name": "task",
            "description": ("Delegate a self-contained job to a subagent with its own fresh context; it returns only a final report. "
                            "Use for broad searches (explore) or separable multi-step work, to keep your own context small. "
                            "Several task calls in one turn run in parallel. Give a complete brief: the subagent can't see this conversation.\n"
                            f"Agent types:\n{types}"),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "Agent type, default 'general'"},
                    "description": {"type": "string", "description": "3-6 word label shown to the user"},
                    "prompt": {"type": "string", "description": "The full task brief"},
                },
                "required": ["description", "prompt"],
            },
        },
    }


def _tools_for(spec):
    names = spec.get("tools")
    allowed = set(names) if names else {t["function"]["name"] for t in TOOLS}
    return [t for t in TOOLS if t["function"]["name"] in allowed - NO_NESTING]


_prompt_lock = threading.Lock()  # parallel subagents take turns at permission prompts


def run_task(args, session, profile_name=None, on_permission=None, on_status=None):
    """Run one subagent to completion and return its report (string)."""
    from .agent import SYSTEM_PROMPT, run_loop
    from .modes import Session

    kind = (args.get("agent") or "general").strip()
    spec = all_agents().get(kind)
    if not spec:
        return f"Error: unknown agent type {kind!r}. Available: {', '.join(all_agents())}"
    label = args.get("description") or kind
    sub = Session(mode=session.mode, allowed=session.allowed,  # same permissions and tool filters; separate todos
                  allowed_tools=session.allowed_tools, disallowed_tools=session.disallowed_tools,
                  thinking=session.thinking, dirs=session.dirs)
    from .agent import _filter_tools
    tools = _filter_tools(_tools_for(spec), sub)
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n" + spec["prompt"]},
                {"role": "user", "content": args.get("prompt", "")}]
    steps = {"n": 0}

    def status(line):
        if on_status and line.lstrip().startswith("●"):
            steps["n"] += 1
            on_status(f"  ⎿  [{label}] {line.strip()[2:]}")

    def perm(name, a, rule):
        if on_permission is None:
            return False, None
        with _prompt_lock:
            return on_permission(name, a, rule)

    try:
        text = run_loop(messages, max_steps=30, profile_name=spec.get("model") or profile_name, session=sub,
                        on_status=status, on_permission=perm if on_permission else None, tools=tools)
    except Exception as e:
        return f"Error: subagent {label} failed: {e}"
    report = (text or "(no report)") + f"\n\n[subagent {kind}: {steps['n']} tool calls]"
    from . import hooks
    blocked, why = hooks.run("SubagentStop", name=kind, result=report)
    if blocked:
        report += f"\n\n[SubagentStop hook feedback]\n{why}"
    return report


CREATE_SYSTEM = """You design a focused subagent for a coding assistant. Given what the user wants, output ONLY a Markdown
file with YAML frontmatter, nothing else:
---
name: kebab-case-name
description: one line saying when to use this agent (the main agent reads it to decide)
tools: comma-separated subset of: {tools}
---
<system prompt: who the agent is, how it should work step by step, what to verify, and the exact shape of its final report>
Give it the fewest tools it needs (read-only tools unless it must change things). Keep the prompt under 250 words."""


def create_agent(description, profile_name=None, scope="project"):
    """Have the model write .zeusai/agents/<name>.md for `description`. Returns the path."""
    import re
    from . import llm
    from .config import get_active_profile
    names = ", ".join(t["function"]["name"] for t in TOOLS if t["function"]["name"] not in NO_NESTING)
    _, prof = get_active_profile(profile_name)
    r = llm.chat(prof, [{"role": "system", "content": CREATE_SYSTEM.format(tools=names)},
                        {"role": "user", "content": description}], max_tokens=1200, thinking=False)
    text = (r.get("content") or "").strip()
    text = re.sub(r"^```\w*\n|\n```$", "", text).strip()
    m = re.match(r"---\n(.*?)\n---", text, re.S)
    if not m:
        raise ValueError("the model didn't return an agent file")
    name = re.search(r"^name:\s*(.+)$", m.group(1), re.M)
    name = re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]", "-", (name.group(1).strip() if name else "custom-agent").lower())).strip("-") or "custom-agent"
    d = _dirs()[1 if scope == "project" else 0]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.md"
    n = 2
    while path.exists():
        path = d / f"{name}-{n}.md"
        n += 1
    path.write_text(text + "\n")
    return path
