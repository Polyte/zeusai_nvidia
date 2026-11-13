"""Session state (mode, enhance toggle, session permissions) and per-mode instructions."""
from dataclasses import dataclass, field

from . import llm
from .config import get_active_profile
from .tools import TOOLS, READ_ONLY_TOOLS

MODE_PROMPTS = {
    "default": "",
    "auto": "\nMode: AUTO. Tools run without confirmation. Work autonomously to completion, verify with tests or commands, then summarize what changed.",
    "fast": (
        "\nMode: FAST. Work as fast as possible: thinking OFF, answer in the fewest words that are still useful, "
        "batch every independent tool call into one turn, prefer the simplest working change over a general one, "
        "skip todos unless there are 5+ clearly ordered steps, and verify with at most one quick command. "
        "No essays, no recap, no extra exploration. If something is unclear, pick the most likely reading and state the assumption in one line."
    ),
    "thinking": (
        "\nMode: THINKING. Think deeply before acting: reason step by step, use todo_write for multi-step work, "
        "read the relevant files in full before editing, check edge cases, existing conventions, and callers, "
        "then verify with tests or commands before finishing. Briefly explain important trade-offs and state assumptions. "
        "Prefer a correct, thorough solution over a quick one."
    ),
    "executor": (
        "\nMode: EXECUTOR (strict instruction follower). Treat the user's instructions as a checklist and follow it literally — you are an executor, not an inventor.\n"
        "Protocol:\n"
        "1. Rewrite the user's instructions as a numbered checklist with todo_write (one step per instruction, in order). Mark each step in_progress before doing it and completed only after verifying it.\n"
        "2. Do exactly what is written, in order, and nothing more: no refactors, no extra files, no guessed names or APIs, no 'improvements'. If an instruction is ambiguous, missing a detail, or would force a guess, call ask_user with 2-4 concrete options (recommended first) and wait before proceeding.\n"
        "3. Verify each step immediately before moving on: re-read edits, run the smallest relevant test or command, and fix only the current step's failure. Never skip verification or batch unverified steps.\n"
        "4. Keep changes minimal and scoped to what the instructions name (plus callers that would otherwise break). Preserve existing style, names, and structure.\n"
        "5. Finish with a checklist report: each step done/skipped/blocked, files changed with paths, verification results (exact commands and outcomes), and anything left undone. If blocked, stop at the exact step, say what blocked you, and do not improvise a different plan."
    ),
    "plan": ("\nMode: PLAN (read-only). Do NOT modify anything. Explore with read/list/search tools, ask the user "
             "(ask_user) about real decisions, then reply with a concise numbered implementation plan: files to change, "
             "what to change, and how you'll verify it. End with the plan; the user will approve it before execution."),
    "research": ("\nMode: RESEARCH. Answer using the web: call web_search, then web_fetch the 2-4 most relevant sources "
                 "(official docs first), cross-check claims, trust what the sources say over your own memory (your training data may be outdated; check dates against today's date), and answer concisely with inline citations like [1]. "
                 "End with a 'Sources' list of numbered URLs. Don't modify files."),
}

PLAN_OPTIONS = ["Yes, execute it (auto-accept edits)", "Yes, execute it (ask before each edit)", "No, keep planning"]

# Model + reasoning presets for the execution modes. Explicit --model / /model always wins;
# these only apply when the caller didn't pick a model (None/"auto").
MODE_PROFILES = {"fast": "laguna", "thinking": "ultra", "executor": "ultra"}
MODE_THINKING = {"fast": False, "thinking": True, "executor": False}


def profile_for_mode(mode, has_image=False):
    """Suggested model alias for a mode (vision always upgrades to the vision model)."""
    if has_image:
        return "super"
    return MODE_PROFILES.get(mode)


def thinking_for_mode(mode):
    """Thinking default for a mode, or None when the mode doesn't set one."""
    return MODE_THINKING.get(mode)


def effective_thinking(session):
    """The thinking value the model will actually get (explicit /think override wins over the mode)."""
    if session is None:
        return None
    if session.thinking is not None:
        return session.thinking
    return thinking_for_mode(session.mode)


def set_mode(session, mode):
    """Switch a session to a mode, applying its thinking default. Returns the suggested model alias or None."""
    from . import permissions
    if mode not in permissions.MODES:
        return None
    session.mode = mode
    if mode in MODE_THINKING:
        session.thinking = MODE_THINKING[mode]
    return profile_for_mode(mode)


def apply_mode_defaults(session, profile_name=None, has_image=False):
    """Fill in thinking + model from the session mode when the caller didn't choose them.

    Explicit choices win: a non-None session.thinking (from /think or --think) and a concrete
    profile_name (from --model / /model) are left alone. Returns the effective profile name.
    """
    if session.thinking is None and session.mode in MODE_THINKING:
        session.thinking = MODE_THINKING[session.mode]
    if profile_name in (None, "auto"):
        suggested = profile_for_mode(session.mode, has_image=has_image)
        if suggested:
            return suggested
    return profile_name


@dataclass
class Session:
    mode: str = "default"
    enhance: bool = False
    allowed: list = field(default_factory=list)  # "don't ask again" rules for this session
    todos: list = field(default_factory=list)    # todo_write list, shown to the user
    id: str = ""                                 # saved-conversation id (sessions.py); "" = not saved
    checkpoints: list = field(default_factory=list)  # per-prompt file snapshots for /rewind
    thinking: object = None                      # None = profile default, True/False = /think override
    auto_continue: bool = True                   # nudge the model to keep going while work remains
    allowed_tools: list = field(default_factory=list)     # headless --allowed-tools (empty = all)
    disallowed_tools: list = field(default_factory=list)  # headless --disallowed-tools
    system_prompt: str = ""                      # replaces the base system prompt (headless --system-prompt)
    append_prompt: str = ""                      # appended to the system prompt (--append-system-prompt)
    style: str = ""                              # output style ("" = settings default, see styles.py)
    dirs: list = field(default_factory=list)     # extra workspace dirs (/add-dir, --add-dir)


def tools_for(mode):
    if mode in ("plan", "research"):
        return [t for t in TOOLS if t["function"]["name"] in READ_ONLY_TOOLS | {"ask_user", "memory", "todo_write"}]
    from . import mcp
    return TOOLS + mcp.tool_defs()


ENHANCE_SYSTEM = """You rewrite a user's request to a coding agent into a clearer, more effective prompt.
Keep the user's intent and every concrete detail. Make it specific: the goal, relevant constraints, expected output, and how to verify success.
Don't invent requirements, file names or technologies the user didn't imply; if something is ambiguous, say the agent should ask.
If the request is vague (e.g. 'fix tests'), write a prompt that has the agent investigate first (run the tests, read the failures) and then fix the root cause.
Keep it short (under 120 words). Output ONLY the rewritten prompt, no preamble."""


def enhance_prompt(prompt, profile="laguna"):
    """One fast call to the lightest model; falls back to the original on any error."""
    try:
        _, prof = get_active_profile(profile)
        r = llm.chat(prof, [{"role": "system", "content": ENHANCE_SYSTEM}, {"role": "user", "content": f"Rewrite this request. Do not answer it or ask questions back; output only the improved prompt.\n<request>\n{prompt}\n</request>"}],
                     max_tokens=400, thinking=False)
        return (r["content"] or "").strip() or prompt
    except Exception:
        return prompt
