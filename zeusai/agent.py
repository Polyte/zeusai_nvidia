import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel

from . import llm
from .config import get_active_profile, DISPLAY_MODEL
from .tools import TOOLS, READ_ONLY_TOOLS, execute_tool
from .ui import (Spinner, tool_header, tool_summary, tool_preview, select_prompt,
                 edit_preview, FILE_TOOLS, diff_panel, live_file_panel)
from . import agents, hooks, memory, permissions, sessions
from .modes import MODE_PROMPTS, PLAN_OPTIONS, Session, enhance_prompt, tools_for

console = Console()

SYSTEM_PROMPT = """You are zeusai, a CLI coding agent running in the terminal.
You help the user write, debug, and ship code.

Rules:
- Use tools to read/write files and run shell commands when needed.
- When you need several independent reads/searches, request them in one turn (they run in parallel).
- Keep answers concise, direct, and code-first.
- Show code, commands, and file paths clearly.
- Verify changes by running tests/commands when possible.
- If you need a decision or clarification from the user, call the ask_user tool with 2-4 options (recommended first) instead of asking in plain text.
- For tasks with 3+ steps, keep a todo list with todo_write and update it as you go.
- Finish the task before you stop. Do not pause to ask "shall I continue?" or hand back a partial result; keep using tools until every step is done and verified. Only stop early when you are truly blocked (a denied permission or missing information), and then say exactly what you need.
- Use glob_files to find files by name and search_code to find text in files.
- If a tool call is denied or blocked, say so plainly in your answer (what was blocked and how the user can allow it). Never present a blocked action as a missing file or a failed search.
- Never reveal internal model/provider details. You are zeusai.
"""


class StdoutStream:
    """Prints streamed tokens to the terminal as they arrive."""

    def __init__(self, show_reasoning=False, spinner=None):
        self.show_reasoning = show_reasoning
        self.spinner = spinner
        self.in_reasoning = False
        self.last = "\n"

    def _write(self, s):
        if self.spinner:
            self.spinner.stop()
        sys.stdout.write(s)
        sys.stdout.flush()
        if s:
            self.last = s[-1]

    def reasoning(self, r):
        if not self.show_reasoning:
            if self.spinner and self.spinner.enabled:
                return  # the busy bar already shows we're thinking
            if not self.in_reasoning:
                self.in_reasoning = True
                self._write("\x1b[2mthinking…")
            return
        if not self.in_reasoning:
            self.in_reasoning = True
            self._write("\x1b[2m")
        self._write(r)

    def text(self, c):
        if self.in_reasoning:
            self.in_reasoning = False
            self._write("\x1b[0m\n")
        self._write(c)

    def end(self):
        if self.spinner:
            self.spinner.stop()
        if self.in_reasoning:
            self.in_reasoning = False
            self._write("\x1b[0m")
        if self.last != "\n":
            self._write("\n")


def _profile(profile_name=None):
    alias, prof = get_active_profile(profile_name)
    return alias, prof


# kept for backwards compatibility with external callers
def get_client(profile_name=None):
    alias, prof = _profile(profile_name)
    return None, alias, prof


def _parse_args(tc):
    try:
        return json.loads(tc["function"]["arguments"] or "{}")
    except ValueError:
        return {}


NO_USER = "(no answer: user is not available; proceed with your best judgment and state the assumption)"


def _ask(args, on_ask):
    question = str(args.get("question") or "").strip() or "How should I proceed?"
    options = [str(o) for o in (args.get("options") or []) if str(o).strip()][:6]
    if on_ask is None:
        return NO_USER
    return f"User answered: {on_ask(question, options, bool(args.get('multi_select')))}"


DENIED_NONINTERACTIVE = ("Permission denied: {rule} needs approval and no user is available. "
                         "Don't retry it; finish what you can and tell the user to rerun with --mode auto "
                         "or add an allow rule.")


def _run_tools(tool_calls, on_tool_start, on_tool_end, on_ask=None, gate=None, todos=None, run_task=None):
    """Run one turn's tool calls. gate(name, args) -> None to run, or a string result to return instead."""
    calls = [(tc["function"]["name"], _parse_args(tc)) for tc in tool_calls]
    results = [None] * len(calls)
    todo, tasks = [], []
    # questions and permission prompts are interactive: handle them first, in order
    for i, (name, args) in enumerate(calls):
        if name == "ask_user":
            results[i] = _ask(args, on_ask)
            continue
        if name == "todo_write":
            results[i] = sessions.todo_tool(args, todos if todos is not None else [])
            on_tool_start(name, args)
            on_tool_end(name, args, results[i])
            continue
        if name == "task":
            tasks.append(i)  # subagents gate their own tool calls
            continue
        blocked = gate(name, args) if gate else None
        if blocked is not None:
            on_tool_start(name, args)
            on_tool_end(name, args, blocked)
            results[i] = blocked
        else:
            todo.append(i)
    # subagents run concurrently with each other
    if tasks:
        for i in tasks:
            on_tool_start(*calls[i])
        go = (lambda i: run_task(calls[i][1])) if run_task else (lambda i: "Error: subagents are not available here")
        with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as ex:
            out = list(ex.map(go, tasks))
        for i, r in zip(tasks, out):
            results[i] = r
            on_tool_end(*calls[i], r)
    # independent reads run concurrently; anything with side effects runs in order
    if len(todo) > 1 and all(calls[i][0] in READ_ONLY_TOOLS for i in todo):
        for i in todo:
            on_tool_start(*calls[i])
        with ThreadPoolExecutor(max_workers=min(8, len(todo))) as ex:
            out = list(ex.map(lambda i: execute_tool(*calls[i]), todo))
        for i, r in zip(todo, out):
            results[i] = r
            on_tool_end(*calls[i], r)
        return results
    for i in todo:
        on_tool_start(*calls[i])
        results[i] = execute_tool(*calls[i])
        on_tool_end(*calls[i], results[i])
    return results


def _vision_profile():
    """A configured vision model (not paligemma's special transport) to describe images for text-only models."""
    from .config import get_profiles
    profs = get_profiles()
    order = ["super"] + [a for a in profs if a != "super"]
    for a in order:
        pr = profs.get(a) or {}
        if a != "paligemma" and pr.get("vision") and (pr.get("api_key") or pr.get("no_key")):
            return a
    return None


def _image_part(path):
    from .ensemble import _load_image_b64
    return {"type": "image_url", "image_url": {"url": _load_image_b64(path)}}


def describe_image(path):
    alias = _vision_profile()
    if not alias:
        return f"Error: {path} is an image and no vision model is configured (configure the 'super' profile)."
    _, vprof = get_active_profile(alias)
    r = llm.chat(vprof, [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image precisely for a coding agent that cannot see it: layout, UI elements, "
                                 "colors, any charts/diagrams, and transcribe all visible text verbatim."},
        _image_part(path)]}], max_tokens=1500, thinking=False)
    return f"Image {path} (described by a vision model; you can't view it directly):\n{(r.get('content') or '').strip()}"


def _handle_images(results, prof):
    """read_file image markers -> (results, paths to show the model this turn)."""
    from .tools import IMAGE_MARK
    show = []
    for i, r in enumerate(results):
        if not r.startswith(IMAGE_MARK):
            continue
        path = r[len(IMAGE_MARK):]
        if prof.get("vision"):
            show.append(path)
            results[i] = f"Image {path} is attached in the next message."
        else:
            try:
                results[i] = describe_image(path)
            except Exception as e:
                results[i] = f"Error: couldn't describe image {path}: {e}"
    return results, show


# Old tool outputs are trimmed in what we SEND (history keeps them in full): every step resends the whole
# conversation, so stale file dumps and logs dominate prefill time in long sessions. The cutoff moves in
# batches so the request prefix stays stable between steps (servers with prefix caching keep their cache).
KEEP_TOOL_RESULTS = int(os.environ.get("ZEUSAI_KEEP_TOOL_RESULTS", 12))
ELIDE_BATCH = 8
ELIDE_OVER = 2000  # chars; smaller results are always sent whole
ELIDE_HEAD = 600


def _elide_stale(rest):
    idx = [i for i, m in enumerate(rest) if m.get("role") == "tool"]
    old = len(idx) - KEEP_TOOL_RESULTS
    if old <= 0:
        return rest
    cutoff = (old // ELIDE_BATCH) * ELIDE_BATCH
    if not cutoff:
        return rest
    names = {tc["id"]: tc["function"] for m in rest for tc in m.get("tool_calls") or []}
    out = list(rest)
    for i in idx[:cutoff]:
        c = out[i].get("content") or ""
        if len(c) <= ELIDE_OVER:
            continue
        fn = names.get(out[i].get("tool_call_id")) or {}
        call = f"{fn.get('name', 'tool')}({(fn.get('arguments') or '')[:160]})"
        out[i] = {**out[i], "content": c[:ELIDE_HEAD] + f"\n… [{len(c) - ELIDE_HEAD:,} more chars of this earlier "
                                                       f"{call} result omitted to save context; call it again if you need them]"}
    return out


_git_cache = {}  # cwd -> (time, text)


def git_context():
    """Branch, short status and recent commits (cached ~30s); "" outside a git repo."""
    import subprocess, time
    cwd = os.getcwd()
    hit = _git_cache.get(cwd)
    if hit and time.time() - hit[0] < 30:
        return hit[1]

    def git(*a):
        r = subprocess.run(["git", *a], capture_output=True, text=True, timeout=5, cwd=cwd)
        return r.stdout.strip() if r.returncode == 0 else None

    text = ""
    try:
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        if branch is not None:
            status = git("status", "--short") or "(clean)"
            lines = status.splitlines()
            if len(lines) > 30:
                status = "\n".join(lines[:30]) + f"\n… {len(lines) - 30} more"
            log = git("log", "--oneline", "-5") or "(no commits)"
            text = f"Git: branch {branch}\nStatus:\n{status}\nRecent commits:\n{log}"
    except (OSError, subprocess.TimeoutExpired):
        pass
    _git_cache[cwd] = (time.time(), text)
    return text


def user_message(prompt, image_src=None, profile_name=None, on_status=None):
    """A user message, with image_src attached for vision models or described by one for text-only models."""
    if not image_src:
        return {"role": "user", "content": prompt}
    try:
        if _profile(profile_name)[1].get("vision"):
            return {"role": "user", "images": [image_src],
                    "content": [{"type": "text", "text": prompt}, _image_part(image_src)]}
        if on_status:
            on_status("describing the image…")
        return {"role": "user", "content": f"{prompt}\n\n<attached-image src=\"{image_src}\">\n{describe_image(image_src)}\n</attached-image>"}
    except Exception as e:
        return {"role": "user", "content": f"{prompt}\n\n(couldn't attach image {image_src}: {e})"}


def _with_mode(messages, mode, todos=None, session=None):
    """Messages to send this turn: the system prompt plus the current mode's instructions."""
    import datetime
    extra = (f"Today's date: {datetime.date.today().isoformat()}. Working directory: {os.getcwd()}"
             + MODE_PROMPTS.get(mode, "") + "\n\n" + memory.system_context())
    git = git_context()
    if git:
        extra += "\n\n" + git + "\n(Git info is a snapshot from the start of this step.)"
    from . import styles
    style = styles.prompt_for((session.style if session is not None else "") or styles.default_style())
    if style:
        extra += "\n\n" + style
    if session is not None and session.append_prompt:
        extra += "\n\n" + session.append_prompt
    if todos:
        extra += "\n\nCurrent todo list:\n" + sessions.render_todos(todos)
    summaries = [m["content"] for m in messages if m.get("summary")]
    if summaries:
        extra += "\n\n" + "\n\n".join(summaries)
    rest = _elide_stale([{k: v for k, v in m.items() if k not in ("images", "reasoning")} if "images" in m or "reasoning" in m else m
                         for m in messages[1:] if not m.get("summary")])
    if not messages or messages[0].get("role") != "system":
        return messages
    base = session.system_prompt if session is not None and session.system_prompt else messages[0]["content"]
    return [{**messages[0], "content": base + "\n" + extra}] + rest


def _filter_tools(toolset, session):
    """Apply headless --allowed-tools / --disallowed-tools (names or labels like Bash, Read, Mcp)."""
    from .ui import tool_label
    allow, deny = set(session.allowed_tools), set(session.disallowed_tools)
    if not allow and not deny:
        return toolset
    keep = []
    for t in toolset:
        n = t["function"]["name"]
        ids = {n, tool_label(n)}
        if ids & deny or (allow and not ids & allow):
            continue
        keep.append(t)
    return keep


# Phrases that mean "I stopped to ask permission instead of finishing". Kept narrow on purpose:
# a normal sign-off like "let me know if you have questions" should not restart the agent.
_PAUSE_MARKERS = (
    "would you like me to continue", "would you like me to proceed", "want me to continue",
    "should i continue", "should i proceed", "shall i continue", "shall i proceed",
    "do you want me to continue", "do you want me to proceed", "ready for me to continue",
    "let me know if you'd like me to continue", "let me know if you want me to continue",
    "i can continue if you", "i'll wait for your go", "waiting for your approval",
    "waiting for your confirmation", "awaiting your confirmation", "awaiting your approval",
    "say the word and i", "confirm and i'll", "let me know and i'll continue",
    # narration that ends just before the work: the model said what it is about to do, then stopped
    "i'll now ", "i will now ", "next, i'll", "next i'll", "now i'll", "now let me",
    "let me now ", "i'm going to ", "i am going to ", "first, i'll", "first i'll",
    "let me check", "let me look", "let me read", "let me run", "let me fix",
    "let me update", "let me add", "let me start", "i'll start by", "i'll begin by",
)

_CONTINUE_NUDGE = (
    "[Continue working — do not pause]\n"
    "The task is not finished yet. Keep going without waiting for me: use the tools, make the remaining "
    "changes, and verify them. Do not ask for confirmation and do not stop to summarise a partial result. "
    "Only stop when every step is done and verified, or when you are truly blocked by a missing permission "
    "or information that only I can provide (use ask_user for that)."
)


def _unfinished_todos(todos):
    """Todo entries still pending or in progress (empty when there is nothing left to do)."""
    return [t for t in todos or [] if isinstance(t, dict) and t.get("status") != "completed"]


def _looks_paused(text):
    """True when a final answer is really a request for permission to keep going."""
    tail = " ".join((text or "").strip().lower().split())[-500:]
    return any(m in tail for m in _PAUSE_MARKERS)


def _chat_routed(msgs, session, toolset, primary_alias, hints, on_text=None,
                 on_reasoning=None, on_status=None, cancel=None, on_tool_delta=None):
    """One model call, routed across models on rate limits (see zeusai.routing).

    Tries the candidate chain in order; only a 429 (after llm.chat's own
    retries) moves on to the next candidate, and only before any reply text
    streamed (an already-shown partial answer can't be restarted elsewhere).
    """
    from .modes import effective_thinking
    from . import routing
    chain, skipped = routing.candidates(primary=primary_alias, tools=bool(toolset), **hints)
    if not chain:
        chain = [primary_alias]
    streamed = {"v": False}

    def _on_text(chunk):
        streamed["v"] = True
        if on_text:
            on_text(chunk)

    last = None
    tried = 0
    for cand in chain:
        try:
            _, cand_prof = _profile(cand)
        except SystemExit:
            continue  # fallback not configured: try the next one
        tried += 1
        try:
            r = llm.chat(cand_prof, msgs, tools=toolset, thinking=effective_thinking(session),
                         on_text=_on_text, on_reasoning=on_reasoning, cancel=cancel,
                         on_tool_delta=on_tool_delta)
        except llm.LLMError as e:
            last = e
            if getattr(e, "status", None) != 429:
                raise
            routing.mark_limited(cand)
            if streamed["v"]:
                raise  # a partial reply is already on screen; restarting elsewhere
            nxt = chain[chain.index(cand) + 1] if chain.index(cand) + 1 < len(chain) else None
            if nxt and on_status:
                on_status(f"↻ {cand} rate-limited — trying {nxt}")
            continue
        routing.record_ok(cand)
        if (tried > 1 or skipped) and on_status:
            on_status(f"↻ routed to {cand}" + (f" ({', '.join(skipped)} limited)" if skipped else ""))
        return r
    if last is not None:
        raise last
    raise llm.LLMError("no configured models available for this task")


def run_loop(messages, max_steps=60, verbose=False, profile_name=None,
             on_text=None, on_reasoning=None, on_status=None, cancel=None,
             on_tool_start=None, on_tool_end=None, on_busy=None, on_ask=None,
             session=None, on_permission=None, tools=None, on_tool_delta=None):
    """Agent tool loop. Streams text via on_text; mutates `messages` in place so history persists.

    Tool calls are reported through on_tool_start/on_tool_end, or as Claude Code-style
    lines ("● Read(x)" / "  ⎿  Read 12 lines") through on_status when those aren't given.
    on_tool_delta(index, name, args_json) fires as a tool call's arguments stream in,
    so a UI can preview e.g. a file being written before the call completes.
    on_busy() is called right before each model request.
    on_ask(question, options, multi) -> answer text handles the ask_user tool;
    without it the model is told the user isn't available.
    session (modes.Session) sets the mode and session permissions. When the model stops
    with unfinished todos or asks permission to keep going, the loop nudges it and
    continues instead of handing control back (session.auto_continue = False disables
    this; plan and research modes never auto-continue).
    on_permission(name, args, rule) -> (allowed: bool, note or None) answers "ask"
    decisions; without it they are denied.
    tools overrides the toolset (used by subagents); the default is the mode's tools plus `task`.
    """
    session = session or Session()
    # Execution modes pair a model with a thinking default; an explicit --model / /think still wins.
    from .modes import effective_thinking, profile_for_mode
    if profile_name in (None, "auto"):
        _has_image = any(m.get("images") for m in messages if isinstance(m, dict))
        _mode_profile = profile_for_mode(session.mode, has_image=_has_image)
        if _mode_profile:
            profile_name = _mode_profile
    # A concrete profile (from --model, /model, or a mode's preset) leads the
    # routing chain; a bare "auto" lets the router order by task + cost.
    chose_primary = profile_name not in (None, "auto")
    alias, prof = _profile(profile_name)
    primary_alias = alias

    if not messages or messages[0].get("role") != "system" or messages[0].get("summary"):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    if on_status is None and verbose:
        on_status = lambda s: console.print(escape(s))
    if on_tool_start is None:
        def on_tool_start(name, args):
            if not on_status:
                return
            on_status(f"● {tool_header(name, args)}")
            for line in edit_preview(name, args):
                on_status(f"  {line}")
    if on_tool_end is None:
        def on_tool_end(name, args, result):
            if on_status:
                on_status(f"  ⎿  {tool_summary(name, args, result)}")
                if verbose:
                    for line in tool_preview(result):
                        on_status(f"     {line}")

    def gate(name, args):
        if (session.allowed_tools or session.disallowed_tools) and not _filter_tools(
                [{"function": {"name": name}}], session):
            return f"Permission denied: {tool_header(name, args)} is not available in this session. Don't retry it."
        blocked, why = hooks.run("PreToolUse", name, args)
        if blocked:
            return f"Blocked by a PreToolUse hook: {why}"
        decision = permissions.check(name, args, session.mode, session.allowed, session.dirs)
        if decision == "allow":
            return None
        rule = permissions.suggest_rule(name, args)
        if decision == "deny":
            why = f"not allowed in {session.mode} mode" if session.mode in ("plan", "research") else "blocked by a deny rule"
            return f"Permission denied: {tool_header(name, args)} is {why}. Don't retry it."
        if on_permission is None:
            outside = permissions.outside_workspace(name, args, session.dirs)
            if outside:
                return (f"Permission denied: {outside} is outside the workspace and no user is available to approve it. "
                        f"This is a permission block, not a missing file: it may well exist. Don't retry it or look for it "
                        f"elsewhere; tell the user it was blocked and that they can rerun with --add-dir <dir> or add "
                        f"\"additionalDirectories\" in settings.")
            why = permissions.danger(name, args)
            if why:
                return (f"Permission denied: {tool_header(name, args)} is destructive ({why}) and needs confirmation, "
                        f"but no user is available. Don't retry it. To allow exactly this command, the user can add "
                        f"the allow rule {permissions.suggest_rule(name, args)}.")
            return DENIED_NONINTERACTIVE.format(rule=tool_header(name, args))
        allowed, note = on_permission(name, args, rule)
        if allowed:
            return None
        return "The user denied this action." + (f" Their instruction: {note}" if note else " Ask what to do instead.")

    top = tools is None  # subagents don't fire prompt/stop hooks
    last = messages[-1] if messages else {}
    if top and last.get("role") == "user":
        content = last.get("content")
        text = content if isinstance(content, str) else " ".join(
            p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text")
        blocked, out = hooks.run("UserPromptSubmit", prompt=text)
        if blocked:
            messages.pop()
            return f"Prompt blocked by a UserPromptSubmit hook: {out}"
        if out:
            hook_text = f"\n\n<hook-context>\n{out}\n</hook-context>"
            if isinstance(content, str):
                last["content"] += hook_text
            else:
                last["content"] = content + [{"type": "text", "text": hook_text}]
        sessions.begin_turn(session, messages)
    stop_retries = 0
    continues = {"todos": 0, "pause": 0}  # auto-continue nudges, per reason (reset by real work)

    for step in range(max_steps):
        sessions.maybe_compact(messages, profile_name, on_status)
        if on_busy:
            on_busy()
        toolset = tools if tools is not None else _filter_tools(tools_for(session.mode) + [agents.tool_schema()], session)
        try:
            from . import routing
            r = _chat_routed(_with_mode(messages, session.mode, session.todos, session), session, toolset,
                             primary_alias if chose_primary else None, routing.classify(messages),
                             on_text=on_text, on_reasoning=on_reasoning, on_status=on_status,
                             cancel=cancel, on_tool_delta=on_tool_delta)
        finally:
            for m in messages:  # images are shown for one request, then kept as text (small, serializable)
                if m.get("images"):
                    text = "".join(x.get("text", "") for x in m["content"] if isinstance(x, dict) and x.get("type") == "text")
                    m["content"] = (text + "\n" if text else "") + "[images shown: " + ", ".join(m.pop("images")) + "]"
        assistant_msg = {"role": "assistant", "content": r["content"] or None}
        if r.get("reasoning"):
            assistant_msg["reasoning"] = r["reasoning"]
        if r["tool_calls"]:
            assistant_msg["tool_calls"] = r["tool_calls"]
        messages.append(assistant_msg)

        if not r["tool_calls"]:
            cancelled = cancel is not None and cancel.is_set()
            if top and stop_retries < 3 and not cancelled:
                blocked, why = hooks.run("Stop", result=r["content"] or "")
                if blocked:
                    stop_retries += 1
                    if on_status:
                        on_status(f"✻ Stop hook: {why}")
                    messages.append({"role": "user", "content": f"[Stop hook feedback — address this before finishing]\n{why}"})
                    continue
            # Auto-continue: never hand control back while work is obviously unfinished.
            # A model that stops with pending todos, or that asks "shall I continue?", gets
            # nudged to keep going instead of pausing the task.
            if (top and not cancelled and getattr(session, "auto_continue", True)
                    and session.mode not in ("plan", "research")):
                pending = _unfinished_todos(session.todos)
                paused = _looks_paused(r["content"]) or not (r["content"] or "").strip()
                if pending and continues["todos"] < 2:
                    continues["todos"] += 1
                    reason = f"{len(pending)} unfinished todo(s)"
                elif paused and continues["pause"] < 3:
                    continues["pause"] += 1
                    reason = ("the reply was empty" if not (r["content"] or "").strip()
                              else "the reply stopped to ask for permission")
                else:
                    reason = None
                if reason:
                    if on_status:
                        on_status(f"↻ continuing ({reason})")
                    messages.append({"role": "user", "content": _CONTINUE_NUDGE, "nudge": True})
                    continue
            return r["content"] or ""
        continues = {"todos": 0, "pause": 0}  # real progress: the nudges may be used again later
        if cancel is not None and cancel.is_set():
            return r["content"] or ""
        if on_text and r["content"]:
            on_text("\n")

        task = None if tools is not None else (lambda a: agents.run_task(a, session, profile_name, on_permission, on_status))
        results = _run_tools(r["tool_calls"], on_tool_start, on_tool_end, on_ask, gate, session.todos, task)
        results, show = _handle_images(results, prof)
        for tc, result in zip(r["tool_calls"], results):
            name = tc["function"]["name"]
            if name not in ("ask_user", "todo_write", "task") and not result.startswith(("Permission denied", "Blocked by", "The user denied")):
                blocked, why = hooks.run("PostToolUse", name, _parse_args(tc), result[:20000])
                if blocked:
                    result += f"\n\n[PostToolUse hook feedback]\n{why}"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        if show:
            try:
                parts = [_image_part(x) for x in show]
                messages.append({"role": "user", "images": show,
                                 "content": [{"type": "text", "text": "Images from read_file:"}] + parts})
            except Exception as e:
                messages.append({"role": "user", "content": f"(couldn't attach images: {e})"})

    return "Reached max steps. Summarize what was done."


ALLOW_ONCE, ALLOW_ALWAYS, DENY = "Yes", "Yes, and don't ask again for {rule} in this project", "No"


def cli_permission(name, args, rule, session, before=None):
    """Terminal permission prompt. Returns (allowed, note)."""
    if before:
        before()
    header = tool_header(name, args)
    console.print(f"[bold yellow]●[/bold yellow] [bold]{escape(header)}[/bold]")
    for line in edit_preview(name, args):
        style = "green" if line.startswith("+") else "red" if line.startswith("-") else "dim"
        console.print(f"    [{style}]{escape(line)}[/{style}]")
    if name == "run_shell":
        console.print(f"    [dim]$ {escape(args.get('command', ''))}[/dim]")
    always = ALLOW_ALWAYS.format(rule=rule)
    answer = select_prompt(f"Allow {header}?", [ALLOW_ONCE, always, DENY])
    if answer == ALLOW_ONCE:
        return True, None
    if answer == always:
        where = permissions.save_rule(rule)
        session.allowed.append(rule)
        console.print(f"    [dim]saved allow rule {escape(rule)} to {escape(where)}[/dim]")
        return True, None
    if answer == DENY or answer.startswith("(skipped"):
        return False, None
    return False, answer  # typed instruction instead


def maybe_enhance(prompt, session, interactive=True):
    """Enhance-prompt mode: rewrite the prompt, let the user pick which to send."""
    if not session.enhance:
        return prompt
    spinner = Spinner()
    spinner.start("Enhancing your prompt")
    try:
        better = enhance_prompt(prompt)
    finally:
        spinner.stop()
    if better == prompt:
        return prompt
    console.print(Panel(escape(better), title="enhanced prompt", border_style="magenta"))
    if not interactive:
        return better
    choice = select_prompt("Send which prompt?", ["Use the enhanced prompt", "Use my original"])
    if choice == "Use the enhanced prompt":
        return better
    if choice == "Use my original" or choice.startswith("(skipped"):
        return prompt
    return choice  # typed an edited version


_learners = []  # background reflection threads, joined before exit


def learn_from(turn, background=False):
    """Auto-learn: reflect on a finished task and save lessons to memory (only when worthwhile)."""
    if not memory.auto_learn_enabled() or not memory.worth_learning(turn):
        return

    def go():
        saved = memory.reflect(turn)
        if saved:
            console.print(f"[magenta]✻ learned:[/magenta] [dim]{escape(', '.join(saved))}[/dim]")

    if background:
        t = threading.Thread(target=go)
        t.start()
        _learners.append(t)
        return
    spinner = Spinner()
    spinner.start("Learning from this task")
    try:
        saved = memory.reflect(turn)
    finally:
        spinner.stop()
    if saved:
        console.print(f"[magenta]✻ learned:[/magenta] [dim]{escape(', '.join(saved))}[/dim]")


def finish_learning(timeout=20):
    for t in _learners:
        t.join(timeout)
    _learners.clear()


def run_streaming(messages, verbose=False, profile_name=None, interactive=True, session=None, learn="sync"):
    """run_loop with live terminal output. Returns final text (already printed).

    In plan mode, ends by asking to approve the plan; approval switches the session
    to auto/default mode and executes it.

    While the model streams a file-writing tool call (write_file / edit_file /
    multi_edit / notebook_edit) the new code is shown live in a syntax-highlighted
    panel that grows token by token, then replaced by a static +/- diff when the
    call completes. This mirrors opencode / Claude Code: the file being built is
    always on screen. Set ZEUSAI_NO_LIVE=1 to disable the live panel.
    """
    session = session or Session()
    spinner = Spinner()
    out = StdoutStream(show_reasoning=verbose, spinner=spinner)
    live_enabled = sys.stdout.isatty() and not os.environ.get("ZEUSAI_NO_LIVE")
    # live file preview state: one Live panel tracks the most recent file tool
    live_state = {"live": None, "name": None, "args_text": "", "last": 0.0, "buf": []}

    def _flush_buffered():
        buf = live_state["buf"]
        if not buf:
            return
        live_state["buf"] = []
        for kind, chunk in buf:
            if kind == "text":
                out.text(chunk)
            else:
                out.reasoning(chunk)

    def _stop_live():
        live = live_state["live"]
        if live is not None:
            live_state["live"] = None
            live_state["name"] = None
            try:
                live.stop()
            except Exception:
                pass
            _flush_buffered()

    def _text_chunk(c):
        if live_state["live"] is not None:
            live_state["buf"].append(("text", c))
        else:
            out.text(c)

    def _reasoning_chunk(c):
        if live_state["live"] is not None:
            live_state["buf"].append(("reasoning", c))
        else:
            out.reasoning(c)

    def on_tool_delta(idx, name, args_text):
        """Realtime preview: called for every streamed chunk of a tool call."""
        if not live_enabled or name not in FILE_TOOLS:
            return
        live_state["args_text"] = args_text
        now = time.monotonic()
        if live_state["live"] is None:
            spinner.stop()
            out.end()
            from rich.live import Live
            live_state["live"] = Live("", console=console, refresh_per_second=15,
                                      transient=True, auto_refresh=False)
            live_state["live"].start()
            live_state["name"] = name
            live_state["last"] = 0.0
        elif live_state["name"] != name:
            # a different file tool started streaming (parallel calls): move the panel
            live_state["name"] = name
        if now - live_state["last"] < 0.05:
            return
        live_state["last"] = now
        panel = live_file_panel(name, args_text, live=True, max_lines=12)
        if panel is not None:
            live_state["live"].update(panel, refresh=True)

    def tool_start(name, args):
        _stop_live()
        out.end()
        header = tool_header(name, args)
        panel = diff_panel(name, args) if name in FILE_TOOLS else None
        if panel is not None:
            console.print(panel)
        else:
            console.print(f"[bold green]●[/bold green] [bold]{escape(header)}[/bold]")
        spinner.start(f"Running {header}")

    def tool_end(name, args, result):
        _stop_live()
        spinner.stop()
        failed = (result.startswith(("Error", "Tool error", "Permission denied", "The user denied", "Blocked by"))
                  or result.startswith("exit_code=") and not result.startswith("exit_code=0"))
        style = "red" if failed else "dim"
        console.print(f"  [{style}]⎿  {escape(tool_summary(name, args, result))}[/{style}]")
        if verbose:
            for line in tool_preview(result):
                console.print(f"     [dim]{escape(line)}[/dim]")

    def on_ask(question, options, multi):
        _stop_live()
        out.end()
        return select_prompt(question, options, multi)

    def on_permission(name, args, rule):
        _stop_live()
        return cli_permission(name, args, rule, session, before=out.end)

    def loop():
        return run_loop(messages, verbose=verbose, profile_name=profile_name,
                        on_text=_text_chunk, on_reasoning=_reasoning_chunk,
                        on_status=lambda s: (out.end(), console.print(f"[dim]{escape(s)}[/dim]")),
                        on_tool_start=tool_start, on_tool_end=tool_end,
                        on_tool_delta=on_tool_delta,
                        on_busy=lambda: (out.end(), spinner.start()),
                        on_ask=on_ask if interactive else None,
                        on_permission=on_permission if interactive else None,
                        session=session)

    # the turn starts at the latest user message (plus the reply it reacts to, for corrections)
    turn_start = max(0, len(messages) - 2)
    try:
        text = loop()
        while session.mode == "plan" and interactive:
            out.end()
            choice = select_prompt("Execute this plan?", PLAN_OPTIONS)
            if choice.startswith("Yes"):
                session.mode = "auto" if "auto-accept" in choice else "default"
                console.print(f"[dim]mode → {session.mode}[/dim]")
                messages.append({"role": "user", "content": "The plan is approved. Execute it now, then verify the result."})
                return loop()
            if choice == PLAN_OPTIONS[2] or choice.startswith("(skipped"):
                return text
            messages.append({"role": "user", "content": f"Revise the plan: {choice}"})
            text = loop()
        return text
    finally:
        _stop_live()
        out.end()
        if learn:
            try:
                learn_from(messages[turn_start:], background=(learn == "background"))
            except Exception:
                pass


INIT_PROMPT = """Explore this repository (list files, read the README, build/config files and a few key sources), then write ZEUSAI.md at the repo root for future coding sessions. Include only what's useful and non-obvious:
- what the project is (2-3 lines)
- how to install, build, run, lint and test (exact commands)
- architecture: main modules and how they fit together
- conventions and gotchas
If ZEUSAI.md exists, improve it instead of replacing good content. Keep it under 80 lines."""


CHAT_HELP = """Commands:
  /mode [default|auto|fast|thinking|executor|plan|research]  switch mode (no arg = cycle)
  /autocontinue [on|off]              keep working until the task is done (default on)
  /enhance                            toggle enhance-prompt
  /permissions                        show allow/deny rules
  /memory [delete <name>]             list (or delete) what zeusai has learned
  /init                               explore the repo and write ZEUSAI.md
  #<text>                             remember <text> (e.g. "#always use pytest -q")
  /model <alias>                      switch model
  /todos                              show the current todo list
  /compact [focus]                    summarize older history to free context
  /resume [id]                        list saved conversations, or resume one
  /rewind [n]                         undo the last n prompts (files + conversation)
  /agents /mcp /hooks /config /cost   subagents, MCP servers, hooks, settings, token usage
  /think [on|off]                     toggle extended reasoning for this session
  /export [file.md]                   save the conversation as Markdown
  /status /bashes                     session overview / background shells
  /style [name]                       output style: default, concise, explanatory, learning, or yours
  /context /doctor                    context-window usage / health checks
  /add-dir <path>                     let the agent use files in another directory
  /permissions [allow|deny|remove …]  view or edit permission rules
  /paste [prompt]                     send the clipboard image (with an optional prompt)
  /agents new <what it should do>     generate a custom subagent in .zeusai/agents/
  /review /commit /pr                 built-in git workflows (override in .zeusai/commands)
  end a line with \\ or wrap in \"\"\"     multi-line input · Tab completes /commands · ↑ history
  !<command>                          run a shell command yourself (output goes into the chat)
  @path                               attach a file or directory to your prompt
  /clear /tools /dir /help /quit"""


CHAT_COMMANDS = ["/help", "/quit", "/exit", "/clear", "/tools", "/dir", "/mode", "/autocontinue", "/enhance", "/permissions",
                 "/memory", "/init", "/model", "/todos", "/compact", "/resume", "/rewind", "/agents", "/mcp",
                 "/hooks", "/config", "/cost", "/think", "/export", "/status", "/bashes", "/style",
                 "/context", "/doctor", "/add-dir", "/paste"]


def setup_readline():
    """Arrow-key editing, persistent history (~/.zeusai/history) and Tab completion of /commands."""
    try:
        import readline
    except ImportError:
        return
    import atexit
    from . import commands
    from .config import get_config_dir
    hist = get_config_dir() / "history"
    try:
        hist.parent.mkdir(parents=True, exist_ok=True)
        readline.read_history_file(hist)
    except OSError:
        pass
    readline.set_history_length(2000)
    atexit.register(lambda: _save_history(readline, hist))

    def complete(text, state):
        if not text.startswith("/"):
            return None
        names = CHAT_COMMANDS + ["/" + n for n in commands.load()]
        hits = sorted(n for n in set(names) if n.startswith(text))
        return hits[state] if state < len(hits) else None

    readline.set_completer(complete)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")


def _save_history(readline, path):
    try:
        readline.write_history_file(path)
    except OSError:
        pass


def read_input(prompt):
    """input() with multi-line support: end a line with \\ to continue, or wrap a block in \"\"\"."""
    line = input(prompt)
    if line.strip() == '"""':
        lines = []
        while True:
            nxt = input("… ")
            if nxt.strip() == '"""':
                return "\n".join(lines)
            lines.append(nxt)
    while line.endswith("\\"):
        line = line[:-1] + "\n" + input("… ")
    return line


def export_markdown(messages, path=None):
    """Write the conversation as Markdown; returns the path."""
    import datetime
    path = path or f"zeusai-{datetime.datetime.now():%Y%m%d-%H%M%S}.md"
    out = [f"# zeusai conversation\n\n_{datetime.datetime.now():%Y-%m-%d %H:%M} · {os.getcwd()}_\n"]
    for m in messages:
        role, c = m.get("role"), m.get("content")
        if m.get("nudge"):
            continue
        if role == "system" and not m.get("summary"):
            continue
        if not isinstance(c, str):
            c = " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
        if m.get("summary"):
            out.append(f"## Summary of earlier conversation\n\n{c}\n")
        elif role == "user":
            out.append(f"## You\n\n{c}\n")
        elif role == "assistant":
            if m.get("reasoning"):
                out.append(f"## Thought\n\n{m['reasoning']}\n")
            if c:
                out.append(f"## zeusai\n\n{c}\n")
            for tc in m.get("tool_calls") or []:
                out.append(f"- `{tool_header(tc['function']['name'], _parse_args(tc))}`")
        elif role == "tool":
            body = (c or "")[:1500] + ("\n…" if len(c or "") > 1500 else "")
            out.append(f"<details><summary>result</summary>\n\n```\n{body}\n```\n</details>\n")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return path


def status_text(session, model):
    from . import tools as _tools
    from .modes import effective_thinking
    u = llm.USAGE
    shells = _tools.background_shells()
    running = sum(1 for _, code in shells.values() if code is None)
    think = {None: "model default", True: "on", False: "off"}[effective_thinking(session)]
    from . import styles
    think += f" · style {session.style or styles.default_style()}"
    git = git_context().split("\n")[0] if git_context() else "not a git repo"
    from . import routing
    return (f"model {model} · mode {session.mode} · thinking {think}\n"
            f"session {session.id or '-'} · {len(session.checkpoints)} checkpoints · todos {len(session.todos)}\n"
            f"tokens {u['prompt'] + u['completion']:,} over {u['requests']} requests · background shells {running} running\n"
            f"cwd {os.getcwd()} · {git}\n"
            f"{routing.status_text()}")


CONTEXT_WINDOW = int(os.environ.get("ZEUSAI_CONTEXT_WINDOW", 128_000))  # tokens; rough default for the hosted models


def context_text(messages, session, model):
    """What fills the context: estimated tokens per part (~4 chars/token), as sent on the next request."""
    import json
    msgs = messages if messages and messages[0].get("role") == "system" else [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    sent = _with_mode(msgs, session.mode, session.todos, session)
    tools = _filter_tools(tools_for(session.mode) + [agents.tool_schema()], session)
    parts = {"system prompt + context": len(sent[0]["content"]), "tool definitions": len(json.dumps(tools)),
             "your messages": 0, "replies + tool calls": 0, "tool results": 0}
    for m in sent[1:]:
        c = m.get("content")
        n = len(c) if isinstance(c, str) else len(json.dumps(c or ""))
        n += sum(len(tc["function"].get("arguments") or "") for tc in m.get("tool_calls") or [])
        key = {"user": "your messages", "assistant": "replies + tool calls", "tool": "tool results"}.get(m["role"])
        if key:
            parts[key] += n
    total = sum(parts.values()) // 4
    pct = total / CONTEXT_WINDOW * 100
    bar = "█" * int(min(pct, 100) / 5) + "░" * (20 - int(min(pct, 100) / 5))
    lines = [f"{model}: ~{total:,} / {CONTEXT_WINDOW:,} tokens  {bar} {pct:.0f}%"]
    lines += [f"  {k:<26} ~{v // 4:>8,} tokens" for k, v in parts.items()]
    lines.append(f"  auto-compact at ~{sessions.COMPACT_AT_CHARS // 4:,} tokens of history · /compact to do it now")
    return "\n".join(lines)


def doctor_text(profile_name=None):
    """Local health checks plus a live ping of the session model."""
    import shutil, subprocess, time
    from . import hooks, mcp
    from .config import get_config_path
    out = []
    ok = lambda good, msg: out.append(("✓ " if good else "✗ ") + msg)
    try:
        alias, prof = _profile(profile_name)
        t = time.monotonic()
        r = llm.chat(prof, [{"role": "user", "content": "Reply with: ok"}], max_tokens=8, thinking=False)
        ok(True, f"model {alias} answered in {time.monotonic() - t:.1f}s ({(r.get('content') or '').strip()[:20]!r})")
    except (Exception, SystemExit) as e:
        ok(False, f"model: {e}")
    cfg = get_config_path()
    if cfg.exists():
        mode = oct(cfg.stat().st_mode & 0o777)
        ok(mode == "0o600", f"config {cfg} (permissions {mode}{'' if mode == '0o600' else ', should be 600: it holds API keys'})")
    else:
        ok(False, f"config {cfg} missing (zeusai configure ...)")
    ok(bool(shutil.which("rg")), "ripgrep for fast search" + ("" if shutil.which("rg") else " (missing: falls back to grep)"))
    ok(bool(shutil.which("git")), "git" + ("" if shutil.which("git") else " (missing)"))
    gh = shutil.which("gh")
    ok(bool(gh), "gh for /pr" + ("" if gh else " (missing: /pr won't work)"))
    try:
        json_ok = all(isinstance(hooks.load(), dict) for _ in [0])
        ok(json_ok, f"hooks: {sum(len(v) for v in hooks.load().values())} configured")
    except Exception as e:
        ok(False, f"hooks: {e}")
    for name, st in mcp.status().items():
        ok(st.startswith("connected"), f"MCP {name}: {st}")
    proj = os.path.join(os.getcwd(), ".zeusai", "settings.json")
    if os.path.isfile(proj):
        try:
            import json
            json.load(open(proj))
            ok(True, f"project settings {proj}")
        except ValueError as e:
            ok(False, f"project settings {proj}: invalid JSON ({e})")
    return "\n".join(out)


def permissions_command(arg, session):
    """`/permissions [allow|ask|deny <rule> | remove <rule>]`; returns text to show."""
    import json
    parts = arg.split(None, 1)
    if len(parts) == 2 and parts[0] in ("allow", "ask", "deny"):
        where = permissions.save_rule(parts[1].strip(), kind=parts[0])
        return f"added {parts[0]} rule {parts[1].strip()} to {where}"
    if len(parts) == 2 and parts[0] == "remove":
        path = permissions._project_settings_path()
        data = permissions._project_settings()
        rule, removed = parts[1].strip(), False
        for kind in ("allow", "ask", "deny"):
            lst = data.get("permissions", {}).get(kind, [])
            if rule in lst:
                lst.remove(rule)
                removed = True
        if rule in session.allowed:
            session.allowed.remove(rule)
            removed = True
        if removed and path.exists():
            path.write_text(json.dumps(data, indent=2))
        return f"removed {rule}" if removed else f"no project or session rule {rule} (user rules live in ~/.zeusai/config.json)"
    rules = permissions.load_rules()
    lines = [f"{k}: {', '.join(rules[k]) or '-'}" for k in ("allow", "ask", "deny")]
    lines.append(f"session: {', '.join(session.allowed) or '-'}")
    lines.append(f"workspace: {', '.join(str(d) for d in permissions.workspace_dirs(session.dirs))}")
    lines.append("edit: /permissions allow|ask|deny <rule> · /permissions remove <rule> · /add-dir <path>")
    return "\n".join(lines)


def add_dir(arg, session):
    p = os.path.abspath(os.path.expanduser(arg.strip())) if arg.strip() else ""
    if not p or not os.path.isdir(p):
        return f"not a directory: {arg.strip() or '(none)'}"
    if p not in session.dirs:
        session.dirs.append(p)
    return f"workspace now includes {p} (this session; add to \"additionalDirectories\" in settings to keep it)"


def session_start(session, source="startup"):
    """SessionStart hook: its stdout becomes extra context for this session."""
    blocked, out = hooks.run("SessionStart", prompt=source)
    if out and not blocked:
        session.append_prompt = (session.append_prompt + "\n\n" if session.append_prompt else "") + \
            f"<session-start-context>\n{out}\n</session-start-context>"


def session_end(session):
    try:
        hooks.run("SessionEnd", prompt=session.id)
    except Exception:
        pass


def bashes_text():
    from . import tools as _tools
    shells = _tools.background_shells()
    if not shells:
        return "No background shells."
    return "\n".join(f"{sid}: {'running' if code is None else f'exit {code}'} · {cmd}" for sid, (cmd, code) in shells.items())


def commands_exist(text):
    from . import commands
    return text[1:].split()[0] in commands.load() if len(text) > 1 else False


def prepare_prompt(text):
    """Expand a custom /command and @file mentions. Returns (prompt, note to show or None)."""
    from . import commands, mentions
    note = None
    expanded = commands.expand(text)
    if expanded is not None:
        note, text = f"/{text[1:].split()[0]} expanded", expanded
    text, attached = mentions.expand(text)
    if attached:
        note = (note + " · " if note else "") + "attached " + ", ".join(attached)
    return text, note


def bash_passthrough(cmd, messages):
    """`!cmd` in chat: run it directly and put the output in the conversation for the model to see."""
    import subprocess
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        out = (r.stdout + r.stderr).rstrip()
        code = r.returncode
    except subprocess.TimeoutExpired:
        out, code = "(timed out after 600s)", -1
    messages.append({"role": "user", "content": f"I ran `{cmd}` myself (exit {code}):\n```\n{out[-10000:]}\n```\n(No reply needed; this is context for later.)"})
    messages.append({"role": "assistant", "content": "Noted."})
    return out, code


def info_command(cmd):
    """Text for /agents, /mcp, /hooks, /cost, /config (shared by chat and TUI), or None."""
    from . import commands, hooks, mcp
    if cmd == "/agents":
        return "\n".join(f"{n}: {a['description']}" + (f"  [{a['path']}]" if a.get("path") else "")
                         for n, a in agents.all_agents().items())
    if cmd == "/mcp":
        st = mcp.status()
        return "\n".join(f"{n}: {v}" for n, v in st.items()) or "No MCP servers configured (zeusai mcp add <name> -- <command> [args])."
    if cmd == "/hooks":
        h = hooks.load()
        lines = [f"{e}: {x.get('matcher') or '*'} → {x['command']}" for e, items in h.items() for x in items]
        return "\n".join(lines) or "No hooks configured (see `hooks` in .zeusai/settings.json)."
    if cmd == "/cost":
        u = llm.USAGE
        est = f" ({u['estimated']} estimated)" if u["estimated"] else ""
        return f"{u['requests']} requests{est} · {u['prompt']:,} prompt + {u['completion']:,} completion tokens"
    if cmd == "/config":
        from .config import get_config_path
        proj = os.path.join(os.getcwd(), ".zeusai", "settings.json")
        cmds = commands.help_lines()
        return (f"user settings: {get_config_path()}\nproject settings: {proj}{'' if os.path.isfile(proj) else ' (none)'}\n"
                "keys: permissions, hooks, mcpServers, ensemble, auto_learn"
                + ("\ncustom commands:\n" + "\n".join(cmds) if cmds else ""))
    return None


def history_entries(messages):
    """Replayable entries for a saved conversation: (kind, text) pairs.

    kind is 'you', 'agent', 'thought', 'tool' or 'summary'. Tool entries carry
    the Claude-style header ("● Bash(...)") so callers can render them as
    separated sections with the full command string.
    """
    entries = []
    for m in messages:
        role = m.get("role")
        c = m.get("content")
        if m.get("nudge"):
            continue
        if role == "system" and not m.get("summary"):
            continue
        if not isinstance(c, str):
            c = " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
        c = c or ""
        if m.get("summary"):
            entries.append(("summary", c))
        elif role == "user":
            entries.append(("you", c))
        elif role == "assistant":
            if m.get("reasoning"):
                entries.append(("thought", m["reasoning"]))
            if c:
                entries.append(("agent", c))
            for tc in m.get("tool_calls") or []:
                from .ui import tool_header as _th
                entries.append(("tool", f"● {_th(tc['function']['name'], _parse_args(tc))}"))
        elif role == "tool":
            entries.append(("tool", f"  ⎿  {(c or '')[:500]}"))
    return entries


def _restore(session, messages, saved):
    """Load a saved conversation into this chat (in place), restoring mode/thinking/model."""
    messages[:] = saved.get("messages", [])
    session.todos[:] = saved.get("todos", [])
    session.id = saved["id"]
    # Restore the exact setup from when the user quit; explicit flags are applied by
    # start_chat after this, so saved values are the defaults here.
    if saved.get("mode") in permissions.MODES:
        session.mode = saved["mode"]
    if saved.get("thinking") is not None:
        session.thinking = saved["thinking"]
    turns = sum(1 for m in messages if m.get("role") == "user")
    console.print(f"[green]✓[/green] resumed [bold]{escape(saved.get('title', ''))}[/bold] "
                  f"[dim]({saved['id']}, {turns} prompts)[/dim]")
    # Replay the full history so the resumed chat shows everything, not just the last reply.
    for kind, text in history_entries(messages):
        text = (text or "").strip()
        if not text:
            continue
        if kind == "you":
            console.print(f"[bold]You:[/bold] {escape(text)}")
        elif kind == "thought":
            console.print(f"[dim]💭 Thought: {escape(text)}[/dim]")
        elif kind == "agent":
            console.print(Panel(Markdown(text[-3000:]), title="zeusai", border_style="dim"))
        elif kind == "tool":
            console.print(f"[dim]{escape(text)}[/dim]")
        elif kind == "summary":
            console.print(f"[dim]{escape(text[:500])}[/dim]")


def _print_sessions():
    items = sessions.listing()
    if not items:
        console.print("[dim]no saved conversations in this directory[/dim]")
    for it in items:
        console.print(f"[bold]{it['id']}[/bold] [dim]{it['updated']} · {it['turns']} prompts[/dim]  {escape(it['title'])}")
    return items


def start_chat(verbose=False, profile_name=None, mode="default", enhance=False, resume=None, dirs=()):
    """resume: None = new conversation, "" = most recent one here, or a session id (prefix)."""
    from .modes import apply_mode_defaults
    session = Session(mode=mode, enhance=enhance, id=sessions.new_id(),
                      dirs=[os.path.abspath(os.path.expanduser(d)) for d in dirs])
    messages = []
    explicitly_picked_model = profile_name not in (None, "auto")
    if resume is not None:
        saved = sessions.load(resume or None)
        if saved:
            _restore(session, messages, saved)
            # A saved profile is the default on resume; an explicit --model still wins.
            if not explicitly_picked_model and saved.get("profile"):
                profile_name = saved["profile"]
        else:
            console.print(f"[yellow]no saved conversation{' ' + escape(resume) if resume else ''} in this directory; starting fresh[/yellow]")
    # Mode presets (fast/thinking/executor) pick the model + thinking when the user didn't.
    profile_name = apply_mode_defaults(session, profile_name)
    # show profile alias, not internal model id
    try:
        alias, prof = _profile(profile_name)
    except SystemExit as e:
        console.print(f"[red]{e}[/red]")
        return
    llm.prewarm(prof["base_url"])
    console.print(Panel(f"[bold]zeusai[/bold] · {DISPLAY_MODEL} · {alias}\n"
                        f"Mode [bold]{session.mode}[/bold]: {permissions.MODES[session.mode]}\n"
                        f"Session [bold]{session.id}[/bold] · resume any time with [bold]zeusai chat -r {session.id}[/bold] or [bold]-c[/bold] for the latest\n"
                        "Type [bold]/help[/bold] for commands, [bold]/mode[/bold] to switch modes, [bold]/quit[/bold] to exit",
                        title="zeusai", border_style="blue"))

    def _save():
        try:
            sessions.save(session.id, messages, session.todos, mode=session.mode,
                          thinking=session.thinking, profile=_profile(profile_name)[0])
        except (OSError, SystemExit):
            pass

    from . import statusline
    setup_readline()
    pending_image = None  # set by /paste for the next prompt
    session_start(session, "resume" if resume is not None else "startup")
    while True:
        if messages:
            try:
                console.print(f"[dim]{escape(statusline.render(_profile(profile_name)[0], session))}[/dim]")
            except Exception:
                pass
        tag = session.mode + ("+enhance" if session.enhance else "")
        try:
            user_in = read_input(f"zeusai [{tag}]> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print()
            continue
        if not user_in:
            continue
        if user_in in ("/quit", "/exit", "quit"):
            break
        if user_in.startswith("#") and len(user_in) > 1:
            note = user_in[1:].strip()
            path = memory.save(" ".join(note.split()[:6]), note, note, "feedback")
            console.print(f"[magenta]✻ remembered[/magenta] [dim]{escape(str(path))}[/dim]")
            continue
        if user_in.startswith("/memory"):
            parts = user_in.split(None, 2)
            if len(parts) == 3 and parts[1] == "delete":
                console.print("[green]✓[/green] deleted" if memory.delete(parts[2]) else f"[red]no memory named {escape(parts[2])}[/red]")
                continue
            items = memory.entries()
            for e in items:
                console.print(f"[bold]{escape(e['name'])}[/bold] [dim]({e['type']})[/dim] {escape(e['description'])}")
            console.print(f"[dim]{len(items)} memories · {memory.project_dir()} · instructions: "
                          f"{', '.join(f for f in memory.INSTRUCTION_FILES if os.path.isfile(f)) or 'none'}[/dim]")
            continue
        if user_in == "/init":
            user_in = INIT_PROMPT
        if user_in.startswith("!") and len(user_in) > 1:
            out, code = bash_passthrough(user_in[1:].strip(), messages)
            console.print(escape(out) if out else "[dim](no output)[/dim]")
            if code:
                console.print(f"[red]exit {code}[/red]")
            continue
        if user_in.startswith("/think"):
            arg = user_in[len("/think"):].strip().lower()
            session.thinking = {"on": True, "off": False}.get(arg, not bool(session.thinking))
            console.print(f"[green]✓[/green] extended thinking {'on' if session.thinking else 'off'} (slower, deeper)")
            continue
        if user_in.startswith("/export"):
            path = export_markdown(messages, user_in[len("/export"):].strip() or None)
            console.print(f"[green]✓[/green] exported to {escape(path)}")
            continue
        if user_in.startswith("/style"):
            from . import styles
            arg = user_in[len("/style"):].strip()
            if not arg:
                console.print(escape(styles.listing(session.style or styles.default_style())))
            elif arg in styles.all_styles():
                session.style = arg
                console.print(f"[green]✓[/green] output style {escape(arg)}")
            else:
                console.print(f"[red]unknown style {escape(arg)}[/red]")
            continue
        if user_in.startswith("/agents new"):
            desc = user_in[len("/agents new"):].strip()
            if not desc:
                console.print("Usage: /agents new <what the agent should do>")
                continue
            spinner = Spinner()
            spinner.start("Designing the agent")
            try:
                path = agents.create_agent(desc, profile_name)
                console.print(f"[green]✓[/green] created {escape(str(path))} (edit it to adjust tools or prompt)")
            except Exception as e:
                console.print(f"[red]couldn't create agent:[/red] {escape(str(e))}")
            finally:
                spinner.stop()
            continue
        if user_in == "/context":
            console.print(escape(context_text(messages, session, _profile(profile_name)[0])))
            continue
        if user_in == "/doctor":
            spinner = Spinner()
            spinner.start("Running checks")
            try:
                text = doctor_text(profile_name)
            finally:
                spinner.stop()
            for line in text.splitlines():
                console.print(("[green]" if line.startswith("✓") else "[red]") + escape(line) + "[/]")
            continue
        if user_in.startswith("/add-dir"):
            console.print(escape(add_dir(user_in[len("/add-dir"):], session)))
            continue
        if user_in.startswith("/paste"):
            from . import clipboard
            try:
                img = clipboard.save_image()
            except RuntimeError as e:
                console.print(f"[red]{escape(str(e))}[/red]")
                continue
            pending_image = img
            user_in = user_in[len("/paste"):].strip() or "Look at this image."
            console.print(f"[dim]pasted image → {escape(img)}[/dim]")
        if user_in == "/status":
            console.print(escape(status_text(session, _profile(profile_name)[0])))
            continue
        if user_in == "/bashes":
            console.print(escape(bashes_text()))
            continue
        info = info_command(user_in.split()[0])
        if info is not None:
            console.print(escape(info))
            continue
        if user_in.startswith("/rewind"):
            arg = user_in[len("/rewind"):].strip()
            if not session.checkpoints:
                console.print("[dim]nothing to rewind[/dim]")
                continue
            what = select_prompt(f"Rewind the last {arg or 1} prompt(s)?",
                                 ["Restore code and conversation", "Restore code only", "Restore conversation only", "Cancel"])
            if what.startswith("Restore"):
                files, n = sessions.rewind(session, messages, int(arg) if arg.isdigit() else 1,
                                           code="conversation only" not in what, chat="code only" not in what)
                console.print(f"[green]✓[/green] rewound {n} prompt(s); restored {len(files)} file(s)"
                              + (": " + escape(", ".join(files)) if files else ""))
                _save()
            continue
        if user_in == "/clear":
            messages = []
            session.todos.clear()
            session.id = sessions.new_id()
            continue
        if user_in == "/todos":
            console.print(sessions.render_todos(session.todos))
            continue
        if user_in.startswith("/compact"):
            before = sessions.size(messages)
            spinner = Spinner()
            spinner.start("Compacting conversation")
            try:
                done = sessions.compact(messages, profile_name, focus=user_in[len("/compact"):].strip() or None)
            except Exception as e:
                done = None
                console.print(f"[red]compaction failed:[/red] {escape(str(e))}")
            finally:
                spinner.stop()
            if done:
                console.print(f"[green]✓[/green] compacted {before:,} → {sessions.size(messages):,} chars")
                _save()
            else:
                console.print("[dim]nothing to compact yet[/dim]")
            continue
        if user_in.startswith("/resume"):
            arg = user_in[len("/resume"):].strip()
            if not arg:
                items = _print_sessions()
                if not items:
                    continue
                pick = select_prompt("Resume which conversation?", [f"{i['id']}  {i['title'][:50]}" for i in items[:9]])
                arg = pick.split()[0] if pick and not pick.startswith("(skipped") else ""
            saved = sessions.load(arg) if arg else None
            if saved:
                _save()  # never lose the current conversation when switching
                _restore(session, messages, saved)
                if saved.get("profile"):
                    try:
                        _profile(saved["profile"])
                        profile_name = saved["profile"]
                    except SystemExit:
                        pass
            elif arg:
                console.print(f"[red]no saved conversation {escape(arg)}[/red]")
            continue
        if user_in == "/help":
            from . import commands
            console.print(CHAT_HELP + ("\nCustom commands:\n" + "\n".join(commands.help_lines()) if commands.load() else ""))
            continue
        if user_in == "/tools":
            console.print("Tools: " + ", ".join(t["function"]["name"] for t in tools_for(session.mode)))
            continue
        if user_in == "/dir":
            import pathlib
            console.print(str(pathlib.Path.cwd()))
            continue
        if user_in.startswith("/mode"):
            from .modes import set_mode
            parts = user_in.split()
            new = parts[1] if len(parts) == 2 else permissions.next_mode(session.mode)
            if new not in permissions.MODES:
                console.print(f"[red]Unknown mode {escape(new)}.[/red] Modes: {', '.join(permissions.MODE_ORDER)}")
            else:
                suggested = set_mode(session, new)
                if suggested:
                    try:
                        alias, _ = _profile(suggested)
                        profile_name = suggested
                        console.print(f"[green]✓[/green] mode {new}: {permissions.MODES[new]} · model {alias} · thinking {'on' if session.thinking else 'off'}")
                    except SystemExit:
                        console.print(f"[green]✓[/green] mode {new}: {permissions.MODES[new]} · thinking {'on' if session.thinking else 'off'}")
                else:
                    console.print(f"[green]✓[/green] mode {new}: {permissions.MODES[new]}")
            continue
        if user_in.startswith("/autocontinue"):
            parts = user_in.split()
            arg = parts[1].lower() if len(parts) == 2 else ""
            session.auto_continue = {"on": True, "off": False}.get(arg, not session.auto_continue)
            console.print(f"[green]✓[/green] auto-continue {'on' if session.auto_continue else 'off'}")
            continue
        if user_in == "/enhance":
            session.enhance = not session.enhance
            console.print(f"[green]✓[/green] enhance-prompt {'on' if session.enhance else 'off'}")
            continue
        if user_in.startswith("/permissions"):
            console.print(escape(permissions_command(user_in[len("/permissions"):].strip(), session)))
            continue
        if user_in.startswith("/model"):
            parts = user_in.split()
            if len(parts) == 2:
                try:
                    alias, _ = _profile(parts[1])
                    profile_name = parts[1]
                    console.print(f"[green]✓[/green] switched to {alias}")
                except SystemExit as e:
                    console.print(f"[red]{e}[/red]")
            else:
                console.print("Usage: /model laguna|ultra|super")
            continue

        try:
            if user_in.startswith("/") and not commands_exist(user_in):
                console.print(f"[red]Unknown command {escape(user_in.split()[0])}.[/red] Try /help")
                continue
            prompt, note = prepare_prompt(user_in)
            if note:
                console.print(f"[dim]{escape(note)}[/dim]")
            prompt = maybe_enhance(prompt, session)
            messages.append(user_message(prompt, pending_image, profile_name,
                                         on_status=lambda s: console.print(f"[dim]{escape(s)}[/dim]")))
            pending_image = None
            import time as _time
            started = _time.time()
            run_streaming(messages, verbose=verbose, profile_name=profile_name, session=session, learn="background")
            if _time.time() - started > float(os.environ.get("ZEUSAI_BELL_AFTER", 20)):
                sys.stdout.write("\a")  # terminal bell: long task finished
                sys.stdout.flush()
        except KeyboardInterrupt:
            console.print("[yellow]interrupted[/yellow]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
        finally:
            _save()
    _save()  # final save on quit so -c resumes this exact conversation
    finish_learning()
    session_end(session)
    if any(m.get("role") == "user" for m in messages):
        console.print(f"[dim]session {session.id} · resume with: zeusai chat --resume {session.id} or zeusai chat -c[/dim]")


def ask_once(prompt, profile_name=None, image_url=None, stream=False):
    """Single ask, optionally with vision image_url. stream=True prints tokens live."""
    # paligemma special
    if profile_name == "paligemma":
        from .ensemble import call_paligemma
        r = call_paligemma("paligemma", prompt, image_url, stream=stream)
        if r.get("error"):
            raise Exception(r["error"])
        return r["content"] or ""
    alias, prof = _profile(profile_name)

    # build message content
    if image_url:
        if not prof.get("vision"):
            console.print(f"[yellow]Note:[/yellow] {alias} is not vision-capable, sending as text URL")
        from .ensemble import _load_image_b64
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _load_image_b64(image_url)}}
        ]
    else:
        content = prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]

    spinner = Spinner()
    if stream:
        out = StdoutStream(show_reasoning=True, spinner=spinner)
        spinner.start()
        try:
            r = llm.chat(prof, messages, on_text=out.text, on_reasoning=out.reasoning)
        finally:
            out.end()
    else:
        spinner.start()
        try:
            r = llm.chat(prof, messages)
        finally:
            spinner.stop()
        if r["reasoning"]:
            console.print(f"[dim]thinking: {r['reasoning'][:1000]}[/dim]")
    content = r["content"]
    if r["reasoning"] and not content:
        content = r["reasoning"]
    return content


def start_ensemble_chat(verbose=False, judge_alias=None):
    from .ensemble import ensemble_ask
    from .config import BASE_URL, ensemble_judge
    judge_alias = ensemble_judge(judge_alias)
    llm.prewarm(BASE_URL)
    console.print(Panel(f"[bold]zeusai[/bold] ensemble · {DISPLAY_MODEL}\nAll models answer, judge merges best.\n/help, /quit, /models", title="zeusai ensemble", border_style="magenta"))
    while True:
        try:
            user_in = input("zeusai-ensemble> ").strip()
        except EOFError:
            break
        if not user_in:
            continue
        if user_in in ("/quit", "/exit", "quit"):
            break
        if user_in == "/help":
            console.print("Commands:\n  /help /quit\n  /models list profiles\n  /judge <alias> set judge\nType 'img:<url> <prompt>' to include image, e.g. img:https://... Describe")
            continue
        if user_in == "/models":
            from .config import list_profiles
            for r in list_profiles():
                console.print(f"{r['alias']}: {r['model_id']} ({'thinking' if r['thinking'] else ''}{' vision' if r['vision'] else ''})")
            continue
        if user_in.startswith("/judge"):
            parts = user_in.split()
            if len(parts) == 2:
                judge_alias = parts[1]
                console.print(f"[green]✓[/green] judge={judge_alias}")
            continue

        img = None
        prompt = user_in
        if prompt.startswith("img:"):
            # img:<url> <prompt>
            rest = prompt[4:].strip()
            # split first token as url
            parts = rest.split(None, 1)
            if len(parts) == 2:
                img, prompt = parts
            else:
                console.print("Usage: img:<url> <prompt>")
                continue

        try:
            ensemble_ask(prompt, image_src=img, judge_alias=judge_alias, verbose=True, stream=True)
        except KeyboardInterrupt:
            console.print("[yellow]interrupted[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
