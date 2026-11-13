"""Saved conversations (resume / continue), context compaction, and the session todo list."""
import datetime
import hashlib
import json
import os
import pathlib
import uuid

from . import llm
from .config import get_active_profile

# rough budget: ~4 chars per token; compact when history passes this many chars
COMPACT_AT_CHARS = int(os.environ.get("ZEUSAI_COMPACT_AT", 240_000))
KEEP_RECENT = 8  # messages kept verbatim after a compaction


def _dir(cwd=None):
    cwd = os.path.abspath(cwd or os.getcwd())
    tag = pathlib.Path(cwd).name + "-" + hashlib.sha1(cwd.encode()).hexdigest()[:8]
    d = pathlib.Path.home() / ".zeusai" / "sessions" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_id():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


def _title(messages):
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return " ".join(m["content"].split())[:70]
    return "(empty)"


def save(session_id, messages, todos=None, mode=None, thinking=None, profile=None, title=None):
    """Write the conversation (without the system prompt, which is rebuilt each run).

    mode/thinking/profile let `--continue` / `--resume` restore the exact setup the
    session was using when the user quit, not the flag defaults. `title` overrides
    the auto-generated first-prompt title (used for /rename).
    """
    body = [m for m in messages if (m.get("role") != "system" or m.get("summary")) and not m.get("nudge")]
    if not any(m.get("role") == "user" for m in body):
        return None
    path = _dir() / f"{session_id}.json"
    tmp = path.with_suffix(".tmp")
    data = {"id": session_id, "cwd": os.getcwd(), "title": title or _title(body),
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
            "todos": todos or [], "messages": body}
    if mode is not None:
        data["mode"] = mode
    if thinking is not None:
        data["thinking"] = thinking
    if profile is not None:
        data["profile"] = profile
    tmp.write_text(json.dumps(data))
    tmp.replace(path)
    return path


def tab_title(messages, fallback="New session", max_len=24):
    """Short title for a TUI tab from the first user prompt."""
    title = _title(messages)
    if not title or title == "(empty)":
        return fallback
    title = " ".join(title.split())
    return title if len(title) <= max_len else title[: max_len - 1] + "…"


def listing(limit=20):
    out = []
    for p in sorted(_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        out.append({"id": d["id"], "title": d.get("title", ""), "updated": d.get("updated", ""),
                    "turns": sum(1 for m in d.get("messages", []) if m.get("role") == "user"),
                    "mode": d.get("mode"), "profile": d.get("profile")})
    return out


def load(session_id=None):
    """Load a session by id (or id prefix); None loads the most recent one. Returns dict or None."""
    files = sorted(_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if session_id:
        files = [p for p in files if p.stem.startswith(session_id)]
    for p in files:
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            continue
    return None


# ---------------------------------------------------------------- compaction

def size(messages):
    n = 0
    for m in messages:
        n += len(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            n += len(tc["function"].get("arguments") or "")
    return n


SUMMARY_SYSTEM = """You compress a coding-agent conversation so work can continue in a fresh context.
Write a dense summary that keeps everything needed to carry on:
- the user's goals and explicit instructions/preferences (quote exact wording when it matters)
- files read or changed, with the key facts learned about them
- commands run and their outcomes (tests passing/failing, errors)
- decisions made and why
- what is done, what is in progress, and the next steps
Omit pleasantries and raw file dumps. Use short headed sections and bullets."""


def _render(messages, limit=180_000):
    parts = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            c = m.get("content") or ""
            parts.append(f"[tool result] {c[:1500]}{' …' if len(c) > 1500 else ''}")
            continue
        if m.get("content"):
            parts.append(f"[{role}] {m['content']}")
        for tc in m.get("tool_calls") or []:
            parts.append(f"[{role} called {tc['function']['name']}] {(tc['function'].get('arguments') or '')[:600]}")
    text = "\n\n".join(parts)
    return text[-limit:]


def _split_point(messages, keep):
    """Index where the kept tail starts; never orphan tool results from their assistant call."""
    i = max(0, len(messages) - keep)
    while i > 0 and messages[i].get("role") == "tool":
        i -= 1
    return i


def compact(messages, profile_name=None, focus=None, keep=KEEP_RECENT):
    """Replace older history with a model-written summary, in place. Returns the summary or None."""
    has_sys = bool(messages) and messages[0].get("role") == "system" and not messages[0].get("summary")
    head = messages[:1] if has_sys else []
    body = messages[len(head):]
    cut = _split_point(body, keep)
    old, recent = body[:cut], body[cut:]
    if not old:
        return None
    from . import hooks
    blocked, extra = hooks.run("PreCompact", prompt=focus or "")
    if blocked:
        return None
    if extra:
        focus = f"{focus}. {extra}" if focus else extra
    _, prof = get_active_profile(profile_name)
    ask = "Summarize this conversation." + (f" Focus especially on: {focus}" if focus else "")
    r = llm.chat(prof, [{"role": "system", "content": SUMMARY_SYSTEM},
                        {"role": "user", "content": f"{ask}\n\n<conversation>\n{_render(old)}\n</conversation>"}],
                 max_tokens=2500, thinking=False)
    summary = (r.get("content") or "").strip()
    if not summary:
        return None
    note = {"role": "system", "summary": True,
            "content": "Summary of the earlier part of this conversation (older messages were compacted):\n\n" + summary}
    messages[:] = head + [note] + recent
    return summary


def maybe_compact(messages, profile_name=None, on_status=None):
    if size(messages) < COMPACT_AT_CHARS:
        return False
    if on_status:
        on_status("✻ context is getting long: compacting older messages…")
    try:
        return compact(messages, profile_name) is not None
    except Exception as e:
        if on_status:
            on_status(f"  compaction failed: {e}")
        return False


# ---------------------------------------------------------------- todos

STATUS_MARK = {"pending": "☐", "in_progress": "◐", "completed": "☑"}


def todo_tool(args, todos):
    """Replace the session todo list (mutated in place) and return a rendering for the model."""
    items = args.get("todos")
    if not isinstance(items, list):
        return "Error: todos must be a list of {content, status}"
    clean = []
    for t in items:
        if not isinstance(t, dict) or not str(t.get("content", "")).strip():
            continue
        st = t.get("status") if t.get("status") in STATUS_MARK else "pending"
        clean.append({"content": str(t["content"]).strip(), "status": st})
    if sum(t["status"] == "in_progress" for t in clean) > 1:
        return "Error: only one todo may be in_progress at a time"
    todos[:] = clean
    return "Todo list updated:\n" + render_todos(todos)


def render_todos(todos):
    if not todos:
        return "(no todos)"
    return "\n".join(f"{STATUS_MARK[t['status']]} {t['content']}" for t in todos)


# ---------------------------------------------------------------- checkpoints (/rewind)

def begin_turn(session, messages):
    """Start a checkpoint for this prompt: the first change to each file stores its prior content."""
    from . import tools
    last = messages[-1].get("content") if messages else ""
    turn = {"at": len(messages) - 1, "files": {}, "prompt": last[:60] if isinstance(last, str) else "(prompt with image)"}
    session.checkpoints.append(turn)

    def snap(p):
        key = str(p)
        if key not in turn["files"]:
            turn["files"][key] = p.read_bytes() if p.exists() else None

    tools.checkpoint_hook = snap


def rewind(session, messages, n=1, code=True, chat=True):
    """Undo the last n prompts: restore files they changed and/or drop them from the conversation.
    Returns (files restored, prompts removed)."""
    n = max(1, min(n, len(session.checkpoints)))
    restored, at = [], None
    for turn in reversed(session.checkpoints[-n:]):
        if code:
            for path, data in turn["files"].items():
                p = pathlib.Path(path)
                if data is None:
                    if p.exists():
                        p.unlink()
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(data)
                restored.append(path)
        at = turn["at"]
    del session.checkpoints[-n:]
    if chat and at is not None and 0 <= at <= len(messages):
        del messages[at:]
    return sorted(set(restored)), n
