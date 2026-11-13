"""Persistent instructions and learned memory, injected into every turn.

Instructions (written by people, read every turn):
    ~/.zeusai/ZEUSAI.md                      user-wide
    ./ZEUSAI.md, ./.zeusai/ZEUSAI.md         project (AGENTS.md / CLAUDE.md also read)

Learned memory (written by the agent, hidden from the repo):
    ~/.zeusai/memory/<project>/<name>.md     one fact per file, with frontmatter
    ~/.zeusai/memory/<project>/MEMORY.md     index; this is what goes into the prompt
    ~/.zeusai/memory/_global/                facts about the user that apply everywhere
"""
import json
import pathlib
import re
import threading

from .config import get_config_dir, get_ensemble_settings, _load_raw

INSTRUCTION_FILES = ["ZEUSAI.md", ".zeusai/ZEUSAI.md", "AGENTS.md", "CLAUDE.md"]
MAX_INSTRUCTIONS = 8000
MAX_INDEX = 4000
TYPES = ("user", "feedback", "project", "reference")
_lock = threading.Lock()


def _root():
    return get_config_dir() / "memory"


def project_dir(cwd=None):
    cwd = pathlib.Path(cwd or pathlib.Path.cwd()).resolve()
    return _root() / re.sub(r"[^A-Za-z0-9._-]+", "-", str(cwd)).strip("-")


def global_dir():
    return _root() / "_global"


def _dir_for(type_):
    return global_dir() if type_ == "user" else project_dir()


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60] or "note"


# ---- instructions ----

def instructions():
    parts = []
    user = get_config_dir() / "ZEUSAI.md"
    if user.is_file():
        parts.append(("~/.zeusai/ZEUSAI.md", user.read_text(errors="ignore")))
    # repo root -> cwd, so the most specific instructions come last (monorepo subpackages)
    cwd = pathlib.Path.cwd()
    chain = [cwd]
    for parent in cwd.parents:
        if parent == pathlib.Path.home() or parent == parent.parent:
            break
        chain.append(parent)
        if (parent / ".git").exists():
            break
    for d in reversed(chain):
        for rel in INSTRUCTION_FILES + ["ZEUSAI.local.md"]:
            p = d / rel
            if p.is_file():
                name = rel if d == cwd else str(p.relative_to(chain[-1].parent))
                parts.append((name, p.read_text(errors="ignore")))
    out, used = [], 0
    for name, text in parts:
        text = text.strip()[: MAX_INSTRUCTIONS - used]
        if text:
            out.append(f"## {name}\n{text}")
            used += len(text)
    return "\n\n".join(out)


# ---- learned memory ----

def _parse(path):
    text = path.read_text(errors="ignore")
    meta, body = {}, text
    m = re.match(r"---\n(.*?)\n---\n?(.*)", text, re.S)
    if m:
        body = m.group(2)
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, body.strip()


def entries():
    """All memories: list of dicts with name, description, type, path."""
    out = []
    for d in (global_dir(), project_dir()):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            if p.name == "MEMORY.md":
                continue
            meta, _ = _parse(p)
            out.append({"name": meta.get("name", p.stem), "description": meta.get("description", ""),
                        "type": meta.get("type", "project"), "path": str(p)})
    return out


def _write_index(d):
    lines = []
    for p in sorted(d.glob("*.md")):
        if p.name == "MEMORY.md":
            continue
        meta, _ = _parse(p)
        lines.append(f"- [{meta.get('name', p.stem)}]({p.name}) ({meta.get('type', 'project')}) — {meta.get('description', '')}")
    (d / "MEMORY.md").write_text("\n".join(lines) + ("\n" if lines else ""))


def index_text():
    lines = []
    for d, label in ((global_dir(), "about the user"), (project_dir(), "this project")):
        idx = d / "MEMORY.md"
        if idx.is_file() and idx.read_text().strip():
            lines.append(f"({label})\n" + idx.read_text().strip())
    return "\n".join(lines)[:MAX_INDEX]


def save(name, description, content, type_="project"):
    type_ = type_ if type_ in TYPES else "project"
    d = _dir_for(type_)
    with _lock:
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{slug(name)}.md"
        p.write_text(f"---\nname: {slug(name)}\ndescription: {' '.join(description.split())}\ntype: {type_}\n---\n\n{content.strip()}\n")
        _write_index(d)
    return p


def delete(name):
    with _lock:
        for d in (global_dir(), project_dir()):
            p = d / f"{slug(name)}.md"
            if p.is_file():
                p.unlink()
                _write_index(d)
                return True
    return False


def read(name):
    for d in (project_dir(), global_dir()):
        p = d / f"{slug(name)}.md"
        if p.is_file():
            return p.read_text(errors="ignore")
    return None


def memory_tool(args):
    action = args.get("action", "save")
    name = str(args.get("name") or "").strip()
    if action == "read":
        return read(name) or f"No memory named {name!r}."
    if action == "delete":
        return f"Deleted memory {name}." if delete(name) else f"No memory named {name!r}."
    if action in ("save", "update"):
        if not name or not args.get("content"):
            return "Error: save needs name and content."
        p = save(name, str(args.get("description") or name), str(args["content"]), str(args.get("type") or "project"))
        return f"Saved memory {slug(name)} ({p})."
    return f"Error: unknown action {action}"


MEMORY_GUIDE = """# Memory
You have persistent memory across sessions (the memory tool). The index below lists what you've learned before; read a memory with memory(action="read") when it's relevant to the task, and follow it.
Save a memory (action="save", one fact each, short kebab-case name) when:
- the user corrects you or states a preference (type "feedback", include why and how to apply it),
- you learn something about the user that applies everywhere (type "user"),
- you discover a non-obvious project fact: how to build/test, a gotcha, a convention (type "project"),
- you made a mistake worth not repeating (type "feedback").
Don't save what's obvious from the code, secrets, or one-off details. Update or delete memories that turn out wrong."""


def system_context():
    """Text appended to the system prompt each turn."""
    parts = []
    ins = instructions()
    if ins:
        parts.append("# Instructions from the user and project (follow these)\n" + ins)
    idx = index_text()
    parts.append(MEMORY_GUIDE + "\n\nMemory index:\n" + (idx or "(empty)"))
    return "\n\n".join(parts)


# ---- auto-learning ----

_BAD = ("Error", "Tool error", "Permission denied", "The user denied")
_CORRECTION = re.compile(r"\b(no|don'?t|do not|instead|wrong|actually|stop|never|always|prefer|should(n'?t)?)\b", re.I)


def auto_learn_enabled():
    return (_load_raw().get("memory") or {}).get("auto_learn", True)


def worth_learning(turn):
    """Only reflect when something happened worth remembering: failures, denials, answers, corrections."""
    for m in turn:
        c = m.get("content") or ""
        if not isinstance(c, str):
            continue
        if m["role"] == "tool" and (c.startswith(_BAD) or (c.startswith("exit_code=") and not c.startswith("exit_code=0"))
                                    or c.startswith("User answered:")):
            return True
        if m["role"] == "user" and _CORRECTION.search(c):
            return True
    return False


def _condense(turn, limit=6000):
    lines = []
    for m in turn:
        role, c = m["role"], m.get("content") or ""
        if not isinstance(c, str):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                lines.append(f"[tool call] {tc['function']['name']} {tc['function']['arguments'][:200]}")
        if c:
            lines.append(f"[{role}] {c[:600]}")
    text = "\n".join(lines)
    return text[-limit:]


REFLECT_SYSTEM = """You review a coding agent's transcript and extract durable lessons for its long-term memory.
A lesson is worth saving only if it will help on FUTURE tasks: a user correction or preference, a mistake the agent made and how to avoid it, a non-obvious fact about this project (how to build/test, gotchas, conventions), or a fact about the user.
Skip anything one-off, obvious from the code, already in the existing memory, or secret.
Output ONLY a JSON array (possibly empty) of at most 3 objects:
[{"name": "kebab-case-name", "type": "feedback|user|project|reference", "description": "one line", "content": "the lesson. For feedback: what to do, then 'Why:' and 'How to apply:'"}]"""


def reflect(turn, profile="laguna"):
    """Ask a fast model for 0-3 lessons and save them. Returns the saved names."""
    from . import llm
    from .config import get_active_profile
    try:
        _, prof = get_active_profile(profile)
        prompt = (f"Existing memory index:\n{index_text() or '(empty)'}\n\n"
                  f"Transcript:\n<transcript>\n{_condense(turn)}\n</transcript>\n\nReturn the JSON array now.")
        r = llm.chat(prof, [{"role": "system", "content": REFLECT_SYSTEM}, {"role": "user", "content": prompt}],
                     max_tokens=700, thinking=False)
        m = re.search(r"\[.*\]", r["content"] or "", re.S)
        items = json.loads(m.group(0)) if m else []
    except Exception:
        return []
    existing = {e["name"] for e in entries()}
    saved = []
    for it in items[:3]:
        if not isinstance(it, dict) or not it.get("name") or not it.get("content"):
            continue
        name = slug(str(it["name"]))
        if name in existing:
            continue
        save(name, str(it.get("description") or name), str(it["content"]), str(it.get("type") or "feedback"))
        saved.append(name)
    return saved
