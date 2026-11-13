import os
import pathlib
import subprocess
import shutil
from rich.console import Console

console = Console()

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", "dist", "build", ".tox"}
RG = shutil.which("rg")
# tools with no side effects: safe to run concurrently in one model turn
READ_ONLY_TOOLS = {"read_file", "list_files", "search_code", "glob_files", "web_search", "web_fetch", "shell_output"}
READ_LIMIT = 2000  # lines returned by read_file when no limit is given
_read_paths = set()  # files the agent has read this process; edits to unread existing files are refused
_bg = {}  # background shells: id -> {"proc", "log", "cmd"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
IMAGE_MARK = "[image] "  # read_file result prefix; agent.run_loop turns it into vision input or a description
WEB_TOOLS = {"web_search", "web_fetch"}

# Tools exposed to the agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file. Output lines are prefixed with their line number and a tab (the prefix is not part of the file; never include it in edit_file strings). Reads up to 2000 lines; use offset/limit for more. You must read a file before editing or overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"},
                               "offset": {"type": "integer", "description": "1-based line to start at"},
                               "limit": {"type": "integer", "description": "Number of lines to read"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with given content.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory, default '.'"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return stdout/stderr. Use for tests, builds, git, etc. Set run_in_background for servers/watchers or long jobs, then poll with shell_output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"},
                               "timeout": {"type": "integer", "description": "Seconds, default 120, max 600"},
                               "run_in_background": {"type": "boolean"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search file contents with grep -rn. Returns matching lines.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": "Find files by glob pattern, e.g. '**/*.py' or 'src/**/test_*.ts'. Returns paths, newest first.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "description": "Base directory, default '.'"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "Track multi-step work as a todo list shown to the user. Send the FULL list each time (it replaces the old one). Use for tasks with 3+ steps: add items up front, mark one in_progress before starting it, and completed right after finishing it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact string old_string with new_string in a file. old_string must be unique unless replace_all is true. Read the file first.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
                               "replace_all": {"type": "boolean", "description": "Replace every occurrence"}},
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_edit",
            "description": "Apply several exact-string edits to one file, in order, all-or-nothing. Each edit is {old_string, new_string, replace_all?}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {"type": "array", "items": {"type": "object", "properties": {
                        "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}},
                        "required": ["old_string", "new_string"]}},
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notebook_edit",
            "description": "Edit a Jupyter notebook cell: replace its source, insert a new cell before it (or at the end with index = number of cells), or delete it. Read the notebook first; cells are numbered from 0.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "cell_index": {"type": "integer"},
                    "new_source": {"type": "string"},
                    "cell_type": {"type": "string", "enum": ["code", "markdown"], "description": "For insert (default code) or to change type on replace"},
                    "edit_mode": {"type": "string", "enum": ["replace", "insert", "delete"], "description": "Default replace"},
                },
                "required": ["path", "cell_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_output",
            "description": "Get new output from a background shell started with run_shell(run_in_background=true), and whether it is still running.",
            "parameters": {"type": "object", "properties": {"shell_id": {"type": "string"}}, "required": ["shell_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_kill",
            "description": "Stop a background shell.",
            "parameters": {"type": "object", "properties": {"shell_id": {"type": "string"}}, "required": ["shell_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (DuckDuckGo). Returns numbered results with title, URL and snippet.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "description": "Default 6"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a web page and return its readable text (truncated). Use on URLs from web_search to read sources.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory",
            "description": "Your persistent memory across sessions. save: store one durable fact/lesson (overwrites same name). read: load a memory listed in the index. delete: remove a wrong or outdated one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["save", "read", "delete"]},
                    "name": {"type": "string", "description": "short kebab-case name"},
                    "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"]},
                    "description": {"type": "string", "description": "one-line summary shown in the index"},
                    "content": {"type": "string", "description": "the fact; for feedback include Why: and How to apply:"},
                },
                "required": ["action", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user a multiple-choice question when you need a decision or clarification you can't resolve from the code. The user picks an option (or types their own). Don't use it for things you can find out with other tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "One short, specific question."},
                    "options": {"type": "array", "items": {"type": "string"}, "description": "2-4 short, distinct answers. Put the recommended one first. An 'Other' free-text option is added automatically."},
                    "multi_select": {"type": "boolean", "description": "Allow picking several options. Default false."},
                },
                "required": ["question", "options"],
            },
        },
    },
]


def _apply_edit(text, e):
    """Returns (new_text, None) or (text, error)."""
    old, new = e.get("old_string", ""), e.get("new_string", "")
    if not old:
        return text, "Error: old_string is empty"
    if old == new:
        return text, "Error: old_string and new_string are identical"
    n = text.count(old)
    if n == 0:
        return text, "Error: old_string not found in file"
    if n > 1 and not e.get("replace_all"):
        return text, f"Error: old_string appears {n} times; add context or set replace_all"
    return text.replace(old, new), None


# file snapshots before each change, for /rewind (set by the agent per turn)
checkpoint_hook = None


def _checkpoint(p):
    if checkpoint_hook:
        checkpoint_hook(p)


def _render_notebook(p):
    import json
    nb = json.loads(p.read_text())
    out = []
    for i, c in enumerate(nb.get("cells", [])):
        src = "".join(c.get("source", []))
        out.append(f"--- cell {i} [{c.get('cell_type')}] ---\n{src}")
        for o in c.get("outputs", []) if c.get("cell_type") == "code" else []:
            text = "".join(o.get("text", [])) or "".join((o.get("data") or {}).get("text/plain", []))
            if o.get("output_type") == "error":
                text = f"{o.get('ename')}: {o.get('evalue')}"
            if text:
                out.append(f"[output]\n{text[:2000]}")
    return "\n".join(out) or "(empty notebook)"


def _notebook_edit(args):
    import json
    p = _safe_path(args["path"])
    if not p.exists():
        return f"Error: file not found: {p}"
    if str(p) not in _read_paths:
        return f"Error: read {p} before editing it"
    nb = json.loads(p.read_text())
    cells = nb.setdefault("cells", [])
    i, mode = int(args["cell_index"]), args.get("edit_mode") or "replace"
    src = (args.get("new_source") or "").splitlines(True)
    limit = len(cells) if mode == "insert" else len(cells) - 1
    if not 0 <= i <= limit:
        return f"Error: cell_index {i} out of range (notebook has {len(cells)} cells)"
    _checkpoint(p)
    if mode == "delete":
        cells.pop(i)
    elif mode == "insert":
        kind = args.get("cell_type") or "code"
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        cells.insert(i, cell)
    else:
        c = cells[i]
        c["source"] = src
        if args.get("cell_type") and args["cell_type"] != c.get("cell_type"):
            c["cell_type"] = args["cell_type"]
            if c["cell_type"] == "code":
                c.setdefault("outputs", []); c.setdefault("execution_count", None)
            else:
                c.pop("outputs", None); c.pop("execution_count", None)
        if c.get("cell_type") == "code":
            c["outputs"], c["execution_count"] = [], None
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return f"{ {'replace': 'Replaced', 'insert': 'Inserted', 'delete': 'Deleted'}.get(mode, mode)} cell {i} in {p} ({len(cells)} cells now)"


def _start_background(cmd):
    import tempfile
    sid = f"bash_{len(_bg) + 1}"
    log = tempfile.NamedTemporaryFile(prefix=f"zeusai-{sid}-", suffix=".log", delete=False)
    proc = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, cwd=str(pathlib.Path.cwd()),
                            start_new_session=True)
    _bg[sid] = {"proc": proc, "log": log.name, "cmd": cmd, "pos": 0}
    return f"Started background shell {sid} (pid {proc.pid}). Poll with shell_output, stop with shell_kill."


def _bg_output(sid):
    b = _bg.get(sid)
    if not b:
        return f"Error: no background shell {sid}"
    with open(b["log"], errors="ignore") as f:
        f.seek(b["pos"])
        new = f.read()
        b["pos"] = f.tell()
    code = b["proc"].poll()
    state = "running" if code is None else f"exited with code {code}"
    if len(new) > 10000:
        new = "... (truncated)\n" + new[-10000:]
    return f"status: {state}\n{new or '(no new output)'}"


def background_shells():
    return {sid: (b["cmd"], b["proc"].poll()) for sid, b in _bg.items()}


def kill_background():
    for b in _bg.values():
        if b["proc"].poll() is None:
            b["proc"].terminate()


def _glob_regex(pattern):
    """Glob -> regex on '/'-separated relative paths: ** spans directories, * and ? stay within one."""
    import re
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
            else:
                body = pattern[i + 1:j]
                out.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
                i = j + 1
        elif c == "{" and "}" in pattern[i:]:  # {a,b} alternation
            j = pattern.index("}", i)
            out.append("(?:" + "|".join(re.escape(x) for x in pattern[i + 1:j].split(",")) + ")")
            i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _glob(base, pattern):
    """Files under base matching pattern, as [(relative path, mtime)]. Heavy dirs (SKIP_DIRS) are pruned
    while walking, unless the pattern names one explicitly (e.g. 'node_modules/foo/**')."""
    rx = _glob_regex(pattern.lstrip("./") if pattern.startswith("./") else pattern)
    wanted = {part for part in pattern.split("/") if part in SKIP_DIRS}
    recursive = "**" in pattern
    max_depth = None if recursive else pattern.count("/")
    hits = []
    for root, dirs, names in os.walk(base):
        rel = os.path.relpath(root, base)
        rel = "" if rel == "." else rel.replace(os.sep, "/") + "/"
        depth = rel.count("/")
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS or d in wanted]
        if max_depth is not None and depth >= max_depth:
            dirs[:] = []
        for n in names:
            if rx.match(rel + n):
                try:
                    hits.append((rel + n, os.stat(os.path.join(root, n)).st_mtime))
                except OSError:
                    pass
    return hits


def _safe_path(p: str):
    base = pathlib.Path.cwd().resolve()
    target = (base / p).resolve() if not os.path.isabs(p) else pathlib.Path(p).resolve()
    # allow only inside cwd or home? keep simple: warn but allow
    return target

def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "read_file":
            p = _safe_path(args["path"])
            if not p.exists():
                return f"Error: file not found: {p}"
            if p.is_dir():
                return f"Error: {p} is a directory; use list_files"
            if p.suffix.lower() in IMAGE_EXTS:
                if p.stat().st_size > 8_000_000:
                    return f"Error: image too large (>8MB): {p}"
                return IMAGE_MARK + str(p)
            if p.suffix == ".ipynb":
                _read_paths.add(str(p))
                return _render_notebook(p)
            offset = max(1, int(args.get("offset") or 1))
            limit = max(1, int(args.get("limit") or READ_LIMIT))
            lines = []
            with p.open(errors="ignore") as f:
                for n, line in enumerate(f, 1):
                    if n < offset:
                        continue
                    if n >= offset + limit:
                        lines.append(f"... (more lines; continue with offset={n})")
                        break
                    lines.append(f"{n:>6}\t{line[:2000].rstrip(chr(10))}")
            _read_paths.add(str(p))
            return "\n".join(lines) if lines else "(empty file)" if offset == 1 else f"(no lines at offset {offset})"
        elif name == "write_file":
            p = _safe_path(args["path"])
            if p.exists() and str(p) not in _read_paths:
                return f"Error: {p} exists and hasn't been read. Read it first, then write (or use edit_file)."
            _checkpoint(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            _read_paths.add(str(p))
            return f"Wrote {len(args['content'])} bytes to {p}"
        elif name == "list_files":
            p = _safe_path(args.get("path", "."))
            if not p.exists():
                return f"Error: path not found: {p}"
            # prune heavy dirs instead of walking them (rglob walked all of .venv/node_modules)
            files = []
            for root, dirs, names in os.walk(p):
                rel_root = pathlib.Path(root).relative_to(p)
                depth = len(rel_root.parts)
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                if depth >= 4:
                    dirs[:] = []
                for d in dirs:
                    files.append(str(rel_root / d) + "/")
                for n in sorted(names):
                    files.append(str(rel_root / n))
                if len(files) > 200:
                    files = files[:200] + ["... (truncated)"]
                    break
            return "\n".join(files) or "(empty)"
        elif name == "run_shell":
            cmd = args["command"]
            if args.get("run_in_background"):
                return _start_background(cmd)
            timeout = min(600, max(1, int(args.get("timeout") or 120)))
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(pathlib.Path.cwd()))
            except subprocess.TimeoutExpired:
                return f"Error: command timed out after {timeout}s (raise timeout, or use run_in_background)"
            out = (result.stdout or "") + (("\nSTDERR:\n" + result.stderr) if result.stderr else "")
            if len(out) > 10000:
                out = out[:10000] + "\n... (truncated)"
            return f"exit_code={result.returncode}\n{out}" if out.strip() else f"exit_code={result.returncode}\n(no output)"
        elif name == "search_code":
            pat = args["pattern"]
            base = args.get("path", ".")
            if RG:
                cmd = [RG, "-n", "--no-heading", "--max-columns", "300", "-e", pat, base]
            else:
                cmd = ["grep", "-rn"] + [f"--exclude-dir={d}" for d in SKIP_DIRS] + ["-e", pat, base]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            out = result.stdout or "(no matches)"
            return out[:10000]
        elif name == "glob_files":
            base = _safe_path(args.get("path", "."))
            hits = _glob(base, args["pattern"])
            if not hits:
                return "(no matches)"
            hits.sort(key=lambda h: h[1], reverse=True)
            return "\n".join(h[0] for h in hits[:300]) + ("\n... (truncated)" if len(hits) > 300 else "")
        elif name == "edit_file":
            p = _safe_path(args["path"])
            if not p.exists():
                return f"Error: file not found: {p}"
            if str(p) not in _read_paths:
                return f"Error: read {p} before editing it"
            text = p.read_text()
            new, err = _apply_edit(text, args)
            if err:
                return err
            _checkpoint(p)
            p.write_text(new)
            return f"Edited {p}"
        elif name == "multi_edit":
            p = _safe_path(args["path"])
            if not p.exists():
                return f"Error: file not found: {p}"
            if str(p) not in _read_paths:
                return f"Error: read {p} before editing it"
            text = p.read_text()
            for i, e in enumerate(args.get("edits") or [], 1):
                text, err = _apply_edit(text, e)
                if err:
                    return f"Error in edit {i}: {err[7:]} (no changes were written)"
            _checkpoint(p)
            p.write_text(text)
            return f"Applied {len(args.get('edits') or [])} edits to {p}"
        elif name == "notebook_edit":
            return _notebook_edit(args)
        elif name == "shell_output":
            return _bg_output(args["shell_id"])
        elif name == "shell_kill":
            b = _bg.get(args["shell_id"])
            if not b:
                return f"Error: no background shell {args['shell_id']}"
            if b["proc"].poll() is None:
                b["proc"].terminate()
                try:
                    b["proc"].wait(5)
                except subprocess.TimeoutExpired:
                    b["proc"].kill()
            return f"Stopped {args['shell_id']}"
        elif name.startswith("mcp__"):
            from .mcp import call
            return call(name, args)
        elif name == "memory":
            from .memory import memory_tool
            return memory_tool(args)
        elif name == "web_search":
            from .web import search
            hits = search(args["query"], int(args.get("max_results") or 6))
            if not hits:
                return "(no results)"
            return "\n\n".join(f"[{i}] {h['title']}\n{h['url']}\n{h['snippet']}" for i, h in enumerate(hits, 1))
        elif name == "web_fetch":
            from .web import fetch
            return fetch(args["url"])
        else:
            return f"Error: unknown tool {name}"
    except Exception as e:
        return f"Tool error ({name}): {e}"

import atexit
atexit.register(kill_background)
