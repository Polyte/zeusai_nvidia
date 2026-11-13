"""Live terminal feedback while the model works: an animated busy bar with
rotating messages, and Claude Code-style tool call lines (● Read(x) / ⎿ result)."""
import json
import random
import sys
import threading
import time

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
BAR_WIDTH = 12

MESSAGES = [
    "Consulting the rubber duck",
    "Herding semicolons",
    "Bribing the compiler",
    "Untangling spaghetti",
    "Negotiating with the linter",
    "Summoning lightning",
    "Polishing the thunderbolt",
    "Reading the docs (for once)",
    "Counting to infinity, twice",
    "Asking Stack Overflow's ghost",
    "Refactoring the universe",
    "Teaching bits to behave",
    "Warming up the GPUs",
    "Chasing an off-by-one",
    "Feeding the hamsters",
    "Aligning the brackets",
    "Hunting for the missing colon",
    "Rewriting it in Rust (not really)",
    "Turning coffee into code",
    "Arguing with tabs vs spaces",
    "Thinking very hard",
    "Dividing by zero, carefully",
    "Reticulating dependencies",
    "Making it work, then making it right",
]


def bar(tick, width=BAR_WIDTH):
    """Indeterminate progress bar: a 3-cell block bouncing left and right."""
    span = width - 3
    pos = tick % (2 * span)
    if pos > span:
        pos = 2 * span - pos
    return "▱" * pos + "▰▰▰" + "▱" * (span - pos)


def busy_line(tick, message, elapsed, hint="ctrl+c to interrupt"):
    return f"{FRAMES[tick % len(FRAMES)]} {message}… {bar(tick)} ({elapsed:.0f}s · {hint})"


class Spinner:
    """Animated status line on the terminal. Safe to start/stop repeatedly.

    Only draws when stdout is a TTY, so piped output stays clean.
    """

    def __init__(self, stream=None, interval=0.1):
        self.stream = stream or sys.stdout
        self.interval = interval
        self.enabled = self.stream.isatty()
        self.lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._label = None

    def start(self, label=None):
        if not self.enabled or self._thread is not None:
            return
        self._label = label
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        t = self._thread
        if t is None:
            return
        self._stop.set()
        t.join()
        self._thread = None
        with self.lock:
            self.stream.write("\r\x1b[2K")
            self.stream.flush()

    def _run(self):
        start = time.monotonic()
        message = self._label or random.choice(MESSAGES)
        tick = 0
        while not self._stop.is_set():
            elapsed = time.monotonic() - start
            # new joke every ~4s unless a fixed label was given
            if self._label is None and tick and tick % 40 == 0:
                message = random.choice(MESSAGES)
            with self.lock:
                self.stream.write("\r\x1b[2K\x1b[2m" + busy_line(tick, message, elapsed) + "\x1b[0m")
                self.stream.flush()
            tick += 1
            self._stop.wait(self.interval)


# ---- Claude Code-style tool lines ----

_TOOL_NAMES = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Update",
    "list_files": "List",
    "run_shell": "Bash",
    "search_code": "Search",
    "ask_user": "Ask",
    "web_search": "WebSearch",
    "web_fetch": "WebFetch",
    "memory": "Memory",
    "glob_files": "Glob",
    "todo_write": "Todo",
    "multi_edit": "Update",
    "shell_output": "BashOutput",
    "shell_kill": "KillShell",
    "task": "Task",
    "notebook_edit": "NotebookEdit",
}


def tool_label(name):
    if name and name.startswith("mcp__"):
        return "Mcp"
    return _TOOL_NAMES.get(name, name)


def _short(s, n=80):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def tool_header(name, args):
    """e.g. 'Read(stats.py)', 'Bash(pytest -q)'."""
    label = _TOOL_NAMES.get(name, name)
    if name.startswith("mcp__"):
        from .mcp import resolve
        srv, tool = resolve(name)
        return f"{srv}:{tool}({_short(json.dumps(args), 60)})"
    if name == "run_shell":
        arg = args.get("command", "")
    elif name == "search_code":
        arg = f'"{args.get("pattern", "")}"' + (f' in {args["path"]}' if args.get("path") else "")
    elif name == "list_files":
        arg = args.get("path", ".")
    elif name == "web_search":
        arg = args.get("query", "")
    elif name == "web_fetch":
        arg = args.get("url", "")
    elif name == "memory":
        arg = f'{args.get("action", "save")} {args.get("name", "")}'.strip()
    elif name == "task":
        arg = f'{args.get("agent") or "general"}: {args.get("description", "")}'
    elif name in ("shell_output", "shell_kill"):
        arg = args.get("shell_id", "")
    elif name == "glob_files":
        arg = args.get("pattern", "") + (f' in {args["path"]}' if args.get("path") else "")
    elif name == "todo_write":
        todos = args.get("todos") or []
        done = sum(1 for x in todos if isinstance(x, dict) and x.get("status") == "completed")
        arg = f"{done}/{len(todos)} done"
    else:
        arg = args.get("path", "")
    return f"{label}({_short(arg)})"


def tool_summary(name, args, result):
    """One-line result summary, e.g. 'Read 42 lines', 'exit 0 · 3 lines'."""
    if result.startswith(("Error", "Tool error", "Permission denied", "The user denied", "Blocked by")):
        return _short(result, 120)
    if name == "read_file" and result.startswith(("Image ", "[image]")):
        return "Read image"
    if name == "notebook_edit":
        return result.split(" in ")[0]
    if name == "read_file":
        return f"Read {result.count(chr(10)) + (0 if result.endswith(chr(10)) else 1)} lines"
    if name == "write_file":
        lines = args.get("content", "").count("\n") + 1
        return f"Wrote {lines} lines to {args.get('path', '')}"
    if name == "edit_file":
        old = args.get("old_string", "").count("\n") + 1
        new = args.get("new_string", "").count("\n") + 1
        return f"Updated {args.get('path', '')} (-{old} +{new} lines)"
    if name == "list_files":
        return f"Listed {len(result.splitlines())} paths"
    if name == "search_code":
        n = 0 if result == "(no matches)" else len(result.splitlines())
        return f"Found {n} matching line{'s' if n != 1 else ''}"
    if name == "web_search":
        n = result.count("\n[")  + result.startswith("[")
        return f"Found {n} result{'s' if n != 1 else ''}"
    if name == "web_fetch":
        return f"Fetched {len(result):,} chars"
    if name == "task":
        return "Done" + (f" ({result.rsplit('[subagent ', 1)[1].rstrip(']').split(': ', 1)[-1]})" if "[subagent " in result else "")
    if name == "multi_edit":
        return f"Applied {len(args.get('edits') or [])} edits to {args.get('path', '')}"
    if name == "shell_output":
        return result.partition("\n")[0]
    if name == "glob_files":
        n = 0 if result == "(no matches)" else len(result.splitlines())
        return f"Found {n} file{'s' if n != 1 else ''}"
    if name == "todo_write":
        return (result.partition("\n")[2] or result).replace("\n", "\n     ")
    if name == "run_shell":
        head, _, body = result.partition("\n")
        code = head.replace("exit_code=", "")
        n = 0 if body.strip() == "(no output)" else len(body.splitlines())
        return f"exit {code} · {n} line{'s' if n != 1 else ''} of output"
    return _short(result, 120)


def edit_preview(name, args, max_lines=30):
    """Unified diff lines for write_file/edit_file, shown before asking permission."""
    import difflib
    import pathlib
    if name == "edit_file":
        old, new = args.get("old_string", ""), args.get("new_string", "")
    elif name == "multi_edit":
        edits = [e for e in args.get("edits") or [] if isinstance(e, dict)]
        old = "".join(e.get("old_string", "") + "\n" for e in edits)
        new = "".join(e.get("new_string", "") + "\n" for e in edits)
    elif name == "write_file":
        try:
            old = pathlib.Path(args.get("path", "")).read_text(errors="ignore")
        except OSError:
            old = ""
        new = args.get("content", "")
    else:
        return []
    diff = [l.rstrip("\n") for l in difflib.unified_diff(old.splitlines(True), new.splitlines(True), n=1)][2:]
    diff = [l for l in diff if not l.startswith("@@")]
    return diff[:max_lines] + ([f"… +{len(diff) - max_lines} lines"] if len(diff) > max_lines else [])


def tool_preview(result, max_lines=6):
    """First lines of a tool result, for verbose mode."""
    lines = result.splitlines()
    out = lines[:max_lines]
    if len(lines) > max_lines:
        out.append(f"… +{len(lines) - max_lines} lines")
    return out


# ---- realtime file-write preview (opencode / Claude Code style) ----
#
# While the model streams a tool call, only the partial JSON of its arguments
# exists (e.g. a half-typed `content` string). These helpers decode that
# prefix on the fly so the CLI/TUI can show the file being built, line by
# line, instead of a blank spinner until the whole call lands.

FILE_TOOLS = ("write_file", "edit_file", "multi_edit", "notebook_edit")

_EXT_LEXER = {
    ".py": "python", ".pyw": "python", ".js": "javascript", ".mjs": "javascript",
    ".jsx": "javascript", ".cjs": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust", ".go": "go", ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp",
    ".cc": "cpp", ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".ps1": "bash",
    ".md": "markdown", ".txt": "text", ".json": "json", ".jsonc": "json",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini", ".cfg": "ini",
    ".xml": "xml", ".html": "html", ".htm": "html", ".css": "css", ".scss": "css",
    ".sql": "sql", ".ipynb": "json",
}


def guess_lexer(path):
    """Syntax lexer for a file path (extension based, 'text' fallback)."""
    import pathlib
    return _EXT_LEXER.get(pathlib.Path(str(path or "")).suffix.lower(), "text")


def _decode_json_string_prefix(s, start):
    """Decode a JSON string that opens at s[start] == '"', tolerating truncation.

    Returns (decoded_prefix, end_index) where end_index is just past the closing
    quote when the string is complete, or len(s) when it runs off the end.
    An incomplete trailing escape (a lone '\\' or '\\u12') is dropped.
    """
    out = []
    i = start + 1
    n = len(s)
    while i < n:
        c = s[i]
        if c == '"':
            return "".join(out), i + 1
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            break  # lone trailing backslash: incomplete escape
        e = s[i + 1]
        if e == "u":
            hexpart = s[i + 2:i + 6]
            if len(hexpart) < 4:
                break  # incomplete \uXXXX
            try:
                out.append(chr(int(hexpart, 16)))
            except ValueError:
                out.append(hexpart)
            i += 6
            continue
        simple = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
                  "/": "/", "\\": "\\", '"': '"'}
        out.append(simple.get(e, e))
        i += 2
    return "".join(out), n


def _find_string_field(s, key, start=0):
    """First '\"key\" : \"...\"' string value at/after start. Returns (value, end) or (None, start)."""
    import re
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"', s[start:])
    if not m:
        return None, start
    quote_at = start + m.end() - 1
    value, end = _decode_json_string_prefix(s, quote_at)
    return value, end


def partial_args(args_text):
    """Best-effort decode of possibly-truncated tool-argument JSON.

    Returns a dict; keys present are path, content, old_string, new_string,
    new_source, command and edits ([{old_string, new_string}, ...]).
    Missing keys are omitted. Works on the complete JSON too.
    """
    if not args_text:
        return {}
    out = {}
    for key in ("path", "content", "old_string", "new_string", "new_source", "command"):
        value, _ = _find_string_field(args_text, key)
        if value is not None:
            out[key] = value
    # multi_edit: every new_string fragment in order (the last one may be partial)
    news, pos = [], 0
    while True:
        value, end = _find_string_field(args_text, "new_string", pos)
        if value is None:
            break
        news.append(value)
        pos = max(end, pos + 1)
    if news:
        out["new_strings"] = news
        out["edits"] = [{"new_string": v} for v in news]
    # combine into a single preview string for the live view
    if out.get("content") is not None:
        out["_preview"] = out["content"]
    elif len(news) > 1:
        out["_preview"] = "\n".join(news)
    elif out.get("new_string") is not None:
        out["_preview"] = out["new_string"]
    elif out.get("new_source") is not None:
        out["_preview"] = out["new_source"]
    elif news:
        out["_preview"] = "\n".join(news)
    return out


def _count_diff_lines(diff):
    adds = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    dels = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    return adds, dels


def diff_panel(name, args, max_lines=40, title=None):
    """Rich Panel with a +/- colored unified diff for a file tool call (final, static)."""
    from rich.panel import Panel
    from rich.text import Text
    diff = edit_preview(name, args, max_lines=max_lines)
    header = title or tool_header(name, args)
    if not diff:
        return None
    body = Text()
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            style = "green"
        elif line.startswith("-") and not line.startswith("---"):
            style = "red"
        elif line.startswith("…"):
            style = "dim"
        else:
            style = "dim"
        body.append(line + "\n", style=style)
    adds, dels = _count_diff_lines(diff)
    stats = f"  +{adds} -{dels}" if adds or dels else ""
    return Panel(body, title=f"[bold green]●[/bold green] {header}{stats}", border_style="dim", title_align="left")


def live_file_panel(name, args_text, *, live=True, max_lines=24, footer=None):
    """Rich Panel showing the file being streamed, syntax highlighted, tail-visible.

    `args_text` is the partial argument JSON from on_tool_delta. Returns None
    for non-file tools. The code shown is the last `max_lines` lines so the
    newest code stays on screen, opencode-style.
    """
    if name not in FILE_TOOLS:
        return None
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    parsed = partial_args(args_text)
    path = parsed.get("path") or "…"
    header = tool_header(name, {"path": path})
    code = parsed.get("_preview", "")
    lines = code.splitlines()
    total = len(lines)
    shown = lines[-max_lines:] if total > max_lines else lines
    body_text = "\n".join(shown) if shown else ""
    if total > max_lines:
        overflow = Text(f"… {total - max_lines} lines above\n", style="dim")
    else:
        overflow = Text()
    hint = footer
    if not hint:
        if total == 0 and parsed.get("old_string"):
            hint = "matching old_string…"
        elif total == 0 and parsed.get("edits") is None and name == "multi_edit":
            hint = "planning edits…"
    lexer = guess_lexer(path)
    code_view = Syntax(body_text, lexer, line_numbers=True, word_wrap=False, theme="default") if body_text else Text("(streaming…)", style="dim")
    from rich.console import Group
    parts = [p for p in (overflow, code_view) if p]
    if hint:
        parts.append(Text(str(hint), style="dim"))
    body = Group(*parts) if len(parts) > 1 else (parts[0] if parts else Text(""))
    state = "building…" if live else "done"
    return Panel(body, title=f"[bold green]●[/bold green] {header} — {state} {total} line{'s' if total != 1 else ''}",
                 border_style="green", title_align="left")


def live_file_text(name, args_text, *, live=True, max_lines=20):
    """Textual-ready text for a live file preview (TUI). Returns None for non-file tools."""
    if name not in FILE_TOOLS:
        return None
    from rich.text import Text
    parsed = partial_args(args_text)
    path = parsed.get("path") or "…"
    header = tool_header(name, {"path": path})
    code = parsed.get("_preview", "")
    lines = code.splitlines()
    total = len(lines)
    shown = lines[-max_lines:] if total > max_lines else lines
    state = "building…" if live else "done"
    text = Text()
    text.append(f"● {header}", style="bold green")
    text.append(f" — {state} {total} lines" if total else f" — {state}", style="dim")
    text.append("\n")
    if total > max_lines:
        text.append(f"… {total - max_lines} lines above\n", style="dim")
    if total == 0 and parsed.get("old_string"):
        text.append("  matching old_string…\n", style="dim")
    elif total == 0:
        text.append("  (streaming…)\n", style="dim")
    for line in shown:
        text.append(f"  {line}\n")
    if live:
        text.append("▍", style="green")
    return text


# ---- interactive choice prompt (for the ask_user tool) ----

OTHER = "Other (type your own answer)"


def _read_key(fd):
    import os
    ch = os.read(fd, 1)
    if ch == b"\x1b":
        import select
        if select.select([fd], [], [], 0.05)[0]:
            ch += os.read(fd, 2)
    return ch


def select_prompt(question, options, multi=False, stream=None):
    """Arrow-key picker. Returns the chosen answer text (several joined by '; ' when multi).

    ↑/↓ or j/k move, 1-9 jump, space toggles (multi), enter submits, esc skips.
    The last row is always a free-text "Other" option. Falls back to a numbered
    prompt when stdin isn't a terminal.
    """
    stream = stream or sys.stdout
    rows = list(options) + [OTHER]
    if not sys.stdin.isatty():
        return _numbered_prompt(question, rows, multi, stream)

    import termios
    import tty
    fd = sys.stdin.fileno()
    cur, picked, drawn = 0, set(), 0
    hint = "↑/↓ move · space select · enter submit · esc skip" if multi else "↑/↓ move · enter select · esc skip"

    def draw():
        nonlocal drawn
        if drawn:
            stream.write(f"\x1b[{drawn}A")
        lines = [f"\x1b[1m? {question}\x1b[0m"]
        for i, opt in enumerate(rows):
            box = ("[x] " if i in picked else "[ ] ") if multi and opt != OTHER else ""
            if i == cur:
                lines.append(f"\x1b[36m❯ {i + 1}. {box}{opt}\x1b[0m")
            else:
                lines.append(f"  {i + 1}. {box}{opt}")
        lines.append(f"\x1b[2m  {hint}\x1b[0m")
        for line in lines:
            stream.write("\r\x1b[2K" + line + "\n")
        stream.flush()
        drawn = len(lines)

    def clear():
        stream.write(f"\x1b[{drawn}A" + "\r\x1b[2K\n" * drawn + f"\x1b[{drawn}A")

    old = termios.tcgetattr(fd)
    answer = None
    try:
        tty.setcbreak(fd)
        stream.write("\x1b[?25l")  # hide cursor
        draw()
        while answer is None:
            k = _read_key(fd)
            if k in (b"\x1b[A", b"k"):
                cur = (cur - 1) % len(rows)
            elif k in (b"\x1b[B", b"j"):
                cur = (cur + 1) % len(rows)
            elif k.isdigit() and 1 <= int(k) <= len(rows):
                cur = int(k) - 1
                if not multi:
                    answer = cur
            elif k == b" " and multi and rows[cur] != OTHER:
                picked ^= {cur}
            elif k in (b"\r", b"\n"):
                answer = cur
            elif k == b"\x1b":
                answer = -1
            elif k == b"\x03":
                raise KeyboardInterrupt
            if answer is None:
                draw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        stream.write("\x1b[?25h")
        clear()
        stream.flush()

    stream.write(f"\x1b[1m? {question}\x1b[0m\n")
    if answer == -1:
        text = "(skipped: user chose not to answer; proceed with your best judgment)"
    elif rows[answer] == OTHER and not (multi and picked):
        text = input("  ❯ ").strip() or "(no answer)"
        return text
    elif multi:
        chosen = [rows[i] for i in sorted(picked)] or [rows[answer]]
        if rows[answer] == OTHER and picked:
            chosen = [rows[i] for i in sorted(picked)]
        text = "; ".join(chosen)
    else:
        text = rows[answer]
    stream.write(f"\x1b[36m  ❯ {'(skipped)' if answer == -1 else text}\x1b[0m\n")
    stream.flush()
    return text


def _numbered_prompt(question, rows, multi, stream):
    stream.write(f"? {question}\n")
    for i, opt in enumerate(rows):
        stream.write(f"  {i + 1}. {opt}\n")
    stream.flush()
    try:
        raw = input("Choose " + ("numbers, comma separated" if multi else "a number") + " or type an answer: ").strip()
    except EOFError:
        return "(no answer: user is not available; proceed with your best judgment and state the assumption)"
    nums = [p.strip() for p in raw.split(",")]
    if nums and all(n.isdigit() and 1 <= int(n) <= len(rows) for n in nums):
        chosen = [rows[int(n) - 1] for n in nums if rows[int(n) - 1] != OTHER]
        if chosen:
            return "; ".join(chosen)
        return input("  answer: ").strip() or "(no answer)"
    return raw or "(no answer)"
