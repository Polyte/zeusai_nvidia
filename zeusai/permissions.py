"""Modes and permission rules for tool calls.

Rules look like `Bash(git status*)`, `Write(src/**)`, `Update(*)`, `Bash` (any call).
They live in ~/.zeusai/config.json (user) and ./.zeusai/settings.json (project):

    "permissions": {"allow": ["Bash(npm test*)"], "deny": ["Bash(rm -rf*)"], "ask": []}

Precedence: deny > ask > allow > mode default.
"""
import fnmatch
import json
import pathlib

from .config import _load_raw, _write_raw
from .ui import tool_label

# label -> the argument rules match against
READ_ONLY = {"Read", "List", "Search", "Glob", "Todo", "BashOutput", "WebSearch", "WebFetch", "Ask", "Memory"}

MODES = {
    "default": "asks before edits and shell commands",
    "auto": "runs every tool without asking (deny rules still apply)",
    "fast": "fastest model, thinking off, short answers, batch tools",
    "thinking": "deep model, extended thinking on, thorough checks",
    "executor": "strict checklist executor: follows instructions step-by-step",
    "plan": "read-only: explores and writes a plan for you to approve",
    "research": "searches the web and cites sources; no edits",
}
MODE_ORDER = ["default", "auto", "fast", "thinking", "executor", "plan", "research"]
# Modes that auto-allow tools (like auto). Deny rules + dangerous commands still ask.
AUTO_MODES = {"auto", "fast", "thinking", "executor"}


# Shell commands that are hard to undo: always confirmed, even in auto mode or under a broad allow rule.
# An explicit deny still wins; when nobody can answer (headless), they are refused.
DANGEROUS = [
    (r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(/\*?|~/?\*?|\$HOME/?\*?|\.\.?/?\*?|\*)(\s|$)", "recursive delete of a top-level path"),
    (r"\bmkfs(\.\w+)?\b", "formats a filesystem"),
    (r"\bdd\b.*\bof=/dev/", "writes a raw device"),
    (r">\s*/dev/(sd|nvme|disk|hd)", "overwrites a disk device"),
    (r":\(\)\s*\{\s*:\|:&\s*\};:", "fork bomb"),
    (r"\bchmod\s+-R\s+0?777\s+/(\s|$)", "opens permissions on /"),
    (r"(^|[;&|]\s*|sudo\s+)(shutdown|reboot|halt|poweroff)\b", "powers off the machine"),
    (r"\bgit\s+push\b.*(\s--force(?!-with-lease)\b|\s-f\b)", "force-push rewrites remote history"),
    (r"\bgit\s+reset\s+--hard\b", "discards uncommitted work"),
    (r"\bgit\s+clean\s+-[a-zA-Z]*f", "deletes untracked files"),
    (r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z)?sh\b", "pipes a download into a shell"),
    (r"\bdrop\s+(database|table)\b", "drops database objects"),
]


def danger(name, args):
    """Why a tool call is destructive, or None."""
    import re
    if name != "run_shell":
        return None
    cmd = args.get("command", "")
    for pat, why in DANGEROUS:
        if re.search(pat, cmd, re.I):
            return why
    return None


def next_mode(mode):
    return MODE_ORDER[(MODE_ORDER.index(mode) + 1) % len(MODE_ORDER)] if mode in MODE_ORDER else "default"


def rule_arg(name, args):
    """The string a rule's pattern is matched against."""
    if name.startswith("mcp__"):
        from .mcp import resolve
        return ":".join(resolve(name))
    if name == "run_shell":
        return args.get("command", "")
    if name in ("search_code", "glob_files"):
        return args.get("pattern", "")
    if name == "web_search":
        return args.get("query", "")
    if name == "web_fetch":
        return args.get("url", "")
    return args.get("path", ".")


def _project_settings_path():
    return pathlib.Path.cwd() / ".zeusai" / "settings.json"


def _project_settings():
    try:
        return json.loads(_project_settings_path().read_text())
    except (OSError, ValueError):
        return {}


def load_rules():
    rules = {"allow": [], "deny": [], "ask": []}
    for src in (_load_raw(), _project_settings()):
        perms = src.get("permissions") or {}
        for kind in rules:
            rules[kind] += [r for r in perms.get(kind, []) if isinstance(r, str)]
    return rules


def _matches(rule, label, arg):
    rule = rule.strip()
    if "(" not in rule:
        return rule == label
    rlabel, _, pat = rule.partition("(")
    pat = pat[:-1] if pat.endswith(")") else pat
    return rlabel == label and fnmatch.fnmatchcase(arg, pat)


PATH_TOOLS = {"read_file", "write_file", "edit_file", "multi_edit", "list_files", "glob_files", "search_code",
              "notebook_edit"}


def workspace_dirs(extra=()):
    """Directories the agent may use freely: cwd, `additionalDirectories` from settings, and /add-dir ones."""
    dirs = [pathlib.Path.cwd()]
    for src in (_load_raw(), _project_settings()):
        dirs += [pathlib.Path(d).expanduser() for d in src.get("additionalDirectories") or [] if isinstance(d, str)]
    dirs += [pathlib.Path(d).expanduser() for d in extra]
    return [d.resolve() for d in dirs]


def outside_workspace(name, args, extra=()):
    """The resolved path if this file tool targets something outside the workspace, else None."""
    if name not in PATH_TOOLS:
        return None
    raw = args.get("path") or "."
    target = (pathlib.Path.cwd() / pathlib.Path(raw).expanduser()).resolve()
    for d in workspace_dirs(extra):
        if target == d or d in target.parents:
            return None
    return str(target)


def check(name, args, mode, session_allow=(), extra_dirs=()):
    """Returns 'allow', 'deny' or 'ask' for this tool call in this mode."""
    label = tool_label(name)
    arg = rule_arg(name, args)
    rules = load_rules()
    if any(_matches(r, label, arg) for r in rules["deny"]):
        return "deny"
    if mode in ("plan", "research") and label not in READ_ONLY:
        return "deny"
    if any(_matches(r, label, arg) for r in rules["ask"]):
        return "ask"
    if danger(name, args):  # only an exact, wildcard-free allow rule for this very command skips the prompt
        exact = f"{label}({arg})"
        return "allow" if exact in list(rules["allow"]) + list(session_allow) else "ask"
    allowed = list(rules["allow"]) + list(session_allow)
    outside = outside_workspace(name, args, extra_dirs)
    if outside:  # files outside the workspace: ask, even in auto mode, unless a rule names that path
        return "allow" if any(_matches(r, label, outside) or _matches(r, label, arg) for r in allowed) else "ask"
    if any(_matches(r, label, arg) for r in allowed):
        return "allow"
    if label in READ_ONLY or mode in AUTO_MODES:
        return "allow"
    return "ask"


def suggest_rule(name, args):
    """A reasonable 'don't ask again' rule for this call."""
    label = tool_label(name)
    arg = rule_arg(name, args)
    if danger(name, args):
        return f"{label}({arg})"  # never generalize a destructive command
    if name == "run_shell":
        words = arg.split()
        return f"{label}({' '.join(words[:2])}*)" if words else label
    if name.startswith("mcp__"):
        return f"{label}({arg})"
    outside = outside_workspace(name, args)
    if outside:
        parent = str(pathlib.PurePath(outside).parent)
        return f"{label}({parent}/*)"
    if name in ("write_file", "edit_file", "multi_edit", "notebook_edit"):
        parent = str(pathlib.PurePath(arg).parent)
        return f"{label}({'' if parent == '.' else parent + '/'}*)"
    return label


def save_rule(rule, kind="allow", scope="project"):
    if scope == "project":
        p = _project_settings_path()
        data = _project_settings()
        perms = data.setdefault("permissions", {})
        lst = perms.setdefault(kind, [])
        if rule not in lst:
            lst.append(rule)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))
        return str(p)
    raw = _load_raw()
    lst = raw.setdefault("permissions", {}).setdefault(kind, [])
    if rule not in lst:
        lst.append(rule)
    _write_raw(raw)
    return "~/.zeusai/config.json"
