"""Hooks: shell commands run on agent events, configured in ~/.zeusai/config.json or .zeusai/settings.json:

    "hooks": {
      "PreToolUse":  [{"matcher": "Bash|Write", "command": "./scripts/guard.sh"}],
      "PostToolUse": [{"matcher": "Update|Write", "command": "ruff format -q $(jq -r .args.path)"}],
      "UserPromptSubmit": [{"command": "echo 'Branch:' $(git branch --show-current)"}],
      "Stop": [{"command": "pytest -q >/dev/null 2>&1 || { echo 'tests fail' >&2; exit 2; }"}]
    }

Each command gets the event as JSON on stdin ({event, tool, label, args, result, prompt, cwd}).
`matcher` is a regex on the tool label (Bash, Write, Update, Read, ...) or tool name; omitted = all.
Exit code 2 blocks: PreToolUse skips the tool, PostToolUse/Stop/SubagentStop feed stderr back to the model,
UserPromptSubmit rejects the prompt, PreCompact cancels the compaction.
stdout (exit 0) is added as context for UserPromptSubmit and SessionStart, and as extra summary instructions
for PreCompact. SessionEnd runs when a chat, TUI or headless run exits.
"""
import json
import os
import re
import subprocess

from .config import _load_raw
from .ui import tool_label

EVENTS = ("PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop", "SubagentStop", "PreCompact",
          "SessionStart", "SessionEnd")
TIMEOUT = 60


def _project():
    from .permissions import _project_settings
    return _project_settings()


def load():
    out = {e: [] for e in EVENTS}
    for src in (_load_raw(), _project()):
        for e, items in (src.get("hooks") or {}).items():
            if e in out and isinstance(items, list):
                out[e] += [h for h in items if isinstance(h, dict) and h.get("command")]
    return out


def _matches(h, name):
    pat = h.get("matcher")
    if not pat or pat == "*" or name is None:
        return True
    try:
        return bool(re.fullmatch(pat, tool_label(name)) or re.fullmatch(pat, name))
    except re.error:
        return pat in (tool_label(name), name)


def run(event, name=None, args=None, result=None, prompt=None):
    """Run matching hooks. Returns (blocked: bool, message: str) — message is stderr if blocked, else stdout."""
    hooks = [h for h in load().get(event, []) if _matches(h, name)]
    if not hooks:
        return False, ""
    payload = json.dumps({"event": event, "tool": name, "label": tool_label(name) if name else None,
                          "args": args, "result": result, "prompt": prompt, "cwd": os.getcwd()})
    outs = []
    for h in hooks:
        try:
            r = subprocess.run(h["command"], shell=True, input=payload, capture_output=True, text=True,
                               timeout=int(h.get("timeout") or TIMEOUT))
        except subprocess.TimeoutExpired:
            outs.append(f"(hook timed out: {h['command']})")
            continue
        if r.returncode == 2:
            return True, (r.stderr or r.stdout or f"blocked by hook: {h['command']}").strip()
        if r.stdout.strip():
            outs.append(r.stdout.strip())
    return False, "\n".join(outs)
