"""Status line shown above the prompt (chat) or in the cwd bar (TUI).

Default: model · mode · tokens · todos · directory. Customize with a command in settings:

    "statusLine": {"command": "echo \"$(git branch --show-current) · $(jq -r .model)\""}

The command gets {model, mode, cwd, session_id, usage, todos} as JSON on stdin; its first output line is shown.
"""
import json
import os
import subprocess

from . import llm
from .config import _load_raw


def _setting():
    from .permissions import _project_settings
    return _project_settings().get("statusLine") or _load_raw().get("statusLine") or {}


def render(model, session):
    u = llm.USAGE
    todos = session.todos
    data = {"model": model, "mode": session.mode, "cwd": os.getcwd(), "session_id": session.id,
            "usage": dict(u), "todos": todos}
    cmd = _setting().get("command")
    if cmd:
        try:
            r = subprocess.run(cmd, shell=True, input=json.dumps(data), capture_output=True, text=True, timeout=5)
            line = (r.stdout.strip().splitlines() or [""])[0]
            if line:
                return line
        except subprocess.TimeoutExpired:
            pass
    parts = [model, session.mode]
    tokens = u["prompt"] + u["completion"]
    if tokens:
        parts.append(f"{tokens / 1000:.1f}k tokens")
    if todos:
        parts.append(f"todos {sum(t['status'] == 'completed' for t in todos)}/{len(todos)}")
    home = os.path.expanduser("~")
    cwd = os.getcwd()
    parts.append("~" + cwd[len(home):] if cwd.startswith(home) else cwd)
    return " · ".join(parts)
