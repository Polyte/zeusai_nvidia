"""Custom slash commands: Markdown prompt templates in .zeusai/commands/ (project) or ~/.zeusai/commands/ (user).

    .zeusai/commands/review.md   ->  /review src/api.py
    ---
    description: Review a file for bugs
    ---
    Review $ARGUMENTS for correctness bugs. Report file:line for each finding.

$ARGUMENTS is the text after the command; $1, $2 ... are its words. A line `!`cmd`` is replaced by
the command's output, and `@path` by the file's contents (see mentions.expand).
Subfolders namespace commands: commands/git/commit.md -> /git:commit.
"""
import pathlib
import re
import subprocess

from .memory import _parse


def _dirs():
    return [pathlib.Path.home() / ".zeusai" / "commands", pathlib.Path.cwd() / ".zeusai" / "commands"]


# Built-ins; a file with the same name in a commands dir overrides them.
BUILTIN = {
    "review": {
        "description": "Review uncommitted changes (or a ref/PR given as argument) for bugs",
        "body": ("Review the code changes for correctness bugs, security issues and missing tests.\n"
                 "Target: $ARGUMENTS (if empty: all uncommitted changes, `git diff HEAD`; if a GitHub PR number, use "
                 "`gh pr diff <n>`; otherwise treat it as a git ref and diff against it).\n"
                 "Read surrounding code where needed to confirm each issue. Report findings ranked by severity, each with "
                 "file:line, what goes wrong and a concrete fix. Say so plainly if you find nothing serious. Don't modify files."),
    },
    "commit": {
        "description": "Stage and commit current changes with a good message",
        "body": ("Create a git commit for the current changes. Run `git status` and `git diff HEAD`, check `git log -5 "
                 "--oneline` for the message style, stage the relevant files (never secrets like .env), and commit with a "
                 "concise message explaining why. Extra instructions: $ARGUMENTS. Don't push."),
    },
    "pr": {
        "description": "Push the branch and open a pull request with gh",
        "body": ("Open a pull request for the current branch. Check `git status`, `git log` and the diff against the base "
                 "branch, push with -u if needed, then `gh pr create` with a clear title and a body summarizing the change "
                 "and how it was tested. Extra instructions: $ARGUMENTS."),
    },
}


def load():
    out = {n: {**c, "path": "(built-in)"} for n, c in BUILTIN.items()}
    for d in _dirs():  # project overrides user
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.md")):
            rel = p.relative_to(d).with_suffix("")
            name = ":".join(rel.parts)
            meta, body = _parse(p)
            out[name] = {"description": meta.get("description") or (body.splitlines()[0][:70] if body else ""),
                         "body": body, "path": str(p)}
    return out


def _shell(m):
    try:
        r = subprocess.run(m.group(1), shell=True, capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return f"(timed out: {m.group(1)})"


def expand(raw):
    """'/name args' -> prompt text, or None when it isn't a custom command."""
    if not raw.startswith("/"):
        return None
    name, _, rest = raw[1:].partition(" ")
    cmd = load().get(name)
    if not cmd:
        return None
    rest = rest.strip()
    words = rest.split()
    text = cmd["body"].replace("$ARGUMENTS", rest)
    text = re.sub(r"\$(\d+)", lambda m: words[int(m.group(1)) - 1] if 0 < int(m.group(1)) <= len(words) else "", text)
    text = re.sub(r"!`([^`]+)`", _shell, text)
    if "$ARGUMENTS" not in cmd["body"] and rest:
        text += f"\n\n{rest}"
    return text


def help_lines():
    return [f"  /{n:<34} {c['description']}" for n, c in sorted(load().items())]
