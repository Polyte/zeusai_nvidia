"""Output styles: how zeusai talks, layered on top of the system prompt.

Built-ins below; add your own as .zeusai/output-styles/<name>.md or ~/.zeusai/output-styles/<name>.md
(frontmatter `description:`, body = instructions). Set a default with "outputStyle": "<name>" in settings.
"""
import pathlib

from .memory import _parse

BUILTIN = {
    "default": {"description": "Concise, code-first engineering answers", "prompt": ""},
    "concise": {"description": "Minimal words: results and code only",
                "prompt": "Output style: terse. Lead with the result. No preamble, no recap, no explanations unless asked. "
                          "Bullets over paragraphs; at most 3 sentences of prose per reply."},
    "explanatory": {"description": "Explains the why behind choices as it works",
                    "prompt": "Output style: explanatory. As you work, briefly explain the reasoning behind important choices "
                              "and point out patterns or trade-offs in the codebase worth knowing, in short 'Insight:' notes. "
                              "Keep code changes the same as you otherwise would."},
    "learning": {"description": "Teaches: leaves small, well-scoped parts for you to write",
                 "prompt": "Output style: learning. Act as a pair-programming mentor. Do the scaffolding, but for one small, "
                           "meaningful piece of logic per task leave a clearly marked TODO(human) with guidance and ask the user "
                           "to implement it, then review their attempt. Explain concepts briefly as you go."},
}


def _dirs():
    return [pathlib.Path.home() / ".zeusai" / "output-styles", pathlib.Path.cwd() / ".zeusai" / "output-styles"]


def all_styles():
    out = dict(BUILTIN)
    for d in _dirs():
        if d.is_dir():
            for p in sorted(d.glob("*.md")):
                meta, body = _parse(p)
                out[meta.get("name") or p.stem] = {"description": meta.get("description", ""), "prompt": body}
    return out


def default_style():
    from .config import _load_raw
    from .permissions import _project_settings
    return _project_settings().get("outputStyle") or _load_raw().get("outputStyle") or "default"


def prompt_for(name):
    return (all_styles().get(name) or {}).get("prompt", "")


def listing(current):
    return "\n".join(f"{'●' if n == current else ' '} {n}: {s['description']}" for n, s in all_styles().items())
