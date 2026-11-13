"""@path mentions: `explain @src/app.py` attaches the file (or a directory listing) to the prompt."""
import os
import pathlib
import re

MENTION = re.compile(r"(?<![\w@])@((?:~|\.{1,2})?/?[\w.\-/]+)")
MAX_BYTES = 100_000


def expand(prompt):
    """Return (prompt_with_attachments, [attached paths])."""
    from .tools import IMAGE_EXTS, execute_tool
    blocks, seen = [], []
    for m in MENTION.finditer(prompt):
        raw = m.group(1).rstrip(".,;:)")
        p = pathlib.Path(os.path.expanduser(raw))
        if not p.exists() or raw in seen:
            continue
        seen.append(raw)
        if p.is_dir():
            blocks.append(f'<directory path="{raw}">\n{execute_tool("list_files", {"path": raw})}\n</directory>')
        elif p.suffix.lower() in IMAGE_EXTS:
            blocks.append(f'<image path="{raw}">The user attached this image; call read_file on it to view it.</image>')
        elif p.stat().st_size <= MAX_BYTES:
            blocks.append(f'<file path="{raw}">\n{execute_tool("read_file", {"path": raw})}\n</file>')
        else:
            blocks.append(f'<file path="{raw}">(too large to attach: {p.stat().st_size:,} bytes; use read_file with offset/limit)</file>')
    if not blocks:
        return prompt, []
    return prompt + "\n\n" + "\n\n".join(blocks), seen
