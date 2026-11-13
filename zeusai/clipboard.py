"""Read an image from the system clipboard (Wayland, X11, macOS) into ~/.zeusai/paste/.
Also copy text to the clipboard for /copy (wl-copy, xclip, xsel, pbcopy)."""
import datetime
import pathlib
import shutil
import subprocess

WRITERS = [
    (["wl-copy"], "wl-copy"),
    (["xclip", "-selection", "clipboard"], "xclip"),
    (["xsel", "--clipboard", "--input"], "xsel"),
    (["pbcopy"], "pbcopy"),
]


def copy_text(text: str) -> str:
    """Copy text to the system clipboard; returns the tool used (RuntimeError if none)."""
    tried = []
    for cmd, tool in WRITERS:
        if not shutil.which(cmd[0]):
            continue
        tried.append(tool)
        try:
            r = subprocess.run(cmd, input=text.encode(), capture_output=True, timeout=5)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0:
            return tool
    if not tried:
        raise RuntimeError("no clipboard tool found: install wl-clipboard (Wayland) or xclip (X11)")
    raise RuntimeError(f"clipboard write failed with {', '.join(tried)}")

READERS = [
    (["wl-paste", "--no-newline", "--type", "image/png"], "wl-paste"),
    (["xclip", "-selection", "clipboard", "-t", "image/png", "-o"], "xclip"),
    (["pngpaste", "-"], "pngpaste"),
]


def save_image():
    """Save the clipboard image as PNG and return its path; RuntimeError if there is none."""
    tried = []
    for cmd, tool in READERS:
        if not shutil.which(cmd[0]):
            continue
        tried.append(tool)
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=5)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0 and r.stdout.startswith(b"\x89PNG"):
            d = pathlib.Path.home() / ".zeusai" / "paste"
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{datetime.datetime.now():%Y%m%d-%H%M%S}.png"
            p.write_bytes(r.stdout)
            return str(p)
    if not tried:
        raise RuntimeError("no clipboard tool found: install wl-clipboard (Wayland) or xclip (X11)")
    raise RuntimeError(f"no image on the clipboard (checked with {', '.join(tried)})")
