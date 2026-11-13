import json

from zeusai.ui import (partial_args, guess_lexer, diff_panel, live_file_panel,
                       live_file_text, edit_preview)


def test_partial_args_write_truncated():
    full = json.dumps({"path": "src/app.py", "content": "def main():\n    print('hi')\n"})
    # stream prefix: cut in the middle of the content string
    prefix = full[: full.index("print") + 10]
    out = partial_args(prefix)
    assert out["path"] == "src/app.py"
    assert out["_preview"].startswith("def main():\n")
    assert "print(" in out["_preview"]
    # lone trailing backslash is dropped, not emitted raw
    out2 = partial_args('{"path": "a.txt", "content": "hi\\')
    assert out2["content"] == "hi"


def test_partial_args_edit_and_multi():
    edit = json.dumps({"path": "a.py", "old_string": "x = 1", "new_string": "x = 2\ny = 3"})
    out = partial_args(edit)
    assert out["old_string"] == "x = 1" and out["new_string"] == "x = 2\ny = 3"
    multi = json.dumps({"path": "m.py", "edits": [
        {"old_string": "one", "new_string": "1"},
        {"old_string": "two", "new_string": "2"},
    ]})
    out2 = partial_args(multi)
    assert out2["new_strings"] == ["1", "2"] and out2["_preview"] == "1\n2"


def test_guess_lexer():
    assert guess_lexer("a.py") == "python"
    assert guess_lexer("src/app.TS") == "typescript"
    assert guess_lexer("README.md") == "markdown"
    assert guess_lexer("data.json") == "json"
    assert guess_lexer("blob.unknownext") == "text"
    assert guess_lexer("") == "text"


def test_diff_panel_write_and_edit(sandbox):
    # new file: every line is an addition
    panel = diff_panel("write_file", {"path": "new.py", "content": "x = 1\ny = 2\n"})
    assert panel is not None
    text = str(panel.renderable) if hasattr(panel, "renderable") else ""
    # panel title carries the tool header
    assert "new.py" in panel.title
    # edit: shows -/+ lines
    (sandbox / "a.py").write_text("x = 1\n")
    diff = edit_preview("edit_file", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"})
    assert any(l.startswith("-") for l in diff) and any(l.startswith("+") for l in diff)
    assert diff_panel("run_shell", {"command": "ls"}) is None


def test_live_panels_show_filename_and_tail():
    args = json.dumps({"path": "src/app.py", "content": "\n".join(f"line {i}" for i in range(30))})
    panel = live_file_panel("write_file", args, live=True, max_lines=10)
    assert panel is not None and "src/app.py" in panel.title and "building" in panel.title
    assert live_file_panel("run_shell", args) is None
    text = live_file_text("write_file", args, live=True, max_lines=10)
    plain = text.plain
    assert "src/app.py" in plain and "line 29" in plain  # tail visible
    assert "line 0" not in plain  # head collapsed
    assert "20 lines above" in plain
    done = live_file_text("write_file", args, live=False, max_lines=10)
    assert "done" in done.plain


def test_run_loop_forwards_tool_deltas(fake_model):
    """on_tool_delta passes through run_loop to llm.chat."""
    from zeusai import agent
    from zeusai.modes import Session
    seen = []

    def model(prof, msgs, **kw):
        cb = kw.get("on_tool_delta")
        if cb:
            cb(0, "write_file", '{"path": "x.py", "content": "hi"}')
        seen.append(kw.get("on_tool_delta") is not None)
        if msgs[-1]["role"] == "user":
            from conftest import tc
            return {"content": "", "tool_calls": [tc("1", "write_file", {"path": "x.py", "content": "hi"})]}
        return {"content": "ok", "tool_calls": []}

    fake_model(model)
    deltas = []
    m = [{"role": "user", "content": "x"}]
    agent.run_loop(m, session=Session(mode="auto"),
                   on_tool_delta=lambda i, n, a: deltas.append((i, n)))
    assert (0, "write_file") in deltas
    assert seen and all(seen)
