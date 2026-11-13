import json
import time

from zeusai.tools import execute_tool as run


def test_read_before_edit_and_write(sandbox):
    (sandbox / "a.py").write_text("x = 1\n")
    assert "before editing" in run("edit_file", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"})
    assert "hasn't been read" in run("write_file", {"path": "a.py", "content": "y"})
    assert run("write_file", {"path": "new.py", "content": "ok"}).startswith("Wrote")  # new files are fine
    run("read_file", {"path": "a.py"})
    assert run("edit_file", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}).startswith("Edited")


def test_read_line_numbers_offset_limit(sandbox):
    (sandbox / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    out = run("read_file", {"path": "f.txt", "offset": 3, "limit": 2})
    assert out.splitlines() == ["     3\tline3", "     4\tline4", "... (more lines; continue with offset=5)"]


def test_edit_uniqueness_and_replace_all(sandbox):
    (sandbox / "d.py").write_text("a = 1\na = 1\n")
    run("read_file", {"path": "d.py"})
    assert "appears 2 times" in run("edit_file", {"path": "d.py", "old_string": "a = 1", "new_string": "b"})
    run("edit_file", {"path": "d.py", "old_string": "a = 1", "new_string": "b", "replace_all": True})
    assert (sandbox / "d.py").read_text() == "b\nb\n"


def test_multi_edit_is_atomic(sandbox):
    (sandbox / "m.py").write_text("one\ntwo\n")
    run("read_file", {"path": "m.py"})
    bad = run("multi_edit", {"path": "m.py", "edits": [{"old_string": "one", "new_string": "1"},
                                                       {"old_string": "nope", "new_string": "x"}]})
    assert "Error in edit 2" in bad and (sandbox / "m.py").read_text() == "one\ntwo\n"
    run("multi_edit", {"path": "m.py", "edits": [{"old_string": "one", "new_string": "1"},
                                                 {"old_string": "two", "new_string": "2"}]})
    assert (sandbox / "m.py").read_text() == "1\n2\n"


def test_glob_skips_heavy_dirs(sandbox):
    (sandbox / "src").mkdir()
    (sandbox / "src" / "app.py").write_text("")
    (sandbox / "node_modules").mkdir()
    (sandbox / "node_modules" / "x.py").write_text("")
    assert run("glob_files", {"pattern": "**/*.py"}) == "src/app.py"


def test_shell_timeout_and_background(sandbox):
    assert "timed out after 1s" in run("run_shell", {"command": "sleep 3", "timeout": 1})
    started = run("run_shell", {"command": "echo hello; sleep 0.2", "run_in_background": True})
    sid = started.split()[3]
    seen = ""
    for _ in range(50):
        out = run("shell_output", {"shell_id": sid})  # returns only output new since the last poll
        seen += out
        if "exited" in out:
            break
        time.sleep(0.1)
    assert "exited with code 0" in out and "hello" in seen
    assert "no background shell" in run("shell_kill", {"shell_id": "bash_99"})


def test_notebook_read_and_edit(sandbox):
    nb = {"cells": [{"cell_type": "code", "source": ["print(1)\n"], "metadata": {},
                     "outputs": [{"output_type": "stream", "text": ["1\n"]}], "execution_count": 1}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (sandbox / "n.ipynb").write_text(json.dumps(nb))
    out = run("read_file", {"path": "n.ipynb"})
    assert "--- cell 0 [code] ---" in out and "[output]\n1" in out
    assert run("notebook_edit", {"path": "n.ipynb", "cell_index": 1, "edit_mode": "insert",
                                 "cell_type": "markdown", "new_source": "# Hi"}).startswith("Inserted cell 1")
    run("notebook_edit", {"path": "n.ipynb", "cell_index": 0, "new_source": "print(2)"})
    cells = json.loads((sandbox / "n.ipynb").read_text())["cells"]
    assert [c["cell_type"] for c in cells] == ["code", "markdown"]
    assert cells[0]["source"] == ["print(2)"] and cells[0]["outputs"] == []
    assert "out of range" in run("notebook_edit", {"path": "n.ipynb", "cell_index": 9})


def test_image_marker(sandbox):
    (sandbox / "p.png").write_bytes(b"\x89PNG fake")
    assert run("read_file", {"path": "p.png"}).startswith("[image] ")


def test_glob_patterns_and_explicit_heavy_dir(sandbox):
    from zeusai.tools import _glob_regex as rx
    for pat, path, want in [("**/*.py", "a/b/c.py", True), ("**/*.py", "c.py", True), ("*.py", "a/c.py", False),
                            ("src/*.ts", "src/a/x.ts", False), ("*.{js,ts}", "a.ts", True),
                            ("file?.txt", "file1.txt", True), ("[!ab].py", "a.py", False)]:
        assert bool(rx(pat).match(path)) == want, (pat, path)
    (sandbox / "node_modules" / "pkg").mkdir(parents=True)
    (sandbox / "node_modules" / "pkg" / "index.js").write_text("")
    assert run("glob_files", {"pattern": "**/*.js"}) == "(no matches)"
    assert run("glob_files", {"pattern": "node_modules/**/*.js"}) == "node_modules/pkg/index.js"
