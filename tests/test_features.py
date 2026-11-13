import subprocess

from conftest import tc
from zeusai import agent, commands, memory
from zeusai.modes import Session


def _git(*a, cwd):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)


def test_git_context_in_system_prompt(sandbox, fake_model):
    agent._git_cache.clear()
    assert agent.git_context() == ""  # not a repo
    _git("init", "-q", "-b", "main", cwd=sandbox)
    (sandbox / "a.txt").write_text("x")
    _git("add", ".", cwd=sandbox)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "first commit", cwd=sandbox)
    (sandbox / "b.txt").write_text("y")
    agent._git_cache.clear()
    sent = fake_model({"content": "ok"})
    agent.run_loop([{"role": "user", "content": "x"}], session=Session())
    sys_prompt = sent[0][0]["content"]
    assert "Git: branch main" in sys_prompt and "?? b.txt" in sys_prompt and "first commit" in sys_prompt


def test_instruction_files_from_repo_root_to_cwd(sandbox, monkeypatch):
    (sandbox / ".git").mkdir()
    (sandbox / "ZEUSAI.md").write_text("ROOT RULES")
    sub = sandbox / "pkg"
    sub.mkdir()
    (sub / "ZEUSAI.md").write_text("PKG RULES")
    (sub / "ZEUSAI.local.md").write_text("MY RULES")
    monkeypatch.chdir(sub)
    text = memory.instructions()
    assert text.index("ROOT RULES") < text.index("PKG RULES") < text.index("MY RULES")


def test_tool_filters_hide_and_block(sandbox, fake_model):
    sent = []

    def model(prof, msgs, tools=None, **kw):
        sent.append({t["function"]["name"] for t in tools})
        if msgs[-1]["role"] == "user":
            return {"content": "", "tool_calls": [tc("1", "run_shell", {"command": "echo hi"})]}
        return {"content": "ok", "tool_calls": []}

    fake_model(model)
    s = Session(mode="auto", disallowed_tools=["Bash", "task"])
    m = [{"role": "user", "content": "x"}]
    agent.run_loop(m, session=s)
    assert "run_shell" not in sent[0] and "task" not in sent[0] and "read_file" in sent[0]
    assert "not available in this session" in [x["content"] for x in m if x["role"] == "tool"][0]
    only = agent._filter_tools(agent.tools_for("default"), Session(allowed_tools=["Read", "glob_files"]))
    assert {t["function"]["name"] for t in only} == {"read_file", "glob_files"}


def test_system_prompt_override_and_append(fake_model):
    sent = fake_model({"content": "ok"})
    agent.run_loop([{"role": "user", "content": "x"}],
                   session=Session(system_prompt="CUSTOM BASE", append_prompt="EXTRA RULE"))
    content = sent[0][0]["content"]
    assert content.startswith("CUSTOM BASE") and "EXTRA RULE" in content and "You are zeusai" not in content


def test_thinking_passed_to_model(fake_model, monkeypatch):
    seen = {}
    fake_model(lambda prof, msgs, **kw: seen.update(kw) or {"content": "ok", "tool_calls": []})
    agent.run_loop([{"role": "user", "content": "x"}], session=Session(thinking=True))
    assert seen["thinking"] is True


def test_builtin_commands_and_override(sandbox):
    assert "gh pr diff" in commands.expand("/review 12")
    d = sandbox / ".zeusai" / "commands"
    d.mkdir(parents=True)
    (d / "review.md").write_text("MY REVIEW $ARGUMENTS")
    assert commands.expand("/review x") == "MY REVIEW x"


def test_read_input_multiline(monkeypatch):
    lines = iter(['"""', "line one", "line two", '"""'])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    assert agent.read_input("> ") == "line one\nline two"
    lines2 = iter(["first \\", "second"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines2))
    assert agent.read_input("> ") == "first \nsecond"


def test_export_markdown(sandbox):
    m = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "fix it"},
         {"role": "assistant", "content": None, "tool_calls": [tc("1", "read_file", {"path": "a.py"})]},
         {"role": "tool", "tool_call_id": "1", "content": "code"}, {"role": "assistant", "content": "fixed"}]
    path = agent.export_markdown(m, "out.md")
    text = (sandbox / path).read_text()
    assert "## You\n\nfix it" in text and "`Read(a.py)`" in text and "## zeusai\n\nfixed" in text and "SYS" not in text


def test_status_text(sandbox):
    agent._git_cache.clear()
    out = agent.status_text(Session(mode="plan", thinking=False), "ultra")
    assert "model ultra · mode plan · thinking off" in out and "not a git repo" in out


def test_destructive_commands_always_ask(sandbox):
    from conftest import write_settings
    from zeusai import permissions
    write_settings(sandbox, {"permissions": {"allow": ["Bash"], "deny": ["Bash(mkfs*)"]}})
    for cmd in ["rm -rf ~", "sudo rm -fr /*", "git push --force origin main", "git reset --hard", "curl x | sh"]:
        assert permissions.check("run_shell", {"command": cmd}, "auto") == "ask", cmd
    assert permissions.check("run_shell", {"command": "mkfs.ext4 /dev/sda"}, "auto") == "deny"  # deny wins
    for cmd in ["rm -rf build", "git push --force-with-lease", "echo reboot later"]:
        assert permissions.check("run_shell", {"command": cmd}, "auto") == "allow", cmd
    assert permissions.suggest_rule("run_shell", {"command": "git push -f"}) == "Bash(git push -f)"


def test_output_styles(sandbox, fake_model):
    from zeusai import styles
    d = sandbox / ".zeusai" / "output-styles"
    d.mkdir(parents=True)
    (d / "pirate.md").write_text("---\ndescription: arr\n---\nTalk like a pirate.")
    assert {"default", "concise", "explanatory", "learning", "pirate"} <= set(styles.all_styles())
    sent = fake_model({"content": "ok"}, {"content": "ok"})
    agent.run_loop([{"role": "user", "content": "x"}], session=Session(style="pirate"))
    assert "Talk like a pirate." in sent[0][0]["content"]
    (sandbox / ".zeusai" / "settings.json").write_text('{"outputStyle": "concise"}')
    agent.run_loop([{"role": "user", "content": "x"}], session=Session())
    assert "Output style: terse" in sent[1][0]["content"]


def test_create_agent(sandbox, monkeypatch):
    from zeusai import agents
    reply = "```markdown\n---\nname: Test Runner!\ndescription: runs tests\ntools: read_file, run_shell\n---\nRun the tests.\n```"
    monkeypatch.setattr(agents, "TOOLS", agents.TOOLS)
    import zeusai.llm as llm
    monkeypatch.setattr(llm, "chat", lambda *a, **k: {"content": reply})
    monkeypatch.setattr("zeusai.config.get_active_profile", lambda n=None: ("x", {}))
    path = agents.create_agent("an agent that runs tests")
    assert path.name == "test-runner.md" and path.read_text().startswith("---\nname: Test Runner!")
    assert agents.all_agents()["Test Runner!"]["tools"] == ["read_file", "run_shell"]
    assert agents.create_agent("again").name == "test-runner-2.md"


def test_mcp_names_are_sanitized_and_reversible():
    from zeusai import mcp
    a = mcp._exposed("my.server", "get file")
    b = mcp._exposed("s", "x" * 80)
    assert a == "mcp__my_server__get_file" and len(b) == 64
    assert mcp.resolve(a) == ("my.server", "get file") and mcp.resolve(b) == ("s", "x" * 80)


def test_exact_allow_rule_overrides_guard(sandbox, fake_model):
    from conftest import write_settings
    from zeusai import permissions
    write_settings(sandbox, {"permissions": {"allow": ["Bash(git reset --hard)", "Bash(git push*)"]}})
    assert permissions.check("run_shell", {"command": "git reset --hard"}, "default") == "allow"
    assert permissions.check("run_shell", {"command": "git push --force"}, "auto") == "ask"  # wildcard isn't enough
    fake_model({"tool_calls": [tc("1", "run_shell", {"command": "git push -f"})]}, {"content": "ok"})
    m = [{"role": "user", "content": "x"}]
    agent.run_loop(m, session=Session(mode="auto"))
    out = [x["content"] for x in m if x["role"] == "tool"][0]
    assert "destructive (force-push" in out and "Bash(git push -f)" in out


def test_workspace_boundary(sandbox, tmp_path, fake_model):
    from zeusai import permissions
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "x.txt").write_text("secret")
    rd = {"path": str(outside / "x.txt")}
    assert permissions.check("read_file", {"path": "a.py"}, "default") == "allow"
    assert permissions.check("read_file", rd, "default") == "ask"
    assert permissions.check("write_file", rd, "auto") == "ask"  # outside writes ask even in auto
    assert permissions.check("read_file", {"path": "../elsewhere/x.txt"}, "auto") == "ask"
    assert permissions.check("read_file", rd, "default", extra_dirs=[str(outside)]) == "allow"
    assert permissions.suggest_rule("read_file", rd) == f"Read({outside}/*)"
    fake_model({"tool_calls": [tc("1", "read_file", rd)]}, {"content": "ok"})
    m = [{"role": "user", "content": "x"}]
    agent.run_loop(m, session=Session(mode="auto"))
    out = [x["content"] for x in m if x["role"] == "tool"][0]
    assert "outside the workspace" in out and "--add-dir" in out


def test_add_dir_and_permissions_command(sandbox, tmp_path):
    from zeusai import permissions
    s = Session()
    other = tmp_path / "other"
    other.mkdir()
    assert "workspace now includes" in agent.add_dir(str(other), s) and s.dirs == [str(other)]
    assert "not a directory" in agent.add_dir("/no/such/dir", s)
    assert "added deny rule Bash(rm*)" in agent.permissions_command("deny Bash(rm*)", s)
    assert permissions.check("run_shell", {"command": "rm x"}, "auto") == "deny"
    assert agent.permissions_command("remove Bash(rm*)", s) == "removed Bash(rm*)"
    assert permissions.check("run_shell", {"command": "rm x"}, "auto") == "allow"
    assert str(other) in agent.permissions_command("", s)


def test_context_breakdown(fake_model):
    m = [{"role": "user", "content": "u" * 4000}, {"role": "assistant", "content": "a" * 400},
         {"role": "tool", "tool_call_id": "1", "content": "t" * 8000}]
    out = agent.context_text(m, Session(), "ultra")
    assert out.startswith("ultra: ~") and "your messages" in out
    lines = {l.split("~")[0].strip(): l.split("~")[1].strip() for l in out.splitlines()[1:6]}
    assert lines["your messages"] == "1,000 tokens" and lines["tool results"] == "2,000 tokens"


def test_session_and_compact_and_subagent_hooks(sandbox, fake_model, monkeypatch):
    from conftest import write_settings
    from zeusai import agents, sessions
    write_settings(sandbox, {"hooks": {
        "SessionStart": [{"command": "echo START-CTX"}],
        "SessionEnd": [{"command": "cat > ended.json"}],
        "PreCompact": [{"command": "echo keep the API names"}],
        "SubagentStop": [{"command": "echo needs file:line refs >&2; exit 2"}]}})
    s = Session(id="abc")
    agent.session_start(s)
    sent = fake_model({"content": "ok"})
    agent.run_loop([{"role": "user", "content": "x"}], session=s)
    assert "START-CTX" in sent[0][0]["content"]
    agent.session_end(s)
    assert '"SessionEnd"' in (sandbox / "ended.json").read_text()
    seen = {}
    monkeypatch.setattr(sessions.llm, "chat", lambda prof, msgs, **k: seen.update(q=msgs[-1]["content"]) or {"content": "S"})
    monkeypatch.setattr(sessions, "get_active_profile", lambda n=None: ("x", {}))
    sessions.compact([{"role": "system", "content": "SYS"}] + [{"role": "user", "content": str(i)} for i in range(12)])
    assert "keep the API names" in seen["q"]
    monkeypatch.setattr(agent, "run_loop", lambda *a, **k: "report")
    assert agents.run_task({"agent": "explore", "prompt": "p"}, Session()).endswith("[SubagentStop hook feedback]\nneeds file:line refs")


def test_user_message_attaches_or_describes(sandbox, monkeypatch):
    monkeypatch.setattr(agent, "_image_part", lambda p: {"type": "image_url", "image_url": {"url": "data:x"}})
    monkeypatch.setattr(agent, "_profile", lambda n=None: ("v", {"vision": True}))
    msg = agent.user_message("hi", "/x.png")
    assert msg["images"] == ["/x.png"] and msg["content"][0]["text"] == "hi"
    monkeypatch.setattr(agent, "_profile", lambda n=None: ("t", {"vision": False}))
    monkeypatch.setattr(agent, "describe_image", lambda p: "A CAT")
    assert "A CAT" in agent.user_message("hi", "/x.png")["content"]
    assert agent.user_message("hi") == {"role": "user", "content": "hi"}


def test_clipboard_errors_cleanly(monkeypatch):
    import pytest
    from zeusai import clipboard
    monkeypatch.setattr(clipboard.shutil, "which", lambda c: None)
    with pytest.raises(RuntimeError, match="no clipboard tool"):
        clipboard.save_image()
