import base64

from conftest import tc, write_settings
from zeusai import agent
from zeusai.modes import Session

def tool_results(m):
    return [x["content"] for x in m if x["role"] == "tool"]


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def test_loop_runs_tools_and_tracks_todos(sandbox, fake_model):
    (sandbox / "a.py").write_text("")
    fake_model({"tool_calls": [tc("1", "todo_write", {"todos": [{"content": "find", "status": "in_progress"}]}),
                               tc("2", "glob_files", {"pattern": "*.py"})]},
               {"content": "done"})
    s = Session(mode="auto", auto_continue=False)
    m = [{"role": "user", "content": "go"}]
    assert agent.run_loop(m, session=s) == "done"
    assert s.todos == [{"content": "find", "status": "in_progress"}]
    assert [x["content"] for x in m if x["role"] == "tool"][1] == "a.py"


def test_todos_survive_in_system_prompt(fake_model):
    sent = fake_model({"content": "ok"})
    s = Session(todos=[{"content": "step", "status": "pending"}], auto_continue=False)
    agent.run_loop([{"role": "user", "content": "x"}], session=s)
    assert "☐ step" in sent[0][0]["content"]


def test_compaction_summary_folded_into_system(fake_model):
    sent = fake_model({"content": "ok"})
    m = [{"role": "system", "content": agent.SYSTEM_PROMPT},
         {"role": "system", "summary": True, "content": "EARLIER SUMMARY"},
         {"role": "user", "content": "x"}]
    agent.run_loop(m, session=Session())
    assert "EARLIER SUMMARY" in sent[0][0]["content"]
    assert all(not x.get("summary") for x in sent[0]) and sum(x["role"] == "system" for x in sent[0]) == 1


def test_default_mode_denies_without_a_user(sandbox, fake_model):
    fake_model({"tool_calls": [tc("1", "run_shell", {"command": "echo hi"})]}, {"content": "ok"})
    m = [{"role": "user", "content": "x"}]
    agent.run_loop(m, session=Session(mode="default"))
    assert tool_results(m)[0].startswith("Permission denied")


def test_plan_mode_blocks_writes(sandbox, fake_model):
    fake_model({"tool_calls": [tc("1", "write_file", {"path": "x.txt", "content": "y"})]}, {"content": "plan"})
    m = [{"role": "user", "content": "x"}]
    agent.run_loop(m, session=Session(mode="plan"))
    assert "not allowed in plan mode" in tool_results(m)[0] and not (sandbox / "x.txt").exists()


def test_parallel_subagents_including_custom(sandbox, fake_model):
    (sandbox / "a.py").write_text("")
    d = sandbox / ".zeusai" / "agents"
    d.mkdir(parents=True)
    (d / "counter.md").write_text("---\nname: counter\ndescription: counts\ntools: glob_files\n---\nCount files.\n")

    def model(prof, msgs, tools=None, **kw):
        names = {t["function"]["name"] for t in tools}
        last = msgs[-1]
        if "task" not in names:  # a subagent
            assert "write_file" not in names  # explore/counter are read-only
            if last["role"] == "user":
                return {"content": "", "tool_calls": [tc("s", "glob_files", {"pattern": "*.py"})]}
            return {"content": "found " + last["content"], "tool_calls": []}
        if last["role"] == "user":
            return {"content": "", "tool_calls": [
                tc("1", "task", {"agent": "explore", "description": "find", "prompt": "find py"}),
                tc("2", "task", {"agent": "counter", "description": "count", "prompt": "count"})]}
        return {"content": "parent done", "tool_calls": []}

    fake_model(model)
    m = [{"role": "user", "content": "go"}]
    assert agent.run_loop(m, session=Session(mode="default")) == "parent done"
    reports = [x["content"] for x in m if x["role"] == "tool"]
    assert reports[0].startswith("found a.py") and "[subagent counter" in reports[1]


def test_hooks_pre_post_prompt_stop(sandbox, fake_model):
    write_settings(sandbox, {"hooks": {
        "PreToolUse": [{"matcher": "Bash", "command": "grep -q forbidden && { echo nope >&2; exit 2; } || exit 0"}],
        "PostToolUse": [{"matcher": "List", "command": "echo checked >&2; exit 2"}],
        "UserPromptSubmit": [{"command": "echo CTX"}],
        "Stop": [{"command": "test -f done.flag || { touch done.flag; echo 'not yet' >&2; exit 2; }"}]}})
    sent = fake_model({"tool_calls": [tc("1", "run_shell", {"command": "echo forbidden"}),
                                      tc("2", "list_files", {})]},
                      {"content": "first"}, {"content": "final"})
    m = [{"role": "user", "content": "go"}]
    assert agent.run_loop(m, session=Session(mode="auto")) == "final"
    assert "<hook-context>\nCTX" in sent[0][1]["content"]
    tools_out = [x["content"] for x in m if x["role"] == "tool"]
    assert tools_out[0] == "Blocked by a PreToolUse hook: nope"
    assert tools_out[1].endswith("[PostToolUse hook feedback]\nchecked")
    assert "not yet" in sent[2][-1]["content"]


def test_prompt_hook_can_block(sandbox, fake_model):
    write_settings(sandbox, {"hooks": {"UserPromptSubmit": [{"command": "echo no secrets >&2; exit 2"}]}})
    fake_model()
    m = [{"role": "user", "content": "x"}]
    assert "blocked" in agent.run_loop(m, session=Session()) and [x["role"] for x in m] == ["system"]


def test_image_shown_once_to_vision_model(sandbox, fake_model, monkeypatch):
    (sandbox / "p.png").write_bytes(PNG)
    sent = fake_model({"tool_calls": [tc("1", "read_file", {"path": "p.png"})]}, {"content": "a pixel"})
    monkeypatch.setattr(agent, "_profile", lambda n=None: ("v", {"vision": True}))
    m = [{"role": "user", "content": "look"}]
    agent.run_loop(m, session=Session(mode="auto"))
    img_msg = sent[1][-1]
    assert img_msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,") and "images" not in img_msg
    assert m[-2]["content"].startswith("Images from read_file:\n[images shown:")  # stored as text afterwards


def test_image_described_for_text_model(sandbox, fake_model, monkeypatch):
    (sandbox / "p.png").write_bytes(PNG)

    def model(prof, msgs, **kw):
        if prof.get("model_id") == "VISION":
            return {"content": "a red pixel"}
        if msgs[-1]["role"] == "user":
            return {"content": "", "tool_calls": [tc("1", "read_file", {"path": "p.png"})]}
        return {"content": "ok", "tool_calls": []}

    fake_model(model)
    monkeypatch.setattr(agent, "_vision_profile", lambda: "super")
    monkeypatch.setattr(agent, "get_active_profile", lambda a: ("super", {"model_id": "VISION"}))
    m = [{"role": "user", "content": "look"}]
    agent.run_loop(m, session=Session(mode="auto"))
    out = tool_results(m)[0]
    assert "described by a vision model" in out and "a red pixel" in out


def test_stale_tool_outputs_trimmed_in_requests_only(fake_model, monkeypatch):
    monkeypatch.setattr(agent, "KEEP_TOOL_RESULTS", 2)
    monkeypatch.setattr(agent, "ELIDE_BATCH", 2)
    big = "x" * 5000
    m = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
    for i in range(6):
        m.append({"role": "assistant", "content": None, "tool_calls": [tc(str(i), "read_file", {"path": f"f{i}"})]})
        m.append({"role": "tool", "tool_call_id": str(i), "content": big})
    out = [x["content"] for x in agent._with_mode(m, "default") if x["role"] == "tool"]
    assert [len(c) < 1000 for c in out] == [True] * 4 + [False] * 2
    assert 'read_file({"path": "f0"})' in out[0]
    assert all(x["content"] == big for x in m if x["role"] == "tool")  # stored history untouched
