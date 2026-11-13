"""The agent keeps working instead of pausing mid-task."""
from conftest import tc
from zeusai import agent
from zeusai.modes import Session


def test_paused_reply_is_nudged_to_continue(sandbox, fake_model):
    sent = fake_model({"content": "I found the bug. Shall I continue and fix it?"},
                      {"content": "fixed and verified"})
    m = [{"role": "user", "content": "fix the bug"}]
    out = agent.run_loop(m, session=Session(mode="auto"))
    assert out == "fixed and verified"
    assert len(sent) == 2
    assert agent._CONTINUE_NUDGE in [x["content"] for x in m if x["role"] == "user"]


def test_a_normal_signoff_does_not_restart_the_agent(sandbox, fake_model):
    sent = fake_model({"content": "Done. Let me know if you have questions."})
    out = agent.run_loop([{"role": "user", "content": "x"}], session=Session(mode="auto"))
    assert out.startswith("Done.")
    assert len(sent) == 1


def test_pending_todos_keep_the_agent_working(sandbox, fake_model):
    sent = fake_model({"content": "Step one done."}, {"content": "still working"},
                      {"content": "all steps done"})
    s = Session(mode="auto", todos=[{"content": "step two", "status": "pending"}])
    out = agent.run_loop([{"role": "user", "content": "do both"}], session=s)
    assert out == "all steps done"
    assert len(sent) == 3  # nudged twice, then handed back


def test_completed_todos_do_not_nudge(sandbox, fake_model):
    sent = fake_model({"content": "all done"})
    s = Session(mode="auto", todos=[{"content": "step", "status": "completed"}])
    assert agent.run_loop([{"role": "user", "content": "x"}], session=s) == "all done"
    assert len(sent) == 1


def test_auto_continue_can_be_disabled(sandbox, fake_model):
    sent = fake_model({"content": "Should I proceed?"})
    s = Session(mode="auto", auto_continue=False)
    assert agent.run_loop([{"role": "user", "content": "x"}], session=s) == "Should I proceed?"
    assert len(sent) == 1


def test_plan_and_research_modes_are_never_nudged(sandbox, fake_model):
    sent = fake_model(*([{"content": "Shall I continue?"}] * 2))
    for mode in ("plan", "research"):
        before = len(sent)
        s = Session(mode=mode, todos=[{"content": "step", "status": "pending"}])
        assert agent.run_loop([{"role": "user", "content": "x"}], session=s) == "Shall I continue?"
        assert len(sent) == before + 1


def test_mid_task_narration_is_nudged(sandbox, fake_model):
    # The model says what it is about to do and then stops: keep it working.
    sent = fake_model({"content": "Now I'll update the parser."}, {"content": "updated"})
    assert agent.run_loop([{"role": "user", "content": "x"}], session=Session(mode="auto")) == "updated"
    assert len(sent) == 2


def test_an_empty_reply_is_nudged(sandbox, fake_model):
    sent = fake_model({"content": ""}, {"content": "done now"})
    assert agent.run_loop([{"role": "user", "content": "x"}], session=Session(mode="auto")) == "done now"
    assert len(sent) == 2


def test_nudges_are_bounded(sandbox, fake_model):
    # A model that never stops asking is nudged at most three times, then handed back.
    sent = fake_model(*([{"content": "Shall I continue?"}] * 5))
    agent.run_loop([{"role": "user", "content": "x"}], session=Session(mode="auto"))
    assert len(sent) == 4  # first answer + 3 nudges


def test_tool_calls_reset_the_nudge_budget(sandbox, fake_model):
    (sandbox / "a.py").write_text("")
    fake_model({"content": "Shall I continue?"},
               {"tool_calls": [tc("1", "glob_files", {"pattern": "*.py"})]},
               {"content": "Shall I continue?"},
               {"content": "done"})
    m = [{"role": "user", "content": "x"}]
    assert agent.run_loop(m, session=Session(mode="auto")) == "done"
    assert sum(1 for x in m if x["role"] == "user" and x["content"] == agent._CONTINUE_NUDGE) == 2
