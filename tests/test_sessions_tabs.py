"""Session persistence, resume metadata, tab titles, and TUI session tabs."""
import asyncio

from zeusai import agent, sessions


def test_save_keeps_mode_thinking_profile_and_title(sandbox):
    sid = sessions.new_id()
    msgs = [{"role": "user", "content": "fix the parser bug"}]
    sessions.save(sid, msgs, [], mode="executor", thinking=False, profile="ultra")
    saved = sessions.load(sid)
    assert saved["mode"] == "executor"
    assert saved["thinking"] is False
    assert saved["profile"] == "ultra"
    assert sessions.tab_title(msgs) == "fix the parser bug"
    assert len(sessions.tab_title([{"role": "user", "content": "x" * 60}])) <= 24


def test_nudges_are_not_saved_or_replayed(sandbox):
    sid = sessions.new_id()
    msgs = [{"role": "user", "content": "hi"},
            {"role": "user", "content": agent._CONTINUE_NUDGE, "nudge": True},
            {"role": "assistant", "content": "done"}]
    sessions.save(sid, msgs, [])
    saved = sessions.load(sid)
    assert all(not m.get("nudge") for m in saved["messages"])
    assert all(text != agent._CONTINUE_NUDGE for _, text in agent.history_entries(msgs))


def _patch_tui(monkeypatch):
    from zeusai import llm, tui
    monkeypatch.setattr(llm, "prewarm", lambda *a, **k: None)
    monkeypatch.setattr(tui, "resolve_auto", lambda a: "laguna")
    monkeypatch.setattr(tui, "get_active_profile", lambda a: ("laguna", {"vision": False}))
    monkeypatch.setattr(agent, "_profile", lambda n=None: ("laguna", {"vision": False}))
    return tui


def test_tui_opens_and_switches_session_tabs(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)

    async def run():
        app = tui.ZeusaiApp(agent_mode="fast")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            tabs = app.query_one("#session_tabs")
            assert len(tabs.query(tui.Tab)) == 1
            app.action_new_session()
            await pilot.pause()
            assert len(tabs.query(tui.Tab)) == 2
            assert app.active_tab == 1
            tabs.active = "tab-0"
            await pilot.pause()
            assert app.active_tab == 0
            assert app.session.mode == "fast"

    asyncio.run(run())


def test_tui_resume_loads_history_and_mode(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    sid = sessions.new_id()
    sessions.save(sid, [{"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"}],
                  [], mode="thinking", thinking=True, profile="ultra")

    async def run():
        app = tui.ZeusaiApp(resume=sid)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.session.id == sid
            assert app.session.mode == "thinking"
            assert app.session.thinking is True
            assert any(m.get("content") == "hello" for m in app.agent_messages)

    asyncio.run(run())
