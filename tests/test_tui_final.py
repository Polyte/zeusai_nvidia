"""Last-mile opencode parity: queue-on-busy, history, expand, rename, pickers, update check."""
import asyncio
import pathlib

from test_sessions_tabs import _patch_tui


def _app(run):
    asyncio.run(run())


def _fake_agent(monkeypatch, replies):
    from zeusai import llm
    def chat(prof, messages, **kw):
        r = replies.pop(0) if replies else {"content": "done", "reasoning": None, "tool_calls": []}
        if kw.get("on_text") and r.get("content"):
            kw["on_text"](r["content"])
        return {"content": r.get("content", ""), "reasoning": r.get("reasoning"), "tool_calls": r.get("tool_calls", [])}
    monkeypatch.setattr(llm, "chat", chat)


def test_submit_while_busy_queues_and_runs_next(sandbox, monkeypatch):
    import threading
    import time
    tui = _patch_tui(monkeypatch)
    release = threading.Event()
    calls = []
    from zeusai import llm
    def chat(prof, messages, **kw):
        calls.append(1)
        if len(calls) == 1:
            release.wait(5)  # first turn blocks a little so a second prompt can be queued
        return {"content": "done", "reasoning": None, "tool_calls": []}
    monkeypatch.setattr(llm, "chat", chat)

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", tui.PromptInput)
            inp.text = "first"
            await pilot.press("enter")
            await pilot.pause()
            inp.text = "second"
            await pilot.press("enter")   # queued, not dropped
            await pilot.pause()
            assert app.message_queue.has_pending()
            release.set()
            for _ in range(80):
                await pilot.pause(0.05)
                if not app.busy and len(calls) >= 2:
                    break
            assert len(calls) == 2
            assert any(m.get("content") == "second" for m in app.agent_messages)

    _app(run)


def test_prompt_history_up_down(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    _fake_agent(monkeypatch, [])

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", tui.PromptInput)
            app._record_history("alpha")
            app._record_history("beta")
            inp.text = "draft"
            await pilot.press("up")
            await pilot.pause()
            assert inp.text == "beta"
            await pilot.press("up")
            await pilot.pause()
            assert inp.text == "alpha"
            await pilot.press("down")
            await pilot.pause()
            assert inp.text == "beta"
            await pilot.press("down")
            await pilot.pause()
            assert inp.text == "draft"  # back to the unsent draft

    _app(run)


def test_tool_section_click_expands(sandbox, monkeypatch):
    from conftest import tc
    tui = _patch_tui(monkeypatch)
    _fake_agent(monkeypatch, [
        {"content": "", "reasoning": None, "tool_calls": [tc("1", "run_shell", {"command": "seq 1 30"})]},
        {"content": "done", "reasoning": None, "tool_calls": []},
    ])

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", tui.PromptInput)
            inp.text = "run seq"
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause(0.05)
                if not app.busy:
                    break
            sec = next(w for w in app.query_one("#chat").children if "tool_section" in w.classes)
            collapsed = str(sec.render())
            assert "click to expand" in collapsed
            app.on_click(type("_", (), {"widget": sec})())
            await pilot.pause()
            expanded = str(sec.render())
            assert "click to expand" not in expanded
            assert "30" in expanded  # full output now visible

    _app(run)


def test_rename_sets_tab_title_and_persists(sandbox, monkeypatch):
    from zeusai import sessions
    tui = _patch_tui(monkeypatch)
    _fake_agent(monkeypatch, [])

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.agent_messages.append({"role": "user", "content": "do work"})
            app._tab_snapshot()
            inp = app.query_one("#input", tui.PromptInput)
            inp.text = "/rename parser fix"
            await pilot.press("enter")
            await pilot.pause()
            assert app._tab_title() == "parser fix"
            app._palette_save()
            saved = sessions.load(app.session.id)
            assert saved["title"] == "parser fix"

    _app(run)


def test_resume_opens_session_dialog(sandbox, monkeypatch):
    from zeusai import sessions
    tui = _patch_tui(monkeypatch)
    _fake_agent(monkeypatch, [])
    sid = sessions.new_id()
    sessions.save(sid, [{"role": "user", "content": "saved work"}], [])

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", tui.PromptInput)
            inp.text = "/resume"
            await pilot.press("enter")
            await pilot.pause()
            assert type(app.screen).__name__ == "CommandScreen"
            assert len(app.screen.query_one("#palette_list")._options) >= 1
            await pilot.press("escape")
            await pilot.pause()
            assert type(app.screen).__name__ == "Screen"

    _app(run)


def test_update_check_cached_and_silent(sandbox, monkeypatch):
    import json
    import pathlib
    tui = _patch_tui(monkeypatch)
    # fresh cache marking a newer version -> notice should appear without any network
    cache = pathlib.Path.home() / ".zeusai" / "update-cache.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"at": __import__("time").time(), "version": "99.0.0"}))

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            notices = [w for w in app.query_one("#chat").children if "notice" in w.classes]
            assert any("Update available" in str(w.render()) for w in notices)

    _app(run)
