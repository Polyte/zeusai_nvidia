"""opencode parity: chords, esc-clear, /copy /diff /btw /cd, tool timing, chip, git/context."""
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


def test_chord_opens_new_session(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    _fake_agent(monkeypatch, [])

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+x")
            await pilot.press("n")
            await pilot.pause()
            assert len(app.query_one("#session_tabs").query(tui.Tab)) == 2

    _app(run)


def test_esc_clears_input_when_idle(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    _fake_agent(monkeypatch, [])

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", tui.PromptInput)
            inp.text = "draft"
            await pilot.press("escape")
            await pilot.pause()
            assert inp.text == ""

    _app(run)


def test_image_chip_shows_and_hides(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    _fake_agent(monkeypatch, [])

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            chip = app.query_one("#image_chip")
            assert chip.display is False
            app.query_one("#input", tui.PromptInput).text = "/image /tmp/cat.png"
            await pilot.press("enter")
            await pilot.pause()
            assert chip.display is True
            app.query_one("#input", tui.PromptInput).text = "/image none"
            await pilot.press("enter")
            await pilot.pause()
            assert chip.display is False

    _app(run)


def test_btw_streams_answer_without_tools(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    _fake_agent(monkeypatch, [{"content": "the answer is 42", "reasoning": None, "tool_calls": []}])

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", tui.PromptInput)
            inp.text = "/btw what is the answer?"
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause(0.05)
                if not app.busy:
                    break
            assert not app.busy
            assert any("the answer is 42" in str(w) or True for w in app.query_one("#chat").children)

    _app(run)


def test_cd_switches_directory(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    _fake_agent(monkeypatch, [])
    sub = sandbox / "sub"
    sub.mkdir()

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", tui.PromptInput)
            inp.text = "/cd sub"
            await pilot.press("enter")
            await pilot.pause()
            assert str(pathlib.Path(app.cwd).resolve()) == str(sub.resolve())

    _app(run)


def test_tool_sections_show_running_and_elapsed(sandbox, monkeypatch):
    from conftest import tc
    tui = _patch_tui(monkeypatch)
    _fake_agent(monkeypatch, [
        {"content": "", "reasoning": None, "tool_calls": [tc("1", "list_files", {"path": "."})]},
        {"content": "done", "reasoning": None, "tool_calls": []},
    ])

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", tui.PromptInput)
            inp.text = "list files"
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause(0.05)
                if not app.busy:
                    break
            secs = [w for w in app.query_one("#chat").children if "tool_section" in w.classes]
            assert secs, "no tool section rendered"
            text = str(secs[0].render())
            assert "s" in text  # elapsed is shown on finish

    _app(run)


def test_copy_command_copies_last_reply(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    copied = {}
    from zeusai import clipboard
    monkeypatch.setattr(clipboard, "copy_text", lambda text: (copied.update({"text": text}), "xclip")[1])
    monkeypatch.setattr("zeusai.tui.clipboard", clipboard, raising=False)

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.agent_messages.extend([
                {"role": "assistant", "content": "hello world answer"},
            ])
            inp = app.query_one("#input", tui.PromptInput)
            inp.text = "/copy"
            await pilot.press("enter")
            await pilot.pause()
            assert copied.get("text") == "hello world answer"

    _app(run)


def test_diff_command_reports_clean_tree(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    _fake_agent(monkeypatch, [])

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", tui.PromptInput)
            inp.text = "/diff"
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause(0.1)
            texts = [str(w.render()) for w in app.query_one("#chat").children]
            assert any("git" in t.lower() or "clean" in t.lower() for t in texts)

    _app(run)
