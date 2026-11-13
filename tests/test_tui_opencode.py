"""opencode-style TUI: multiline composer, suggest popups, command palette, stats."""
import asyncio

from test_sessions_tabs import _patch_tui


def _app(run):
    asyncio.run(run())


def test_enter_submits_and_shift_enter_newlines(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    from zeusai import llm
    monkeypatch.setattr(llm, "chat", lambda prof, messages, **kw: {"content": "ok", "reasoning": None, "tool_calls": []})

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", tui.PromptInput)
            inp.text = "first line"
            inp.move_cursor((0, len("first line")))  # like a real user: cursor at the end
            await pilot.press("shift+enter")
            await pilot.pause()
            assert inp.text.startswith("first line") and "\n" in inp.text  # newline in box, not sent
            assert not app.busy
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause(0.05)
                if not app.busy:
                    break
            assert inp.text == ""  # submitted and cleared
            assert app.agent_messages and app.agent_messages[-1]["role"] == "assistant"

    _app(run)


def test_slash_popup_completions(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    from zeusai import llm
    monkeypatch.setattr(llm, "chat", lambda prof, messages, **kw: {"content": "ok", "reasoning": None, "tool_calls": []})

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", tui.PromptInput)
            for ch in "/mod":
                await pilot.press(ch)
            await pilot.pause()
            assert app.query_one("#suggest").display is True
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert inp.text.startswith("/model")
            assert app.query_one("#suggest").display is False
            assert app.focused is inp

    _app(run)


def test_mention_popup_completions(sandbox, monkeypatch):
    (sandbox / "README.md").write_text("# hi")
    tui = _patch_tui(monkeypatch)
    from zeusai import llm
    monkeypatch.setattr(llm, "chat", lambda prof, messages, **kw: {"content": "ok", "reasoning": None, "tool_calls": []})

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.query_one("#input", tui.PromptInput)
            for ch in "@READ":
                await pilot.press(ch)
            await pilot.pause()
            assert app.query_one("#suggest").display is True
            await pilot.press("enter")
            await pilot.pause()
            assert "README.md" in inp.text

    _app(run)


def test_palette_opens_filters_and_dismisses(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    from zeusai import llm
    monkeypatch.setattr(llm, "chat", lambda prof, messages, **kw: {"content": "ok", "reasoning": None, "tool_calls": []})

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+p")
            await pilot.pause()
            assert type(app.screen).__name__ == "CommandScreen"
            app.screen.query_one("#palette_input").value = "new session"
            await pilot.pause()
            assert len(app.screen.query_one("#palette_list")._options) == 1
            await pilot.press("enter")
            await pilot.pause()
            assert type(app.screen).__name__ == "Screen"  # dialog closed
            assert len(app.query_one("#session_tabs").query(tui.Tab)) == 2  # new tab created

    _app(run)


def test_palette_model_picker(sandbox, monkeypatch):
    tui = _patch_tui(monkeypatch)
    from zeusai import llm
    monkeypatch.setattr(llm, "chat", lambda prof, messages, **kw: {"content": "ok", "reasoning": None, "tool_calls": []})

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+p")
            await pilot.pause()
            app.screen.query_one("#palette_input").value = "switch model"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert type(app.screen).__name__ == "CommandScreen"  # models screen
            assert len(app.screen.query_one("#palette_list")._options) >= 1
            await pilot.press("escape")
            await pilot.pause()
            assert type(app.screen).__name__ == "Screen"

    _app(run)


def test_boot_shows_notice_stats_and_bottom_bar(sandbox, monkeypatch):
    # fresh HOME has no API keys -> opencode-style auth notice; stats/bottom bars rendered
    tui = _patch_tui(monkeypatch)

    async def run():
        app = tui.ZeusaiApp(agent_mode="auto")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            notices = [w for w in app.query_one("#chat").children if "notice" in w.classes]
            assert notices, "missing API keys should show a notice banner"
            assert app.query_one("#stats_left").render() is not None
            assert "MCP" in str(app.query_one("#bottom_left").render())
            assert app.query_one("#meta_right").render() is not None

    _app(run)
