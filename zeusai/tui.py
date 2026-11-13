"""zeusai TUI — full-screen, opencode-style. Launched by `zeusai` / `zeusai tui`."""
from __future__ import annotations

import json
import os
import pathlib
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

from rich.markup import escape
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (Button, Input, Markdown, OptionList,
                             SelectionList, Static, Tab, Tabs, TextArea)
from textual.screen import ModalScreen

from . import llm
from .config import BASE_URL, ensemble_judge, get_default_profile_name, get_profiles, list_profiles, get_active_profile, resolve_auto
from .ensemble import ensemble_ask
from .agent import (run_loop, SYSTEM_PROMPT, ALLOW_ONCE, ALLOW_ALWAYS, DENY, bash_passthrough,
                    commands_exist, info_command, prepare_prompt, export_markdown, status_text,
                    bashes_text, context_text, doctor_text, permissions_command, add_dir, history_entries)
from . import permissions, sessions
from .modes import PLAN_OPTIONS, Session, enhance_prompt, set_mode
from .ui import edit_preview, tool_header, tool_summary
from .ui import FILE_TOOLS, live_file_text
from .ui import MESSAGES, OTHER, busy_line
import random

@dataclass
class QueuedPrompt:
    """A prompt in the message queue."""
    prompt: str
    image_src: Optional[str] = None
    want_tools: bool = True
    enhanced: bool = False
    original_prompt: Optional[str] = None  # Store original for comparison

    def display_text(self) -> str:
        """Text to show in the queue display."""
        prefix = "✨ " if self.enhanced else "📝 "
        tools = "🔧 " if self.want_tools else "💬 "
        img = " 🖼️" if self.image_src else ""
        return f"{prefix}{tools}{self.prompt[:80]}{'...' if len(self.prompt) > 80 else ''}{img}"


class MessageQueue:
    """Manages a queue of prompts to be processed."""
    
    def __init__(self):
        self.queue: List[QueuedPrompt] = []
        self.processing = False
        self.current_index = -1
    
    def add(self, prompt: str, image_src: Optional[str] = None, want_tools: bool = True, enhanced: bool = False, original_prompt: Optional[str] = None):
        """Add a prompt to the queue."""
        qp = QueuedPrompt(prompt=prompt, image_src=image_src, want_tools=want_tools, enhanced=enhanced, original_prompt=original_prompt)
        self.queue.append(qp)
        return len(self.queue) - 1
    
    def remove(self, index: int) -> bool:
        """Remove a prompt at the given index."""
        if 0 <= index < len(self.queue):
            self.queue.pop(index)
            if self.current_index >= index:
                self.current_index -= 1
            return True
        return False
    
    def clear(self):
        """Clear all prompts from the queue."""
        self.queue.clear()
        self.current_index = -1
    
    def get_next(self) -> Optional[QueuedPrompt]:
        """Get the next unprocessed prompt."""
        if self.current_index + 1 < len(self.queue):
            self.current_index += 1
            return self.queue[self.current_index]
        return None
    
    def has_pending(self) -> bool:
        """Check if there are pending prompts."""
        return self.current_index + 1 < len(self.queue)
    
    def get_display_items(self) -> List[str]:
        """Get display items for the queue panel."""
        items = []
        for i, qp in enumerate(self.queue):
            status = "▶ " if i == self.current_index else ("✓ " if i < self.current_index else "  ")
            items.append(f"{status}{i+1}. {qp.display_text()}")
        return items


HELP_TEXT = r"""## zeusai TUI

**Keys:** `Enter` send · `Shift+Enter` new line · `Shift+Tab` cycle mode (default → auto → fast → thinking → executor → plan → research) · `Ctrl+N` new session tab · `Ctrl+P` commands · `Ctrl+E` enhance-prompt · `Ctrl+C` quit
**Queue Keys:** `Ctrl+Q` add to queue * `Ctrl+Shift+Q` process queue * `Ctrl+Alt+Q` clear queue
**Input:** `/` completes slash commands · `@` completes file paths · `!cmd` runs a shell command
**Slash commands (in input):**

| Command | What it does |
|---|---|
| `/help` | this help |
| `/models` | list models, keys, thinking/vision flags |
| `/model <alias>` | single-model mode (laguna/ultra/super/glimmer/paligemma) |
| `/model auto` | default model for the session (ultra: best on the coding eval) |
| `/ensemble` | fan-out to all + judge (slower) |
| `/judge <alias>` | set judge for ensemble |
| `/tools on|off` | enable tool use (coding agent, read/write/run_shell) |
| `/image <path|url>` | attach image for next prompt (vision) |
| `/imagedir` | show image status |
| `/clear` | clear chat and start a new conversation |
| `/todos` | show the agent's todo list |
| `/think [on|off]` | toggle extended reasoning |
| `/export [file.md]` | save the conversation as Markdown |
| `/status` `/bashes` | session overview / background shells |
| `/copy` | copy the last reply to the clipboard |
| `/diff` | show uncommitted changes (git diff) |
| `/btw <text>` | one-shot answer, no tools |
| `/cd <path>` | change working directory |
| `/style [name]` | output style (default, concise, explanatory, learning, custom) |
| `/context` `/doctor` | context-window usage / health checks |
| `/add-dir <path>` | let the agent use another directory |
| `/permissions [allow\|deny\|remove <rule>]` | view or edit rules |
| `/paste` | attach the clipboard image to the next prompt |
| `/agents new <description>` | generate a custom subagent |
| `/review` `/commit` `/pr` | built-in git workflows |
| `/rewind [n]` | undo the last n prompts (restores files and conversation) |
| `/agents` | pick a subagent (dialog) |
| `/mcp` `/mcps` | pick an MCP server (dialog) |
| `!cmd` | run a shell command yourself; output goes into the conversation |
| `@path` | attach a file or directory to your prompt (images too) |
| `Esc` | stop the running agent turn |
| `/compact [focus]` | summarize older agent history to free context |
| `/resume [id]` | open a session picker, or resume one by id |
| `/rename <title>` | rename this session tab |
| `/quit` | quit |
| `/mode [default\|auto\|fast\|thinking\|executor\|plan\|research]` | set agent mode (no arg: show status) |
| `/new` | open a new session in a new tab |
| `/autocontinue [on\|off]` | keep working until the task is done (default on) |
| `/enhance` | toggle enhance-prompt |

**Prompts:**
- `Which is larger, 9.11 or 9.8?` → ensemble vote
- `img path/to.jpg Describe?` or `/image /tmp/sign.jpg` then prompt → vision (super+paligemma)
- `run: list files, create hello.py` → tool agent (needs `/tools on`)
- `Write a limerick about GPU computing` → ultra judge merges creative
"""


class PromptInput(TextArea):
    """Multiline prompt box, opencode-style: Enter sends, Shift+Enter inserts a newline."""

    BINDINGS = [
        Binding("enter", "submit_prompt", "Send", priority=True),
        Binding("shift+enter", "insert_newline", "New line", priority=True),
    ]

    def action_submit_prompt(self):
        self.app.submit_from_input(self.text)

    def action_insert_newline(self):
        self.insert("\n", maintain_selection_offset=False)

    def on_key(self, event):
        """Intercept ctrl+x chord keys (l/n/m) and prompt-history arrows."""
        app = self.app
        chord = getattr(app, "_chord", None)
        if chord:
            if time.monotonic() - chord > 2:
                app._chord = None
            elif event.key in ("l", "n", "m"):
                event.stop()
                event.prevent_default()
                app._run_chord(event.key)
                return
            else:
                app._chord = None
        if event.key == "up" and self.cursor_location[0] == 0:
            if app._history_prev():
                event.stop()
                event.prevent_default()
        elif event.key == "down" and self.cursor_location[0] == len((self.text or "").split("\n")) - 1:
            if app._history_next():
                event.stop()
                event.prevent_default()


# Slash-command completions, opencode-style: command, one-line description.
COMMANDS = [
    ("/help", "Show help"),
    ("/models", "List models and keys"),
    ("/model <alias>", "Switch model (laguna/ultra/super/glimmer/paligemma)"),
    ("/ensemble", "Fan-out to all models + judge"),
    ("/judge <alias>", "Set judge for ensemble"),
    ("/mode <name>", "Set mode (default/auto/fast/thinking/executor/plan/research)"),
    ("/autocontinue on|off", "Keep working until the task is done"),
    ("/tools on|off", "Enable or disable tool use"),
    ("/image <path>", "Attach an image for the next prompt"),
    ("/clear", "Clear the conversation"),
    ("/new", "New session in a new tab"),
    ("/resume", "List or resume a saved session"),
    ("/todos", "Show the todo list"),
    ("/think on|off", "Toggle extended reasoning"),
    ("/enhance", "Toggle enhance-prompt"),
    ("/export", "Export the conversation as Markdown"),
    ("/status", "Session overview"),
    ("/compact", "Compact older history"),
    ("/rewind", "Undo the last prompts (restores files)"),
    ("/permissions", "View or edit permission rules"),
    ("/add-dir <path>", "Allow another directory"),
    ("/style <name>", "Output style"),
    ("/context", "Context window usage"),
    ("/doctor", "Health checks"),
    ("/copy", "Copy the last reply to the clipboard"),
    ("/diff", "Show uncommitted changes (git diff)"),
    ("/btw <text>", "One-shot answer, no tools"),
    ("/rename <title>", "Rename this session tab"),
    ("/cd <path>", "Change working directory"),
    ("/agents", "List subagents"),
    ("/mcp", "MCP servers"),
    ("/hooks", "Hook configuration"),
    ("/config", "Show config paths"),
    ("/cost", "Token usage"),
    ("/quit", "Quit zeusai"),
]


class AskPanel(Vertical):
    """Multiple-choice question from the agent (ask_user tool)."""

    DEFAULT_CSS = """
    AskPanel { height: auto; border: round $accent; padding: 0 1; margin: 1 0; }
    AskPanel OptionList, AskPanel SelectionList { height: auto; max-height: 10; }
    AskPanel Button { margin-top: 1; }
    """

    def __init__(self, question, options, multi, on_done):
        super().__init__()
        self.question, self.options, self.multi, self.on_done = question, options, multi, on_done

    def compose(self) -> ComposeResult:
        yield Static(f"[b]? {escape(self.question)}[/b]")
        if self.multi:
            yield SelectionList(*[(o, i) for i, o in enumerate(self.options)])
            yield Button("Submit", variant="primary")
            yield Static("[dim]space toggles · tab to Submit · or type your own answer below[/dim]")
        else:
            yield OptionList(*(self.options + [OTHER]))
            yield Static("[dim]↑/↓ + enter · or type your own answer below[/dim]")

    def on_mount(self):
        self.query_one(SelectionList if self.multi else OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        rows = self.options + [OTHER]
        if rows[event.option_index] == OTHER:
            inp = self.app.query_one("#input", PromptInput)
            inp.placeholder = "Type your answer and press Enter"
            inp.focus()
        else:
            self.on_done(rows[event.option_index])

    def on_button_pressed(self, event: Button.Pressed):
        picked = self.query_one(SelectionList).selected
        if picked:
            self.on_done("; ".join(self.options[i] for i in sorted(picked)))


class CommandScreen(ModalScreen):
    """opencode-style command palette / model picker dialog (ctrl+p)."""

    DEFAULT_CSS = """
    CommandScreen { align: center middle; }
    #palette { width: 72; height: auto; border: round #2dd4bf; background: $surface; padding: 0 1; }
    #palette_title { padding: 0 1; color: $text-muted; }
    #palette_input { margin: 0 1; height: 3; border: round #2dd4bf; }
    #palette_list { height: auto; max-height: 12; margin: 0 1 1 1; }
    """

    BINDINGS = [Binding("escape", "dismiss", "Close", priority=True)]

    def __init__(self, app, title, items):
        super().__init__()
        self._app = app
        self._title = title
        self._items = items  # [(label, hint, callable)]

    def compose(self) -> ComposeResult:
        with Vertical(id="palette"):
            yield Static(self._title, id="palette_title")
            yield Input(placeholder="Type to filter…", id="palette_input")
            yield OptionList(id="palette_list")

    def on_mount(self):
        self.query_one("#palette_input", Input).focus()
        self._fill("")

    def _fill(self, filt: str):
        opt = self.query_one("#palette_list", OptionList)
        opt.clear_options()
        self._shown = [fn for lbl, hint, fn in self._items
                       if not filt or filt.lower() in lbl.lower()]
        opt.add_options([escape(f"{lbl}  {hint}") for lbl, hint, fn in self._items
                         if not filt or filt.lower() in lbl.lower()])
        try:
            opt.highlighted = 0
        except Exception:
            pass

    @on(Input.Changed, "#palette_input")
    def _filter(self, event: Input.Changed):
        self._fill(event.value or "")

    @on(Input.Submitted, "#palette_input")
    def _pick_first(self, event: Input.Submitted):
        opt = self.query_one("#palette_list", OptionList)
        idx = opt.highlighted or 0
        fn = self._shown[idx] if 0 <= idx < len(self._shown) else None
        self.dismiss()
        if fn is not None:
            fn()

    @on(OptionList.OptionSelected, "#palette_list")
    def _chosen(self, event: OptionList.OptionSelected):
        fn = self._shown[event.option_index] if 0 <= event.option_index < len(self._shown) else None
        self.dismiss()
        if fn is not None:
            fn()


class ZeusaiApp(App):
    """Full-screen zeusai ensemble agent."""

    title = "zeusai"
    sub_title = "terminal coding agent"
    CSS = """
    Screen { layout: vertical; background: $background; color: $text; }
    #session_tabs { height: 3; border-bottom: solid $surface-lighten-1; }
    #session_tabs Tab { padding: 0 1; }
    #session_tabs Underline > .underline--bar { color: #2dd4bf; }
    #chat_area { height: 1fr; }
    #hero { height: 1fr; align: center middle; text-align: center; color: $text-muted; }
    #chat { height: 1fr; padding: 0 2; }
    .user_msg { background: $boost; border: round $surface-lighten-1; padding: 0 1; margin: 1 0; }
    #busy { height: 1; padding: 0 2; color: #2dd4bf; display: none; }
    #status_line { height: 1; padding: 0 2; color: $text-muted; }
    #queue_bar { height: auto; padding: 0 2; color: $text-muted; display: none; }
    #bashes_bar { height: 1; padding: 0 2; color: $text-muted; display: none; }
    #suggest { height: auto; max-height: 12; border: round #2dd4bf; margin: 0 2; display: none; }
    #suggest OptionList { height: auto; max-height: 10; }
    .file_live { height: auto; border: round #2dd4bf; padding: 0 1; margin: 1 0; }
    .thought_live { height: auto; max-height: 12; padding: 0 1; margin: 1 0; color: $text-muted; }
    .tool_section { height: auto; border-left: thick #2dd4bf; padding: 0 1; margin: 1 0; }
    .tool_section.shell { border-left: thick $warning; }
    .notice { height: auto; border: round #2dd4bf; padding: 0 1; margin: 1 0; }
    #input_box { height: auto; border: round #2dd4bf; margin: 0 2; padding: 0 1; }
    #input { height: 3; border: none; padding: 0 1; }
    #image_chip { height: 1; color: $text-muted; display: none; }
    #input_stats { height: 1; color: $text-muted; }
    #stats_left { width: 1fr; }
    #stats_right { width: auto; text-align: right; }
    #input_meta { height: 1; padding: 0 2; color: $text-muted; }
    #cwd_bar { width: 1fr; }
    #meta_right { width: auto; text-align: right; }
    #bottom_bar { height: 1; padding: 0 2; color: $text-muted; }
    #bottom_left { width: 1fr; }
    #bottom_right { width: auto; text-align: right; }
    OptionList { border: none !important; }
    OptionList > .option-list--option-highlighted { background: #134e4a; color: #ffffff; }
    OptionList:focus > .option-list--option-highlighted { background: #0f766e; color: #ffffff; }
    """

    # priority=True: the focused Input otherwise swallows ctrl+e/ctrl+c etc.
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear", "Clear", priority=True),
        Binding("ctrl+t", "tools", "Tools", priority=True),
        Binding("shift+tab", "cycle_mode", "Mode", priority=True),
        Binding("ctrl+n", "new_session", "New session", priority=True),
        Binding("ctrl+e", "enhance_prompt", "Enhance", priority=True),
        Binding("ctrl+x", "chord_start", "Chord", priority=True, show=False),
        Binding("ctrl+q", "add_to_queue", "Queue", priority=True, show=False),
        Binding("ctrl+shift+q", "process_queue", "Process Queue", priority=True, show=False),
        Binding("ctrl+alt+q", "clear_queue", "Clear Queue", priority=True, show=False),
        Binding("escape", "interrupt", "Stop", priority=True),
        Binding("f1", "help", "Help"),
    ]

    def __init__(self, initial_path: str | None = None, judge_alias: str | None = None,
                 model: str | None = None, agent_mode: str = "default", resume: str | None = None):
        super().__init__()
        if initial_path:
            p = pathlib.Path(initial_path).expanduser()
            if p.is_dir():
                os.chdir(p)
        self.agent_mode = agent_mode if agent_mode in permissions.MODES else "default"
        self.resume = resume  # None = fresh, "" = latest here, or a session id/prefix
        self.judge_alias = ensemble_judge(judge_alias)
        # one model answers unless ensemble is explicitly requested (--model ensemble or /ensemble)
        self.mode = model or resolve_auto(get_default_profile_name())
        if self.mode not in ("ensemble",):
            # one model per session: resolve auto once here, not per prompt
            try:
                self.single_alias = resolve_auto(self.mode)
                get_active_profile(self.single_alias)
                self.mode = "single"
            except Exception:
                self.mode = "ensemble"
                self.single_alias = None
        else:
            self.single_alias = None
        self.tools_enabled = True
        self.session = Session(id=sessions.new_id(), mode=self.agent_mode)
        self.agent_messages = []  # tool-agent conversation, kept across prompts
        self.tabs_state = []      # one dict per session tab (id, title, messages, session, ...)
        self.active_tab = 0
        self._switching_tab = False
        self._resumed = None
        self._suggest = None   # active completion popup state (slash / @)
        self._sink = None      # current streaming Sink (for elapsed + tok/s stats)
        self._chord = None     # timestamp when ctrl+x chord was pressed
        self._ph_index = -1    # rotating placeholder example
        self._ph_tick = 0
        self._history: list[str] = []  # submitted prompts (up/down in the composer)
        self._hist_idx = -1    # position in _history while navigating (-1 = current)
        self._hist_saved = ""  # unsent draft preserved while navigating history
        self._renamed_title: str | None = None  # /rename override for the active tab
        self.image_src: str | None = None
        self._busy = False
        self._ask_done = None  # set while an AskPanel waits for an answer
        self.busy_tick = 0
        self.busy_started = 0.0
        self.busy_msg = ""
        self.cwd = str(pathlib.Path.cwd())
        self.cancel = threading.Event()  # Esc stops the running agent turn
        # Message queue
        self.message_queue = MessageQueue()
        self._queue_processing = False
        self._load_resume()

    # ---------- sessions (tabs) ----------
    def _load_resume(self):
        """Load a saved conversation into the first tab (full history, todos, mode)."""
        if self.resume is None:
            return
        saved = sessions.load(self.resume or None)
        if not saved:
            return
        mode = saved.get("mode") if saved.get("mode") in permissions.MODES else self.agent_mode
        self.session = Session(id=saved["id"], mode=mode,
                               thinking=saved.get("thinking"))
        self.agent_messages[:] = saved.get("messages", [])
        self.session.todos[:] = saved.get("todos", [])
        if saved.get("profile"):
            try:
                self.single_alias = resolve_auto(saved["profile"])
                get_active_profile(self.single_alias)
                self.mode = "single"
            except Exception:
                pass
        self._renamed_title = self._detect_renamed(saved)
        self._resumed = saved

    def _detect_renamed(self, saved):
        """Recover a /rename override from a saved session (saved title != auto title)."""
        if not saved.get("title"):
            return None
        try:
            auto = sessions._title(saved.get("messages", []))
        except Exception:
            return None
        return saved["title"] if saved["title"] != auto else None

    def _tab_title(self):
        """The active tab's title: a /rename override, else from the first prompt."""
        return self._renamed_title or sessions.tab_title(self.agent_messages, fallback="New session")

    def _tab_snapshot(self):
        return {"id": self.session.id,
                "title": self._tab_title(),
                "renamed": self._renamed_title,
                "messages": self.agent_messages, "session": self.session,
                "single_alias": self.single_alias, "mode": self.mode}

    def _stash_tab(self):
        if self.tabs_state and self.active_tab < len(self.tabs_state):
            self.tabs_state[self.active_tab] = self._tab_snapshot()

    def _restore_tab(self, index: int):
        state = self.tabs_state[index]
        self.session = state["session"]
        self.agent_messages = state["messages"]
        self.single_alias = state["single_alias"]
        self.mode = state["mode"]
        self._renamed_title = state.get("renamed")
        self.active_tab = index

    def _new_tab_state(self):
        sid = sessions.new_id()
        return {"id": sid, "title": "New session", "messages": [],
                "session": Session(id=sid, mode=self.agent_mode),
                "single_alias": self.single_alias, "mode": self.mode}

    def _open_resumed(self, saved):
        """Open a saved conversation in a new tab, keeping the current one intact."""
        if self._busy:
            self._append_plain("Busy; try again when the current task finishes.")
            return
        self._stash_tab()
        mode = saved.get("mode") if saved.get("mode") in permissions.MODES else self.agent_mode
        session = Session(id=saved["id"], mode=mode, thinking=saved.get("thinking"))
        session.todos[:] = saved.get("todos", [])
        state = {"id": saved["id"], "title": saved.get("title", "session"),
                 "renamed": self._detect_renamed(saved),
                 "messages": list(saved.get("messages", [])), "session": session,
                 "single_alias": self.single_alias, "mode": self.mode}
        self.tabs_state.append(state)
        index = len(self.tabs_state) - 1
        try:
            tabs = self.query_one("#session_tabs", Tabs)
            tabs.add_tab(Tab(f"{index + 1}: {self._renamed_title or sessions.tab_title(state['messages'], fallback='session')}",
                             id=f"tab-{index}"))
            tabs.active = f"tab-{index}"
        except Exception as e:
            self._status(f"tab error: {e}")
        self._restore_tab(index)
        self._clear_log()
        self._log_title()
        self._replay_history()
        self._append_plain(f"Resumed {saved.get('title', '')} ({saved['id']}) in tab {index + 1}.", style="green")

    def _refresh_tab_label(self, index: int | None = None):
        index = self.active_tab if index is None else index
        try:
            tabs = self.query_one("#session_tabs", Tabs)
            tab = tabs.query(Tab)[index]
            title = self._tab_title()
            tab.label = f"{index + 1}: {title}"
        except Exception:
            pass

    def _replay_history(self):
        """Render a resumed conversation in the log, with tool calls as their own sections."""
        for kind, text in history_entries(self.agent_messages):
            text = (text or "").strip()
            if not text:
                continue
            if kind == "you":
                self._append_user(text)
            elif kind == "thought":
                self._append_plain(f"💭 Thought\n{text[-2000:]}", style="dim italic")
            elif kind == "agent":
                self._append_markdown(text[-3000:])
            elif kind == "tool":
                self._append_plain(text, style="dim")
            elif kind == "summary":
                self._append_plain(text[:500], style="dim")

    # ---------- UI layout ----------
    def compose(self) -> ComposeResult:
        yield Tabs(id="session_tabs")
        with Vertical(id="chat_area"):
            yield Static(self._hero_text(), id="hero")
            yield VerticalScroll(id="chat")
        yield Static("", id="busy")
        yield Static("", id="status_line")
        yield Static("", id="queue_bar")
        yield Static("", id="bashes_bar")
        with Vertical(id="suggest"):
            yield OptionList(id="suggest_list")
        with Vertical(id="input_box"):
            yield PromptInput(placeholder="Ask anything…  / for commands, @ for files, ! for shell",
                              id="input", show_line_numbers=False, soft_wrap=True)
            yield Static("", id="image_chip")
            with Horizontal(id="input_stats"):
                yield Static("", id="stats_left")
                yield Static("", id="stats_right")
        with Horizontal(id="input_meta"):
            yield Static("", id="cwd_bar")
            yield Static("ctrl+p commands", id="meta_right")
        with Horizontal(id="bottom_bar"):
            yield Static("", id="bottom_left")
            yield Static("", id="bottom_right")

    @staticmethod
    def _hero_text() -> Text:
        """Empty-state hero, opencode-style: block wordmark + tagline + example prompts.

        Glyphs use the same light half-block vocabulary as opencode's wordmark.
        """
        from zeusai import __version__
        glyphs = {
            "z": ["█▀▀▀", "  █ ", "█▄▄█"],
            "e": ["█▀▀█", "█▀▀▀", "▀▀▀ "],
            "u": ["█  █", "█  █", "█▀▀█"],
            "s": ["█▀▀▀", "▀▀▀█", "█▄▄█"],
            "a": ["█▀▀▀", "█  █", "█  █"],
            "i": ["█",    "█",    "█"],
        }
        t = Text()
        for row in range(3):
            t.append(" ".join(glyphs[c][row] for c in "zeusai") + "\n", style="bold")
        t.append(f"\nzeusai {__version__} · terminal coding agent\n\n", style="dim")
        t.append("Fix the failing tests and run pytest\n", style="dim")
        t.append("enter send · shift+tab mode · ctrl+n new session · /help", style="dim")
        return t

    # ---------- busy bar ----------
    @property
    def busy(self):
        return self._busy

    @busy.setter
    def busy(self, value):
        self._busy = value
        try:
            bar = self.query_one("#busy", Static)
        except Exception:
            return
        if value:
            self.busy_tick, self.busy_started = 0, time.monotonic()
            self.busy_msg = random.choice(MESSAGES)
            bar.display = True
            self._tick_busy()
        else:
            bar.display = False
            self._update_stats()

    def _tick_busy(self):
        if not self._busy:
            # rotate the empty-state placeholder examples slowly (opencode-style)
            self._ph_tick += 1
            if self._ph_tick % 75 == 0:
                try:
                    inp = self.query_one("#input", PromptInput)
                    if not inp.text:
                        examples = ["Fix a TODO in the codebase", "Explain this repo",
                                    "Add input validation to the API", "Fix the failing tests and run pytest"]
                        self._ph_index = (self._ph_index + 1) % len(examples)
                        inp.placeholder = f'Ask anything…  "{examples[self._ph_index]}"'
                except Exception:
                    pass
            return
        if self.busy_tick and self.busy_tick % 40 == 0:
            self.busy_msg = random.choice(MESSAGES)
        self.query_one("#busy", Static).update(
            busy_line(self.busy_tick, self.busy_msg, time.monotonic() - self.busy_started, hint="working"))
        self._update_stats()
        self._update_bashes()
        self.busy_tick += 1

    def _update_bashes(self):
        """opencode-style background-task row (one line when any shells are running)."""
        try:
            from . import tools as _tools
            shells = _tools.background_shells()
            running = [sid for sid, (_out, code) in shells.items() if code is None]
            bar = self.query_one("#bashes_bar", Static)
            if running:
                bar.update(Text(f"◉ {len(running)} background shell(s): {', '.join(running)}  ·  /bashes"))
                bar.display = True
            else:
                bar.update(Text(""))
                bar.display = False
        except Exception:
            pass

    # ---------- prompt history (up/down in the composer) ----------
    def _history_path(self):
        from .config import get_config_dir
        return get_config_dir() / "history"

    def _load_history(self):
        try:
            entries = self._history_path().read_text().splitlines()
            self._history = [e.replace("\\n", "\n") for e in entries if e][-100:]
        except OSError:
            self._history = []

    def _record_history(self, text: str):
        if not text or text.startswith("!"):
            return
        if self._history and self._history[-1] == text:
            return
        self._history.append(text)
        del self._history[:-100]
        self._hist_idx = -1
        self._hist_saved = ""
        try:
            with open(self._history_path(), "a") as f:
                f.write(text.replace("\n", "\\n") + "\n")
        except OSError:
            pass

    def _history_apply(self):
        inp = self.query_one("#input", PromptInput)
        inp.text = self._history[self._hist_idx] if self._hist_idx >= 0 else self._hist_saved
        upto = inp.text or ""
        inp.move_cursor((upto.count("\n"), len(upto) - (upto.rfind("\n") + 1)))
        self._hide_suggest()

    def _history_prev(self) -> bool:
        if not self._history:
            return False
        inp = self.query_one("#input", PromptInput)
        if self._hist_idx < 0:
            self._hist_saved = inp.text or ""
            self._hist_idx = len(self._history) - 1
        else:
            self._hist_idx = max(0, self._hist_idx - 1)
        self._history_apply()
        return True

    def _history_next(self) -> bool:
        if self._hist_idx < 0:
            return False
        self._hist_idx += 1
        if self._hist_idx >= len(self._history):
            self._hist_idx = -1
        self._history_apply()
        return True

    def on_mount(self):
        self.set_interval(0.2, self._tick_busy)  # 5 FPS instead of 10
        llm.prewarm(BASE_URL)
        try:
            self.query_one("#chat", VerticalScroll).show_vertical_scrollbar = True
        except Exception:
            pass
        self.query_one("#input", PromptInput).focus()
        # first session tab (a resumed conversation fills it, otherwise a fresh session)
        self.tabs_state = [self._tab_snapshot()]
        tabs = self.query_one("#session_tabs", Tabs)
        tabs.add_tab(Tab(f"1: {self._tab_title()}",
                         id="tab-0"))
        tabs.active = "tab-0"
        self._log_title()
        self._status(f"{self.session.mode} mode · judge {self.judge_alias} · tools {'on' if self.tools_enabled else 'off'}")
        self._set_hero_visible(not self._resumed)
        if self._resumed:
            self._append_notice(f"Resumed {self._resumed.get('title', '')} — full history below. Mode {self.session.mode}.",
                                f"({self.session.id})")
            self._replay_history()
        else:
            self._check_api_keys()
        self._update_statusline()
        self._update_stats()
        self._update_bottom()
        self._update_queue_display()
        self._load_history()
        self._update_bashes()
        self._check_update()
        from .agent import session_start
        threading.Thread(target=session_start, args=(self.session,), daemon=True).start()

    def _newer(self, latest) -> bool:
        from zeusai import __version__
        try:
            return ([int(x or 0) for x in str(latest).replace("v", "").split(".")]
                    > [int(x or 0) for x in __version__.split(".")])
        except Exception:
            return False

    def _check_update(self):
        """Passive update check (opencode's /update notice), cached weekly, fail silent."""
        import json
        import pathlib as _pl
        import time as _t
        import urllib.request
        from zeusai import __version__
        cache = _pl.Path.home() / ".zeusai" / "update-cache.json"
        latest = None
        try:
            data = json.loads(cache.read_text())
            if _t.time() - data.get("at", 0) < 7 * 24 * 3600:
                latest = data.get("version")
        except Exception:
            pass
        if latest is None:
            def fetch():
                try:
                    req = urllib.request.Request("https://pypi.org/pypi/zeusai/json",
                                                 headers={"User-Agent": "zeusai"})
                    with urllib.request.urlopen(req, timeout=4) as r:
                        v = json.loads(r.read())["info"]["version"]
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps({"at": _t.time(), "version": v}))
                    if self._newer(v):
                        self.call_from_thread(self._append_notice,
                                              f"Update available to v{v} › run `pip install -U zeusai`")
                except Exception:
                    pass
            threading.Thread(target=fetch, daemon=True).start()
            return
        if self._newer(latest):
            self._append_notice(f"Update available to v{latest} › run `pip install -U zeusai`")

    def on_unmount(self):
        from .agent import session_end
        self._stash_tab()
        for state in self.tabs_state:
            if any(m.get("role") == "user" for m in state["messages"]):
                try:
                    s = state["session"]
                    sessions.save(s.id, state["messages"], s.todos, mode=s.mode,
                                  thinking=s.thinking, profile=state["single_alias"],
                                  title=state.get("renamed"))
                except (OSError, SystemExit):
                    pass
        session_end(self.session)
        self._status(f"cwd={pathlib.Path.cwd()}")

    # ---------- helpers (log helpers) ----------
    def _log_title(self):
        model = "ensemble" if self.mode == "ensemble" else (self.single_alias or "auto")
        think = "on" if self.session.thinking else "off" if self.session.thinking is not None else "default"
        img = " · img:✓" if self.image_src else ""
        line = f"{model} · {self.session.mode} · thinking {think} · judge {self.judge_alias}{img}"
        try:
            self.query_one("#stats_left", Static).update(Text(line))
        except Exception:
            pass
        self._update_statusline()
        self._update_image_chip()

    # ---------- ctrl+x chords (opencode-style: x then l/n/m) ----------
    def action_chord_start(self):
        if self._busy:
            return
        self._chord = time.monotonic()
        self.query_one("#input", PromptInput).focus()
        self._status("ctrl+x … (l session · n new · m model)")

    def _run_chord(self, key: str):
        self._chord = None
        if key == "l":
            self._palette_next_tab()
        elif key == "n":
            self.action_new_session()
        elif key == "m":
            self._open_models()

    def _set_hero_visible(self, visible: bool):
        try:
            self.query_one("#hero", Static).display = visible
        except Exception:
            pass

    def _append_markdown(self, md: str, cls: str | None = None):
        log = self.query_one("#chat", VerticalScroll)
        node = Markdown(md)
        if cls:
            node.add_class(cls)
        log.mount(node)
        log.scroll_end(animate=False)

    def _append_user(self, text: str):
        """User prompt shown opencode-style: in a subtle box, no 'You:' prefix, no markdown parsing."""
        self._set_hero_visible(False)
        log = self.query_one("#chat", VerticalScroll)
        node = Static(Text(text), classes="user_msg")
        log.mount(node)
        log.scroll_end(animate=False)

    def _append_plain(self, text: str, style: str | None = None):
        log = self.query_one("#chat", VerticalScroll)
        log.mount(Static(Text(text, style=style or "")))  # Text: never parse [..] as markup
        log.scroll_end(animate=False)

    def _status(self, line: str):
        """Ephemeral status shown above the input (opencode-style), not a side panel."""
        try:
            self.query_one("#status_line", Static).update(Text(line))
        except Exception:
            pass

    def _show_ask(self, question, options, multi, deliver):
        """App thread: show the question; deliver(answer) unblocks the worker."""
        panel = AskPanel(question, options, multi, None)

        def done(answer):
            self._ask_done = None
            panel.remove()
            inp = self.query_one("#input", PromptInput)
            inp.placeholder = "Ask anything…  / for commands, @ for files, ! for shell"
            inp.focus()
            self._append_plain(f"? {question}\n  ❯ {answer}", style="cyan")
            self.busy_msg = random.choice(MESSAGES)
            deliver(answer)

        panel.on_done = done
        self._ask_done = done
        self.busy_msg = "Waiting for your answer"
        log = self.query_one("#chat", VerticalScroll)
        log.mount(panel)
        log.scroll_end(animate=False)

    def _models_text(self) -> str:
        try:
            rows = list_profiles()
            lines = []
            default = get_default_profile_name()
            for r in rows:
                mark = "★" if r["alias"] == default else " "
                extra = ""
                if r["thinking"]:
                    extra += " [thinking]"
                if r["vision"]:
                    extra += " [vision]"
                lines.append(f"{mark} {r['alias']:<12} {r['masked_key']}{extra}")
            lines.append(f"judge={self.judge_alias}  mode={self.mode}")
            return "\n".join(lines)
        except Exception as e:
            return f"models err: {e}"

    # ---------- actions ----------
    def action_quit(self):
        self.exit()
    def action_clear(self):
        try:
            self.query_one("#chat", VerticalScroll).remove_children()
        except Exception:
            pass
        self.agent_messages.clear()
        self.session.todos.clear()
        self.session.id = sessions.new_id()
        self._refresh_tab_label()
        self._status("cleared")
    def action_tools(self):
        self.tools_enabled = not self.tools_enabled
        self._log_title()
        self._status(f"tools={self.tools_enabled}")
    def action_new_session(self):
        """Open a fresh conversation in a new tab."""
        if self._busy:
            self._status("busy; wait for the current turn to finish before opening a tab")
            return
        self._stash_tab()
        state = self._new_tab_state()
        self.tabs_state.append(state)
        index = len(self.tabs_state) - 1
        try:
            tabs = self.query_one("#session_tabs", Tabs)
            tabs.add_tab(Tab(f"{index + 1}: New session", id=f"tab-{index}"))
            tabs.active = f"tab-{index}"
        except Exception as e:
            self._status(f"tab error: {e}")
        self._restore_tab(index)
        self._clear_log()
        self._log_title()
        self._status(f"new session {self.session.id}")
    def _clear_log(self):
        try:
            self.query_one("#chat", VerticalScroll).remove_children()
        except Exception:
            pass
    @on(Tabs.TabActivated, "#session_tabs")
    def _tab_activated(self, event: Tabs.TabActivated):
        if self._switching_tab:
            return
        try:
            index = list(self.query_one("#session_tabs", Tabs).query(Tab)).index(event.tab)
        except ValueError:
            return
        if index == self.active_tab or self._busy:
            return
        self._switching_tab = True
        try:
            self._stash_tab()
            self._restore_tab(index)
            self._clear_log()
            self._log_title()
            self._replay_history()
            self._update_statusline()
            self._status(f"session {index + 1} · {self.session.id} · mode {self.session.mode}")
        finally:
            self._switching_tab = False
    def action_cycle_mode(self):
        if self._busy:
            return
        new = permissions.next_mode(self.session.mode)
        self._apply_mode(new)
    def _apply_mode(self, new):
        """Switch mode, applying its model + thinking preset (an explicit /model still wins)."""
        suggested = set_mode(self.session, new)
        if suggested:
            try:
                alias = resolve_auto(suggested)
                get_active_profile(alias)
                self.single_alias = alias
                self.mode = "single"
            except Exception:
                pass
        self._log_title()
        think = "on" if self.session.thinking else "off"
        self._status(f"mode={new}: {permissions.MODES[new]} · thinking {think}"
                     + (f" · model {self.single_alias}" if suggested and self.single_alias else ""))
    def action_toggle_enhance(self):
        self.session.enhance = not self.session.enhance
        self._log_title()
        self._status(f"enhance-prompt={'on' if self.session.enhance else 'off'}")
    
    def action_enhance_prompt(self):
        """Enhance the current input prompt using the LLM."""
        inp = self.query_one("#input", PromptInput)
        raw = (inp.text or "").strip()
        if not raw:
            self._status("Enhance: input is empty")
            return
        
        # Check if it's a slash command or special command
        if raw.startswith("/") or raw.startswith("!") or raw in ("quit", "exit"):
            self._status("Enhance: cannot enhance slash commands or shell commands")
            return
        
        self._status("Enhancing prompt...")
        
        def enhance_worker():
            try:
                enhanced = enhance_prompt(raw)
                if enhanced != raw:
                    self.call_from_thread(self._show_enhance_dialog, raw, enhanced)
                else:
                    self.call_from_thread(self._status, "Enhance: prompt unchanged")
            except Exception as e:
                self.call_from_thread(self._status, f"Enhance error: {e}")
        
        threading.Thread(target=enhance_worker, daemon=True).start()
    
    def _show_enhance_dialog(self, original: str, enhanced: str):
        """Show a dialog to choose between original and enhanced prompt."""
        def on_done(choice):
            inp = self.query_one("#input", PromptInput)
            if choice == "Use enhanced":
                inp.text = enhanced
                self._status("Enhanced prompt applied")
            elif choice == "Keep original":
                self._status("Kept original prompt")
            else:
                inp.text = choice
                self._status("Custom prompt applied")
        
        panel = AskPanel(
            f"Enhanced prompt:\n\n{enhanced}\n\nOriginal:\n{original}\n\nChoose:",
            ["Use enhanced", "Keep original", "Edit manually"],
            False,
            on_done
        )
        log = self.query_one("#chat", VerticalScroll)
        log.mount(panel)
        log.scroll_end(animate=False)
    
    def action_add_to_queue(self):
        """Add current input to the message queue."""
        inp = self.query_one("#input", PromptInput)
        raw = (inp.text or "").strip()
        if not raw:
            self._status("Queue: input is empty")
            return
        
        # Check if it's a slash command or special command
        if raw.startswith("/") or raw.startswith("!") or raw in ("quit", "exit"):
            self._status("Queue: cannot queue slash commands or shell commands")
            return
        
        want_tools = self.tools_enabled or raw.lower().startswith("run:")
        prompt = raw[4:].strip() if raw.lower().startswith("run:") else raw
        prompt, note = prepare_prompt(prompt)
        
        # Add to queue
        idx = self.message_queue.add(
            prompt=prompt,
            image_src=self.image_src,
            want_tools=want_tools,
            enhanced=False,
            original_prompt=raw
        )
        
        # Update queue display
        self._update_queue_display()
        
        # Clear input
        inp.text = ""
        self._status(f"Queued #{idx + 1}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    
    def action_process_queue(self):
        """Process all queued prompts."""
        if self._queue_processing:
            self._status("Queue: already processing")
            return
        
        if not self.message_queue.has_pending():
            self._status("Queue: no pending prompts")
            return
        
        self._queue_processing = True
        self._status("Queue: processing...")
        self._process_next_in_queue()
    
    def action_clear_queue(self):
        """Clear the message queue."""
        count = len(self.message_queue.queue)
        self.message_queue.clear()
        self._update_queue_display()
        self._status(f"Queue: cleared {count} prompt(s)")
    
    def _update_queue_display(self):
        """Update the queue strip above the input (hidden when empty)."""
        try:
            items = self.message_queue.get_display_items()
            bar = self.query_one("#queue_bar", Static)
            if items:
                bar.update(Text("  ".join(items)))
                bar.display = True
            else:
                bar.update(Text(""))
                bar.display = False
        except Exception:
            pass
    
    def _process_next_in_queue(self):
        """Process the next prompt in the queue."""
        qp = self.message_queue.get_next()
        if qp is None:
            self._queue_processing = False
            self._update_queue_display()
            self._status("Queue: all done")
            return
        
        self._update_queue_display()
        self._status(f"Queue: processing #{self.message_queue.current_index + 1}/{len(self.message_queue.queue)}")
        
        # Show the prompt being processed
        self._append_user("queued: " + qp.prompt + (f"\n\nimage: {qp.image_src}" if qp.image_src else ""))
        
        # Run the prompt
        self.busy = True
        t = threading.Thread(target=self._work, args=(qp.prompt, qp.want_tools, qp.image_src, self.mode, self.single_alias, self.judge_alias), daemon=True)
        t.start()
        
        # Store callback to process next after this one completes
        # We'll use a wrapper to chain the next queue item
        self._queue_worker_thread = t
    def action_ensemble(self):
        self.mode = "ensemble"
        self.single_alias = None
        self._log_title()
        self._status("mode=ensemble")
    def action_help(self):
        self._append_markdown(HELP_TEXT)

    # ---------- command palette (ctrl+p, opencode-style dialogs) ----------
    def action_command_palette(self):
        """Override Textual's built-in palette with opencode-style commands."""
        self.action_open_palette()

    def action_open_palette(self):
        items = [
            ("New session", "ctrl+n", self.action_new_session),
            ("Switch session", "next tab", self._palette_next_tab),
            ("Switch model", "models", self._open_models),
            ("Clear conversation", "/clear", self.action_clear),
            ("Cycle mode", "shift+tab", self.action_cycle_mode),
            ("Toggle tools", "ctrl+t", self.action_tools),
            ("Toggle enhance-prompt", "ctrl+e", self.action_toggle_enhance),
            ("Export conversation", "/export", self._palette_export),
            ("Save session", "resume-friendly", self._palette_save),
            ("Help", "/help", self.action_help),
            ("Quit", "ctrl+c", self.action_quit),
        ]
        self.push_screen(CommandScreen(self, "Commands", items))

    def _palette_next_tab(self):
        if len(self.tabs_state) < 2:
            self._status("only one session tab open")
            return
        nxt = (self.active_tab + 1) % len(self.tabs_state)
        self.query_one("#session_tabs", Tabs).active = f"tab-{nxt}"

    def _resume_pick(self, sid: str):
        saved = sessions.load(sid)
        if saved:
            self._open_resumed(saved)
        else:
            self._append_plain(f"No saved conversation {sid}")

    def _pick_agent(self, name: str):
        """Prefill the composer so the task runs through a chosen subagent."""
        inp = self.query_one("#input", PromptInput)
        inp.text = f"Use the {name} subagent to "
        inp.move_cursor((0, len(inp.text)))
        inp.focus()
        self._status(f"agent: {name} — describe the task and press Enter")

    def _pick_mcp(self, name: str):
        from . import mcp
        try:
            st = mcp.status().get(name, "not started")
        except Exception as e:
            st = f"error: {e}"
        self._append_plain(f"MCP {name}: {st}")

    def _open_models(self):
        try:
            rows = list_profiles()
        except Exception as e:
            self._status(f"models err: {e}")
            return
        default = get_default_profile_name()
        items = []
        for r in rows:
            mark = "✓" if r["alias"] == self.single_alias else "★" if r["alias"] == default else " "
            extra = (" [thinking]" if r["thinking"] else "") + (" [vision]" if r["vision"] else "")
            alias = r["alias"]
            items.append((f"{mark} {alias}  {r['masked_key']}{extra}".strip(), "model",
                          (lambda a=alias: self._palette_set_model(a))))
        items.append(("Auto (session default)", "model", lambda: self._palette_set_model("auto")))
        items.append(("Ensemble (all models + judge)", "ensemble", self.action_ensemble))
        self.push_screen(CommandScreen(self, "Switch model", items))

    def _palette_set_model(self, alias: str):
        err = self._set_model(alias)
        if err:
            self._append_plain(f"Error: {err}")
        else:
            self._append_plain(f"Single mode: {self.single_alias}")
            self._status(f"mode=single alias={self.single_alias}")

    def _set_model(self, alias: str) -> str:
        """Switch to a single model. Returns '' on success or an error message."""
        try:
            alias = resolve_auto(alias)
            get_active_profile(alias)
        except Exception as e:
            return str(e)
        self.mode = "single"
        self.single_alias = alias
        self._log_title()
        return ""

    def _palette_export(self):
        path = export_markdown(self.agent_messages)
        self._append_plain(f"Exported to {path}")

    def _palette_save(self):
        try:
            sessions.save(self.session.id, self.agent_messages, self.session.todos,
                          mode=self.session.mode, thinking=self.session.thinking,
                          profile=self.single_alias, title=self._renamed_title)
            self._status(f"saved session {self.session.id}")
        except OSError as e:
            self._status(f"save failed: {e}")
    def action_interrupt(self):
        if isinstance(self.screen, CommandScreen):
            self.screen.dismiss()
            return
        if self._suggest is not None:
            self._hide_suggest()
            self.query_one("#input", PromptInput).focus()
            return
        if self._ask_done:
            self._ask_done("(skipped)")
            return
        if self.busy:
            self.cancel.set()
            self._status("interrupting…")
            return
        # idle: Esc clears the composer (opencode-style)
        inp = self.query_one("#input", PromptInput)
        inp.text = ""
        inp.focus()

    # ---------- input ----------
    def submit_from_input(self, raw_text: str):
        """Send the prompt box contents (Enter in the multiline input lands here)."""
        raw = (raw_text or "").strip()
        self.query_one("#input", PromptInput).text = ""
        self._hide_suggest()
        if raw and self._ask_done:
            self._record_history(raw)
            self._ask_done(raw)  # typed answer to a pending question
            return
        if not raw:
            return
        self._record_history(raw)
        if self.busy and not raw.startswith("/") and not raw.startswith("!"):
            # opencode-style: queue the prompt; it runs automatically after this turn
            follow = raw[4:].strip() if raw.lower().startswith("run:") else raw
            prompt, _note = prepare_prompt(follow)
            idx = self.message_queue.add(
                prompt=prompt, image_src=self.image_src,
                want_tools=self.tools_enabled or raw.lower().startswith("run:"),
                original_prompt=raw)
            self._update_queue_display()
            self._status(f"queued #{idx + 1} (runs next)")
            return
        if self.busy:
            return  # slash/! commands while busy are dropped; run them when idle
        if raw.startswith("!") and len(raw) > 1:
            cmd = raw[1:].strip()
            self._append_plain(f"$ {cmd}")
            def go():
                out, code = bash_passthrough(cmd, self.agent_messages)
                self.call_from_thread(self._append_plain, (out or "(no output)") + (f"\nexit {code}" if code else ""))
            threading.Thread(target=go, daemon=True).start()
            return
        # slash commands (custom /commands from .zeusai/commands expand into a prompt)
        if raw.startswith("/") and not commands_exist(raw):
            self._slash(raw)
            return
        if raw in ("quit", "exit"):
            self.exit()
            return
        # run: prefix forces tools
        want_tools = self.tools_enabled or raw.lower().startswith("run:")
        prompt = raw[4:].strip() if raw.lower().startswith("run:") else raw
        shown = prompt
        prompt, note = prepare_prompt(prompt)
        if note:
            shown += f"\n\n*{note}*"

        self._append_user(shown + (f"\n\nimage: {self.image_src}" if self.image_src else ""))
        self.busy = True
        # run in worker thread
        t = threading.Thread(target=self._work, args=(prompt, want_tools, self.image_src, self.mode, self.single_alias, self.judge_alias), daemon=True)
        t.start()
        # consume image after one use? keep until /image none
        # keep image (user can /image none to clear)

    # ---------- suggest popups (slash commands / @ files) ----------
    def _cursor_offset(self, ta: PromptInput) -> int:
        """Character offset of the cursor within the full text."""
        line_no, col = ta.cursor_location
        lines = ta.text.split("\n")
        return sum(len(l) + 1 for l in lines[:line_no]) + col

    @on(TextArea.Changed, "#input")
    def _input_changed(self, event: TextArea.Changed):
        """Drive the / and @ completion popups as the prompt is typed."""
        if self._busy:
            self._hide_suggest()
            return
        ta = event.text_area
        text = ta.text or ""
        off = self._cursor_offset(ta)
        prefix = text[:off]
        boundary = max(prefix.rfind(" "), prefix.rfind("\n"))
        token = prefix[boundary + 1:]
        if token.startswith("/"):
            needle = token[1:].lower()
            items = [(cmd, desc) for cmd, desc in COMMANDS if cmd[1:].lower().startswith(needle)]
            if items:
                self._show_suggest(boundary + 1, "/", "command", items)
                return
        elif token.startswith("@"):
            needle = token[1:]
            items = self._file_suggest(needle)
            if items:
                self._show_suggest(boundary + 1, "@", "file", items)
                return
        self._hide_suggest()

    def _file_suggest(self, needle: str):
        """Files in the working directory matching the prefix (@ completion, cwd only)."""
        import pathlib
        try:
            names = sorted(os.listdir(self.cwd))
        except OSError:
            return []
        items = []
        for name in names:
            if not name.startswith(needle) or name.startswith("."):
                continue
            is_dir = pathlib.Path(self.cwd, name).is_dir()
            items.append((name + ("/" if is_dir else ""), "directory" if is_dir else "file"))
            if len(items) >= 10:
                break
        return items

    def _show_suggest(self, offset: int, trigger: str, kind: str, items):
        self._suggest = {"offset": offset, "trigger": trigger, "kind": kind,
                         "values": [v.split()[0] if kind == "command" else v for v, _d in items]}
        panel = self.query_one("#suggest", Vertical)
        panel.display = True
        opt = self.query_one("#suggest_list", OptionList)
        opt.clear_options()
        opt.add_options([escape(f"{v}  {d}") for v, d in items])
        try:
            opt.highlighted = 0
        except Exception:
            pass
        opt.focus()

    def _hide_suggest(self):
        self._suggest = None
        try:
            self.query_one("#suggest", Vertical).display = False
        except Exception:
            pass

    @on(OptionList.OptionSelected, "#suggest_list")
    def _suggest_chosen(self, event: OptionList.OptionSelected):
        info = self._suggest
        if not info or not (0 <= event.option_index < len(info["values"])):
            return
        value = info["values"][event.option_index]
        ta = self.query_one("#input", PromptInput)
        off = self._cursor_offset(ta)
        start = info["offset"]
        replacement = value + (" " if info["kind"] == "command" else "")
        ta.text = (ta.text or "")[:start] + replacement + (ta.text or "")[off:]
        upto = ta.text[: start + len(replacement)]
        new_line = upto.count("\n")
        new_col = len(upto) - (upto.rfind("\n") + 1)
        ta.move_cursor((new_line, new_col))
        self._hide_suggest()
        ta.focus()

    # ---------- stats (in-box info row + context meter) ----------
    def _fmt_tokens(self, n: int) -> str:
        return f"{n / 1000:.1f}K" if n >= 1000 else str(n)

    def _usage_tokens(self) -> int:
        return llm.USAGE["prompt"]

    def _update_stats(self):
        """Left: mode + model (opencode-style in-box row). Right: timing/throughput while busy."""
        left = f"{self.session.mode} · {self.single_alias or f'ensemble/{self.judge_alias}'}"
        right = ""
        sink = getattr(self, "_sink", None)
        if self._busy and sink is not None:
            elapsed = time.monotonic() - sink.started
            tps = (sink.chars / 4 / elapsed) if elapsed > 0 else 0.0
            right = f"{elapsed:.1f}s · {tps:.1f} tok/s"
        else:
            think = "on" if self.session.thinking else "off" if self.session.thinking is not None else "default"
            right = f"thinking {think} · judge {self.judge_alias}"
        try:
            self.query_one("#stats_left", Static).update(Text(left))
            self.query_one("#stats_right", Static).update(Text(right))
        except Exception:
            pass

    def _update_statusline(self):
        """Under-box status: short cwd · git branch + unclean count · tokens (context %)."""
        from . import statusline
        try:
            from .statusline import _setting
            if _setting().get("command"):
                line = statusline.render(self.single_alias or f"ensemble/{self.judge_alias}", self.session)
                self.query_one("#cwd_bar", Static).update(Text(line))
                return
        except Exception:
            pass
        home = os.path.expanduser("~")
        cwd = self.cwd
        short = "~" + cwd[len(home):] if cwd.startswith(home) else cwd
        parts = [short]
        git = self._git_info()
        if git:
            parts.extend(git)
        tokens = self._usage_tokens()
        if tokens:
            usage = f"{self._fmt_tokens(tokens)} tok"
            limit = self._context_limit()
            if limit:
                usage += f" ({tokens / limit * 100:.0f}%)"
            parts.append(usage)
        if self.session.todos:
            parts.append(f"{len(self.session.todos)} todos")
        try:
            self.query_one("#cwd_bar", Static).update(Text(" · ".join(parts)))
        except Exception:
            pass

    def _git_info(self):
        """[branch, unclean-count] from the cached git context snapshot (no subprocess here)."""
        try:
            from .agent import git_context
            text = git_context()
        except Exception:
            return []
        if not text:
            return []
        branch = None
        unclean = 0
        in_status = False
        for line in text.splitlines():
            if line.startswith("Git: branch "):
                branch = line[len("Git: branch "):]
            elif line.startswith("Status:"):
                in_status = True
            elif line.startswith("Recent commits:"):
                in_status = False
            elif in_status and line.strip() and line.strip() != "(clean)":
                unclean += 1
        out = [branch] if branch else []
        if unclean:
            out.append(str(unclean))
        return out

    def _context_limit(self):
        """Context window size for the active model, from the profile ("context" key)."""
        try:
            alias = self.judge_alias if self.mode == "ensemble" else self.single_alias
            if not alias:
                return None
            _, prof = get_active_profile(resolve_auto(alias))
            return prof.get("context") or None
        except Exception:
            return None

    def _update_image_chip(self):
        try:
            chip = self.query_one("#image_chip", Static)
            if self.image_src:
                base = os.path.basename(self.image_src)
                chip.update(Text(f"🖼 {base}  ·  /image none to clear"))
                chip.display = True
            else:
                chip.update(Text(""))
                chip.display = False
        except Exception:
            pass

    def _update_bottom(self):
        """Opencode-style bottom bar: MCP count left, version + hints right."""
        try:
            from . import mcp
            n = len(mcp.configured())
        except Exception:
            n = 0
        try:
            from zeusai import __version__
            self.query_one("#bottom_left", Static).update(Text(f"⊙ {n} MCP /mcps"))
            self.query_one("#bottom_right", Static).update(
                Text(f"zeusai {__version__} · shift+tab mode · ctrl+n new · ctrl+c quit"))
        except Exception:
            pass

    def _append_notice(self, text: str, action: str | None = None):
        """Opencode-style notice card rendered in the chat (update/auth notices)."""
        body = f"{text}  › {action}" if action else text
        log = self.query_one("#chat", VerticalScroll)
        log.mount(Static(Text(body), classes="notice"))
        log.scroll_end(animate=False)

    def _check_api_keys(self):
        """Boot notice when no profile has an API key (opencode's auth notice)."""
        try:
            if any(p.get("api_key") for p in get_profiles().values()):
                return
        except Exception:
            return
        self._append_notice("No API keys configured",
                            "run `zeusai configure --profile laguna --api-key <key>`, then `zeusai doctor`")

    def _slash(self, raw: str):
        parts = raw.split()
        cmd = parts[0].lower()
        if cmd == "/help":
            self._append_markdown(HELP_TEXT)
        elif cmd == "/models":
            self._append_plain(self._models_text())
        elif cmd == "/model" and len(parts) == 2:
            err = self._set_model(parts[1])
            if err:
                self._append_plain(f"Error: {err}")
            else:
                self._append_plain(f"Single mode: {self.single_alias}")
                self._status(f"mode=single alias={self.single_alias}")
        elif cmd == "/ensemble":
            self.mode = "ensemble"
            self.single_alias = None
            self._log_title()
            self._append_plain("Ensemble mode (all models + judge).")
        elif cmd == "/judge" and len(parts) == 2:
            self.judge_alias = parts[1]
            self._log_title()
            self._append_plain(f"Judge = {self.judge_alias}")
        elif cmd == "/tools":
            if len(parts) == 2:
                self.tools_enabled = parts[1].lower() in ("on", "1", "true", "yes")
            else:
                self.tools_enabled = not self.tools_enabled
            self._log_title()
            self._append_plain(f"Tools {'ON' if self.tools_enabled else 'OFF'} — coding agent uses read_file/write_file/list_files/run_shell.")
        elif cmd == "/image":
            if len(parts) == 2 and parts[1].lower() in ("none", "-", "clear"):
                self.image_src = None
                self._log_title()
                self._append_plain("Image cleared.")
            elif len(parts) == 2:
                self.image_src = parts[1]
                self._log_title()
                self._append_plain(f"Image attached: {self.image_src} — next prompt will be vision.")
            else:
                self._append_plain("Usage: /image <path|url>  or  /image none")
        elif cmd == "/mode" and len(parts) == 2:
            if parts[1] in permissions.MODES:
                self._apply_mode(parts[1])
                self._append_plain(f"Mode {parts[1]}: {permissions.MODES[parts[1]]}"
                                   f" · thinking {'on' if self.session.thinking else 'off'}")
            else:
                self._append_plain(f"Unknown mode. Modes: {', '.join(permissions.MODE_ORDER)}")
        elif cmd == "/new":
            self.action_new_session()
            self._append_plain(f"New session {self.session.id} in tab {self.active_tab + 1}.")
        elif cmd == "/autocontinue":
            arg = parts[1].lower() if len(parts) == 2 else ""
            self.session.auto_continue = {"on": True, "off": False}.get(arg, not self.session.auto_continue)
            self._append_plain(f"Auto-continue {'on' if self.session.auto_continue else 'off'}")
        elif cmd == "/enhance":
            self.action_toggle_enhance()
            self._append_plain(f"Enhance-prompt {'on' if self.session.enhance else 'off'}")
        elif cmd == "/permissions":
            self._append_plain(permissions_command(raw[len("/permissions"):].strip(), self.session))
        elif cmd == "/mode":
            self._append_plain(f"agent mode={self.session.mode} thinking={'on' if self.session.thinking else 'off' if self.session.thinking is not None else 'model default'} "
                               f"autocontinue={'on' if self.session.auto_continue else 'off'} enhance={self.session.enhance} "
                               f"model mode={self.mode} single={self.single_alias} judge={self.judge_alias} tools={self.tools_enabled} image={self.image_src or '-'}")
        elif cmd == "/clear":
            self.action_clear()
        elif cmd == "/think":
            arg = parts[1].lower() if len(parts) == 2 else ""
            self.session.thinking = {"on": True, "off": False}.get(arg, not bool(self.session.thinking))
            self._append_plain(f"Extended thinking {'on' if self.session.thinking else 'off'}")
        elif cmd == "/export":
            path = export_markdown(self.agent_messages, parts[1] if len(parts) == 2 else None)
            self._append_plain(f"Exported to {path}")
        elif cmd == "/style":
            from . import styles
            if len(parts) == 1:
                self._append_plain(styles.listing(self.session.style or styles.default_style()))
            elif parts[1] in styles.all_styles():
                self.session.style = parts[1]
                self._append_plain(f"Output style: {parts[1]}")
            else:
                self._append_plain(f"Unknown style {parts[1]}")
        elif cmd == "/agents" and len(parts) > 2 and parts[1] == "new":
            desc = raw.split(None, 2)[2]
            self._status("designing agent…")
            def go():
                from .agents import create_agent
                try:
                    msg = f"Created {create_agent(desc, self.single_alias)}"
                except Exception as e:
                    msg = f"Couldn't create agent: {e}"
                self.call_from_thread(self._append_plain, msg)
            threading.Thread(target=go, daemon=True).start()
        elif cmd == "/context":
            self._append_plain(context_text(self.agent_messages, self.session, self.single_alias or f"ensemble/{self.judge_alias}"))
        elif cmd == "/doctor":
            self._status("running checks…")
            threading.Thread(target=lambda: self.call_from_thread(self._append_plain, doctor_text(self.single_alias)), daemon=True).start()
        elif cmd == "/add-dir":
            self._append_plain(add_dir(raw[len("/add-dir"):], self.session))
        elif cmd == "/paste":
            from . import clipboard
            try:
                self.image_src = clipboard.save_image()
                self._log_title()
                self._append_plain(f"Clipboard image attached: {self.image_src} (sent with your next prompt)")
            except RuntimeError as e:
                self._append_plain(str(e))
        elif cmd == "/status":
            self._append_plain(status_text(self.session, self.single_alias or f"ensemble/{self.judge_alias}"))
        elif cmd == "/copy":
            text = ""
            for m in reversed(self.agent_messages):
                if m.get("role") == "assistant" and m.get("content"):
                    text = m["content"] if isinstance(m["content"], str) else ""
                    break
            if not text:
                self._append_plain("Nothing to copy yet.")
                return
            def go_copy():
                from . import clipboard
                try:
                    tool = clipboard.copy_text(text)
                    msg = f"Copied the last reply ({len(text)} chars) via {tool}."
                except RuntimeError as e:
                    msg = f"Copy failed: {e}"
                self.call_from_thread(self._append_plain, msg)
            threading.Thread(target=go_copy, daemon=True).start()
        elif cmd == "/diff":
            self._status("computing diff…")
            def go_diff():
                import subprocess
                try:
                    stat = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True,
                                          timeout=10, cwd=os.getcwd())
                    diff = subprocess.run(["git", "diff"], capture_output=True, text=True,
                                          timeout=10, cwd=os.getcwd())
                    if stat.returncode != 0:
                        msg = "Not a git repo (no diff)."
                    elif not (stat.stdout.strip() or diff.stdout.strip()):
                        msg = "No uncommitted changes."
                    else:
                        body = "\n".join(diff.stdout.strip().splitlines()[:80])
                        msg = f"Uncommitted changes:\n{stat.stdout.strip()}\n\n{body}" + ("\n…" if len(diff.stdout.strip().splitlines()) > 80 else "")
                except Exception as e:
                    msg = f"Diff failed: {e}"
                self.call_from_thread(self._append_plain, msg)
            threading.Thread(target=go_diff, daemon=True).start()
        elif cmd == "/btw":
            text = raw[len("/btw"):].strip()
            if not text:
                self._append_plain("Usage: /btw <question> (one-shot answer, no tools)")
                return
            if self.busy:
                self._append_plain("Busy; try again when the current task finishes.")
                return
            self._append_user(text)
            self.busy = True
            def go_btw():
                try:
                    _, prof = get_active_profile(self.single_alias or self.judge_alias)
                    stream = [None]
                    def on_text(chunk):
                        if stream[0] is None:
                            stream[0] = self.call_from_thread(self._open_stream, "**Answer:**\n\n")
                        self.call_later(self._stream_write, stream[0], chunk)
                    r = llm.chat(prof, [{"role": "system", "content": SYSTEM_PROMPT},
                                         {"role": "user", "content": text}],
                                 on_text=on_text,
                                 thinking=self.session.thinking)
                    if stream[0] is not None:
                        self.call_later(stream[0].stop)
                    if r.get("reasoning"):
                        self.call_from_thread(self._append_plain, "💭 " + r["reasoning"][-500:], "dim italic")
                except Exception as e:
                    self.call_from_thread(self._append_plain, f"Error: {e}")
                finally:
                    self.call_from_thread(lambda: setattr(self, "busy", False))
            threading.Thread(target=go_btw, daemon=True).start()
        elif cmd == "/cd":
            arg = raw[len("/cd"):].strip()
            if not arg:
                self._append_plain(f"cwd: {os.getcwd()}\nUsage: /cd <path>")
                return
            p = pathlib.Path(arg).expanduser()
            if not p.is_dir():
                self._append_plain(f"Not a directory: {arg}")
                return
            p = p.resolve()
            os.chdir(p)
            self.cwd = str(p)
            self._update_statusline()
            self._append_plain(f"Working directory: {self.cwd}  (sessions are stored per directory)")
        elif cmd == "/bashes":
            self._append_plain(bashes_text())
        elif info_command(cmd) is not None:
            self._append_plain(info_command(cmd))
        elif cmd == "/rewind":
            if not self.session.checkpoints:
                self._append_plain("Nothing to rewind.")
                return
            n = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 1
            files, n = sessions.rewind(self.session, self.agent_messages, n)
            sessions.save(self.session.id, self.agent_messages, self.session.todos, title=self._renamed_title)
            self._append_plain(f"Rewound {n} prompt(s); restored {len(files)} file(s)" + (": " + ", ".join(files) if files else ""))
        elif cmd == "/todos":
            self._append_plain(sessions.render_todos(self.session.todos))
        elif cmd == "/compact":
            if self.busy:
                self._append_plain("Busy; try again when the current task finishes.")
                return
            focus = raw[len("/compact"):].strip() or None
            self._status("compacting…")
            def go():
                try:
                    done = sessions.compact(self.agent_messages, self.single_alias, focus=focus)
                    if done:
                        sessions.save(self.session.id, self.agent_messages, self.session.todos, title=self._renamed_title)
                    msg = "Compacted older history into a summary." if done else "Nothing to compact yet."
                except Exception as e:
                    msg = f"Compaction failed: {e}"
                self.call_from_thread(self._append_plain, msg)
            threading.Thread(target=go, daemon=True).start()
        elif cmd == "/resume":
            if len(parts) == 1:
                items = sessions.listing()
                if not items:
                    self._append_plain("No saved conversations in this directory.")
                    return
                opts = [(f"{i['title'] or '(untitled)'}  ·  {i['id'][:12]}  ·  {i['updated']}  ·  {i['turns']} turns",
                         "session", (lambda sid=i["id"]: self._resume_pick(sid))) for i in items]
                self.push_screen(CommandScreen(self, "Resume session", opts))
                return
            saved = sessions.load(parts[1])
            if not saved:
                self._append_plain(f"No saved conversation {parts[1]}")
                return
            self._open_resumed(saved)
        elif cmd == "/rename":
            new_title = raw[len("/rename"):].strip()
            if not new_title:
                self._append_plain(f"Session title: {self._tab_title()}")
                return
            self._renamed_title = new_title
            self._stash_tab()
            self._refresh_tab_label()
            self._append_plain(f"Session renamed to {new_title}")
        elif cmd == "/agents":
            from . import agents
            opts = [(f"{n}  {a.get('description', '')}", "agent",
                     (lambda n=n: self._pick_agent(n))) for n, a in agents.all_agents().items()]
            self.push_screen(CommandScreen(self, "Subagents", opts))
        elif cmd in ("/mcp", "/mcps"):
            from . import mcp
            conf = mcp.configured()
            if not conf:
                self._append_plain("No MCP servers configured (zeusai mcp add <name> -- <command> [args]).")
                return
            opts = [(f"{n}  {spec.get('url') or spec.get('command', '')}", "server",
                     (lambda n=n: self._pick_mcp(n))) for n, spec in conf.items()]
            self.push_screen(CommandScreen(self, "MCP servers", opts))
        elif cmd in ("/quit", "/exit"):
            self.exit()
        else:
            self._append_plain(f"Unknown command: {cmd}. Try /help")

    # ---------- streaming ----------
    def _open_stream(self, header: str):
        """Runs on the app thread: mount a Markdown widget and return its stream."""
        log = self.query_one("#chat", VerticalScroll)
        node = Markdown(header)
        log.mount(node)
        log.scroll_end(animate=False)
        return Markdown.get_stream(node)

    async def _stream_write(self, stream, text: str):
        await stream.write(text)
        self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

    def _final_file_text(self, name, args):
        """Static +/- diff text for a completed file tool call (shown after streaming)."""
        header = tool_header(name, args)
        text = Text()
        text.append(f"● {header}", style="bold green")
        diff = edit_preview(name, args, max_lines=30)
        if diff:
            adds = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
            dels = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
            if adds or dels:
                text.append(f"  +{adds} -{dels}", style="dim")
        text.append("\n")
        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                style = "green"
            elif line.startswith("-") and not line.startswith("---"):
                style = "red"
            else:
                style = "dim"
            text.append(f"  {line}\n", style=style)
        return text

    def _open_file_stream(self):
        """Runs on the app thread: mount a live file panel and return its widget."""
        log = self.query_one("#chat", VerticalScroll)
        node = Static("", classes="file_live")
        log.mount(node)
        log.scroll_end(animate=False)
        return node

    def _update_file_widget(self, node, text):
        try:
            node.update(text)
        except Exception:
            pass
        try:
            self.query_one("#chat", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    def _open_thought(self):
        """Runs on the app thread: mount the live 'thinking' panel."""
        log = self.query_one("#chat", VerticalScroll)
        node = Static("", classes="thought_live")
        log.mount(node)
        log.scroll_end(animate=False)
        return node

    def _update_thought(self, node, text):
        tail = "\n".join((text or "").splitlines()[-12:])
        try:
            node.update(Text("💭 Thinking\n" + tail, style="dim italic"))
        except Exception:
            pass
        try:
            self.query_one("#chat", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    def _tool_section_text(self, name, args, result=None, elapsed=None, running=False, expanded=False):
        """Header + result for one tool call, rendered as its own section.

        Shell commands get the full command string as the heading, so the exact
        command the agent ran is always visible. While a call is running the
        section shows a running state; finished sections show elapsed time.
        Click a finished section to expand its full output.
        """
        text = Text()
        if name == "run_shell":
            text.append("$ ", style="bold yellow")
            text.append(args.get("command", ""), style="bold")
        else:
            text.append("● ", style="bold rgb(45,212,191)")
            text.append(tool_header(name, args), style="bold")
        if running:
            text.append("\n  running…", style="dim")
        if result is not None:
            text.append("\n  ⎿  " + tool_summary(name, args, result), style="dim")
            if elapsed is not None and elapsed >= 0:
                text.append(f" · {elapsed:.1f}s", style="dim")
            if name == "run_shell":
                body = result.partition("\n")[2].rstrip()
                lines = body.splitlines()
                keep = 60 if expanded else 8
                if lines and body != "(no output)":
                    for line in lines[-keep:]:
                        text.append("\n    " + line, style="dim")
                    if len(lines) > keep:
                        text.append(f"\n    … {len(lines) - keep} more lines", style="dim")
                if not expanded and len(lines) > 8:
                    text.append("\n  click to expand", style="dim")
            elif expanded:
                for line in result.splitlines()[:100]:
                    text.append("\n    " + line, style="dim")
            elif len(result) > 300 or "\n" in result:
                text.append("\n  click to expand", style="dim")
        return text

    def _open_tool_section(self, name, args):
        """Runs on the app thread: mount a section for a tool call (shell commands especially)."""
        log = self.query_one("#chat", VerticalScroll)
        node = Static(self._tool_section_text(name, args, running=True),
                      classes="tool_section" + (" shell" if name == "run_shell" else ""))
        node._sec = {"name": name, "args": args, "result": None, "elapsed": None, "expanded": False}
        log.mount(node)
        log.scroll_end(animate=False)
        return node

    def _render_section(self, node):
        sec = getattr(node, "_sec", None)
        if not sec:
            return
        node.update(self._tool_section_text(sec["name"], sec["args"], sec["result"],
                                            elapsed=sec["elapsed"], running=sec["result"] is None,
                                            expanded=sec["expanded"]))

    def _finish_tool_section(self, node, name, args, result, elapsed=None):
        sec = getattr(node, "_sec", None)
        if sec is not None:
            sec.update(name=name, args=args, result=result, elapsed=elapsed)
            self._render_section(node)
        try:
            self.query_one("#chat", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    def on_click(self, event):
        """Click a finished tool section to expand/collapse its full output."""
        node = event.widget
        if not node.has_class("tool_section"):
            return
        sec = getattr(node, "_sec", None)
        if not sec or sec["result"] is None:
            return
        sec["expanded"] = not sec["expanded"]
        self._render_section(node)

    # ---------- background work ----------
    def _work(self, prompt: str, want_tools: bool, image_src: str | None, mode: str, single_alias: str | None, judge_alias: str):
        app = self

        class Sink:
            """Collects tokens in the worker thread and flushes to the UI ~10x/sec (batched)."""
            def __init__(self, header):
                self.stream = app.call_from_thread(app._open_stream, header)
                self.buf = []
                self.last = 0.0
                self.pending = False
                self.started = time.monotonic()  # for the in-box elapsed + tok/s stats
                self.chars = 0
                app._sink = self
            def __call__(self, chunk):
                self.buf.append(chunk)
                self.chars += len(chunk)
                now = time.monotonic()
                if now - self.last >= 0.1:  # 10 FPS instead of 20
                    self.flush()
                    self.last = now
            def flush(self):
                if self.buf:
                    text = "".join(self.buf)
                    self.buf.clear()
                    # Use call_later to batch multiple flushes in one event loop iteration
                    app.call_later(app._stream_write, self.stream, text)
            def close(self):
                self.flush()
                app.call_later(self.stream.stop)

        def status(line: str):
            self.call_from_thread(lambda: self._status(line))

        sink = None
        try:
            if want_tools:
                plan_note = ""
                if mode == "ensemble":
                    status("agent: gathering ensemble plan…")
                    plan, _, _, meta = ensemble_ask(
                        f"Plan how to complete this task (concise steps, files):\n{prompt}",
                        image_src=image_src, judge_alias=judge_alias, verbose=False, on_status=status,
                    )
                    plan_note = f"\n\nExpert plan:\n{plan}"
                session = self.session

                def on_ask(question, options, multi=False):
                    answer, ready = {}, threading.Event()
                    def deliver(a):
                        answer["a"] = a
                        ready.set()
                    self.call_from_thread(self._show_ask, question, options, multi, deliver)
                    ready.wait()
                    return answer["a"]

                def on_permission(name, args, rule):
                    for line in edit_preview(name, args):
                        status("    " + line)
                    always = ALLOW_ALWAYS.format(rule=rule)
                    what = f"\n$ {args.get('command', '')}" if name == "run_shell" else ""
                    answer = on_ask(f"Allow {tool_header(name, args)}?{what}", [ALLOW_ONCE, always, DENY])
                    if answer == ALLOW_ONCE:
                        return True, None
                    if answer == always:
                        permissions.save_rule(rule)
                        session.allowed.append(rule)
                        status(f"saved allow rule {rule}")
                        return True, None
                    return False, (None if answer == DENY else answer)

                if session.enhance:
                    status("enhancing prompt…")
                    better = enhance_prompt(prompt)
                    if better != prompt:
                        choice = on_ask(f"Enhanced prompt:\n\n{better}\n\nSend which prompt?",
                                        ["Use the enhanced prompt", "Use my original"])
                        prompt = better if choice == "Use the enhanced prompt" else prompt if choice == "Use my original" else choice

                messages = self.agent_messages
                if not messages:
                    messages.append({"role": "system", "content": SYSTEM_PROMPT})
                if plan_note:
                    messages.append({"role": "system", "content": f"Ensemble expert plan for the next task:{plan_note}"})
                user_msg = {"role": "user", "content": prompt}
                if image_src:  # vision models see the image; text-only ones get a vision model's description
                    from .agent import user_message
                    user_msg = user_message(prompt, image_src, judge_alias if mode == "ensemble" else single_alias, status)
                    self.image_src = None  # one prompt per attachment in agent mode
                    self.call_from_thread(self._log_title)
                messages.append(user_msg)
                self.call_from_thread(self._refresh_tab_label)
                status(f"agent ({session.mode} mode): working…")
                sink = Sink("**Agent:**\n\n")

                # realtime file preview: one widget per streaming tool-call index
                live_widgets = {}
                live_last = {}
                live_args = {}
                live_lock = threading.Lock()

                # one section per tool call, so every command shows up as its own block
                pending_sections = []
                section_lock = threading.Lock()

                # live "Thinking" panel: reasoning tokens are shown while they stream
                thought = {"node": None, "buf": [], "last": 0.0}

                def on_reasoning(chunk):
                    thought["buf"].append(chunk)
                    now = time.monotonic()
                    if now - thought["last"] < 0.2:  # 5 FPS instead of 10
                        return
                    thought["last"] = now
                    text = "".join(thought["buf"])
                    if thought["node"] is None:
                        thought["node"] = self.call_from_thread(self._open_thought)
                    self.call_later(self._update_thought, thought["node"], text)

                def on_tool_delta(idx, name, args_text):
                    if name not in FILE_TOOLS:
                        return
                    now = time.monotonic()
                    with live_lock:
                        last = live_last.get(idx, 0.0)
                        if now - last < 0.1:  # 10 FPS instead of 20
                            return
                        live_last[idx] = now
                        live_args[idx] = (name, args_text)
                        node = live_widgets.get(idx)
                    if node is None:
                        node = self.call_from_thread(self._open_file_stream)
                        with live_lock:
                            live_widgets[idx] = node
                    text = live_file_text(name, args_text, live=True, max_lines=16)
                    if text is not None:
                        self.call_later(self._update_file_widget, node, text)

                def _finalize_all(name, args, is_start):
                    """Point the newest live widget (or any pending index) at the final diff."""
                    with live_lock:
                        idx = None
                        # tool start/end carry no index; use the most recent widget
                        if live_widgets:
                            idx = next(reversed(live_widgets))
                            node = live_widgets.pop(idx)
                            live_last.pop(idx, None)
                            live_args.pop(idx, None)
                    if idx is None:
                        return None
                    text = self._final_file_text(name, args)
                    self.call_from_thread(self._update_file_widget, node, text)
                    return node

                def on_tool_start(name, args):
                    status(f"● {tool_header(name, args)}")
                    if name in FILE_TOOLS:
                        if not _finalize_all(name, args, True):
                            # no live widget (non-streaming provider): show the diff directly
                            text = self._final_file_text(name, args)
                            self.call_from_thread(self._update_file_widget,
                                                  self.call_from_thread(self._open_file_stream), text)
                        return
                    # every non-file tool gets its own visible section (full command string for shells)
                    node = self.call_from_thread(self._open_tool_section, name, args)
                    with section_lock:
                        pending_sections.append({"name": name, "args": args, "node": node, "done": False,
                                                 "t0": time.monotonic()})

                def on_tool_end(name, args, result):
                    status(f"  ⎿  {tool_summary(name, args, result)}")
                    if name in FILE_TOOLS:
                        return
                    with section_lock:
                        hit = next((s for s in pending_sections if not s["done"] and s["name"] == name), None)
                        if hit:
                            hit["done"] = True
                    if hit:
                        elapsed = time.monotonic() - hit["t0"]
                        self.call_from_thread(self._finish_tool_section, hit["node"], name, args, result, elapsed)

                self.cancel.clear()

                def loop():
                    return run_loop(messages, verbose=False, on_text=sink, on_status=status, on_ask=on_ask,
                                    on_permission=on_permission, session=session, cancel=self.cancel,
                                    on_tool_start=on_tool_start, on_tool_end=on_tool_end,
                                    on_tool_delta=on_tool_delta, on_reasoning=on_reasoning,
                                    profile_name=judge_alias if mode == "ensemble" else single_alias)

                loop()
                while session.mode == "plan":
                    choice = on_ask("Execute this plan?", PLAN_OPTIONS)
                    if choice.startswith("Yes"):
                        session.mode = "auto" if "auto-accept" in choice else "default"
                        self.call_from_thread(self._log_title)
                        status(f"mode → {session.mode}")
                        messages.append({"role": "user", "content": "The plan is approved. Execute it now, then verify the result."})
                        loop()
                        break
                    if choice == PLAN_OPTIONS[2]:
                        break
                    messages.append({"role": "user", "content": f"Revise the plan: {choice}"})
                    loop()
                try:
                    sessions.save(session.id, messages, session.todos, mode=session.mode,
                                  thinking=session.thinking, profile=single_alias,
                                  title=self._renamed_title)
                except OSError:
                    pass
                # the tab title follows the first prompt of this session
                self.call_from_thread(self._refresh_tab_label)
                self.call_from_thread(self._stash_tab)
            elif mode == "ensemble":
                started = {}
                def ens_text(chunk):
                    if "sink" not in started:
                        started["sink"] = Sink("**Final Answer (ensemble):**\n\n")
                    started["sink"](chunk)
                try:
                    _, _, _, meta = ensemble_ask(prompt, image_src=image_src, judge_alias=judge_alias,
                                                 verbose=False, on_text=ens_text, on_status=status)
                finally:
                    sink = started.get("sink")
            else:
                if single_alias == "paligemma":
                    from .agent import ask_once
                    out = ask_once(prompt, profile_name=single_alias, image_url=image_src, stream=False)
                    sink = Sink(f"**{single_alias}:**\n\n")
                    sink(out)
                else:
                    from .ensemble import _load_image_b64
                    _, prof = get_active_profile(single_alias)
                    if image_src and not prof.get("vision"):
                        status(f"{single_alias} has no vision; /model super for images")
                    content = prompt
                    if image_src:
                        content = [{"type": "text", "text": prompt},
                                   {"type": "image_url", "image_url": {"url": _load_image_b64(image_src)}}]
                    sink = Sink(f"**{single_alias}:**\n\n")
                    thought = {"node": None, "buf": [], "last": 0.0}
                    def on_reasoning(chunk):
                        thought["buf"].append(chunk)
                        now = time.monotonic()
                        if now - thought["last"] < 0.1:
                            return
                        thought["last"] = now
                        if thought["node"] is None:
                            thought["node"] = self.call_from_thread(self._open_thought)
                        self.call_from_thread(self._update_thought, thought["node"],
                                              "".join(thought["buf"]))
                    llm.chat(prof, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
                             on_text=sink, on_reasoning=on_reasoning)
        except Exception as e:
            self.call_from_thread(self._append_plain, f"Error: {e}")
        finally:
            # settle any live file panels left mid-stream (e.g. Esc mid-send)
            try:
                pending = locals().get("live_widgets")
                if pending:
                    with locals()["live_lock"]:
                        items = list(pending.items())
                        pending.clear()
                        args_map = dict(locals().get("live_args") or {})
                    for idx, node in items:
                        name, args_text = args_map.get(idx, ("write_file", ""))
                        text = live_file_text(name, args_text, live=False, max_lines=16)
                        if text is not None:
                            self.call_from_thread(self._update_file_widget, node, text)
            except Exception:
                pass
            if sink is not None:
                try:
                    sink.close()
                except Exception:
                    pass
                self._sink = None
            # settle any tool sections left in "running…" (e.g. Esc mid-call)
            try:
                stuck = locals().get("pending_sections")
                if stuck:
                    with locals()["section_lock"]:
                        unfinished = [s for s in stuck if not s["done"]]
                        stuck.clear()
                    for s in unfinished:
                        self.call_from_thread(self._finish_tool_section, s["node"], s["name"],
                                              s.get("args") or {}, "interrupted", time.monotonic() - s["t0"])
            except Exception:
                pass
            self.call_from_thread(lambda: setattr(self, "busy", False))
            self.call_from_thread(self._update_statusline)
            # opencode-style: queued prompts run automatically after the current turn
            if self._queue_processing:
                self.call_from_thread(self._process_next_in_queue)
            elif self.message_queue.has_pending():
                self._queue_processing = True
                self.call_from_thread(self._process_next_in_queue)

def run_tui(initial_path: str | None = None, judge_alias: str | None = None,
            model: str | None = None, agent_mode: str = "default", resume: str | None = None):
    app = ZeusaiApp(initial_path=initial_path, judge_alias=judge_alias, model=model,
                    agent_mode=agent_mode, resume=resume)
    app.run()
    # Tell the user how to come back to this exact conversation.
    try:
        from rich.console import Console
        console = Console()
        app._stash_tab()
        for i, state in enumerate(app.tabs_state, 1):
            if any(m.get("role") == "user" for m in state["messages"]):
                title = sessions.tab_title(state["messages"], fallback="session")
                console.print(f"[dim]session {i}: {state['id']} · {title} · "
                              f"resume with `zeusai tui --resume {state['id']}`[/dim]")
    except Exception:
        pass
