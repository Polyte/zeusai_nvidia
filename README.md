# zeusai

**zeusai** is a CLI coding agent. Every prompt is fanned out to **all configured models in parallel**, then a judge scores and merges the best answer by weight/accuracy — so you get one consolidated response.

## Models (ensemble)

| Alias | Model ID | Special |
|-------|----------|---------|
| `laguna` | `poolside/laguna-xs-2.1` | Fast, coding |
| `ultra` | `nvidia/nemotron-3-ultra-550b-a55b` | Deep reasoning, `enable_thinking`, default judge |
| `super` | `nvidia/nemotron-3.5-super-vl-120b-a12b` | Vision + thinking `enable_thinking` |
| `glimmer` | `meta/muse-glimmer-30b` | `reasoning_effort=high`, math/logic |
| `paligemma` | `google/paligemma` | VLM on `https://ai.api.nvidia.com/v1/vlm/google/paligemma`, base64 image (needs `--image`) |

All use the same base theme: `https://integrate.api.nvidia.com/v1` (except paligemma VLM). Presentation name is **zeusai** — internal model IDs are hidden in normal output, use `zeusai models` to see them.

## Install (this repo)

```bash
uv venv .venv
uv pip install -e .
# or
pip install -e .
chmod +x install.sh
./install.sh   # symlinks ~/.local/bin/zeusai
```

## Configure

Each profile has its own key (`~/.zeusai/config.json`, mode 600):

```bash
zeusai configure --profile laguna --api-key nvapi-...
zeusai configure --profile ultra --api-key nvapi-...
zeusai configure --profile super --api-key nvapi-...
zeusai configure --profile glimmer --api-key nvapi-...   # same as laguna
zeusai configure --profile paligemma --api-key nvapi-...
zeusai use auto       # default: one model answers each prompt
zeusai doctor         # test all 5
zeusai show-config
```

## Usage

opencode-style map:

| opencode | zeusai |
|----------|--------|
| `opencode` | `zeusai` (interactive ensemble) |
| `opencode ~/code/my-project` | `zeusai ~/code/my-project` |
| `opencode mini` | `zeusai mini [path] [--model laguna]` (light single-model REPL) |
| `opencode run "explain this repo"` | `zeusai run "explain this repo"` (automation, tools) |
| `opencode --help` | `zeusai --help` |

```bash
zeusai                          # interactive ensemble chat (opencode TUI equivalent, line-based)
zeusai ~/code/my-project        # same, in that directory
zeusai mini ~/code/my-project   # light REPL (laguna, opencode mini equivalent)
zeusai chat -v
zeusai chat --model ultra       # single-model chat
zeusai run "fix bug in app.py" --ensemble -v    # ensemble plan + tool execution (judge=ultra, opencode run)
zeusai run "fix bug" --model laguna -v          # single-model with tools
zeusai run --path ~/proj "fix bug"               # opencode run in dir
zeusai ask "Which number is larger, 9.11 or 9.8?" --ensemble
zeusai ask "Write a Python reverse fn" --model ultra --stream
zeusai ensemble "Which number is larger, 9.11 or 9.8? Explain." --raw   # show all drafts
zeusai ensemble "What does this sign say?" --image https://.../example1a.jpg
zeusai ask "Describe the image" --model paligemma --image /path/to/img.jpg
zeusai --help
```

### How ensemble works

1. **Detect task**: code / reasoning / vision / creative / general → weights (e.g. code: laguna 5, ultra 4, glimmer 3…)
2. **Fan-out in parallel** (ThreadPool) to all applicable models. `paligemma` only if `--image` given.
3. **Judge merge**: `ultra` (or `--judge`) scores each 0-10 on accuracy/completeness/correctness, then merges into one best answer (preserving code blocks, preferring weighted consensus + logical checks).
4. **Agent run**: `zeusai run --ensemble` uses the merged plan as system context, then the judge model executes tools (`read_file`, `write_file`, `list_files`, `run_shell`, `search_code`, `edit_file`).

### Interactive commands

Ensemble chat:
```
zeusai-ensemble> Which is larger, 9.11 or 9.8?
zeusai-ensemble> img:https://assets...jpg Read the sign
zeusai-ensemble> /help /models /judge ultra /quit
```

Single chat:
```
zeusai> fix the login bug
zeusai> /help /tools /dir /clear /quit
```

### Examples

```bash
# All five vote, judge merges
zeusai ensemble "Write a limerick about GPU computing"

# Vision ensemble: super + paligemma lead, others provide context
zeusai ensemble "Read the traffic sign, color?" \
  --image https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1a.jpg

# Paligemma VLM directly (base64 image, special transport)
zeusai ask "Describe the image" --model paligemma --image /tmp/sign.jpg

# Glimmer high-reasoning
zeusai ask "Which number is larger, 9.11 or 9.8?" --model glimmer

# Ultra with thinking (stream shows reasoning_content first)
zeusai ask "Write a limerick about GPU computing." --model ultra --stream

# Super vision + thinking via image_url dict
zeusai ask "What does it say?" --model super \
  --image https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1a.jpg

# Coding agent with ensemble plan
zeusai run "list files, find TODOs, fix them, run tests" --ensemble -v
```

## Modes and permissions

| Mode | What it does |
|------|--------------|
| `default` | Reads freely; asks before every file write/edit and shell command |
| `auto` | Runs every tool without asking (deny rules still apply) |
| `plan` | Read-only. Explores, asks questions, writes a plan, then asks you to approve it. Approving switches to `auto` or `default` and executes |
| `research` | Searches the web (DuckDuckGo, no key), reads 2-4 sources, answers with `[1]` citations and a Sources list |

```bash
zeusai run --mode plan "add input validation to the API"
zeusai run --mode auto "fix the failing tests"
zeusai research "what changed in the latest Django release?"
zeusai run --enhance "fix tests"      # rewrite the prompt first, then pick which version to send
```

In `chat`: `/mode [name]` (no name = cycle), `/enhance`, `/permissions`. In the TUI: **Shift+Tab** cycles modes, **Ctrl+E** toggles enhance-prompt.

**Permission prompts** show the command or a diff, then let you pick *Yes*, *Yes, and don't ask again for `Rule`* (saved to `.zeusai/settings.json`), *No*, or type an instruction instead. Rules can also be written by hand in `.zeusai/settings.json` (project) or `~/.zeusai/config.json` (user):

```json
"permissions": {
  "allow": ["Bash(npm test*)", "Bash(git status*)", "Update(src/*)"],
  "deny":  ["Bash(rm -rf*)", "Write(.env)"],
  "ask":   ["Bash(git push*)"]
}
```

**Destructive commands always ask**, even in `auto` mode or under a broad allow rule like `Bash`: recursive deletes of `/`, `~` or `.`, `mkfs`, `dd of=/dev/…`, `git push --force`, `git reset --hard`, `git clean -f`, `curl … | sh`, `DROP TABLE`, shutdown/reboot. Only an exact allow rule for that command (e.g. `Bash(git reset --hard)`) skips the prompt; headless runs refuse them otherwise.

Labels: `Read`, `Write`, `Update`, `List`, `Search`, `Bash`, `WebSearch`, `WebFetch`. Precedence: deny > ask > allow > mode default. When no one is there to answer (piped stdin), anything that would ask is denied; use `--mode auto` or allow rules for automation.

**Questions from the AI:** when the agent needs a decision it calls `ask_user` and you get a picker: arrow keys + Enter, Space for multi-select, a number to jump, *Other* to type your own answer, Esc to skip.

## Sessions, context and checkpoints

Every conversation is saved to `~/.zeusai/sessions/<dir>/` after each prompt.

```bash
zeusai -c                        # continue the most recent conversation here (line chat)
zeusai chat -r 20261108-0538     # resume by id (prefix ok)
zeusai sessions                  # list saved conversations for this directory
zeusai run -c "now add tests"    # one-shot, continuing the last conversation
```

- **Auto-compaction:** when history passes ~240k chars (`ZEUSAI_COMPACT_AT`), older messages are summarized and the last 8 kept verbatim. `/compact [focus]` does it on demand.
- **Todos:** the agent tracks multi-step work with `todo_write`; `/todos` shows the list.
- **Rewind:** each prompt checkpoints the files it changes. `/rewind [n]` restores code and/or conversation.

## Tools

`read_file` (line-numbered, `offset`/`limit`; images are shown to vision models or described by `super` for text-only ones; notebooks render cells + outputs), `notebook_edit`, `write_file`, `edit_file` (`replace_all`), `multi_edit`, `list_files`, `glob_files`, `search_code` (ripgrep), `run_shell` (`timeout`, `run_in_background`) + `shell_output` / `shell_kill`, `web_search`, `web_fetch`, `memory`, `todo_write`, `ask_user`, `task` (subagents), and any MCP server tools (`mcp__<server>__<tool>`).
Files must be read before they are edited or overwritten.

## Subagents

The `task` tool runs a job in a fresh context and returns only the report; several run in parallel.
Built-in: `explore` (read-only search) and `general`. Add your own in `.zeusai/agents/<name>.md` or `~/.zeusai/agents/`:

```markdown
---
name: test-runner
description: Runs the tests and explains failures
tools: read_file, search_code, glob_files, run_shell
model: laguna
---
You run the test suite and report each failure with its likely cause.
```

`zeusai agents` / `/agents` lists them. `/agents new <what it should do>` has the model write one for you (least-privilege tools), then you can edit it.

## Custom slash commands

`.zeusai/commands/review.md` becomes `/review` (subfolders namespace: `git/commit.md` → `/git:commit`).
`$ARGUMENTS`, `$1`, `$2` are substituted, and `` !`cmd` `` is replaced by the command's output.

## Hooks

Shell commands that run on agent events, in `.zeusai/settings.json` or `~/.zeusai/config.json`:

```json
"hooks": {
  "PreToolUse":  [{"matcher": "Bash", "command": "./scripts/guard.sh"}],
  "PostToolUse": [{"matcher": "Update|Write", "command": "ruff format -q $(jq -r .args.path)"}],
  "UserPromptSubmit": [{"command": "git status --short"}],
  "Stop": [{"command": "pytest -q >/dev/null 2>&1 || { echo 'tests fail' >&2; exit 2; }"}]
}
```

Hooks get the event as JSON on stdin. Exit code 2 blocks: PreToolUse skips the tool, PostToolUse/Stop send stderr back to the model, UserPromptSubmit rejects the prompt. UserPromptSubmit stdout is added as context.

## MCP servers

```bash
zeusai mcp add fs -- npx -y @modelcontextprotocol/server-filesystem .
zeusai mcp list
zeusai mcp remove fs
```

Remote (streamable HTTP) servers: `zeusai mcp add docs https://example.com/mcp -H 'Authorization: Bearer ${DOCS_TOKEN}'`.

Or in settings: `"mcpServers": {"fs": {"command": ..., "args": [...], "env": {...}}, "docs": {"url": "https://…/mcp", "headers": {...}}}` (`${VAR}` expands from the environment). Tools need approval in `default` mode (rule label `Mcp`, e.g. `Mcp(fs:read_file)`).

## Headless / scripting

```bash
zeusai -p "summarize this repo"                       # prints the final answer
git diff | zeusai -p "review this diff" --output-format json
zeusai -p "fix the tests" --mode auto --output-format stream-json   # one JSON event per line
```

Flags: `--allowed-tools Read,Glob,Search` / `--disallowed-tools Bash,Write` (names or labels), `--system-prompt`, `--append-system-prompt`, `--think/--no-think`, `--max-steps`, `-c` / `-r <id>`.

`--style <name>` sets the output style.

`json` gives `{result, is_error, session_id, num_tool_calls, duration_ms, usage}`. Anything needing approval is denied unless `--mode auto` or allow rules cover it.

## Context the agent gets automatically

- Instruction files from the repo root down to the current directory: `ZEUSAI.md`, `.zeusai/ZEUSAI.md`, `AGENTS.md`, `CLAUDE.md`, plus your personal, uncommitted `ZEUSAI.local.md`; and `~/.zeusai/ZEUSAI.md`.
- Git snapshot: branch, `git status --short`, last 5 commits (refreshed every 30s).
- Today's date, working directory, mode, todo list and learned memories.

## Output styles

`/style` lists them, `/style <name>` switches: `default`, `concise` (results only), `explanatory` (adds "Insight:" notes on why), `learning` (leaves small `TODO(human)` pieces for you to write). Add your own in `.zeusai/output-styles/<name>.md`; set a default with `"outputStyle": "concise"` in settings.

## Built-in commands

`/review [ref|PR#]` reviews changes for bugs · `/commit` writes a commit · `/pr` pushes and opens a PR with `gh`. Override any of them with a file of the same name in `.zeusai/commands/`.

## In-chat extras

↑/↓ history (saved in `~/.zeusai/history`) · Tab completes `/commands` · multi-line input: end a line with `\` or wrap a block in `"""` · `/think [on|off]` extended reasoning · `/export [file.md]` · `/status` · `/bashes` · a terminal bell when a turn takes over 20s (`ZEUSAI_BELL_AFTER`) · `@path` attaches a file or directory · status line above the prompt (`"statusLine": {"command": ...}` in settings; gets JSON with model/mode/usage/todos on stdin) · `!cmd` runs a shell command and puts the output in the conversation · `/cost` token usage · `/mcp` `/hooks` `/config` · `Esc` stops the agent (TUI) · `/image <path|url>` in the TUI attaches an image to the next agent prompt (described by `super` if the agent model is text-only).

## Tests

```bash
uv pip install -e '.[dev]'
.venv/bin/python -m pytest -q     # offline: the model is mocked; MCP tests use local stub servers
```

## Project layout

```
zeusai/
  zeusai/
    cli.py        # typer CLI (chat/run/ask/ensemble/models/use/doctor)
    agent.py      # tool loop + ensemble chat
    ensemble.py   # fan-out, weighting, judge merge, paligemma transport
    tools.py      # built-in tools (read/write/edit/multi_edit/glob/search/shell/background shells/web)
    config.py     # 5 profiles, keys, default (ensemble)
    sessions.py   # saved conversations, compaction, todos, checkpoints/rewind
    agents.py     # task tool: subagents (explore/general/custom)
    hooks.py      # PreToolUse/PostToolUse/UserPromptSubmit/Stop hooks
    commands.py   # custom slash commands
    mentions.py   # @path attachments
    mcp.py        # MCP stdio client
    statusline.py # status line (default or custom command)
    permissions.py, modes.py, memory.py, llm.py, tui.py, ui.py, web.py
  tests/          # pytest suite (tools, sessions, agent loop, subagents, hooks, MCP, ...)
  pyproject.toml
  requirements.txt
  README.md
  install.sh
  hello.py        # example file created by run --ensemble demo
```

## Speed

- Streams tokens everywhere (CLI, REPL, TUI); `ask`/`ensemble` stream by default (`--no-stream` to disable).
- No `openai` SDK: a stdlib client (`zeusai/llm.py`) with pooled keep-alive HTTPS connections. Startup is about 0.1s (was about 2s).
- Independent read-only tool calls (`read_file`, `list_files`, `search_code`) in one turn run in parallel. `search_code` uses ripgrep when it's installed.
- Default mode is `auto`: one model per session, one round trip per prompt, no judge (~1.5s vs ~6s for ensemble). Sessions (`zeusai`, `chat`, `mini`) lock to `ultra` (10/10 on a 5-task coding eval with hidden tests, vs laguna 6/10 and super 10/10 but slower); one-shot `ask --image` goes to `super`. Switch in-session with `/model <alias>`; Multiple models answer only when asked per prompt: `zeusai ensemble "…"`, `--ensemble`, or `/ensemble` in the TUI. `zeusai use ensemble` is refused.
- Any OpenAI-compatible endpoint can be a profile, including keyless local servers: `zeusai configure --profile local --base-url http://localhost:11434/v1 --model-id <model> --no-key`.
- Ensemble knobs live in `~/.zeusai/config.json`:

```json
"ensemble": {
  "draft_thinking": false,
  "draft_max_tokens": 2048,
  "grace_seconds": 8,
  "judge_thinking": false,
  "skip_judge_if_single": true
}
```

`grace_seconds` caps how long to wait for slower models after the first good draft (the actual wait adapts to how fast that first draft arrived). Models that miss the window are skipped. Turn `draft_thinking`/`judge_thinking` back on to trade speed for depth.

## Notes

- `super`/`ultra` use `extra_body={"chat_template_kwargs":{"enable_thinking":true}}`
- `glimmer` uses `extra_body={"reasoning_effort":"high"}`
- `paligemma` uses raw `POST https://ai.api.nvidia.com/v1/vlm/google/paligemma` with `image_url` as **string** (`data:image/jpeg;base64,...`), not dict, and stream `true` → `text/event-stream`. URL images are downloaded → base64 locally.
- Keys are masked in `show-config`/`models` output.
# zeusai_nvidia
