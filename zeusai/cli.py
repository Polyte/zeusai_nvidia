import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import (
    ensemble_judge,
    get_active_profile, get_profiles, get_default_profile_name,
    save_profile, set_default, get_config_path, DISPLAY_MODEL, list_profiles
)
from .agent import start_chat, run_streaming, ask_once, start_ensemble_chat

app = typer.Typer(name="zeusai", help="zeusai - CLI coding agent (single + ensemble). opencode-style: `zeusai` = interactive, `zeusai mini` = light REPL, `zeusai run \"prompt\"` = automation.", add_completion=False)
console = Console()

def _prewarm(model=None):
    """Start the TCP+TLS handshake now, overlapping it with prompt preparation."""
    try:
        from . import llm
        from .config import get_active_profile
        llm.prewarm(get_active_profile(model)[1]["base_url"])
    except BaseException:
        pass  # a missing key is reported properly later


def _check_mode(mode):
    from .permissions import MODES
    if mode not in MODES:
        console.print(f"[red]Unknown mode {mode}. Modes: {', '.join(MODES)}[/red]")
        raise typer.Exit(1)
    return mode

@app.command()
def research(question: str = typer.Argument(..., help="Question to research on the web"), verbose: bool = typer.Option(False, "--verbose", "-v"), model: str = typer.Option(None, "--model", "-m")):
    """Research a question on the web and answer with cited sources (run --mode research)."""
    import sys
    from .modes import Session
    try:
        run_streaming([{"role": "user", "content": question}], verbose=verbose, profile_name=model,
                      interactive=sys.stdin.isatty(), session=Session(mode="research"))
    except SystemExit as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

@app.command()
def version():
    console.print("zeusai 2.0.0")

@app.command()
def models():
    """List available models/profiles"""
    rows = list_profiles()
    table = Table(title="zeusai models", show_header=True)
    table.add_column("Alias")
    table.add_column("Model")
    table.add_column("Key")
    table.add_column("Thinking")
    table.add_column("Vision")
    for r in rows:
        mark = " *default*" if r["is_default"] else ""
        table.add_row(f"{r['alias']}{mark}", r["model_id"], r["masked_key"], str(r["thinking"]), str(r["vision"]))
    console.print(table)
    console.print("[dim]Switch default: zeusai use <alias> | Ensemble: zeusai ensemble \"prompt\"[/dim]")

@app.command()
def use(alias: str = typer.Argument(..., help="auto | laguna | ultra | super | glimmer | paligemma | ensemble")):
    """Set default model (auto = one fast model per prompt, ensemble = fan-out to all + judge)"""
    if alias == "auto":
        set_default("auto")
        console.print("[green]✓[/green] Default model set to auto (one fast model per prompt; images go to the vision model)")
        return
    if alias == "ensemble":
        console.print("[yellow]Ensemble can't be the default: one model answers each prompt.[/yellow] Use it per prompt: zeusai ensemble \"...\" or --ensemble")
        raise typer.Exit(1)
    profiles = get_profiles()
    if alias not in profiles:
        console.print(f"[red]Unknown profile {alias}. Available: {', '.join(list(profiles.keys()) + ['auto', 'ensemble'])}[/red]")
        raise typer.Exit(1)
    set_default(alias)
    console.print(f"[green]✓[/green] Default model set to {alias}")

@app.command()
def configure(api_key: str = typer.Option(None, "--api-key"), profile: str = typer.Option(None, "--profile", help="laguna|ultra|super|glimmer|paligemma or a new name"), model_id: str = typer.Option(None, "--model-id"), base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1 for Ollama"), no_key: bool = typer.Option(False, "--no-key", help="Endpoint needs no API key (local servers)")):
    """Configure API key (or a custom/local endpoint) for a profile"""
    if not profile:
        # default to default profile if not ensemble, else laguna
        d = get_default_profile_name()
        profile = "laguna" if d in ("auto", "ensemble") else d
    if not api_key and not no_key:
        profiles = get_profiles()
        cur = profiles.get(profile, {})
        if not cur.get("api_key") and not cur.get("no_key"):
            api_key = typer.prompt(f"API key for {profile}", hide_input=True)
    if api_key or model_id or base_url or no_key:
        save_profile(profile, api_key=api_key, model_id=model_id, base_url=base_url, no_key=no_key or None)
        console.print(f"[green]✓[/green] Saved {profile} to {get_config_path()}")
    show_config()

@app.command()
def show_config():
    """Show config (keys masked)"""
    rows = list_profiles()
    table = Table(title="zeusai config", show_header=True)
    table.add_column("Profile")
    table.add_column("Model ID")
    table.add_column("API Key")
    for r in rows:
        table.add_row(f"{r['alias']} {'✓' if r['is_default'] else ''}", r["model_id"], r["masked_key"])
    console.print(table)
    console.print(f"Config file: [bold]{get_config_path()}[/bold]")

@app.command()
def tui(path: str = typer.Argument(None, help="Directory to work in (opencode-style)."), judge: str = typer.Option(None, "--judge"), model: str = typer.Option(None, "--model", "-m", help="Single model instead of ensemble"), mode: str = typer.Option("default", "--mode", help="default | auto | fast | thinking | executor | plan | research"), cont: bool = typer.Option(False, "--continue", "-c", help="Continue the most recent conversation in this directory"), resume: str = typer.Option(None, "--resume", "-r", help="Resume a saved conversation by id (see `zeusai sessions`)")):
    """Full-screen TUI (like `opencode`).

    Examples:
      zeusai tui
      zeusai tui ~/code/my-project
      zeusai tui -c          # continue the latest session (full history + todos)
      zeusai tui -r <id>     # resume a specific session
      zeusai tui --mode fast # start in fast execution mode
    """
    import os, pathlib
    if path:
        p = pathlib.Path(path).expanduser()
        if p.is_dir():
            os.chdir(p)
        else:
            console.print(f"[red]Not a directory: {path}[/red]")
            raise typer.Exit(1)
    from .tui import run_tui
    try:
        run_tui(initial_path=None, judge_alias=judge, model=model, agent_mode=_check_mode(mode),
                resume=resume if resume else ("" if cont else None))
    except KeyboardInterrupt:
        pass

@app.command()
def mini(path: str = typer.Argument(None, help="Directory to work in (like `opencode ~/code/my-project`)"), verbose: bool = typer.Option(False, "--verbose", "-v"), model: str = typer.Option(None, "--model", "-m", help="laguna|ultra|super|glimmer|paligemma|ensemble"), mode: str = typer.Option("default", "--mode", help="default | auto | fast | thinking | executor | plan | research"), cont: bool = typer.Option(False, "--continue", "-c", help="Continue the most recent conversation in this directory"), resume: str = typer.Option(None, "--resume", "-r", help="Resume a saved conversation by id (see `zeusai sessions`)")):
    """Minimal interactive REPL (like `opencode mini`).

    Examples:
      zeusai mini
      zeusai mini ~/code/my-project
      zeusai mini --model laguna
      zeusai mini --mode executor
    """
    import os, pathlib
    if path:
        p = pathlib.Path(path).expanduser()
        if p.is_dir():
            os.chdir(p)
        else:
            console.print(f"[red]Not a directory: {path}[/red]")
            raise typer.Exit(1)
    # mini = light single-model chat by default (fast), falls back to ensemble if requested
    if model == "ensemble":
        start_ensemble_chat(verbose=verbose)
        return
    resume_arg = resume if resume else ("" if cont else None)
    try:
        start_chat(verbose=verbose, profile_name=model, mode=_check_mode(mode), resume=resume_arg)
    except SystemExit as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

@app.command()
def chat(add_dir: list[str] = typer.Option([], "--add-dir", help="Extra directory the agent may use (repeatable)"), mode: str = typer.Option("default", "--mode", help="default (ask before edits) | auto | fast | thinking | executor | plan | research"), enhance: bool = typer.Option(False, "--enhance", help="Rewrite the prompt into a clearer one first"), verbose: bool = typer.Option(False, "--verbose", "-v"), model: str = typer.Option(None, "--model", "-m", help="laguna|ultra|super|glimmer|paligemma|ensemble"), ensemble: bool = typer.Option(False, "--ensemble", help="Fan-out to all models + judge"), cont: bool = typer.Option(False, "--continue", "-c", help="Continue the most recent conversation in this directory"), resume: str = typer.Option(None, "--resume", "-r", help="Resume a saved conversation by id (see `zeusai sessions`)")):
    """Interactive chat (opencode-style: `zeusai` / `zeusai chat` = full interface)"""
    use_ens = ensemble or model == "ensemble"
    if use_ens:
        # `--ensemble --model ultra` means ultra judges
        start_ensemble_chat(verbose=verbose, judge_alias=model if model not in (None, "ensemble") else None)
        return
    start_chat(verbose=verbose, profile_name=model, mode=_check_mode(mode), enhance=enhance,
               resume=resume if resume else ("" if cont else None), dirs=add_dir)

@app.command()
def run(task: str = typer.Argument(..., help="Task prompt (like `opencode run \"explain this repo\"`)"), add_dir: list[str] = typer.Option([], "--add-dir", help="Extra directory the agent may use (repeatable)"), mode: str = typer.Option("default", "--mode", help="default (ask before edits) | auto | fast | thinking | executor | plan | research"), enhance: bool = typer.Option(False, "--enhance", help="Rewrite the prompt into a clearer one first"), verbose: bool = typer.Option(False, "--verbose", "-v"), model: str = typer.Option(None, "--model", "-m", help="laguna|ultra|super|glimmer|paligemma|ensemble"), ensemble: bool = typer.Option(False, "--ensemble", help="Ensemble drafts then tool execution"), judge: str = typer.Option(None, "--judge", help="Judge model for ensemble"), models_str: str = typer.Option(None, "--models", help="Comma list for ensemble, e.g. laguna,ultra"), path: str = typer.Option(None, "--path", help="Work directory (like `opencode <path>`). If set, cd there first."), cont: bool = typer.Option(False, "--continue", "-c", help="Continue the most recent conversation in this directory"), resume: str = typer.Option(None, "--resume", "-r", help="Resume a saved conversation by id (see `zeusai sessions`)")):
    """One-shot agent task (uses tools) — opencode `run` equivalent. With --ensemble (or default ensemble), gather drafts then execute with judge."""
    import os, pathlib
    if path:
        p = pathlib.Path(path).expanduser()
        if p.is_dir():
            os.chdir(p)
        else:
            console.print(f"[red]Not a directory: {path}[/red]")
            raise typer.Exit(1)
    use_ens = ensemble or model == "ensemble"
    if not use_ens and model in (None, "ensemble"):
        # if default is single, don't force ensemble; only when flag set
        pass
    # Actually: only use_ens if explicitly requested or default==ensemble AND no explicit single model
    if use_ens:
        from .ensemble import ensemble_ask
        judge = ensemble_judge(judge)
        try:
            # first get ensemble plan (no tools) - brief drafts
            console.print("[bold]zeusai ensemble[/bold] gathering drafts...")
            merged_plan, _, _, _ = ensemble_ask(
                f"Plan how to complete this task (be concise, list steps and files):\n{task}",
                models=models_str, judge_alias=judge, verbose=True, stream=True, header="Plan:"
            )
            console.print(f"[dim]Executing with {judge}...[/dim]")
            messages = [
                {"role": "system", "content": f"Task: {task}\n\nEnsemble expert plan (use this, verify with tools):\n{merged_plan}\n\nNow execute the task using tools. Be concise."},
                {"role": "user", "content": task}
            ]
            run_streaming(messages, verbose=verbose, profile_name=judge)
            return
        except SystemExit as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    import sys
    from . import sessions
    from .agent import maybe_enhance
    from .modes import Session, apply_mode_defaults
    from .permissions import MODES as _MODES
    mode = _check_mode(mode)
    interactive = sys.stdin.isatty()
    session = Session(mode=mode, enhance=enhance, id=sessions.new_id(),
                      dirs=[str(pathlib.Path(d).expanduser().resolve()) for d in add_dir])
    messages = []
    resume_id = resume if resume else (None if not cont else "")
    if resume_id is not None or cont:
        saved = sessions.load(resume_id or None)
        if saved:
            messages, session.id = saved["messages"], saved["id"]
            session.todos[:] = saved.get("todos", [])
            # Restore the setup from when the user quit; an explicit --mode/--model still wins.
            if mode == "default" and saved.get("mode") in _MODES:
                session.mode = saved["mode"]
            if saved.get("thinking") is not None and session.thinking is None:
                session.thinking = saved["thinking"]
            if model in (None, "auto", "ensemble") and saved.get("profile"):
                model = saved["profile"]
            console.print(f"[dim]continuing {saved['id']}: {saved.get('title', '')} (mode {session.mode})[/dim]")
        else:
            console.print("[yellow]no saved conversation here; starting fresh[/yellow]")
    model = apply_mode_defaults(session, model)
    _prewarm(model)
    try:
        task = maybe_enhance(task, session, interactive=interactive)
        messages.append({"role": "user", "content": task})
        run_streaming(messages, verbose=verbose, profile_name=model, interactive=interactive, session=session)
    except SystemExit as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        try:
            from .agent import _profile
            sessions.save(session.id, messages, session.todos, mode=session.mode,
                          thinking=session.thinking, profile=_profile(model)[0])
        except (OSError, SystemExit):
            pass
    if any(m.get("role") == "user" for m in messages):
        console.print(f"[dim]session {session.id} · resume with: zeusai chat --resume {session.id} or zeusai chat -c[/dim]")

@app.command()
def ask(prompt: str = typer.Argument(..., help="Prompt"), model: str = typer.Option(None, "--model", "-m", help="laguna|ultra|super|glimmer|paligemma|ensemble"), image: str = typer.Option(None, "--image", help="Image URL or local path for vision"), stream: bool = typer.Option(True, "--stream/--no-stream", "-s/-S", help="Stream tokens live (default on)"), ensure: bool = typer.Option(False, "--ensemble", help="Fan-out ensemble"), judge: str = typer.Option(None, "--judge"), models_str: str = typer.Option(None, "--models")):
    """One-shot ask (no tools). With --ensemble, answers from all models merged."""
    use_ens = ensure or model == "ensemble"
    if use_ens:
        from .ensemble import ensemble_ask
        try:
            merged, _, _, _ = ensemble_ask(prompt, image_src=image, models=models_str, judge_alias=judge, verbose=True, stream=stream, header="Final Answer (ensemble):")
            if not stream:
                from rich.markdown import Markdown
                console.print("[bold]Final Answer (ensemble):[/bold]")
                console.print(Markdown(merged))
            return
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    # paligemma needs special handling (VLM endpoint)
    if model == "paligemma":
        from .ensemble import call_paligemma
        try:
            r = call_paligemma("paligemma", prompt, image, stream=stream)
            if r.get("error"):
                console.print(f"[red]Error: {r['error']}[/red]")
                raise typer.Exit(1)
            if not stream:
                from rich.markdown import Markdown
                console.print(Markdown(r["content"]))
            return
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    from .config import resolve_auto
    try:
        out = ask_once(prompt, profile_name=resolve_auto(model, has_image=bool(image)), image_url=image, stream=stream)
        if out and not stream:
            from rich.markdown import Markdown
            console.print(Markdown(out))
    except SystemExit as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def ensemble(prompt: str = typer.Argument(..., help="Prompt"), image: str = typer.Option(None, "--image", help="Image URL or local path"), judge: str = typer.Option(None, "--judge", help="Judge model"), models_str: str = typer.Option(None, "--models", help="Comma list, default all"), stream: bool = typer.Option(True, "--stream/--no-stream", "-s/-S"), raw: bool = typer.Option(False, "--raw", help="Show raw candidate answers")):
    """Ensemble: send prompt to all models, compile best answer by weight."""
    from .ensemble import ensemble_ask
    try:
        merged, results, task, meta = ensemble_ask(prompt, image_src=image, models=models_str, judge_alias=judge, verbose=True, stream=stream, header="Final Answer (ensemble):")
        if not stream:
            from rich.markdown import Markdown
            console.print("\n[bold]Final Answer (ensemble):[/bold]")
            console.print(Markdown(merged))
        if raw:
            console.print("\n[bold]Raw candidates:[/bold]")
            for r in results:
                console.print(f"\n### {r['alias']}\n{r.get('content') or r.get('error')}")
    except SystemExit as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@app.command("print", hidden=True)
def print_cmd(prompt: str = typer.Argument(None, help="Prompt; piped stdin is appended"),
              output_format: str = typer.Option("text", "--output-format", help="text | json | stream-json"),
              mode: str = typer.Option("default", "--mode", help="default (anything needing approval is denied) | auto | fast | thinking | executor | plan | research"),
              model: str = typer.Option(None, "--model", "-m"),
              cont: bool = typer.Option(False, "--continue", "-c"), resume: str = typer.Option(None, "--resume", "-r"),
              max_steps: int = typer.Option(40, "--max-steps"),
              allowed_tools: str = typer.Option(None, "--allowed-tools", help="Comma list of tools/labels to offer, e.g. Read,Glob,Search"),
              disallowed_tools: str = typer.Option(None, "--disallowed-tools", help="Comma list to remove, e.g. Bash,Write"),
              system_prompt: str = typer.Option(None, "--system-prompt", help="Replace the base system prompt"),
              append_system_prompt: str = typer.Option(None, "--append-system-prompt", help="Extra instructions appended to the system prompt"),
              think: bool = typer.Option(None, "--think/--no-think", help="Force extended reasoning on/off"),
              style: str = typer.Option("", "--style", help="Output style (default, concise, explanatory, learning, custom)"),
              add_dir: list[str] = typer.Option([], "--add-dir", help="Extra directory the agent may use (repeatable)")):
    """Headless mode for scripts: `zeusai -p "prompt" [--output-format json]`."""
    import json, sys, time
    from . import llm, sessions
    from .agent import prepare_prompt, run_loop
    from .modes import Session, apply_mode_defaults
    from .permissions import MODES as _MODES
    if output_format not in ("text", "json", "stream-json"):
        raise typer.BadParameter("--output-format must be text, json or stream-json")
    piped = "" if sys.stdin.isatty() else sys.stdin.read()
    text = "\n\n".join(x for x in (prompt, piped) if x and x.strip())
    if not text:
        raise typer.BadParameter("no prompt given (argument or stdin)")
    split = lambda v: [x.strip() for x in (v or "").split(",") if x.strip()]
    mode = _check_mode(mode)
    session = Session(mode=mode, id=sessions.new_id(), thinking=think,
                      allowed_tools=split(allowed_tools), disallowed_tools=split(disallowed_tools),
                      system_prompt=system_prompt or "", append_prompt=append_system_prompt or "", style=style,
                      dirs=[str(__import__("pathlib").Path(d).expanduser().resolve()) for d in add_dir])
    from .agent import session_start, session_end
    session_start(session, "resume" if (cont or resume) else "startup")
    messages = []
    saved = sessions.load(resume) if resume else sessions.load() if cont else None
    if saved:
        messages, session.id = saved["messages"], saved["id"]
        session.todos[:] = saved.get("todos", [])
        if mode == "default" and saved.get("mode") in _MODES:
            session.mode = saved["mode"]
        if saved.get("thinking") is not None and session.thinking is None:
            session.thinking = saved["thinking"]
        if model in (None, "auto") and saved.get("profile"):
            model = saved["profile"]
    model = apply_mode_defaults(session, model)
    _prewarm(model)
    text, _ = prepare_prompt(text)
    messages.append({"role": "user", "content": text})
    emit = (lambda **e: print(json.dumps(e), flush=True)) if output_format == "stream-json" else (lambda **e: None)
    emit(type="system", subtype="init", session_id=session.id, cwd=str(__import__("os").getcwd()), mode=session.mode)
    t0, steps, err = time.time(), {"n": 0}, None
    def start(name, args):
        steps["n"] += 1
        emit(type="tool_use", name=name, input=args)
    def end(name, args, result):
        emit(type="tool_result", name=name, content=result[:20000])
    try:
        result = run_loop(messages, max_steps=max_steps, profile_name=model, session=session,
                          on_tool_start=start, on_tool_end=end)
    except Exception as e:
        result, err = f"Error: {e}", str(e)
    finally:
        try:
            from .agent import _profile
            sessions.save(session.id, messages, session.todos, mode=session.mode,
                          thinking=session.thinking, profile=_profile(model)[0])
        except (OSError, SystemExit):
            pass
    session_end(session)
    summary = dict(type="result", is_error=err is not None, result=result, session_id=session.id,
                   num_tool_calls=steps["n"], duration_ms=int((time.time() - t0) * 1000), usage=dict(llm.USAGE))
    if output_format == "text":
        print(result)
    elif output_format == "json":
        print(json.dumps(summary))
    else:
        emit(**summary)
    if err:
        raise typer.Exit(1)

mcp_app = typer.Typer(help="Manage MCP servers (tools from external processes)")
app.add_typer(mcp_app, name="mcp")

@mcp_app.command("list")
def mcp_list():
    """Show configured MCP servers and whether they connect."""
    from . import mcp
    st = mcp.status()
    if not st:
        console.print("[dim]no MCP servers configured[/dim]")
    for n, v in st.items():
        console.print(f"[bold]{n}[/bold]: {v}")

@mcp_app.command("add", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def mcp_add(ctx: typer.Context, name: str = typer.Argument(...), command: str = typer.Argument(...),
            env: list[str] = typer.Option([], "--env", "-e", help="KEY=VALUE, repeatable (stdio)"),
            header: list[str] = typer.Option([], "--header", "-H", help="'Name: value', repeatable (HTTP)")):
    """Add a server. stdio: zeusai mcp add fs -- npx -y @modelcontextprotocol/server-filesystem .
    HTTP: zeusai mcp add docs https://example.com/mcp -H 'Authorization: Bearer ${TOKEN}'"""
    from . import mcp
    mcp.add(name, command, ctx.args, dict(e.split("=", 1) for e in env if "=" in e),
            dict((h.split(":", 1)[0].strip(), h.split(":", 1)[1].strip()) for h in header if ":" in h))
    console.print(f"[green]✓[/green] added {name} to ~/.zeusai/config.json")

@mcp_app.command("remove")
def mcp_remove(name: str):
    from . import mcp
    console.print(f"[green]✓[/green] removed {name}" if mcp.remove(name) else f"[red]no server {name} in ~/.zeusai/config.json[/red]")

@app.command("agents")
def agents_cmd():
    """List subagent types (built-in and from .zeusai/agents/*.md)."""
    from .agent import info_command
    console.print(info_command("/agents"))

@app.command("sessions")
def sessions_cmd(limit: int = typer.Option(20, "--limit", "-n")):
    """List saved conversations for this directory (resume with `zeusai chat -r <id>`, `zeusai tui -r <id>`, or `-c`)."""
    from . import sessions
    items = sessions.listing(limit)
    if not items:
        console.print("[dim]no saved conversations in this directory[/dim]")
    for it in items:
        extra = f" · {it['mode']}" if it.get("mode") else ""
        console.print(f"[bold]{it['id']}[/bold] [dim]{it['updated']} · {it['turns']} prompts{extra}[/dim]  {it['title']}")

@app.command()
def doctor(model: str = typer.Option(None, "--model", "-m", help="Test single profile, else test all")):
    """Check connectivity (tests all or one profile)"""
    from concurrent.futures import ThreadPoolExecutor
    from .ensemble import call_paligemma, call_openai_model
    import time
    profiles = get_profiles()
    targets = [model] if model else [k for k in profiles.keys() if profiles[k].get("api_key")]

    def check(alias):
        """Returns (alias, seconds, reply, error)."""
        t = time.monotonic()
        try:
            _, prof = get_active_profile(alias)
            if prof.get("type") == "vlm":
                # paligemma needs image to test; use tiny sign image
                r = call_paligemma(alias, "Describe the image.", "https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1a.jpg")
            else:
                r = call_openai_model(alias, "say hi in 3 words", None, max_tokens=32, thinking=False)
        except (Exception, SystemExit) as e:
            return alias, 0, "", str(e)
        return alias, time.monotonic() - t, r.get("content") or "", r.get("error")

    console.print(f"Testing {len(targets)} profile(s) in parallel...")
    with ThreadPoolExecutor(max_workers=max(1, len(targets))) as ex:
        for alias, dt, reply, err in ex.map(check, targets):
            if err:
                console.print(f"[red]✗ {alias} failed: {err}[/red]")
            else:
                console.print(f"[green]✓ {alias} connected[/green] ({dt:.1f}s) Response: {reply[:80]}")

def main():
    import sys, os, pathlib
    # opencode-style: `zeusai <path>` opens TUI in that directory
    SUBCOMMANDS = {"research","version","models","use","configure","show-config","chat","mini","tui","run","ask","ensemble","doctor","sessions","print","mcp","agents"}
    if len(sys.argv) >= 2 and sys.argv[1] in ("-p", "--print"):
        sys.argv[1] = "print"
    if len(sys.argv) >= 2 and sys.argv[1] in ("-c", "--continue", "-r", "--resume"):
        sys.argv.insert(1, "chat")
    if len(sys.argv) >= 2 and sys.argv[1] not in SUBCOMMANDS and not sys.argv[1].startswith("-"):
        arg = sys.argv[1]
        p = pathlib.Path(arg).expanduser()
        if p.is_dir():
            os.chdir(p)
            sys.argv = [sys.argv[0]]
    if len(sys.argv) == 1:
        try:
            d = get_default_profile_name()
            profiles = get_profiles()
            if not any(profiles[k].get("api_key") for k in profiles):
                raise SystemExit("no key")
        except SystemExit:
            console.print(Panel("Welcome to [bold]zeusai[/bold]\n\nFirst, configure your API key:\n  zeusai configure --profile laguna --api-key <key>\n  zeusai models\n  zeusai doctor\n\nTUI: `zeusai` (full-screen) or `zeusai tui <path>`\nLight REPL: `zeusai mini`\nAutomation: `zeusai run \"prompt\"`", title="zeusai", border_style="blue"))
            return
        # opencode-style: bare `zeusai` = full-screen TUI (line REPL still via `zeusai mini` / `zeusai chat`)
        try:
            from .tui import run_tui
            run_tui()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            # fallback to single-model line chat if TUI fails (e.g. no tty)
            console.print(f"[yellow]TUI failed ({e}), falling back to line chat[/yellow]")
            start_chat(verbose=False)
        return
    app()

if __name__ == "__main__":
    main()
