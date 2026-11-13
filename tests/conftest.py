import json

import pytest


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Each test gets its own HOME and working directory, and fresh module state."""
    home, proj = tmp_path / "home", tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(proj)
    from zeusai import config, mcp, tools
    config._raw_cache.update(mtime=None, data={})
    tools._read_paths.clear()
    tools.checkpoint_hook = None
    yield proj
    mcp._shutdown()
    mcp._servers.clear()
    mcp._errors.clear()
    mcp._names.clear()
    tools.kill_background()
    tools._bg.clear()


def tc(i, name, args):
    """An OpenAI-shaped tool call."""
    return {"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture
def fake_model(monkeypatch):
    """Script the model: fake_model(reply, reply, ...) or fake_model(fn). Returns the list of sent message lists."""
    from zeusai import agent

    sent = []

    def install(*replies):
        it = iter(replies)

        def chat(prof, messages, **kw):
            sent.append(json.loads(json.dumps(messages)))
            if len(replies) == 1 and callable(replies[0]):
                return replies[0](prof, messages, **kw)
            r = next(it)
            return {"content": r.get("content", ""), "tool_calls": r.get("tool_calls", [])}

        monkeypatch.setattr(agent.llm, "chat", chat)
        monkeypatch.setattr(agent, "_profile", lambda n=None: ("test", {"vision": False}))
        return sent

    return install


def write_settings(proj, data):
    d = proj / ".zeusai"
    d.mkdir(exist_ok=True)
    (d / "settings.json").write_text(json.dumps(data))
