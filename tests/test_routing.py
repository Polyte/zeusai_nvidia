"""Intelligent routing: capability + cost aware selection with fallback & circuit breaker."""
import copy

import pytest

from zeusai import agent, llm, routing
from zeusai.modes import Session


@pytest.fixture
def keyed(monkeypatch):
    """DEFAULT_PROFILES with fake keys so routing sees usable profiles."""
    from zeusai import config
    profs = copy.deepcopy(config.DEFAULT_PROFILES)
    for p in profs.values():
        p["api_key"] = "k"
        p.setdefault("model_id", "m")
    monkeypatch.setattr(config, "DEFAULT_PROFILES", profs)
    return profs


@pytest.fixture(autouse=True)
def _fresh_routing():
    routing._state.clear()
    yield
    routing._state.clear()


# --------------------------------------------------------------- candidates & order

def test_explicit_primary_leads(keyed):
    chain, skipped = routing.candidates(primary="glimmer")
    assert chain[0] == "glimmer"


def test_trivial_prefers_cheapest(keyed):
    chain, _ = routing.candidates(trivial=True)
    assert chain[0] == "laguna"
    assert routing.cost(chain[0]) <= routing.cost("ultra")


def test_complex_prefers_strongest(keyed):
    chain, _ = routing.candidates(complex=True)
    assert chain[0] == "ultra"


def test_default_keeps_quality_order(keyed):
    chain, _ = routing.candidates()
    assert chain[0] == "ultra"
    # tools are in play, so the VLM profile is dropped from the chain
    assert set(chain) == {"ultra", "glimmer", "super", "laguna"}
    assert "paligemma" not in chain
    chain, _ = routing.candidates(tools=False)
    assert "paligemma" in chain


def test_vision_order_and_tools_drop_vlm(keyed):
    chain, _ = routing.candidates(has_image=True, tools=True)
    assert chain[0] == "super"
    assert "paligemma" not in chain          # VLM profile can't take tools
    chain, _ = routing.candidates(has_image=True, tools=False)
    assert chain[0] == "super" and "paligemma" in chain


def test_cooling_profiles_are_skipped(keyed, monkeypatch):
    routing.mark_limited("ultra")
    chain, skipped = routing.candidates(complex=True)
    assert "ultra" in skipped
    assert "ultra" not in chain
    assert chain[0] == "glimmer"


def test_all_cooling_returns_chain_anyway(keyed):
    for a in list(keyed):
        routing.mark_limited(a)
    chain, skipped = routing.candidates(primary="ultra")
    assert chain and skipped


# --------------------------------------------------------------- circuit breaker

def test_breaker_growth_and_cap(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(routing, "time", type("T", (), {"monotonic": staticmethod(lambda: now["t"])})())
    routing.mark_limited("ultra")
    routing.mark_limited("ultra")
    routing.mark_limited("ultra")
    assert routing.cooldown_left("ultra") == 60 * 4  # fails=3 -> 60s base * 2^(3-1)
    now["t"] = 1000 + 60 * 4 + 1
    assert not routing.is_cooling("ultra")
    routing.record_ok("ultra")
    assert routing.state("ultra")["fails"] == 0


def test_status_text_mentions_limited():
    routing.mark_limited("super")
    assert "super" in routing.status_text()


# --------------------------------------------------------------- run_loop integration

def _patch_chat(monkeypatch, outcomes):
    """outcomes: {alias: [llm result dict or Exception, ...]}; cycles per alias."""
    calls = []
    def chat(prof, messages, **kw):
        alias = prof.get("alias") or prof
        calls.append(alias)
        seq = outcomes[alias]
        r = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(llm, "chat", chat)
    return calls


OK = {"content": "done", "reasoning": None, "tool_calls": []}


def test_run_loop_falls_back_on_429(keyed, monkeypatch):
    calls = _patch_chat(monkeypatch, {"ultra": [llm.LLMError("HTTP 429", status=429)], "glimmer": [OK]})
    seen = []
    s = Session(mode="auto", auto_continue=False)
    out = agent.run_loop([{"role": "user", "content": "fix the bug and refactor the parser"}], session=s,
                         profile_name="ultra", on_status=seen.append)
    assert out == "done"
    assert calls == ["ultra", "glimmer"]
    assert routing.is_cooling("ultra")
    assert any("glimmer" in m for m in seen)


def test_run_loop_raises_when_everything_limited(keyed, monkeypatch):
    _patch_chat(monkeypatch, {a: [llm.LLMError("HTTP 429", status=429)]
                              for a in ("ultra", "glimmer", "super", "laguna")})
    s = Session(mode="auto", auto_continue=False)
    with pytest.raises(llm.LLMError) as e:
        agent.run_loop([{"role": "user", "content": "fix and refactor everything"}], session=s,
                       profile_name="ultra")
    assert e.value.status == 429


def test_run_loop_hard_error_does_not_fall_back(keyed, monkeypatch):
    calls = _patch_chat(monkeypatch, {"ultra": [llm.LLMError("HTTP 400 bad", status=400)], "glimmer": [OK]})
    s = Session(mode="auto", auto_continue=False)
    with pytest.raises(llm.LLMError):
        agent.run_loop([{"role": "user", "content": "x"}], session=s, profile_name="ultra")
    assert calls == ["ultra"]


def test_run_loop_never_leaves_the_requested_model_on_success_without_429(keyed, monkeypatch):
    calls = _patch_chat(monkeypatch, {"ultra": [OK], "glimmer": [OK]})
    s = Session(mode="auto", auto_continue=False)
    assert agent.run_loop([{"role": "user", "content": "say hi"}], session=s, profile_name="ultra") == "done"
    assert calls == ["ultra"]
    assert not routing.is_cooling("ultra")


def test_trivial_prompts_route_cheap_when_auto(keyed, monkeypatch):
    calls = _patch_chat(monkeypatch, {a: [OK] for a in ("laguna", "ultra", "glimmer", "super", "paligemma")})
    s = Session(mode="default", auto_continue=False)
    agent.run_loop([{"role": "user", "content": "hi"}], session=s, profile_name=None)
    assert calls == ["laguna"]  # cost optimisation: cheapest capable model first
