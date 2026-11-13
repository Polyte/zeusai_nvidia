"""HTTP 429 / transient 5xx retries with backoff (no more dead turns on rate limits)."""
import threading

import pytest

from zeusai import llm


class _Conn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Resp:
    def __init__(self, status, body=b"", sse=(), retry_after=None):
        self.status = status
        self._body = body
        self._sse = list(sse)
        self._ra = retry_after

    def read(self):
        return self._body

    def getheader(self, name):
        return self._ra if name.lower() == "retry-after" else None

    def __iter__(self):
        return iter(self._sse)


def _ok(text):
    return _Resp(200, sse=[f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}\n'.encode(),
                           b"data: [DONE]\n"])


PROF = {"base_url": "https://api.test", "api_key": "k", "model_id": "m"}


def _script(monkeypatch, resps, sleeps=None):
    """Patch llm.post to serve scripted responses; record sleeps when asked."""
    it = iter(resps)
    calls = {"n": 0}
    def fake_post(url, payload, api_key, accept=None):
        calls["n"] += 1
        return ("key", _Conn(), next(it))
    monkeypatch.setattr(llm, "post", fake_post)
    if sleeps is not None:
        monkeypatch.setattr(llm.time, "sleep", lambda d: sleeps.append(d))
    return calls


MSGS = [{"role": "user", "content": "hi"}]


def test_429_then_success_honors_retry_after(monkeypatch):
    sleeps = []
    calls = _script(monkeypatch, [
        _Resp(429, b"rate limited", retry_after="2"),
        _Resp(429, b"rate limited", retry_after="3"),
        _ok("hello"),
    ], sleeps)
    r = llm.chat(PROF, MSGS)
    assert r["content"] == "hello"
    assert calls["n"] == 3
    assert sleeps == [2.0, 3.0]  # exactly what the server asked for


def test_429_exhausts_budget_and_raises_helpful_error(monkeypatch):
    monkeypatch.setenv("ZEUSAI_MAX_RETRIES", "2")
    sleeps = []
    _script(monkeypatch, [_Resp(429, b"slow down", retry_after="1")] * 9, sleeps)
    with pytest.raises(llm.LLMError) as e:
        llm.chat(PROF, MSGS)
    assert "429" in str(e.value) and "rate limited" in str(e.value)
    assert len(sleeps) == 2  # initial attempt + 2 retries


def test_hard_error_is_not_retried(monkeypatch):
    sleeps = []
    calls = _script(monkeypatch, [_Resp(400, b"bad request")], sleeps)
    with pytest.raises(llm.LLMError) as e:
        llm.chat(PROF, MSGS)
    assert "HTTP 400" in str(e.value)
    assert calls["n"] == 1 and sleeps == []


def test_transient_5xx_backoff_grows(monkeypatch):
    sleeps = []
    _script(monkeypatch, [_Resp(503, b"overloaded"), _ok("back")], sleeps)
    r = llm.chat(PROF, MSGS)
    assert r["content"] == "back"
    assert len(sleeps) == 1 and 0.5 <= sleeps[0] <= 1.0  # 1s backoff with jitter


def test_cancel_during_backoff_returns_briefly(monkeypatch):
    cancel = threading.Event()
    cancel.set()  # already cancelled: must not wait out the backoff
    _script(monkeypatch, [_Resp(429, b"nope", retry_after="30")], [])
    r = llm.chat(PROF, MSGS, cancel=cancel)
    assert r == {"content": "", "reasoning": None, "tool_calls": []}


def test_retry_after_is_capped(monkeypatch):
    sleeps = []
    _script(monkeypatch, [_Resp(429, b"x", retry_after="120"), _ok("y")], sleeps)
    assert llm.chat(PROF, MSGS)["content"] == "y"
    assert sleeps == [30.0]
