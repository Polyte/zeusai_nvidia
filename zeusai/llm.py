"""Minimal OpenAI-compatible chat client on the stdlib.

Why not the `openai` SDK: importing it costs ~1.4s per launch, and building a
new client per call threw away the TLS connection every time. This module keeps
a pool of keep-alive HTTPS connections per host and streams SSE tokens as they
arrive.

Rate limits: providers (NVIDIA free tiers especially) answer HTTP 429 whenever
they feel like it. chat() retries those (and transient 5xx) with exponential
backoff, honoring the Retry-After header, so a blip doesn't kill the turn.
ZEUSAI_MAX_RETRIES overrides the retry budget (default 5).
"""
import http.client
import json
import os
import random
import ssl
import threading
import time
import urllib.parse

TIMEOUT = 300  # seconds of socket inactivity before giving up

# Statuses worth retrying: rate limits + transient server-side failures.
_RETRYABLE = {408, 429, 500, 502, 503, 504}
_MAX_DELAY = 30.0  # never wait longer than this before one attempt

_ssl_ctx = None
_pool = {}  # (scheme, host, port) -> [idle connections]
_pool_lock = threading.Lock()


class LLMError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status  # HTTP status, so routing can detect rate limits


class _Retryable(Exception):
    """A response the caller may retry (rate limit / transient 5xx)."""

    def __init__(self, status, body, retry_after):
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status
        self.retry_after = retry_after  # seconds the server asked for, or None


def _retry_after(resp):
    """The Retry-After header in seconds (its seconds form), or None."""
    try:
        v = resp.getheader("Retry-After")
    except Exception:
        return None
    if not v:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        return None  # HTTP-date form: plain exponential backoff applies


def _ctx():
    global _ssl_ctx
    if _ssl_ctx is None:
        _ssl_ctx = ssl.create_default_context()
    return _ssl_ctx


def _key(url):
    u = urllib.parse.urlsplit(url)
    port = u.port or (443 if u.scheme == "https" else 80)
    return (u.scheme, u.hostname, port), u.path.rstrip("/")


def _new_conn(key):
    scheme, host, port = key
    if scheme == "https":
        return http.client.HTTPSConnection(host, port, timeout=TIMEOUT, context=_ctx())
    return http.client.HTTPConnection(host, port, timeout=TIMEOUT)


def _acquire(key):
    with _pool_lock:
        idle = _pool.get(key)
        if idle:
            return idle.pop(), True
    return _new_conn(key), False


def _release(key, conn):
    with _pool_lock:
        _pool.setdefault(key, []).append(conn)


def prewarm(base_url):
    """Open TCP+TLS to the API in the background so the first prompt skips the handshake."""
    def _go():
        try:
            key, _ = _key(base_url)
            conn = _new_conn(key)
            conn.connect()
            _release(key, conn)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def post(url, payload, api_key, accept="application/json"):
    """POST JSON over a pooled connection. Returns (conn_key, conn, response).

    Caller must fully read the response and then call _release(key, conn).
    A reused connection that the server already closed is retried once on a fresh one.
    """
    key, path = _key(url)
    body = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": accept,
        "Connection": "keep-alive",
        "User-Agent": "zeusai",
    }
    for attempt in range(2):
        conn, reused = _acquire(key)
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            return key, conn, resp
        except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError,
                http.client.CannotSendRequest, http.client.BadStatusLine, ssl.SSLError):
            conn.close()
            if not reused or attempt:
                raise
    raise LLMError("unreachable")


def iter_sse(resp):
    """Yield parsed JSON objects from an SSE response body."""
    for raw in resp:
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            yield json.loads(data)
        except ValueError:
            continue


# token usage across this process: {"requests", "prompt", "completion", "estimated"} (shown by /cost)
USAGE = {"requests": 0, "prompt": 0, "completion": 0, "estimated": 0}
_usage_lock = threading.Lock()


def _count(usage, messages, out_chars):
    with _usage_lock:
        USAGE["requests"] += 1
        if usage:
            USAGE["prompt"] += usage.get("prompt_tokens") or 0
            USAGE["completion"] += usage.get("completion_tokens") or 0
        else:  # server sent no usage: estimate ~4 chars/token
            USAGE["estimated"] += 1
            USAGE["prompt"] += len(json.dumps(messages)) // 4
            USAGE["completion"] += out_chars // 4


def build_payload(prof, messages, tools=None, max_tokens=None, thinking=None, stream=True):
    payload = {
        "model": prof["model_id"],
        "messages": messages,
        "temperature": prof.get("temperature", 0.6),
        "top_p": prof.get("top_p", 0.95),
        "max_tokens": max_tokens or prof.get("max_tokens", 8192),
        "stream": stream,
    }
    if stream and prof.get("stream_usage", True):
        payload["stream_options"] = {"include_usage": True}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    use_thinking = prof.get("thinking") if thinking is None else thinking
    if use_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    elif prof.get("thinking"):
        # model thinks by default; explicitly turn it off for speed
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if prof.get("reasoning_effort"):
        payload["reasoning_effort"] = prof["reasoning_effort"]
    if prof.get("extra_body"):
        payload.update(prof["extra_body"])
    return payload


def chat(prof, messages, tools=None, max_tokens=None, thinking=None,
         on_text=None, on_reasoning=None, cancel=None, on_tool_delta=None):
    """Streamed chat completion.

    on_text / on_reasoning are called with each token chunk as it arrives.
    on_tool_delta(index, name, arguments) is called whenever a streamed tool
    call's name or arguments grow (arguments is the partial JSON built so far);
    it fires before the tool call is complete, so a UI can preview e.g. a file
    being written while the model is still generating it.
    cancel: optional threading.Event; when set, the stream is abandoned.
    Returns {"content", "reasoning", "tool_calls"} where tool_calls is a list of
    OpenAI-shaped dicts ({"id", "type", "function": {"name", "arguments"}}).
    """
    url = prof["base_url"].rstrip("/") + "/chat/completions"
    payload = build_payload(prof, messages, tools, max_tokens, thinking, stream=True)

    def attempt():
        key, conn, resp = post(url, payload, prof["api_key"], accept="text/event-stream")
        if resp.status >= 400:
            err = resp.read().decode(errors="replace")[:1000]
            retry_after = _retry_after(resp)
            conn.close()
            if resp.status in _RETRYABLE:
                raise _Retryable(resp.status, err, retry_after)
            raise LLMError(f"HTTP {resp.status}: {err}", status=resp.status)
        return key, conn, resp

    # Retry rate limits and transient failures with exponential backoff, honoring
    # Retry-After. Waiting uses the cancel event so Esc stays responsive.
    retries = max(0, int(os.environ.get("ZEUSAI_MAX_RETRIES", "5")))
    backoff = 1.0
    while True:
        try:
            key, conn, resp = attempt()
            break
        except _Retryable as e:
            if retries <= 0:
                hint = (" — rate limited by the provider; wait a few seconds and retry"
                        if e.status == 429 else "")
                raise LLMError(f"{e} after retries{hint}", status=e.status) from None
            retries -= 1
            delay = e.retry_after if e.retry_after is not None else backoff * random.uniform(0.5, 1.0)
            delay = min(delay, _MAX_DELAY)
            backoff = min(backoff * 2, _MAX_DELAY)
            if cancel is not None:
                if cancel.wait(delay):
                    return {"content": "", "reasoning": None, "tool_calls": []}
            else:
                time.sleep(delay)

    content, reasoning = [], []
    calls = {}  # index -> accumulated tool call
    usage = None
    finished = False
    try:
        for ev in iter_sse(resp):
            if cancel is not None and cancel.is_set():
                break
            if ev.get("usage"):
                usage = ev["usage"]
            choices = ev.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                reasoning.append(r)
                if on_reasoning:
                    on_reasoning(r)
            c = delta.get("content")
            if c:
                content.append(c)
                if on_text:
                    on_text(c)
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", len(calls))
                slot = calls.setdefault(idx, {
                    "id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                grew = False
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                    grew = True
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
                    grew = True
                if grew and on_tool_delta is not None and slot["function"]["name"]:
                    on_tool_delta(idx, slot["function"]["name"], slot["function"]["arguments"])
        else:
            finished = True
    finally:
        if finished:
            resp.read()  # drain so the connection can be reused
            _release(key, conn)
        else:
            conn.close()

    _count(usage, messages, sum(map(len, content)) + sum(map(len, reasoning)))
    tool_calls = [calls[i] for i in sorted(calls)]
    for i, tc in enumerate(tool_calls):
        if not tc["id"]:
            tc["id"] = f"call_{i}"
    return {"content": "".join(content), "reasoning": "".join(reasoning) or None, "tool_calls": tool_calls}
