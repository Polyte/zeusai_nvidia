"""Minimal MCP client (stdio and streamable HTTP transports). Servers come from "mcpServers" in
~/.zeusai/config.json or .zeusai/settings.json:

    "mcpServers": {
      "fs":   {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."], "env": {}},
      "docs": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"}}
    }

${VAR} in headers/env is expanded from the environment.

Their tools are exposed to the agent as mcp__<server>__<tool> and go through normal permission checks.
"""
import atexit
import itertools
import json
import os
import subprocess
import threading

from .config import _load_raw, _write_raw

PROTOCOL = "2025-06-18"
TIMEOUT = 60


def configured():
    from .permissions import _project_settings
    out = dict(_load_raw().get("mcpServers") or {})
    out.update(_project_settings().get("mcpServers") or {})
    return {k: v for k, v in out.items()
            if isinstance(v, dict) and (v.get("command") or v.get("url")) and not v.get("disabled")}


def _expand(d):
    return {k: os.path.expandvars(str(v)) for k, v in (d or {}).items()}


class Server:
    def __init__(self, name, spec):
        self.name = name
        env = {**os.environ, **_expand(spec.get("env"))}
        self.proc = subprocess.Popen([spec["command"], *spec.get("args", [])], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env, bufsize=1)
        self.ids = itertools.count(1)
        self.lock = threading.Lock()
        self.pending = {}  # id -> [Event, response]
        threading.Thread(target=self._reader, daemon=True).start()
        self._handshake()

    def _handshake(self):
        self.tools = []
        self._request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                     "clientInfo": {"name": "zeusai", "version": "2"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        cursor, tools = None, []
        while True:
            r = self._request("tools/list", {"cursor": cursor} if cursor else {})
            tools += r.get("tools", [])
            cursor = r.get("nextCursor")
            if not cursor:
                break
        self.tools = tools

    def _reader(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            slot = self.pending.get(msg.get("id"))
            if slot and ("result" in msg or "error" in msg):
                slot[1] = msg
                slot[0].set()
        for slot in list(self.pending.values()):  # server exited: wake waiters
            slot[0].set()

    def _send(self, msg):
        with self.lock:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()

    def _request(self, method, params):
        rid = next(self.ids)
        slot = self.pending[rid] = [threading.Event(), None]
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            slot[0].wait(TIMEOUT)
        finally:
            self.pending.pop(rid, None)
        msg = slot[1]
        if msg is None:
            raise RuntimeError(f"MCP server {self.name}: no response to {method}")
        if "error" in msg:
            raise RuntimeError(f"MCP server {self.name}: {msg['error'].get('message', msg['error'])}")
        return msg["result"]

    def call(self, tool, args):
        r = self._request("tools/call", {"name": tool, "arguments": args})
        parts = []
        for c in r.get("content", []):
            if c.get("type") == "text":
                parts.append(c["text"])
            elif c.get("type") == "resource":
                parts.append(c.get("resource", {}).get("text", "(resource)"))
            else:
                parts.append(f"({c.get('type')} content)")
        text = "\n".join(parts) or json.dumps(r.get("structuredContent", "")) or "(no content)"
        return ("Error: " if r.get("isError") else "") + text[:20000]

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()


class HttpServer(Server):
    """Streamable HTTP transport: each JSON-RPC message is a POST; replies are JSON or an SSE stream."""

    def __init__(self, name, spec):
        self.name = name
        self.url = spec["url"]
        self.headers = _expand(spec.get("headers"))
        self.session_id = None
        self.ids = itertools.count(1)
        self._handshake()

    def _post(self, msg):
        import urllib.request
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": PROTOCOL, **self.headers}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        req = urllib.request.Request(self.url, data=json.dumps(msg).encode(), headers=headers, method="POST")
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        self.session_id = resp.headers.get("Mcp-Session-Id") or self.session_id
        return resp

    def _send(self, msg):
        self._post(msg).close()

    def _request(self, method, params):
        import urllib.error
        rid = next(self.ids)
        try:
            resp = self._post({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"MCP server {self.name}: HTTP {e.code} {e.read()[:200].decode(errors='ignore')}")
        with resp:
            if "text/event-stream" in (resp.headers.get("Content-Type") or ""):
                msg = None
                data = []
                for raw in resp:
                    line = raw.decode(errors="ignore").rstrip("\r\n")
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip())
                    elif not line and data:
                        try:
                            ev = json.loads("\n".join(data))
                        except ValueError:
                            ev = None
                        data = []
                        if isinstance(ev, dict) and ev.get("id") == rid and ("result" in ev or "error" in ev):
                            msg = ev
                            break
            else:
                body = resp.read()
                msg = json.loads(body) if body.strip() else None
                if isinstance(msg, list):
                    msg = next((m for m in msg if m.get("id") == rid), None)
        if msg is None:
            raise RuntimeError(f"MCP server {self.name}: no response to {method}")
        if "error" in msg:
            raise RuntimeError(f"MCP server {self.name}: {msg['error'].get('message', msg['error'])}")
        return msg["result"]

    def close(self):
        if self.session_id:
            import urllib.request
            try:
                headers = {"Mcp-Session-Id": self.session_id, **self.headers}
                urllib.request.urlopen(urllib.request.Request(self.url, headers=headers, method="DELETE"), timeout=5)
            except Exception:
                pass


_servers, _errors, _started = {}, {}, threading.Lock()


def ensure_started():
    with _started:
        for name, spec in configured().items():
            if name in _servers or name in _errors:
                continue
            try:
                _servers[name] = (HttpServer if spec.get("url") else Server)(name, spec)
            except Exception as e:
                _errors[name] = str(e)
    return _servers


_names = {}  # exposed tool name -> (server, original tool name)


def _safe(s):
    import re
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


def _exposed(server, tool):
    """Model-safe name (^[A-Za-z0-9_-]{1,64}$) for a server tool, unique and reversible via _names."""
    import hashlib
    name = f"mcp__{_safe(server)}__{_safe(tool)}"
    if len(name) > 64:
        name = name[:55] + "_" + hashlib.sha1(f"{server}/{tool}".encode()).hexdigest()[:8]
    while name in _names and _names[name] != (server, tool):
        name = name[:-1] + "x"
    _names[name] = (server, tool)
    return name


def tool_defs():
    defs = []
    for name, srv in ensure_started().items():
        for t in srv.tools:
            defs.append({"type": "function", "function": {
                "name": _exposed(name, t["name"]),
                "description": (t.get("description") or "")[:1000] + f" (MCP server: {name})",
                "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
            }})
    return defs


def resolve(full_name):
    """(server, tool) for an exposed name."""
    if full_name not in _names:
        tool_defs()
    if full_name in _names:
        return _names[full_name]
    _, server, tool = (full_name.split("__", 2) + ["", ""])[:3]
    return server, tool


def call(full_name, args):
    server, tool = resolve(full_name)
    srv = ensure_started().get(server)
    if not srv:
        return f"Error: MCP server {server} is not running ({_errors.get(server, 'not configured')})"
    return srv.call(tool, args)


def status():
    ensure_started()
    out = {n: f"connected · {len(s.tools)} tools" for n, s in _servers.items()}
    out.update({n: f"failed: {e}" for n, e in _errors.items()})
    return out


def add(name, command, args, env=None, headers=None):
    """command may be an http(s) URL for a remote server (then env is ignored, headers are sent)."""
    raw = _load_raw()
    if command.startswith(("http://", "https://")):
        spec = {"url": command, "headers": headers or {}}
    else:
        spec = {"command": command, "args": list(args), "env": env or {}}
    raw.setdefault("mcpServers", {})[name] = spec
    _write_raw(raw)


def remove(name):
    raw = _load_raw()
    found = (raw.get("mcpServers") or {}).pop(name, None) is not None
    _write_raw(raw)
    return found


@atexit.register
def _shutdown():
    for s in _servers.values():
        s.close()
