import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from conftest import write_settings
from zeusai import commands, mcp, mentions, permissions, statusline
from zeusai.modes import Session, tools_for

ECHO_SERVER = r'''
import sys, json
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m: continue
    if m["method"] == "initialize": r = {"protocolVersion": "2025-06-18", "capabilities": {}}
    elif m["method"] == "tools/list": r = {"tools": [{"name": "echo", "description": "Echo", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}
    else: r = {"content": [{"type": "text", "text": "echo: " + m["params"]["arguments"]["text"]}]}
    print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}), flush=True)
'''


def test_custom_commands(sandbox):
    d = sandbox / ".zeusai" / "commands" / "git"
    d.mkdir(parents=True)
    (d.parent / "hi.md").write_text("---\ndescription: Say hi\n---\nHello $1, all: $ARGUMENTS. !`echo SHELL`\n")
    (d / "commit.md").write_text("Commit staged")
    assert commands.expand("/hi bob and al") == "Hello bob, all: bob and al. SHELL"
    assert commands.expand("/git:commit now") == "Commit staged\n\nnow"
    assert commands.expand("/missing") is None


def test_mentions(sandbox):
    (sandbox / "a.py").write_text("print(1)\n")
    (sandbox / "pic.png").write_bytes(b"x")
    text, attached = mentions.expand("see @a.py and @pic.png, mail me@x.com, @nope.py")
    assert attached == ["a.py", "pic.png"]
    assert '<file path="a.py">\n     1\tprint(1)\n</file>' in text and "call read_file" in text


def test_permissions_rules(sandbox):
    write_settings(sandbox, {"permissions": {"allow": ["Bash(git status*)"], "deny": ["Bash(rm -rf*)"]}})
    assert permissions.check("run_shell", {"command": "git status -s"}, "default") == "allow"
    assert permissions.check("run_shell", {"command": "rm -rf /"}, "auto") == "deny"
    assert permissions.check("run_shell", {"command": "ls"}, "default") == "ask"
    assert permissions.check("glob_files", {"pattern": "*"}, "plan") == "allow"


def test_mcp_stdio(sandbox):
    server = sandbox / "echo.py"
    server.write_text(ECHO_SERVER)
    write_settings(sandbox, {"mcpServers": {"echo": {"command": sys.executable, "args": [str(server)]}}})
    assert mcp.status() == {"echo": "connected · 1 tools"}
    assert "mcp__echo__echo" in [t["function"]["name"] for t in tools_for("default")]
    assert "mcp__echo__echo" not in [t["function"]["name"] for t in tools_for("plan")]
    assert mcp.call("mcp__echo__echo", {"text": "hi"}) == "echo: hi"
    assert permissions.check("mcp__echo__echo", {}, "default") == "ask"
    assert permissions.suggest_rule("mcp__echo__echo", {}) == "Mcp(echo:echo)"


def test_mcp_http_json_and_sse(sandbox, monkeypatch):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_DELETE(self):
            seen["deleted"] = self.headers.get("Mcp-Session-Id")
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            m = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.setdefault("auth", self.headers.get("Authorization"))
            if "id" not in m:
                self.send_response(202)
                self.end_headers()
                return
            if m["method"] != "initialize":
                assert self.headers.get("Mcp-Session-Id") == "sess-1"
            if m["method"] == "initialize":
                r = {"protocolVersion": "2025-06-18", "capabilities": {}}
            elif m["method"] == "tools/list":
                r = {"tools": [{"name": "add", "inputSchema": {"type": "object"}}]}
            else:
                a = m["params"]["arguments"]
                r = {"content": [{"type": "text", "text": str(a["a"] + a["b"])}]}
            body = json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r})
            self.send_response(200)
            if m["method"] == "tools/call":
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n')
                self.wfile.write(f"event: message\ndata: {body}\n\n".encode())
            else:
                self.send_header("Content-Type", "application/json")
                if m["method"] == "initialize":
                    self.send_header("Mcp-Session-Id", "sess-1")
                self.end_headers()
                self.wfile.write(body.encode())

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("TOKEN", "s3cret")
    try:
        mcp.add("calc", f"http://127.0.0.1:{srv.server_address[1]}/mcp", [], headers={"Authorization": "Bearer ${TOKEN}"})
        assert mcp.status() == {"calc": "connected · 1 tools"}
        assert mcp.call("mcp__calc__add", {"a": 2, "b": 40}) == "42"
        mcp._shutdown()
        assert seen == {"auth": "Bearer s3cret", "deleted": "sess-1"}
    finally:
        srv.shutdown()


def test_statusline_default_and_custom(sandbox):
    s = Session(mode="auto", todos=[{"content": "a", "status": "completed"}, {"content": "b", "status": "pending"}])
    line = statusline.render("ultra", s)
    assert line.startswith("ultra · auto") and "todos 1/2" in line
    write_settings(sandbox, {"statusLine": {"command": "python3 -c 'import json,sys; print(\"M=\" + json.load(sys.stdin)[\"model\"])'"}})
    assert statusline.render("ultra", s) == "M=ultra"
