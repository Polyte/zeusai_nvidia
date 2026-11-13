from zeusai import sessions, tools
from zeusai.modes import Session


def test_save_load_resume(sandbox):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi"}]
    sid = sessions.new_id()
    sessions.save(sid, msgs, [{"content": "t", "status": "pending"}])
    d = sessions.load()
    assert d["id"] == sid and d["messages"][0]["role"] == "user"  # system prompt isn't stored
    assert sessions.load(sid[:12])["todos"][0]["content"] == "t"
    assert sessions.listing()[0]["title"] == "hello world"


def test_empty_conversation_not_saved(sandbox):
    assert sessions.save(sessions.new_id(), [{"role": "system", "content": "S"}]) is None


def test_split_point_keeps_tool_results_with_their_call():
    body = [{"role": "user"}, {"role": "assistant", "tool_calls": [1]}, {"role": "tool"}, {"role": "tool"},
            {"role": "assistant"}]
    assert sessions._split_point(body, 3) == 1


def test_compact_replaces_old_history(monkeypatch):
    monkeypatch.setattr(sessions.llm, "chat", lambda *a, **k: {"content": "SUMMARY"})
    monkeypatch.setattr(sessions, "get_active_profile", lambda n=None: ("x", {}))
    m = [{"role": "system", "content": "SYS"}] + [{"role": "user", "content": f"u{i}"} for i in range(12)]
    assert sessions.compact(m) == "SUMMARY"
    assert m[0]["content"] == "SYS" and m[1]["summary"] and len(m) == 2 + sessions.KEEP_RECENT


def test_todo_tool_validates():
    todos = []
    out = sessions.todo_tool({"todos": [{"content": "a", "status": "completed"},
                                        {"content": "b", "status": "in_progress"}]}, todos)
    assert "☑ a" in out and "◐ b" in out and len(todos) == 2
    two = [{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}]
    assert sessions.todo_tool({"todos": two}, todos).startswith("Error") and len(todos) == 2


def test_rewind_restores_files_and_conversation(sandbox):
    s = Session()
    (sandbox / "keep.txt").write_text("original")
    messages = [{"role": "user", "content": "first"}]
    sessions.begin_turn(s, messages)
    tools.execute_tool("read_file", {"path": "keep.txt"})
    tools.execute_tool("write_file", {"path": "keep.txt", "content": "changed"})
    tools.execute_tool("write_file", {"path": "made.txt", "content": "new"})
    messages.append({"role": "assistant", "content": "done"})
    files, n = sessions.rewind(s, messages)
    assert n == 1 and len(files) == 2
    assert (sandbox / "keep.txt").read_text() == "original" and not (sandbox / "made.txt").exists()
    assert messages == []
