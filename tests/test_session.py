"""Validation test for agent/session.py"""
import os, sys, types, tempfile

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _project_root)

from database.models import DatabaseManager

# Load session.py directly, bypassing agent/__init__.py
_session_path = os.path.join(_project_root, "agent", "session.py")
_mod = types.ModuleType("agent.session")
_mod.__file__ = _session_path
sys.modules.setdefault("agent", types.ModuleType("agent"))
sys.modules["agent.session"] = _mod
with open(_session_path, encoding="utf-8") as _f:
    exec(compile(_f.read(), _session_path, "exec"), _mod.__dict__)

SessionManager = _mod.SessionManager
Session = _mod.Session
SessionStatus = _mod.SessionStatus

# Create temp DB
f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
db_path = f.name
f.close()

try:
    db = DatabaseManager(db_path)
    sm = SessionManager(db)

    print("=== FLEEA agent/session.py Validation ===\n")

    # ── 1. No session initially ──
    assert sm.get_current_session() is None
    assert sm.session_id is None
    assert sm.short_id is None
    assert sm.is_active is False
    print("  OK  Initial state: no session")

    # ── 2. Session creation ──
    session = sm.create_session()
    assert session is not None
    assert len(session.session_id) == 36  # UUID format: 8-4-4-4-12
    assert len(session.short_id) == 8
    assert session.status == SessionStatus.ACTIVE
    assert session.is_active is True
    assert session.total_interactions == 0
    assert session.total_tool_calls == 0
    assert session.total_tokens == 0
    assert session.total_cost == 0.0
    assert session.last_active is not None
    assert sm.is_active is True
    assert sm.session_id == session.session_id
    assert sm.short_id == session.short_id
    print("  OK  Session creation: UUID, status, counters")

    # ── 3. Record interactions ──
    # Note: SessionManager only tracks interaction counts + tool call counts.
    # Token/cost tracking is owned by ExecutionLogger (separate concern).
    sm.record_interaction(tool_used=True)
    assert session.total_interactions == 1
    assert session.total_tool_calls == 1

    sm.record_interaction(tool_used=False)
    assert session.total_interactions == 2
    assert session.total_tool_calls == 1  # no tool this time

    sm.record_interaction(tool_used=True)
    assert session.total_interactions == 3
    assert session.total_tool_calls == 2
    print("  OK  record_interaction: counters updated correctly")

    # ── 4. Record extra tool calls ──
    sm.record_tool_call()
    assert session.total_tool_calls == 3
    print("  OK  record_tool_call: separate tool tracking")

    # ── 5. Activity timestamp updates ──
    assert session.last_active is not None
    assert session.last_active >= session.created_at
    print("  OK  last_active: updated on interactions")

    # ── 6. Derived properties ──
    assert session.duration_seconds >= 0
    assert isinstance(session.duration_display, str)
    assert "s" in session.duration_display  # at least shows seconds
    assert "$" in session.cost_display
    print("  OK  Derived properties: duration, cost_display")

    # ── 7. to_dict() serialization ──
    data = session.to_dict()
    assert data["session_id"] == session.session_id
    assert data["short_id"] == session.short_id
    assert data["status"] == "active"
    assert data["total_interactions"] == 3
    assert data["total_tool_calls"] == 3
    assert "duration" in data
    assert "avg_tokens_per_interaction" in data
    print("  OK  to_dict: all fields present")

    # ── 8. summary_lines() ──
    lines = session.summary_lines()
    assert len(lines) >= 7
    assert any("Session:" in l for l in lines)
    assert any("Interactions:" in l for l in lines)
    assert any("Tool calls:" in l for l in lines)
    assert any("Tokens used:" in l for l in lines)
    assert any("Est. cost:" in l for l in lines)
    print("  OK  summary_lines: all 7 lines present")

    # ── 9. flush_to_db ──
    sm.flush_to_db()
    db_session = db.get_session(session.session_id)
    assert db_session is not None
    assert db_session["message_count"] == 3
    print("  OK  flush_to_db: stats persisted to SQLite")

    # ── 10. End session ──
    completed = sm.end_session()
    assert completed is not None
    assert completed.status == SessionStatus.ENDED
    assert completed.ended_at is not None
    assert completed.is_active is False
    assert sm.get_current_session() is None
    assert sm.is_active is False

    # Verify DB was updated
    db_session = db.get_session(completed.session_id)
    assert db_session["ended_at"] is not None
    print("  OK  end_session: status, timestamp, DB update")

    # ── 11. End session when none active ──
    result = sm.end_session()
    assert result is None
    print("  OK  end_session (no session): returns None gracefully")

    # ── 12. record_interaction with no active session ──
    sm.record_interaction()         # should not crash
    sm.record_tool_call()           # should not crash
    sm.flush_to_db()                # should not crash
    print("  OK  No-session calls: all degrade gracefully")

    # ── 13. Force-end stale session on create ──
    s1 = sm.create_session()
    s2 = sm.create_session()  # should force-end s1
    assert sm.get_current_session() == s2
    assert s1.status == SessionStatus.ENDED
    print("  OK  Force-end stale session on new create")

    # ── 14. get_session_info ──
    info = sm.get_session_info()
    assert info is not None
    assert info["status"] == "active"
    # End and check None
    sm.end_session()
    assert sm.get_session_info() is None
    print("  OK  get_session_info: dict when active, None when ended")

    # ── 15. get_past_session ──
    past = sm.get_past_session(completed.session_id)
    assert past is not None
    assert past["session_id"] == completed.session_id
    print("  OK  get_past_session: retrieves from DB")

    print("\nAll checks passed.")

finally:
    os.unlink(db_path)
