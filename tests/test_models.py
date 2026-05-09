"""Quick validation for database/models.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import DatabaseManager, SCHEMA_VERSION

f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
db_path = f.name
f.close()

try:
    db = DatabaseManager(db_path)
    print("=== FLEEA database/models.py Validation ===\n")
    print(f"  Schema version: {SCHEMA_VERSION}")

    # 1. All 6 tables exist
    health = db.health_check()
    print("\n  Tables:")
    for table, count in health["table_counts"].items():
        status = "OK" if count >= 0 else "MISSING"
        print(f"    {status}  {table}: {count} rows")

    # 2. execution_logs
    db.insert_execution_log(
        session_id="s1", user_input="What is AI?", detected_intent="question",
        selected_tool="web_search", tool_input={"query": "AI"},
        tool_result={"results": []}, success=True, execution_duration_ms=345.6,
        prompt_tokens=150, completion_tokens=300, total_tokens=450, estimated_cost=0.005,
    )
    db.insert_execution_log(
        session_id="s1", user_input="Weather?", detected_intent="weather",
        selected_tool="web_search", tool_input={"query": "weather"},
        tool_result="", success=False, error="Timeout", retry_count=2,
        execution_duration_ms=30000, prompt_tokens=50, completion_tokens=0,
        total_tokens=50, estimated_cost=0.0001,
    )
    logs = db.get_recent_logs(5)
    assert len(logs) == 2
    assert logs[0]["selected_tool"] == "web_search"
    summary = db.get_session_cost_summary("s1")
    assert summary["tool_calls"] == 2
    assert summary["successes"] == 1
    assert summary["failures"] == 1
    assert summary["total_tokens"] == 500
    print("\n  OK  execution_logs: insert, query, cost_summary")

    # 3. conversations
    db.insert_conversation(session_id="s1", user_message="What is AI?",
                           assistant_response="Artificial Intelligence...", tool_invoked="web_search")
    db.insert_conversation(session_id="s1", user_message="Thanks!",
                           assistant_response="You are welcome!")
    history = db.get_conversation_history("s1")
    assert len(history) == 2
    assert history[0]["user_message"] == "What is AI?"
    assert history[1]["tool_invoked"] is None
    recent = db.get_recent_conversations("s1", n=1)
    assert len(recent) == 1
    assert recent[0]["user_message"] == "Thanks!"
    assert db.count_conversations("s1") == 2
    print("  OK  conversations: insert, history, recent, count")

    # 4. sessions
    db.insert_session("s1")
    db.update_session_stats("s1", message_count=2, total_tokens=500, total_cost=0.005)
    db.end_session("s1")
    sess = db.get_session("s1")
    assert sess is not None
    assert sess["message_count"] == 2
    assert sess["ended_at"] is not None
    print("  OK  sessions: insert, update, end, get")

    # 5. user_profile — category-based
    db.set_profile("identity", "name", "John Doe")
    db.set_profile("identity", "timezone", "US/Eastern")
    db.set_profile("work_style", "focus_hours", "9AM-12PM")
    db.set_profile("work_style", "communication", "Concise, direct")
    db.set_profile("schedule", "meeting_hours", "2PM-5PM")
    db.set_profile("writing", "tone", "Professional but warm")
    db.set_profile("habits", "morning_routine", "Review tasks, check calendar")
    db.set_profile("priorities", "ranking", "Urgent > High > Medium > Low")
    db.set_profile("preferences", "theme", "dark")

    assert db.get_profile_value("identity", "name") == "John Doe"
    assert db.get_profile_value("nonexistent", "key") is None

    # Upsert
    db.set_profile("identity", "name", "Jane Doe")
    assert db.get_profile_value("identity", "name") == "Jane Doe"

    # Category query
    work = db.get_profile_category("work_style")
    assert len(work) == 2
    assert work["focus_hours"] == "9AM-12PM"

    # Full profile
    full = db.get_full_profile()
    assert "identity" in full
    assert "schedule" in full
    assert "work_style" in full
    assert "writing" in full
    assert "habits" in full
    assert "priorities" in full
    assert "preferences" in full
    assert len(full) == 7  # 7 categories

    # Delete
    assert db.delete_profile_entry("preferences", "theme") is True
    assert db.get_profile_value("preferences", "theme") is None
    assert db.delete_profile_entry("preferences", "theme") is False

    # Context string
    ctx = db.profile_to_context_string()
    assert "### Identity" in ctx
    assert "Jane Doe" in ctx
    assert "### Work Style" in ctx
    print("  OK  user_profile: set, get, upsert, category, full, delete, context_string")

    # 6. Health check final
    health = db.health_check()
    assert health["schema_version"] == 1
    assert health["file_size_mb"] > 0
    assert health["table_counts"]["tasks"] == 0
    assert health["table_counts"]["calendar_events"] == 0
    print("  OK  health_check: schema version, file size, future tables ready")

    print("\nAll checks passed.")

finally:
    os.unlink(db_path)
