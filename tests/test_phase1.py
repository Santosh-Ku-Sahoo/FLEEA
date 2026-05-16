"""
FLEEA Phase 1 — Validation Test Suite.
Tests all core modules without requiring API keys.
"""

import asyncio
import sys
import os
import tempfile

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_database():
    """Test DatabaseManager: table creation, CRUD operations."""
    from database.models import DatabaseManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = DatabaseManager(db_path)

        # Test execution log insert + query
        row_id = db.insert_execution_log(
            session_id="test-session",
            user_input="What is AI?",
            detected_intent="question",
            selected_tool="web_search",
            tool_input={"query": "what is AI"},
            tool_result={"results": []},
            success=True,
            execution_duration_ms=123.4,
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            estimated_cost=0.001,
        )
        assert row_id >= 1, f"Expected row_id >= 1, got {row_id}"

        logs = db.get_recent_logs(5)
        assert len(logs) == 1, f"Expected 1 log, got {len(logs)}"
        assert logs[0]["selected_tool"] == "web_search"
        assert logs[0]["total_tokens"] == 300

        # Test conversation insert + query
        db.insert_conversation(
            session_id="test-session",
            user_message="Hello",
            assistant_response="Hi there",
        )
        db.insert_conversation(
            session_id="test-session",
            user_message="How are you?",
            assistant_response="I'm doing well!",
        )

        history = db.get_conversation_history("test-session")
        assert len(history) == 2, f"Expected 2 conversations, got {len(history)}"

        # Test session lifecycle
        db.insert_session("test-session")
        db.update_session_stats("test-session", 2, 300, 0.001)
        db.end_session("test-session")

        session = db.get_session("test-session")
        assert session is not None
        assert session["message_count"] == 2
        assert session["total_tokens"] == 300

        # Test user profile (category-based API)
        db.set_profile("identity", "name", "John")
        assert db.get_profile_value("identity", "name") == "John"

        db.set_profile("identity", "name", "Jane")  # Upsert
        assert db.get_profile_value("identity", "name") == "Jane"

        assert db.get_profile_value("nonexistent", "key") is None

        print("  [PASS] Database: All tests passed")
    finally:
        os.unlink(db_path)


def test_logger():
    """Test ExecutionLogger: cost calculation, usage tracking."""
    from agent.logger import ExecutionLogger, TokenUsage
    from database.models import DatabaseManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = DatabaseManager(db_path)
        logger = ExecutionLogger(db)

        # Test cost calculation with Gemini model
        cost = ExecutionLogger.calculate_cost("gemini-2.5-flash", 1000, 500)
        expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert abs(cost - expected) < 0.0001, f"Cost mismatch: {cost} vs {expected}"

        # Test with unknown model (falls back to default pricing)
        cost_unknown = ExecutionLogger.calculate_cost("unknown-model", 1000, 500)
        expected_default = (1000 / 1_000_000) * 0.10 + (500 / 1_000_000) * 0.40
        assert abs(cost_unknown - expected_default) < 0.0001

        # Test log + session stats
        db.insert_session("s1")
        logger.log_tool_execution(
            session_id="s1",
            user_input="test",
            detected_intent="test",
            selected_tool="web_search",
            tool_input={"query": "test"},
            tool_result="results",
            success=True,
            usage=TokenUsage(100, 200, 300, 0.001),
        )

        stats = logger.get_session_stats("s1")
        assert stats.tool_calls == 1

        print("  [PASS] Logger: All tests passed")
    finally:
        os.unlink(db_path)


def test_safety():
    """Test SafetyManager: action classification."""
    from agent.safety import SafetyManager, SafetyVerdict
    from config.settings import Settings

    settings = Settings()  # type: ignore[call-arg]
    safety = SafetyManager(settings)

    # Web search should be SAFE
    check = safety.classify("web_search", {"query": "AI news"})
    assert check.verdict == SafetyVerdict.SAFE, f"Expected SAFE, got {check.verdict}"

    # File deletion should be DANGEROUS (by tool name)
    check = safety.classify("file_delete", {"path": "/tmp/test.txt"})
    assert check.verdict == SafetyVerdict.DANGEROUS

    # rm -rf / should be DANGEROUS (by pattern)
    check = safety.classify("system_command", {"command": "rm -rf / "})
    assert check.verdict == SafetyVerdict.DANGEROUS

    # Unknown tool should default to SENSITIVE (default-deny policy)
    check = safety.classify("unknown_tool", {"foo": "bar"})
    assert check.verdict == SafetyVerdict.SENSITIVE

    # Credential pattern in input
    check = safety.classify("web_search", {"query": "password reset token"})
    assert check.verdict == SafetyVerdict.SENSITIVE  # credential pattern

    print("  [PASS] Safety: All tests passed")


def test_session():
    """Test SessionManager: lifecycle management."""
    from agent.session import SessionManager
    from database.models import DatabaseManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = DatabaseManager(db_path)
        sm = SessionManager(db)

        # No session initially
        assert sm.get_current_session() is None

        # Create session
        session = sm.create_session()
        assert session is not None
        assert len(session.session_id) == 36  # UUID format
        assert len(session.short_id) == 8

        # End session
        completed = sm.end_session()
        assert completed is not None
        assert completed.ended_at is not None
        assert sm.get_current_session() is None

        print("  [PASS] Session: All tests passed")
    finally:
        os.unlink(db_path)


def test_tool_registry():
    """Test ToolRegistry: registration, schema generation, execution."""
    from tools.tool_registry import ToolRegistry, ToolCategory
    from tools.base_tool import BaseTool, ToolResult

    class MockTool(BaseTool):
        @property
        def name(self):
            return "mock_tool"

        @property
        def description(self):
            return "A mock tool for testing"

        def get_schema(self):
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            }

        async def execute(self, **kwargs):
            return ToolResult(success=True, data=f"Echo: {kwargs.get('message', '')}")

    registry = ToolRegistry()
    mock = MockTool()

    # Register (requires category argument)
    registry.register(mock, category=ToolCategory.SAFE)
    assert len(registry) == 1
    assert "mock_tool" in registry

    # Get schemas
    schemas = registry.get_all_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "mock_tool"

    # Execute
    result = asyncio.run(registry.execute("mock_tool", message="hello"))
    assert result.success
    assert result.data == "Echo: hello"

    # Not found
    result = asyncio.run(registry.execute("nonexistent"))
    assert not result.success

    # Duplicate registration
    try:
        registry.register(mock, category=ToolCategory.SAFE)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("  [PASS] Tool Registry: All tests passed")


def test_profile():
    """Test UserProfile: CRUD and context generation."""
    from agent.profile import UserProfile
    from database.models import DatabaseManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = DatabaseManager(db_path)
        profile = UserProfile(db)

        profile.set("name", "John Doe", category="identity")
        profile.set("work_style", "Deep focus mornings", category="preferences")

        assert profile.get("name", category="identity") == "John Doe"
        assert profile.get("work_style", category="preferences") == "Deep focus mornings"

        all_profile = profile.get_all()
        assert len(all_profile) >= 2

        context = profile.to_context_string()
        assert "John Doe" in context

        print("  [PASS] Profile: All tests passed")
    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    print("\n== FLEEA Phase 1 -- Validation Tests ==\n")

    tests = [
        test_database,
        test_logger,
        test_safety,
        test_session,
        test_tool_registry,
        test_profile,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
