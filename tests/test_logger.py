"""Validation test for agent/logger.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from database.models import DatabaseManager
from agent.logger import ExecutionLogger, TokenUsage, SessionStats, Timer, MODEL_PRICING

f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
db_path = f.name
f.close()

try:
    db = DatabaseManager(db_path)
    logger = ExecutionLogger(db)

    print("=== FLEEA agent/logger.py Validation ===\n")

    # ── 1. Cost calculation ──
    cost = ExecutionLogger.calculate_cost("claude-sonnet-4-20250514", 1000, 500)
    expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
    assert abs(cost - expected) < 0.0001
    # Unknown model falls back to default pricing (0.10 input, 0.40 output)
    cost_unknown = ExecutionLogger.calculate_cost("unknown-model", 1000, 500)
    expected_default = (1000 / 1_000_000) * 0.10 + (500 / 1_000_000) * 0.40
    assert abs(cost_unknown - expected_default) < 0.0001
    print("  OK  Cost calculation: known model + fallback")

    # ── 2. Pricing table coverage ──
    assert "claude-sonnet-4-20250514" in MODEL_PRICING
    assert "claude-3-5-haiku-20241022" in MODEL_PRICING
    assert "gpt-4o" in MODEL_PRICING
    print(f"  OK  Pricing table: {len(MODEL_PRICING)} models")

    # ── 3. Token usage extraction (mock response — Gemini format) ──
    class MockUsageMetadata:
        prompt_token_count = 200
        candidates_token_count = 400

    class MockResponse:
        usage_metadata = MockUsageMetadata()
        model_version = "gemini-2.5-flash"
        model = "gemini-2.5-flash"

    usage = ExecutionLogger.extract_usage(MockResponse())
    assert usage.prompt_tokens == 200
    assert usage.completion_tokens == 400
    assert usage.total_tokens == 600
    assert usage.estimated_cost > 0
    assert not usage.is_empty

    # No usage attribute
    empty_usage = ExecutionLogger.extract_usage(object())
    assert empty_usage.is_empty
    print("  OK  extract_usage: mock response + missing usage")

    # ── 4. Tool execution logging ──
    db.insert_session("s1")

    row_id = logger.log_tool_execution(
        session_id="s1",
        user_input="What is AI?",
        detected_intent="knowledge_question",
        selected_tool="web_search",
        tool_input={"query": "what is AI"},
        tool_result={"results": [{"title": "AI", "url": "https://example.com"}]},
        success=True,
        execution_duration_ms=345.6,
        usage=TokenUsage(150, 300, 450, 0.005),
    )
    assert row_id >= 1

    # Failed tool call
    logger.log_tool_execution(
        session_id="s1",
        user_input="Weather?",
        detected_intent="weather",
        selected_tool="web_search",
        tool_input={"query": "weather"},
        tool_result="",
        success=False,
        error="Timeout after 30s",
        retry_count=2,
        execution_duration_ms=30000.0,
        usage=TokenUsage(50, 0, 50, 0.0001),
    )

    stats = logger.get_session_stats("s1")
    assert stats.tool_calls == 2
    assert stats.tool_successes == 1
    assert stats.tool_errors == 1
    # Note: log_tool_execution tracks tool_calls counts only in session stats;
    # token accumulation is handled by track_api_usage() to avoid double-counting.
    assert stats.success_rate == 50.0
    print("  OK  log_tool_execution: success + failure + stats")

    # ── 5. Conversation logging ──
    logger.log_conversation(
        session_id="s1",
        user_message="What is AI?",
        assistant_response="AI stands for Artificial Intelligence...",
        tool_invoked="web_search",
    )
    logger.log_conversation(
        session_id="s1",
        user_message="Thanks!",
        assistant_response="You are welcome!",
    )
    assert stats.message_count == 2

    history = logger.get_conversation_history("s1")
    assert len(history) == 2
    assert history[0]["user_message"] == "What is AI?"
    assert history[1]["tool_invoked"] is None
    print("  OK  log_conversation: paired turns + history query")

    # ── 6. Error logging ──
    error_id = logger.log_error(
        session_id="s1",
        user_input="Do something dangerous",
        error="Action blocked by safety layer",
        error_type="safety_blocked",
        selected_tool="file_delete",
    )
    assert error_id >= 1
    assert stats.tool_errors == 2  # 1 from failed tool + 1 from error log
    print("  OK  log_error: error recorded in execution_logs")

    # ── 7. API usage tracking ──
    logger.track_api_usage("s1", TokenUsage(100, 200, 300, 0.003))
    assert stats.api_calls == 1
    assert stats.total_tokens == 300  # only track_api_usage accumulates tokens
    print("  OK  track_api_usage: in-memory accumulation")

    # ── 8. Session stats flush ──
    logger.flush_session_stats("s1")
    sess = db.get_session("s1")
    assert sess is not None
    assert sess["message_count"] == 2
    assert sess["total_tokens"] == 300  # matches in-memory stats

    # Flush non-existent session — should not crash
    logger.flush_session_stats("nonexistent")
    print("  OK  flush_session_stats: persisted + graceful on missing")

    # ── 9. Cost warning checks ──
    # Session warning
    warning = logger.check_cost_warning("s1", session_limit=0.001, monthly_limit=100.0)
    assert warning is not None
    assert "Session cost" in warning

    # No warning
    no_warning = logger.check_cost_warning("s1", session_limit=100.0, monthly_limit=1000.0)
    assert no_warning is None
    print("  OK  check_cost_warning: session threshold + no-warning")

    # ── 10. Recent logs query ──
    logs = logger.get_recent_logs(10)
    assert len(logs) == 3  # 2 tool calls + 1 error
    assert logs[0]["selected_tool"] in ("web_search", "file_delete")
    print("  OK  get_recent_logs: returns correct count")

    # ── 11. Timer utility ──
    import time
    with Timer() as t:
        time.sleep(0.05)
    assert t.duration_ms >= 40  # ~50ms sleep, allow some variance
    assert "Timer(" in repr(t)
    print("  OK  Timer: context manager + repr")

    # ── 12. SessionStats properties ──
    s = SessionStats(tool_calls=10, tool_successes=8, tool_errors=2, total_cost=0.005)
    assert s.success_rate == 80.0
    assert "$" in s.cost_display
    print("  OK  SessionStats: success_rate + cost_display")

    print(f"\nAll checks passed.")

finally:
    os.unlink(db_path)
