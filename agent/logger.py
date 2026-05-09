"""
FLEEA Execution Logger — Structured logging, token tracking, and cost observability.

Architecture:
    Every tool call, every API request, every conversation turn, and every
    error is logged to SQLite via the DatabaseManager.  In-memory session
    stats provide real-time aggregates for the /status command without
    re-querying the database on every keystroke.

    The logger also owns the model pricing table and cost calculation logic.
    This keeps financial awareness in one place — the brain doesn't need
    to know what anything costs; it just hands response objects to the logger.

Design:
    - DatabaseManager injected via constructor (no globals, no singletons).
    - Five public logging methods, each maps to one concern:
          log_tool_execution   → execution_logs table
          log_conversation     → conversations table
          log_error            → execution_logs table (with error details)
          track_api_usage      → in-memory stats (no DB write, fast)
          flush_session_stats  → persists in-memory stats to sessions table
    - TokenUsage and SessionStats are plain dataclasses — no Pydantic
      overhead for internal data that never crosses an API boundary.
    - Timer context manager for precise tool execution timing.

Usage:
    logger = ExecutionLogger(db)

    usage = logger.extract_usage(claude_response)
    logger.log_tool_execution(
        session_id=session.session_id,
        user_input="What is AI?",
        detected_intent="knowledge_question",
        selected_tool="web_search",
        tool_input={"query": "what is AI"},
        tool_result={"results": [...]},
        success=True,
        execution_duration_ms=timer.duration_ms,
        usage=usage,
    )
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from database.models import DatabaseManager


# ── Module logger for internal diagnostics ─────────────────────────
_log = logging.getLogger("fleea.logger")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODEL PRICING TABLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  USD per 1 million tokens.  Update when Anthropic / OpenAI change pricing.
#  Source: https://docs.anthropic.com/en/docs/about-claude/models
#  Last updated: 2025-05-15

MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic — Claude
    "claude-sonnet-4-20250514":     {"input": 3.00,  "output": 15.00},
    "claude-3-5-sonnet-20241022":   {"input": 3.00,  "output": 15.00},
    "claude-3-5-haiku-20241022":    {"input": 0.80,  "output": 4.00},
    "claude-3-opus-20240229":       {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307":      {"input": 0.25,  "output": 1.25},
    # OpenAI — stub pricing for future fallback
    "gpt-4o":                       {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":                  {"input": 0.15,  "output": 0.60},
}

# Fallback when the model ID isn't in the table
_DEFAULT_PRICING: dict[str, float] = {"input": 3.00, "output": 15.00}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATA CLASSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class TokenUsage:
    """Token counts and estimated cost from a single API call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.total_tokens == 0


@dataclass
class SessionStats:
    """
    Running in-memory aggregates for a single session.

    These are the live counters displayed by /status.
    They are flushed to the sessions table periodically and on shutdown.
    """

    message_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    tool_calls: int = 0
    tool_successes: int = 0
    tool_errors: int = 0
    api_calls: int = 0

    @property
    def cost_display(self) -> str:
        """Formatted cost string for terminal display."""
        if self.total_cost < 0.01:
            return f"${self.total_cost:.6f}"
        return f"${self.total_cost:.4f}"

    @property
    def success_rate(self) -> float:
        """Tool call success rate as a percentage."""
        if self.tool_calls == 0:
            return 100.0
        return (self.tool_successes / self.tool_calls) * 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EXECUTION LOGGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ExecutionLogger:
    """
    Central observability hub for FLEEA.

    Five concerns, five methods:
    1. log_tool_execution  — record a tool call cycle
    2. log_conversation    — record a user ↔ assistant turn
    3. log_error           — record an error without a tool call
    4. track_api_usage     — accumulate token stats (fast, no DB write)
    5. flush_session_stats — persist in-memory stats to DB
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._session_stats: dict[str, SessionStats] = {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  COST CALCULATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def calculate_cost(
        model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """
        Calculate estimated USD cost for a single API call.

        Uses the MODEL_PRICING table.  Falls back to Sonnet-class pricing
        if the model is unknown — errs on the side of overestimating.
        """
        pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 8)

    @staticmethod
    def extract_usage(response: Any) -> TokenUsage:
        """
        Extract token usage from an Anthropic API response object.

        Safely handles missing attributes — returns empty TokenUsage
        if the response doesn't have usage data (e.g., error responses).
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenUsage()

        prompt = getattr(usage, "input_tokens", 0) or 0
        completion = getattr(usage, "output_tokens", 0) or 0
        model = getattr(response, "model", "")

        cost = ExecutionLogger.calculate_cost(model, prompt, completion)
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            estimated_cost=cost,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  1. TOOL EXECUTION LOGGING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def log_tool_execution(
        self,
        *,
        session_id: str,
        user_input: str,
        detected_intent: str,
        selected_tool: str,
        tool_input: dict | str,
        tool_result: dict | str,
        success: bool,
        error: str | None = None,
        retry_count: int = 0,
        execution_duration_ms: float = 0.0,
        usage: TokenUsage | None = None,
    ) -> int:
        """
        Record a complete tool execution cycle to the database.

        All parameters are keyword-only to prevent positional bugs
        in a function with 12+ args.  Returns the log row ID.
        """
        if usage is None:
            usage = TokenUsage()

        try:
            row_id = self._db.insert_execution_log(
                session_id=session_id,
                user_input=user_input,
                detected_intent=detected_intent,
                selected_tool=selected_tool,
                tool_input=tool_input,
                tool_result=tool_result,
                success=success,
                error=error,
                retry_count=retry_count,
                execution_duration_ms=execution_duration_ms,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost=usage.estimated_cost,
            )
        except Exception as exc:
            # Logging must never crash the agent.  Degrade gracefully.
            _log.error("Failed to write execution log: %s", exc)
            row_id = -1

        # Update in-memory session stats (counts only — token/cost
        # accumulation is handled by track_api_usage() to avoid
        # double-counting, since the same API call triggers both methods)
        stats = self._ensure_stats(session_id)
        stats.tool_calls += 1
        if success:
            stats.tool_successes += 1
        else:
            stats.tool_errors += 1

        return row_id

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  2. CONVERSATION LOGGING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def log_conversation(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_response: str,
        tool_invoked: str | None = None,
    ) -> None:
        """
        Record one complete conversation turn (user → assistant pair).

        Called after the brain produces its final response for a user message.
        """
        try:
            self._db.insert_conversation(
                session_id=session_id,
                user_message=user_message,
                assistant_response=assistant_response,
                tool_invoked=tool_invoked,
            )
        except Exception as exc:
            _log.error("Failed to write conversation log: %s", exc)

        stats = self._ensure_stats(session_id)
        stats.message_count += 1

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  3. ERROR LOGGING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def log_error(
        self,
        *,
        session_id: str,
        user_input: str,
        error: str,
        error_type: str = "unknown",
        selected_tool: str = "_system",
        retry_count: int = 0,
        usage: TokenUsage | None = None,
    ) -> int:
        """
        Record an error that occurred outside of a normal tool call.

        Examples: API connection failure, safety-blocked action,
        unhandled exception in the agentic loop.  Stored in
        execution_logs with success=False for unified querying.
        """
        if usage is None:
            usage = TokenUsage()

        try:
            row_id = self._db.insert_execution_log(
                session_id=session_id,
                user_input=user_input,
                detected_intent=f"error:{error_type}",
                selected_tool=selected_tool,
                tool_input={},
                tool_result={},
                success=False,
                error=error,
                retry_count=retry_count,
                execution_duration_ms=0.0,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost=usage.estimated_cost,
            )
        except Exception as exc:
            _log.error("Failed to write error log: %s", exc)
            row_id = -1

        stats = self._ensure_stats(session_id)
        stats.tool_errors += 1
        stats.total_tokens += usage.total_tokens
        stats.total_cost += usage.estimated_cost

        return row_id

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  4. API USAGE TRACKING (in-memory only, fast)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def track_api_usage(self, session_id: str, usage: TokenUsage) -> None:
        """
        Accumulate token usage from an API call into session stats.

        This is for API calls that don't result in a tool execution
        (e.g., a direct text response with no tool_use).  No DB write —
        just in-memory accumulation for the /status display.
        """
        stats = self._ensure_stats(session_id)
        stats.api_calls += 1
        stats.total_tokens += usage.total_tokens
        stats.total_cost += usage.estimated_cost

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  5. SESSION STATS — read + flush
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_session_stats(self, session_id: str) -> SessionStats:
        """Get live in-memory stats for a session.  Used by /status."""
        return self._ensure_stats(session_id)

    def flush_session_stats(self, session_id: str) -> None:
        """
        Persist in-memory session stats to the sessions table.

        Called periodically by the brain and on graceful shutdown.
        Safe to call multiple times — idempotent update.
        """
        stats = self._session_stats.get(session_id)
        if stats is None:
            return
        try:
            self._db.update_session_stats(
                session_id,
                message_count=stats.message_count,
                total_tokens=stats.total_tokens,
                total_cost=stats.total_cost,
            )
        except Exception as exc:
            _log.error("Failed to flush session stats: %s", exc)

    def check_cost_warning(
        self, session_id: str, session_limit: float, monthly_limit: float
    ) -> str | None:
        """
        Check if session or monthly cost exceeds warning thresholds.

        Returns a warning message string, or None if under budget.
        Used by the brain to inject cost warnings into the REPL.
        """
        stats = self._ensure_stats(session_id)

        if stats.total_cost >= session_limit:
            return (
                f"Session cost ({stats.cost_display}) has reached the "
                f"warning threshold (${session_limit:.2f})."
            )

        # Monthly check: query DB for all sessions this month
        try:
            month_start = datetime.now(timezone.utc).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            ).isoformat()

            with self._db.connection() as conn:
                row = conn.execute(
                    "SELECT COALESCE(SUM(total_cost), 0.0) AS monthly_cost "
                    "FROM sessions WHERE started_at >= ?",
                    (month_start,),
                ).fetchone()

            monthly_cost = (row["monthly_cost"] if row else 0.0) + stats.total_cost
            if monthly_cost >= monthly_limit:
                return (
                    f"Monthly cost (${monthly_cost:.4f}) has reached the "
                    f"warning threshold (${monthly_limit:.2f})."
                )
        except Exception as exc:
            _log.warning("Failed to check monthly cost: %s", exc)

        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  QUERY METHODS (read from DB)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_recent_logs(self, n: int = 20) -> list[dict[str, Any]]:
        """Retrieve the N most recent execution logs (newest first)."""
        return self._db.get_recent_logs(n)

    def get_session_logs(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all execution logs for a specific session."""
        return self._db.get_session_logs(session_id)

    def get_conversation_history(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve full conversation history for a session."""
        return self._db.get_conversation_history(session_id)

    def get_session_cost_summary(self, session_id: str) -> dict[str, Any]:
        """Get aggregate cost data for a session from the database."""
        return self._db.get_session_cost_summary(session_id)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  INTERNAL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _ensure_stats(self, session_id: str) -> SessionStats:
        """Get or create in-memory stats for a session."""
        if session_id not in self._session_stats:
            self._session_stats[session_id] = SessionStats()
        return self._session_stats[session_id]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TIMER UTILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Timer:
    """
    Context-manager timer for measuring tool execution duration.

    Usage:
        with Timer() as t:
            result = await tool.execute(**args)
        print(f"Took {t.duration_ms:.1f}ms")
    """

    __slots__ = ("_start", "duration_ms")

    def __init__(self) -> None:
        self._start: float = 0.0
        self.duration_ms: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.duration_ms = (time.perf_counter() - self._start) * 1000

    def __repr__(self) -> str:
        return f"Timer(duration_ms={self.duration_ms:.2f})"
