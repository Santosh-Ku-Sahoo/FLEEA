"""
FLEEA Session Manager — Session lifecycle, metadata tracking, and persistence.

Architecture:
    Every REPL run is a session.  The SessionManager owns the session
    lifecycle from creation to teardown.  All logs, conversations, and
    cost data are tagged with the session_id, making it the primary key
    for debugging and observability.

    The Session dataclass is the in-memory representation of session state.
    It is authoritative during a session — the database is updated from it,
    not the other way around.  This avoids unnecessary reads.

Design:
    - DatabaseManager injected via constructor (no globals).
    - One active session at a time (single-user desktop app).
    - Session state held in memory, flushed to SQLite on:
          1. Every flush_to_db() call (periodic)
          2. end_session() (graceful shutdown)
    - UUID4 for session IDs — universally unique, no collisions.
    - All timestamps are UTC, stored as ISO 8601.
    - Integration-ready: logger.py reads session_id from here;
      app.py calls create/end; brain.py calls record_interaction.

Usage:
    sm = SessionManager(db)
    session = sm.create_session()
    sm.record_interaction(tokens=450, cost=0.005, tool_used=True)
    sm.record_interaction(tokens=200, cost=0.002, tool_used=False)
    completed = sm.end_session()
    print(completed.summary_table())
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from database.models import DatabaseManager


_log = logging.getLogger("fleea.session")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SESSION STATUS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SessionStatus(Enum):
    """Lifecycle states of a session."""

    ACTIVE = "active"
    ENDED = "ended"
    CRASHED = "crashed"  # set if the process exits without end_session()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SESSION DATA CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Session:
    """
    In-memory representation of a single FLEEA session.

    This is the single source of truth while the session is active.
    The database is updated FROM this object, never the reverse.
    """

    # ── Identity ──
    session_id: str
    created_at: datetime

    # ── Lifecycle ──
    status: SessionStatus = SessionStatus.ACTIVE
    ended_at: datetime | None = None
    last_active: datetime | None = None

    # ── Counters ──
    total_interactions: int = 0     # user messages processed
    total_tool_calls: int = 0       # tools invoked
    total_tokens: int = 0           # cumulative prompt + completion
    total_cost: float = 0.0         # USD estimate

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DERIVED PROPERTIES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def short_id(self) -> str:
        """First 8 characters of the UUID for display."""
        return self.session_id[:8]

    @property
    def is_active(self) -> bool:
        return self.status == SessionStatus.ACTIVE

    @property
    def duration_seconds(self) -> float:
        """Wall-clock seconds since session creation."""
        end = self.ended_at or datetime.now(timezone.utc)
        return (end - self.created_at).total_seconds()

    @property
    def duration_display(self) -> str:
        """Human-readable session duration."""
        total = int(self.duration_seconds)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    @property
    def cost_display(self) -> str:
        """Formatted cost string — more precision when cost is small."""
        if self.total_cost < 0.01:
            return f"${self.total_cost:.6f}"
        return f"${self.total_cost:.4f}"

    @property
    def avg_tokens_per_interaction(self) -> float:
        """Average tokens used per user interaction."""
        if self.total_interactions == 0:
            return 0.0
        return self.total_tokens / self.total_interactions

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DISPLAY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state for API responses or logging."""
        return {
            "session_id": self.session_id,
            "short_id": self.short_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration": self.duration_display,
            "total_interactions": self.total_interactions,
            "total_tool_calls": self.total_tool_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.cost_display,
            "avg_tokens_per_interaction": round(self.avg_tokens_per_interaction, 1),
        }

    def summary_lines(self) -> list[str]:
        """
        Generate a list of summary lines for terminal display.
        Used by app.py for the session-end recap.
        """
        return [
            f"Session:      {self.short_id}",
            f"Status:       {self.status.value}",
            f"Duration:     {self.duration_display}",
            f"Interactions: {self.total_interactions}",
            f"Tool calls:   {self.total_tool_calls}",
            f"Tokens used:  {self.total_tokens:,}",
            f"Est. cost:    {self.cost_display}",
        ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SESSION MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SessionManager:
    """
    Manages session lifecycle: creation → tracking → teardown.

    - One active session at a time.
    - All sessions persisted to SQLite.
    - In-memory Session dataclass is the live source of truth.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._current: Session | None = None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SESSION CREATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_session(self) -> Session:
        """
        Create and register a new session.

        If a previous session is still active, it is force-ended first
        (defensive — should not happen in normal flow).
        """
        if self._current is not None and self._current.is_active:
            _log.warning(
                "Force-ending previous session %s before creating new one",
                self._current.short_id,
            )
            self.end_session()

        now = datetime.now(timezone.utc)
        session = Session(
            session_id=str(uuid.uuid4()),
            created_at=now,
            last_active=now,
        )

        try:
            self._db.insert_session(session.session_id)
        except Exception as exc:
            _log.error("Failed to persist new session: %s", exc)
            # Continue anyway — in-memory tracking still works

        self._current = session
        _log.info("Session created: %s", session.short_id)
        return session

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  ACTIVE SESSION RETRIEVAL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_current_session(self) -> Session | None:
        """Get the current active session, or None."""
        return self._current

    @property
    def session_id(self) -> str | None:
        """Convenience: current session ID, or None."""
        return self._current.session_id if self._current else None

    @property
    def short_id(self) -> str | None:
        """Convenience: current short session ID, or None."""
        return self._current.short_id if self._current else None

    @property
    def is_active(self) -> bool:
        """Whether a session is currently running."""
        return self._current is not None and self._current.is_active

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  ACTIVITY UPDATES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def record_interaction(
        self,
        *,
        tool_used: bool = False,
    ) -> None:
        """
        Record one user interaction (one complete request → response cycle).

        Called by the brain after processing each user message.
        Updates interaction count and the last_active timestamp.

        Note: Token/cost tracking is owned exclusively by ExecutionLogger.
        SessionManager only tracks lifecycle counters.
        """
        if self._current is None:
            _log.warning("record_interaction called with no active session")
            return

        self._current.total_interactions += 1
        self._current.last_active = datetime.now(timezone.utc)

        if tool_used:
            self._current.total_tool_calls += 1

    def record_tool_call(self) -> None:
        """
        Record a single tool call within the current interaction.

        Use this when the brain invokes multiple tools per interaction.
        record_interaction() should still be called once per user message.
        """
        if self._current is None:
            return

        self._current.total_tool_calls += 1
        self._current.last_active = datetime.now(timezone.utc)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DATABASE FLUSH
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def flush_to_db(self) -> None:
        """
        Persist current in-memory interaction count to SQLite.

        Only writes message_count.  Token/cost persistence is handled
        exclusively by ExecutionLogger.flush_session_stats().
        Safe to call multiple times — idempotent update.
        """
        if self._current is None:
            return

        try:
            self._db.update_session_message_count(
                self._current.session_id,
                message_count=self._current.total_interactions,
            )
        except Exception as exc:
            _log.error("Failed to flush session stats: %s", exc)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SESSION CLOSING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def end_session(self) -> Session | None:
        """
        Gracefully end the current session.

        1. Sets ended_at timestamp and status.
        2. Marks session as ended in the database.
        3. Clears the current session reference.
        4. Returns the completed Session for summary display.

        Note: The caller (app.py) must call
        ExecutionLogger.flush_session_stats() BEFORE this method to
        persist token/cost data.  This method only persists lifecycle.
        """
        if self._current is None:
            return None

        self._current.ended_at = datetime.now(timezone.utc)
        self._current.status = SessionStatus.ENDED

        # Persist interaction count
        self.flush_to_db()

        try:
            self._db.end_session(self._current.session_id)
        except Exception as exc:
            _log.error("Failed to mark session as ended: %s", exc)

        completed = self._current
        self._current = None

        _log.info(
            "Session ended: %s | %s | %d interactions",
            completed.short_id,
            completed.duration_display,
            completed.total_interactions,
        )
        return completed

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DIAGNOSTICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_session_info(self) -> dict[str, Any] | None:
        """
        Get current session as a serializable dict.
        Used by /status command.  Returns None if no session.
        """
        if self._current is None:
            return None
        return self._current.to_dict()

    def get_past_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a past session's data from the database."""
        return self._db.get_session(session_id)
