"""
FLEEA Database Manager — SQLite schema, operations, and query layer.

Architecture:
    - Pure sqlite3. No ORM, no connection pooling, no abstraction layers.
    - One file, one module, one class.  Easy to read, easy to debug.
    - Context-manager connections: auto-commit on success, rollback on error.
    - WAL journal mode for safe concurrent reads during async operations.
    - All tables use IF NOT EXISTS — safe to call init_db() on every boot.
    - Schema versioned via SCHEMA_VERSION constant for future migrations.

Tables (Phase 1):
    execution_logs   — every tool call: input, output, tokens, cost, timing
    conversations    — paired user message + assistant response per turn
    sessions         — session lifecycle and aggregate stats
    user_profile     — structured user preferences and behavioral data

Tables (future-ready stubs):
    tasks            — Phase 3 task management
    calendar_events  — Phase 3 calendar integration

Usage:
    db = DatabaseManager(settings.db_path_resolved)   # settings injected
    db.insert_execution_log(session_id=..., ...)
    with db.connection() as conn:
        conn.execute("SELECT ...")                    # escape hatch
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


# Bump this when the schema changes.
SCHEMA_VERSION: int = 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATABASE MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DatabaseManager:
    """
    Simple, robust SQLite database manager.

    No pooling — opens a fresh connection per operation via context manager.
    This is the correct design for a single-user desktop app on SQLite:
    connection overhead is ~0.1 ms, pooling adds complexity with zero benefit.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def path(self) -> Path:
        """Absolute path to the database file."""
        return self._db_path

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for safe database operations.

        - Auto-commits on success.
        - Rolls back on any exception.
        - Always closes the connection.
        - Enables WAL mode and foreign keys per connection.
        """
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=10.0,                     # wait up to 10s if locked
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row        # dict-like row access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create all tables on first run.  Idempotent."""
        with self.connection() as conn:
            conn.executescript(_SCHEMA_SQL)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  EXECUTION LOGS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def insert_execution_log(
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
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        estimated_cost: float = 0.0,
    ) -> int:
        """
        Record a single tool execution cycle.

        Every field is keyword-only to prevent positional-argument bugs
        in a function with 14+ parameters.  Returns the row ID.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO execution_logs (
                    session_id, timestamp, user_input, detected_intent,
                    selected_tool, tool_input, tool_result,
                    success, error, retry_count, execution_duration_ms,
                    prompt_tokens, completion_tokens, total_tokens,
                    estimated_cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    _now_iso(),
                    user_input,
                    detected_intent,
                    selected_tool,
                    _to_json(tool_input),
                    _to_json(tool_result),
                    success,
                    error,
                    retry_count,
                    execution_duration_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost,
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_recent_logs(self, n: int = 20) -> list[dict[str, Any]]:
        """Most recent execution logs, newest first."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM execution_logs ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_session_logs(self, session_id: str) -> list[dict[str, Any]]:
        """All execution logs for a specific session, chronological."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM execution_logs WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_session_cost_summary(self, session_id: str) -> dict[str, Any]:
        """Aggregate token and cost stats for a session."""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                   AS tool_calls,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) AS successes,
                    SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) AS failures,
                    COALESCE(SUM(prompt_tokens), 0)     AS total_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                    COALESCE(SUM(total_tokens), 0)      AS total_tokens,
                    COALESCE(SUM(estimated_cost), 0.0)  AS total_cost,
                    COALESCE(SUM(execution_duration_ms), 0.0) AS total_duration_ms
                FROM execution_logs
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            return dict(row) if row else {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  CONVERSATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def insert_conversation(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_response: str,
        tool_invoked: str | None = None,
    ) -> int:
        """
        Record one complete conversation turn (user → assistant pair).

        Storing both sides in a single row makes history reconstruction
        trivial — one SELECT returns the full dialogue in order.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversations (
                    session_id, user_message, assistant_response,
                    tool_invoked, timestamp
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_message, assistant_response, tool_invoked, _now_iso()),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_conversation_history(self, session_id: str) -> list[dict[str, Any]]:
        """Full conversation history for a session, chronological."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT user_message, assistant_response, tool_invoked, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_recent_conversations(self, session_id: str, n: int = 10) -> list[dict[str, Any]]:
        """Last N conversation turns for a session (for context windowing)."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT user_message, assistant_response, tool_invoked, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, n),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]  # restore chronological order

    def count_conversations(self, session_id: str) -> int:
        """Number of conversation turns in a session."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM conversations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["cnt"] if row else 0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SESSIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def insert_session(self, session_id: str) -> None:
        """Record a new session start."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, started_at, message_count,
                                      total_tokens, total_cost)
                VALUES (?, ?, 0, 0, 0.0)
                """,
                (session_id, _now_iso()),
            )

    def update_session_stats(
        self,
        session_id: str,
        message_count: int,
        total_tokens: int,
        total_cost: float,
    ) -> None:
        """Update running session statistics."""
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET message_count = ?, total_tokens = ?, total_cost = ?
                WHERE session_id = ?
                """,
                (message_count, total_tokens, total_cost, session_id),
            )

    def update_session_message_count(
        self,
        session_id: str,
        message_count: int,
    ) -> None:
        """Update only the message count — leaves token/cost untouched."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE sessions SET message_count = ? WHERE session_id = ?",
                (message_count, session_id),
            )

    def end_session(self, session_id: str) -> None:
        """Mark a session as ended with current timestamp."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                (_now_iso(), session_id),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve session details, or None if not found."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  USER PROFILE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def set_profile(self, category: str, key: str, value: str) -> None:
        """
        Upsert a user profile entry.

        Categories group related preferences:
            identity     — name, timezone, location
            work_style   — deep-focus hours, communication preferences
            schedule     — preferred meeting hours, availability windows
            writing      — tone, formality, signature
            habits       — recurring tasks, daily routines
            priorities   — task ranking rules, urgency definitions
            preferences  — theme, language, notification settings

        The (category, key) pair is unique.  Repeated calls overwrite.
        """
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO user_profile (category, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(category, key)
                DO UPDATE SET value = excluded.value,
                             updated_at = excluded.updated_at
                """,
                (category, key, value, _now_iso()),
            )

    def get_profile_value(self, category: str, key: str) -> str | None:
        """Get a single profile value, or None if not set."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT value FROM user_profile WHERE category = ? AND key = ?",
                (category, key),
            ).fetchone()
            return row["value"] if row else None

    def get_profile_category(self, category: str) -> dict[str, str]:
        """Get all key-value pairs in a profile category."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM user_profile WHERE category = ? ORDER BY key",
                (category,),
            ).fetchall()
            return {row["key"]: row["value"] for row in rows}

    def get_full_profile(self) -> dict[str, dict[str, str]]:
        """
        Get the entire user profile, organized by category.

        Returns:
            {
                "identity":    {"name": "John", "timezone": "US/Eastern"},
                "work_style":  {"focus_hours": "9AM-12PM", ...},
                "schedule":    {"meeting_hours": "2PM-5PM", ...},
                ...
            }
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT category, key, value FROM user_profile ORDER BY category, key"
            ).fetchall()

        profile: dict[str, dict[str, str]] = {}
        for row in rows:
            cat = row["category"]
            if cat not in profile:
                profile[cat] = {}
            profile[cat][row["key"]] = row["value"]
        return profile

    def delete_profile_entry(self, category: str, key: str) -> bool:
        """Delete a single profile entry.  Returns True if a row was removed."""
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM user_profile WHERE category = ? AND key = ?",
                (category, key),
            )
            return cursor.rowcount > 0

    def profile_to_context_string(self) -> str:
        """
        Format the entire profile as a markdown string for system prompt injection.

        Example output:
            ### Identity
            - **name**: John Doe
            - **timezone**: US/Eastern

            ### Work Style
            - **focus_hours**: 9AM–12PM
            - **communication**: Concise, direct
        """
        profile = self.get_full_profile()
        if not profile:
            return ""

        lines: list[str] = []
        for category, entries in profile.items():
            # Convert category slug to title case
            title = category.replace("_", " ").title()
            lines.append(f"### {title}")
            for key, value in entries.items():
                display_key = key.replace("_", " ")
                lines.append(f"- **{display_key}**: {value}")
            lines.append("")  # blank line between categories

        return "\n".join(lines)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  AUTONOMOUS TASKS (Phase 4 — Supervised Autonomy)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    #  Maps TaskStep fields to the existing `tasks` table:
    #      step_id         → title (UUID, unique identifier)
    #      goal_id         → session_id (groups steps under one goal)
    #      instruction     → description
    #      status          → status
    #      sequence        → priority (reused for ordering)
    #      result          → tags (JSON — stores result + metadata)
    #      requires_approval → recurrence (reused as flag field)

    def insert_task_step(
        self,
        *,
        step_id: str,
        goal_id: str,
        sequence: int,
        instruction: str,
        status: str = "pending",
        requires_approval: bool = False,
    ) -> int:
        """
        Persist a single autonomous task step.

        Returns the auto-incremented row ID.
        """
        now = _now_iso()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tasks (
                    session_id, title, description, status, priority,
                    recurrence, tags, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal_id,
                    step_id,            # title = step UUID
                    instruction,        # description = instruction text
                    status,
                    str(sequence),      # priority = sequence number
                    "approval_required" if requires_approval else "auto",
                    json.dumps({"retry_count": 0}),
                    now,
                    now,
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def update_task_status(
        self,
        step_id: str,
        *,
        status: str,
        result: str | None = None,
        error: str | None = None,
        retry_count: int | None = None,
    ) -> None:
        """
        Transition a task step to a new status.

        Atomically updates status, result/error metadata, and timestamp.
        """
        now = _now_iso()
        completed_at = now if status in ("completed", "failed", "skipped") else None

        # Build tags JSON with result/error/retry metadata
        tags_data: dict[str, Any] = {}
        if result is not None:
            tags_data["result"] = result[:5000]  # cap stored result size
        if error is not None:
            tags_data["error"] = error[:2000]
        if retry_count is not None:
            tags_data["retry_count"] = retry_count
        tags_json = json.dumps(tags_data, default=str) if tags_data else None

        with self.connection() as conn:
            if tags_json and completed_at:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, tags = ?, completed_at = ?, updated_at = ?
                    WHERE title = ?
                    """,
                    (status, tags_json, completed_at, now, step_id),
                )
            elif tags_json:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, tags = ?, updated_at = ?
                    WHERE title = ?
                    """,
                    (status, tags_json, now, step_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, updated_at = ?
                    WHERE title = ?
                    """,
                    (status, now, step_id),
                )

    def get_pending_tasks(self, goal_id: str) -> list[dict[str, Any]]:
        """
        Get all pending task steps for a goal, ordered by sequence.

        Returns dicts with keys: step_id, goal_id, sequence, instruction,
        status, requires_approval, tags, created_at.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT title AS step_id, session_id AS goal_id,
                       CAST(priority AS INTEGER) AS sequence,
                       description AS instruction, status,
                       recurrence AS approval_flag, tags,
                       created_at, completed_at
                FROM tasks
                WHERE session_id = ? AND status = 'pending'
                ORDER BY CAST(priority AS INTEGER) ASC
                """,
                (goal_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_tasks_by_goal(self, goal_id: str) -> list[dict[str, Any]]:
        """Get all task steps for a goal, ordered by sequence."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT title AS step_id, session_id AS goal_id,
                       CAST(priority AS INTEGER) AS sequence,
                       description AS instruction, status,
                       recurrence AS approval_flag, tags,
                       created_at, completed_at
                FROM tasks
                WHERE session_id = ?
                ORDER BY CAST(priority AS INTEGER) ASC
                """,
                (goal_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_task_step(self, step_id: str) -> dict[str, Any] | None:
        """Get a single task step by its step_id (title)."""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT title AS step_id, session_id AS goal_id,
                       CAST(priority AS INTEGER) AS sequence,
                       description AS instruction, status,
                       recurrence AS approval_flag, tags,
                       created_at, completed_at
                FROM tasks
                WHERE title = ?
                """,
                (step_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_task_summary(self, goal_id: str) -> dict[str, int]:
        """
        Aggregate status counts for all steps in a goal.

        Returns: {total, pending, in_progress, completed, failed, skipped}
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS cnt
                FROM tasks
                WHERE session_id = ?
                GROUP BY status
                """,
                (goal_id,),
            ).fetchall()

        summary = {
            "total": 0,
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
        }
        for row in rows:
            status = row["status"]
            count = row["cnt"]
            summary["total"] += count
            if status in summary:
                summary[status] = count

        return summary

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  USER AUTHENTICATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_user(
        self, user_id: str, name: str, email: str, password_hash: str
    ) -> bool:
        """Create a new user. Returns True if successful, False if user_id exists."""
        with self.connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO users (user_id, name, email, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, name, email, password_hash, _now_iso()),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def authenticate_user(self, user_id: str) -> dict[str, Any] | None:
        """Fetch user by user_id for authentication."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_users(self) -> list[dict[str, Any]]:
        """Fetch all registered users for admin dashboard."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT id, user_id, name, email, role, created_at FROM users ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_user(self, user_id: str) -> bool:
        """Delete a user. Returns True if a user was deleted."""
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            return cursor.rowcount > 0

    def update_user(self, user_id: str, name: str, email: str) -> bool:
        """Update user profile info. Returns True if successful."""
        with self.connection() as conn:
            try:
                cursor = conn.execute(
                    "UPDATE users SET name = ?, email = ? WHERE user_id = ?",
                    (name, email, user_id)
                )
                return cursor.rowcount > 0
            except sqlite3.IntegrityError:
                return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  HEALTH / DIAGNOSTICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def health_check(self) -> dict[str, Any]:
        """
        Quick diagnostic: table row counts, DB file size, schema version.
        Used by the /status CLI command.
        """
        tables = [
            "users", "execution_logs", "conversations", "sessions",
            "user_profile", "tasks", "calendar_events",
        ]
        counts: dict[str, int] = {}

        with self.connection() as conn:
            for table in tables:
                try:
                    row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
                    counts[table] = row["cnt"] if row else 0
                except sqlite3.OperationalError:
                    counts[table] = -1  # table doesn't exist yet

        file_size_mb = self._db_path.stat().st_size / (1024 * 1024) if self._db_path.exists() else 0.0

        return {
            "schema_version": SCHEMA_VERSION,
            "db_path": str(self._db_path),
            "file_size_mb": round(file_size_mb, 3),
            "table_counts": counts,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _to_json(data: Any) -> str:
    """
    Safely serialize data to a JSON string for storage.
    Already-serialized strings pass through unchanged.
    """
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SCHEMA — All CREATE TABLE statements
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SCHEMA_SQL: str = """

-- ─── USERS ───────────────────────────────────────────────────────
-- Registered users for authentication and multi-tenant persistence.

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT    UNIQUE NOT NULL,
    name          TEXT    NOT NULL,
    email         TEXT    UNIQUE NOT NULL,
    password_hash TEXT    NOT NULL,
    role          TEXT    DEFAULT 'user',
    created_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_userid ON users(user_id);
CREATE INDEX IF NOT EXISTS idx_users_email  ON users(email);


-- ─── EXECUTION LOGS ──────────────────────────────────────────────
-- Every tool call the agent makes is recorded here.
-- This is the single most important table for debugging and cost control.

CREATE TABLE IF NOT EXISTS execution_logs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id            TEXT    NOT NULL,
    timestamp             TEXT    NOT NULL,

    -- What the user said and what the agent understood
    user_input            TEXT,
    detected_intent       TEXT,

    -- Which tool was selected and what happened
    selected_tool         TEXT    NOT NULL,
    tool_input            TEXT,                    -- JSON of arguments sent to tool
    tool_result           TEXT,                    -- JSON of tool response
    success               BOOLEAN NOT NULL DEFAULT 0,
    error                 TEXT,                    -- error message if failed

    -- Resilience tracking
    retry_count           INTEGER DEFAULT 0,
    execution_duration_ms REAL    DEFAULT 0.0,     -- wall-clock time for tool execution

    -- Token accounting (from Claude API response.usage)
    prompt_tokens         INTEGER DEFAULT 0,
    completion_tokens     INTEGER DEFAULT 0,
    total_tokens          INTEGER DEFAULT 0,

    -- Cost tracking (USD, calculated from model pricing table)
    estimated_cost        REAL    DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_exec_session   ON execution_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_exec_timestamp ON execution_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_exec_tool      ON execution_logs(selected_tool);


-- ─── CONVERSATIONS ───────────────────────────────────────────────
-- Each row is one complete turn: what the user said + what FLEEA replied.
-- Pairing both sides in one row simplifies history queries.

CREATE TABLE IF NOT EXISTS conversations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT    NOT NULL,
    user_message        TEXT    NOT NULL,
    assistant_response  TEXT    NOT NULL,
    tool_invoked        TEXT,                      -- which tool was used (NULL if none)
    timestamp           TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);


-- ─── SESSIONS ────────────────────────────────────────────────────
-- Lifecycle tracking for each REPL run.

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    UNIQUE NOT NULL,
    started_at      TEXT    NOT NULL,
    ended_at        TEXT,
    message_count   INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    total_cost      REAL    DEFAULT 0.0
);


-- ─── USER PROFILE ────────────────────────────────────────────────
-- Structured preference store, organized by category.
--
-- Predefined categories:
--   identity     — name, timezone, location
--   work_style   — deep-focus hours, communication preferences, productivity patterns
--   schedule     — preferred meeting hours, availability windows, blocked times
--   writing      — tone, formality level, signature, preferred language
--   habits       — recurring tasks, daily routines, health reminders
--   priorities   — task ranking rules, urgency thresholds, project order
--   preferences  — UI theme, notification settings, default search engine
--
-- The (category, key) pair enforces uniqueness.  Upsert via ON CONFLICT.

CREATE TABLE IF NOT EXISTS user_profile (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    category    TEXT    NOT NULL,
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,

    UNIQUE(category, key)
);

CREATE INDEX IF NOT EXISTS idx_profile_category ON user_profile(category);


-- ─── TASKS (Phase 3 — schema ready) ─────────────────────────────
-- Natural-language task management with priority and recurrence.

CREATE TABLE IF NOT EXISTS tasks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,                          -- session that created it (NULL if via API)
    title           TEXT    NOT NULL,
    description     TEXT,
    status          TEXT    NOT NULL DEFAULT 'pending',   -- pending, in_progress, done, cancelled
    priority        TEXT    NOT NULL DEFAULT 'medium',    -- low, medium, high, urgent
    due_date        TEXT,                          -- ISO 8601 datetime or NULL
    recurrence      TEXT,                          -- cron-like pattern or NULL
    tags            TEXT,                          -- JSON array of tag strings
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    completed_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status   ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
CREATE INDEX IF NOT EXISTS idx_tasks_due      ON tasks(due_date);


-- ─── CALENDAR EVENTS (Phase 3 — schema ready) ───────────────────
-- Local mirror of calendar events for search and context.

CREATE TABLE IF NOT EXISTS calendar_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id     TEXT,                          -- Google Calendar event ID
    title           TEXT    NOT NULL,
    description     TEXT,
    location        TEXT,
    start_time      TEXT    NOT NULL,              -- ISO 8601
    end_time        TEXT    NOT NULL,              -- ISO 8601
    all_day         BOOLEAN DEFAULT 0,
    source          TEXT    NOT NULL DEFAULT 'local',  -- local, google, outlook
    synced_at       TEXT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cal_start  ON calendar_events(start_time);
CREATE INDEX IF NOT EXISTS idx_cal_source ON calendar_events(source);

"""
