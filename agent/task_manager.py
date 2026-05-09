"""
FLEEA Task Manager — Persistent state machine for autonomous task steps.

Architecture:
    The TaskManager is the state-management layer of the autonomous pipeline.
    It owns the lifecycle of TaskSteps — from initial persistence through
    status transitions to completion or failure.

    All state is persisted to SQLite via DatabaseManager, making task state
    survive application restarts.  The in-memory TaskStep dataclasses are
    ephemeral views — the database is the source of truth.

Design:
    - DatabaseManager injected via constructor (no globals).
    - Every state transition is persisted atomically.
    - Retry budget enforcement prevents infinite retry loops.
    - GoalSummary aggregates provide at-a-glance progress.
    - All methods are synchronous (DB ops are fast, < 1ms).

State Machine:
    PENDING → IN_PROGRESS → COMPLETED
                          → FAILED → (retry) → IN_PROGRESS
                          → SKIPPED
    PENDING → AWAITING_APPROVAL → IN_PROGRESS
                                → SKIPPED

Usage:
    manager = TaskManager(db, max_retries=2)
    manager.save_plan(steps)
    step = manager.get_next_pending(goal_id)
    manager.mark_in_progress(step.step_id)
    manager.mark_completed(step.step_id, result="Done")
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.planner import GoalSummary, TaskStatus, TaskStep
from database.models import DatabaseManager


_log = logging.getLogger("fleea.task_manager")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TASK MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TaskManager:
    """
    Persistent task state machine for autonomous execution.

    Provides:
        1. Plan persistence     → save_plan()
        2. Step retrieval       → get_next_pending(), get_all_steps()
        3. State transitions    → mark_in_progress(), mark_completed(), etc.
        4. Retry management     → can_retry(), increment_retry()
        5. Goal progress        → get_goal_summary()

    All operations go through DatabaseManager for persistence.
    """

    def __init__(self, db: DatabaseManager, *, max_retries: int = 2) -> None:
        self._db = db
        self._max_retries = max_retries

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  1. PLAN PERSISTENCE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def save_plan(self, steps: list[TaskStep]) -> None:
        """
        Persist all steps of a plan to the database.

        Each step is inserted as a new row in the tasks table.
        Steps must have unique step_ids (enforced by the Planner).
        """
        for step in steps:
            try:
                self._db.insert_task_step(
                    step_id=step.step_id,
                    goal_id=step.goal_id,
                    sequence=step.sequence,
                    instruction=step.instruction,
                    status=step.status.value,
                    requires_approval=step.requires_approval,
                )
            except Exception as exc:
                _log.error(
                    "Failed to persist step %d [%s]: %s",
                    step.sequence,
                    step.step_id[:8],
                    exc,
                )
                raise

        _log.info(
            "Saved plan: goal=%s  steps=%d",
            steps[0].goal_id[:8] if steps else "?",
            len(steps),
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  2. STEP RETRIEVAL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_next_pending(self, goal_id: str) -> TaskStep | None:
        """
        Get the next pending step for a goal (lowest sequence number).

        Returns None if no pending steps remain.
        """
        rows = self._db.get_pending_tasks(goal_id)
        if not rows:
            return None
        return self._row_to_step(rows[0])

    def get_all_steps(self, goal_id: str) -> list[TaskStep]:
        """Get all steps for a goal, ordered by sequence."""
        rows = self._db.get_tasks_by_goal(goal_id)
        return [self._row_to_step(row) for row in rows]

    def get_step(self, step_id: str) -> TaskStep | None:
        """Get a single step by its step_id."""
        row = self._db.get_task_step(step_id)
        return self._row_to_step(row) if row else None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  3. STATE TRANSITIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def mark_in_progress(self, step_id: str) -> None:
        """Transition a step to IN_PROGRESS."""
        self._db.update_task_status(
            step_id,
            status=TaskStatus.IN_PROGRESS.value,
        )
        _log.debug("Step %s → IN_PROGRESS", step_id[:8])

    def mark_completed(self, step_id: str, result: str) -> None:
        """Transition a step to COMPLETED with its result."""
        self._db.update_task_status(
            step_id,
            status=TaskStatus.COMPLETED.value,
            result=result,
        )
        _log.info("Step %s → COMPLETED", step_id[:8])

    def mark_failed(self, step_id: str, error: str) -> None:
        """Transition a step to FAILED with its error."""
        self._db.update_task_status(
            step_id,
            status=TaskStatus.FAILED.value,
            error=error,
        )
        _log.warning("Step %s → FAILED: %s", step_id[:8], error[:100])

    def mark_skipped(self, step_id: str, reason: str = "User declined") -> None:
        """Transition a step to SKIPPED."""
        self._db.update_task_status(
            step_id,
            status=TaskStatus.SKIPPED.value,
            error=reason,
        )
        _log.info("Step %s → SKIPPED: %s", step_id[:8], reason)

    def mark_awaiting_approval(self, step_id: str) -> None:
        """Transition a step to AWAITING_APPROVAL."""
        self._db.update_task_status(
            step_id,
            status=TaskStatus.AWAITING_APPROVAL.value,
        )
        _log.debug("Step %s → AWAITING_APPROVAL", step_id[:8])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  4. RETRY MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def can_retry(self, step_id: str) -> bool:
        """Check if a step has retry budget remaining."""
        step = self.get_step(step_id)
        if step is None:
            return False
        return step.retry_count < self._max_retries

    def increment_retry(self, step_id: str) -> int:
        """
        Increment retry count and reset status to PENDING.

        Returns the new retry count.
        """
        step = self.get_step(step_id)
        if step is None:
            return 0

        new_count = step.retry_count + 1
        self._db.update_task_status(
            step_id,
            status=TaskStatus.PENDING.value,
            retry_count=new_count,
        )
        _log.info(
            "Step %s retry %d/%d → PENDING",
            step_id[:8],
            new_count,
            self._max_retries,
        )
        return new_count

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  5. GOAL PROGRESS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_goal_summary(self, goal_id: str) -> GoalSummary:
        """Get aggregate progress for a goal."""
        raw = self._db.get_task_summary(goal_id)
        return GoalSummary(
            goal_id=goal_id,
            total_steps=raw.get("total", 0),
            completed=raw.get("completed", 0),
            failed=raw.get("failed", 0),
            pending=raw.get("pending", 0),
            skipped=raw.get("skipped", 0),
            in_progress=raw.get("in_progress", 0),
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  INTERNAL HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _row_to_step(row: dict[str, Any]) -> TaskStep:
        """Convert a DB row dict into a TaskStep dataclass."""
        # Parse tags JSON for result/error/retry metadata
        tags_str = row.get("tags") or "{}"
        try:
            tags = json.loads(tags_str) if isinstance(tags_str, str) else {}
        except (json.JSONDecodeError, TypeError):
            tags = {}

        # Map approval flag
        approval_flag = row.get("approval_flag", "auto")
        requires_approval = approval_flag == "approval_required"

        # Map status string to enum
        status_str = row.get("status", "pending")
        try:
            status = TaskStatus(status_str)
        except ValueError:
            status = TaskStatus.PENDING

        return TaskStep(
            step_id=row["step_id"],
            goal_id=row["goal_id"],
            sequence=int(row.get("sequence", 0)),
            instruction=row.get("instruction", ""),
            status=status,
            result=tags.get("result"),
            error=tags.get("error"),
            retry_count=int(tags.get("retry_count", 0)),
            created_at=row.get("created_at", ""),
            completed_at=row.get("completed_at"),
            requires_approval=requires_approval,
        )
