"""
FLEEA Task Planner — LLM-powered goal decomposition into actionable steps.

Architecture:
    The Planner is the first stage of the autonomous agent pipeline.
    Given a high-level user goal, it calls Claude with a structured prompt
    that decomposes the goal into 3–7 concrete, actionable TaskSteps.

    Each TaskStep is a self-contained instruction that the Executor can
    feed directly to Brain.think().  Steps are ordered by dependency and
    flagged for approval if they might trigger sensitive operations.

    The Planner does NOT execute anything — it only plans.

Design:
    - Settings injected via constructor (API key, model).
    - Output is a list of TaskStep dataclasses with PENDING status.
    - Claude is forced to output JSON via a structured prompt.
    - Fallback: if Claude returns unparseable output, wrap goal as single step.
    - TaskStatus and TaskStep are defined here as shared types.

Usage:
    planner = TaskPlanner(settings)
    steps = planner.create_plan("Research AI trends and write a summary")
    # → [TaskStep(sequence=1, ...), TaskStep(sequence=2, ...), ...]
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import anthropic

from config.settings import Settings


_log = logging.getLogger("fleea.planner")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHARED DATA TYPES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TaskStatus(Enum):
    """Lifecycle states of an autonomous task step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"                # dependency failed or user declined
    AWAITING_APPROVAL = "awaiting_approval"


@dataclass
class TaskStep:
    """
    A single actionable step within an autonomous execution plan.

    Fields map to the existing `tasks` table in SQLite:
        title       → instruction
        description → goal context
        status      → TaskStatus.value
        priority    → 'medium' (default)
        session_id  → goal_id
        tags        → JSON metadata
    """

    step_id: str                            # UUID — unique per step
    goal_id: str                            # groups steps under one goal
    sequence: int                           # 1-based execution order
    instruction: str                        # concrete action for Brain
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None               # Brain's response on success
    error: str | None = None                # error message on failure
    retry_count: int = 0                    # times retried
    created_at: str = ""                    # ISO timestamp
    completed_at: str | None = None
    requires_approval: bool = False         # pre-flagged by Planner

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class GoalSummary:
    """Aggregate status of all steps in a goal."""

    goal_id: str
    total_steps: int = 0
    completed: int = 0
    failed: int = 0
    pending: int = 0
    skipped: int = 0
    in_progress: int = 0

    @property
    def is_done(self) -> bool:
        """True when no steps are pending or in progress."""
        return self.pending == 0 and self.in_progress == 0

    @property
    def success_rate(self) -> float:
        """Percentage of steps that completed successfully."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed / self.total_steps) * 100


# ── Backward compatibility — old Plan dataclass for existing callers ──

@dataclass
class Plan:
    """Legacy structured execution plan (Phase 1 compat)."""

    steps: list[str]
    is_multi_step: bool = False
    original_request: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PLANNING PROMPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_PLANNING_PROMPT: str = """\
You are a task planner for an AI agent called FLEEA. Your job is to break down \
a user's high-level goal into a sequence of concrete, actionable steps.

## Rules
1. Generate between 3 and 7 steps. Never fewer, never more.
2. Each step must be a specific, self-contained instruction — NOT vague.
   - BAD:  "Analyze the data"
   - GOOD: "Search the web for the top 5 AI trends in 2025"
3. Steps must be ordered by dependency — step N can only depend on steps 1..N-1.
4. Each step should describe ONE action, not multiple.
5. Mark `requires_approval` as true for steps that involve:
   - Writing or modifying files
   - Sending communications (email, messages)
   - Any financial operations
   - System commands or installations
   - Any action that changes external state
6. Mark `requires_approval` as false for steps that only:
   - Search the web
   - Read or summarize information
   - Perform calculations
   - Answer questions

## Available Tools
The agent has access to: web_search. More tools may be added in the future.
Plan steps that leverage these capabilities.

## Output Format
Respond with ONLY a JSON array — no markdown, no explanation, no code fences.
Each element must have these exact keys:
- "instruction": string — the concrete action to perform
- "requires_approval": boolean — whether this step needs user confirmation

Example:
[
  {"instruction": "Search the web for the latest developments in quantum computing", "requires_approval": false},
  {"instruction": "Summarize the top 3 findings into a concise briefing", "requires_approval": false},
  {"instruction": "Draft a one-page report with key insights and future outlook", "requires_approval": false}
]

## User Goal
{goal}
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TASK PLANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TaskPlanner:
    """
    LLM-powered goal decomposition engine.

    Takes a natural-language goal and returns a list of concrete
    TaskStep instances ready for the Executor.

    The Planner only plans — it never executes.  All tool invocations
    happen downstream through the Executor → Brain pipeline.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = anthropic.Anthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=settings.API_TIMEOUT_SECONDS,
        )
        self._model = settings.DEFAULT_MODEL
        self._max_tokens = 1024  # plans are short — save tokens

    def create_plan(self, goal: str) -> list[TaskStep]:
        """
        Decompose a goal into 3–7 actionable TaskSteps.

        Args:
            goal: High-level natural-language goal from the user.

        Returns:
            List of TaskStep instances in execution order,
            each with status=PENDING and a unique step_id.
        """
        goal_id = str(uuid.uuid4())

        _log.info("Planning goal [%s]: %s", goal_id[:8], goal[:100])

        try:
            raw_steps = self._call_llm(goal)
            steps = self._parse_steps(raw_steps, goal_id)
        except Exception as exc:
            _log.warning(
                "Plan generation failed (%s), falling back to single step: %s",
                type(exc).__name__,
                exc,
            )
            steps = self._fallback_plan(goal, goal_id)

        # Clamp to 3–7 steps
        if len(steps) < 3:
            _log.info("Plan has %d steps (< 3), padding with synthesis step", len(steps))
            steps = self._ensure_minimum_steps(steps, goal, goal_id)
        elif len(steps) > 7:
            _log.info("Plan has %d steps (> 7), truncating to 7", len(steps))
            steps = steps[:7]
            # Re-sequence
            for i, step in enumerate(steps, start=1):
                step.sequence = i

        _log.info(
            "Plan created [%s]: %d steps",
            goal_id[:8],
            len(steps),
        )
        return steps

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  LLM INTERACTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _call_llm(self, goal: str) -> str:
        """Call Claude to generate a task plan as JSON."""
        prompt = _PLANNING_PROMPT.format(goal=goal)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        text_parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        return "\n".join(text_parts).strip()

    def _parse_steps(
        self, raw_json: str, goal_id: str
    ) -> list[TaskStep]:
        """
        Parse Claude's JSON output into TaskStep instances.

        Handles common LLM quirks:
        - Markdown code fences around JSON
        - Extra text before/after the array
        - Missing optional fields
        """
        # Strip markdown code fences if present
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Find the JSON array boundaries
        start_idx = cleaned.find("[")
        end_idx = cleaned.rfind("]")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise ValueError(f"No JSON array found in LLM output: {cleaned[:200]}")

        json_str = cleaned[start_idx:end_idx + 1]
        raw_list: list[dict[str, Any]] = json.loads(json_str)

        if not isinstance(raw_list, list) or not raw_list:
            raise ValueError(f"Expected non-empty JSON array, got: {type(raw_list)}")

        steps: list[TaskStep] = []
        for i, item in enumerate(raw_list, start=1):
            if not isinstance(item, dict):
                _log.warning("Skipping non-dict step at index %d: %s", i, item)
                continue

            instruction = item.get("instruction", "").strip()
            if not instruction:
                _log.warning("Skipping step with empty instruction at index %d", i)
                continue

            steps.append(TaskStep(
                step_id=str(uuid.uuid4()),
                goal_id=goal_id,
                sequence=i,
                instruction=instruction,
                requires_approval=bool(item.get("requires_approval", False)),
            ))

        if not steps:
            raise ValueError("No valid steps parsed from LLM output")

        # Re-sequence after filtering
        for i, step in enumerate(steps, start=1):
            step.sequence = i

        return steps

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FALLBACK & PADDING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _fallback_plan(self, goal: str, goal_id: str) -> list[TaskStep]:
        """
        Create a minimal 3-step plan when LLM planning fails.

        Steps: understand → execute → summarize.
        """
        now = datetime.now(timezone.utc).isoformat()
        return [
            TaskStep(
                step_id=str(uuid.uuid4()),
                goal_id=goal_id,
                sequence=1,
                instruction=f"Understand and clarify the goal: {goal}",
                created_at=now,
            ),
            TaskStep(
                step_id=str(uuid.uuid4()),
                goal_id=goal_id,
                sequence=2,
                instruction=f"Execute the core action: {goal}",
                created_at=now,
            ),
            TaskStep(
                step_id=str(uuid.uuid4()),
                goal_id=goal_id,
                sequence=3,
                instruction="Summarize the results and key findings",
                created_at=now,
            ),
        ]

    def _ensure_minimum_steps(
        self, steps: list[TaskStep], goal: str, goal_id: str
    ) -> list[TaskStep]:
        """Pad a plan to at least 3 steps with a synthesis step."""
        while len(steps) < 3:
            steps.append(TaskStep(
                step_id=str(uuid.uuid4()),
                goal_id=goal_id,
                sequence=len(steps) + 1,
                instruction=(
                    "Synthesize all previous findings into a clear, "
                    "actionable summary for the user"
                ),
            ))
        return steps

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  LEGACY COMPAT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_simple_plan(self, user_request: str) -> Plan:
        """
        Phase 1 compatibility — returns a legacy Plan dataclass.

        Use create_plan() for the full autonomous pipeline.
        """
        return Plan(
            steps=[user_request],
            is_multi_step=False,
            original_request=user_request,
        )
