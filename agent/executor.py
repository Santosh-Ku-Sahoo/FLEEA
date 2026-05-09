"""
FLEEA Task Executor — Executes individual task steps through the Brain.

Architecture:
    The Executor is the bridge between the autonomous pipeline and the
    existing Brain.  It takes a single TaskStep, constructs a task-specific
    prompt, and delegates to Brain.think() for the full agentic loop.

    The Executor NEVER accesses tools directly.  All tool invocations,
    safety gating, retry logic, and memory operations go through the
    Brain's existing infrastructure.  This keeps the autonomous system
    as a pure orchestration layer with zero duplication.

Design:
    - Brain injected via constructor (no globals).
    - Each execution gets a context-rich prompt that includes the goal
      and previous step results for continuity.
    - Retry logic (up to 2 attempts) for transient failures.
    - ExecutionResult dataclass captures success, response, timing, error.
    - All executions are logged via the brain's internal logger.

Usage:
    executor = TaskExecutor(brain)
    result = await executor.execute(step, goal="Research AI trends")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from agent.planner import TaskStep


_log = logging.getLogger("fleea.executor")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EXECUTION RESULT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ExecutionResult:
    """Outcome of executing a single task step."""

    success: bool
    response: str                   # Brain's full text response
    duration_ms: float = 0.0        # wall-clock execution time
    error: str | None = None        # error message if failed
    retry_count: int = 0            # number of retries attempted

    @property
    def response_preview(self) -> str:
        """First 200 characters of the response for display."""
        if len(self.response) <= 200:
            return self.response
        return self.response[:197] + "..."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TASK EXECUTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TaskExecutor:
    """
    Executes autonomous task steps by delegating to the Brain.

    The Executor is deliberately thin — it constructs a contextual
    prompt and calls brain.think().  The Brain handles everything else:
    tool selection, safety gating, retry, memory, and logging.

    This ensures the autonomous system inherits all existing safeguards
    without duplicating any logic.
    """

    def __init__(self, brain: Any, *, max_retries: int = 2) -> None:
        """
        Args:
            brain:       A FLEEABrain instance (typed as Any to avoid
                         circular import — validated at runtime).
            max_retries: Maximum retry attempts per step on transient errors.
        """
        self._brain = brain
        self._max_retries = max_retries

    async def execute(
        self,
        step: TaskStep,
        *,
        goal: str,
        previous_results: list[str] | None = None,
    ) -> ExecutionResult:
        """
        Execute a single task step through the Brain.

        Args:
            step:             The TaskStep to execute.
            goal:             The high-level user goal (for context).
            previous_results: Results from earlier steps (for continuity).

        Returns:
            ExecutionResult with the Brain's response and metadata.
        """
        prompt = self._build_prompt(step, goal, previous_results)
        attempt = 0
        last_error: str | None = None

        while attempt <= self._max_retries:
            attempt += 1

            _log.info(
                "Executing step %d [%s] (attempt %d/%d): %s",
                step.sequence,
                step.step_id[:8],
                attempt,
                self._max_retries + 1,
                step.instruction[:80],
            )

            start = time.perf_counter()

            try:
                response = await self._brain.think(prompt)
                duration_ms = (time.perf_counter() - start) * 1000

                # Check for error indicators in the response
                if self._is_error_response(response):
                    last_error = f"Brain returned error response: {response[:200]}"
                    _log.warning(
                        "Step %d [%s] got error response (attempt %d): %s",
                        step.sequence,
                        step.step_id[:8],
                        attempt,
                        last_error[:100],
                    )
                    if attempt <= self._max_retries:
                        continue
                    return ExecutionResult(
                        success=False,
                        response=response,
                        duration_ms=duration_ms,
                        error=last_error,
                        retry_count=attempt - 1,
                    )

                _log.info(
                    "Step %d [%s] completed: %.1fms  response=%d chars",
                    step.sequence,
                    step.step_id[:8],
                    duration_ms,
                    len(response),
                )

                return ExecutionResult(
                    success=True,
                    response=response,
                    duration_ms=duration_ms,
                    retry_count=attempt - 1,
                )

            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000
                last_error = f"{type(exc).__name__}: {exc}"

                _log.error(
                    "Step %d [%s] failed (attempt %d): %s",
                    step.sequence,
                    step.step_id[:8],
                    attempt,
                    last_error,
                )

                if attempt <= self._max_retries:
                    _log.info("Retrying step %d...", step.sequence)
                    continue

                return ExecutionResult(
                    success=False,
                    response="",
                    duration_ms=duration_ms,
                    error=last_error,
                    retry_count=attempt - 1,
                )

        # Unreachable, but defensive
        return ExecutionResult(
            success=False,
            response="",
            error=last_error or "Max retries exceeded",
            retry_count=self._max_retries,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PROMPT CONSTRUCTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _build_prompt(
        step: TaskStep,
        goal: str,
        previous_results: list[str] | None = None,
    ) -> str:
        """
        Construct a task-specific prompt for the Brain.

        The prompt provides three layers of context:
        1. The overall goal (so Brain understands purpose)
        2. Results from previous steps (for continuity)
        3. The specific instruction for this step
        """
        sections: list[str] = []

        # 1. Goal context
        sections.append(
            f"You are executing step {step.sequence} of an autonomous task plan.\n"
            f"Overall goal: {goal}"
        )

        # 2. Previous results (if any)
        if previous_results:
            context_lines = []
            for i, result in enumerate(previous_results, start=1):
                # Truncate long results to keep prompt size manageable
                preview = result[:500] + "..." if len(result) > 500 else result
                context_lines.append(f"Step {i} result: {preview}")
            sections.append(
                "Previous step results:\n" + "\n".join(context_lines)
            )

        # 3. Current instruction
        sections.append(
            f"Your current task (step {step.sequence}):\n{step.instruction}\n\n"
            "Execute this step thoroughly. Use available tools if needed. "
            "Provide a clear, complete response."
        )

        return "\n\n".join(sections)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  ERROR DETECTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _is_error_response(response: str) -> bool:
        """
        Heuristic check for error-like Brain responses.

        The Brain always returns a string (never raises), so we check
        for known error patterns in the response text.
        """
        if not response or len(response.strip()) < 10:
            return True

        error_indicators = [
            "I encountered an unexpected error",
            "I've reached the maximum number of tool calls",
            "I apologize, but I'm unable to",
        ]
        response_lower = response.lower()
        return any(indicator.lower() in response_lower for indicator in error_indicators)
