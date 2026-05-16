"""
FLEEA Autonomous Loop — Supervised autonomous agent orchestrator.

Architecture:
    The AutonomousLoop is the top-level orchestrator that ties together
    the Planner, Executor, TaskManager, and Reflector into a coherent
    supervised autonomy pipeline.

    Execution flow:
        1. User provides a high-level goal
        2. Planner decomposes goal into 3–7 TaskSteps via Gemini
        3. Plan is displayed to user for approval before execution
        4. Each step is executed sequentially through the Executor → Brain
        5. Steps flagged requires_approval pause for user confirmation
        6. After each step, the Reflector evaluates success and extracts learnings
        7. Learnings are stored in VectorStore for future reference
        8. Final GoalResult summarizes the entire execution

    Safety guarantees:
        - User approves the plan before any execution begins
        - SENSITIVE actions pause for approval (via Brain's SafetyManager)
        - DANGEROUS actions are unconditionally blocked (via SafetyManager)
        - Ctrl+C triggers graceful shutdown — remaining steps stay PENDING
        - Failed steps ask user whether to continue or abort
        - Max 2 retries per step to prevent runaway loops

Design:
    - All dependencies injected via constructor (no globals).
    - Rich terminal output for clear progress visibility.
    - Reflector uses a lightweight Gemini call (~$0.001 per reflection).
    - GoalResult dataclass provides structured execution summary.

Usage:
    loop = AutonomousLoop(
        settings=settings, brain=brain, task_manager=tm,
        planner=planner, executor=executor, logger=logger,
    )
    result = await loop.run("Research AI trends and write a summary")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from agent.executor import ExecutionResult, TaskExecutor
from agent.planner import GoalSummary, TaskPlanner, TaskStatus, TaskStep
from agent.task_manager import TaskManager
from config.settings import Settings


_log = logging.getLogger("fleea.autonomous")
_console = Console()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RESULT TYPES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Reflection:
    """Outcome of reflecting on a completed step."""

    success_assessment: bool        # did the step achieve its objective?
    learning: str                   # key insight or takeaway
    improvement: str                # what to do differently next time

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success_assessment,
            "learning": self.learning,
            "improvement": self.improvement,
        }


@dataclass
class StepResult:
    """Combined execution + reflection result for one step."""

    step: TaskStep
    execution: ExecutionResult
    reflection: Reflection | None = None


@dataclass
class GoalResult:
    """Final summary of an autonomous goal execution."""

    goal_id: str
    goal: str
    step_results: list[StepResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    interrupted: bool = False       # True if user interrupted with Ctrl+C

    @property
    def completed_count(self) -> int:
        return sum(1 for sr in self.step_results if sr.execution.success)

    @property
    def failed_count(self) -> int:
        return sum(1 for sr in self.step_results if not sr.execution.success)

    @property
    def total_steps(self) -> int:
        return len(self.step_results)

    @property
    def success_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return (self.completed_count / self.total_steps) * 100

    @property
    def learnings(self) -> list[str]:
        """All learnings extracted from reflections."""
        return [
            sr.reflection.learning
            for sr in self.step_results
            if sr.reflection and sr.reflection.learning
        ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  REFLECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_REFLECTION_PROMPT: str = """\
You are an AI agent evaluating the result of a task step.

## Goal
{goal}

## Step {sequence}: {instruction}

## Result
{result}

## Your Assessment
Evaluate this result and respond with ONLY a JSON object (no markdown, no code fences):
{{
  "success": true/false,
  "learning": "One sentence about what was learned or accomplished",
  "improvement": "One sentence about what could be done better next time"
}}
"""


class _Reflector:
    """
    Lightweight reflection engine using Claude.

    Evaluates task step outcomes and extracts learnings.
    Each reflection costs ~$0.001 — negligible overhead.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self._model = settings.DEFAULT_MODEL
        self._max_tokens = 256  # reflections are very short

    def reflect(
        self,
        step: TaskStep,
        result: str,
        goal: str,
    ) -> Reflection:
        """
        Evaluate a step's result and extract learnings.

        Returns a Reflection with success assessment, learning, and
        improvement suggestion. Falls back to a default Reflection
        if Claude returns unparseable output.
        """
        prompt = _REFLECTION_PROMPT.format(
            goal=goal,
            sequence=step.sequence,
            instruction=step.instruction,
            result=result[:2000],  # cap to control token usage
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=self._max_tokens,
                ),
            )

            raw = response.text or ""
            return self._parse_reflection(raw.strip())

        except Exception as exc:
            _log.warning("Reflection failed: %s", exc)
            return Reflection(
                success_assessment=True,
                learning="Step completed (reflection unavailable)",
                improvement="No improvement data available",
            )

    @staticmethod
    def _parse_reflection(raw: str) -> Reflection:
        """Parse Claude's JSON reflection output."""
        # Strip code fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.index("\n")
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Find JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            return Reflection(
                success_assessment=True,
                learning=cleaned[:200] if cleaned else "Step completed",
                improvement="",
            )

        try:
            data = json.loads(cleaned[start:end + 1])
            return Reflection(
                success_assessment=bool(data.get("success", True)),
                learning=str(data.get("learning", ""))[:500],
                improvement=str(data.get("improvement", ""))[:500],
            )
        except (json.JSONDecodeError, TypeError):
            return Reflection(
                success_assessment=True,
                learning="Step completed (parse error in reflection)",
                improvement="",
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AUTONOMOUS LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AutonomousLoop:
    """
    Supervised autonomous agent orchestrator.

    Ties together Planner → TaskManager → Executor → Reflector
    into a coherent execution pipeline with human oversight.

    Key safety properties:
        1. Plan approval before execution begins
        2. Step-level approval for sensitive operations
        3. Graceful Ctrl+C handling (remaining steps stay PENDING)
        4. Fail-stop on critical errors with user choice to continue
        5. All tool invocations go through Brain → SafetyManager
    """

    def __init__(
        self,
        *,
        settings: Settings,
        brain: Any,
        task_manager: TaskManager,
        planner: TaskPlanner,
        executor: TaskExecutor,
        vector_store: Any | None = None,
    ) -> None:
        self._settings = settings
        self._brain = brain
        self._task_manager = task_manager
        self._planner = planner
        self._executor = executor
        self._vector_store = vector_store
        self._reflector = _Reflector(settings)

    async def run(self, goal: str) -> GoalResult:
        """
        Execute a high-level goal autonomously with supervision.

        Flow:
            1. Plan: decompose goal into steps
            2. Display: show plan to user for approval
            3. Persist: save plan to database
            4. Execute: run steps sequentially
            5. Reflect: evaluate each step's outcome
            6. Store: persist learnings in vector memory
            7. Summarize: return GoalResult

        Args:
            goal: Natural-language description of the user's goal.

        Returns:
            GoalResult with all step outcomes, learnings, and timing.
        """
        start_time = time.perf_counter()
        step_results: list[StepResult] = []
        interrupted = False

        # ── 1. PLAN ─────────────────────────────────────────────────
        _console.print()
        _console.print(
            Panel(
                f"[bold cyan]Goal:[/] {goal}",
                title="[bold white]  AUTONOMOUS MODE  [/]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        _console.print("[dim]Planning...[/]")

        try:
            steps = self._planner.create_plan(goal)
        except Exception as exc:
            _console.print(f"[bold red]Planning failed:[/] {exc}")
            _log.error("Plan creation failed: %s", exc)
            return GoalResult(
                goal_id="",
                goal=goal,
                total_duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        goal_id = steps[0].goal_id

        # ── 2. DISPLAY PLAN ────────────────────────────────────────
        self._display_plan(steps)

        # ── 3. APPROVAL ───────────────────────────────────────────
        try:
            approved = Confirm.ask(
                "\n[bold yellow]Execute this plan?[/]",
                default=True,
                console=_console,
            )
        except (KeyboardInterrupt, EOFError):
            _console.print("[dim]Cancelled.[/]")
            return GoalResult(
                goal_id=goal_id,
                goal=goal,
                interrupted=True,
                total_duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        if not approved:
            _console.print("[dim]Plan rejected. No actions taken.[/]")
            return GoalResult(
                goal_id=goal_id,
                goal=goal,
                total_duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        # ── 4. PERSIST PLAN ───────────────────────────────────────
        try:
            self._task_manager.save_plan(steps)
        except Exception as exc:
            _console.print(f"[bold red]Failed to persist plan:[/] {exc}")
            _log.error("Plan persistence failed: %s", exc)

        # ── 5. EXECUTE STEPS ───────────────────────────────────────
        previous_results: list[str] = []

        for step in steps:
            try:
                step_result = await self._execute_step(
                    step, goal, previous_results,
                )
            except KeyboardInterrupt:
                _console.print(
                    "\n[bold yellow]Interrupted![/] "
                    "Remaining steps preserved as PENDING."
                )
                interrupted = True
                break

            step_results.append(step_result)

            if step_result.execution.success:
                previous_results.append(step_result.execution.response)

                # ── 6. REFLECT ─────────────────────────────────────
                try:
                    reflection = self._reflector.reflect(
                        step, step_result.execution.response, goal,
                    )
                    step_result.reflection = reflection

                    # Store learning in vector memory
                    self._store_learning(reflection, step, goal)

                except Exception as exc:
                    _log.warning("Reflection failed for step %d: %s", step.sequence, exc)

            else:
                # Step failed — ask user whether to continue or abort
                if not self._handle_failure(step, step_result.execution):
                    # User chose to abort
                    interrupted = True
                    break

        # ── 7. SUMMARIZE ───────────────────────────────────────────
        total_duration = (time.perf_counter() - start_time) * 1000
        result = GoalResult(
            goal_id=goal_id,
            goal=goal,
            step_results=step_results,
            total_duration_ms=total_duration,
            interrupted=interrupted,
        )

        self._display_summary(result)
        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STEP EXECUTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _execute_step(
        self,
        step: TaskStep,
        goal: str,
        previous_results: list[str],
    ) -> StepResult:
        """Execute a single step with approval check and state management."""

        # ── Step-level approval for flagged steps ──────────────────
        if step.requires_approval:
            self._task_manager.mark_awaiting_approval(step.step_id)

            _console.print()
            content = Text()
            content.append(f"Step {step.sequence}: ", style="bold yellow")
            content.append(step.instruction, style="white")
            content.append("\n\nThis step may modify external state.", style="dim")

            _console.print(
                Panel(
                    content,
                    title="[bold yellow]  STEP APPROVAL  [/]",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )

            try:
                approved = Confirm.ask(
                    "[bold yellow]Execute this step?[/]",
                    default=True,
                    console=_console,
                )
            except (KeyboardInterrupt, EOFError):
                approved = False

            if not approved:
                self._task_manager.mark_skipped(step.step_id, "User declined")
                _console.print(f"[dim]Step {step.sequence} skipped.[/]")
                return StepResult(
                    step=step,
                    execution=ExecutionResult(
                        success=False,
                        response="Step skipped by user",
                        error="User declined approval",
                    ),
                )

        # ── Execute ────────────────────────────────────────────────
        self._task_manager.mark_in_progress(step.step_id)
        _console.print(
            f"\n[bold cyan]> Step {step.sequence}/{len(previous_results) + 1}:[/] "
            f"{step.instruction}"
        )
        _console.print("[dim]  Executing...[/]")

        exec_result = await self._executor.execute(
            step,
            goal=goal,
            previous_results=previous_results,
        )

        # ── Update state ───────────────────────────────────────────
        if exec_result.success:
            self._task_manager.mark_completed(step.step_id, exec_result.response)
            _console.print(
                f"[bold green]  [OK] Step {step.sequence} completed[/] "
                f"[dim]({exec_result.duration_ms:.0f}ms)[/]"
            )
        else:
            self._task_manager.mark_failed(step.step_id, exec_result.error or "Unknown error")
            _console.print(
                f"[bold red]  [FAIL] Step {step.sequence} failed:[/] "
                f"{exec_result.error or 'Unknown error'}"
            )

        return StepResult(step=step, execution=exec_result)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FAILURE HANDLING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _handle_failure(
        self,
        step: TaskStep,
        exec_result: ExecutionResult,
    ) -> bool:
        """
        Handle a failed step — ask user whether to continue or abort.

        Returns True to continue, False to abort.
        """
        _console.print()
        content = Text()
        content.append(f"Step {step.sequence} failed\n", style="bold red")
        content.append(f"Error: {exec_result.error or 'Unknown'}\n\n", style="white")
        content.append(
            "You can continue with the remaining steps or abort the plan.",
            style="dim",
        )

        _console.print(
            Panel(
                content,
                title="[bold red]  STEP FAILED  [/]",
                border_style="red",
                padding=(1, 2),
            )
        )

        try:
            return Confirm.ask(
                "[bold yellow]Continue with remaining steps?[/]",
                default=True,
                console=_console,
            )
        except (KeyboardInterrupt, EOFError):
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  LEARNING STORAGE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _store_learning(
        self,
        reflection: Reflection,
        step: TaskStep,
        goal: str,
    ) -> None:
        """Store a reflection's learning in the vector store for future recall."""
        if not self._vector_store or not reflection.learning:
            return

        try:
            self._vector_store.add_memory(
                text=f"[Autonomous learning] Goal: {goal} | "
                     f"Step: {step.instruction} | "
                     f"Learning: {reflection.learning}",
                metadata={
                    "source": "autonomous_reflection",
                    "goal_id": step.goal_id,
                    "step_sequence": step.sequence,
                    "success": reflection.success_assessment,
                },
            )
            _log.debug(
                "Stored learning for step %d: %s",
                step.sequence,
                reflection.learning[:80],
            )
        except Exception as exc:
            _log.warning("Failed to store learning: %s", exc)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DISPLAY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _display_plan(self, steps: list[TaskStep]) -> None:
        """Display the execution plan as a numbered list."""
        _console.print()

        table = Table(
            title="Execution Plan",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            pad_edge=True,
        )
        table.add_column("#", style="bold", width=3, justify="right")
        table.add_column("Step", min_width=40)
        table.add_column("Approval", width=10, justify="center")

        for step in steps:
            approval = "[yellow]Required[/]" if step.requires_approval else "[green]Auto[/]"
            table.add_row(
                str(step.sequence),
                step.instruction,
                approval,
            )

        _console.print(table)

    def _display_summary(self, result: GoalResult) -> None:
        """Display the final execution summary."""
        _console.print()

        # Build summary content
        content = Text()
        content.append("Goal: ", style="bold white")
        content.append(f"{result.goal}\n\n", style="cyan")

        for sr in result.step_results:
            if sr.execution.success:
                icon = "[OK]"
                style = "green"
            else:
                icon = "[FAIL]"
                style = "red"
            content.append(f"  {icon} Step {sr.step.sequence}: ", style=f"bold {style}")
            content.append(f"{sr.step.instruction}\n", style="white")

            if sr.reflection and sr.reflection.learning:
                content.append(f"    -> {sr.reflection.learning}\n", style="dim")

        content.append(f"\nCompleted: ", style="bold white")
        content.append(f"{result.completed_count}/{result.total_steps}", style="bold green")
        content.append(f"  |  Duration: ", style="bold white")
        content.append(f"{result.total_duration_ms / 1000:.1f}s", style="bold cyan")

        if result.interrupted:
            content.append(f"\n[!] Execution was interrupted", style="bold yellow")

        if result.learnings:
            content.append(f"\n{len(result.learnings)} learning(s) stored in memory", style="dim")

        border = "green" if result.failed_count == 0 else "yellow"
        _console.print(
            Panel(
                content,
                title="[bold white]  EXECUTION COMPLETE  [/]",
                border_style=border,
                padding=(1, 2),
            )
        )
