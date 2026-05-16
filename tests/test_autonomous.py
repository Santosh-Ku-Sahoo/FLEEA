r"""
FLEEA Autonomous Agent — Integration Test Script.

Wires all dependencies (mirroring app.py), then runs a simple goal
through the AutonomousLoop to verify the full pipeline:
    Planner → TaskManager → Executor → Brain → Reflector

Usage:
    cd FLEEA
    .\venv\Scripts\python.exe tests/test_autonomous.py
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

from agent.brain import FLEEABrain
from agent.logger import ExecutionLogger
from agent.safety import SafetyManager
from agent.session import SessionManager
from agent.planner import TaskPlanner
from agent.task_manager import TaskManager
from agent.executor import TaskExecutor
from agent.autonomous_loop import AutonomousLoop
from config.settings import Settings
from database.models import DatabaseManager
from tools.tool_registry import build_default_registry


console = Console()


def _configure_logging() -> None:
    """Set up logging to file for the test."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "fleea.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger("fleea")
    root.setLevel(logging.INFO)
    root.addHandler(handler)


async def main() -> None:
    """Wire dependencies and run a test goal."""

    _configure_logging()

    # ── 1. Load settings ──────────────────────────────────────────
    console.print("[bold cyan]Loading settings...[/]")
    try:
        settings = Settings()  # type: ignore[call-arg]
    except Exception as exc:
        console.print(f"[bold red]Settings failed:[/] {exc}")
        console.print("[yellow]Make sure .env is configured.[/]")
        return

    # ── 2. Wire core dependencies (same as app.py) ────────────────
    db = DatabaseManager(settings.db_path_resolved)
    logger = ExecutionLogger(db)
    session_mgr = SessionManager(db)
    safety = SafetyManager(settings)
    registry = build_default_registry(settings)

    # Create session
    session = session_mgr.create_session()
    console.print(f"[dim]Session: {session.short_id}[/]")

    # Create brain
    brain = FLEEABrain(
        settings=settings,
        tool_registry=registry,
        logger=logger,
        safety=safety,
        session_manager=session_mgr,
    )

    # ── 3. Wire autonomous components ─────────────────────────────
    planner = TaskPlanner(settings)
    task_mgr = TaskManager(db, max_retries=2)
    executor = TaskExecutor(brain, max_retries=2)

    autonomous = AutonomousLoop(
        settings=settings,
        brain=brain,
        task_manager=task_mgr,
        planner=planner,
        executor=executor,
        # vector_store not wired in app.py yet — pass None
    )

    # ── 4. Run test goal ──────────────────────────────────────────
    goal = "Search for the top 3 AI trends in 2025 and summarize the findings"
    console.print(f"\n[bold green]Testing goal:[/] {goal}\n")

    try:
        result = await autonomous.run(goal)

        # ── 5. Print results ──────────────────────────────────────
        console.print(f"\n[bold]Result summary:[/]")
        console.print(f"  Goal ID:    {result.goal_id[:8]}")
        console.print(f"  Steps:      {result.total_steps}")
        console.print(f"  Completed:  {result.completed_count}")
        console.print(f"  Failed:     {result.failed_count}")
        console.print(f"  Duration:   {result.total_duration_ms / 1000:.1f}s")
        console.print(f"  Interrupted: {result.interrupted}")

        if result.learnings:
            console.print(f"\n[bold]Learnings ({len(result.learnings)}):[/]")
            for i, learning in enumerate(result.learnings, 1):
                console.print(f"  {i}. {learning}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user.[/]")

    # ── 6. Cleanup ────────────────────────────────────────────────
    logger.flush_session_stats(session.session_id)
    session_mgr.end_session()
    console.print("\n[dim]Session ended. Test complete.[/]")


if __name__ == "__main__":
    asyncio.run(main())
