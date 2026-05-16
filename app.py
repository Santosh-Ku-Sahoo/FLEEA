"""
FLEEA Application — Terminal REPL with CLI commands.

This is the entry point. It wires all dependencies together using
explicit injection, starts a session, and runs the interactive loop.

Design:
    - Rich terminal UX with branding, spinners, markdown rendering
    - CLI command system: /help, /clear, /history, /status, /logs, /exit
    - Graceful error handling and shutdown
    - All dependencies wired here and passed down — no globals

Integration points:
    - FLEEABrain      → think(), clear_history(), get_status()
    - SessionManager   → create_session(), end_session(), get_current_session()
    - ExecutionLogger  → get_session_stats(), get_recent_logs(),
                         get_conversation_history(), flush_session_stats()
    - ToolRegistry     → build_default_registry() factory, list_tools()
    - SafetyManager    → constructed with Settings, passed to Brain
    - DatabaseManager  → constructed with db_path, passed to Logger/SessionMgr
    - Settings         → loaded from .env, passed to every component

Usage:
    python app.py
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from agent.brain import FLEEABrain
from agent.logger import ExecutionLogger
from agent.safety import SafetyManager
from agent.session import SessionManager
from config.settings import Settings
from database.models import DatabaseManager
from tools.tool_registry import ToolRegistry, ToolCategory, build_default_registry


# ── Module logger ──────────────────────────────────────────────────
_log = logging.getLogger("fleea.app")

# ── Constants ──────────────────────────────────────────────────────

_VERSION = "0.1.0"

_THEME = Theme(
    {
        "brand": "bold cyan",
        "info": "dim cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "prompt": "bold white",
        "muted": "dim white",
    }
)


# ── Logging Setup ──────────────────────────────────────────────────

def _configure_logging(level: str) -> None:
    """
    Configure Python's logging for FLEEA modules.

    All FLEEA loggers write to logs/fleea.log with rotation so the
    terminal stays clean and the log file doesn't grow unbounded.

    Rotation policy: 5 MB per file, 3 backup files retained.
    """
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "fleea.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
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
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  APPLICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FLEEAApp:
    """
    Main application class. Wires dependencies and runs the REPL.

    Lifecycle:
        1. run()          → enters async event loop
        2. _main()        → loads settings, wires dependencies, starts session
        3. _repl_loop()   → processes input until /exit or Ctrl-C
        4. _shutdown()    → flushes stats, ends session, prints summary
    """

    def __init__(self) -> None:
        self._console = Console(theme=_THEME)

        # Dependencies — set during _main()
        self._settings: Settings | None = None
        self._db: DatabaseManager | None = None
        self._brain: FLEEABrain | None = None
        self._session_mgr: SessionManager | None = None
        self._logger: ExecutionLogger | None = None
        self._registry: ToolRegistry | None = None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  ENTRY POINT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def run(self) -> None:
        """Entry point — initialize and start the async event loop."""
        try:
            asyncio.run(self._main())
        except KeyboardInterrupt:
            self._shutdown()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  INITIALIZATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _main(self) -> None:
        """Async main — initialize dependencies and run REPL."""

        # ── 1. Load Settings ──────────────────────────────────────
        try:
            self._settings = Settings()  # type: ignore[call-arg]
        except Exception as exc:
            self._console.print(
                f"[error]Failed to load settings: {exc}[/error]"
            )
            self._console.print(
                "[warning]Make sure your .env file is configured "
                "with valid API keys.[/warning]"
            )
            return

        # ── 2. Configure Logging ──────────────────────────────────
        _configure_logging(self._settings.LOG_LEVEL)

        # ── 3. Wire Dependencies ──────────────────────────────────
        self._db = DatabaseManager(self._settings.db_path_resolved)
        self._logger = ExecutionLogger(self._db)
        self._session_mgr = SessionManager(self._db)
        safety = SafetyManager(self._settings)

        # Build tool registry via the centralized factory
        self._registry = build_default_registry(self._settings)

        # Phase 2 memory wiring (graceful degradation)
        short_term = None
        retriever = None
        profile_mgr = None
        vector_store = None

        try:
            from memory.short_term import ShortTermMemory
            from memory.retriever import MemoryRetriever
            from memory.profile_manager import ProfileManager
            from memory.vector_store import VectorStore

            persist_dir = str(
                self._settings.project_root / "memory" / "chroma_db"
            )
            vector_store = VectorStore(persist_dir=persist_dir)
            retriever = MemoryRetriever(vector_store)
            profile_mgr = ProfileManager(self._db)
            short_term = ShortTermMemory(max_turns=10)
            _log.info("Memory subsystems wired successfully")
        except Exception as exc:
            _log.warning("Memory subsystem init failed (non-fatal): %s", exc)

        # Create the brain with all injected dependencies
        self._brain = FLEEABrain(
            settings=self._settings,
            tool_registry=self._registry,
            logger=self._logger,
            safety=safety,
            session_manager=self._session_mgr,
            short_term_memory=short_term,
            memory_retriever=retriever,
            profile_manager=profile_mgr,
            vector_store=vector_store,
        )

        # ── 4. Start Session ──────────────────────────────────────
        session = self._session_mgr.create_session()
        self._display_banner(session.short_id)

        # ── 5. REPL Loop ──────────────────────────────────────────
        await self._repl_loop()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  REPL LOOP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _repl_loop(self) -> None:
        """
        Main Read-Eval-Print loop.

        Flow:
            1. Read user input
            2. If empty → skip
            3. If starts with '/' → dispatch to command handler
            4. Otherwise → route to Brain.think()
            5. Render response in a branded panel
        """
        while True:
            try:
                self._console.print()
                user_input = self._console.input("[bold cyan]You >> [/]").strip()

                if not user_input:
                    continue

                # ── CLI command dispatch ──────────────────────────
                if user_input.startswith("/"):
                    should_exit = self._handle_command(user_input)
                    if should_exit:
                        break
                    continue

                # ── Route to Brain ────────────────────────────────
                with self._console.status(
                    "[info]Thinking...[/info]", spinner="dots"
                ):
                    response = await self._brain.think(user_input)  # type: ignore[union-attr]

                # ── Display response ──────────────────────────────
                self._console.print()
                self._console.print(
                    Panel(
                        Markdown(response),
                        title="[brand]FLEEA[/brand]",
                        border_style="cyan",
                        padding=(1, 2),
                    )
                )

            except KeyboardInterrupt:
                self._console.print("\n[muted]Use /exit to quit.[/muted]")
                continue
            except EOFError:
                break
            except Exception as exc:
                self._console.print(f"[error]Error: {exc}[/error]")
                if self._settings and self._settings.LOG_LEVEL == "DEBUG":
                    self._console.print_exception()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  CLI COMMAND DISPATCH
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _handle_command(self, command: str) -> bool:
        """
        Handle CLI commands. Returns True if the app should exit.

        Routing is dictionary-based — O(1) lookup, no if/else chain.
        """
        cmd = command.lower().strip()

        handlers: dict[str, Any] = {
            "/help":    self._cmd_help,
            "/clear":   self._cmd_clear,
            "/history": self._cmd_history,
            "/status":  self._cmd_status,
            "/logs":    self._cmd_logs,
            "/exit":    lambda: True,
            "/quit":    lambda: True,
        }

        handler = handlers.get(cmd)
        if handler is None:
            self._console.print(
                f"[warning]Unknown command: {cmd}. "
                "Type /help for available commands.[/warning]"
            )
            return False

        result = handler()
        if result is True:
            self._shutdown()
            return True
        return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  COMMANDS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _cmd_help(self) -> None:
        """Display available commands."""
        table = Table(
            title="FLEEA Commands",
            border_style="cyan",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Command", style="bold white", width=12)
        table.add_column("Description", style="white")

        table.add_row("/help",    "Show this help message")
        table.add_row("/clear",   "Clear conversation history")
        table.add_row("/history", "Show conversation history")
        table.add_row("/status",  "Show session and system status")
        table.add_row("/logs",    "Show recent tool execution logs")
        table.add_row("/exit",    "Exit FLEEA gracefully")

        self._console.print()
        self._console.print(table)

    def _cmd_clear(self) -> None:
        """Clear conversation history for the current session."""
        if self._brain:
            self._brain.clear_history()
        self._console.print("[success]Conversation history cleared.[/success]")

    def _cmd_history(self) -> None:
        """
        Display conversation history for the current session.

        Reads from the database via ExecutionLogger, which returns rows
        with 'user_message', 'assistant_response', and 'timestamp' fields.
        """
        session = (
            self._session_mgr.get_current_session()
            if self._session_mgr
            else None
        )
        if not session or not self._logger:
            self._console.print("[muted]No active session.[/muted]")
            return

        history = self._logger.get_conversation_history(session.session_id)
        if not history:
            self._console.print("[muted]No conversation history yet.[/muted]")
            return

        self._console.print()
        for entry in history:
            timestamp = entry.get("timestamp", "")[:19]

            # User turn
            user_msg = entry.get("user_message", "")
            self._console.print(
                f"  [bold white]You[/bold white]  [muted]{timestamp}[/muted]"
            )
            self._console.print(f"    {user_msg[:200]}")
            self._console.print()

            # Assistant turn
            assistant_msg = entry.get("assistant_response", "")
            tool = entry.get("tool_invoked")
            tool_tag = f"  [muted]via {tool}[/muted]" if tool else ""
            self._console.print(
                f"  [brand]FLEEA[/brand]  [muted]{timestamp}[/muted]{tool_tag}"
            )
            self._console.print(f"    {assistant_msg[:200]}")
            self._console.print()

    def _cmd_status(self) -> None:
        """
        Display comprehensive session and system status.

        Pulls live in-memory stats from ExecutionLogger (no DB query)
        and session metadata from SessionManager.
        """
        session = (
            self._session_mgr.get_current_session()
            if self._session_mgr
            else None
        )
        if not session:
            self._console.print("[muted]No active session.[/muted]")
            return

        # ── Session table ─────────────────────────────────────────
        table = Table(
            title="Session Status",
            border_style="cyan",
            show_header=False,
        )
        table.add_column("Metric", style="bold white", width=22)
        table.add_column("Value", style="cyan")

        table.add_row("Session ID",    session.short_id)
        table.add_row("Duration",      session.duration_display)
        table.add_row("Interactions",  str(session.total_interactions))
        table.add_row("Tool Calls",    str(session.total_tool_calls))

        # Token/cost data lives in ExecutionLogger (single source of truth)
        stats = (
            self._logger.get_session_stats(session.session_id)
            if self._logger
            else None
        )
        if stats:
            table.add_row("Tokens Used",   f"{stats.total_tokens:,}")
            table.add_row("Est. Cost",     stats.cost_display)
            table.add_row("API Calls",     str(stats.api_calls))
            table.add_row("Tool Errors",   str(stats.tool_errors))
            table.add_row("Success Rate",  f"{stats.success_rate:.0f}%")

        # Model and tool info from brain
        if self._brain:
            brain_status = self._brain.get_status()
            table.add_row("Model",          brain_status["model"])
            table.add_row("Max Tokens",     str(brain_status["max_tokens"]))
            table.add_row("Registered Tools",
                          ", ".join(brain_status["registered_tools"]) or "-")

        self._console.print()
        self._console.print(table)

    def _cmd_logs(self) -> None:
        """
        Display the 10 most recent tool execution logs.

        Reads from the execution_logs table via ExecutionLogger.
        Each row contains: timestamp, selected_tool, success,
        total_tokens, estimated_cost, execution_duration_ms.
        """
        if not self._logger:
            self._console.print("[muted]Logger not initialized.[/muted]")
            return

        logs = self._logger.get_recent_logs(10)
        if not logs:
            self._console.print("[muted]No execution logs yet.[/muted]")
            return

        table = Table(
            title="Recent Execution Logs",
            border_style="cyan",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Time",     style="muted",  width=12)
        table.add_column("Tool",     style="cyan",   width=14)
        table.add_column("Status",                    width=8)
        table.add_column("Tokens",   style="white",  width=8)
        table.add_column("Cost",     style="white",  width=10)
        table.add_column("Duration", style="muted",  width=10)

        for log in logs:
            timestamp = str(log.get("timestamp", ""))[:19]
            tool = log.get("selected_tool", "unknown")
            success = log.get("success", False)
            status = "[green]OK[/green]" if success else "[red]FAIL[/red]"
            tokens = str(log.get("total_tokens", 0))
            cost = f"${log.get('estimated_cost', 0):.6f}"
            duration = f"{log.get('execution_duration_ms', 0):.0f}ms"

            table.add_row(timestamp, tool, status, tokens, cost, duration)

        self._console.print()
        self._console.print(table)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DISPLAY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _display_banner(self, session_id: str) -> None:
        """Display the startup banner with branding and session info."""
        banner = Text()
        banner.append("  F.L.E.E.A.\n", style="bold cyan")
        banner.append(
            "  Future Learning Executive & Everyday Assistant\n\n",
            style="white",
        )
        banner.append("  Phase 5 ", style="bold green")
        banner.append("- Production Ready (Memory + Voice + Auth + Admin)\n", style="white")
        banner.append(f"  Version: {_VERSION}\n", style="muted")
        banner.append(f"  Session: {session_id}\n", style="muted")

        self._console.print()
        self._console.print(
            Panel(
                banner,
                border_style="cyan",
                padding=(1, 2),
                title="FLEEA",
                subtitle="Type /help for commands",
            )
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SHUTDOWN
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _shutdown(self) -> None:
        """
        Graceful shutdown — flush stats, end session, print summary.

        Called on /exit, /quit, or KeyboardInterrupt.  Safe to call
        multiple times (end_session returns None on second call).
        """
        if self._session_mgr and self._logger:
            session = self._session_mgr.get_current_session()

            # Capture logger stats BEFORE ending (session_id needed)
            stats = (
                self._logger.get_session_stats(session.session_id)
                if session
                else None
            )

            if session:
                # Flush logger's in-memory stats to DB before closing
                self._logger.flush_session_stats(session.session_id)

            # End session — returns the completed Session for display
            completed = self._session_mgr.end_session()
            if completed:
                tokens = stats.total_tokens if stats else 0
                cost = stats.cost_display if stats else "$0.00"

                self._console.print()
                summary = Text()
                summary.append("Session ended\n", style="bold white")
                summary.append(
                    f"  Duration:      {completed.duration_display}\n",
                    style="muted",
                )
                summary.append(
                    f"  Interactions:  {completed.total_interactions}\n",
                    style="muted",
                )
                summary.append(
                    f"  Tool Calls:    {completed.total_tool_calls}\n",
                    style="muted",
                )
                summary.append(
                    f"  Tokens:        {tokens:,}\n",
                    style="muted",
                )
                summary.append(
                    f"  Cost:          {cost}\n",
                    style="muted",
                )

                self._console.print(
                    Panel(summary, border_style="cyan", title="Goodbye")
                )

        self._console.print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    app = FLEEAApp()
    app.run()


if __name__ == "__main__":
    main()
