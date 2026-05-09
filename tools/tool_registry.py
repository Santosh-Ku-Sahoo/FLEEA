"""
FLEEA Tool Registry — Centralized, declarative tool registration and routing.

Architecture:
    Every tool in FLEEA lives here.  The Brain queries the registry for
    Claude-compatible schemas and routes tool_use calls to the correct
    handler — all via dictionary lookup, never via if/else branching.

    Each registered tool carries a ToolMetadata descriptor that stores:
        - name:              unique identifier (matches Claude tool_use name)
        - function:          the BaseTool instance that executes the work
        - description:       human-readable purpose string
        - requires_approval: whether the safety layer must gate execution
        - category:          ToolCategory (SAFE / SENSITIVE / DANGEROUS)

Integration points:
    - safety.py   →  registry.requires_approval(name) and
                      registry.get_category(name) feed the SafetyManager
                      so it doesn't need its own hardcoded tool lists.
    - brain.py    →  registry.get_all_schemas() for Claude API,
                      registry.execute(name, **kwargs) for tool routing.

Design:
    - Pure dictionary-based lookup.  O(1) register, retrieve, execute.
    - ToolMetadata is a frozen dataclass — immutable after registration.
    - ToolCategory is an enum with an ordering property for future
      escalation logic (SAFE < SENSITIVE < DANGEROUS).
    - build_default_registry() is the single factory used by app.py to
      wire up all tools at startup.  Adding a new tool = one line.
    - No singleton, no global state — instance is passed via constructor.

Usage:
    from config.settings import Settings
    from tools.tool_registry import build_default_registry

    settings = Settings()
    registry = build_default_registry(settings)
    schemas  = registry.get_all_schemas()           # → Claude API tools=
    result   = await registry.execute("web_search", query="AI news")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tools.base_tool import BaseTool, ToolResult


_log = logging.getLogger("fleea.tools.registry")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TOOL CATEGORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToolCategory(Enum):
    """
    Three-tier risk classification for registered tools.

    Semantics mirror safety.py's SafetyVerdict so the safety layer can
    use the category as a *baseline* classification before applying its
    own pattern-based escalation rules.

    Ordering:  SAFE (0) < SENSITIVE (1) < DANGEROUS (2)
    """

    SAFE = "safe"
    SENSITIVE = "sensitive"
    DANGEROUS = "dangerous"

    @property
    def severity(self) -> int:
        """Numeric severity for comparison.  Higher = more dangerous."""
        return {"safe": 0, "sensitive": 1, "dangerous": 2}[self.value]

    def __lt__(self, other: ToolCategory) -> bool:
        if not isinstance(other, ToolCategory):
            return NotImplemented
        return self.severity < other.severity

    def __le__(self, other: ToolCategory) -> bool:
        if not isinstance(other, ToolCategory):
            return NotImplemented
        return self.severity <= other.severity


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TOOL METADATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class ToolMetadata:
    """
    Immutable descriptor for a registered tool.

    Attributes:
        name:              Unique identifier matching the Claude tool_use name.
        function:          The BaseTool instance that does the actual work.
        description:       Human-readable one-liner for diagnostics/display.
        requires_approval: If True, the safety layer must prompt before executing.
        category:          ToolCategory tier — drives default safety classification.
    """

    name: str
    function: BaseTool
    description: str
    requires_approval: bool
    category: ToolCategory

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Tool name cannot be empty.")
        if self.function is None:
            raise ValueError(f"Tool '{self.name}' must have a non-None function.")
        if not isinstance(self.category, ToolCategory):
            raise TypeError(
                f"Tool '{self.name}' category must be a ToolCategory enum, "
                f"got {type(self.category).__name__}."
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TOOL REGISTRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToolRegistry:
    """
    Centralized registry for all FLEEA tools.

    Provides:
        1. Tool registration    → register()
        2. Tool retrieval       → get(), get_metadata()
        3. Tool execution       → execute()
        4. Available tool list  → list_tools(), tool_names
        5. Approval checking    → requires_approval(), get_category()

    All lookups are O(1) dictionary operations.
    No hardcoded routing — tools self-register via metadata.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolMetadata] = {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  1. REGISTRATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def register(self, tool: BaseTool, category: ToolCategory) -> None:
        """
        Register a tool with its risk category.

        The tool's ``name``, ``description``, and ``requires_approval``
        properties are read from the BaseTool instance.  The caller
        supplies the ``category`` to keep classification explicit and
        auditable — never inferred.

        Raises:
            ValueError:  If a tool with the same name is already registered.
            TypeError:   If ``tool`` is not a BaseTool or ``category`` is invalid.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"Expected a BaseTool instance, got {type(tool).__name__}."
            )

        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered.  "
                "Each tool must have a unique name."
            )

        metadata = ToolMetadata(
            name=tool.name,
            function=tool,
            description=tool.description,
            requires_approval=tool.requires_approval,
            category=category,
        )

        self._tools[tool.name] = metadata
        _log.info(
            "Registered tool: %-20s  category=%-10s  approval=%s",
            tool.name, category.value, tool.requires_approval,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  2. RETRIEVAL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get(self, name: str) -> BaseTool | None:
        """Retrieve a tool instance by name.  Returns None if not found."""
        meta = self._tools.get(name)
        return meta.function if meta else None

    def get_metadata(self, name: str) -> ToolMetadata | None:
        """Retrieve full metadata for a tool.  Returns None if not found."""
        return self._tools.get(name)

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """
        Return all tool schemas in Claude-compatible format.
        Passed directly to the ``tools`` parameter of the Messages API.
        """
        return [meta.function.get_schema() for meta in self._tools.values()]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  3. EXECUTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """
        Execute a registered tool by name.

        Returns a ToolResult with ``success=False`` if the tool is not
        found — never raises to the caller.
        """
        meta = self._tools.get(name)
        if meta is None:
            available = ", ".join(sorted(self._tools.keys())) or "(none)"
            _log.warning("Tool '%s' not found.  Available: %s", name, available)
            return ToolResult(
                success=False,
                error=(
                    f"Tool '{name}' not found in registry.  "
                    f"Available tools: {available}"
                ),
            )

        _log.debug("Executing tool '%s' with args: %s", name, list(kwargs.keys()))
        return await meta.function.execute(**kwargs)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  4. LISTING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def list_tools(self) -> list[dict[str, str]]:
        """
        Return a summary of all registered tools for display or diagnostics.

        Each entry contains: name, description, requires_approval, category.
        """
        return [
            {
                "name": meta.name,
                "description": meta.description,
                "requires_approval": str(meta.requires_approval),
                "category": meta.category.value,
            }
            for meta in self._tools.values()
        ]

    @property
    def tool_names(self) -> list[str]:
        """Sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def tools_by_category(self, category: ToolCategory) -> list[str]:
        """Return tool names that belong to the given category."""
        return [
            meta.name
            for meta in self._tools.values()
            if meta.category == category
        ]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  5. APPROVAL & SAFETY QUERIES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def requires_approval(self, name: str) -> bool:
        """
        Check if a tool requires user approval before execution.

        Returns True if:
            - The tool is registered and its ``requires_approval`` flag is set, OR
            - The tool is not registered (fail-safe: unknown tools need approval).

        This method is designed as the integration hook for safety.py,
        so the safety layer can query the registry instead of maintaining
        its own parallel approval list.
        """
        meta = self._tools.get(name)
        if meta is None:
            # Unknown tool → fail-safe: require approval
            return True
        return meta.requires_approval

    def get_category(self, name: str) -> ToolCategory | None:
        """
        Return the ToolCategory for a registered tool.

        Returns None if the tool is not registered.  The safety layer
        can use this as a baseline before applying its own escalation
        rules (e.g., pattern-based input scanning may upgrade SAFE → SENSITIVE).
        """
        meta = self._tools.get(name)
        return meta.category if meta else None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DUNDER SUPPORT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        tool_list = ", ".join(sorted(self._tools.keys()))
        return f"ToolRegistry(count={len(self)}, tools=[{tool_list}])"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DEFAULT REGISTRY FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_default_registry(settings: Any) -> ToolRegistry:
    """
    Construct the production ToolRegistry with all active tools.

    This is the single wiring point called by app.py at startup.
    To add a new tool to FLEEA:
        1. Create the tool class (extends BaseTool)
        2. Add one ``registry.register(...)`` line below

    Args:
        settings:  The application Settings instance, forwarded to
                   tool constructors that need configuration.

    Returns:
        A fully populated ToolRegistry ready for the Brain.
    """
    from tools.web_search import WebSearchTool

    registry = ToolRegistry()

    # ── Phase 1 tools ─────────────────────────────────────────────
    registry.register(WebSearchTool(settings), category=ToolCategory.SAFE)

    # ── Future tools (uncomment as implemented) ───────────────────
    # registry.register(FileWriteTool(settings),      category=ToolCategory.SENSITIVE)
    # registry.register(CalendarTool(settings),        category=ToolCategory.SENSITIVE)
    # registry.register(TaskManagerTool(settings),     category=ToolCategory.SENSITIVE)
    # registry.register(ComputerControlTool(settings), category=ToolCategory.DANGEROUS)

    _log.info(
        "Tool registry built: %d tool(s) registered  %s",
        len(registry),
        registry.tool_names,
    )

    return registry
