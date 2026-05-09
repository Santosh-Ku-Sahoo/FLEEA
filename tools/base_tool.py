"""
FLEEA Base Tool — Abstract interface for all tools.

All tools must implement this interface. The ToolRegistry uses it
to auto-discover schemas and route execution. The Brain uses it
to get Claude-compatible tool definitions.

Design: ABC + dataclass result type. No heavy frameworks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Standardized result from any tool execution."""

    success: bool
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0

    def to_content_string(self) -> str:
        """Convert result to a string suitable for Claude's tool_result block."""
        if self.success and self.data is not None:
            if isinstance(self.data, str):
                return self.data
            if isinstance(self.data, (list, dict)):
                import json
                return json.dumps(self.data, indent=2, default=str, ensure_ascii=False)
            return str(self.data)
        if self.error:
            return f"Error: {self.error}"
        return "Tool executed with no output."


class BaseTool(ABC):
    """
    Abstract base class for all FLEEA tools.

    Subclasses must implement:
    - name: str property
    - description: str property
    - get_schema() -> dict
    - execute(**kwargs) -> ToolResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name used in Claude API tool definitions."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for Claude to understand when to use it."""
        ...

    @property
    def requires_approval(self) -> bool:
        """Override to True for tools that always need safety approval."""
        return False

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """
        Return a Claude-compatible tool definition.
        Format:
        {
            "name": "tool_name",
            "description": "...",
            "input_schema": { "type": "object", "properties": {...}, "required": [...] }
        }
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given arguments. Must handle its own errors."""
        ...
