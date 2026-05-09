"""
FLEEA Short-Term Memory — Bounded conversation context buffer.

Architecture:
    ShortTermMemory is a bounded, session-scoped message buffer that
    maintains the most recent conversation turns for Claude's context
    window.  It is the bridge between raw user/assistant exchanges
    and the structured message list that the Brain sends to the API.

    The Brain owns the full API message list (which includes tool_use
    and tool_result blocks).  ShortTermMemory tracks only the *semantic*
    conversation: user messages and assistant text responses.  This
    separation is intentional:

        Brain._messages   →  raw API payload (user, assistant, tool_use,
                             tool_result blocks) — drives the agentic loop
        ShortTermMemory   →  semantic history (user text ↔ assistant text)
                             — drives context recall and future summarization

    By keeping semantic history separate, we gain:
        1. Clean input for future summarization pipelines (Phase 2+)
        2. Accurate turn-count limits (tool exchanges don't inflate the count)
        3. Portable context — can be serialized, exported, or fed to a
           different model without Anthropic-specific tool block formatting
        4. Natural integration point for long-term memory retrieval

Design:
    - Configurable capacity with sane default (10 turns = 20 messages).
    - Automatic FIFO trimming — oldest turns drop first.
    - Session-aware: can be scoped to a session_id and reset cleanly.
    - Immutable message dicts: stored as copies, returned as copies.
    - Thread-safe for read operations; write operations are single-
      threaded (desktop app, one user, one brain).
    - Zero external dependencies — pure Python dataclass + list.

Integration:
    # In brain.py __init__:
    self._short_term = ShortTermMemory(max_turns=10)

    # After each completed user→assistant exchange:
    self._short_term.add_message("user", user_message)
    self._short_term.add_message("assistant", response_text)

    # To get recent context (e.g., for summarization or retrieval):
    recent = self._short_term.get_recent_messages(n=5)

Usage:
    mem = ShortTermMemory(max_turns=10)
    mem.add_message("user", "What's the weather?")
    mem.add_message("assistant", "It's 72°F and sunny in Seattle.")
    mem.add_message("user", "And tomorrow?")
    mem.add_message("assistant", "Rain expected, high of 58°F.")

    recent = mem.get_recent_messages(n=2)
    # → [{"role": "user", "content": "And tomorrow?", ...},
    #    {"role": "assistant", "content": "Rain expected...", ...}]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


_log = logging.getLogger("fleea.memory.short_term")

# Default maximum conversation turns (1 turn = 1 user + 1 assistant msg).
# 10 turns = 20 messages — fits comfortably in Claude's context alongside
# tool exchanges, and is enough for natural conversation continuity.
_DEFAULT_MAX_TURNS: int = 10


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MESSAGE DATA CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class Message:
    """
    Immutable record of a single conversation message.

    Frozen to prevent accidental mutation after storage.
    Slots for memory efficiency — we may hold hundreds of these
    across sessions in a long-running process.
    """

    role: str                  # "user" or "assistant"
    content: str               # the message text
    timestamp: str             # ISO 8601 UTC, set automatically
    session_id: str = ""       # optional — scopes to a session

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for API consumption or logging."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }

    def to_api_dict(self) -> dict[str, str]:
        """
        Minimal dict for Claude API's messages parameter.

        Only includes 'role' and 'content' — the two fields Claude
        expects.  Strips metadata to keep API payloads clean.
        """
        return {"role": self.role, "content": self.content}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHORT-TERM MEMORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ShortTermMemory:
    """
    Bounded FIFO buffer for recent conversation messages.

    Stores up to `max_turns` complete conversation turns (each turn
    is one user message + one assistant response = 2 messages).

    When the buffer exceeds capacity, the oldest messages are
    trimmed automatically in pairs (one full turn at a time) to
    maintain a valid alternating user/assistant sequence.

    Thread-safety:
        Single-writer (the brain), safe for concurrent reads.
        No locking needed for a single-user desktop app.
    """

    def __init__(
        self,
        max_turns: int = _DEFAULT_MAX_TURNS,
        session_id: str = "",
    ) -> None:
        """
        Initialize short-term memory.

        Args:
            max_turns: Maximum conversation turns to retain.
                       1 turn = 1 user msg + 1 assistant msg.
                       Default: 10 turns (20 messages).
            session_id: Optional session scope. When set, all messages
                        are tagged with this ID for traceability.
        """
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")

        self._max_turns = max_turns
        self._max_messages = max_turns * 2  # 1 turn = 2 messages
        self._session_id = session_id
        self._messages: list[Message] = []
        self._total_added: int = 0     # lifetime count (never resets on trim)
        self._total_trimmed: int = 0   # lifetime count of trimmed messages

        _log.debug(
            "ShortTermMemory initialized: max_turns=%d  session=%s",
            max_turns,
            session_id or "(unscoped)",
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PUBLIC API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def add_message(self, role: str, content: str) -> None:
        """
        Append a message to the conversation buffer.

        Automatically trims oldest messages when the buffer exceeds
        capacity.  Trims in pairs (full turns) to keep the
        user→assistant alternation intact.

        Args:
            role: "user" or "assistant".
            content: The message text.  Empty content is stored
                     (an empty assistant response is still a response).

        Raises:
            ValueError: If role is not "user" or "assistant".
        """
        if role not in ("user", "assistant"):
            raise ValueError(
                f"role must be 'user' or 'assistant', got '{role}'"
            )

        msg = Message(
            role=role,
            content=content,
            timestamp=_now_iso(),
            session_id=self._session_id,
        )

        self._messages.append(msg)
        self._total_added += 1

        # Trim after adding — keeps capacity invariant tight
        self._trim_memory()

    def get_recent_messages(
        self,
        n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most recent messages as serialized dicts.

        Returns full message dicts including metadata (timestamp,
        session_id).  For Claude API format, use get_api_messages().

        Args:
            n: Number of messages to return.  None returns all
               buffered messages.  Values exceeding the buffer
               size return everything available.

        Returns:
            List of message dicts, oldest first (chronological).
        """
        if n is None or n >= len(self._messages):
            source = self._messages
        else:
            source = self._messages[-n:]

        return [msg.to_dict() for msg in source]

    def get_api_messages(
        self,
        n: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Retrieve recent messages in Claude API format.

        Returns minimal dicts with only 'role' and 'content' —
        ready to be appended to the Brain's messages list or
        used as standalone context for a separate API call.

        Args:
            n: Number of messages to return.  None returns all.

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        if n is None or n >= len(self._messages):
            source = self._messages
        else:
            source = self._messages[-n:]

        return [msg.to_api_dict() for msg in source]

    def get_last_user_message(self) -> str | None:
        """
        Get the content of the most recent user message, or None.

        Useful for context-aware tool selection and intent detection.
        """
        for msg in reversed(self._messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant_message(self) -> str | None:
        """
        Get the content of the most recent assistant message, or None.

        Useful for continuity checks and response deduplication.
        """
        for msg in reversed(self._messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def clear(self) -> None:
        """
        Clear all buffered messages.

        Does NOT reset lifetime counters (total_added, total_trimmed)
        — those are diagnostic and persist for the session.
        """
        count = len(self._messages)
        self._messages.clear()
        _log.info(
            "Short-term memory cleared: removed %d messages "
            "(lifetime: %d added, %d trimmed)",
            count,
            self._total_added,
            self._total_trimmed,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TRIMMING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _trim_memory(self) -> None:
        """
        Trim the buffer to stay within the max_messages ceiling.

        Removes oldest messages in pairs (one full turn) so the
        alternating user/assistant sequence stays valid.

        Called automatically after every add_message().  Safe to
        call manually as a no-op when under capacity.
        """
        if len(self._messages) <= self._max_messages:
            return

        excess = len(self._messages) - self._max_messages

        # Round up to nearest even number to keep turn pairs intact
        if excess % 2 != 0:
            excess += 1

        # Don't trim more than we have
        excess = min(excess, len(self._messages))

        self._messages = self._messages[excess:]
        self._total_trimmed += excess

        _log.debug(
            "Trimmed %d messages (buffer: %d/%d, lifetime trimmed: %d)",
            excess,
            len(self._messages),
            self._max_messages,
            self._total_trimmed,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SESSION MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def set_session(self, session_id: str) -> None:
        """
        Update the session scope.

        Future messages will be tagged with this session_id.
        Existing messages keep their original session_id.

        Call this when a new session starts without creating
        a new ShortTermMemory instance.
        """
        self._session_id = session_id
        _log.debug("Session updated: %s", session_id)

    @property
    def session_id(self) -> str:
        """Current session scope (empty string if unscoped)."""
        return self._session_id

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DIAGNOSTICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def message_count(self) -> int:
        """Number of messages currently in the buffer."""
        return len(self._messages)

    @property
    def turn_count(self) -> int:
        """Approximate number of complete turns in the buffer."""
        return len(self._messages) // 2

    @property
    def is_empty(self) -> bool:
        """Whether the buffer has no messages."""
        return len(self._messages) == 0

    @property
    def is_full(self) -> bool:
        """Whether the buffer is at maximum capacity."""
        return len(self._messages) >= self._max_messages

    @property
    def capacity(self) -> int:
        """Maximum number of messages the buffer can hold."""
        return self._max_messages

    @property
    def max_turns(self) -> int:
        """Maximum conversation turns the buffer can hold."""
        return self._max_turns

    def get_stats(self) -> dict[str, Any]:
        """
        Return diagnostic stats for /status or logging.

        Returns:
            Dict with current size, capacity, lifetime counters.
        """
        return {
            "message_count": self.message_count,
            "turn_count": self.turn_count,
            "max_turns": self._max_turns,
            "max_messages": self._max_messages,
            "is_full": self.is_full,
            "total_added": self._total_added,
            "total_trimmed": self._total_trimmed,
            "session_id": self._session_id or None,
        }

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory("
            f"messages={self.message_count}/{self._max_messages}, "
            f"turns={self.turn_count}/{self._max_turns}, "
            f"session={self._session_id!r})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
