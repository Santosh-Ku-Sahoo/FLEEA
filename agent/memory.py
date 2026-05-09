"""
FLEEA Memory — Short-term and long-term memory system.

Phase 1: In-memory conversation history only.
Phase 2+: ChromaDB vector store for semantic recall,
SQLite for structured facts, memory summarization pipeline.
"""

from __future__ import annotations


class MemoryManager:
    """
    Phase 1 stub — conversation history managed by the Brain.
    Future: Vector memory (ChromaDB), structured memory (SQLite),
    memory summarization, and context window management.
    """

    def __init__(self) -> None:
        pass

    def store(self, content: str, metadata: dict | None = None) -> None:
        """Store a memory entry. Phase 1: no-op."""
        pass

    def recall(self, query: str, top_k: int = 5) -> list[str]:
        """Recall relevant memories. Phase 1: returns empty."""
        return []

    def summarize(self) -> str:
        """Summarize stored memories. Phase 1: returns empty."""
        return ""
