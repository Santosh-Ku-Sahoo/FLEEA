"""
FLEEA Memory Retriever — Query, filter, and format long-term memories.

Architecture:
    MemoryRetriever is the *read-side* interface to FLEEA's vector memory.
    It wraps VectorStore with two responsibilities that don't belong in
    the storage layer:

        1. Quality filtering — discard low-similarity results that would
           pollute the prompt with irrelevant context.
        2. Prompt formatting — convert structured MemoryResult objects
           into clean, human-readable text ready for injection into
           Claude's system prompt.

    The retriever does NOT decide what to store — that responsibility
    belongs to higher-level orchestration (e.g., MemoryManager).  It
    is a pure read-side concern.

Design:
    - Similarity threshold is configurable (default: 0.45 for cosine).
      Memories below this score are silently dropped — they'd hurt
      more than they help.
    - Maximum injection limit prevents memory from dominating the
      prompt.  Default: 5 memories ≈ 100–200 tokens.
    - Output format is a simple bulleted list under a header, designed
      to be appended to the system prompt without further processing.
    - All operations are safe — if the VectorStore is uninitialized or
      errors, the retriever returns empty results (never crashes).

Integration:
    # In app.py (wiring):
    from memory.vector_store import VectorStore
    from memory.retriever import MemoryRetriever

    vector_store = VectorStore(persist_dir="memory/chroma_db")
    retriever = MemoryRetriever(vector_store)

    # In brain.py (before Claude API call):
    memory_context = retriever.retrieve_and_format(user_message)
    if memory_context:
        system_prompt += "\\n" + memory_context

    # Or use the two-step API for inspection / logging:
    memories = retriever.retrieve_relevant("Python interview prep")
    formatted = retriever.format_for_prompt(memories)

Usage:
    retriever = MemoryRetriever(vector_store, threshold=0.5, max_memories=3)
    text = retriever.retrieve_and_format("what language does the user prefer?")
    # Returns:
    # "## Relevant Past Context\n\n- User prefers Python 3.12\n- User is preparing for interviews"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from memory.vector_store import MemoryResult, VectorStore


_log = logging.getLogger("fleea.memory.retriever")

# ── Defaults ───────────────────────────────────────────────────────

_DEFAULT_THRESHOLD: float = 0.45
_DEFAULT_MAX_MEMORIES: int = 5
_DEFAULT_TOP_K: int = 10
_PROMPT_HEADER: str = "## Relevant Past Context"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RETRIEVAL RESULT (for the two-step API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """
    A filtered, retriever-level view of a memory.

    Thinner than MemoryResult — exposes only what the prompt needs,
    plus the score for logging / diagnostics.

    Attributes:
        text: The stored memory text.
        score: Similarity score that passed the threshold.
        source: The ``source`` metadata tag, if present (e.g.,
                "preference", "conversation", "fact").
    """

    text: str
    score: float
    source: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MEMORY RETRIEVER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MemoryRetriever:
    """
    Read-side interface to FLEEA's long-term vector memory.

    Wraps VectorStore with quality filtering and prompt-ready formatting.

    Typical flow:
        1. ``retrieve_relevant(query)`` — search + filter
        2. ``format_for_prompt(results)``  — bullet-list text
        Or use the convenience one-liner:
        3. ``retrieve_and_format(query)``  — both in one call
    """

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        threshold: float = _DEFAULT_THRESHOLD,
        max_memories: int = _DEFAULT_MAX_MEMORIES,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            vector_store: The VectorStore instance to query.
            threshold: Minimum cosine similarity score (0.0–1.0).
                       Memories below this are discarded.
                       Default 0.45 balances recall vs. noise.
            max_memories: Maximum number of memories to return / inject.
                          Caps prompt bloat regardless of how many pass
                          the threshold.
        """
        self._store = vector_store
        self._threshold = threshold
        self._max_memories = max_memories

        _log.debug(
            "MemoryRetriever configured: threshold=%.2f  max=%d",
            threshold,
            max_memories,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  RETRIEVE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def retrieve_relevant(
        self,
        query: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Search vector memory and return only high-quality matches.

        Queries the VectorStore for ``top_k`` candidates, then filters
        out any result below the similarity threshold.  The surviving
        results are capped at ``max_memories`` and returned as lean
        RetrievalResult objects.

        Args:
            query: Natural-language query (typically the user's message).
            top_k: Number of candidates to fetch from ChromaDB before
                   filtering.  Set higher than max_memories so the
                   threshold filter has room to work.
            where: Optional ChromaDB metadata filter.
                   Example: ``{"source": "preference"}``

        Returns:
            List of RetrievalResult objects, best match first.
            Empty list if nothing passes the threshold, the store is
            empty, or an error occurs.
        """
        if not query or not query.strip():
            return []

        try:
            # Fetch candidates from the vector store
            raw_results: list[MemoryResult] = self._store.search(
                query=query,
                top_k=top_k,
                where=where,
            )

            # Filter by similarity threshold
            filtered = [r for r in raw_results if r.score >= self._threshold]

            # Cap at max_memories (results are already sorted best-first)
            capped = filtered[: self._max_memories]

            if capped:
                _log.info(
                    "Retrieved %d/%d memories (threshold=%.2f, query=%.50s...)",
                    len(capped),
                    len(raw_results),
                    self._threshold,
                    query,
                )
            else:
                _log.debug(
                    "No memories passed threshold %.2f for query: %.50s...",
                    self._threshold,
                    query,
                )

            return [_to_retrieval_result(r) for r in capped]

        except Exception as exc:
            _log.error("Memory retrieval failed: %s", exc)
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FORMAT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def format_for_prompt(memories: list[RetrievalResult]) -> str:
        """
        Format retrieved memories as clean text for prompt injection.

        Output example::

            ## Relevant Past Context

            - User prefers Python 3.12
            - User is preparing for backend interviews
            - User's timezone is US/Eastern

        Args:
            memories: List of RetrievalResult objects from
                      ``retrieve_relevant()``.

        Returns:
            A formatted string ready to append to the system prompt.
            Empty string if no memories are provided — callers should
            check truthiness before appending.
        """
        if not memories:
            return ""

        lines = [_PROMPT_HEADER, ""]
        for mem in memories:
            lines.append(f"- {mem.text}")

        return "\n".join(lines)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  CONVENIENCE: RETRIEVE + FORMAT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def retrieve_and_format(
        self,
        query: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
        where: dict[str, Any] | None = None,
    ) -> str:
        """
        One-call convenience: retrieve relevant memories and format
        them for prompt injection.

        Equivalent to::

            memories = retriever.retrieve_relevant(query)
            return retriever.format_for_prompt(memories)

        Args:
            query: Natural-language query.
            top_k: Candidates to fetch before filtering.
            where: Optional metadata filter.

        Returns:
            Formatted string, or empty string if nothing is relevant.
        """
        memories = self.retrieve_relevant(query, top_k=top_k, where=where)
        return self.format_for_prompt(memories)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DIAGNOSTICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def threshold(self) -> float:
        """Current similarity threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Update the similarity threshold at runtime."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be 0.0–1.0, got {value}")
        self._threshold = value
        _log.info("Retriever threshold updated to %.2f", value)

    @property
    def max_memories(self) -> int:
        """Current maximum memories cap."""
        return self._max_memories

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic info for /status or logging."""
        return {
            "threshold": self._threshold,
            "max_memories": self._max_memories,
            "store_initialized": self._store.is_initialized,
            "store_count": self._store.count,
        }

    def __repr__(self) -> str:
        return (
            f"MemoryRetriever("
            f"threshold={self._threshold}, "
            f"max_memories={self._max_memories})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _to_retrieval_result(result: MemoryResult) -> RetrievalResult:
    """
    Convert a VectorStore MemoryResult to a lean RetrievalResult.

    Extracts the ``source`` tag from metadata if present, defaulting
    to ``"unknown"`` for memories stored without a source label.
    """
    return RetrievalResult(
        text=result.text,
        score=result.score,
        source=result.metadata.get("source", "unknown"),
    )
