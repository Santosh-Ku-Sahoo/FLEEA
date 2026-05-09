"""
FLEEA Vector Store — Persistent semantic memory via ChromaDB.

Architecture:
    VectorStore is the long-term memory backend for FLEEA.  It stores
    text snippets as dense vector embeddings in a persistent ChromaDB
    collection, enabling semantic search over the user's accumulated
    knowledge, preferences, and conversation highlights.

    This module is a *storage and retrieval* layer only.  It does NOT
    decide what to store — that responsibility belongs to higher-level
    modules (e.g., a MemoryManager that extracts salient facts from
    conversations and calls add_memory() selectively).

    Embedding model: all-MiniLM-L6-v2 (384-dim, ~80 MB, fast on CPU).
    Storage: ChromaDB with persistent SQLite+Parquet backend.
    Both run fully local — nothing leaves the machine.

Design:
    - ChromaDB client and SentenceTransformer loaded lazily on first use.
      This avoids a 3-5 second startup penalty if memory isn't used in
      a given session.
    - Single collection per instance (default: "fleea_memory").
    - ChromaDB handles deduplication via deterministic IDs.
    - All public methods catch and log exceptions — memory failures
      must never crash the agent.  Degraded recall is acceptable;
      a crashed REPL is not.
    - Structured MemoryResult dataclass for type-safe search returns.

Integration:
    # In app.py (wiring):
    from memory.vector_store import VectorStore
    vector_store = VectorStore(persist_dir=str(settings.project_root / "memory" / "chroma_db"))

    # Storing a memory:
    vector_store.add_memory(
        text="User prefers dark mode and uses Python 3.12",
        metadata={"source": "preference", "session_id": "abc-123"},
    )

    # Semantic search:
    results = vector_store.search("what programming language does the user use?", top_k=3)
    for r in results:
        print(r.text, r.score, r.metadata)

Usage:
    store = VectorStore(persist_dir="./data/chroma_db")
    mid = store.add_memory("The user's timezone is US/Eastern")
    results = store.search("timezone", top_k=1)
    store.delete_memory(mid)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_log = logging.getLogger("fleea.memory.vector_store")

# ── Defaults ───────────────────────────────────────────────────────

_DEFAULT_COLLECTION: str = "fleea_memory"
_DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
_DEFAULT_PERSIST_DIR: str = "memory/chroma_db"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SEARCH RESULT DATA CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class MemoryResult:
    """
    A single search result from the vector store.

    Frozen + slotted for immutability and memory efficiency.

    Attributes:
        memory_id: The unique identifier of the stored memory.
        text: The original text that was embedded and stored.
        score: Similarity score (0.0–1.0, higher = more similar).
               Derived from ChromaDB's L2 distance.
        metadata: Arbitrary metadata dict attached at storage time.
    """

    memory_id: str
    text: str
    score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging or API consumption."""
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VECTOR STORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VectorStore:
    """
    Persistent semantic memory backed by ChromaDB + sentence-transformers.

    Provides three operations:
        1. add_memory   — embed and store text with metadata
        2. search       — semantic similarity search over stored memories
        3. delete_memory — remove a specific memory by ID

    All heavy dependencies (ChromaDB client, embedding model) are loaded
    lazily on first use to keep startup fast.

    Thread-safety:
        ChromaDB's PersistentClient is thread-safe for reads.
        Single-writer assumption (desktop app, one brain) for writes.
    """

    def __init__(
        self,
        persist_dir: str = _DEFAULT_PERSIST_DIR,
        collection_name: str = _DEFAULT_COLLECTION,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """
        Initialize the vector store.

        Heavy resources (ChromaDB client, SentenceTransformer) are NOT
        loaded here — they are created lazily on the first operation
        to avoid a 3-5s startup delay when memory isn't used.

        Args:
            persist_dir: Directory for ChromaDB's persistent storage.
                         Created automatically if it doesn't exist.
            collection_name: Name of the ChromaDB collection.
            embedding_model: HuggingFace model ID for embeddings.
                             Default: all-MiniLM-L6-v2 (384-dim, fast).
        """
        self._persist_dir = Path(persist_dir)
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model

        # Lazy-loaded resources
        self._client: Any = None       # chromadb.PersistentClient
        self._collection: Any = None   # chromadb.Collection
        self._embedder: Any = None     # SentenceTransformer

        _log.debug(
            "VectorStore configured: dir=%s  collection=%s  model=%s",
            self._persist_dir,
            collection_name,
            embedding_model,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  LAZY INITIALIZATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _ensure_initialized(self) -> bool:
        """
        Initialize ChromaDB client and embedding model on first use.

        Returns True if ready, False if initialization failed.
        Failures are logged but never raised — the caller should
        degrade gracefully (e.g., return empty search results).
        """
        if self._collection is not None and self._embedder is not None:
            return True

        try:
            return self._initialize()
        except Exception as exc:
            _log.error("VectorStore initialization failed: %s", exc)
            return False

    def _initialize(self) -> bool:
        """
        Perform the actual heavy initialization.

        Separated from _ensure_initialized for clarity and testability.
        """
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        # ── ChromaDB client ───────────────────────────────────────
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
            ),
        )

        # get_or_create: idempotent — safe on every boot
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        _log.info(
            "ChromaDB ready: collection=%s  count=%d  path=%s",
            self._collection_name,
            self._collection.count(),
            self._persist_dir,
        )

        # ── Embedding model ───────────────────────────────────────
        from sentence_transformers import SentenceTransformer

        t0 = time.perf_counter()
        self._embedder = SentenceTransformer(self._embedding_model_name)
        load_ms = (time.perf_counter() - t0) * 1000

        _log.info(
            "Embedding model loaded: %s  (%.0fms)",
            self._embedding_model_name,
            load_ms,
        )

        return True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  ADD MEMORY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> str | None:
        """
        Embed and store a text memory with optional metadata.

        Args:
            text: The text to embed and store.  Must be non-empty.
            metadata: Arbitrary key-value pairs attached to the memory.
                      ChromaDB requires metadata values to be str, int,
                      float, or bool — complex types are auto-converted
                      to strings.
            memory_id: Optional deterministic ID.  If None, a content-
                       based hash is generated so identical text doesn't
                       create duplicate entries.

        Returns:
            The memory_id of the stored entry, or None if storage failed.
        """
        if not text or not text.strip():
            _log.warning("Attempted to store empty text — skipped")
            return None

        if not self._ensure_initialized():
            return None

        try:
            # Generate deterministic ID from content if not provided
            mid = memory_id or _content_hash(text)

            # Sanitize metadata for ChromaDB compatibility
            safe_meta = _sanitize_metadata(metadata) if metadata else {}

            # Embed the text
            embedding = self._embedder.encode(text).tolist()

            # Upsert: insert-or-update semantics — safe for retries
            # and deduplication of identical content.
            # ChromaDB rejects empty metadata dicts, so only include
            # metadatas when we have actual key-value pairs.
            upsert_kwargs: dict[str, Any] = {
                "ids": [mid],
                "embeddings": [embedding],
                "documents": [text],
            }
            if safe_meta:
                upsert_kwargs["metadatas"] = [safe_meta]

            self._collection.upsert(**upsert_kwargs)

            _log.info(
                "Memory stored: id=%s  text=%.60s...  keys=%s",
                mid[:12],
                text,
                list(safe_meta.keys()) if safe_meta else "none",
            )
            return mid

        except Exception as exc:
            _log.error("Failed to store memory: %s", exc)
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SEARCH
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def search(
        self,
        query: str,
        top_k: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryResult]:
        """
        Semantic similarity search over stored memories.

        Embeds the query text, finds the nearest neighbors in the
        vector space, and returns structured MemoryResult objects
        sorted by descending similarity.

        Args:
            query: The natural-language query to search for.
            top_k: Maximum number of results to return.
            where: Optional ChromaDB metadata filter.
                   Example: {"source": "preference"} to search
                   only preference memories.

        Returns:
            List of MemoryResult objects, best match first.
            Empty list if the store is empty, uninitialized, or
            an error occurs.
        """
        if not query or not query.strip():
            return []

        if not self._ensure_initialized():
            return []

        try:
            # Clamp top_k to collection size to avoid ChromaDB errors
            count = self._collection.count()
            if count == 0:
                return []
            effective_k = min(top_k, count)

            # Embed the query
            query_embedding = self._embedder.encode(query).tolist()

            # Build query kwargs
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": effective_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            raw = self._collection.query(**kwargs)

            return _parse_results(raw)

        except Exception as exc:
            _log.error("Search failed: %s", exc)
            return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DELETE MEMORY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by its ID.

        Args:
            memory_id: The ID returned by add_memory().

        Returns:
            True if the delete operation succeeded, False otherwise.
            Note: ChromaDB doesn't error on deleting non-existent IDs,
            so True doesn't guarantee the ID existed.
        """
        if not memory_id:
            return False

        if not self._ensure_initialized():
            return False

        try:
            self._collection.delete(ids=[memory_id])
            _log.info("Memory deleted: id=%s", memory_id[:12])
            return True
        except Exception as exc:
            _log.error("Failed to delete memory %s: %s", memory_id[:12], exc)
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  BULK OPERATIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def clear_collection(self) -> bool:
        """
        Delete ALL memories in the collection.

        Destructive — use with caution.  Intended for testing
        and explicit user-initiated "forget everything" commands.

        Returns:
            True if the operation succeeded.
        """
        if not self._ensure_initialized():
            return False

        try:
            count_before = self._collection.count()

            # ChromaDB: delete collection and recreate empty
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            _log.warning(
                "Collection cleared: removed %d memories", count_before,
            )
            return True
        except Exception as exc:
            _log.error("Failed to clear collection: %s", exc)
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DIAGNOSTICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def count(self) -> int:
        """Number of memories currently stored."""
        if not self._ensure_initialized():
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    @property
    def is_initialized(self) -> bool:
        """Whether the store has been initialized."""
        return self._collection is not None and self._embedder is not None

    def get_stats(self) -> dict[str, Any]:
        """
        Return diagnostic info for /status or logging.
        """
        initialized = self.is_initialized
        return {
            "initialized": initialized,
            "persist_dir": str(self._persist_dir),
            "collection": self._collection_name,
            "embedding_model": self._embedding_model_name,
            "memory_count": self._collection.count() if initialized and self._collection else 0,
        }

    def __repr__(self) -> str:
        count = self._collection.count() if self.is_initialized and self._collection else "?"
        return (
            f"VectorStore("
            f"memories={count}, "
            f"collection={self._collection_name!r}, "
            f"model={self._embedding_model_name!r})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _content_hash(text: str) -> str:
    """
    Generate a deterministic ID from text content.

    Uses SHA-256 truncated to 16 hex chars.  This ensures identical
    text always maps to the same ID, which ChromaDB's upsert uses
    for deduplication — re-storing the same text is a no-op update.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure all metadata values are ChromaDB-compatible types.

    ChromaDB accepts: str, int, float, bool.
    Anything else is converted to its string representation.
    """
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif value is None:
            # ChromaDB doesn't accept None — skip or convert
            safe[key] = ""
        else:
            safe[key] = str(value)
    return safe


def _parse_results(raw: dict[str, Any]) -> list[MemoryResult]:
    """
    Parse ChromaDB query response into structured MemoryResult objects.

    ChromaDB returns nested lists (one per query).  We always send
    one query at a time, so we unpack the first (and only) element.

    Distance is converted to a similarity score:
        score = 1.0 - (distance / 2.0)   for cosine distance
        (cosine distance ranges from 0 to 2; 0 = identical)
    """
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    results: list[MemoryResult] = []
    for i, mid in enumerate(ids):
        # Convert cosine distance to similarity score (0.0–1.0)
        distance = distances[i] if i < len(distances) else 0.0
        score = max(0.0, 1.0 - (distance / 2.0))

        results.append(MemoryResult(
            memory_id=mid,
            text=docs[i] if i < len(docs) else "",
            score=score,
            metadata=metas[i] if i < len(metas) else {},
        ))

    return results
