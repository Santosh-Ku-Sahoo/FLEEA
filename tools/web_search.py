"""
FLEEA Web Search Tool — Tavily API integration for real-time web retrieval.

Architecture:
    Tavily is purpose-built for AI agents: it returns clean, pre-ranked
    results with relevance scores and an optional AI-generated answer.
    Free tier: 1,000 searches/month.  https://tavily.com

    This tool implements the BaseTool interface and integrates with:
    - ToolRegistry: for schema auto-discovery and execution routing
    - SafetyManager: classified as SAFE (no approval required)
    - ExecutionLogger: tool_input/tool_result are structured dicts
    - Brain: returns Claude-ready tool_result content strings

Design:
    - Settings injected via constructor for API key, timeout, max results.
    - Built-in retry with exponential backoff for transient failures
      (429, 500, 502, 503, 504, connection errors).
    - Structured SearchResult dataclass — never returns raw JSON.
    - All failures return a ToolResult with success=False and a
      human-readable error message.  Never raises to the caller.

Usage:
    tool = WebSearchTool(settings)
    result = await tool.execute(query="latest AI news", max_results=5)
    if result.success:
        for r in result.data["results"]:
            print(r["title"], r["url"], r["confidence"])
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from config.settings import Settings
from tools.base_tool import BaseTool, ToolResult


_log = logging.getLogger("fleea.tools.web_search")

_TAVILY_API_URL: str = "https://api.tavily.com/search"

# HTTP status codes that are safe to retry on (transient errors)
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRUCTURED RESULT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class SearchResult:
    """
    One structured search result.
    Never returned raw — always formatted by _format_results().
    """

    title: str
    summary: str        # first ~500 chars of page content
    source: str         # domain name extracted from URL
    url: str
    confidence: float   # Tavily relevance score [0.0 – 1.0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "confidence": self.confidence,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WEB SEARCH TOOL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class WebSearchTool(BaseTool):
    """
    Search the web using Tavily Search API.

    Returns structured results with title, summary, source, URL,
    and a confidence score.  Handles retries, timeouts, and all
    failure modes gracefully — never raises.
    """

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.TAVILY_API_KEY
        self._timeout = settings.SEARCH_TIMEOUT_SECONDS
        self._max_results = settings.SEARCH_MAX_RESULTS
        self._max_retries = settings.MAX_RETRIES
        self._retry_min_wait = settings.RETRY_MIN_WAIT
        self._retry_max_wait = settings.RETRY_MAX_WAIT

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  INTERFACE IMPLEMENTATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current information, news, facts, or any topic. "
            "Use this when the user asks about recent events, needs real-time data "
            "(weather, stocks, scores), or asks something you're unsure about. "
            "Returns structured results with titles, summaries, source domains, "
            "URLs, and relevance confidence scores."
        )

    @property
    def requires_approval(self) -> bool:
        return False  # web_search is SAFE tier

    def get_schema(self) -> dict[str, Any]:
        """Claude-compatible tool definition for web_search."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query. Write like a skilled researcher: "
                            "concise, keyword-rich, no personal details."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Maximum number of results to return (1–10). "
                            f"Default: {self._max_results}."
                        ),
                        "default": self._max_results,
                    },
                },
                "required": ["query"],
            },
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute a web search via Tavily API.

        Returns ToolResult with structured data on success, or a
        human-readable error on any failure.  Never raises.
        """
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", self._max_results)

        # ── Input validation ──
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                success=False,
                error="Search query cannot be empty.",
            )

        # Clamp max_results to safe range
        max_results = max(1, min(int(max_results), 10))

        # ── Execute with retry ──
        return await self._execute_with_retry(query.strip(), max_results)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  RETRY ENGINE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _execute_with_retry(
        self, query: str, max_results: int
    ) -> ToolResult:
        """
        Execute the search with built-in exponential backoff retry.

        Retries on:
            - HTTP 429 (rate limit)
            - HTTP 5xx (server errors)
            - Connection errors (network flake)
            - Timeout errors (slow response)

        Does NOT retry on:
            - HTTP 401 (bad API key — no point retrying)
            - HTTP 400 (bad request — our fault)
            - Successful empty results (not an error)
        """
        last_error: str = ""
        start = time.perf_counter()

        for attempt in range(1 + self._max_retries):
            try:
                result = await self._make_request(query, max_results)

                # Successful response (any HTTP code handled inside)
                if result is not None:
                    result_duration = (time.perf_counter() - start) * 1000
                    return ToolResult(
                        success=result["success"],
                        data=result.get("data"),
                        error=result.get("error"),
                        duration_ms=result_duration,
                    )

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                _log.warning(
                    "Search attempt %d/%d failed: %s",
                    attempt + 1, 1 + self._max_retries, last_error,
                )

            except Exception as exc:
                # Non-retryable unknown error
                duration_ms = (time.perf_counter() - start) * 1000
                return ToolResult(
                    success=False,
                    error=f"Search failed: {type(exc).__name__}: {exc}",
                    duration_ms=duration_ms,
                )

            # Backoff before next retry (skip on last attempt)
            if attempt < self._max_retries:
                wait = min(
                    self._retry_min_wait * (2 ** attempt),
                    self._retry_max_wait,
                )
                _log.info("Retrying in %.1fs...", wait)
                await asyncio.sleep(wait)

        # All retries exhausted
        duration_ms = (time.perf_counter() - start) * 1000
        return ToolResult(
            success=False,
            error=(
                f"Search failed after {1 + self._max_retries} attempts. "
                f"Last error: {last_error}"
            ),
            duration_ms=duration_ms,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  HTTP REQUEST
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _make_request(
        self, query: str, max_results: int
    ) -> dict[str, Any] | None:
        """
        Make a single HTTP request to the Tavily API.

        Returns:
            dict with "success", "data", "error" keys on HTTP response.
            None if a retryable network error occurs (caller handles retry).

        Raises:
            httpx.TimeoutException, httpx.ConnectError — retryable
            Exception — non-retryable
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                _TAVILY_API_URL,
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "include_raw_content": False,
                    "search_depth": "basic",
                },
            )

        # ── Handle HTTP errors ──
        status = response.status_code

        if status == 401:
            return {
                "success": False,
                "error": (
                    "Invalid Tavily API key. "
                    "Check TAVILY_API_KEY in your .env file."
                ),
            }

        if status == 400:
            return {
                "success": False,
                "error": f"Bad search request: {response.text[:200]}",
            }

        if status in _RETRYABLE_STATUS_CODES:
            _log.warning("Tavily returned HTTP %d — will retry", status)
            raise httpx.ReadError(f"Retryable HTTP {status}")

        if status != 200:
            return {
                "success": False,
                "error": f"Tavily API error: HTTP {status}",
            }

        # ── Parse response ──
        try:
            data = response.json()
        except Exception:
            return {
                "success": False,
                "error": "Tavily returned invalid JSON response.",
            }

        # ── Validate response structure ──
        if not isinstance(data, dict):
            return {
                "success": False,
                "error": "Tavily returned unexpected response format.",
            }

        # ── Format results ──
        formatted = _format_results(query, data)
        return {"success": True, "data": formatted}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RESULT FORMATTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _format_results(query: str, data: dict[str, Any]) -> dict[str, Any]:
    """
    Transform raw Tavily response into clean structured output.

    Returns:
        {
            "query": "...",
            "answer": "...",               # Tavily's AI answer (may be empty)
            "result_count": 5,
            "results": [
                {
                    "title": "...",
                    "summary": "...",       # first ~500 chars of content
                    "source": "nytimes.com",
                    "url": "https://...",
                    "confidence": 0.95      # Tavily relevance score
                },
                ...
            ]
        }
    """
    answer = data.get("answer", "") or ""
    raw_results = data.get("results", [])

    results: list[dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue

        url = item.get("url", "")
        result = SearchResult(
            title=item.get("title", "Untitled"),
            summary=_truncate(item.get("content", ""), 500),
            source=_extract_domain(url),
            url=url,
            confidence=round(float(item.get("score", 0.0)), 3),
        )
        results.append(result.to_dict())

    return {
        "query": query,
        "answer": answer,
        "result_count": len(results),
        "results": results,
    }


def _extract_domain(url: str) -> str:
    """
    Extract the domain name from a URL.

    'https://www.nytimes.com/2025/article' → 'nytimes.com'
    'https://docs.python.org/3/library'    → 'docs.python.org'
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.hostname or ""
        # Strip 'www.' prefix for cleaner display
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, adding ellipsis if truncated."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len - 3].rstrip() + "..."
