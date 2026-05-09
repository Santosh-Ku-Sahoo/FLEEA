"""
FLEEA Brain — Core agentic loop with multi-tool calling and memory.

Architecture:
    The Brain is the central nervous system of FLEEA.  It manages the
    full lifecycle of a user request:

        User message  →  memory retrieval  →  Claude API  →  detect tool_use
        →  safety gate  →  tool execution via ToolRegistry  →  tool results
        back to Claude  →  final text response  →  memory storage

    Every dependency is injected via the constructor — no globals, no
    singletons.  Memory subsystems are optional: if not provided the
    brain operates in Phase 1 mode with no degradation.

Phase 2 memory integration:
    BEFORE LLM call:
        - Retrieve relevant long-term memories via MemoryRetriever
        - Retrieve user profile via ProfileManager
        - Inject both into the system prompt as context sections
    AFTER response:
        - Store user + assistant messages in ShortTermMemory
        - Extract profile signals from user input (rule-based)
        - Store important user inputs in VectorStore for future recall

Design:
    - Multi-tool support: Claude can request multiple tools per turn
    - Agentic loop with configurable max iterations
    - Retry with exponential backoff on transient API errors
    - Cost tracking on every API call
    - Memory failures never crash the agent — degrade gracefully
    - Safety gate is non-negotiable: the LLM cannot bypass it

Usage:
    brain = FLEEABrain(
        settings=settings,
        tool_registry=registry,
        logger=logger,
        safety=safety,
        session_manager=session_mgr,
        short_term_memory=stm,
        memory_retriever=retriever,
        profile_manager=profile_mgr,
        vector_store=vector_store,
    )
    response = await brain.think("What's the latest AI news?")
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import anthropic
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent.logger import ExecutionLogger, Timer, TokenUsage
from agent.safety import SafetyManager, SafetyVerdict
from agent.session import SessionManager
from config.prompts import build_system_prompt
from config.settings import Settings
from tools.tool_registry import ToolRegistry

# Phase 2 memory imports — guarded so brain works without them
try:
    from memory.short_term import ShortTermMemory
    from memory.retriever import MemoryRetriever
    from memory.profile_manager import ProfileManager
    from memory.vector_store import VectorStore
except ImportError:  # pragma: no cover
    ShortTermMemory = None  # type: ignore[assignment,misc]
    MemoryRetriever = None  # type: ignore[assignment,misc]
    ProfileManager = None   # type: ignore[assignment,misc]
    VectorStore = None      # type: ignore[assignment,misc]


_log = logging.getLogger("fleea.brain")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BRAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FLEEABrain:
    """
    Core agentic loop: User → Claude → Tool Use → Tool Result → Claude → Response.

    Supports multi-turn, multi-tool calling with:
        - Safety gating before every tool execution
        - Token + cost tracking on every API call
        - Max iteration guard to prevent infinite loops
        - Retry with exponential backoff on transient API failures
        - Conversation history management
        - Cost threshold warnings
    """

    def __init__(
        self,
        *,
        settings: Settings,
        tool_registry: ToolRegistry,
        logger: ExecutionLogger,
        safety: SafetyManager,
        session_manager: SessionManager,
        user_profile_context: str = "",
        # Phase 2 — memory subsystems (all optional)
        short_term_memory: Any | None = None,
        memory_retriever: Any | None = None,
        profile_manager: Any | None = None,
        vector_store: Any | None = None,
    ) -> None:
        # ── Claude API client ─────────────────────────────────────
        self._client = anthropic.Anthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=settings.API_TIMEOUT_SECONDS,
        )
        self._model = settings.DEFAULT_MODEL
        self._max_tokens = settings.MAX_RESPONSE_TOKENS

        # ── Agentic loop tuning ───────────────────────────────────
        self._max_tool_iterations = settings.MAX_TOOL_ITERATIONS
        self._max_retries = settings.MAX_RETRIES
        self._retry_min_wait = settings.RETRY_MIN_WAIT
        self._retry_max_wait = settings.RETRY_MAX_WAIT
        self._max_context_messages = settings.MAX_CONTEXT_MESSAGES

        # ── Cost thresholds ───────────────────────────────────────
        self._session_cost_warning = settings.SESSION_COST_WARNING
        self._monthly_cost_limit = settings.MONTHLY_COST_LIMIT
        self._cost_tracking_enabled = settings.COST_TRACKING_ENABLED

        # ── Injected dependencies ─────────────────────────────────
        self._registry = tool_registry
        self._logger = logger
        self._safety = safety
        self._session = session_manager

        # ── Phase 2: Memory subsystems (graceful if absent) ───────
        self._short_term = short_term_memory
        self._retriever = memory_retriever
        self._profile_mgr = profile_manager
        self._vector_store = vector_store

        # ── System prompt (base — enriched per-turn dynamically) ──
        self._base_system_prompt = build_system_prompt(user_profile_context)
        self._system_prompt = self._base_system_prompt

        # ── Conversation state ────────────────────────────────────
        self._messages: list[dict[str, Any]] = []
        self._trimmed_count: int = 0  # total messages trimmed this session

        _log.info(
            "Brain initialized: model=%s  max_tokens=%d  tools=%s  "
            "memory=%s  retriever=%s  profile=%s  vector=%s",
            self._model,
            self._max_tokens,
            self._registry.tool_names,
            bool(self._short_term),
            bool(self._retriever),
            bool(self._profile_mgr),
            bool(self._vector_store),
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PUBLIC INTERFACE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def think(self, user_message: str) -> str:
        """
        Process a user message through the full agentic loop.

        Flow:
            1. PRE-LLM: Retrieve memories + profile → enrich system prompt
            2. Append user message to conversation history
            3. Call Claude API with tool schemas
            4. If Claude wants to use tools:
               a. Safety-check each tool call
               b. Execute approved tools via ToolRegistry
               c. Append tool results to messages
               d. Loop back to step 3 (up to max_tool_iterations)
            5. Return final text response
            6. POST-LLM: Store in short-term memory, extract profile,
               store important inputs in vector store

        Always returns a string — never raises to the caller.
        """
        session_id = self._session.session_id or "unknown"

        # ── PRE-LLM: Enrich system prompt with memory context ─────
        self._system_prompt = self._build_dynamic_prompt(user_message)

        # Append user message to conversation history
        self._messages.append({"role": "user", "content": user_message})

        # ── Run the agentic loop ──────────────────────────────────
        tools_invoked: list[str] = []
        total_usage = TokenUsage()

        try:
            response_text, tools_invoked, total_usage = await self._agentic_loop(
                session_id, user_message,
            )
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            _log.exception("Unhandled error in agentic loop")

            self._logger.log_error(
                session_id=session_id,
                user_input=user_message,
                error=error_msg,
                error_type=type(exc).__name__,
            )

            response_text = (
                "I encountered an unexpected error while processing your request. "
                "Please try again, or rephrase your question."
            )

        # ── Append assistant response to history ──────────────────
        self._messages.append({"role": "assistant", "content": response_text})

        # ── POST-LLM: Memory storage ──────────────────────────────
        self._post_response_memory(user_message, response_text, session_id)

        # ── Log the conversation turn ─────────────────────────────
        self._logger.log_conversation(
            session_id=session_id,
            user_message=user_message,
            assistant_response=response_text,
            tool_invoked=", ".join(tools_invoked) if tools_invoked else None,
        )

        # ── Update session counters ───────────────────────────────
        self._session.record_interaction(
            tool_used=bool(tools_invoked),
        )

        # ── Flush stats to DB periodically ────────────────────────
        self._logger.flush_session_stats(session_id)

        # ── Cost warnings ─────────────────────────────────────────
        if self._cost_tracking_enabled:
            warning = self._logger.check_cost_warning(
                session_id,
                self._session_cost_warning,
                self._monthly_cost_limit,
            )
            if warning:
                _log.warning("Cost threshold: %s", warning)

        return response_text

    @property
    def conversation_history(self) -> list[dict[str, Any]]:
        """Read-only copy of the current conversation history."""
        return list(self._messages)

    def clear_history(self) -> None:
        """Clear conversation history for the current session."""
        self._messages.clear()
        self._trimmed_count = 0
        _log.info("Conversation history cleared")

    def update_system_prompt(self, user_profile_context: str = "") -> None:
        """
        Rebuild the base system prompt with updated profile context.

        Called when the user profile changes mid-session so Claude
        can personalize subsequent responses.
        """
        self._base_system_prompt = build_system_prompt(user_profile_context)
        self._system_prompt = self._base_system_prompt
        _log.info("Base system prompt updated with new profile context")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  AGENTIC LOOP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _agentic_loop(
        self,
        session_id: str,
        user_input: str,
    ) -> tuple[str, list[str], TokenUsage]:
        """
        The core loop: call Claude → handle tool use → repeat.

        Runs until Claude returns a final text response (stop_reason=end_turn)
        or the iteration ceiling is hit.

        Returns:
            (response_text, tools_invoked, cumulative_usage)
        """
        tools_invoked: list[str] = []
        cumulative_usage = TokenUsage()
        iteration = 0

        while iteration < self._max_tool_iterations:
            iteration += 1
            _log.debug("Agentic loop iteration %d/%d", iteration, self._max_tool_iterations)

            # ── Trim history to stay within context budget ─────────
            self._trim_history()

            # ── Call Claude ────────────────────────────────────────
            response = await self._call_claude_api()

            # ── Track token usage ─────────────────────────────────
            usage = self._logger.extract_usage(response)
            self._logger.track_api_usage(session_id, usage)
            cumulative_usage = _accumulate_usage(cumulative_usage, usage)

            # ── Check stop reason ─────────────────────────────────
            stop_reason = response.stop_reason

            if stop_reason == "end_turn":
                # Claude is done — extract final text
                return _extract_text(response), tools_invoked, cumulative_usage

            if stop_reason == "tool_use":
                # Claude wants to use one or more tools
                tool_result_blocks, invoked = await self._handle_tool_calls(
                    response, session_id, user_input, usage,
                )
                tools_invoked.extend(invoked)

                # Append the assistant's response (with tool_use blocks)
                self._messages.append({
                    "role": "assistant",
                    "content": response.content,
                })

                # Append tool results so Claude can process them
                self._messages.append({
                    "role": "user",
                    "content": tool_result_blocks,
                })

                # Loop back — Claude will now process the tool results
                continue

            # ── Unexpected stop reason ─────────────────────────────
            # max_tokens, stop_sequence, etc.  Return whatever we have.
            _log.warning("Unexpected stop_reason: %s", stop_reason)
            return _extract_text(response), tools_invoked, cumulative_usage

        # ── Max iterations reached ─────────────────────────────────
        _log.warning(
            "Max tool iterations reached (%d) for input: %s",
            self._max_tool_iterations,
            user_input[:100],
        )
        return (
            "I've reached the maximum number of tool calls for this request. "
            "Here's what I have so far based on the information gathered.",
            tools_invoked,
            cumulative_usage,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TOOL CALL HANDLING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _handle_tool_calls(
        self,
        response: Any,
        session_id: str,
        user_input: str,
        usage: TokenUsage,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Process all tool_use blocks from a single Claude response.

        For each tool_use block:
            1. Extract tool name and arguments
            2. Run through safety gate (evaluate_and_gate)
            3. If approved → execute via ToolRegistry
            4. If denied → return error result to Claude
            5. Log the execution (success or denial)

        Returns:
            (tool_result_blocks, list_of_invoked_tool_names)
        """
        tool_result_blocks: list[dict[str, Any]] = []
        invoked_names: list[str] = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name: str = block.name
            tool_input: dict[str, Any] = block.input
            tool_use_id: str = block.id

            _log.info("Tool call detected: %s", tool_name)

            # ── Safety Gate ────────────────────────────────────────
            allowed, safety_check = self._safety.evaluate_and_gate(
                tool_name, tool_input,
            )

            if not allowed:
                # Denied by safety layer — send error back to Claude
                deny_reason = safety_check.reason
                _log.warning(
                    "Tool '%s' denied by safety: %s [%s]",
                    tool_name,
                    deny_reason,
                    safety_check.verdict.value,
                )

                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": f"Tool execution denied by safety system: {deny_reason}",
                    "is_error": True,
                })

                # Log the denied tool call
                self._logger.log_tool_execution(
                    session_id=session_id,
                    user_input=user_input,
                    detected_intent=f"tool_call:{tool_name}",
                    selected_tool=tool_name,
                    tool_input=tool_input,
                    tool_result={"denied": True, "reason": deny_reason},
                    success=False,
                    error=f"Safety {safety_check.verdict.value}: {deny_reason}",
                    usage=usage,
                )
                continue

            # ── Execute Tool ───────────────────────────────────────
            invoked_names.append(tool_name)

            with Timer() as timer:
                result = await self._registry.execute(tool_name, **tool_input)

            content_str = result.to_content_string()
            _log.info(
                "Tool '%s' executed: success=%s  duration=%.1fms",
                tool_name,
                result.success,
                timer.duration_ms,
            )

            # Build tool_result block for Claude
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content_str,
                "is_error": not result.success,
            })

            # Log the execution
            self._logger.log_tool_execution(
                session_id=session_id,
                user_input=user_input,
                detected_intent=f"tool_call:{tool_name}",
                selected_tool=tool_name,
                tool_input=tool_input,
                tool_result=content_str[:2000],  # Cap log size
                success=result.success,
                error=result.error,
                execution_duration_ms=timer.duration_ms,
                usage=usage,
            )

            # Record tool call on session (count only — logger owns cost)
            self._session.record_tool_call()

        return tool_result_blocks, invoked_names

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  CONVERSATION HISTORY MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _trim_history(self) -> None:
        """
        Trim conversation history to stay within the context budget.

        Keeps the most recent messages up to MAX_CONTEXT_MESSAGES.
        Always preserves at least the last user message so Claude
        has something to respond to.
        """
        limit = self._max_context_messages
        if len(self._messages) <= limit:
            return

        excess = len(self._messages) - limit
        self._messages = self._messages[excess:]
        self._trimmed_count += excess

        _log.info(
            "Trimmed %d old messages (total trimmed this session: %d, "
            "remaining: %d)",
            excess,
            self._trimmed_count,
            len(self._messages),
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  CLAUDE API CALL (with retry)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _call_claude_api(self) -> Any:
        """
        Call the Claude Messages API with retry on transient errors.

        The synchronous Anthropic client is run in a thread executor
        to avoid blocking the async event loop.

        Retries on:
            - APIConnectionError (network flake)
            - RateLimitError     (HTTP 429)
            - InternalServerError (HTTP 5xx)

        Does NOT retry on:
            - AuthenticationError (bad API key — no point retrying)
            - BadRequestError    (our fault — fix the request)
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._create_message)

    def _create_message(self) -> Any:
        """
        Synchronous Claude Messages API call — runs in thread executor.

        Uses tenacity.Retrying as a context manager so retry parameters
        (max_retries, backoff bounds) are read from instance attributes
        set by Settings, not hardcoded at class-definition time.
        """
        tools = self._registry.get_all_schemas()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": self._system_prompt,
            "messages": self._messages,
        }

        # Only include tools param if we have tools registered
        # (Claude rejects an empty tools list)
        if tools:
            kwargs["tools"] = tools

        _log.debug(
            "Calling Claude: model=%s  messages=%d  tools=%d",
            self._model,
            len(self._messages),
            len(tools),
        )

        retryer = Retrying(
            retry=retry_if_exception_type((
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.InternalServerError,
            )),
            wait=wait_exponential(
                multiplier=1,
                min=self._retry_min_wait,
                max=self._retry_max_wait,
            ),
            stop=stop_after_attempt(1 + self._max_retries),
            reraise=True,
            before_sleep=lambda rs: _log.warning(
                "Claude API retry %d: %s",
                rs.attempt_number,
                rs.outcome.exception() if rs.outcome else "unknown",
            ),
        )

        return retryer(self._client.messages.create, **kwargs)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DIAGNOSTICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def model(self) -> str:
        """The model ID currently in use."""
        return self._model

    @property
    def message_count(self) -> int:
        """Number of messages in the conversation history."""
        return len(self._messages)

    def get_status(self) -> dict[str, Any]:
        """
        Return brain diagnostics for the /status command.
        """
        status: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "max_tool_iterations": self._max_tool_iterations,
            "max_context_messages": self._max_context_messages,
            "message_count": self.message_count,
            "trimmed_count": self._trimmed_count,
            "registered_tools": self._registry.tool_names,
            "system_prompt_length": len(self._system_prompt),
            "memory_enabled": {
                "short_term": bool(self._short_term),
                "retriever": bool(self._retriever),
                "profile": bool(self._profile_mgr),
                "vector_store": bool(self._vector_store),
            },
        }
        if self._short_term:
            status["short_term_stats"] = self._short_term.get_stats()
        if self._profile_mgr:
            status["profile_stats"] = self._profile_mgr.get_stats()
        return status

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  MEMORY INTEGRATION (Phase 2)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_dynamic_prompt(self, user_message: str) -> str:
        """
        Enrich the base system prompt with memory and profile context.

        Called BEFORE each Claude API turn.  Builds the prompt in a
        deliberate order for maximum Claude comprehension:

            1. Base prompt (identity, safety, tools)
            2. User Profile (stable structured facts — highest trust)
            3. Relevant Past Context (semantically retrieved — use with care)

        Profile comes before memories because it's structured, verified,
        and higher-trust.  Retrieved memories are volatile and may contain
        noise — placing them last signals lower priority.

        Deduplication: if a memory repeats information already in the
        profile (e.g., "user prefers Python"), it's stripped to avoid
        wasting tokens and confusing the model with redundancy.

        Returns the enriched system prompt string.
        """
        sections: list[str] = [self._base_system_prompt]

        # ── 1. User Profile (structured, high-trust) ─────────────
        profile_text = ""
        if self._profile_mgr:
            try:
                profile_text = self._profile_mgr.format_for_prompt()
                if profile_text:
                    sections.append(profile_text)
                    _log.debug("Injected profile context (%d chars)", len(profile_text))
            except Exception as exc:
                _log.error("Profile retrieval failed (non-fatal): %s", exc)

        # ── 2. Relevant Past Context (semantic, lower-trust) ─────
        if self._retriever:
            try:
                memory_text = self._retriever.retrieve_and_format(user_message)
                if memory_text:
                    # Deduplicate: strip memory lines that are already
                    # covered by the profile to avoid redundant injection
                    memory_text = _deduplicate_memory(memory_text, profile_text)
                    if memory_text:
                        sections.append(memory_text)
                        _log.debug(
                            "Injected memory context (%d chars)",
                            len(memory_text),
                        )
            except Exception as exc:
                _log.error("Memory retrieval failed (non-fatal): %s", exc)

        return "\n\n".join(sections)

    def _post_response_memory(
        self,
        user_message: str,
        response_text: str,
        session_id: str,
    ) -> None:
        """
        Post-response memory operations.  Called AFTER the agentic loop.

        1. Store both messages in short-term memory (semantic history)
        2. Extract and persist profile signals from user input
        3. Store important user inputs in the vector store

        All operations are wrapped in try/except — memory failures
        must never crash the agent.
        """
        # ── 1. Short-term memory ──────────────────────────────────
        if self._short_term:
            try:
                self._short_term.add_message("user", user_message)
                self._short_term.add_message("assistant", response_text)
            except Exception as exc:
                _log.error("Short-term memory storage failed: %s", exc)

        # ── 2. Profile extraction ─────────────────────────────────
        if self._profile_mgr:
            try:
                updates = self._profile_mgr.extract_and_store(user_message)
                if updates:
                    _log.info(
                        "Extracted %d profile signal(s) from user input",
                        len(updates),
                    )
            except Exception as exc:
                _log.error("Profile extraction failed: %s", exc)

        # ── 3. Vector store (important inputs only) ───────────────
        if self._vector_store and _is_important(user_message):
            try:
                self._vector_store.add_memory(
                    text=user_message,
                    metadata={
                        "source": "conversation",
                        "session_id": session_id,
                        "role": "user",
                    },
                )
                _log.debug("Stored user input in vector memory")
            except Exception as exc:
                _log.error("Vector store failed: %s", exc)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _extract_text(response: Any) -> str:
    """
    Extract text content from a Claude API response.

    Claude responses contain a list of content blocks.  This pulls
    all text blocks and joins them.  Returns a fallback message
    if no text is found.
    """
    text_parts: list[str] = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    return "\n".join(text_parts) if text_parts else "I have no response."


def _accumulate_usage(total: TokenUsage, delta: TokenUsage) -> TokenUsage:
    """
    Accumulate token usage across multiple API calls in one agentic loop.
    Returns a new TokenUsage with summed values.
    """
    return TokenUsage(
        prompt_tokens=total.prompt_tokens + delta.prompt_tokens,
        completion_tokens=total.completion_tokens + delta.completion_tokens,
        total_tokens=total.total_tokens + delta.total_tokens,
        estimated_cost=total.estimated_cost + delta.estimated_cost,
    )


# ── Importance keywords that signal a message worth remembering ────
_IMPORTANCE_PATTERNS: re.Pattern[str] = re.compile(
    r"\b("
    r"i\s+prefer|i\s+like|i\s+use|i\s+need|i\s+want|"
    r"my\s+goal|my\s+name|i(?:\s+am|'m)\s+(?:working|building|studying|preparing)|"
    r"remember|don't\s+forget|important|always|never|"
    r"help\s+me|explain|teach|show\s+me|"
    r"project|deadline|interview|career|resume|portfolio"
    r")\b",
    re.IGNORECASE,
)

# Trivial messages to skip (greetings, single words, commands)
_TRIVIAL_CEILING = 15  # messages shorter than this are usually noise


def _is_important(text: str) -> bool:
    """
    Lightweight heuristic to decide if a user message is worth storing
    in the vector store for long-term recall.

    Returns True if the message is likely to contain a preference,
    goal, project detail, or other high-signal information.  Returns
    False for greetings, short commands, and trivial chatter.

    This is intentionally conservative — we'd rather miss some
    borderline messages than fill the vector store with noise.
    """
    if not text or len(text.strip()) < _TRIVIAL_CEILING:
        return False

    return bool(_IMPORTANCE_PATTERNS.search(text))


def _deduplicate_memory(memory_text: str, profile_text: str) -> str:
    """
    Remove memory lines that duplicate information already in the profile.

    When both the profile and retrieved memory mention "Python", injecting
    both wastes tokens and makes the model over-weight that signal.
    This function strips memory lines whose key content words are already
    present in the profile section.

    Args:
        memory_text: The formatted memory block (with header + bullet lines).
        profile_text: The formatted profile block.

    Returns:
        Cleaned memory text with redundant lines removed.
        Empty string if all lines were redundant.
    """
    if not profile_text or not memory_text:
        return memory_text

    profile_lower = profile_text.lower()

    # Split into header and content lines
    lines = memory_text.split("\n")
    header_lines: list[str] = []
    content_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            content_lines.append(line)
        else:
            header_lines.append(line)

    # Filter: keep content lines NOT already covered by profile
    kept: list[str] = []
    for line in content_lines:
        # Extract the semantic payload after "- "
        payload = line.strip().lstrip("- ").lower().strip()
        if not payload:
            continue

        # Check if all significant words in the payload appear in profile.
        # "User prefers Python" → check if "python" is in profile_lower.
        # Use content words (skip stop words like "user", "the", "is").
        _stop = {"user", "the", "is", "a", "an", "to", "in", "for", "and", "of", "it"}
        words = [w for w in payload.split() if w not in _stop and len(w) > 2]

        # If >70% of content words appear in profile, it's redundant
        if words:
            overlap = sum(1 for w in words if w in profile_lower)
            if overlap / len(words) >= 0.7:
                _log.debug("Deduped memory line (profile overlap): %s", line.strip())
                continue

        kept.append(line)

    if not kept:
        return ""

    return "\n".join(header_lines + kept)

