"""
FLEEA Settings — Centralized configuration via Pydantic BaseSettings.

Architecture:
    - Single source of truth for all runtime configuration.
    - Loaded from environment variables with .env fallback.
    - Instantiated ONCE in app.py, then passed via constructor
      to every module that needs it. No singleton, no global state.
    - All fields carry safe defaults so the system can boot with
      only the two mandatory API keys set.
    - Validators enforce invariants at startup, not at runtime.

Usage:
    settings = Settings()                        # loads .env automatically
    brain    = FLEEABrain(settings, ...)         # explicit injection
    logger   = ExecutionLogger(settings, ...)    # explicit injection
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project Root ───────────────────────────────────────────────────
# Resolved once at import time. Every path default is relative to this.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application-wide configuration.

    Required env vars (will fail fast if absent):
        GOOGLE_API_KEY
        TAVILY_API_KEY

    Everything else has safe defaults and can be tuned via .env
    or environment variables on any OS.
    """

    # ── Pydantic Settings wiring ───────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",                    # silently drop unknown env vars
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  LLM PROVIDER CONFIGURATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Primary brain — Gemini (Google). REQUIRED.
    GOOGLE_API_KEY: str

    # Default model for all Gemini API calls.
    # Flash strikes the best cost / tool-calling-quality balance.
    DEFAULT_MODEL: str = "gemini-2.0-flash"

    # Maximum tokens the LLM may generate per response.
    MAX_RESPONSE_TOKENS: int = 4096

    # Maximum number of messages (user + assistant turns) to keep in
    # the conversation history.  Older messages are trimmed to prevent
    # exceeding the model's context window.  Each tool-use cycle adds
    # multiple messages, so 50 messages ≈ 15–20 user turns.
    MAX_CONTEXT_MESSAGES: int = 50

    # OpenAI fallback — STUB in Phase 1.  Set the key when ready;
    # the brain will not attempt fallback until Phase 2+.
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SEARCH CONFIGURATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Tavily Search API. REQUIRED.
    # Free tier: 1 000 searches / month → https://tavily.com
    TAVILY_API_KEY: str

    # Default maximum results per web search call.
    SEARCH_MAX_RESULTS: int = 5

    # HTTP timeout for outbound search requests (seconds).
    SEARCH_TIMEOUT_SECONDS: int = 30

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  RETRY / RESILIENCE CONFIGURATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # How many times to retry a failed LLM API call
    # (connection errors, 429 rate-limits, 5xx server errors).
    MAX_RETRIES: int = 3

    # Exponential backoff bounds (seconds) for retries.
    RETRY_MIN_WAIT: float = 2.0
    RETRY_MAX_WAIT: float = 30.0

    # Hard ceiling on consecutive tool-call iterations inside
    # a single user turn.  Prevents infinite agentic loops.
    MAX_TOOL_ITERATIONS: int = 10

    # Per-request timeout for the Gemini SDK (seconds).
    API_TIMEOUT_SECONDS: int = 60

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DATABASE / STORAGE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Path to the single SQLite database used for logs, conversations,
    # sessions, and user profile.  Relative paths are resolved from
    # the project root.
    DB_PATH: str = "database/fleea.db"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  LOGGING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Python log level name — DEBUG, INFO, WARNING, ERROR, CRITICAL.
    LOG_LEVEL: str = "INFO"

    # When True, raw API request/response payloads are written to
    # the execution log.  Useful for debugging; disable in prod
    # to keep log volume manageable.
    LOG_RAW_API: bool = False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  COST TRACKING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Monthly budget ceiling (USD).  When estimated spend crosses
    # this threshold the agent emits a warning — it does NOT hard-
    # block because the user may intentionally exceed it.
    MONTHLY_COST_LIMIT: float = 10.0

    # Session-level cost warning threshold (USD).
    SESSION_COST_WARNING: float = 1.0

    # Enable / disable real-time cost tracking in the logger.
    COST_TRACKING_ENABLED: bool = True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SAFETY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # When True, unknown / unregistered tools default to
    # NEEDS_APPROVAL instead of SAFE.  Always keep this on.
    SAFETY_DEFAULT_DENY: bool = True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  VALIDATORS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(
                f"LOG_LEVEL must be one of {allowed}, got '{v}'"
            )
        return upper

    @field_validator("MAX_RETRIES")
    @classmethod
    def _validate_max_retries(cls, v: int) -> int:
        if v < 0 or v > 10:
            raise ValueError("MAX_RETRIES must be between 0 and 10")
        return v

    @field_validator("MAX_TOOL_ITERATIONS")
    @classmethod
    def _validate_max_iterations(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError("MAX_TOOL_ITERATIONS must be between 1 and 50")
        return v

    @field_validator("MONTHLY_COST_LIMIT", "SESSION_COST_WARNING")
    @classmethod
    def _validate_cost_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Cost thresholds cannot be negative")
        return v

    @model_validator(mode="after")
    def _validate_relationships(self) -> Settings:
        """Cross-field validation that runs after all fields are set."""
        if self.RETRY_MIN_WAIT > self.RETRY_MAX_WAIT:
            raise ValueError(
                f"RETRY_MIN_WAIT ({self.RETRY_MIN_WAIT}) cannot exceed "
                f"RETRY_MAX_WAIT ({self.RETRY_MAX_WAIT})"
            )
        if self.SESSION_COST_WARNING > self.MONTHLY_COST_LIMIT:
            raise ValueError(
                f"SESSION_COST_WARNING ({self.SESSION_COST_WARNING}) should not "
                f"exceed MONTHLY_COST_LIMIT ({self.MONTHLY_COST_LIMIT})"
            )
        return self

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DERIVED PROPERTIES (not stored, computed on access)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def project_root(self) -> Path:
        """Absolute path to the FLEEA project directory."""
        return _PROJECT_ROOT

    @property
    def db_path_resolved(self) -> Path:
        """
        Fully resolved database path.
        If DB_PATH is relative, it is resolved against project_root.
        """
        p = Path(self.DB_PATH)
        if p.is_absolute():
            return p
        return _PROJECT_ROOT / p

    @property
    def log_level_int(self) -> int:
        """LOG_LEVEL as a Python logging integer constant."""
        return getattr(logging, self.LOG_LEVEL, logging.INFO)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DISPLAY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def to_safe_dict(self) -> dict[str, Any]:
        """
        Return all settings as a dict with API keys masked.
        Safe for logging, debugging, and /status display.
        """
        data = self.model_dump()
        for key in data:
            if "KEY" in key or "SECRET" in key or "TOKEN" in key:
                val = data[key]
                if isinstance(val, str) and len(val) > 8:
                    data[key] = val[:4] + "****" + val[-4:]
                elif isinstance(val, str) and val:
                    data[key] = "****"
        return data
