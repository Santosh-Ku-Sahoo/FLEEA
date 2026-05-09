"""
FLEEA Profile Manager — Structured user profile with auto-extraction.

Architecture:
    ProfileManager is the Phase 2 evolution of agent/profile.py.  It adds
    automatic extraction of user preferences, goals, and communication
    style from natural conversation, then persists them in the existing
    SQLite user_profile table via DatabaseManager.

    Extraction is rule-based (regex + keyword matching) — deliberately
    simple, fast, and deterministic.  No external NLP models, no network
    calls, no GPU.  The rules are grouped by profile category and each
    rule produces a (category, key, value) triple for the DB.

    This module is a *detection and storage* layer.  The caller (e.g.,
    brain.py or a future MemoryManager) decides *when* to call
    extract_profile_info() — typically after each user turn.

Design:
    - Rule-based extraction via compiled regex patterns for speed.
    - Three extraction categories: preferences, goals, communication.
    - Extraction returns a list of ProfileUpdate dataclasses so the
      caller can inspect / log / filter before committing to the DB.
    - All DB writes use upsert semantics — safe for repeated extraction
      of the same signal across turns.
    - Safe defaults: empty profile returns gracefully, extraction on
      irrelevant text returns an empty list.

Integration:
    # In app.py (wiring):
    from memory.profile_manager import ProfileManager
    profile_mgr = ProfileManager(db)

    # After each user message (in brain.py or MemoryManager):
    updates = profile_mgr.extract_profile_info(user_message)
    for u in updates:
        profile_mgr.update_profile(u.category, u.key, u.value)

    # For system prompt injection:
    profile = profile_mgr.get_profile()
    context = profile_mgr.format_for_prompt(profile)

Usage:
    mgr = ProfileManager(db)
    updates = mgr.extract_profile_info("I prefer Python and keep it short")
    # → [ProfileUpdate(category="preferences", key="language", value="Python"),
    #    ProfileUpdate(category="communication", key="style", value="concise")]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from database.models import DatabaseManager


_log = logging.getLogger("fleea.memory.profile_manager")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATA CLASSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class ProfileUpdate:
    """
    A single extracted profile signal, ready to be persisted.

    Attributes:
        category: Profile category (e.g., "preferences", "goals").
        key: The specific attribute (e.g., "language", "career_goal").
        value: The extracted value (e.g., "Python", "preparing for interviews").
    """

    category: str
    key: str
    value: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EXTRACTION RULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Each rule is a tuple: (compiled_regex, category, key, group_index)
# group_index is the regex capture group that contains the value.
# Rules are tried in order; multiple can match per input.

_ExtractionRule = tuple[re.Pattern[str], str, str, int]

_EXTRACTION_RULES: list[_ExtractionRule] = [
    # ─────────────────────────────────────────────────────────────
    # RULE ORDERING:  Specific patterns MUST come before broad
    # catch-alls.  "I prefer a casual tone" must match the tone
    # rule, not the generic "I prefer X" preference rule.
    # ─────────────────────────────────────────────────────────────

    # ── Communication style (SPECIFIC — checked first) ────────────

    # "I prefer formal/casual/informal tone"
    (
        re.compile(
            r"\b(?:i\s+prefer|use)\s+(?:a\s+)?(formal|casual|informal|professional|friendly)\s+(?:tone|style|language)",
            re.IGNORECASE,
        ),
        "communication", "tone", 1,
    ),
    # "keep it short" / "keep it concise" / "keep it brief"
    (
        re.compile(
            r"\bkeep\s+it\s+(short|concise|brief|simple|minimal)",
            re.IGNORECASE,
        ),
        "communication", "style", 1,
    ),
    # "be detailed" / "be verbose" / "be thorough"
    (
        re.compile(
            r"\bbe\s+(detailed|verbose|thorough|comprehensive|elaborate)",
            re.IGNORECASE,
        ),
        "communication", "style", 1,
    ),
    # "respond in Hindi" / "answer in Spanish"
    (
        re.compile(
            r"\b(?:respond|answer|reply|speak|talk)\s+(?:to\s+me\s+)?in\s+(\w+)",
            re.IGNORECASE,
        ),
        "communication", "language", 1,
    ),

    # ── Identity (SPECIFIC — checked before broad rules) ──────────

    # "My name is John" / "I'm called John"
    (
        re.compile(
            r"\b(?:my\s+name\s+is|i(?:\s+am|'m)\s+called|call\s+me)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",
            re.IGNORECASE,
        ),
        "identity", "name", 1,
    ),
    # "My timezone is US/Eastern" / "I'm in IST"
    (
        re.compile(
            r"\b(?:my\s+timezone\s+is|i(?:\s+am|'m)\s+in)\s+([A-Z][A-Za-z/_+-]+)",
            re.IGNORECASE,
        ),
        "identity", "timezone", 1,
    ),

    # ── Preferences (BROAD — checked after specific rules) ────────

    # "My favorite language is Rust"
    (
        re.compile(
            r"\bmy\s+(?:fav(?:ou?rite)?|preferred)\s+(?:language|editor|ide|framework|tool)\s+is\s+(.+?)(?:\.|,|\band\b|$)",
            re.IGNORECASE,
        ),
        "preferences", "favorite_tool", 1,
    ),
    # "I prefer Python" / "I prefer dark mode"
    # Breaks on sentence-end AND conjunctions to avoid capturing
    # "Python and keep it short" as one value.
    (
        re.compile(r"\bi\s+prefer\s+(.+?)(?:\.|,|\band\b|\bbut\b|\bso\b|$)", re.IGNORECASE),
        "preferences", "preference", 1,
    ),
    # "I like using TypeScript" / "I like dark themes"
    (
        re.compile(r"\bi\s+like\s+(?:using\s+)?(.+?)(?:\.|,|\band\b|\bbut\b|\bso\b|$)", re.IGNORECASE),
        "preferences", "preference", 1,
    ),
    # "I use Python 3.12" / "I use VS Code"
    (
        re.compile(r"\bi\s+use\s+(.+?)(?:\.|,|\band\b|\bbut\b|\bso\b|$)", re.IGNORECASE),
        "preferences", "tool_in_use", 1,
    ),

    # ── Goals ─────────────────────────────────────────────────────

    # "I am preparing for interviews"
    (
        re.compile(
            r"\bi(?:\s+am|'m)\s+(?:preparing|studying|training|practicing)\s+for\s+(.+?)(?:\.|,|\band\b|$)",
            re.IGNORECASE,
        ),
        "goals", "preparing_for", 1,
    ),
    # "I want to learn machine learning"
    (
        re.compile(
            r"\bi\s+want\s+to\s+(?:learn|study|master|understand)\s+(.+?)(?:\.|,|\band\b|$)",
            re.IGNORECASE,
        ),
        "goals", "learning_goal", 1,
    ),
    # "My goal is to become a senior engineer"
    (
        re.compile(r"\bmy\s+goal\s+is\s+(?:to\s+)?(.+?)(?:\.|,|\band\b|$)", re.IGNORECASE),
        "goals", "primary_goal", 1,
    ),
    # "I'm working on a portfolio project"
    (
        re.compile(
            r"\bi(?:\s+am|'m)\s+working\s+on\s+(.+?)(?:\.|,|\band\b|$)",
            re.IGNORECASE,
        ),
        "goals", "current_project", 1,
    ),
    # "I'm building an AI agent"
    (
        re.compile(
            r"\bi(?:\s+am|'m)\s+building\s+(.+?)(?:\.|,|\band\b|$)",
            re.IGNORECASE,
        ),
        "goals", "building", 1,
    ),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PROFILE MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ProfileManager:
    """
    Manages structured user profile data with auto-extraction.

    Combines three responsibilities:
        1. CRUD operations on the user_profile table (via DatabaseManager)
        2. Rule-based extraction of profile signals from user text
        3. Formatting profile data for prompt injection

    Thread-safety:
        Safe for single-writer (desktop app).  DatabaseManager handles
        connection-per-operation with WAL mode.
    """

    def __init__(self, db: DatabaseManager) -> None:
        """
        Initialize the ProfileManager.

        Args:
            db: The shared DatabaseManager instance.
        """
        self._db = db
        _log.debug("ProfileManager initialized")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  READ
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_profile(self) -> dict[str, dict[str, str]]:
        """
        Retrieve the full user profile, organized by category.

        Returns:
            Nested dict: ``{category: {key: value, ...}, ...}``
            Empty dict if no profile data exists.

        Example::

            {
                "preferences": {"preference": "Python", "tool_in_use": "VS Code"},
                "goals": {"preparing_for": "backend interviews"},
                "communication": {"style": "concise"},
            }
        """
        try:
            return self._db.get_full_profile()
        except Exception as exc:
            _log.error("Failed to read profile: %s", exc)
            return {}

    def get_value(
        self, key: str, category: str = "preferences",
    ) -> str | None:
        """
        Get a single profile value.

        Args:
            key: The profile attribute name.
            category: Profile category (default: "preferences").

        Returns:
            The value string, or None if not set.
        """
        try:
            return self._db.get_profile_value(category, key)
        except Exception as exc:
            _log.error("Failed to read profile value %s/%s: %s", category, key, exc)
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  WRITE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def update_profile(self, category: str, key: str, value: str) -> bool:
        """
        Upsert a single profile entry.

        Args:
            category: Profile category (e.g., "preferences", "goals").
            key: The attribute name.
            value: The value to store.

        Returns:
            True if the write succeeded, False on error.
        """
        if not key or not value:
            return False

        try:
            self._db.set_profile(category, key, value)
            _log.info(
                "Profile updated: [%s] %s = %s",
                category, key, value,
            )
            return True
        except Exception as exc:
            _log.error("Failed to update profile %s/%s: %s", category, key, exc)
            return False

    def delete_entry(self, key: str, category: str = "preferences") -> bool:
        """
        Delete a single profile entry.

        Returns:
            True if a row was actually removed.
        """
        try:
            removed = self._db.delete_profile_entry(category, key)
            if removed:
                _log.info("Profile entry deleted: [%s] %s", category, key)
            return removed
        except Exception as exc:
            _log.error("Failed to delete profile %s/%s: %s", category, key, exc)
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  EXTRACT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def extract_profile_info(self, text: str) -> list[ProfileUpdate]:
        """
        Extract profile signals from user text using rule-based matching.

        Scans the input against compiled regex patterns to detect:
            - Preferences (language, tools, themes)
            - Goals (learning, career, projects)
            - Communication style (tone, verbosity, language)
            - Identity (name, timezone)

        Each match produces a ProfileUpdate dataclass.  Multiple signals
        can be extracted from a single message.

        Args:
            text: Raw user message text.

        Returns:
            List of ProfileUpdate objects.  Empty list if nothing was
            detected — this is the common case for most messages.
        """
        if not text or not text.strip():
            return []

        updates: list[ProfileUpdate] = []
        seen: set[tuple[str, str]] = set()  # deduplicate (category, key) per call

        for pattern, category, key, group_idx in _EXTRACTION_RULES:
            match = pattern.search(text)
            if match:
                raw_value = match.group(group_idx).strip()
                value = _clean_extracted_value(raw_value)

                if not value:
                    continue

                # Deduplicate: if a more specific rule already matched
                # the same (category, key), skip the broader one
                dedup_key = (category, key)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                updates.append(ProfileUpdate(
                    category=category,
                    key=key,
                    value=value,
                ))

        if updates:
            _log.info(
                "Extracted %d profile signal(s) from: %.60s...",
                len(updates),
                text,
            )

        return updates

    def extract_and_store(self, text: str) -> list[ProfileUpdate]:
        """
        Convenience: extract profile signals and persist them immediately.

        Equivalent to::

            updates = mgr.extract_profile_info(text)
            for u in updates:
                mgr.update_profile(u.category, u.key, u.value)
            return updates

        Args:
            text: Raw user message text.

        Returns:
            List of ProfileUpdate objects that were stored.
        """
        updates = self.extract_profile_info(text)
        for update in updates:
            self.update_profile(update.category, update.key, update.value)
        return updates

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FORMAT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def format_for_prompt(
        self, profile: dict[str, dict[str, str]] | None = None,
    ) -> str:
        """
        Format profile data as clean text for system prompt injection.

        Output example::

            ## User Profile

            ### Preferences
            - preference: Python
            - tool_in_use: VS Code

            ### Goals
            - preparing_for: backend interviews

            ### Communication
            - style: concise

        Args:
            profile: Profile dict from ``get_profile()``.  If None,
                     fetches the current profile from the DB.

        Returns:
            Formatted markdown string.  Empty string if the profile
            is empty — callers should check truthiness before appending.
        """
        if profile is None:
            profile = self.get_profile()

        if not profile:
            return ""

        lines: list[str] = ["## User Profile", ""]
        for category, entries in sorted(profile.items()):
            title = category.replace("_", " ").title()
            lines.append(f"### {title}")
            for key, value in sorted(entries.items()):
                display_key = key.replace("_", " ")
                lines.append(f"- {display_key}: {value}")
            lines.append("")

        return "\n".join(lines).rstrip()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DIAGNOSTICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic info for /status or logging."""
        profile = self.get_profile()
        total_entries = sum(len(v) for v in profile.values())
        return {
            "categories": list(profile.keys()),
            "total_entries": total_entries,
        }

    def __repr__(self) -> str:
        profile = self.get_profile()
        total = sum(len(v) for v in profile.values())
        return f"ProfileManager(entries={total}, categories={list(profile.keys())})"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _clean_extracted_value(raw: str) -> str:
    """
    Clean a regex-captured value for storage.

    - Strips whitespace and trailing punctuation
    - Collapses internal whitespace
    - Returns empty string if result is too short to be meaningful
    """
    # Strip trailing punctuation and whitespace
    cleaned = raw.strip().rstrip(".,;:!?")

    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Reject values that are too short or too long to be useful
    if len(cleaned) < 2 or len(cleaned) > 200:
        return ""

    return cleaned
