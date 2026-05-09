"""
FLEEA User Profile — Structured user preference store.

Separate from vector memory. Stores deterministic facts about the user
(name, work style, schedule preferences) in SQLite key-value format.

Phase 1: Basic CRUD against the user_profile table.
Phase 2+: Auto-extraction of preferences from conversations.

Design:
    The profile table uses a (category, key) composite key.
    Categories: identity, work_style, schedule, writing,
    habits, priorities, preferences.
    This class wraps the DatabaseManager's profile methods with
    a default category for simple key-value access.
"""

from __future__ import annotations

from database.models import DatabaseManager


# Default category for simple set/get that don't specify one.
_DEFAULT_CATEGORY: str = "preferences"


class UserProfile:
    """
    Manages structured user profile data in SQLite.

    The underlying table uses (category, key) pairs.
    For simple access, methods default to the 'preferences' category.
    Use set_categorized / get_categorized for explicit category control.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # ── Simple API (default category) ─────────────────────────────

    def set(self, key: str, value: str, category: str = _DEFAULT_CATEGORY) -> None:
        """Set a profile value under a given category (default: preferences)."""
        self._db.set_profile_value(category, key, value)

    def get(self, key: str, category: str = _DEFAULT_CATEGORY) -> str | None:
        """Get a profile value, or None if not set."""
        return self._db.get_profile_value(category, key)

    def delete(self, key: str, category: str = _DEFAULT_CATEGORY) -> bool:
        """Delete a profile entry. Returns True if a row was removed."""
        return self._db.delete_profile_entry(category, key)

    # ── Full profile access ───────────────────────────────────────

    def get_all(self) -> dict[str, dict[str, str]]:
        """
        Get all profile data, organized by category.

        Returns:
            {
                "identity":   {"name": "John", "timezone": "US/Eastern"},
                "work_style": {"focus_hours": "9AM-12PM"},
                ...
            }
        """
        return self._db.get_full_profile()

    def to_context_string(self) -> str:
        """Format profile as a context string for the system prompt."""
        return self._db.profile_to_context_string()
