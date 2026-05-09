"""
FLEEA Safety Manager — Code-enforced action classification and approval flow.

Architecture:
    This is NOT prompt-only safety.  Every tool call passes through a
    code-level gate BEFORE execution.  The LLM cannot bypass this layer
    no matter what the user says or how the prompt is manipulated.

    Three-tier classification:
        SAFE       → execute immediately, no confirmation
        SENSITIVE  → show intended action, require explicit 'yes'
        DANGEROUS  → refuse unconditionally, no override

    Evaluation order (highest priority first):
        1. Blocked patterns in tool input  → DANGEROUS
        2. Sensitive patterns in tool input → SENSITIVE
        3. Tool name in dangerous registry → DANGEROUS
        4. Tool name in sensitive registry → SENSITIVE
        5. Tool name in safe registry      → SAFE
        6. Default (unknown tool)          → SENSITIVE  (fail-safe)

Design:
    - Settings injected via constructor for the safety_default_deny flag.
    - All pattern matching is regex-based, compiled once at import time.
    - Rich terminal prompts for approval flow — clear, scannable, branded.
    - evaluate_and_gate() is the single entry point used by the brain.
    - No LLM call needed — classification is pure code.

Usage:
    safety = SafetyManager(settings)
    allowed, check = safety.evaluate_and_gate("web_search", {"query": "AI"})
    if allowed:
        result = await registry.execute("web_search", query="AI")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from config.settings import Settings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLASSIFICATION TYPES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SafetyVerdict(Enum):
    """Three-tier safety classification."""

    SAFE = "safe"            # Execute without asking
    SENSITIVE = "sensitive"  # Show action, wait for explicit confirmation
    DANGEROUS = "dangerous"  # Refuse unconditionally — no override


@dataclass(frozen=True)
class SafetyCheck:
    """Immutable result of a safety evaluation."""

    verdict: SafetyVerdict
    reason: str               # Why this classification was applied
    intended_action: str      # Human-readable description shown to user
    tool_name: str
    tool_input: dict[str, Any]
    matched_rule: str = ""    # Which rule triggered (for debugging/logs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLASSIFICATION RULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Tools classified by name ───────────────────────────────────────

SAFE_TOOLS: frozenset[str] = frozenset({
    "web_search",
})

SENSITIVE_TOOLS: frozenset[str] = frozenset({
    "send_email",
    "file_write",
    "file_move",
    "file_copy",
    "file_rename",
    "calendar_create",
    "calendar_update",
    "calendar_delete",
    "task_create",
    "task_update",
    "browser_navigate",
    "browser_action",
    "computer_control",
    "app_launch",
})

DANGEROUS_TOOLS: frozenset[str] = frozenset({
    "file_delete",
    "shell_command",
    "system_command",
    "financial_action",
    "financial_transfer",
    "credential_access",
    "system_shutdown",
    "system_reboot",
    "disk_format",
    "registry_edit",
})


# ── Input content patterns ────────────────────────────────────────
# Each tuple: (compiled_regex, human_reason, verdict)
# Patterns are evaluated against a flattened string of all tool input values.

@dataclass(frozen=True)
class _PatternRule:
    """A single regex-based safety rule."""
    pattern: re.Pattern[str]
    reason: str
    verdict: SafetyVerdict
    rule_id: str


def _compile_rules() -> list[_PatternRule]:
    """Compile all pattern rules once at import time."""
    rules: list[tuple[str, str, SafetyVerdict, str]] = [

        # ── DANGEROUS: blocked unconditionally ─────────────────
        (r"rm\s+-rf\s+/\s*$",
         "Recursive deletion of root directory",
         SafetyVerdict.DANGEROUS, "BLK_RM_ROOT"),

        (r"rm\s+-rf\s+~\s*$",
         "Recursive deletion of home directory",
         SafetyVerdict.DANGEROUS, "BLK_RM_HOME"),

        (r"rm\s+-rf\s+/(?:usr|etc|var|boot|sys|proc)",
         "Recursive deletion of critical system directory",
         SafetyVerdict.DANGEROUS, "BLK_RM_SYSDIR"),

        (r"format\s+[cC]:",
         "Format system drive",
         SafetyVerdict.DANGEROUS, "BLK_FORMAT_C"),

        (r"del\s+/[sS]\s+[cC]:\\",
         "Delete all files on system drive",
         SafetyVerdict.DANGEROUS, "BLK_DEL_C"),

        (r"mkfs\s+.*\s+/dev/sd[a-z]",
         "Format a mounted disk partition",
         SafetyVerdict.DANGEROUS, "BLK_MKFS"),

        (r"dd\s+if=.+\s+of=/dev/",
         "Raw disk write operation",
         SafetyVerdict.DANGEROUS, "BLK_DD"),

        (r":(\\|/)etc/(passwd|shadow|sudoers)",
         "System credential file access",
         SafetyVerdict.DANGEROUS, "BLK_SYSAUTH"),

        (r"curl\s+.+\|\s*(ba)?sh",
         "Remote code execution via pipe to shell",
         SafetyVerdict.DANGEROUS, "BLK_CURL_PIPE"),

        (r"wget\s+.+\|\s*(ba)?sh",
         "Remote code execution via pipe to shell",
         SafetyVerdict.DANGEROUS, "BLK_WGET_PIPE"),

        (r"eval\s*\(",
         "Dynamic code evaluation",
         SafetyVerdict.DANGEROUS, "BLK_EVAL"),

        (r"net\s+user\s+.+\s+/add",
         "Creating system user account",
         SafetyVerdict.DANGEROUS, "BLK_NET_USER"),

        # ── SENSITIVE: require confirmation ────────────────────
        (r"rm\s+(-[rRf]+|--recursive|--force)",
         "Recursive or forced file deletion",
         SafetyVerdict.SENSITIVE, "SEN_RM_RF"),

        (r"del\s+/[sS]",
         "Recursive delete command",
         SafetyVerdict.SENSITIVE, "SEN_DEL_S"),

        (r"rmdir\s+/[sS]",
         "Recursive directory removal",
         SafetyVerdict.SENSITIVE, "SEN_RMDIR"),

        (r"sudo\s+",
         "Elevated privilege command",
         SafetyVerdict.SENSITIVE, "SEN_SUDO"),

        (r"chmod\s+",
         "File permission modification",
         SafetyVerdict.SENSITIVE, "SEN_CHMOD"),

        (r"chown\s+",
         "File ownership change",
         SafetyVerdict.SENSITIVE, "SEN_CHOWN"),

        (r"DROP\s+(TABLE|DATABASE|INDEX)",
         "Database destructive operation",
         SafetyVerdict.SENSITIVE, "SEN_DROP"),

        (r"TRUNCATE\s+TABLE",
         "Database table truncation",
         SafetyVerdict.SENSITIVE, "SEN_TRUNCATE"),

        (r"DELETE\s+FROM\s+\w+\s*(;|\s*$)",
         "Delete all rows from table (no WHERE clause)",
         SafetyVerdict.SENSITIVE, "SEN_DELETE_ALL"),

        (r"password|passwd|credential|secret_key|api_key|access_token",
         "Credential or secret detected in input",
         SafetyVerdict.SENSITIVE, "SEN_CREDENTIAL"),

        (r"shutdown|reboot|restart|poweroff|halt",
         "System power operation",
         SafetyVerdict.SENSITIVE, "SEN_POWER"),

        (r"registry|regedit|reg\s+add|reg\s+delete",
         "Windows Registry modification",
         SafetyVerdict.SENSITIVE, "SEN_REGISTRY"),

        (r":(\\|/)Windows(\\|/)(System32|SysWOW64)",
         "Windows system directory access",
         SafetyVerdict.SENSITIVE, "SEN_WIN_SYS"),

        (r"pip\s+install|npm\s+install|apt\s+install|brew\s+install",
         "Package installation",
         SafetyVerdict.SENSITIVE, "SEN_PKG_INSTALL"),

        (r"git\s+push\s+.*--force",
         "Force push to git remote",
         SafetyVerdict.SENSITIVE, "SEN_GIT_FORCE"),

        (r"ssh\s+|scp\s+|rsync\s+",
         "Remote server access",
         SafetyVerdict.SENSITIVE, "SEN_REMOTE"),

        (r"mailto:|smtp://|sendmail|mutt\s+",
         "Email sending operation",
         SafetyVerdict.SENSITIVE, "SEN_EMAIL"),

        (r"\b(transfer|payment|invoice|charge|debit|credit)\b.*\b\d+\.?\d*\b",
         "Financial operation with amount",
         SafetyVerdict.SENSITIVE, "SEN_FINANCIAL"),
    ]

    return [
        _PatternRule(
            pattern=re.compile(pat, re.IGNORECASE),
            reason=reason,
            verdict=verdict,
            rule_id=rule_id,
        )
        for pat, reason, verdict, rule_id in rules
    ]


# Compile once at module load
_PATTERN_RULES: list[_PatternRule] = _compile_rules()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAFETY MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SafetyManager:
    """
    Code-enforced safety gate for all tool executions.

    Every tool call passes through evaluate_and_gate() before the
    tool registry is allowed to execute it.  This is NOT advisory —
    the brain cannot proceed without this layer's permission.
    """

    def __init__(self, settings: Settings) -> None:
        self._console = Console()
        self._default_deny = settings.SAFETY_DEFAULT_DENY
        # Audit trail: track how many actions were blocked / approved / denied
        self._audit_counts: dict[str, int] = {
            "safe_allowed": 0,
            "sensitive_approved": 0,
            "sensitive_denied": 0,
            "dangerous_blocked": 0,
        }

    @property
    def audit(self) -> dict[str, int]:
        """Safety audit counters for this session."""
        return dict(self._audit_counts)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  CLASSIFICATION ENGINE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def classify(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> SafetyCheck:
        """
        Classify a tool call into SAFE / SENSITIVE / DANGEROUS.

        Evaluation order (first match wins):
            1. Dangerous patterns in input → DANGEROUS
            2. Sensitive patterns in input → SENSITIVE
            3. Tool name in DANGEROUS_TOOLS → DANGEROUS
            4. Tool name in SENSITIVE_TOOLS → SENSITIVE
            5. Tool name in SAFE_TOOLS     → SAFE
            6. Default → SENSITIVE (if safety_default_deny) or SAFE
        """
        input_str = _flatten_input(tool_input)
        action_desc = _describe_action(tool_name, tool_input)

        # ── 1 & 2. Pattern-based classification (input content) ────
        # Dangerous patterns are checked first (they appear first in the list),
        # so ordering within _PATTERN_RULES matters.
        for rule in _PATTERN_RULES:
            if rule.pattern.search(input_str):
                return SafetyCheck(
                    verdict=rule.verdict,
                    reason=rule.reason,
                    intended_action=action_desc,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    matched_rule=rule.rule_id,
                )

        # ── 3. Tool name → DANGEROUS ──────────────────────────────
        if tool_name in DANGEROUS_TOOLS:
            return SafetyCheck(
                verdict=SafetyVerdict.DANGEROUS,
                reason=f"Tool '{tool_name}' is classified as dangerous",
                intended_action=action_desc,
                tool_name=tool_name,
                tool_input=tool_input,
                matched_rule=f"TOOL_DANGEROUS:{tool_name}",
            )

        # ── 4. Tool name → SENSITIVE ──────────────────────────────
        if tool_name in SENSITIVE_TOOLS:
            return SafetyCheck(
                verdict=SafetyVerdict.SENSITIVE,
                reason=f"Tool '{tool_name}' requires explicit approval",
                intended_action=action_desc,
                tool_name=tool_name,
                tool_input=tool_input,
                matched_rule=f"TOOL_SENSITIVE:{tool_name}",
            )

        # ── 5. Tool name → SAFE ──────────────────────────────────
        if tool_name in SAFE_TOOLS:
            return SafetyCheck(
                verdict=SafetyVerdict.SAFE,
                reason="Tool is classified as safe",
                intended_action=action_desc,
                tool_name=tool_name,
                tool_input=tool_input,
                matched_rule=f"TOOL_SAFE:{tool_name}",
            )

        # ── 6. Default: fail-safe ─────────────────────────────────
        if self._default_deny:
            return SafetyCheck(
                verdict=SafetyVerdict.SENSITIVE,
                reason=f"Unknown tool '{tool_name}' — approval required (default-deny policy)",
                intended_action=action_desc,
                tool_name=tool_name,
                tool_input=tool_input,
                matched_rule="DEFAULT_DENY",
            )

        return SafetyCheck(
            verdict=SafetyVerdict.SAFE,
            reason=f"Unknown tool '{tool_name}' — allowed (default-allow policy)",
            intended_action=action_desc,
            tool_name=tool_name,
            tool_input=tool_input,
            matched_rule="DEFAULT_ALLOW",
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  APPROVAL FLOW (terminal interaction)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def request_approval(self, check: SafetyCheck) -> bool:
        """
        Display a rich-formatted approval prompt.
        Returns True if user explicitly approves, False otherwise.
        Default is NO — safety-first.
        """
        content = Text()
        content.append("Tool:     ", style="bold white")
        content.append(f"{check.tool_name}\n", style="bold cyan")
        content.append("Action:   ", style="bold white")
        content.append(f"{check.intended_action}\n", style="white")
        content.append("Reason:   ", style="bold white")
        content.append(f"{check.reason}\n", style="yellow")
        content.append("Rule:     ", style="bold white")
        content.append(f"{check.matched_rule}", style="dim")

        self._console.print()
        self._console.print(
            Panel(
                content,
                title="[bold yellow]  APPROVAL REQUIRED  [/]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        try:
            return Confirm.ask(
                "[bold yellow]Allow this action?[/]",
                default=False,
                console=self._console,
            )
        except (KeyboardInterrupt, EOFError):
            self._console.print("[dim]Cancelled.[/dim]")
            return False

    def display_blocked(self, check: SafetyCheck) -> None:
        """Display a refusal message for dangerous actions.  No override."""
        content = Text()
        content.append("Tool:     ", style="bold white")
        content.append(f"{check.tool_name}\n", style="bold cyan")
        content.append("Action:   ", style="bold white")
        content.append(f"{check.intended_action}\n", style="white")
        content.append("Reason:   ", style="bold white")
        content.append(f"{check.reason}\n", style="bold red")
        content.append("Rule:     ", style="bold white")
        content.append(f"{check.matched_rule}\n\n", style="dim")
        content.append(
            "This action has been permanently blocked.  "
            "No override is available.",
            style="red",
        )

        self._console.print()
        self._console.print(
            Panel(
                content,
                title="[bold red]  ACTION BLOCKED  [/]",
                border_style="red",
                padding=(1, 2),
            )
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PRIMARY ENTRY POINT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def evaluate_and_gate(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[bool, SafetyCheck]:
        """
        Full safety gate.  This is the ONLY method the brain calls.

        Returns:
            (allowed: bool, check: SafetyCheck)

        Flow:
            SAFE      → allowed=True   (silent, no output)
            SENSITIVE → prompt user    → allowed=True/False
            DANGEROUS → display block  → allowed=False (always)
        """
        check = self.classify(tool_name, tool_input)

        if check.verdict == SafetyVerdict.SAFE:
            self._audit_counts["safe_allowed"] += 1
            return True, check

        if check.verdict == SafetyVerdict.DANGEROUS:
            self.display_blocked(check)
            self._audit_counts["dangerous_blocked"] += 1
            return False, check

        if check.verdict == SafetyVerdict.SENSITIVE:
            approved = self.request_approval(check)
            if approved:
                self._audit_counts["sensitive_approved"] += 1
            else:
                self._audit_counts["sensitive_denied"] += 1
            return approved, check

        # Unreachable, but fail safe
        return False, check

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DIAGNOSTICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_rules_summary(self) -> dict[str, Any]:
        """Return a summary of all safety rules for /status display."""
        return {
            "safe_tools": sorted(SAFE_TOOLS),
            "sensitive_tools": sorted(SENSITIVE_TOOLS),
            "dangerous_tools": sorted(DANGEROUS_TOOLS),
            "pattern_rules": len(_PATTERN_RULES),
            "default_deny": self._default_deny,
            "audit": self.audit,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _flatten_input(tool_input: dict[str, Any]) -> str:
    """
    Flatten all tool input values into a single string for pattern matching.
    Handles nested dicts, lists, and primitives recursively.
    """
    parts: list[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _walk(v)
        else:
            parts.append(str(value))

    _walk(tool_input)
    return " ".join(parts)


def _describe_action(tool_name: str, tool_input: dict[str, Any]) -> str:
    """
    Generate a human-readable description of the intended action.
    Truncates long values for terminal readability.
    """
    parts = [f"Execute '{tool_name}'"]
    for key, value in tool_input.items():
        display = str(value)
        if len(display) > 120:
            display = display[:117] + "..."
        parts.append(f"  {key}: {display}")
    return "\n".join(parts)
