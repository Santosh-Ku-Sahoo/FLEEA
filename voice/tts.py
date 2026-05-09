"""
FLEEA TTS — Text-to-Speech via pyttsx3.

Architecture:
    TextToSpeech wraps the pyttsx3 offline TTS engine with a clean,
    thread-safe interface.  It converts assistant text responses to
    audible speech using the system's native TTS voices (SAPI5 on
    Windows, NSSpeechSynthesizer on macOS, espeak on Linux).

Design:
    - Engine initialized once at construction.
    - ``speak()`` blocks until the utterance is complete (pyttsx3's
      runAndWait() is inherently synchronous).
    - Markdown artifacts (code blocks, URLs, headers) are stripped
      before speaking to produce clean, natural audio.
    - Rate, volume, and voice are configurable at init and runtime.
    - All exceptions are caught — TTS failure never crashes the agent.

Integration:
    tts = TextToSpeech(rate=175)
    tts.speak("The weather in Seattle is 62°F and partly cloudy.")

Dependencies:
    pip install pyttsx3
"""

from __future__ import annotations

import logging
import re
from typing import Any

_log = logging.getLogger("fleea.voice.tts")

# ── Defaults ───────────────────────────────────────────────────────
_DEFAULT_RATE: int = 175       # words per minute (natural pace)
_DEFAULT_VOLUME: float = 0.9   # 0.0 – 1.0
_MAX_SPEAK_LENGTH: int = 2000  # truncate very long responses


class TextToSpeech:
    """
    Offline text-to-speech engine using pyttsx3.

    Converts text to audible speech via the OS's native TTS voices.
    Strips markdown formatting for clean spoken output.

    Usage:
        tts = TextToSpeech()
        tts.speak("Hello, how can I help you today?")
    """

    def __init__(
        self,
        *,
        rate: int = _DEFAULT_RATE,
        volume: float = _DEFAULT_VOLUME,
        voice_index: int | None = None,
    ) -> None:
        """
        Initialize the TTS engine.

        Args:
            rate: Speech rate in words per minute.
            volume: Output volume (0.0 = mute, 1.0 = max).
            voice_index: Index of the system voice to use.
                         None = system default.  Use ``list_voices()``
                         to discover available voices.
        """
        self._engine: Any = None
        self._rate = rate
        self._volume = volume
        self._voice_index = voice_index
        self._initialized = False

        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize the pyttsx3 engine with configured settings."""
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)

            # Set voice if specified
            if self._voice_index is not None:
                voices = self._engine.getProperty("voices")
                if 0 <= self._voice_index < len(voices):
                    self._engine.setProperty("voice", voices[self._voice_index].id)
                    _log.info("TTS voice set to: %s", voices[self._voice_index].name)
                else:
                    _log.warning(
                        "Voice index %d out of range (0–%d), using default",
                        self._voice_index, len(voices) - 1,
                    )

            self._initialized = True
            _log.info("TTS initialized: rate=%d  volume=%.1f", self._rate, self._volume)

        except Exception as exc:
            _log.error("TTS initialization failed: %s", exc)
            self._initialized = False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TEXT CLEANING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """
        Strip markdown and formatting artifacts for natural speech.

        Removes:
            - Code blocks (``` ... ```)
            - Inline code (`...`)
            - Markdown headers (##)
            - Bold/italic markers (** __)
            - Bullet points (- *)
            - URLs (keep link text if available)
            - Multiple whitespace
        """
        # Remove fenced code blocks entirely (not useful as speech)
        text = re.sub(r"```[\s\S]*?```", " ", text)

        # Remove inline code backticks
        text = re.sub(r"`([^`]*)`", r"\1", text)

        # Remove markdown links, keep display text
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

        # Remove standalone URLs
        text = re.sub(r"https?://\S+", "", text)

        # Remove markdown formatting
        text = re.sub(r"#{1,6}\s*", "", text)       # headers
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)  # bold/italic
        text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)    # underscores

        # Remove bullet points
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PUBLIC API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def speak(self, text: str) -> bool:
        """
        Speak the given text aloud.

        Cleans markdown formatting, truncates if too long, and
        synthesizes speech via the system's native TTS engine.

        Args:
            text: The text to speak.  Markdown is stripped automatically.

        Returns:
            True if speech completed successfully, False otherwise.
        """
        if not self._initialized or not self._engine:
            _log.warning("TTS not initialized, skipping speech")
            return False

        if not text or not text.strip():
            return False

        # Clean and truncate
        clean = self._clean_for_speech(text)
        if not clean:
            return False

        if len(clean) > _MAX_SPEAK_LENGTH:
            clean = clean[:_MAX_SPEAK_LENGTH] + "... I'll stop here for brevity."
            _log.debug("Truncated speech to %d chars", _MAX_SPEAK_LENGTH)

        try:
            _log.debug("Speaking %d chars", len(clean))
            self._engine.say(clean)
            self._engine.runAndWait()
            return True

        except Exception as exc:
            _log.error("TTS speech failed: %s", exc)
            # Attempt to reinitialize engine for next call
            self._init_engine()
            return False

    def stop(self) -> None:
        """Stop any ongoing speech."""
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  CONFIGURATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def set_rate(self, rate: int) -> None:
        """Update speech rate (words per minute)."""
        self._rate = rate
        if self._engine:
            self._engine.setProperty("rate", rate)

    def set_volume(self, volume: float) -> None:
        """Update output volume (0.0 – 1.0)."""
        self._volume = max(0.0, min(1.0, volume))
        if self._engine:
            self._engine.setProperty("volume", self._volume)

    def list_voices(self) -> list[dict[str, str]]:
        """Return available system voices for discovery."""
        if not self._engine:
            return []

        voices = self._engine.getProperty("voices")
        return [
            {"index": i, "name": v.name, "id": v.id}
            for i, v in enumerate(voices)
        ]

    @property
    def is_available(self) -> bool:
        """Whether the TTS engine is initialized and ready."""
        return self._initialized

    def __repr__(self) -> str:
        return (
            f"TextToSpeech(rate={self._rate}, volume={self._volume}, "
            f"initialized={self._initialized})"
        )
