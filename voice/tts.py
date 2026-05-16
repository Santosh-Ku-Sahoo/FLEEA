"""
FLEEA TTS — Text-to-Speech via pyttsx3.

Architecture:
    TextToSpeech wraps the pyttsx3 offline TTS engine with a clean,
    thread-safe interface.  It converts assistant text responses to
    audible speech using the system's native TTS voices (SAPI5 on
    Windows, NSSpeechSynthesizer on macOS, espeak on Linux).

Design:
    - NO engine is created on the main thread.  pyttsx3 uses a COM
      singleton on Windows (SAPI5) that is bound to the OS thread it
      was first initialized on.  Calling pyttsx3.init() on one thread
      and runAndWait() on another corrupts the COM apartment and causes
      silent failure.  To avoid this, speak() spawns a fresh daemon
      thread and creates a brand-new engine *inside* that thread.
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
import threading

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
        Configure TTS settings (no engine created here).

        The pyttsx3 engine is intentionally NOT created on the calling
        thread.  Each speak() call spawns a fresh daemon thread and
        creates the engine there, ensuring COM-apartment safety on
        Windows (SAPI5) and thread safety everywhere else.

        Args:
            rate: Speech rate in words per minute.
            volume: Output volume (0.0 = mute, 1.0 = max).
            voice_index: Index of the system voice to use.
                         None = system default.  Use ``list_voices()``
                         to discover available voices.
        """
        self._rate = rate
        self._volume = volume
        self._voice_index = voice_index

        # Verify pyttsx3 is importable — don't create an engine here.
        try:
            import pyttsx3  # noqa: F401
            self._available = True
            _log.info("TTS ready (pyttsx3 found): rate=%d  volume=%.1f", rate, volume)
        except ImportError:
            self._available = False
            _log.error("pyttsx3 not installed — TTS unavailable. Run: pip install pyttsx3")

    # keep _init_engine as a no-op stub so any callers don't crash
    def _init_engine(self) -> None:  # noqa: D401
        """No-op: engine is now created per-thread inside speak()."""
        pass

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

        Runs in a dedicated daemon thread so that it is always safe
        to call from Streamlit's worker threads or any async context.
        On Windows, SAPI5 is COM-based and must be initialized on the
        same thread that calls runAndWait(); spawning a fresh thread
        per utterance guarantees that invariant.

        Args:
            text: The text to speak.  Markdown is stripped automatically.

        Returns:
            True if speech completed successfully, False otherwise.
        """
        if not text or not text.strip():
            return False

        # Clean and truncate
        clean = self._clean_for_speech(text)
        if not clean:
            return False

        if len(clean) > _MAX_SPEAK_LENGTH:
            clean = clean[:_MAX_SPEAK_LENGTH] + "... I'll stop here for brevity."
            _log.debug("Truncated speech to %d chars", _MAX_SPEAK_LENGTH)

        result: list[bool] = []

        def _run() -> None:
            """Initialize a fresh engine on this thread and speak."""
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self._rate)
                engine.setProperty("volume", self._volume)
                if self._voice_index is not None:
                    voices = engine.getProperty("voices")
                    if 0 <= self._voice_index < len(voices):
                        engine.setProperty("voice", voices[self._voice_index].id)
                _log.debug("Speaking %d chars", len(clean))
                engine.say(clean)
                engine.runAndWait()
                result.append(True)
            except Exception as exc:
                _log.error("TTS speech failed: %s", exc)
                result.append(False)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=30)  # max 30 s; prevents indefinite hangs
        return bool(result and result[0])

    def stop(self) -> None:
        """No-op: each speak() runs in its own thread and cleans up automatically."""
        pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  CONFIGURATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def set_rate(self, rate: int) -> None:
        """Update speech rate (words per minute)."""
        self._rate = rate

    def set_volume(self, volume: float) -> None:
        """Update output volume (0.0 – 1.0)."""
        self._volume = max(0.0, min(1.0, volume))

    def list_voices(self) -> list[dict[str, str]]:
        """Return available system voices. Creates a temporary engine for discovery."""
        result: list[dict[str, str]] = []

        def _get_voices() -> None:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                voices = engine.getProperty("voices")
                result.extend(
                    {"index": str(i), "name": v.name, "id": v.id}
                    for i, v in enumerate(voices)
                )
                engine.stop()
            except Exception as exc:
                _log.warning("list_voices failed: %s", exc)

        t = threading.Thread(target=_get_voices, daemon=True)
        t.start()
        t.join(timeout=10)
        return result

    @property
    def is_available(self) -> bool:
        """Whether pyttsx3 is installed and TTS can be used."""
        return self._available

    def __repr__(self) -> str:
        return (
            f"TextToSpeech(rate={self._rate}, volume={self._volume}, "
            f"available={self._available})"
        )
