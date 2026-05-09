"""
FLEEA Voice Interface — Continuous listen-think-speak loop.

Architecture:
    VoiceInterface is the voice-mode entry point for FLEEA.  It wires
    together the STT, TTS, and Brain modules into a continuous loop:

        listen (mic) → transcribe (Whisper) → think (Brain) → speak (pyttsx3)

    The class owns zero business logic — it delegates everything to the
    Brain and merely handles the I/O transport layer (audio in, audio out).

Design:
    - Backend wiring mirrors app.py and streamlit_app.py exactly:
      Settings → DB → Logger → Session → Safety → Registry → Brain.
    - Brain.think() is async; this module runs it via asyncio per turn.
    - "exit" / "quit" / "stop" voice commands end the session gracefully.
    - Silence is handled naturally — if STT returns empty, the loop
      simply re-listens without sending anything to the Brain.
    - All subsystem failures (mic, model, TTS) are caught and logged.
      The loop continues unless the user says "exit".
    - Console feedback is printed so the user has visual confirmation
      of what was heard and what the agent replied.

Integration:
    - Imports SpeechToText from voice.stt (faster-whisper backend)
    - Imports TextToSpeech from voice.tts (pyttsx3 backend)
    - Calls FLEEABrain.think() for agent processing
    - No modification of any Brain, STT, or TTS logic

Usage (from project root):
    python -m interfaces.voice_interface
    python -m interfaces.voice_interface --model small --duration 7

Dependencies:
    pip install faster-whisper sounddevice numpy pyttsx3
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── Ensure project root is on sys.path ─────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent.brain import FLEEABrain
from agent.logger import ExecutionLogger
from agent.safety import SafetyManager
from agent.session import SessionManager
from config.settings import Settings
from database.models import DatabaseManager
from tools.tool_registry import build_default_registry
from voice.stt import SpeechToText
from voice.tts import TextToSpeech

# Phase 2 memory (optional — graceful if missing)
try:
    from memory.short_term import ShortTermMemory
    from memory.retriever import MemoryRetriever
    from memory.profile_manager import ProfileManager
    from memory.vector_store import VectorStore
    _MEMORY_AVAILABLE = True
except ImportError:
    _MEMORY_AVAILABLE = False


_log = logging.getLogger("fleea.interfaces.voice")

# ── Exit commands (normalized to lowercase) ────────────────────────
_EXIT_COMMANDS = frozenset({"exit", "quit", "stop", "goodbye", "bye"})

# ── Defaults ──────────────────────────────────────────────────────
_LISTEN_DURATION: float = 5.0   # seconds per recording cycle
_MAX_RETRIES: int = 2           # retries on mic / transcription failure
_SILENCE_PATIENCE: int = 3      # consecutive silences before hint


def _configure_logging(level: str) -> None:
    """
    Set up file + console logging for voice mode.

    File handler uses rotation (5 MB, 3 backups) to keep disk usage bounded.
    Console handler is minimal to avoid cluttering the voice UX.
    """
    log_dir = os.path.join(_PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "fleea.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(levelname)-8s  %(message)s",
    ))
    console_handler.setLevel(logging.WARNING)  # keep console quiet

    root = logging.getLogger("fleea")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        root.addHandler(file_handler)
        root.addHandler(console_handler)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VOICE INTERFACE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VoiceInterface:
    """
    Continuous voice interaction loop for FLEEA.

    Combines SpeechToText, FLEEABrain, and TextToSpeech into a
    seamless listen → think → speak pipeline.

    Lifecycle:
        1. __init__()  → loads settings, wires all dependencies
        2. run()       → blocks in continuous loop until exit
        3. _shutdown() → flushes stats, ends session, cleans up

    Usage:
        vi = VoiceInterface()
        vi.run()   # blocks until user says "exit"
    """

    def __init__(
        self,
        *,
        model_size: str = "base",
        listen_duration: float = _LISTEN_DURATION,
        language: str = "en",
    ) -> None:
        """
        Initialize the voice interface and wire all dependencies.

        Args:
            model_size: Whisper model size for STT (tiny/base/small/medium).
                        "base" is the best speed/accuracy tradeoff on CPU.
            listen_duration: Seconds to record per listen cycle.
            language: Target language for speech recognition.
        """
        self._listen_duration = listen_duration
        self._consecutive_silences = 0

        # ── 1. Load Settings ──────────────────────────────────────
        self._settings = Settings()  # type: ignore[call-arg]
        _configure_logging(self._settings.LOG_LEVEL)

        # ── 2. Wire Backend Dependencies ──────────────────────────
        #    Mirrors app.py exactly — explicit DI, no globals.
        self._db = DatabaseManager(self._settings.db_path_resolved)
        self._logger = ExecutionLogger(self._db)
        self._session_mgr = SessionManager(self._db)
        self._safety = SafetyManager(self._settings)
        self._registry = build_default_registry(self._settings)

        # Phase 2 memory wiring (graceful degradation)
        short_term = None
        retriever = None
        profile_mgr = None
        vector_store = None

        if _MEMORY_AVAILABLE:
            try:
                persist_dir = str(Path(_PROJECT_ROOT) / "memory" / "chroma_db")
                vector_store = VectorStore(persist_dir=persist_dir)
                retriever = MemoryRetriever(vector_store)
                profile_mgr = ProfileManager(self._db)
                short_term = ShortTermMemory(max_turns=10)
            except Exception as exc:
                _log.warning("Memory subsystem init failed (non-fatal): %s", exc)

        # ── 3. Create Brain ───────────────────────────────────────
        self._brain = FLEEABrain(
            settings=self._settings,
            tool_registry=self._registry,
            logger=self._logger,
            safety=self._safety,
            session_manager=self._session_mgr,
            short_term_memory=short_term,
            memory_retriever=retriever,
            profile_manager=profile_mgr,
            vector_store=vector_store,
        )

        # ── 4. Create STT + TTS ──────────────────────────────────
        self._stt = SpeechToText(
            model_size=model_size,
            language=language,
        )
        self._tts = TextToSpeech(rate=175)

        # ── 5. Start session ──────────────────────────────────────
        self._session = self._session_mgr.create_session()
        self._running = False

        _log.info(
            "VoiceInterface initialized — session %s  model=%s  lang=%s",
            getattr(self._session, "short_id", "?"),
            model_size,
            language,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  MAIN LOOP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def run(self) -> None:
        """
        Start the continuous voice interaction loop.

        Blocks until the user says "exit" or a fatal error occurs.

        Loop:
            1. Print prompt + listen for audio
            2. Transcribe via Whisper (with retry on failure)
            3. Check for exit command
            4. Send to Brain.think()
            5. Print + speak response
            6. Repeat
        """
        self._running = True
        self._consecutive_silences = 0

        _print_banner()
        self._tts.speak("Hello! I'm FLEEA, your voice assistant. How can I help?")

        while self._running:
            try:
                self._loop_iteration()
            except KeyboardInterrupt:
                print("\n\n  Interrupted by user.")
                break
            except Exception as exc:
                _log.error("Loop error (continuing): %s", exc)
                print(f"\n  [Error: {exc} — retrying...]\n")

        # Cleanup
        self._shutdown()

    def _loop_iteration(self) -> None:
        """Execute one listen → think → speak cycle."""

        # ── Listen ────────────────────────────────────────────────
        print("\n  🎤 Listening... (speak now)")
        text = self._listen_with_retry()

        if not text:
            self._consecutive_silences += 1

            if self._consecutive_silences >= _SILENCE_PATIENCE:
                print("  ... (still listening — say something, or 'exit' to quit)")
                self._consecutive_silences = 0  # reset after hint
            else:
                print("  ... (silence)")

            return

        # Reset silence counter on successful speech
        self._consecutive_silences = 0

        # ── Display what was heard ────────────────────────────────
        print(f"\n  🗣️  You: {text}")

        # ── Check exit command ────────────────────────────────────
        normalized = text.strip().lower().rstrip(".!?")
        if normalized in _EXIT_COMMANDS:
            print("\n  👋 Goodbye!")
            self._tts.speak("Goodbye! Have a great day.")
            self._running = False
            return

        # ── Think ─────────────────────────────────────────────────
        print("  🤔 Thinking...")
        start = time.perf_counter()

        response = asyncio.run(self._brain.think(text))

        elapsed = time.perf_counter() - start
        _log.info("Brain response in %.1fs", elapsed)

        # ── Display + Speak ───────────────────────────────────────
        print(f"\n  🤖 FLEEA: {response}")
        self._tts.speak(response)

    def _listen_with_retry(self) -> str:
        """
        Listen for speech with retry on transient failures.

        Retries up to _MAX_RETRIES times if the microphone or
        transcription fails.  Returns empty string if all retries
        fail or silence is detected.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                text = self._stt.listen(duration=self._listen_duration)
                return text
            except Exception as exc:
                _log.warning(
                    "Listen attempt %d/%d failed: %s",
                    attempt, _MAX_RETRIES, exc,
                )
                if attempt < _MAX_RETRIES:
                    print(f"  [Retry {attempt}/{_MAX_RETRIES}...]")

        return ""

    def _shutdown(self) -> None:
        """
        Clean up resources on exit.

        Order: stop TTS → flush logs → end session → print summary.
        Safe to call multiple times.
        """
        self._running = False
        self._tts.stop()

        # End session + flush stats
        session_id = getattr(self._session, "session_id", None)

        try:
            if session_id:
                self._logger.flush_session_stats(session_id)
        except Exception:
            pass

        try:
            completed = self._session_mgr.end_session()
            if completed:
                print(f"\n  Session: {completed.short_id}")
                print(f"  Duration: {completed.duration_display}")
                print(f"  Interactions: {completed.total_interactions}")
        except Exception:
            pass

        _log.info("Voice session ended")
        print("\n  Session ended. Logs saved.\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  DIAGNOSTICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_status(self) -> dict[str, Any]:
        """Return voice interface diagnostics."""
        return {
            "stt": repr(self._stt),
            "tts": repr(self._tts),
            "tts_available": self._tts.is_available,
            "stt_available": self._stt.is_available(),
            "brain_status": self._brain.get_status(),
            "running": self._running,
            "consecutive_silences": self._consecutive_silences,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _print_banner() -> None:
    """Print the voice mode startup banner."""
    print("""
  ╔══════════════════════════════════════════╗
  ║        FLEEA — Voice Mode               ║
  ║   Future Learning Executive & Everyday  ║
  ║              Assistant                  ║
  ╠══════════════════════════════════════════╣
  ║  🎤  Speak naturally after the prompt   ║
  ║  🛑  Say "exit" to quit                 ║
  ║  ⏱️   Default: 5s recording window      ║
  ╚══════════════════════════════════════════╝
    """)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """Launch FLEEA in voice mode."""
    import argparse

    parser = argparse.ArgumentParser(
        description="FLEEA Voice Interface — hands-free AI assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size (tiny=fastest, medium=most accurate)",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Listening duration in seconds per cycle",
    )
    parser.add_argument(
        "--language", default="en",
        help="Speech recognition language code (e.g. en, es, fr, hi)",
    )
    args = parser.parse_args()

    try:
        vi = VoiceInterface(
            model_size=args.model,
            listen_duration=args.duration,
            language=args.language,
        )
        vi.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        _log.exception("Fatal error in voice interface")
        sys.exit(1)


if __name__ == "__main__":
    main()
