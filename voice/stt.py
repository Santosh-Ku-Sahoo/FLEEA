"""
FLEEA STT — Speech-to-Text via faster-whisper.

Architecture:
    SpeechToText wraps two concerns:
        1. Microphone capture — uses ``sounddevice`` to record PCM audio
           chunks from the system's default input device.
        2. Transcription — uses ``faster-whisper`` (CTranslate2 backend)
           for fast, local, offline speech-to-text.

    The module is designed for a push-to-listen loop: the caller invokes
    ``listen()`` which blocks while recording, then returns the transcribed
    text.  Silence detection via RMS energy avoids sending blank audio to
    the model.

Design:
    - Model loaded once at init (lazy-loaded on first ``listen()`` call
      to avoid blocking import).
    - ``compute_type="int8"`` for CPU-friendly inference (~2x faster
      than float32 on most machines).
    - Recording uses 16kHz mono (Whisper's native sample rate).
    - Silence detection: if peak RMS across the recording is below the
      threshold, we skip transcription entirely.
    - All exceptions are caught and logged — never crashes the caller.

Integration:
    stt = SpeechToText(model_size="base")
    text = stt.listen(duration=5)
    if text:
        response = await brain.think(text)

Dependencies:
    pip install faster-whisper sounddevice numpy
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

_log = logging.getLogger("fleea.voice.stt")

# ── Defaults ───────────────────────────────────────────────────────
_DEFAULT_MODEL: str = "base"
_SAMPLE_RATE: int = 16_000          # Whisper's native sample rate
_CHANNELS: int = 1                   # mono
_DEFAULT_DURATION: float = 5.0       # seconds to record per listen()
_SILENCE_RMS_THRESHOLD: float = 0.01 # below this → silence


class SpeechToText:
    """
    Local speech-to-text engine using faster-whisper.

    Captures microphone audio via sounddevice and transcribes with
    the CTranslate2-backed Whisper model for fast CPU inference.

    Usage:
        stt = SpeechToText()
        text = stt.listen()          # blocks for ~5s, returns string
        text = stt.listen(duration=3)  # shorter recording
    """

    def __init__(
        self,
        *,
        model_size: str = _DEFAULT_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ) -> None:
        """
        Initialize the STT engine.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3).
                        "base" is the best balance of speed and accuracy for
                        real-time voice on CPU.
            device: Compute device ("cpu" or "cuda").
            compute_type: CTranslate2 quantization ("int8", "float16", "float32").
            language: Target language for transcription.
        """
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model: Any = None  # lazy-loaded

        _log.info(
            "STT configured: model=%s  device=%s  compute=%s  lang=%s",
            model_size, device, compute_type, language,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  MODEL LOADING (lazy)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _ensure_model(self) -> None:
        """Load the Whisper model on first use."""
        if self._model is not None:
            return

        _log.info("Loading faster-whisper model '%s' (first call)...", self._model_size)
        start = time.perf_counter()

        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )

        elapsed = time.perf_counter() - start
        _log.info("Model loaded in %.1fs", elapsed)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  RECORDING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _record(duration: float) -> np.ndarray:
        """
        Record audio from the default microphone.

        Returns a float32 numpy array of shape (samples,) at 16kHz mono.
        """
        import sounddevice as sd

        _log.debug("Recording %.1fs of audio...", duration)
        audio = sd.rec(
            int(duration * _SAMPLE_RATE),
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype="float32",
        )
        sd.wait()  # block until recording is complete

        return audio.flatten()

    @staticmethod
    def _is_silence(audio: np.ndarray) -> bool:
        """Check if the recorded audio is effectively silence."""
        # Clean audio of NaNs or Infs that might have crept in from hardware issues
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use absolute mean as a more robust proxy for volume if RMS overflows
        try:
            rms = float(np.sqrt(np.mean(np.square(audio))))
        except (RuntimeError, OverflowError):
            rms = float(np.mean(np.abs(audio)))

        _log.debug("Audio RMS: %.4f (threshold: %.4f)", rms, _SILENCE_RMS_THRESHOLD)
        
        # Handle cases where rms might be NaN after calculations
        if np.isnan(rms):
            return True
            
        return rms < _SILENCE_RMS_THRESHOLD

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TRANSCRIPTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a numpy audio array to text.

        Returns the concatenated text from all detected segments,
        stripped of whitespace.  Returns empty string on failure.
        """
        self._ensure_model()

        try:
            segments, info = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=5,
                vad_filter=True,       # voice activity detection
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                ),
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()
            _log.debug("Transcribed (%s, %.0f%% prob): '%s'",
                       info.language, info.language_probability * 100, text)
            return text

        except Exception as exc:
            _log.error("Transcription failed: %s", exc)
            return ""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PUBLIC API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def listen(self, duration: float = _DEFAULT_DURATION) -> str:
        """
        Record from the microphone and return transcribed text.

        Blocks for ``duration`` seconds while recording, then runs
        transcription.  Returns empty string if silence or failure.

        Args:
            duration: Recording duration in seconds.

        Returns:
            Transcribed text, or empty string.
        """
        try:
            audio = self._record(duration)
        except Exception as exc:
            _log.error("Microphone capture failed: %s", exc)
            return ""

        if self._is_silence(audio):
            _log.debug("Silence detected, skipping transcription")
            return ""

        return self._transcribe(audio)

    def is_available(self) -> bool:
        """Check if the microphone and model dependencies are available."""
        try:
            import sounddevice as sd  # noqa: F401
            import faster_whisper     # noqa: F401
            return True
        except ImportError:
            return False

    def __repr__(self) -> str:
        return (
            f"SpeechToText(model={self._model_size!r}, "
            f"device={self._device!r}, lang={self._language!r})"
        )
